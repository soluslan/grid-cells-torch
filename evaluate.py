"""Unattended evaluation of a checkpoint: run the model, measure, save.

Measures the two things the paper reports for the supervised network:

1. Does it path-integrate? Decoded-position error vs. the true trajectory
   (Fig. 1b,c: 16cm after 15s trained, 91cm untrained). Runs first -- if this
   fails there is nothing to interpret in the gridness numbers.
2. Did grid-like units emerge in the bottleneck? (Fig. 1d,g: 129/512.)
3. What else is in that layer, and is it stable? Border cells (8.7%),
   conjunctive grid x head-direction cells (14, 11% of the grid units), the
   clustering of grid scale (Fig. 1e: 3 clusters at 47/70/106cm), and the
   ratemap correlation between 2e5 and 3e5 training steps (Ext. Data Fig. 3b).

Library only: notebooks/03_path_integration.ipynb, 04_bottleneck_units.ipynb
and 05_cell_types.ipynb drive these. Per-unit measures live in scores.py and
every plot in figures.py, so nothing here overlaps those.
"""

import os

import numpy as np
import torch

from ensembles import build_ensembles, encode_initial_conditions
from scores import (BORDER_BINS, BORDER_THRESHOLD, HD_THRESHOLD, GridScorer,
                    border_score, cluster_scales, directional_ratemap,
                    discreteness_significance, grid_scale, paper_mask_parameters,
                    ratemap_stability, resultant_vector_length)
from train import build_model

# The paper's operative criterion: one cutoff for every unit. Every
# grid-like count it reports (Fig. 1g, the 129/512 headline) uses this.
GRIDNESS_THRESHOLD = 0.37

# The paper does not say how it decoded position from the place cells, so all
# three plausible readouts are measured. See PlaceCellEnsemble.decode_position.
DECODERS = ("argmax", "weighted_mean", "top3")

STEP_SECONDS = 0.15  # 100 stored steps spanning the paper's T=15s trajectory


def collect_batches(loader, n_trajectories):
    """Materialise the evaluation trajectories once, so that every model and
    control below is measured on exactly the same data."""
    batches, n = [], 0
    for batch in loader:
        batches.append(batch)
        n += batch["init_pos"].shape[0]
        if n >= n_trajectories:
            break
    return batches, n


@torch.no_grad()
def run_model(model, batches, place_ensembles, hd_ensembles, device,
              want_activations=False):
    """-> ({decoder: decoded_pos [N,T,2]}, bottleneck [N,T,U] or None, lstm or None).

    Positions are decoded inside the loop: keeping the raw [N,T,256] place-cell
    posteriors would cost ~400MB to say the same thing as ~3MB of coordinates.
    """
    model.eval()
    place = place_ensembles[0]
    decoded = {mode: [] for mode in DECODERS}
    bottleneck_all, lstm_all = [], []

    for batch in batches:
        init_conds = encode_initial_conditions(
            batch["init_pos"].to(device), batch["init_hd"].to(device),
            place_ensembles, hd_ensembles)
        out = model(init_conds, batch["ego_vel"].to(device))

        probs = torch.softmax(out.logits[0], dim=-1)  # [B,T,n_pc]
        for mode in DECODERS:
            decoded[mode].append(place.decode_position(probs, mode).cpu().numpy())
        if want_activations:
            bottleneck_all.append(out.bottleneck.cpu().numpy())
            lstm_all.append(out.lstm_output.cpu().numpy())

    decoded = {m: np.concatenate(v, axis=0) for m, v in decoded.items()}
    if not want_activations:
        return decoded, None, None
    return (decoded, np.concatenate(bottleneck_all, axis=0),
            np.concatenate(lstm_all, axis=0))


@torch.no_grad()
def decode_ground_truth(batches, place_ensembles, device):
    """The error floor: decode the place-cell code of the TRUE position.

    This is what a network that knew its location perfectly would still score,
    because the readout can only name cell centres. Without it there is no way
    to tell how much of the measured error is the network's and how much is the
    resolution of a 256-cell code.
    """
    place = place_ensembles[0]
    decoded = {mode: [] for mode in DECODERS}
    for batch in batches:
        probs = place.posterior(batch["target_pos"].to(device))
        for mode in DECODERS:
            decoded[mode].append(place.decode_position(probs, mode).cpu().numpy())
    return {m: np.concatenate(v, axis=0) for m, v in decoded.items()}


def errors(decoded, true_pos):
    """[N,T,2] -> per-(trajectory,timestep) Euclidean error [N,T], in metres."""
    return np.linalg.norm(decoded - true_pos, axis=-1)


def effect_size(a, b):
    """Suppl. Methods 3f eqs. (9)-(10): mean difference over pooled s.d."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                     / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled


def path_integration_errors(model, untrained, batches, place_ensembles,
                            hd_ensembles, device, want_activations=False):
    """-> (err, decoded, bottleneck, lstm).

    `err` and `decoded` share a shape: {condition: {decoder: ...}} over the
    conditions trained / untrained / floor, holding [N,T] errors and [N,T,2]
    positions respectively.
    """
    true_pos = np.concatenate([b["target_pos"].numpy() for b in batches], axis=0)
    decoded_trained, bottleneck, lstm = run_model(
        model, batches, place_ensembles, hd_ensembles, device, want_activations)
    decoded_untrained, _, _ = run_model(
        untrained, batches, place_ensembles, hd_ensembles, device)

    decoded = {"trained": decoded_trained, "untrained": decoded_untrained,
               "floor": decode_ground_truth(batches, place_ensembles, device)}
    err = {cond: {m: errors(d[m], true_pos) for m in DECODERS}
           for cond, d in decoded.items()}
    return err, decoded, bottleneck, lstm


def best_decoder(err):
    """Whichever readout the trained network ends the trajectory closest under."""
    return min(DECODERS, key=lambda m: err["trained"][m][:, -1].mean())


def report_path_integration(err, dt=STEP_SECONDS):
    n_steps = err["trained"][DECODERS[0]].shape[1]
    print(f"\n=== path integration ({n_steps} steps x {dt}s = "
          f"{n_steps * dt:.1f}s trajectories) ===")
    print(f"  {'decoder':<14}{'condition':<12}{'t=0':>9}{'final':>9}{'mean over t':>13}")
    for mode in DECODERS:
        for cond in ("trained", "untrained", "floor"):
            e = err[cond][mode]
            print(f"  {mode:<14}{cond:<12}"
                  f"{100 * e[:, 0].mean():>8.1f}c{100 * e[:, -1].mean():>8.1f}c"
                  f"{100 * e.mean():>12.1f}c")
        trained_final = err["trained"][mode][:, -1]
        untrained_final = err["untrained"][mode][:, -1]
        floor_final = err["floor"][mode][:, -1].mean()
        print(f"  {'':<14}-> network error above floor: "
              f"{100 * (trained_final.mean() - floor_final):.1f}cm,  "
              f"effect size vs untrained = "
              f"{abs(effect_size(trained_final, untrained_final)):.2f}")
    print("  (paper: 16cm trained, 91cm untrained after 15s, effect size 2.83)")


def score_directional(headings, activations):
    """-> (tuning [n_units, HD_BINS], resultant lengths [n_units]).

    `headings` is target_hd flattened to [N]; `activations` [N, n_units].
    """
    tuning = directional_ratemap(headings, activations)
    return tuning, resultant_vector_length(tuning)


def report_directional(name, lengths):
    n = int((lengths > HD_THRESHOLD).sum())
    print(f"  {name}: mean resultant={lengths.mean():.3f}, max={lengths.max():.3f}, "
          f">{HD_THRESHOLD} (paper's threshold) = {n}/{len(lengths)} "
          f"({100 * n / len(lengths):.1f}%)")
    return n


def report_gridness(name, scores_60):
    n = int((scores_60 > GRIDNESS_THRESHOLD).sum())
    print(f"  {name}: mean score_60={scores_60.mean():.3f}, "
          f"max={scores_60.max():.3f}, score_60>{GRIDNESS_THRESHOLD} "
          f"(paper's threshold) = {n}/{len(scores_60)} "
          f"({100 * n / len(scores_60):.1f}%)")
    return n


# --------------------------------------------------------------------------
# Cell types beyond gridness -- Fig. 1e,f,g and Ext. Data Fig. 3b
# --------------------------------------------------------------------------

def score_borders(cfg, xy, activations, nbins=BORDER_BINS):
    """Border score per unit, on the paper's own 20x20 binning.

    Suppl. 3d specifies 20x20 bins for the border score while using 32x32 for
    gridness, so this rebins rather than reusing the ratemaps notebook 04
    already has.
    """
    scorer = build_scorer(cfg, nbins)
    return np.array([
        border_score(scorer.calculate_ratemap(xy[:, 0], xy[:, 1], activations[:, i]))
        for i in range(activations.shape[1])])


def grid_scales(sacs, cfg, nbins, grid_like_mask):
    """Grid scale in metres per unit; NaN wherever `grid_like_mask` is False.

    The mask is required, not optional: the paper's Reporting Summary states
    "Assessment of grid scale was limited to units determined to be grid-like",
    and a scale read off a non-periodic autocorrelogram is noise.
    """
    bin_m = cfg.task.env_size / nbins
    out = np.full(len(sacs), np.nan)
    for i in np.flatnonzero(np.asarray(grid_like_mask)):
        out[i] = grid_scale(sacs[i], bin_m)
    return out


def report_border(name, scores):
    n = int((scores > BORDER_THRESHOLD).sum())
    print(f"  {name}: mean border={scores.mean():.3f}, max={scores.max():.3f}, "
          f">{BORDER_THRESHOLD} (paper's threshold) = {n}/{len(scores)} "
          f"({100 * n / len(scores):.1f}%)   [paper: 8.7%]")
    return n


def report_conjunctive(name, scores_60, resultants):
    """Fig. 1f,g: units that are both grid-like and directionally tuned."""
    grid = scores_60 > GRIDNESS_THRESHOLD
    directional = resultants > HD_THRESHOLD
    both = int((grid & directional).sum())
    n_grid = int(grid.sum())
    pct = 100 * both / n_grid if n_grid else float("nan")
    print(f"  {name}: {both} conjunctive (grid AND directional) = {pct:.1f}% "
          f"of the {n_grid} grid-like units   [paper: 14, 11% of 129]")
    return both


def report_grid_scales(name, scales):
    s = scales[np.isfinite(scales)]
    if len(s) == 0:
        print(f"  {name}: no unit has a measurable grid scale")
        return s
    print(f"  {name}: {len(s)} scales, range {100*s.min():.0f}-{100*s.max():.0f}cm, "
          f"mean {100*s.mean():.0f}cm   [paper: 28-115cm, mean 66cm]")
    return s


# The paper clustered 129 scales. Below this many, a mixture model has more
# freedom than the data constrains and BIC starts selecting noise, so the
# output is reported but flagged rather than read as a result.
MIN_SCALES_TO_CLUSTER = 30


def report_scale_clustering(name, scales, n_shuffles=500, seed=0):
    """-> (k, means, bics, ratios, discreteness, null, p). Fig. 1e."""
    s = scales[np.isfinite(scales)]
    if len(s) < 10:
        print(f"  {name}: only {len(s)} scales, too few to cluster")
        return 0, np.array([]), np.array([]), np.array([]), np.nan, np.array([]), np.nan
    if len(s) < MIN_SCALES_TO_CLUSTER:
        print(f"  {name}: UNDERPOWERED -- {len(s)} scales against the paper's 129. "
              f"Both lines below are reported for completeness, not as findings: "
              f"a mixture over this few points fits noise, and the shuffle test "
              f"has little power to reject anything.")
    obs, null, p = discreteness_significance(s, n_shuffles=n_shuffles, seed=seed)
    k, means, bics, ratios = cluster_scales(s, seed=seed)
    print(f"  {name}: discreteness {obs:.2f} vs null {null.mean():.2f}+/-{null.std():.2f}, "
          f"p={p:.4f}   [paper: p<0.002]")
    print(f"  {name}: BIC picks {k} cluster(s) at "
          f"{', '.join(f'{100*m:.0f}cm' for m in means)}, "
          f"ratios {np.round(ratios, 2)}   [paper: 3 at 47/70/106cm, ~1.5]")
    return k, means, bics, ratios, obs, null, p


@torch.no_grad()
def layer_ratemaps(cfg, checkpoint, batches, device, nbins, layer="bottleneck"):
    """-> (ratemaps [n_units][nbins,nbins], epoch) for one checkpoint.

    Pass the same `batches` for both checkpoints so the two sets of maps come
    from identical trajectories -- otherwise the correlation between them
    mixes representational drift with a different sample of the arena.
    """
    model, _, epoch, place, hd = load_models(cfg, checkpoint, device)
    _, bottleneck, lstm = run_model(model, batches, place, hd, device,
                                    want_activations=True)
    acts = {"bottleneck": bottleneck, "lstm": lstm}[layer]
    xy = np.concatenate([b["target_pos"].numpy() for b in batches],
                        axis=0).reshape(-1, 2)
    flat = acts.reshape(-1, acts.shape[-1])
    scorer = build_scorer(cfg, nbins)
    return [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], flat[:, i])
            for i in range(flat.shape[1])], epoch


def stability(ratemaps_a, ratemaps_b):
    """Per-unit ratemap correlation between two training checkpoints."""
    return np.array([ratemap_stability(a, b)
                     for a, b in zip(ratemaps_a, ratemaps_b)])


def report_stability(name, stab, scores_60, resultants):
    """Ext. Data Fig. 3b: grid-like units are stable, directional ones are not."""
    grid = scores_60 > GRIDNESS_THRESHOLD
    directional = resultants > HD_THRESHOLD
    print(f"  {name}: all units mean r={np.nanmean(stab):.3f}")
    for label, mask in (("grid-like", grid), ("directional", directional)):
        if mask.sum():
            print(f"    {label:<12} n={int(mask.sum()):<4} mean r="
                  f"{np.nanmean(stab[mask]):.3f}")
    print("  [paper: grid-like units highly stable, directional units not]")
    return stab


def load_models(cfg, checkpoint, device, untrained_seed=0):
    """-> (trained, untrained, epoch, place_ensembles, hd_ensembles)."""
    place_ensembles, hd_ensembles = build_ensembles(cfg, device)
    assert len(place_ensembles) == 1, "decoding assumes a single place ensemble"
    target_ensembles = place_ensembles + hd_ensembles

    model = build_model(cfg, target_ensembles, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    torch.manual_seed(untrained_seed)
    untrained = build_model(cfg, target_ensembles, device)
    return model, untrained, ckpt["epoch"], place_ensembles, hd_ensembles


def build_scorer(cfg, nbins):
    half = cfg.task.env_size / 2.0
    return GridScorer(nbins, ((-half, half), (-half, half)), paper_mask_parameters())


def default_out_dir(checkpoint):
    """data/checkpoints/<run>/ckpt.pt -> results/<run>/.

    Everything large and regenerable lives under data/ and is gitignored;
    results/ holds the small figures worth keeping. The two are paired by run
    name, so naming a run at training time is enough to keep its outputs
    separate from every other run's.
    """
    ckpt_dir = os.path.abspath(os.path.dirname(checkpoint))
    parts = ckpt_dir.split(os.sep)
    if "checkpoints" in parts:
        i = len(parts) - 1 - parts[::-1].index("checkpoints")
        run = os.sep.join(parts[i + 1:])
        root = os.sep.join(parts[:i - 1]) if i >= 1 else os.sep
        return os.path.join(root, "results", run)
    return os.path.join(ckpt_dir, "eval")
