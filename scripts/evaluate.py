"""Load a trained checkpoint, run inference (no training) over a batch of
trajectories, and check whether grid-cell-like (hexagonally periodic)
activity emerged in the bottleneck / LSTM units.

Ported from google-deepmind/grid-cells' train.py evaluation block +
utils.get_scores_and_plot, now a standalone script instead of something
interleaved into the training loop.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # headless: must be set before `scores` imports pyplot

from config import Config
from dataset import build_dataloader
from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from model import GridCellsRNN
from scores import GridScorer
from utils import encode_initial_conditions

# Paper's own shuffle-derived population threshold (Banino et al. 2018,
# Supplementary Methods 3d) -- see README "Known caveats" for how this
# differs from the paper's per-unit shuffle procedure.
GRIDNESS_THRESHOLD = 0.37


def build_ensembles(cfg, device):
    place_cell_ensembles = [
        PlaceCellEnsemble(n, stdev=s, pos_min=-cfg.task.env_size / 2.0,
                           pos_max=cfg.task.env_size / 2.0, seed=cfg.task.neurons_seed).to(device)
        for n, s in zip(cfg.task.n_pc, cfg.task.pc_scale)
    ]
    head_direction_ensembles = [
        HeadDirectionCellEnsemble(n, concentration=c, seed=cfg.task.neurons_seed).to(device)
        for n, c in zip(cfg.task.n_hdc, cfg.task.hdc_concentration)
    ]
    return place_cell_ensembles, head_direction_ensembles


@torch.no_grad()
def collect_activations(model, place_cell_ensembles, head_direction_ensembles,
                         loader, n_trajectories, device):
    """Runs the model over n_trajectories worth of data (in eval mode, no
    dropout) and returns flattened (xy, bottleneck_acts, lstm_acts) arrays,
    each row one (trajectory, timestep) sample."""
    model.eval()
    xy_all, bottleneck_all, lstm_all = [], [], []
    n_collected = 0
    for batch in loader:
        init_pos = batch["init_pos"].to(device)
        init_hd = batch["init_hd"].to(device)
        ego_vel = batch["ego_vel"].to(device)
        target_pos = batch["target_pos"].to(device)  # [B,T,2], already center-origin

        init_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles,
                                                head_direction_ensembles)
        out = model(init_conds, ego_vel)

        b, t, _ = target_pos.shape
        xy_all.append(target_pos.reshape(b * t, 2).cpu().numpy())
        bottleneck_all.append(out.bottleneck.reshape(b * t, -1).cpu().numpy())
        lstm_all.append(out.lstm_output.reshape(b * t, -1).cpu().numpy())

        n_collected += b
        if n_collected >= n_trajectories:
            break

    return (np.concatenate(xy_all, axis=0),
            np.concatenate(bottleneck_all, axis=0),
            np.concatenate(lstm_all, axis=0))


def score_units(scorer: GridScorer, xy: np.ndarray, activations: np.ndarray):
    """activations: [N, n_units] -> per-unit (score_60, ratemap, sac)."""
    n_units = activations.shape[1]
    ratemaps = [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], activations[:, i])
                for i in range(n_units)]
    results = [scorer.get_scores(rm) for rm in ratemaps]
    scores_60 = np.array([r[0] for r in results])
    sacs = [r[4] for r in results]
    mask_60 = [r[2] for r in results]
    return scores_60, ratemaps, sacs, mask_60


def compute_shuffle_thresholds(scorer: GridScorer, ratemaps, n_shuffles=100,
                                min_peak_frac=0.2, seed=None, progress_every=32):
    """Per-unit gridness thresholds via the paper's own null distribution
    (field-shuffle, see scores.py) instead of one fixed cutoff for every
    unit. Prints progress since this is the slow part of evaluation (each
    unit needs `n_shuffles` full SAC computations)."""
    rng = np.random.default_rng(seed)
    thresholds = np.empty(len(ratemaps))
    for i, rm in enumerate(ratemaps):
        thresholds[i] = scorer.shuffled_gridness_threshold(
            rm, n_shuffles=n_shuffles, min_peak_frac=min_peak_frac, rng=rng)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    shuffle threshold: {i + 1}/{len(ratemaps)} units done")
    return thresholds


def plot_units(scorer, ratemaps, sacs, mask_60, scores_60, out_path, title, cols=16,
               threshold=GRIDNESS_THRESHOLD):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    n_units = len(ratemaps)
    ordering = np.argsort(-scores_60)
    rows = int(np.ceil(n_units / cols))
    # threshold may be one shared cutoff (float) or one value per unit
    # (per-unit shuffle threshold, possibly NaN for units with no fields --
    # those never pass, since scores_60[index] > nan is always False).
    threshold_arr = np.broadcast_to(np.asarray(threshold, dtype=float), scores_60.shape)
    threshold_label = f"{float(threshold):.2f}" if np.ndim(threshold) == 0 else "per-unit shuffle 95th pct"
    fig = plt.figure(figsize=(24, rows * 4))
    fig.suptitle(f"{title}  (red = gridness > {threshold_label})")
    for i in range(n_units):
        index = ordering[i]
        unit_threshold = threshold_arr[index]
        passed = scores_60[index] > unit_threshold
        rf = plt.subplot(rows * 2, cols, i + 1)
        acr = plt.subplot(rows * 2, cols, n_units + i + 1)
        # cmap="jet" matches the paper's Fig. 1d ratemap/SAC color scheme
        # (blue=low -> red=high), not matplotlib's default viridis.
        rf_title = (f"{scores_60[index]:.2f}" if np.ndim(threshold) == 0
                    else f"{scores_60[index]:.2f}/{unit_threshold:.2f}")
        scorer.plot_ratemap(ratemaps[index], ax=rf, title=rf_title, cmap="jet")
        scorer.plot_sac(sacs[index], mask_params=mask_60[index], ax=acr, cmap="jet")
        if passed:
            rf.title.set_color("red")
            rf.title.set_fontweight("bold")
            for ax in (rf, acr):
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                        edgecolor="red", linewidth=2))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shard_dir", default="data/shards")
    parser.add_argument("--n_trajectories", type=int, default=4000)
    parser.add_argument("--nbins", type=int, default=32)
    parser.add_argument("--out_dir", default="eval")
    parser.add_argument("--shuffle_threshold", action=argparse.BooleanOptionalAction, default=True,
                         help="per-unit gridness threshold via the paper's field-shuffle null "
                              "distribution (Supplementary Methods 3d), instead of one fixed "
                              f"cutoff ({GRIDNESS_THRESHOLD}) for every unit. Slower: "
                              "n_units * n_shuffles extra SAC computations. Use --no-shuffle_threshold "
                              "for the old fast fixed-cutoff behavior.")
    parser.add_argument("--n_shuffles", type=int, default=100,
                         help="shuffles per unit for --shuffle_threshold (paper uses 100)")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    place_cell_ensembles, head_direction_ensembles = build_ensembles(cfg, device)
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    model = GridCellsRNN(
        target_ensembles=target_ensembles, nh_lstm=cfg.model.nh_lstm,
        nh_bottleneck=cfg.model.nh_bottleneck, dropout_rates=cfg.model.dropout_rates,
        bottleneck_has_bias=cfg.model.bottleneck_has_bias,
        init_weight_disp=cfg.model.init_weight_disp, ego_vel_dim=cfg.model.ego_vel_dim,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint from epoch {ckpt['epoch']}")

    loader = build_dataloader(args.shard_dir, shard_indices=None,
                               batch_size=cfg.train.minibatch_size, shuffle=True)

    xy, bottleneck_acts, lstm_acts = collect_activations(
        model, place_cell_ensembles, head_direction_ensembles, loader,
        args.n_trajectories, device)
    print(f"collected {xy.shape[0]} (trajectory,timestep) samples "
          f"from ~{args.n_trajectories} trajectories")

    half_env = cfg.task.env_size / 2.0
    coord_range = ((-half_env, half_env), (-half_env, half_env))
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10).tolist()
    mask_parameters = list(zip(starts, ends))
    scorer = GridScorer(args.nbins, coord_range, mask_parameters)

    for name, acts in [("bottleneck", bottleneck_acts), ("lstm", lstm_acts)]:
        print(f"scoring {acts.shape[1]} {name} units...")
        scores_60, ratemaps, sacs, mask_60 = score_units(scorer, xy, acts)
        n_grid_like_fixed = (scores_60 > GRIDNESS_THRESHOLD).sum()
        print(f"  {name}: mean score_60={scores_60.mean():.3f}, "
              f"max={scores_60.max():.3f}, "
              f"n_units with score_60>{GRIDNESS_THRESHOLD} (fixed threshold) = "
              f"{n_grid_like_fixed}/{len(scores_60)}")

        save_kwargs = {"scores_60": scores_60}
        if args.shuffle_threshold:
            print(f"  computing per-unit shuffle thresholds "
                  f"({args.n_shuffles} shuffles/unit, this is the slow part)...")
            thresholds = compute_shuffle_thresholds(
                scorer, ratemaps, n_shuffles=args.n_shuffles, seed=args.shuffle_seed)
            grid_like = scores_60 > thresholds  # nan threshold (no fields) -> never passes
            n_grid_like = int(np.sum(grid_like))
            print(f"  {name}: n_units grid-like (per-unit shuffle threshold) = "
                  f"{n_grid_like}/{len(scores_60)} "
                  f"(mean threshold={np.nanmean(thresholds):.3f}, "
                  f"{np.isnan(thresholds).sum()} units had no detectable fields)")
            plot_threshold = thresholds
            save_kwargs["shuffle_thresholds"] = thresholds
        else:
            plot_threshold = GRIDNESS_THRESHOLD
            save_kwargs["threshold"] = GRIDNESS_THRESHOLD

        out_path = os.path.join(args.out_dir, f"{name}_ratemaps_epoch{ckpt['epoch']}.pdf")
        plot_units(scorer, ratemaps, sacs, mask_60, scores_60, out_path,
                   title=f"{name} units, checkpoint epoch {ckpt['epoch']}", threshold=plot_threshold)
        print(f"  saved plot: {out_path}")
        np.savez(os.path.join(args.out_dir, f"{name}_scores_epoch{ckpt['epoch']}.npz"), **save_kwargs)


if __name__ == "__main__":
    main()
