"""Analyse a trained supervised checkpoint against the paper's Fig. 1 / Extended Data Fig. 3.

Analyses (--analysis, repeatable; default all):
    path        decoded-position error vs time, trained vs untrained (Fig. 1b,c)
    gridness    gridness of every bottleneck and LSTM unit, score histogram, ratemap PDFs (Fig. 1d,g)
    cell-types  border, head-direction, conjunctive, grid-scale clustering, stability (Fig. 1e-g, ED 3)

Figures go to --out-dir (default: results/<run>/ paired with data/checkpoints/<run>/).

Examples:
    python analyze_supervised.py --checkpoint data/checkpoints/baseline/checkpoint_epoch299.pt
    python analyze_supervised.py --checkpoint ... --analysis path --n-trajectories 1000
"""
import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from grid_cells import evaluate, figures
from grid_cells.dataset import build_dataloader
from grid_cells.scores import BORDER_BINS, BORDER_THRESHOLD, HD_THRESHOLD, score_units
from train import default_shard_dir

ANALYSES = ("path", "gridness", "cell-types")
NBINS = 32


def run_path(cfg, args, model, untrained, epoch, place, hd, batches, device, out_dir):
    true_pos = np.concatenate([b["target_pos"].numpy() for b in batches], axis=0)
    err, decoded, _, _ = evaluate.path_integration_errors(
        model, untrained, batches, place, hd, device, want_activations=False)
    evaluate.report_path_integration(err)
    mode = evaluate.best_decoder(err)
    fig = figures.plot_path_integration(err, true_pos, decoded, mode, evaluate.DECODERS)
    path = out_dir / f"path_integration_epoch{epoch}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("saved", path)


def score_layers(model, batches, place, hd, device, cfg):
    true_pos = np.concatenate([b["target_pos"].numpy() for b in batches], axis=0)
    headings = np.concatenate([b["target_hd"].numpy() for b in batches]).reshape(-1)
    xy = true_pos.reshape(-1, 2)
    _, bottleneck_acts, lstm_acts = evaluate.run_model(
        model, batches, place, hd, device, want_activations=True)
    scorer = evaluate.build_scorer(cfg, NBINS)
    scored, tuned = {}, {}
    for name, acts in (("bottleneck", bottleneck_acts), ("lstm", lstm_acts)):
        flat = acts.reshape(-1, acts.shape[-1])
        print(f"scoring {flat.shape[1]} {name} units ...")
        scored[name] = score_units(scorer, xy, flat)
        tuned[name] = evaluate.score_directional(headings, flat)
    return xy, bottleneck_acts.reshape(-1, bottleneck_acts.shape[-1]), scored, tuned


def run_gridness(epoch, scored, tuned, out_dir):
    for name, (s60, _, _, _) in scored.items():
        evaluate.report_gridness(name, s60)
        evaluate.report_directional(name, tuned[name][1])
    print("paper: gridness 129/512 (25.2%) in the bottleneck; head direction 10.2%")

    fig, ax = plt.subplots(figsize=(9, 4))
    figures.plot_score_distribution({k: v[0] for k, v in scored.items()},
                                    evaluate.GRIDNESS_THRESHOLD, ax=ax)
    ax.set_title(f"gridness distribution, epoch {epoch}")
    path = out_dir / f"gridness_epoch{epoch}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("saved", path)

    for name, (s60, rm, sc, mp) in scored.items():
        path = figures.save_unit_pdf(
            str(out_dir / f"{name}_ratemaps_epoch{epoch}.pdf"),
            ratemaps=rm, sacs=sc, mask_params=mp, scores=s60, nbins=NBINS, cols=16,
            threshold=evaluate.GRIDNESS_THRESHOLD, hd_tuning=tuned[name][0],
            title=f"{name} units, epoch {epoch} "
                  f"(red = gridness > {evaluate.GRIDNESS_THRESHOLD})")
        print("saved", path)


def run_cell_types(cfg, args, epoch, xy, acts, scored, tuned, batches, device, out_dir):
    scores_60, _, sacs, _ = scored["bottleneck"]
    resultants = tuned["bottleneck"][1]
    border = evaluate.score_borders(cfg, xy, acts, nbins=BORDER_BINS)
    evaluate.report_border("bottleneck", border)
    n_conj = evaluate.report_conjunctive("bottleneck", scores_60, resultants)

    scales = evaluate.grid_scales(sacs, cfg, NBINS, scores_60 > evaluate.GRIDNESS_THRESHOLD)
    evaluate.report_grid_scales("bottleneck", scales)
    _, means, bics, _, _, _, _ = evaluate.report_scale_clustering(
        "bottleneck", scales, n_shuffles=500, seed=cfg.train.seed)

    stab, ep_early = None, None
    if args.early_checkpoint and os.path.exists(args.early_checkpoint):
        rm_late, _ = evaluate.layer_ratemaps(cfg, args.checkpoint, batches, device, NBINS)
        rm_early, ep_early = evaluate.layer_ratemaps(cfg, args.early_checkpoint, batches,
                                                     device, NBINS)
        stab = evaluate.stability(rm_early, rm_late)
        evaluate.report_stability("bottleneck", stab, scores_60, resultants)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    figures.plot_gridness_vs_directional(scores_60, resultants, evaluate.GRIDNESS_THRESHOLD,
                                         HD_THRESHOLD, ax=axes[0, 0])
    axes[0, 0].set_title(f"gridness vs directional tuning ({n_conj} conjunctive)", fontsize=10)
    figures.plot_scale_distribution(scales, means, bics, ax=axes[0, 1])
    axes[1, 0].hist(border, bins=40, color="#1a4f7a", alpha=0.7)
    axes[1, 0].axvline(BORDER_THRESHOLD, color="k", ls="--", lw=1)
    axes[1, 0].set_xlabel("border score")
    axes[1, 0].set_ylabel("units")
    axes[1, 0].set_title(f"border cells: {int((border > BORDER_THRESHOLD).sum())}/"
                         f"{len(border)} > {BORDER_THRESHOLD}  [paper: 8.7%]", fontsize=10)
    if stab is not None:
        figures.plot_stability(stab, scores_60, resultants, evaluate.GRIDNESS_THRESHOLD,
                               HD_THRESHOLD, ax=axes[1, 1])
        axes[1, 1].set_title(f"stability, epoch {ep_early} vs {epoch}", fontsize=10)
    else:
        axes[1, 1].axis("off")
    fig.suptitle(f"Cell types in the linear layer, epoch {epoch} ({acts.shape[1]} units)",
                 fontsize=13)
    fig.tight_layout()
    path = out_dir / f"cell_types_epoch{epoch}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--early-checkpoint", default=None,
                        help="earlier checkpoint for the stability analysis (paper: 2e5 vs 3e5 "
                             "steps); default: checkpoint_epoch200.pt next to --checkpoint")
    parser.add_argument("--analysis", action="append", choices=ANALYSES + ("all",))
    parser.add_argument("--n-trajectories", type=int, default=4000)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    analyses = set(args.analysis or ["all"])
    if "all" in analyses:
        analyses = set(ANALYSES)
    if args.early_checkpoint is None:
        args.early_checkpoint = os.path.join(os.path.dirname(args.checkpoint),
                                             "checkpoint_epoch200.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    out_dir = Path(args.out_dir or evaluate.default_out_dir(args.checkpoint))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, untrained, epoch, place, hd = evaluate.load_models(cfg, args.checkpoint, device)
    loader = build_dataloader(args.shard_dir or default_shard_dir(), shard_indices=None,
                              batch_size=cfg.train.minibatch_size, shuffle=True)
    batches, n_traj = evaluate.collect_batches(loader, args.n_trajectories)
    print(f"epoch {epoch} | {n_traj} trajectories | device {device} | out {out_dir}")

    if "path" in analyses:
        run_path(cfg, args, model, untrained, epoch, place, hd, batches, device, out_dir)
    if analyses & {"gridness", "cell-types"}:
        xy, acts, scored, tuned = score_layers(model, batches, place, hd, device, cfg)
        if "gridness" in analyses:
            run_gridness(epoch, scored, tuned, out_dir)
        if "cell-types" in analyses:
            run_cell_types(cfg, args, epoch, xy, acts, scored, tuned, batches, device, out_dir)


if __name__ == "__main__":
    main()
