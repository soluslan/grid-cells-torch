"""Train the supervised path-integration network (paper Fig. 1).

Default run: 300 epochs x 1000 steps = Supplementary Table 1's 300,000 updates (~30 min on a
GTX 1650). Checkpoints go to --results-dir; per-epoch mean losses to <results-dir>/epoch_losses.csv.

Examples:
    python train.py
    python train.py --results-dir data/checkpoints/dropout0 --dropout 0.0
    python train.py --epochs 2 --steps-per-epoch 50 --results-dir /tmp/smoke
"""
import argparse
import csv
import os
import time

from config import Config
from grid_cells.training import train


def default_shard_dir() -> str:
    for path in ("data/latest/shards", "data/shards"):
        if os.path.isdir(path):
            return path
    return "data/latest/shards"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shard-dir", default=None,
                        help="directory of *.npz shards (default: data/latest/shards, "
                             "falling back to data/shards)")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.results_dir:
        cfg.train.results_dir = args.results_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.steps_per_epoch is not None:
        cfg.train.steps_per_epoch = args.steps_per_epoch
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.dropout is not None:
        cfg.model.dropout_rates = (args.dropout,)
    shard_dir = args.shard_dir or default_shard_dir()

    print(f"shards: {shard_dir} | results: {cfg.train.results_dir}")
    start = time.time()
    losses = train(cfg, shard_dir)
    with open(os.path.join(cfg.train.results_dir, "epoch_losses.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "mean_loss"])
        writer.writerows(enumerate(losses))
    print(f"done in {time.time() - start:.0f}s: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
