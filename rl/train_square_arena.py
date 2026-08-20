"""Real training run for the RL-agent roadmap plan's M5: the paper's literal spec (32 actors,
1e9 env steps, Supplementary Table 2 hyperparameters -- all Config() defaults already match).
Checkpoints to cfg.rl.results_dir every checkpoint_every_env_steps; per-actor episode-reward
logs and vision/grid loss logs land in the same directory. See rl_train.py's module docstring
for the process architecture.

Sets OMP/MKL/OPENBLAS thread-count env vars *before* importing torch -- belt-and-suspenders
alongside rl_train.py's own `_cap_thread_pools()` (`torch.set_num_threads(1)`) called inside
each worker; env vars must be set pre-import to reliably constrain the native BLAS libraries'
own internal thread pools, which torch's own API doesn't fully control on all backends. See the
RL-agent roadmap plan's "Throughput fix" section.

Run from grid-cells-torch/: python rl/train_square_arena.py
To resume a stopped run: python rl/train_square_arena.py --resume-from data/checkpoints/rl_baseline/checkpoint_step{N}_final.pt
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys

sys.path.insert(0, ".")

from config import Config
from rl_train import train_rl_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", default=None,
                         help="checkpoint*.pt path to resume from (must include optimizer "
                              "state, i.e. saved by this codebase's checkpoint_worker); "
                              "omit to start fresh at step 0")
    args = parser.parse_args()

    cfg = Config()  # defaults: num_actors=32, total_env_steps=1e9, results_dir=data/checkpoints/rl_baseline
    cfg.rl.resume_from = args.resume_from
    train_rl_agent(cfg)
    print("training run complete")
