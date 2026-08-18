"""M4 verification: does the real multiprocess wiring (torch.multiprocessing spawn, shared
model/optimizer state across actual OS process boundaries) survive a short run without
deadlocking or crashing? Small num_actors + tiny total_env_steps, per the roadmap plan's M4
verification step ("short smoke run e.g. 1e5 steps instead of 1e9").

Run from grid-cells-torch/: python rl/mp_smoke_test.py
"""

import sys
import time

sys.path.insert(0, ".")

from config import Config
from rl_train import train_rl_agent


def main():
    cfg = Config()
    cfg.rl.env.num_actors = 2
    cfg.rl.total_env_steps = 2000  # tiny; just needs to exercise every code path a few times
    cfg.rl.actor_critic.n_step = 20
    cfg.rl.replay.frame_capacity = 2000
    cfg.rl.replay.sequence_capacity = 20

    start = time.time()
    train_rl_agent(cfg)
    print(f"mp_smoke_test: all processes joined cleanly in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
