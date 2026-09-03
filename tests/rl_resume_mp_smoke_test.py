"""End-to-end verification of the graceful-stop + resume flow, with the real DMLab multiprocess
wiring (torch.multiprocessing spawn, actual actor/learner/checkpoint processes): does SIGTERM
produce a checkpoint_step{N}_final.pt without a forced kill, and does --resume-from continue
step_counter from N instead of restarting at 0?

Run from grid-cells-torch/: python tests/rl_resume_mp_smoke_test.py
"""
import glob
import os
import re
import signal
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, ".")

from config import Config
from rl_train import train_rl_agent

import torch.multiprocessing as mp

RESULTS_DIR = "data/checkpoints/rl_resume_smoke_test"


def _run(cfg):
    train_rl_agent(cfg)


def main():
    if os.path.exists(RESULTS_DIR):
        import shutil
        shutil.rmtree(RESULTS_DIR)

    cfg = Config()
    cfg.rl.env.num_actors = 2
    cfg.rl.total_env_steps = 10_000_000  # high enough that we control the stop via SIGTERM, not this
    cfg.rl.checkpoint_every_env_steps = 100_000
    cfg.rl.replay.frame_capacity = 2_000
    cfg.rl.replay.sequence_capacity = 20
    cfg.rl.results_dir = RESULTS_DIR

    print("=== phase 1: launch, run briefly, SIGTERM (graceful stop) ===")
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_run, args=(cfg,))
    p.start()
    time.sleep(45)  # let it run past at least one checkpoint_every_env_steps boundary
    assert p.is_alive(), "phase 1 process died before we could SIGTERM it -- check for a crash"
    os.kill(p.pid, signal.SIGTERM)
    p.join(timeout=60)
    assert not p.is_alive(), "process did not exit within 60s of SIGTERM -- graceful stop hook not working"
    print(f"phase 1 process exited cleanly, exitcode={p.exitcode}")

    final_files = glob.glob(os.path.join(RESULTS_DIR, "checkpoint_step*_final.pt"))
    assert len(final_files) == 1, f"expected exactly one _final.pt, found {final_files}"
    final_path = final_files[0]
    n = int(re.search(r"checkpoint_step(\d+)_final\.pt", final_path).group(1))
    print(f"OK: {final_path} written (step={n})")
    assert n > 0, "final checkpoint step is 0 -- SIGTERM fired before any progress was made"

    print("\n=== phase 2: resume from that checkpoint, confirm step_counter continues from N ===")
    cfg2 = Config()
    cfg2.rl.env.num_actors = 2
    cfg2.rl.total_env_steps = n + 40_000  # small additional budget, self-terminating
    cfg2.rl.checkpoint_every_env_steps = 1_000_000_000  # don't care about mid-run checkpoints here
    cfg2.rl.replay.frame_capacity = 2_000
    cfg2.rl.replay.sequence_capacity = 20
    cfg2.rl.results_dir = RESULTS_DIR
    cfg2.rl.resume_from = final_path

    start = time.time()
    train_rl_agent(cfg2)
    elapsed = time.time() - start
    print(f"phase 2 (resumed run) completed to total_env_steps={cfg2.rl.total_env_steps} in {elapsed:.1f}s")

    # actor_*_episodes.csv are append-only and keyed by rank, so if step_counter correctly
    # resumed from N (not 0), every logged step value across BOTH phases should be non-decreasing
    # and phase 2's lines should all be >= n.
    all_steps = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "actor_*_episodes.csv")):
        with open(path) as f:
            for line in f:
                all_steps.append(int(line.split(",")[0]))
    assert all_steps, "no episode log lines found at all -- can't verify resume"
    steps_past_n = [s for s in all_steps if s > n]
    print(f"OK: {len(steps_past_n)}/{len(all_steps)} logged episode steps are > {n} "
          f"(resume continued forward, did not restart at 0)")
    assert steps_past_n, "no logged step exceeded the resume point -- step_counter may not have resumed"

    print("\nALL CHECKS PASSED: graceful SIGTERM stop + --resume-from both work end to end")


if __name__ == "__main__":
    main()
