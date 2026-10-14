---
paths:
  - "grid_cells/rl_*.py"
  - "grid_cells/actor_critic.py"
  - "grid_cells/vision.py"
  - "grid_cells/replay_buffer.py"
  - "grid_cells/vec_env.py"
  - "train_rl_agent.py"
  - "eval_rl_agent.py"
  - "levels/**"
  - "rl_train.py"
  - "actor_critic.py"
  - "vision.py"
  - "replay_buffer.py"
---

# RL agent files

- Read `docs/paper/spec-rl-agent.md` and the RL sections of `docs/paper/deviations.md` before
  changing behaviour here.
- Anything you claim matches the paper: grep the cited paragraph in `docs/paper/text/main.txt`
  or `supp.txt` and confirm it. For levels, open `docs/paper/figures/fig2.png`.
- Gradient flow: stop-gradient from vision into the grid network and from the grid network into
  the actor–critic (ED Fig. 5). Vision and grid are trained only by their own supervised losses.
- Units: one environment step is one frame (1/60 s); one action = 4 frames; budgets and
  `step_counter` are in frames.
- `grid_cells/rl_a3c.py` is the frozen legacy loop kept as the A3C baseline. Do not fix bugs in
  it; tests mark its known defects as `xfail(strict=True)`.
