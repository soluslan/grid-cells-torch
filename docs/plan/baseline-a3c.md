# Baseline: the A3C run before the A2C switch

Frozen record of the last A3C run, kept as the "before" reference for every later comparison.
Code at commit `55ef4ea` (`rl_train.py`, preserved after Phase 3 as `grid_cells/rl_a3c.py`).

## Run

- Command: `python train_rl_agent.py` (32 Hogwild actor processes + vision/grid/checkpoint
  workers, CPU only, `CUDA_VISIBLE_DEVICES=""`), resumed several times.
- Level: `levels/square_arena.lua`. Budget: 1e9 environment frames (action repeat 4).
- Finished 2026-10-02 ~03:00, `data/checkpoints/rl_baseline/checkpoint_step1000000092_final.pt`
  (19.7GB, weights + optimizers + both float32 replay buffers). 1,870 light
  `checkpoint_step{N}.pt` files (~34.6MB each) in the same directory.
- Throughput: ~1,000 frames/s (1e9 frames in ~11.6 days of wall time).

## Learning curve: flat

Per-actor logs `actor_{rank}_episodes.csv` (`step,episode_reward`), 185,299 episodes. A goal
arrival is +10, so "goal rate" is the fraction of episodes with any reward.

| frames | episodes | mean reward | goal rate | max |
|---|---|---|---|---|
| 0–100M | 18563 | 0.932 | 7.6% | 90 |
| 100–200M | 18584 | 0.952 | 7.6% | 70 |
| 200–300M | 18520 | 0.913 | 7.6% | 60 |
| 300–400M | 18515 | 0.944 | 7.8% | 60 |
| 400–500M | 18521 | 0.873 | 7.3% | 70 |
| 500–600M | 18519 | 0.931 | 7.7% | 70 |
| 600–700M | 18518 | 0.983 | 8.0% | 70 |
| 700–800M | 18522 | 0.951 | 7.8% | 90 |
| 800–900M | 18515 | 0.974 | 8.1% | 60 |
| 900–1000M | 18522 | 0.910 | 7.5% | 50 |

Overall mean 0.936, goal rate 7.7%. The paper's square-arena agent reaches ~289 (Fig. 2f,
Supplementary Results 1a). No trend at any point in the budget.

## Behaviour

`results/rl_eval_step199832848/` (5 episodes, seeds 0–4, checkpoint at step 199,832,848):
first-person and top-down videos plus Fig. 2h-style trajectory plots. In 5/5 episodes the
agent walks straight to a wall within ~10s of spawning and then stays still for the
remaining ~80s, with position and view fixed to the pixel.

## Defects found in this code (plan §6)

| ID | Location (`rl_train.py` at `55ef4ea`) | Defect |
|---|---|---|
| A | line 165 | n-step bootstrap uses `V(s_t)` instead of `V(s_{t+n})` |
| B | line 108 | reward input to the policy LSTM is never updated, always 0 |
| C | lines 155–156 | grid LSTM re-initialized on teleport, from the pre-teleport frame's vision output |
| D | lines 142–145 | goal grid code and previous action not cleared at episode end |
| E | lines 157, 165 | goal arrival treated as terminal (policy LSTM reset, bootstrap 0) |
| F | `vision.py` 47–50 | per-unit 95% masking; paper wording is ambiguous |
| G | `replay_buffer.py` | grid sequences are cut at teleports, so the grid net never sees one |

## Learner update counts (from the logs, approximate — logs mix several resumes)

- vision: ~4.5M minibatch updates over 1e9 frames, i.e. one per ~250 frames
- grid: ~153k minibatch updates over 1e9 frames, i.e. one per ~6,500 frames

These ratios seed the A2C defaults `vision_updates_per_rollout` / `grid_updates_per_rollout`.

## Preservation

Nothing in `data/checkpoints/rl_baseline/` is to be deleted as part of this plan.
