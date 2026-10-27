# Status — goal-finding plan

Plan: [`2026-10-goal-finding.md`](2026-10-goal-finding.md) (Korean). This file tracks where
execution currently stands; update it at the end of every phase and commit it with that
phase's work.

Updated: 2026-10-27

## Phases

- [x] 0 plan copied into repo, this file created (account switch deferred — plan §2.3)
- [x] 1 baseline frozen (`docs/plan/baseline-a3c.md`)
- [x] 2 context restructure (`docs/paper/`, CLAUDE.md, `.claude/rules/`)
- [x] 3 package restructure (`grid_cells/`, entry points, pytest, notebooks removed, `scripts/setup_dmlab.sh`)
- [x] 4 direct simulator -> npz data path
- [x] 5 KAT + wiring tests, A3C red lights confirmed
- [x] 6 A2C implementation
- [x] 7 gates: toy env, tiny_goal (passed 2026-10-09 after the spawn fix)
- [x] 8 square-arena training, full 1e9 frames (finished 2026-10-14)
- [ ] 9 gridness shortfall investigation (parallel)
- [ ] 10 comparison with the paper

## Decisions made

- 2026-10-08: switch training from A3C (CPU Hogwild) to A2C (GPU, synchronous). A3C code is
  kept frozen as `grid_cells/rl_a3c.py` for baseline reproduction and diagnosis.
- 2026-10-08: unroll length stays 100, action repeat 4 (Supplementary Table 2, p.24 lists them
  separately; Supplementary 3c defines the n-step return with `γ^n V(s_{t+n})`).
- 2026-10-08: lab-account switch deferred until SSH/direct access is available (currently
  Remote Control only).
- 2026-10-09: the square-arena goal is drawn invisibly (`--hidden-goal`) for all
  paper comparisons; a visible goal would let the policy CNN bypass the grid code
  (deviations.md, item I).
- 2026-10-08: notebooks are deleted after conversion to CLI scripts; notebook 06 becomes
  `scripts/setup_dmlab.sh`. The uncommitted notebook diffs (01/03/04/05) were only output-path
  strings (`/home/user` -> `/home/junoLee`), nothing to carry over.
- 2026-10-08: paper PDFs/figures/text extracts are gitignored by default; citation and own
  notes are committed. Re-confirm with the user before Phase 2 commit.

## Results so far

- Phase 2: spec notes verified line by line by a separate agent; 7 citation/precision issues
  fixed. Workspace CLAUDE.md no longer requires reading everything up front.
- Phase 3: package move is bit-identical (`tests/test_golden.py`). `scripts/setup_dmlab.sh` +
  patch rebuilt DeepMind Lab from a fresh worktree of the pinned commit (bazel build OK, wheel OK).
- Phase 4: `generate_data.py` regenerates shard 0 bit-identical to `data/shards/0000-of-0099.npz`.
- Phase 5: all six wiring contracts fail on frozen A3C for the documented reasons (strict xfail).
  Scorer KAT: clean hexagonal lattices score 1.1–1.7 at spacings 0.3–1.2 m, noise < 0.37, so the
  supervised gridness shortfall (7.8%) is not a scorer failure on clean grids (Phase 9 input).
- Newly identified while writing the A2C plan: replayed grid sequences in A3C are initialised
  from ground-truth place/HD cells (`train_step`), while the paper sets the grid LSTM to zero
  (main p.7). A2C trains from a zero state (deviations.md, item H).

- Phase 6: `grid_cells/rl_a2c.py` + `vec_env.py` + `rl_common.py`. All wiring contracts pass for
  A2C; each paper-violating ablation flag is caught by its contract; A2C learns a bandit with
  Table 2 hyperparameters (`test_a2c_learns_a_bandit_with_paper_hyperparameters`).
  Throughput on the GTX 1650 with 32 envs: ~4,600 frames/s (gate: 4,000) → 1e9 frames ≈ 60 h.
  Per rollout (12,800 frames, ~2.8 s): env stepping 1.0 s, batched forward 1.0 s, AC update
  0.5 s, 51 vision updates 0.3 s, 2 grid updates 0.05 s. GPU memory 0.73 GB.
- Random-policy baselines (uniform over the 6 actions, 100 episodes each): square_arena
  **5.40** (goal found in 38% of episodes), tiny_goal **17.2** (77%). The 1e9-frame A3C agent
  (0.93) scored far *below* a random policy.
- Phase 7 gate, attempt 1 (tiny_goal, RMSProp momentum 0.99): **failed** — entropy 1.79 → 0.001
  within ~6e5 frames, agent froze (still fraction 1.00): the A3C "freeze" again, with A–G fixed.
- Diagnostic, momentum 0 (only change), 3e6 frames: entropy stayed 1.4–1.6, reward per episode
  17.5 → 28.7 → 35.7 → 48.0 → 46.9 → 46.8 per 0.5M frames (~2.7x random by 1.5M, then flat).
  Momentum 0 is now the A2C default (deviations.md).
- Phase 7 gate, attempt 2 (momentum 0, lr 1e-4, 1e7 frames, `gate_tiny_goal_v2`): **not passed**.
  Reward per 1M frames: 30 → 48 → 49 → 56 → 56 → 55 → 57 → 51 → 57 → 56. Entropy 1.1–1.5 (no
  collapse), still fraction 0.3–0.6. Learns to ~3.2x random by 3e6 frames, then flat for 7e6.
  Eval of the final checkpoint at native 64 px: 0 / 100 / 30 over seeds 0–2.
- Diagnostic lr 2e-4 (Table 2 upper end), 3.5e6 frames: same plateau (~55–60) reached faster
  (45 at 0.5M vs 29). Learning rate is not what caps performance.
- Bandit test still passes with momentum 0.
- The square-arena goal object turned out to be visible (deviation I); `--hidden-goal` added.

- Phase 7: the first two gate attempts stalled at ~55 on tiny_goal. The cause was the level,
  not the learner: spawn points could land inside wall geometry (within the player's ~16-unit
  collision radius), freezing the agent for the rest of the episode — about 8% of spawn
  segments. With the spawn margin fixed (`levels/tiny_goal.lua`, commit 974714a) and nothing
  else changed, the score went 55 -> 113 against a random-policy baseline of 22.6, passing the
  5x gate. `tests/test_levels.py` pins the clearance.
- Phase 8: square arena with `--hidden-goal`, from scratch, **completed the full 1e9 frames**
  on 2026-10-14 (`data/checkpoints/a2c_square_hidden/checkpoint_step1000000000_final.pt`,
  185,184 episodes, ~4,500 frames/s at the end). Mean episode reward per 100M frames:

  | frames | 100M | 200M | 300M | 400M | 500M | 600M | 700M | 800M | 900M | 1000M |
  |---|---|---|---|---|---|---|---|---|---|---|
  | reward | 38 | **81** | 28 | 27 | 27 | 28 | 30 | 30 | 31 | 31 |

  The agent learns — unlike the frozen A3C loop, which never left 0.9 — but performance peaked
  around 2e8 frames (85.0 mean, 8.45 goals/episode) and fell to ~30 (32.5 mean, 3.24
  goals/episode) for the remaining 8e8 without recovering. Four earlier dips between 6e7 and
  1.2e8 did recover on their own; the fifth did not. Random policy on the same level: 5.40.
  Paper: 289.

## Next

- **Explain the collapse at ~2e8 frames** (see Results). Nothing in the logs identifies a
  cause yet; entropy did not collapse and the still-fraction stayed low. This blocks any
  claim about final performance and is the first thing to investigate.
- Phase 9 (gridness shortfall) and Phase 10 (comparison with the paper) are untouched.
- Codebase study sessions by region are planned; see `docs/study/` once it exists.

## Open questions

- Commit paper PDFs to the public repo or not (default: no).
- Delete `data/square_room_100steps_2.2m_1000000/` (17GB CSV) after Phase 4 verification?
- Delete `lab/build_dmlab.sh` and its two build logs after `scripts/setup_dmlab.sh` absorbs them?
