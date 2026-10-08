# Status — goal-finding plan

Plan: [`2026-10-goal-finding.md`](2026-10-goal-finding.md) (Korean). This file tracks where
execution currently stands; update it at the end of every phase and commit it with that
phase's work.

Updated: 2026-10-08

## Phases

- [x] 0 plan copied into repo, this file created (account switch deferred — plan §2.3)
- [x] 1 baseline frozen (`docs/plan/baseline-a3c.md`)
- [x] 2 context restructure (`docs/paper/`, CLAUDE.md, `.claude/rules/`)
- [x] 3 package restructure (`grid_cells/`, entry points, pytest, notebooks removed, `scripts/setup_dmlab.sh`)
- [x] 4 direct simulator -> npz data path
- [x] 5 KAT + wiring tests, A3C red lights confirmed
- [x] 6 A2C implementation
- [ ] 7 gates: toy env, tiny_goal
- [ ] 8 square-arena training with milestone gates
- [ ] 9 gridness shortfall investigation (parallel)
- [ ] 10 comparison with the paper

## Decisions made

- 2026-10-08: switch training from A3C (CPU Hogwild) to A2C (GPU, synchronous). A3C code is
  kept frozen as `grid_cells/rl_a3c.py` for baseline reproduction and diagnosis.
- 2026-10-08: unroll length stays 100, action repeat 4 (Supplementary Table 2, p.24 lists them
  separately; Supplementary 3c defines the n-step return with `γ^n V(s_{t+n})`).
- 2026-10-08: lab-account switch deferred until SSH/direct access is available (currently
  Remote Control only).
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

## Next (needs the user's decisions)

- Plateau at ~3x random on tiny_goal. Candidate causes not yet tested: (1) not enough frames —
  A3C-family agents in DMLab usually need several 1e7 frames even for simple tasks; (2) entropy
  cost (Table 2's 6e-5–1e-4 is tiny next to +10 rewards); (3) vision mask (F); (4) reward scale.
  Cheapest informative next step: tiny_goal at lr 2e-4 for 3e7 frames (~2 h).
- Square-arena milestone gates must be re-based: a random policy already scores 5.4, so the old
  "1e7 frames >= 5" gate is meaningless. Proposed: 1e7 >= 11 (2x random), 5e7 >= 50,
  1e8 >= 150, 2.5e8 >= 250.
- Decide visible vs hidden goal before Phase 8.

## Open questions

- The square-arena goal is a visible object in our level; the paper's is unmarked (deviation I).
  Making it invisible needs a position-triggered reward. Decide before Phase 8.

- Commit paper PDFs to the public repo or not (default: no).
- Delete `data/square_room_100steps_2.2m_1000000/` (17GB CSV) after Phase 4 verification?
- Delete `lab/build_dmlab.sh` and its two build logs after `scripts/setup_dmlab.sh` absorbs them?
