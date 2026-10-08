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
- Random-policy baselines (uniform over the 6 actions): square_arena ~13 per episode (6 episodes;
  100-episode run pending), tiny_goal ~17. The 1e9-frame A3C agent (0.93) was far *below* random.
- Phase 7 gate, first attempt (tiny_goal, Table 2 settings, RMSProp momentum 0.99): **failed**.
  Policy entropy collapsed from 1.79 to 0.001 within ~600k frames (47 updates) and the agent
  stopped moving (still fraction 1.00) — the same "freeze" as the A3C run, now with defects A–G
  fixed. Prime suspect: Table 2's "gradient momentum 0.99" applied as RMSProp *momentum* (on top
  of decay 0.99), which multiplies the effective step ~100x. Diagnostic: momentum 0
  (`--ac-momentum 0`), everything else unchanged.

## Next

- Read out `data/checkpoints/diag_tiny_mom0` (3e6 frames, momentum 0). If entropy stays healthy
  and reward rises above the random level, rerun the full tiny_goal gate (1e7 frames) with
  momentum 0, then decide whether momentum 0 becomes the default (deviations.md entry).
- Then Phase 8: square_arena from scratch with milestone gates.

## Open questions

- The square-arena goal is a visible object in our level; the paper's is unmarked (deviation I).
  Making it invisible needs a position-triggered reward. Decide before Phase 8.

- Commit paper PDFs to the public repo or not (default: no).
- Delete `data/square_room_100steps_2.2m_1000000/` (17GB CSV) after Phase 4 verification?
- Delete `lab/build_dmlab.sh` and its two build logs after `scripts/setup_dmlab.sh` absorbs them?
