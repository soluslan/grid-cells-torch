# Status — goal-finding plan

Plan: [`2026-10-goal-finding.md`](2026-10-goal-finding.md) (Korean). This file tracks where
execution currently stands; update it at the end of every phase and commit it with that
phase's work.

Updated: 2026-10-08

## Phases

- [x] 0 plan copied into repo, this file created (account switch deferred — plan §2.3)
- [x] 1 baseline frozen (`docs/plan/baseline-a3c.md`)
- [ ] 2 context restructure (`docs/paper/`, CLAUDE.md, `.claude/rules/`)
- [ ] 3 package restructure (`grid_cells/`, entry points, pytest, notebooks removed, `scripts/setup_dmlab.sh`)
- [ ] 4 direct simulator -> npz data path
- [ ] 5 KAT + wiring tests, A3C red lights confirmed
- [ ] 6 A2C implementation
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

## Next

- Phase 2: context restructure (`docs/paper/`, CLAUDE.md).

## Open questions

- Commit paper PDFs to the public repo or not (default: no).
- Delete `data/square_room_100steps_2.2m_1000000/` (17GB CSV) after Phase 4 verification?
- Delete `lab/build_dmlab.sh` and its two build logs after `scripts/setup_dmlab.sh` absorbs them?
