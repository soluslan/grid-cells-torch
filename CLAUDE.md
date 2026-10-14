# CLAUDE.md — grid-cells-torch

PyTorch reimplementation of Banino et al. (2018), "Vector-based navigation using grid-like
representations in artificial agents" (Nature 557:429): the supervised path-integration network
and a from-scratch RL agent in DeepMind Lab. Citation and the paper's local files:
`docs/paper/README.md`.

## Start here

1. `docs/plan/STATUS.md` — where the current plan stands and what to do next. Read it first in
   every new session and after every compaction; update it at the end of each phase.
2. `docs/plan/2026-10-goal-finding.md` — the current plan (Korean).
3. `docs/maintenance.md` — package layout, entry points, conventions, commands.

Do **not** read the whole paper, every figure, and all the code at session start. Use the table
below to read only what the task needs.

## What to read for which task

| Task | Read |
|---|---|
| RL agent code (`rl_*`, `actor_critic`, `vision`, `replay_buffer`, `vec_env`) | `docs/paper/spec-rl-agent.md`, `docs/paper/deviations.md` (RL sections), ED Fig. 5 entry in `docs/paper/figures.md` |
| DMLab level (`levels/*.lua`) | `spec-rl-agent.md` "Environment", Fig. 2 entry in `figures.md` **and open `docs/paper/figures/fig2.png`**; for DMLab APIs, the `lab/` reading list in `../CLAUDE.md` |
| Supervised network, dataset, training | `docs/paper/spec-supervised.md`, `deviations.md` (supervised sections) |
| Analysis / scoring (`scores`, `evaluate`, `figures`) | `spec-supervised.md` "Analysis definitions", Fig. 1 / ED 3 entries in `figures.md` |
| Port fidelity vs the original TF code | the matching file in `../grid-cells/` (reference only, never modify) |
| Comparing results with the paper | "Reported results" tables in the two spec notes |

## The one hard rule about the paper

Before writing or changing code that claims to follow the paper, open the **primary source** the
spec note cites — `grep` the paragraph in `docs/paper/text/main.txt` / `supp.txt` (page markers
`===== PAGE N =====`), and open the figure image when the spec note or `figures.md` says the image
matters. Spec notes are summaries and can be wrong; the PDF and figures are ground truth. If the
paper is silent or ambiguous, say so and record the choice in `docs/paper/deviations.md`.

If `docs/paper/text/` or the PDFs are missing (they are gitignored), regenerate them as described
in `docs/paper/README.md`; the originals are in `../supplements/논문/` and `../supplements/figures/`.

## Working rules

- Tests: `pytest -m "not dmlab and not slow"` for the fast suite, `pytest` for everything. A test
  that guards a bug must be seen failing on the buggy code first.
- Never start a long training run without written milestone gates (see the plan).
- `data/` is large and gitignored; never delete anything under `data/checkpoints/` without asking.
- Python environment: conda `env1` (`/home/junoLee/miniconda3/envs/env1/bin/python`), which has
  `torch` (CUDA), `deepmind_lab`, and `pytest`. The system `python3` has neither torch nor DMLab.
