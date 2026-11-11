# Maintenance notes

Conventions for working in this repository. Commands: `bash run_scripts.sh`.

## Layout and package boundaries

- Root `*.py` files are the only entry points: `generate_data.py`, `train.py`,
  `analyze_supervised.py`, `train_rl_agent.py`, `eval_rl_agent.py`. Each has a module docstring
  with usage examples; keep it current.
- `grid_cells/` is library code and never has `if __name__ == "__main__"` blocks
  (`tests/test_package_rules.py` enforces this). It stays flat until it passes ~20 modules.
  - data: `data_generation.py` (trajectory simulator, shard writer), `dataset.py` (shard loading)
  - supervised network: `ensembles.py`, `model.py`, `training.py`
  - analysis: `scores.py`, `evaluate.py`, `figures.py`
  - RL agent: `vision.py`, `actor_critic.py`, `replay_buffer.py`, `rl_a3c.py` (frozen legacy),
    and the A2C learner modules
- `config.py` stays at the root and holds every default as dataclasses. New settings go into the
  matching dataclass; entry points expose only the flags people actually change.
- `scripts/` holds shell tooling (`setup_dmlab.sh`) and the DeepMind Lab build patch.
- `levels/` holds our DeepMind Lab levels, loaded through `levelDirectory`; `lab/` is the
  upstream submodule and is never edited except through `scripts/dmlab_patches/`.

## Data and results

- `data/` is gitignored. Datasets: `data/datasets/<id>/{shards/,meta.json}` with `data/latest`
  pointing at the current one (`data/shards/` is the legacy location of the same shards).
  Checkpoints: `data/checkpoints/<run>/`.
- `results/<run>/` is committed: small figures and videos only. `evaluate.default_out_dir()`
  pairs `data/checkpoints/<run>/` with `results/<run>/`.
- Never delete anything under `data/checkpoints/` without asking.

## Tests

- `pytest` from the repo root. Markers: `dmlab` (auto-skipped without `deepmind_lab`), `slow`,
  `gpu`. Fast suite: `pytest -m "not dmlab and not slow"`.
- `tests/golden/` holds outputs recorded before the package restructure; `test_golden.py` must
  stay bit-identical unless a change is meant to alter behaviour, in which case regenerate the
  golden file in the same commit and say why in the message.
- A test that guards a bug must be seen failing on the buggy code first. Known defects of the
  frozen A3C loop are pinned with `pytest.mark.xfail(strict=True)`.

## The paper

- `docs/paper/spec-*.md` summarise what the paper specifies, with PDF page citations;
  `docs/paper/figures.md` records what only the images show; `docs/paper/deviations.md` lists
  every departure and every gap we filled. Update `deviations.md` in the same commit as the
  code that departs. Bugs (code that did something other than intended) do not go there; they
  go in the commit message (see Commits).
- The PDFs, figure images and text extraction under `docs/paper/` are local, gitignored copies.

## Plans and status

- Plans live in `docs/plan/`; `docs/plan/STATUS.md` is the single place that says where work
  stands. Update it at the end of every phase.
- Long RL runs need written milestone gates before they start (see the plan).

## Commits

- `COMMIT_DRAFT.md` at the repo root (gitignored) is the draft message for the next commit. Each
  time code or docs change, add what changed and why. The first line is the subject; then a blank
  line and a bulleted body.
- Commit with `git commit -F COMMIT_DRAFT.md`, then empty the file.

## Local environment

- Interpreter: conda `grid` (`/home/soluslan/miniconda3/envs/grid/bin/python`, Python 3.12) with
  everything in `requirements.txt`. The system `python3` (3.14) has neither torch nor DMLab.
- Machine: WSL2, Ubuntu 26.04 (GCC 15), 16 threads, 15 GB RAM, **no GPU**. Every entry point
  falls back to CPU when CUDA is unavailable (A2C's `device="cuda"` default included). Memory is
  the tight resource for RL: the A2C replay buffer alone is ~4.9 GB at 400k frames, plus one
  DeepMind Lab process per environment.
- DeepMind Lab: `bash scripts/setup_dmlab.sh` builds the `lab/` submodule and installs it into
  `grid`. It needs these apt packages (installed with sudo, by hand): `build-essential libffi-dev
  gettext freeglut3-dev libsdl2-dev zip libosmesa6-dev python3-dev python3-numpy python3-pil
  libglib2.0-dev`. Several of these are in `universe`, so run `sudo apt-get update` first on a
  fresh system. Bazelisk is fetched into `~/.local/bin` if missing. Both renderers work without
  a GPU (`hardware` = EGL on Mesa's llvmpipe, `software` = OSMesa); the `libEGL warning: failed
  to open /dev/dri/...` lines they print are harmless.
- Timings recorded elsewhere in the docs (e.g. "~30 min on a GTX 1650", A2C frames/s in
  `docs/plan/STATUS.md`) were measured on an earlier GPU machine; they are kept as records.
