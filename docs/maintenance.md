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
  code that departs.
- The PDFs, figure images and text extraction under `docs/paper/` are local, gitignored copies.

## Plans and status

- Plans live in `docs/plan/`; `docs/plan/STATUS.md` is the single place that says where work
  stands. Update it at the end of every phase.
- Long RL runs need written milestone gates before they start (see the plan).

## Local environment

- Interpreter: conda `env1` (`/home/junoLee/miniconda3/envs/env1/bin/python`), with CUDA
  PyTorch, `deepmind_lab` and `pytest`. The system `python3` has neither.
- Hardware this was tuned on: Ryzen 9 3900X (12 cores / 24 threads), 62 GB RAM, GTX 1650 4 GB,
  WSL2.
