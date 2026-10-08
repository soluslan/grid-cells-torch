---
paths:
  - "grid_cells/model.py"
  - "grid_cells/training.py"
  - "grid_cells/data_generation.py"
  - "grid_cells/dataset.py"
  - "grid_cells/ensembles.py"
  - "grid_cells/scores.py"
  - "grid_cells/evaluate.py"
  - "train.py"
  - "generate_data.py"
  - "analyze_supervised.py"
  - "model.py"
  - "dataset.py"
  - "ensembles.py"
  - "scores.py"
  - "evaluate.py"
---

# Supervised network files

- Read `docs/paper/spec-supervised.md` and the supervised sections of
  `docs/paper/deviations.md` before changing behaviour here.
- Confirm any "as in the paper" claim against `docs/paper/text/main.txt` (Methods, PDF p.6) or
  `supp.txt` (Table 1, PDF p.24; analysis definitions, PDF p.16–19).
- The original TF implementation in `../grid-cells/` is the reference for port fidelity;
  `deviations.md` lists every intentional difference from it.
