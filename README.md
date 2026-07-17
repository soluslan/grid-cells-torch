# grid-cells-torch

PyTorch reimplementation of [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)
(Banino et al. 2018, Nature, "Vector-based navigation using grid-like
representations in artificial agents"). The original repo is TF1 + Sonnet v1
and does not run on current package versions (`tf.contrib.*` was removed in
TF2, Sonnet v1's `snt.AbstractModule`/`snt.RNNCore` were redesigned in v2, the
TF1 queue-based data API was removed in TF2, `tf.flags`/`tf.app.run()` were
removed, and there's even a bare Python-2 `xrange`) -- so this is a from-scratch
port, not a patch.

Uses our own dataset at `square_room_100steps_2.2m_1000000/` (generated in
`make_dataset.ipynb` using the paper's own Supplementary Information
Table 1 motion-model parameters) instead of the original's TFRecord release.

## Two things ported deliberately differently from the original

1. **Weight decay is actually applied here.** The original's `model.py`
   registered Sonnet `regularizers` on the bottleneck/output-head weights, but
   `train.py`'s optimized loss only ever summed the two cross-entropy terms --
   nothing anywhere calls `tf.losses.get_regularization_loss()`. As published,
   `model_weight_decay=1e-5` never affected training. Here it's wired into
   real optimizer param groups (`model.decay_parameters()` /
   `model.no_decay_parameters()` in `model.py`, used in `train.py`). Set
   `ModelConfig.weight_decay = 0.0` to reproduce the original's literal (no
   regularization) behavior.
2. **`nh_embed` is dropped.** It was accepted and stored in the original's
   `GridCellsRNNCell.__init__` but never used to build anything in `_build()`
   -- dead config surface, not carried over.

## Pipeline

```
square_room_100steps_2.2m_1000000/*.csv
        │  scripts/convert_csv_to_shards.py
        ▼
data/shards/*.npz   (init_pos, init_hd, ego_vel, target_pos, target_hd)
        │  dataset.py (GridCellsDataset / DataLoader)
        ▼
train.py  (ensembles.py targets + model.py GridCellsRNN)
        ▼
results/checkpoint_epoch*.pt
        │  scripts/evaluate.py (scores.py GridScorer)
        ▼
eval/*_ratemaps_epoch*.pdf, eval/*_scores_epoch*.npz
```

### Data field mapping (our CSV -> model input/targets)

Our CSV has corner-origin `pos_x,pos_y` (range ~[0, 2.2]), allocentric
`vel_x,vel_y`, and `head_direction_x,head_direction_y` unit vectors. The
original TFRecord dataset used **center-origin** coordinates
(`coord_range=(-1.1,1.1)`) and an **egocentric** 3-component velocity input
(`ego_vel`, shape `[100,3]`) whose exact composition was never published (only
the TFRecord *reader* is open-sourced, not the trajectory-generation code that
produced `ego_vel`). `scripts/convert_csv_to_shards.py` derives:

- `pos_centered = pos_{x,y} - 1.1`
- `theta = arctan2(head_direction_y, head_direction_x)`
- `init_pos/init_hd` = `pos_centered`/`theta` at step 0; `target_pos/target_hd` = all 100 steps
- `speed = sqrt(vel_x**2 + vel_y**2)`
- `dtheta`: computed from the **angle between consecutive `head_direction` unit
  vectors** (`atan2(cross, dot)`), *not* from `rot_vel * dt` -- our CSV's
  `rot_vel` is a linearly-interpolated instantaneous sample of a `dt=0.02s`
  fine-step signal, and was measured to disagree with the true coarse (0.15s)
  step-to-step heading change by ~12% on a sample trajectory.
- `ego_vel = stack([speed, sin(dtheta), cos(dtheta)])` -- **this 3-component
  composition is a well-reasoned reconstruction, not a confirmed fact.**
  Treat it as tunable -- see "Known caveats" below.

## Running it

```bash
# (run from inside this repo's root)

# 1. smoke test first: converts only 2 shards + a few training iterations
python scripts/smoke_test.py

# 2. once that passes, convert the full dataset (~2-5 min)
python scripts/convert_csv_to_shards.py --csv_dir square_room_100steps_2.2m_1000000 --out_dir data/shards

# 3. train (1000 epochs x 1000 steps/epoch = 1,000,000 gradient steps total,
#    minibatch=10 -- see "Training adjustments made" below. ~1.8h on a GTX 1650)
python train.py

# 4. evaluate a checkpoint for grid-cell-like periodicity
python scripts/evaluate.py --checkpoint results/checkpoint_epoch998.pt
```

## Training adjustments made

The original's `lr=1e-5` + gradient clip `1e-5` (`training_clipping`,
applied via `tf.clip_by_value` -- ported here as
`torch.nn.utils.clip_grad_value_`, an element-wise abs-clamp, **not**
`clip_grad_norm_`) is an aggressive combination. `scripts/smoke_test.py`
step 7/8 measured real gradient magnitudes at init (~0.07-0.15) and found
them ~7000x larger than the clip value -- meaning every step would have
clamped to a constant `+-1e-5`, making training too slow to be useful in any
finite run.

**Decision: loosened `grad_clip_value` to `1.0`** (`config.py`,
`TrainConfig.grad_clip_value`) so the clip only guards against extreme
outliers instead of saturating on every step. Set it back to `1e-5` to
reproduce the original's literal (very slow) behavior. `learning_rate`
itself was left at the paper's literal `1e-5` for this run -- see "Known
caveats" below for why it may still need raising.

## Evaluating a trained model

`scripts/evaluate.py` loads a checkpoint, runs inference only (no
training/dropout) over a batch of trajectories, and uses `scores.py`'s
`GridScorer` (ported from the original's `scores.py`: ratemaps via
`scipy.stats.binned_statistic_2d`, spatial autocorrelograms via
`scipy.signal.convolve2d`, gridness scores via rotation-based Pearson
correlation) to check whether grid-cell-like hexagonal periodicity emerged,
scoring the bottleneck (256 units) and raw LSTM output (128 units) layers
separately.

```bash
python scripts/evaluate.py --checkpoint results/checkpoint_epoch998.pt
```

Produces, per layer, in `eval/`:
- `{bottleneck,lstm}_ratemaps_epoch{N}.pdf` -- one ratemap + autocorrelogram
  per unit, sorted by gridness score
- `{bottleneck,lstm}_scores_epoch{N}.npz` -- raw `scores_60` array

A checkpoint (`results/checkpoint_epoch*.pt`) is a zip archive containing a
pickled dict with `model` (state_dict), `optimizer` (state_dict, 2 param
groups), and `epoch` (int) -- saved every `save_every_n_epochs` (2) epochs
by `train.py`.

## Results so far

First full training run: 1,000,000 gradient steps (1000 epochs x 1000
steps/epoch, minibatch=10), ~1.8h on a GTX 1650. Loss converged from ~8.04
(near the `log(256)+log(12)≈8.03` random-init sanity bound) to ~2.18.
Evaluated at `checkpoint_epoch998.pt`:

| layer      | units | gridness > 0.3 | max gridness |
|------------|-------|-----------------|--------------|
| bottleneck | 256   | 8 (3%)          | 1.07         |
| raw LSTM   | 128   | 0 (0%)          | --           |

This qualitatively reproduces the paper's central claim -- grid-like
periodicity is bottleneck-specific, not a property of the raw recurrent
state -- but at a much lower rate than the paper's reported ~129/512
(~25%) grid-like units. See "Known caveats" below for the leading
hypotheses on why, in priority order.

## Known caveats / tuning candidates for the next run

Ranked roughly by how likely each is to close the gap with the paper's
reported grid-cell emergence rate:

1. **`parameter_updates` count mismatch.** The paper's own Supplementary
   Table 1 states 300,000 total gradient steps, but the public GitHub
   `train.py` flag defaults imply 1,000,000 (`epochs=1000 x
   steps_per_epoch=1000`, `config.py`'s current defaults). We trained with
   the code's 1,000,000, not the paper's stated 300,000 -- unclear which
   figure actually produced the paper's published results.
2. **`ego_vel` composition is a reconstruction, not a confirmed fact.**
   DeepMind's exact 3-component egocentric velocity encoding was never
   published (only the TFRecord *reader* is open-sourced). We use
   `[speed, sin(dtheta), cos(dtheta)]` in
   `scripts/convert_csv_to_shards.py` -- see "Data field mapping" above.
   Considered the single most likely source of the gap if #1 doesn't close
   it.
3. **Learning rate still at the paper's literal `1e-5`**
   (`TrainConfig.learning_rate`), independent of the gradient-clip value
   already loosened above. RMSprop's per-parameter normalization made the
   clip fix matter less for training speed than expected; lr itself may
   need raising (e.g. 1e-4 or 1e-3) -- watch for instability (NaN/exploding
   loss) via `scripts/smoke_test.py` first.
4. **Grid-cell significance test methodology differs from the paper's.**
   The paper calls a unit "grid-like" via a shuffle-based null distribution
   (95th percentile of gridness scores from spatially-shuffled ratemaps,
   Supplementary Methods 3d), not a fixed threshold. `evaluate.py` uses a
   fixed `gridness > 0.3` cutoff (a common literature rule of thumb) -- not
   a true apples-to-apples comparison to the paper's 129/512 figure.

## Data / large files

Not committed to git (see `.gitignore`), all regeneratable:

| path             | size                  | regenerate via                      |
|-------------------|-----------------------|--------------------------------------|
| `data/shards/`    | ~2.3GB, 100 files     | `scripts/convert_csv_to_shards.py`   |
| `results/`        | ~1.4GB, 500 checkpoints | `train.py`                         |

`eval/` (PDFs + score `.npz` files, a few MB) is small and **not**
ignored -- kept as evidence of what a given checkpoint actually produced.
