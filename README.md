# grid-cells-torch

PyTorch reimplementation of [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)
(Banino et al. 2018, Nature, "Vector-based navigation using grid-like
representations in artificial agents"). The original repo is TF1 + Sonnet v1
and does not run on current package versions (`tf.contrib.*` was removed in
TF2, Sonnet v1's `snt.AbstractModule`/`snt.RNNCore` were redesigned in v2, the
TF1 queue-based data API was removed in TF2, `tf.flags`/`tf.app.run()` were
removed, and there's even a bare Python-2 `xrange`) -- so this is a from-scratch
port, not a patch.

Uses our own dataset at `data/square_room_100steps_2.2m_1000000/` (generated in
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

## A detail the paper's equations omit but the code (both original and here) implements

The paper's Methods describes the LSTM's initial cell/hidden state as a
plain linear transform of the t=0 place/head-direction encoding, e.g.
`l_0 = W^cp c_0 + W^cd h_0` (Extended Data Fig. 1) -- no bias term shown.
Both the original (`snt.Linear(nh_lstm, name="state_init")` /
`"cell_init"`, and Sonnet's `Linear` defaults to `use_bias=True`) and this
port (`state_init`/`cell_init` are plain `nn.Linear(n_init_in, nh_lstm)`,
which also defaults to `bias=True`) actually include a learned bias in this
computation. This isn't a deviation this port introduces -- both
implementations agree with each other, just not with the paper's simplified
written equation -- but it's easy to miss if you're cross-referencing the
paper's math against the code.

## Pipeline

```
data/square_room_100steps_2.2m_1000000/*.csv
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
- `ego_vel = stack([speed, sin(dtheta), cos(dtheta)])`, where `dtheta` is the
  net heading change between consecutive stored `head_direction` unit
  vectors (`arctan2(cross, dot)` of consecutive steps) -- **not** the CSV's
  raw `rot_vel` column, even though `[v_t, sin(phi_dot_t), cos(phi_dot_t)]`
  built from `rot_vel` looks more literally paper-like on paper. `rot_vel`
  was tried (in three variants) and consistently made training stall --
  see "Why `rot_vel` doesn't work for `ego_vel`" below for the full
  root-cause investigation and "Known caveats" #3 for the summary. Still a
  reconstruction either way, not a confirmed fact (DeepMind's exact
  `ego_vel` composition was never published, only the TFRecord *reader*).

### Two different Δt's in `make_dataset.ipynb`

The Supplementary Information's Table 1 (`41586_2018_102_MOESM1_ESM.pdf`)
lists `T=15` ("Duration of simulated trajectories"), `Δt=0.02`
("**Simulation**-step time increment"), and `trajectory length=100`
("Number of time steps in the trajectories used for the supervised learning
task") as three *separate* rows -- Δt=0.02s is explicitly the fine physics-
simulation step, not the spacing between the 100 steps fed to the network.
`make_dataset.ipynb` matches this: it simulates the rat's continuous motion
at `dt=0.02s` (750 fine steps, needed for the near-wall behaviour to work at
all -- `d=0.03m` is a thin perimeter that a coarser step could skip over
without ever registering "near wall"), then resamples down to the 100
target times spaced `T/100=0.15s` apart (`t_target =
np.linspace(0, T, 100, endpoint=False)`) that are actually stored/trained
on. `rot_vel` is generated at the fine 0.02s resolution and then
interpolated onto those 100 coarse times.

### Why `rot_vel` doesn't work for `ego_vel` -- `dtheta` is used instead

**The bug.** `rot_vel` is a *rate* (rad/s), but `sin`/`cos` only meaningfully
distinguish values within one period (`2*pi ≈ 6.28`). `sigma_phi` (the
rotational-velocity std, `330 deg/s ≈ 5.76 rad/s`, Supplementary Table 1) is
already almost as large as that whole period. Under a zero-mean Gaussian
with that std, **58.5% of *all* rotation samples -- ordinary ones, nothing
to do with walls -- have `|rot_vel| > pi`** and wrap the unit circle at
least partway before ever reaching `sin`/`cos`; `27.5%` wrap a full turn or
more. On top of that, the motion model's wall-avoidance behaviour
(`_wall_redirect` in `make_dataset.ipynb`) can turn the heading by up to
~90 degrees in a single *fine* `dt=0.02s` step (see "Two different Δt's"
above), and dividing that near-instant turn by `0.02s` produces true
outliers -- `rot_vel` was measured reaching **+-92 rad/s** at those moments,
versus roughly +-20 rad/s away from walls. A from-scratch training run with
raw `rot_vel` visibly stalled (loss plateaued around 6.96 instead of
continuing down to ~3.5 like the healthy prior run).

**Two attempted fixes, both failed.** Clipping `rot_vel` to `+-3*sigma_phi`
(~17.3 rad/s) before `sin`/`cos` only touches the ~0.4% wall-bounce tail and
leaves the other 58% of ordinary-but-large samples unaddressed -- loss still
stalled (~7.07). Rescaling by `COARSE_DT` (~0.15s, this dataset's actual
per-training-step spacing) into an angle-like quantity, then clipping to
`+-pi`, also stalled (~7.06) -- essentially identical to the unscaled case.

**Root cause, and the actual fix: use `dtheta` instead.** Both `rot_vel`
variants above share the same underlying problem regardless of scale/
clipping: `rot_vel` is an *instantaneous* rate sampled at a single fine
(0.02s) instant within each 0.15s coarse step (which spans ~7.5 fine
sub-steps), so it only reflects whatever was happening at that one moment
and misses the rest of the step. `dtheta` -- the angle between consecutive
stored `head_direction` unit vectors, i.e. the *net* heading change actually
realized over the full 0.15s step -- measures the right quantity instead,
and is what `scripts/convert_csv_to_shards.py` uses (confirmed not a
dataset-reseed confound: `dtheta` reaches ~4.9 on both the original and a
reseeded dataset, matching the healthy prior run; see that file's module
docstring for the full investigation).

## Running it

```bash
# (run from inside this repo's root)

# 1. smoke test first: converts only 2 shards + a few training iterations
python scripts/smoke_test.py

# 2. once that passes, convert the full dataset (~2-5 min)
python scripts/convert_csv_to_shards.py --csv_dir data/square_room_100steps_2.2m_1000000 --out_dir data/shards

# 3. train (300 epochs x 1000 steps/epoch = 300,000 gradient steps total,
#    minibatch=10 -- see "Training adjustments made" below. ~34min on a GTX 1650)
python train.py

# 4. evaluate a checkpoint for grid-cell-like periodicity
python scripts/evaluate.py --checkpoint results/checkpoint_epoch299.pt
```

## Training adjustments made

The original's `lr=1e-5` + gradient clip `1e-5` (`training_clipping`,
applied via `tf.clip_by_value` -- ported here as
`torch.nn.utils.clip_grad_value_`, an element-wise abs-clamp, **not**
`clip_grad_norm_`) is an aggressive combination. `scripts/smoke_test.py`
step 7/8 measured real gradient magnitudes at init (~0.07-0.15) and found
them ~7000x larger than the clip value -- meaning every step would clamp to
a constant `+-1e-5`, regardless of the true gradient's magnitude.

**This was previously "fixed" by loosening `grad_clip_value` to `1.0`**, on
the assumption that the literal value would just make training pointlessly
slow. That run completed (1e6 steps, `results/checkpoint_epoch998.pt`), but
inspecting it turned up a bigger problem than slowness: `lstm.weight_ih_l0`
had grown ~20x (max|w| 0.11 -> 2.11) and gradients at that checkpoint were
~150-200x *larger* than at init (max|grad| ~14-25 vs ~0.1) -- the LSTM's
recurrent weights drifted into an exploding-gradient regime over the 100-step
BPTT unroll, unconstrained by anything else (no `hidden_clip_value`/
`cell_clip_value`, which the original's `snt.LSTM` supports but neither the
original `model.py` nor this port ever enables).

That reframes the literal `1e-5` clip: under a constant-magnitude clip,
RMSprop's own per-parameter magnitude normalization becomes redundant (its
running average of squared gradients converges to `(1e-5)^2` regardless of
the true gradient), so the whole optimizer degenerates toward a fixed-step,
sign-of-gradient-only update (the Rprop lineage RMSprop was itself
generalized from) -- slow per step, but structurally incapable of the kind
of runaway weight growth seen above, since no single step can ever exceed
the clip. **Reverted `grad_clip_value` back to the paper's literal `1e-5`**
(`config.py`, `TrainConfig.grad_clip_value`) on the hypothesis that it may be
an implicit stabilizer against this exact failure mode rather than just an
oversight.

**Confirmed (2026-07-22): a from-scratch run at the literal `1e-5` clip,
`nh_bottleneck=512`, 300,000 steps (300 epochs x 1000 steps/epoch) completed
cleanly** -- loss decreased smoothly and monotonically (7.79 -> 3.57) with no
NaN/divergence, and evaluation found grid-like units at a much higher rate
than the earlier loosened-clip/256-bottleneck run (see "Results so far"
below). This is consistent with, though not yet definitive proof of, the
stabilizer hypothesis -- LSTM weight norms were not directly tracked during
this run, so "loss didn't explode" is suggestive but not the same as
confirming weights stayed bounded the way the earlier `clip=1.0` run's
post-hoc weight inspection showed they hadn't. `learning_rate` itself
remains at the paper's literal `1e-5`.

## Evaluating a trained model

`scripts/evaluate.py` loads a checkpoint, runs inference only (no
training/dropout) over a batch of trajectories, and uses `scores.py`'s
`GridScorer` (ported from the original's `scores.py`: ratemaps via
`scipy.stats.binned_statistic_2d`, spatial autocorrelograms via
`scipy.signal.convolve2d`, gridness scores via rotation-based Pearson
correlation) to check whether grid-cell-like hexagonal periodicity emerged,
scoring the bottleneck (`ModelConfig.nh_bottleneck`, 512 units by default)
and raw LSTM output (128 units) layers separately.

```bash
# default: per-unit shuffle threshold (the paper's actual methodology --
# see "Known caveats" #5 -- but slow: ~30-40min for 512+128 units at 100
# shuffles/unit)
python scripts/evaluate.py --checkpoint results/checkpoint_epoch299.pt

# fast alternative: one fixed GRIDNESS_THRESHOLD=0.37 cutoff for every unit
python scripts/evaluate.py --checkpoint results/checkpoint_epoch299.pt --no-shuffle_threshold
```

Produces, per layer, in `eval/`:
- `{bottleneck,lstm}_ratemaps_epoch{N}.pdf` -- one ratemap + autocorrelogram
  per unit, sorted by gridness score. Units that pass their threshold are
  highlighted with a red border and a red bold title (`score`, or
  `score/threshold` in the default per-unit shuffle mode) so they're
  visible at a glance in the 16-column grid.
- `{bottleneck,lstm}_scores_epoch{N}.npz` -- raw `scores_60` array (plus
  `shuffle_thresholds`, one per unit, in the default shuffle mode)

`n_trajectories=4000` is sampled with `shuffle=True`, so re-running
`evaluate.py` on the same checkpoint gives slightly different scores/counts
each time (different random trajectory sample) -- expect small run-to-run
noise, not a bug.

A checkpoint (`results/checkpoint_epoch*.pt`) is a zip archive containing a
pickled dict with `model` (state_dict), `optimizer` (state_dict, 2 param
groups), and `epoch` (int) -- saved every `save_every_n_epochs` (2) epochs
by `train.py`.

## Results so far

**Current run** (2026-07-23): 300,000 gradient steps (300 epochs x 1000
steps/epoch, minibatch=10, ~34min on a GTX 1650), using the current
`config.py` defaults -- `nh_bottleneck=512`, `grad_clip_value=1e-5` (the
paper's literal value, reverted from an earlier loosened `1.0` -- see
"Training adjustments made"), `dtheta`-based `ego_vel`. Loss converged from
~7.79 to ~3.60. Evaluated at `checkpoint_epoch299.pt` with one
`evaluate.py --shuffle_threshold` run, which computes both the fixed
`GRIDNESS_THRESHOLD=0.37` cutoff and the paper's actual per-unit shuffle
threshold (100 shuffles/unit -- see "Known caveats" #5 and
`notebooks/gridness_shuffle_explainer.ipynb` for how the latter works):

| layer      | units | mean gridness | max gridness | gridness > fixed 0.37 | gridness > own shuffle threshold | mean per-unit threshold |
|------------|-------|----------------|--------------|------------------------|-----------------------------------|--------------------------|
| bottleneck | 512   | 0.144          | 1.481        | 92 (18.0%)             | **39 (7.6%)**                     | 0.557                    |
| raw LSTM   | 128   | -0.011         | 1.169        | 11 (8.6%)              | **5 (3.9%)**                      | 0.562                    |

(counts/means fluctuate slightly run-to-run -- `n_trajectories=4000` is a
different random sample each time, see note above; figures here are from
one representative run. Separately re-confirmed these numbers aren't a
seed artifact: the dataset was regenerated from scratch with the original
per-file RNG seeding (`np.random.seed(file_idx)` in `make_dataset.ipynb`,
after a personal detour testing an offset seed), and both loss (~3.60 vs
the offset-seed run's ~3.57) and grid-like rates land in the same range
either way.)

**Provisional note on the shuffle-threshold column.** A follow-up check
found that real trained units' ratemaps get inconsistently
over-segmented by the current (unsmoothed) watershed step -- field counts
across the 512 bottleneck units ranged from 1 to **86** (out of 1024 bins),
unlike the clean, few-field synthetic example in the explainer notebook.
This likely inflates or destabilizes individual per-unit thresholds for
the noisier units. A fix (Gaussian-smoothing the ratemap before field
segmentation only, not before scoring) has been identified but is
deliberately not yet applied -- **treat the 39/512 and 5/128 numbers above
as provisional**, likely to shift once that's addressed.

The per-unit shuffle procedure is noticeably *stricter* here either way:
its average per-unit threshold (~0.56) is well above the paper's own fixed
0.37, meaning our network's ratemaps -- for whatever reason (the
over-segmentation issue above, field count/size/shape, resolution, or a
difference between our shuffle implementation and DeepMind's undisclosed
original) -- can already fake a higher gridness score than 0.37 just by
rearranging their own fields at random. So the honest, paper-methodology
grid-like rate for this checkpoint is closer to **~7.6% (bottleneck)**, not
~18.0% -- further from the paper's 25.2% than the fixed-threshold number
suggested, not closer.

This is a sizeable jump from the previous run below, though still short of
the paper's reported 129/512 (25.2%). Notably, the raw LSTM layer now also
shows some grid-like units, unlike the previous run's 0/128 -- worth
watching whether that persists across re-evaluations, since the paper
frames grid-like periodicity as bottleneck-specific.

**Previous run** (256-unit bottleneck, loosened `grad_clip_value=1.0`,
threshold `0.3`, 1,000,000 steps, `checkpoint_epoch998.pt`, loss ~8.04 ->
~2.18) -- kept here for comparison, not reproducible from current
`config.py` defaults:

| layer      | units | gridness > 0.3 | max gridness |
|------------|-------|-----------------|--------------|
| bottleneck | 256   | 8 (3%)          | 1.07         |
| raw LSTM   | 128   | 0 (0%)          | --           |

Both runs qualitatively reproduce the paper's central claim -- grid-like
periodicity concentrates in the bottleneck far more than in the raw
recurrent state. See "Known caveats" below for remaining gaps to the
paper's reported rate.

## Known caveats / tuning candidates for the next run

Ranked roughly by how likely each is to close the gap with the paper's
reported grid-cell emergence rate:

1. **`parameter_updates` count mismatch.** The paper's own Supplementary
   Table 1 states 300,000 total gradient steps, but the public GitHub
   `train.py` flag defaults imply 1,000,000 (`epochs=1000 x
   steps_per_epoch=1000`, `config.py`'s current defaults). We trained with
   the code's 1,000,000, not the paper's stated 300,000 -- unclear which
   figure actually produced the paper's published results.
2. **Bottleneck width -- now matched to the paper's headline number, and
   retrained.** The Nature main text's Fig. 1 network (the one reporting
   "129/512 (25.2%) grid-like units") uses a 512-unit linear bottleneck. The
   publicly released `google-deepmind/grid-cells` code instead defaulted to
   256 units -- the same width later reused for the RL agent's grid code in
   the paper's Fig. 2-4 (Extended Data Fig. 6a: "256 linear layer units",
   21.4% grid-like there) -- which is what the "Previous run" in "Results so
   far" used. `ModelConfig.nh_bottleneck` is now `512`, and the "Current
   run" above was retrained/evaluated at that width (set it back to `256`
   to match the released code / RL agent instead). Still short of the
   paper's 25.2%, so this alone doesn't fully close the gap.
3. **`ego_vel` composition -- three `rot_vel`-based reconstructions tried
   and rejected; `dtheta` kept.** `scripts/convert_csv_to_shards.py` uses
   `[speed, sin(dtheta), cos(dtheta)]`, where `dtheta` is the net heading
   change directly measured between consecutive stored `head_direction`
   vectors -- not the paper-literal-looking `[v_t, sin(phi_dot_t),
   cos(phi_dot_t)]` built from the raw CSV `rot_vel` rate, which was tried
   in three variants (raw, clipped to `+-3*sigma_phi`, rescaled by
   `COARSE_DT` and clipped to `+-pi`) and failed identically in all three
   (loss plateaued ~6.96-7.07 instead of `dtheta`'s ~3.5-4.9) -- see "Why
   `rot_vel` doesn't work for `ego_vel`" above for the root cause (it's a
   single-instant sample within a 0.15s window, not the window's net
   change, so no amount of clipping/rescaling fixes it). This was confirmed
   end to end across from-scratch runs with an otherwise-identical setup
   (512-unit bottleneck, literal `1e-5` grad clip), and a dedicated check
   ruled out the dataset reseed as a confound.
4. **Learning rate at the paper's literal `1e-5`, and `grad_clip_value`
   reverted to the paper's literal `1e-5` -- confirmed workable.** The
   "Current run" above trained cleanly at this setting (loss 7.79 -> 3.57,
   no divergence) and produced a much higher grid-like rate than the
   earlier loosened-clip run, consistent with (but not rigorous proof of)
   the clip-as-stabilizer hypothesis in "Training adjustments made" --
   LSTM weight norms still haven't been tracked *during* a run to confirm
   they stay bounded, only inferred from the absence of loss blowup.
5. **Grid-cell significance test now uses the paper's actual per-unit
   shuffle procedure -- and it lowers the grid-like rate, not raises it.**
   `scores.py`'s `field_labels`/`shuffle_fields`/
   `GridScorer.shuffled_gridness_threshold` implement the paper's null
   distribution (Supplementary Methods 3d): watershed-segment each unit's
   ratemap into fields, relocate each field's peak to a random bin, refill
   the rest with the unit's own background level, repeat 100x, take the
   95th percentile as that unit's own threshold. `evaluate.py
   --shuffle_threshold` (the default) uses this instead of one fixed cutoff
   for every unit; `--no-shuffle_threshold` restores the old fast fixed-0.37
   behavior. See `notebooks/gridness_shuffle_explainer.ipynb` for a worked,
   visual walkthrough of the whole procedure on a synthetic example.
   Caveat: DeepMind never released this analysis code, only the paper's
   prose description -- exact implementation choices this repo had to make
   on its own (field-boundary convention, how overlapping relocated fields
   combine, what fills vacated background, and the watershed algorithm
   itself: a dependency-free steepest-ascent walk rather than
   `scipy.ndimage.watershed_ift`/skimage) are reasonable but unverified
   against any original source. Applying it to `checkpoint_epoch299.pt`
   gives a *stricter* result than the fixed 0.37 (see "Results so far"):
   bottleneck 39/512 (7.6%) vs. 92/512 (18.0%) with the fixed cutoff -- the
   per-unit thresholds average ~0.56, well above 0.37, meaning this
   network's ratemaps can already fake a gridness score above 0.37 just by
   rearranging their own fields at random. This widens, not narrows, the
   gap to the paper's 25.2%. **Known unresolved issue (deliberately on
   hold):** real units' field counts range from 1 to 86 (of 1024 ratemap
   bins) under the current unsmoothed watershed step -- much wider than the
   clean synthetic example in `notebooks/gridness_shuffle_explainer.ipynb`
   -- which likely makes per-unit thresholds for the noisier units
   unreliable. The fix under consideration is smoothing the ratemap before
   field segmentation only (not before scoring); not yet implemented, so
   treat current shuffle-threshold numbers as provisional.

## Data / large files

Not committed to git (see `.gitignore`), all regeneratable:

| path                                        | size                    | regenerate via                     |
|----------------------------------------------|-------------------------|-------------------------------------|
| `data/square_room_100steps_2.2m_1000000/`   | ~17GB, 100 CSV files    | `make_dataset.ipynb`               |
| `data/shards/`                              | ~2.3GB, 100 files       | `scripts/convert_csv_to_shards.py` |
| `results/`                                  | ~600MB, 151 checkpoints (current `epochs=300`; scales with `TrainConfig.epochs`) | `train.py` |

`eval/` (PDFs + score `.npz` files, a few MB) is small and **not**
ignored -- kept as evidence of what a given checkpoint actually produced.

## Roadmap: beyond the supervised path-integration network

This repo currently reimplements only Banino et al. 2018's **supervised**
grid-cell network (Fig. 1 / Methods / Supplementary Methods 3a-b). The
paper's other half -- a vision-based RL agent that reuses this network's
representations to navigate DeepMind Lab mazes (Fig. 2-4) -- was never
open-sourced by DeepMind ("the codebase for the deep RL agents makes use
of proprietary components... unable to publicly release", per the paper).
Planned next phases, in the order they'd naturally build on each other:

1. **Vision module.** The RL agent doesn't get privileged ground-truth
   position/head-direction at every step like the supervised network does
   -- it only gets a first-person visual frame, processed through a CNN
   into a feature vector (paper's Methods / Extended Data Fig. 2). Needs:
   confirming the exact CNN architecture the paper specifies (layer count/
   sizes), and how its output feeds into the rest of the agent.
2. **Actor-critic policy/value network.** An A3C-style network (Fig. 2a)
   that takes the vision features (+ this repo's pretrained grid-cell LSTM
   representations, either frozen or fine-tuned) and outputs an action
   distribution plus a value estimate, trained via reinforcement learning
   against a navigation-to-goal reward. Needs: confirming action space
   (discrete movement/rotation commands), reward shaping, and whether the
   grid-cell network is frozen or continues training during RL.
3. **Environment.** The paper trains and evaluates in DeepMind Lab mazes,
   which are not freely reusable/reproducible here in the same form. Needs
   a decision on a substitute (a custom simple maze environment, or an
   open equivalent) that preserves the task structure (visual navigation
   to a goal in a partially observable maze) closely enough to be a fair
   comparison.
4. **Validation.** The paper's headline RL-agent finding is that grid-like
   periodicity *re-emerges* in the agent's own recurrent units under this
   setup (Fig. 2d, Extended Data Fig. 6a, 21.4% grid-like at 256 units) --
   this repo's existing `scores.py`/`evaluate.py` gridness pipeline should
   carry over directly for checking that once an RL agent exists.

None of this is implemented yet -- this section exists to record the
intended scope before work starts, so later commits build toward it
incrementally rather than the roadmap being reconstructed after the fact.
