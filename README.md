# grid-cells-torch

PyTorch reimplementation of [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)
(Banino et al. 2018, Nature, "Vector-based navigation using grid-like
representations in artificial agents") — the paper's **supervised**
path-integration network (Fig. 1 / Methods / Supplementary Methods 3a–b).

The original is TF1 + Sonnet v1 and does not run on current versions
(`tf.contrib.*`, `snt.AbstractModule`, the TF1 queue data API and
`tf.flags` are all gone; there is even a bare Python-2 `xrange`), so this is
a from-scratch port, not a patch. It also uses our own dataset, generated in
`notebooks/00_make_dataset.ipynb` from the paper's Supplementary Table 1
motion-model parameters, rather than the original's unpublished TFRecord
release.

## Following the paper, not the released code

Where the paper and the released code disagree, this port follows the
**paper**. In every case the released code contains the machinery for the
paper's version but never switches it on.

| | Paper | Released code | Here |
|---|---|---|---|
| Gradient clipping scope | output heads (`g→y`, `g→z`) | all parameters | output heads |
| Weight decay scope | output-head **weights** | + bottleneck weight | output-head weights |
| Gridness formula | `min(c60,c120) − max(c30,c90,c150)` | mean-based variant | paper's |
| Gridness annulus | 8→20 bins, step 2 | 0.2→0.4‥1.0 × nbins | paper's |
| Ratemap bins | 32×32 (20×20 for border score) | 20×20 | 32×32 / 20×20 |
| Bottleneck width | 512 (Fig. 1) / 256 (Ext. Data Fig. 3d, RL agent) | 256 | 256 |
| Grid-like criterion | one 0.37 cutoff for all units | — | 0.37 |

Both scope items come from one Methods sentence pattern — *"parameters
projecting from the dropout layer, g⃗ₜ, to the place and head-direction cell
predictions y⃗ₜ and z⃗ₜ"* for clipping, and the same phrase with **weights**
for weight decay. Note "parameters" (weights + biases) vs "weights" only.
`TrainConfig.grad_clip_scope` / `ModelConfig.weight_decay_scope` accept
`"bottleneck_and_heads"` (the original's unused `clip_bottleneck_gradient`)
and, for clipping, `"all"` (its actual default).

**Caveat on clipping.** The paper justifies clipping by exploding gradients
in *recurrent* networks, yet scopes it to the feedforward heads only, leaving
the LSTM unclipped. That tension is unresolvable from the paper alone, which
is why all three scopes remain selectable.

### Other deliberate differences

1. **Weight decay is actually applied.** The original registered Sonnet
   `regularizers` but never summed them into the optimized loss, so
   `model_weight_decay=1e-5` never affected training. Set
   `ModelConfig.weight_decay = 0.0` for the original's literal behavior.
2. **`nh_embed` is dropped** — stored but never used to build anything.
3. **A learned bias in the LSTM init.** The paper's Extended Data Fig. 1
   writes `l₀ = W^cp c₀ + W^cd h₀` with no bias, but both the original
   (`snt.Linear` defaults to `use_bias=True`) and this port include one. The
   two implementations agree with each other, just not with the equation.

### Two numbers worth knowing

**The 0.37 cutoff.** A unit is grid-like iff `gridness > 0.37`; one cutoff for
every unit, which is what the paper actually does. It derived that number from
a per-unit field shuffle and then collapsed it (Methods): *"The means, over
units, of the thresholds obtained were >0.37 […] Units exceeding these
thresholds were considered to be grid-like."* So the shuffle is how 0.37 was
derived, not how units were judged. Re-deriving our own was dropped — it
cannot improve the model, only relabel it.

**The annulus inner radius is ours, not the paper's.** `8→20` is the outer
radius; the inner one the paper gives only as "the central peak excluded",
with no value. Excluding it is not optional — the central peak is rotationally
symmetric, so it correlates with itself equally at every rotation and washes
out the contrast gridness measures. `DEFAULT_INNER_RADIUS_BINS = 4.0` is what
the released code's fractional `0.2` came to at its own `nbins=20`. Far from
negligible — see Results, "Everything about the representation," for the full
sensitivity sweep at 256 units (a ~5.7× swing) and why it isn't just adopted
as a fix.

## Pipeline

```
data/square_room_100steps_2.2m_1000000/*.csv      00_make_dataset.ipynb
        │  dataset.py: convert_all()               01_prepare_data.ipynb
        ▼
data/shards/*.npz   (init_pos, init_hd, ego_vel, target_pos, target_hd)
        │  train.py (ensembles.py targets, model.py GridCellsRNN)
        ▼                                          02_train.ipynb
data/checkpoints/<run>/checkpoint_epoch*.pt
        │  evaluate.py (scores.py measures, figures.py draws)
        ▼                                          03_path_integration.ipynb
results/<run>/path_integration_epoch*.png          04_bottleneck_units.ipynb
results/<run>/{bottleneck,lstm}_ratemaps_epoch*.pdf
results/<run>/cell_types_epoch*.png                05_cell_types.ipynb
```

The three analysis notebooks split by question, not by layer: 03 asks whether
the network path-integrates at all, 04 whether grid-like units emerged, 05 what
else is in that layer and whether it holds still across training.

### Where things live

**`.py` is library, notebooks are execution.** No module has a CLI or a
`__main__`; the notebooks import them and implement nothing themselves. Run the
notebooks in numeric order.

**`data/` is large and regenerable, `results/` is small and kept.** That single
split is what `.gitignore` follows, so nothing large can leak into the
repository by accident and nothing worth keeping can be lost to a wildcard.

| | holds | tracked? |
|---|---|---|
| `*.py` (repo root) | the whole pipeline, as importable modules | yes |
| `notebooks/` | every workflow: make data → prepare → train → analyse | yes |
| `data/` | **everything large**: datasets, shards, weights | **no** |
| `results/<run>/` | figures, one folder per training run | yes |

`<run>` is whatever `cfg.train.results_dir` was named at training time.
**A fresh clone has `results/` but no `data/`** — the figures are committed, the
things that produce them are not, and the notebooks rebuild those:

```
data/                                    (gitignored, ~19GB)
    square_room_100steps_2.2m_1000000/   ~17GB, 100 CSVs   00_make_dataset.ipynb
    shards/                              ~2.3GB, 100 .npz  01_prepare_data.ipynb
    checkpoints/<run>/                   ~45MB per run     02_train.ipynb

results/                                 (tracked, ~12MB)
    baseline/                            the 256-unit run these Results describe
    nodropout/                           its dropout_rates=(0.0,) control
    superseded_512unit/                  the older 512-unit run, kept for reference
```

`data/checkpoints/` and `results/` pair up by run name, except for the six-run
seed sweep: those trained into `data/checkpoints/seed{1,2}_dropout{0.5,0.0}/`
and produced only the summary numbers in Results, no figures of their own.

New large artefacts belong under `data/` too, with their own line in
`.gitignore`. The subfolders are ignored individually rather than ignoring
`data/` wholesale, so that the ignore list doubles as the inventory of what
should be there.

To compare configurations, set `cfg.train.results_dir =
"data/checkpoints/<name>"` in `02_train.ipynb`; `evaluate.default_out_dir()`
pairs it with `results/<name>/` automatically, so naming the run once keeps a
sweep's outputs apart.

Training is ~28 minutes for the paper's 300,000 steps and there is **no
resume** — an interrupted run restarts from epoch 0. Checkpoints land every
`save_every_n_epochs` (20 — ~16 per run rather than 151, while still hitting
epoch 200 and 299, the two points the paper's inter-trial stability analysis
compares). Only figures are saved from evaluation: nothing reads intermediate
arrays and re-running is a couple of minutes. `n_trajectories=4000` is sampled
with `shuffle=True`, so re-running shifts the numbers slightly — run-to-run
noise, not a bug.

## Dataset reconstruction

Our CSV has corner-origin `pos_x,pos_y` (~[0, 2.2]), allocentric `vel_x,vel_y`
and `head_direction_{x,y}` unit vectors. The original TFRecord used
**center-origin** coordinates and a 3-component **egocentric** `ego_vel`
whose composition was never published (only the *reader* was open-sourced).
`dataset.py`'s `csv_shard_to_arrays` derives:

- `pos_centered = pos − 1.1`, `theta = arctan2(hd_y, hd_x)`
- `init_*` = step 0; `target_*` = all 100 steps
- `ego_vel = [speed, sin(dtheta), cos(dtheta)]`

### Two different Δt's

Supplementary Table 1 lists `T=15`, `Δt=0.02` ("**simulation**-step time
increment") and `trajectory length=100` as three separate rows — Δt is the
fine physics step, not the spacing of the 100 stored steps.
`make_dataset.ipynb` simulates at `dt=0.02s` (750 steps; needed for the
`d=0.03m` wall perimeter to register at all), then resamples to 100 steps
spaced `T/100 = 0.15s` apart.

### Why `dtheta`, not `rot_vel`

`ego_vel`'s rotation component looks like it should be the paper's literal
`[v_t, sin(φ̇_t), cos(φ̇_t)]` built from the CSV's `rot_vel`. It was tried in
three variants (raw; clipped to ±3σ_φ; rescaled by the coarse step and
clipped to ±π) and **all three stalled training** (loss plateaued ~6.96–7.07
vs `dtheta`'s ~3.5–4.9).

Root cause: `rot_vel` is an *instantaneous rate* sampled at one fine 0.02s
instant inside each 0.15s step (~7.5 sub-steps), so it misses most of the
step — no rescaling fixes *which quantity is measured*. It is also badly
scaled for `sin`/`cos`: with `σ_φ = 330°/s ≈ 5.76 rad/s`, 58.5% of ordinary
samples exceed π and wrap, and wall-avoidance turns produce ±92 rad/s
outliers. `dtheta` — the net heading change actually realized over the step,
from consecutive stored `head_direction` vectors — measures the right thing.
Confirmed not a dataset-reseed confound.

## Results

**Run of 2026-08-07**, at the default `nh_bottleneck=256`: 300,000 gradient
steps (300 epochs × 1000, minibatch=10, 26m36s on a GTX 1650), paper-scoped
clipping and weight decay, literal `1e-5` learning rate and clip, bottleneck
bias per the paper, `seed=0`. Loss **7.820 → 3.673**, monotone. Evaluated at
`checkpoint_epoch299.pt` over 4000 trajectories.

**Why 4000.** It is the released code's `training_evaluation_minibatch_size`
(`grid-cells/train.py`); the paper never states an evaluation sample size.
It buys ratemap resolution: 4000 × 100 steps = 400,000 (position, activation)
samples spread over 32×32 = 1024 bins, so ~390 samples per bin. At 1000
trajectories that falls to ~98 and at 500 to ~49, where single-bin noise starts
showing up in the autocorrelogram and so in gridness. Nothing about training
depends on it — it only sets how well-estimated each ratemap is.

**There is no held-out split.** Training and evaluation both call
`build_dataloader(..., shard_indices=None)`, so the 4000 evaluation
trajectories are drawn from the same 1,000,000-trajectory pool the network
trained on. This is inherited from the released code, which evaluates off the
same input queue it trains from, and the paper does not describe a split
either. It matters less here than it would elsewhere: training presents
3,000,000 trajectories total, so each of the million is seen about three times,
and the quantities of interest are representational (does a ratemap look
hexagonal) rather than predictive. The path-integration error is the exception —
that *is* a performance number, and it is measured in-sample. Holding out ten
shards would cost nothing but has not been done.

> **At 256 units the paper's comparison is 56/256 = 21.9%** (Extended Data
> Fig. 3d, circular arena), not the 25.2% headline, which is a 512-unit figure.
> A superseded 512-unit run of 2026-07-30 scored 37/512 (7.2%); its figures are
> kept under `results/superseded_512unit/` and its checkpoint no longer loads
> into the default model.

### Path integration — reproduces the paper

Decoded-position error against the true trajectory
(`results/baseline/path_integration_epoch299.png`):

| decoder | trained, t=15s | untrained | floor | effect size |
|---|---|---|---|---|
| argmax | 17.2cm | 112.0cm | 7.0cm | 2.53 |
| weighted mean | 25.2cm | 86.8cm | 6.9cm | 2.47 |
| **top3** | **14.7cm** | **88.8cm** | 6.9cm | **2.87** |
| *paper* | *16cm* | *91cm* | — | *2.83* |

**The paper does not say how it decoded position from the place cells**, so
all three readouts are measured. `top3` matches the paper on all three numbers
and is what `evaluate.py` selects; `argmax` is 23% off on the untrained
control. Any claim comparing our error to the paper's 16cm must name the
decoder.

*Floor* is the error left when the **true** position's place-cell code is
decoded — what a perfect path integrator would still score, since a 256-cell
code can only name cell centres. It is not in the paper, and it is what makes
the trained number interpretable: 14.7cm is 7.8cm of network error on top of
6.9cm of readout resolution, not 14.7cm of network error.

The error curve is more informative than either endpoint: 25.4cm at t=0 (the
LSTM settling out of its injected initial condition), down to **9.5cm at
t=1.80s** — 2.6cm above the floor — then drifting up over the remaining 13s.
Settling transient first, accumulating drift after.

### Everything about the representation — short, over, or absent

| measure | ours (256 units) | paper | paper's figure |
|---|---|---|---|
| gridness > 0.37, bottleneck | **11 (4.3%)** | 21.9% at 256 | Fig. 1d, ED 3d |
| gridness > 0.37, raw LSTM | 1 (0.8%) | ~0 | — |
| head direction > 0.47 | 65 (25.4%) — *see below, not measurable* | 10.2% | Fig. 1f |
| border score > 0.50 | **5 (2.0%)** | 8.7% | Suppl. eq. 8 |
| conjunctive (both cutoffs) | 1 (9.1% of grid units) | 14 (11% of 129) | Fig. 1g |
| grid scale | 80–120cm, mean 107 | 28–115cm, mean 66 | Suppl. 3d |
| scale clusters (BIC) | 4, ratios 1.10–1.17 | 3, ratios ~1.5 | Fig. 1e |
| discreteness of scale | p = 0.108 | p < 0.002 | Fig. 1e |
| stability, 2e5 vs 3e5 | all 0.887 / grid 0.880 / directional 0.815 | grid stable, directional **not** | ED 3b |

Read this table with three cautions.

**The scale rows are underpowered, not results.** They rest on 10 grid-like
units against the paper's 129. A mixture model over 10 points has more freedom
than the data constrains — the BIC curve has no clean minimum — and the shuffle
test has almost no power. `evaluate.report_scale_clustering` prints a warning
below 30 scales for exactly this reason.

**But the scale itself is solid, and it is the most concrete lead here.** The
peak-finding was checked directly against the autocorrelograms: every top unit
shows the textbook hexagonal signature of six off-centre peaks in three mirror
pairs (unit 74: 14.8/14.8, 15.6/15.6, 18.0/18.0 bins), with the next peak far
beyond at 30 bins, so the 20-bin search cap truncates nothing real. Mean 107cm
is genuine — and in a 2.2m arena that is a lattice with barely two periods
across it, the lowest spatial frequency that can express hexagonal structure at
all. Our grid-like units are only the coarsest ones; the paper's 28–70cm
population is simply absent. Whatever is short here is short specifically at
high spatial frequency, which is a narrower question than "not enough grid
cells".

**The annulus inner radius turns out to be a much bigger lever than the earlier
512-unit sweep suggested, and it's real, not an artefact.** Sweeping it at 256
units on this same checkpoint (SAC computed once per unit, reused across every
radius tried):

| inner radius | 1 | 2 | 3 | 4 (default) | 5 | 6 | 7 | 9 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| grid-like > 0.37 | 4.3% | 4.3% | 4.3% | 4.3% | 4.7% | 5.5% | 6.6% | 11.7% | 20.7% | 24.6% |

That is a ~5.7× swing, not the ~2× seen at 512 units, and inner=12 bins
(82.5cm) alone lands within a point of the paper's 21.9%.

Checked against controls before trusting this: ten draws of pure Gaussian
noise stay flat and low across the same sweep (mean 0.00–0.05, never near
0.37), while a synthetic hexagonal lattice at our own units' scale (16-bin
spacing, 110cm) actually *rises* with inner radius (1.30 → 1.99), matching
what the real units do. So this is not the same failure mode as the
`grid_scale` SAC edge-artifact found earlier — it reflects something genuine
about these units' periodicity, not noise inflating a thin ring's correlation.

**Why it moves so much here specifically:** `DEFAULT_INNER_RADIUS_BINS = 4.0`
was ported as an *absolute bin count* from the released code's `0.2` fraction
at its own `nbins=20`. Applied literally at our `nbins=32` (the paper's own
ratemap resolution), `0.2` would give 6.4 bins, not 4 — worth fixing for
consistency, though the sweep shows that alone only moves 4.3%→6.6%. The much
larger effect at 9+ bins makes sense given what these units actually are:
their own grid scale (mean 107cm) is far coarser than the paper's population
average (66cm), so a fixed inner radius tuned for the paper's mix of scales
increasingly under-excludes the trivial near-centre autocorrelation falloff
for a unit whose true wavelength is this much longer — the ring at radius 4
is measuring too close to centre relative to *this* unit's own periodicity.

**Why this is not "the gap is closed."** Inner=12 was found by sweeping until
the count approached the paper's number, using the very network the number is
meant to validate — adopting it as the new default would be tuning the ruler
to the answer. There is no independent principle here for what the "right"
inner radius is (the paper never states one), so the honest update is not "the
network makes 21.9% grid cells after all" but **"confidence in the 4.3–10.2%
shortfall being real is now lower than it was"** — the measurement has enough
free-parameter room, at this population's scale specifically, to swing between
"clearly short" and "matches the paper" depending on a choice nobody has
justified. `evaluate.report_gridness` still uses the inner=4 default; changing
it needs a principled reason, not a fit to this outcome.

**Checked, and ruled out: a hidden fine-scale population just below the
cutoff.** If the 0.37 threshold were hiding real but weaker fine-scale grid
cells, units just under it should trend toward *smaller* measured scale as
gridness rises toward the cutoff. It doesn't happen: near-miss units
(gridness 0.20–0.37, the 11 with a measurable scale) average 97cm — same
neighbourhood as the confirmed grid cells (107cm) — and every band down to
gridness < 0 sits at the same ~96–100cm. Whatever periodicity these low-scoring
units carry is the same coarse flavour as the units that clear the cutoff, not
a finer one waiting to be found. This is independent of the annulus question
above: it rules out the threshold as the explanation, while the annulus
finding questions the scoring parameter instead.

**The head-direction row should not be read as a disagreement.** That measure
turns out not to be recoverable from a linear layer at all: the non-negativity
shift it requires collapses a cosine tuning curve to r̄ = ½ regardless of how
weak the modulation is, which is above the 0.47 cutoff, and the choice of
convention alone moves the answer across 0–71%. Known gaps 3 has the algebra
and the measurements. The border and gridness rows carry no such problem —
both rest on quantities that survive an affine shift, or on a ratio whose
convention was checked not to invert.

**The stability dissociation does not reproduce.** The paper's finding is that
grid-like units hold their map across training while directionally modulated
ones do not. Here everything is stable (0.82–0.89) and the two groups barely
separate. A population that stable at 2e5 steps is one that stopped changing
early, which fits the loss curve flattening after ~epoch 150.

### The dropout ablation: no effect either way

The paper's claim (Extended Data Fig. 4) is that grid-like units *do not emerge*
without regularization. Six runs — seeds 0, 1, 2 crossed with `dropout_rates`
0.5 and 0.0, everything else identical, all scored on the same evaluation
trajectories (`results/nodropout/dropout_ablation_epoch299.png` shows the seed-0
pair):

| dropout | seed 0 | seed 1 | seed 2 | mean | s.d. | grid scale |
|---|---|---|---|---|---|---|
| 0.5 | 4.3% | 6.6% | 7.4% | **6.1%** | 1.6 | 102cm |
| 0.0 | 10.2% | 5.1% | 6.2% | **7.2%** | 2.7 | 88cm |

**No detectable effect.** The 1.1-point gap is effect size 0.47 on n=3 per
group — nothing. Removing dropout does lower the training loss as it should
(2.9–3.0 vs 3.7) without changing gridness, so the regularizer is doing
something; just not this.

> An earlier version of this section reported the seed-0 pair alone (4.3% vs
> 10.2%) as contradicting the paper. That was a single-seed artefact: 10.2% is
> the highest of all six runs, and the replication removes it. The claim is
> withdrawn.

Two things the sweep does establish, both firmer than anything a single run
could say:

**Our run-to-run s.d. is ~2.7 points, against the paper's 2.8 across 100
retrainings.** The noise level matches, so the setup is not unusually unstable —
and no single-run difference smaller than about 5 points means anything here.
The earlier 5.9-vs-7.2 clip/decay-scope comparison is retroactively confirmed as
noise.

**Under the default scoring, the shortfall is systematic across seeds, and the
scale finding is robust regardless of scoring.** All six runs land in
4.3–10.2% (mean 6.6%) against the paper's 21.9% at this width — far outside
seed noise, though (see above) the annulus question means the true magnitude
of that shortfall is now uncertain, not the fact that it's consistent across
seeds. Every one of the six has a mean grid scale of 84–107cm,
never near the paper's 66cm, while individual clean grid cells do form (best
unit reaches 1.07). So the network reliably produces *a few, very coarse* grid
cells rather than *many at several scales*. That is a sharper target than "too
few grid cells", and it is what the next hypothesis has to explain.

**What the path-integration result rules out.** The shortfall is not a failed
task: the network solves path integration to within 8% of the paper's own
figures, with a *better* effect size. It performs the same computation and
represents it differently, so the open question is what selects the
representation — and with the dropout hypothesis refuted, there is no current
candidate.

An earlier observation, still standing: the LSTM being unclipped under the
paper's scope did **not** destabilise training (weakening the idea that the
tiny `1e-5` clip acts as an implicit stabiliser — though loss not exploding is
weaker evidence than bounded weights, which have never been tracked).

## Known gaps

Ranked by how likely each is to matter.

1. **The gridness shortfall's evidential status just weakened.** See Results,
   "Everything about the representation," for the full annulus sensitivity
   sweep and the ruled-out hidden-fine-population check — in short, confidence
   that 4.3–10.2% reflects the network's true periodicity, rather than an
   unlucky scoring choice, is now lower, not fixed. Untried: the place-cell
   scale `pc_scale` (sets the spatial frequency of the *targets*, though
   already far finer than the observed grid scale, so its leverage here is
   uncertain); the circular arena, the paper's actual 256-unit condition; and
   a principled way to pin down the inner radius (e.g. per-unit, from each
   map's own central-peak width) rather than one global guess.
2. **Multi-seed statistics: 3 seeds, not the paper's 100.** `TrainConfig.seed`
   fixes weight init and batch order, and the dropout sweep above ran 3 seeds
   per configuration — enough to measure the noise floor (s.d. ~2.7 points,
   matching the paper's 2.8 across 100 retrainings) but not enough to resolve
   anything smaller than ~5 points. Any future single-run comparison should be
   read against that floor; the checkpoints are under
   `data/checkpoints/seed{1,2}_dropout{0.5,0.0}/`.
3. **Head-direction tuning is implemented but not measurable on this layer.**
   It reports 65/256 (25.4%) above the paper's 0.47 against its 10.2%, and
   26.6% on the superseded 512-unit run — but that difference should not be
   read as a result, for the reason below. The resultant-vector formula
   (Suppl. eqs. 6–7) assumes non-negative firing rates, but the bottleneck is
   a plain linear layer with signed activations, so each unit's tuning is
   shifted by its own minimum first — without that the measure is not even in
   [0,1] (a signed total passes through zero; mean length comes out at 32.9).
   The paper analysed linear-layer units too and never says how it handled this.
   The shift is `resultant_vector_length(..., shift_to_nonnegative=)`.

   **The shift IS the cause of the gap, and the measure is not trustworthy on
   a linear layer.** Two facts, both verified numerically against this run:

   *The shift destroys exactly the quantity being measured.* For a tuning curve
   `β = A·cos(α−φ) + B`, min-shifting gives `A·(1 + cos(α−φ))`, whose resultant
   is `A·n/2 ÷ A·n = ½` — **independent of both A and B**. Measured: r̄ = 0.506
   for every amplitude from 0.001 to 50 and every baseline from 0 to 1000. The
   same curves unshifted give 0.0001 … 0.49, tracking modulation depth as they
   should. The measure works by comparing modulation against baseline, and the
   min-shift deletes the baseline. Since 0.506 > 0.47, **any** unit with even a
   faintly cosine-shaped directional component is counted as tuned.

   That is visible in the data: 72 of 256 units sit in r̄ ∈ [0.42, 0.52], piled
   against the cutoff. Adding just 3.7% of the mean tuning span as a constant
   moves the count from 25.4% to exactly the paper's 10.2%, and 9.1% of the
   span drives it to 0.4%.

   *The convention alone spans the whole range.* Per-unit min → 25.4%,
   population-wide min → 0.0%, half-wave rectification → 71.1%. The paper's
   10.2% lies inside that interval. So our "overshoot" is not evidence that
   this network is more directional than the paper's; it is evidence that the
   number is set by an unstated convention.

   An earlier note here claimed the min-shift was the *most conservative*
   baseline and so could not inflate the count. That compared it against
   half-wave rectification and against no shift at all. Against the case that
   matters — a firing rate with its own positive baseline — min-shift is the
   **most liberal** constant shift, because it is the smallest constant that
   still clears zero, and r̄ falls monotonically in that constant.

   Any fix has to restore a baseline the linear layer does not have. Until one
   is justified rather than picked, the honest reading of this row is "not
   measurable here", not "25.4% vs 10.2%".

   Related: the 0.47 cutoff is the Rayleigh critical value at **n = 20**
   (0.4720 at α = 0.01, against 0.4966 at n = 18 and 0.4507 at n = 22), so the
   paper treated its 20 angular bins as the 20 observations. Changing
   `scores.HD_BINS` invalidates the threshold.
4. **Two ambiguities inherited from the paper's own text**, both in
   Supplementary Methods 3d and both resolved here by a documented choice
   rather than by guessing at the paper's:
   - *The discreteness histogram range.* The paper fixes it at "scales 10 to 36
     spatial bins". That does not reconcile with its own reported 28–115cm at
     any bin width: its shuffle jitter of ±7 bins is half the smallest scale,
     putting that scale at 14 bins = 28cm and so a bin at 2cm — under which the
     stated top of the range, 36 bins = 72cm, falls short of the reported
     maximum of 115cm. `scores.discreteness` works in metres and lets each
     histogram span its own data instead.
   - *The annulus inner radius*, as above.
5. **Square environment only.** The paper also validates a 2.2m-diameter
   circular arena (Extended Data Fig. 3d, 21.9%).
6. **`ego_vel` composition is a reconstruction**, not a confirmed fact —
   DeepMind never published it.
7. **`parameter_updates` ambiguity.** Table 1 states 300,000; the released
   flag defaults imply 1,000,000. We use 300,000.
8. **The bottleneck bias is the paper's, and untested.** Methods: *"The linear
   decoder consists of three sets of weights **and biases**. The first set …
   map from the LSTM hidden state m⃗ₜ to the linear layer activations
   g⃗ₜ ∈ ℝ⁵¹²."* `ModelConfig.bottleneck_has_bias = True` follows that; the
   released code has no bias here. Set it `False` for the released code's
   behaviour. No run has compared the two. Note this is the mirror of
   deliberate difference 3 above: the paper omits a bias the code has in the
   LSTM init, and specifies one the code lacks in the bottleneck.

## Roadmap: beyond the supervised network

The paper's other half — a vision-based RL agent reusing these
representations to navigate DeepMind Lab mazes (Fig. 2–4) — was never
open-sourced ("the codebase for the deep RL agents makes use of proprietary
components"). Planned, in build order:

1. **Vision module** — CNN over 64×64 RGB (Supplementary 3b: four conv
   layers, 16/32/64/128 filters, 5×5, stride 2, pad 2, ReLU, then FC-256).
2. **Actor–critic policy** — A3C-style, taking vision features plus this
   network's representations, six discrete actions.
3. **Environment** — DeepMind Lab (`lab/`, a git submodule) is buildable
   here; see "RL environment setup" below. Fig. 2's open-field task is
   reproducible via `random_goal_factory.lua`; the goal-driven/goal-doors
   multi-room environments (Fig. 3) have no public source and need to be
   reconstructed from the Methods description.
4. **Validation** — the paper's finding is that grid-like periodicity
   *re-emerges* in the agent's own units (21.4% at 256 units); the existing
   `scores.py` pipeline carries over directly.

### RL environment setup

DeepMind Lab targets a 2018-era toolchain and doesn't build out of the box on
a modern machine. It's included as the `lab/` git submodule (pinned to a
specific upstream commit, unmodified — no fork needed) rather than copied in:
its own asset-packaging step bundles its entire ~2.4GB `assets/` tree
regardless of which level is actually used, and `engine/`/`q3map2/`/
`assets_oa/` are GPL-licensed, so copying that content into this repository
would both bloat every future clone and risk entangling this project's
license with the GPL.

```
git clone --recurse-submodules <this repo>
# or, after a plain clone:
git submodule update --init
```

Then, once (before touching any RL code):

1. `pip install -r requirements-rl.txt`
2. Run `notebooks/06_setup_rl_environment.ipynb` top to bottom. It applies
   the handful of small patches this toolchain needs directly (Bazel version
   pin, a couple of dependency-version swaps, one Python-3.12+ compatibility
   fix), builds `lab/` with Bazel, and installs the result as a
   `deepmind_lab` wheel — importable alongside `torch` in the same process.
