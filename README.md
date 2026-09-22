# grid-cells-torch

PyTorch reimplementation of [google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells)
(Banino et al. 2018, Nature, "Vector-based navigation using grid-like
representations in artificial agents"): the paper's **supervised**
path-integration network (Fig. 1), plus a **from-scratch RL agent**
(Fig. 2–4) that reuses it — the paper's RL codebase was never open-sourced.

The original is TF1 + Sonnet v1 and does not run on current versions, so this
is a from-scratch port, not a patch. It also uses our own dataset, generated
in `notebooks/00_make_dataset.ipynb` from the paper's Supplementary Table 1
motion-model parameters, rather than the original's unpublished TFRecord
release.

## Setup

Requires Python 3.12 (the floor set by `numpy`/`scipy` in `requirements.txt`;
every other pin there supports 3.12+ too).

```
pip install -r requirements.txt
```

**Supervised network** — run `notebooks/00_make_dataset.ipynb` through
`05_cell_types.ipynb` in order. Each stage's output feeds the next:

```
data/square_room_100steps_2.2m_1000000/*.csv   00_make_dataset.ipynb
data/shards/*.npz                              01_prepare_data.ipynb
data/checkpoints/<run>/checkpoint_epoch*.pt     02_train.ipynb (~28 min on a GTX 1650)
results/<run>/*.png, *.pdf                      03/04/05_*.ipynb (path integration,
                                                 bottleneck ratemaps, cell types)
```

`.py` files are library code; notebooks are the only entry points, with one
exception (`train_rl_agent.py`, see below) — no other module has a CLI.
`data/` is gitignored (large, regenerable from the notebooks); `results/<run>/`
is committed (small, one folder per training run — `<run>` is whatever
`cfg.train.results_dir` was set to). `evaluate.default_out_dir()` pairs
`data/checkpoints/<run>/` with `results/<run>/` automatically. A fresh clone
has `results/` but no `data/`. `tests/` holds verification scripts (currently
all RL, since the supervised pipeline's checks live inline in `02_train.ipynb`'s
own cells instead) — not part of the pipeline, run to confirm nothing broke.

**RL agent** additionally needs DeepMind Lab, pulled in as the `lab/` git
submodule (kept as a submodule, not copied in, because its asset-packaging
step bundles an unrelated ~2.4GB and part of it is GPL-licensed):

```
git clone --recurse-submodules <this repo>   # or, after a plain clone:
git submodule update --init
```

Then run `notebooks/06_setup_rl_environment.ipynb` top to bottom once — it
patches the 2018-era toolchain (Bazel version pin, a couple of dependency
swaps, one Python-3.12+ fix), builds `lab/` with Bazel, and installs
`deepmind_lab` as a wheel. **06 is the last setup notebook, not the last
notebook overall** — `07_play_checkpoint.ipynb` exists (below), but there is
deliberately no notebook for RL *training* itself, for a reason worth stating
plainly:

Training itself is `python3 train_rl_agent.py` (`--resume-from <checkpoint>`
continues an interrupted run; see "RL agent" below) — a genuine CLI script,
not a notebook, because it isn't a bounded interactive step the way 00–06 are.
`train_rl_agent()` (`rl_train.py`) spawns 32 actor processes plus separate
vision-, grid-, and checkpoint-worker processes via `torch.multiprocessing`,
and runs for up to the paper's 1e9 env steps — observed at roughly 12–18 days
of wall-clock time for the full budget. A Jupyter kernel would have to stay
alive that entire time, doesn't recover cleanly from an interrupted session
the way a plain `nohup ... &` process does, and spawning multiprocessing
children from inside a notebook kernel is known to fight with Jupyter's own
IPC. `02_train.ipynb` already anticipates this same tradeoff for the
supervised pipeline even though it never needs to act on it: *"train.py is a
library with no CLI, so a detached run would need a driver script written for
the purpose."* `train_rl_agent.py` is exactly that driver script, built
because the RL loop is the one stage here that actually needs it.

Evaluating a checkpoint, by contrast, *is* a bounded operation — measured at
18.1s end-to-end for one 30-second episode video (15.2s of live DMLab
env+inference, 1.7s video encoding) — so `notebooks/07_play_checkpoint.ipynb`
handles that interactively, importing the same model-reconstruction helpers
(`build_models_from_checkpoint` etc.) that `rl_train.py` itself uses.

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
for weight decay. `TrainConfig.grad_clip_scope` / `ModelConfig.weight_decay_scope`
accept `"bottleneck_and_heads"` (the original's unused `clip_bottleneck_gradient`)
and, for clipping, `"all"` (its actual default).

**Caveat on clipping.** The paper justifies clipping by exploding gradients
in *recurrent* networks, yet scopes it to the feedforward heads only, leaving
the LSTM unclipped — unresolvable from the paper alone, which is why all
three scopes remain selectable.

Other deliberate differences: **weight decay is actually applied** (the
original registered Sonnet `regularizers` but never summed them into the
loss — set `ModelConfig.weight_decay = 0.0` for the original's literal
behaviour); **`nh_embed` is dropped** (stored but never used); **a learned
bias in the LSTM init** that the paper's equation omits but both
implementations include.

**The 0.37 gridness cutoff** is one fixed value for every unit, derived by
the paper from a per-unit shuffle and then collapsed (Methods: *"the means,
over units, of the thresholds obtained were >0.37"*) — re-deriving our own
per-unit threshold was dropped, since it can only relabel the model, not
improve it.

**The annulus inner radius is ours, not the paper's.** `8→20` is the outer
radius the paper gives; the inner one it gives only as "the central peak
excluded", no value — excluding it is not optional, since the rotationally
symmetric central peak washes out the gridness contrast otherwise.
`DEFAULT_INNER_RADIUS_BINS = 4.0` is what the released code's fractional
`0.2` came to at its own `nbins=20`; see Known gaps.

## RL agent

The paper's RL agent — vision-based navigation in DeepMind Lab, reusing the
supervised network's representations — is **built and training**, not just
planned:

1. **Vision module** (`vision.py`) — CNN over 64×64 RGB (Supplementary 3b:
   four conv layers, 16/32/64/128 filters, 5×5, stride 2, pad 2, ReLU, then
   FC-256) decoding to place/head-direction probabilities, masked per-unit.
2. **Grid network** (`model.py`'s `GridCellsRNN`, extended with an `rl_lstm`
   branch) — takes egocentric velocity *and* the vision module's decoded
   place/HD probabilities every step, not velocity alone.
3. **Actor–critic policy** (`actor_critic.py`) — A3C-style LSTM, taking
   vision features, the grid code, a goal grid code, reward, and previous
   action; six discrete actions.
4. **Environment** (`levels/square_arena.lua`) — a self-contained DMLab
   level (deliberately not built on `factories/random_goal_factory.lua`; see
   the file's own header) reproducing Fig. 2's open-field task. The
   goal-driven/goal-doors multi-room environments (Fig. 3) have no public
   source and still need reconstructing from the Methods description.
5. **Training loop** (`rl_train.py`, driven by `train_rl_agent.py`) —
   Hogwild-style multiprocessing: 32 actor processes plus separate vision-,
   grid-, and checkpoint-worker processes sharing CPU model/optimizer state.
   Checkpointing is full-fidelity and resumable: `checkpoint_worker` writes
   a light `checkpoint_step{N}.pt` (weights + optimizer) and a full
   `checkpoint_latest.pt` / `checkpoint_step{N}_final.pt` (+ both replay
   buffers, ~19.7GB) every `checkpoint_every_env_steps` (500,000) and on
   graceful SIGTERM; `--resume-from <path>` continues `step_counter` from
   `N`, not 0 — verified end-to-end (`tests/rl_resume_mp_smoke_test.py` and
   in a real crash-recovery). There is currently no automatic restart on
   reboot; resuming after any interruption is a manual re-invocation.
6. **Validation** (not yet run) — the paper's finding is that grid-like
   periodicity *re-emerges* in the agent's own units (21.4% at 256 units);
   `scores.py` carries over directly once training has progressed further.

### Fixes made since this was first wired up

Roughly chronological; every item below is a real behavioural bug already
fixed in the current code, not a style change.

- **Grid-network optimizer used the supervised (Table 1) LR/weight decay
  instead of the RL-specific (Table 2) ones** — fixed via `build_optimizer`
  override params and `GridRLConfig.weight_decay`.
- **Vision module's place/HD predictions were computed but never fed into
  the grid LSTM** — `vision_t = cat(place_probs, hd_probs)` now reaches
  `grid_network.step()`.
- **Vision masking was whole-vector, not per-unit** (one coin flip per
  timestep instead of independent per-unit masking), and was applied before
  the place/HD heads instead of after — both corrected.
- **Intra-maze cue was a floating pickup**; Fig. 2b shows a wall-mounted
  texture — now a wall decal.
- **Spawn position was one of 36 fixed grid-cell centres**, not continuous —
  fixed to uniform sampling over the full 1.5×1.5m central region.
- **RL velocity input collapsed `[speed, sin, cos]` (3-dim)**, discarding
  strafe — fixed to `[u, v, sin, cos]` (4-dim), since DMLab allows
  independent forward/lateral motion.
- **The paper-mandated σ=0.01 Gaussian velocity noise was missing** from the
  RL rollout — now injected via `RLEnvConfig.velocity_noise_std`.
- **The wall-redirect angle (θ_RH) in `00_make_dataset.ipynb`** was a
  variable "turn parallel to the wall" instead of the paper's literal fixed
  ±90° — this is the *supervised* dataset's generator, caught while
  auditing the same Table 1 parameters this work also depends on.
  Regenerating the dataset with the fix is why the Results below carry a
  new date.
- **`SequenceReplayBuffer` didn't store images**, needed to retrain vision
  jointly with the grid network from replayed sequences — added.
- **A CUDA driver crash under sustained 32-actor load.** All RL models are
  CPU tensors by design, yet PyTorch's `RMSprop.step()` unconditionally
  queries the CUDA driver whenever one is *present in the system* — with 32
  processes doing this every step, this produced `torch.AcceleratorError`
  and crashed most actors at once (observed on a GTX 1650 under WSL2).
  Fixed via `CUDA_VISIBLE_DEVICES=""` in `train_rl_agent.py`, since
  nothing here touches the GPU.

## Dataset reconstruction

Our CSV has corner-origin `pos_x,pos_y` (~[0, 2.2]), allocentric `vel_x,vel_y`
and `head_direction_{x,y}` unit vectors — the original TFRecord used
center-origin coordinates and an unpublished 3-component egocentric `ego_vel`.
`dataset.py`'s `csv_shard_to_arrays` derives `pos_centered = pos − 1.1`,
`theta = arctan2(hd_y, hd_x)`, and `ego_vel = [speed, sin(dtheta), cos(dtheta)]`
— `dtheta` (the net heading change actually realized over the step, from
consecutive stored `head_direction` vectors) rather than the raw `rot_vel`
column, which is an instantaneous rate sampled at one fine 0.02s instant
inside each 0.15s step and structurally can't represent the step's net
change. Supplementary Table 1 lists `T=15`, `Δt=0.02` and `trajectory
length=100` as three separate rows: `Δt` is the fine physics step (needed for
the `d=0.03m` wall perimeter to register at all), not the spacing of the 100
stored steps — `00_make_dataset.ipynb` simulates at 750 fine steps, then
resamples to 100 steps spaced `T/100 = 0.15s` apart.

## Results

### Supervised network

Run of 2026-08-29, at the default `nh_bottleneck=256`: 300,000 gradient steps
(29m13s on a GTX 1650), trained on the wall-redirect-corrected dataset,
paper-scoped clipping/weight decay, `seed=0`. Loss **7.679 → 3.852**,
monotone. Evaluated at `checkpoint_epoch299.pt` over 4000 trajectories
(`results/baseline/`).

**Path integration reproduces the paper.** The paper never says how it
decodes position from the place cells, so three readouts are measured; `top3`
is what `evaluate.py` selects:

| decoder | trained, t=15s | untrained | floor | effect size |
|---|---|---|---|---|
| argmax | 18.3cm | 89.7cm | 6.9cm | 2.49 |
| weighted mean | 26.5cm | 82.5cm | 6.8cm | 2.31 |
| **top3** | **16.0cm** | **95.3cm** | 6.8cm | **2.53** |
| *paper* | *16cm* | *91cm* | — | *2.83* |

*Floor* is the error a perfect path integrator would still score, since a
256-cell code can only name cell centres — 16.0cm is 9.2cm of network error
on top of 6.8cm of readout resolution. The error curve: 28.3cm at t=0 (LSTM
settling out of its injected init), down to 9.6cm at t=1.80s, then drifting
up at ~0.49cm/s over the remaining ~13s.

**The representation itself is short, over, or absent relative to the
paper**, at 256 units against its 21.9% (Ext. Data Fig. 3d) headline:

| measure | ours | paper |
|---|---|---|
| gridness > 0.37, bottleneck | 20/256 (7.8%) | 21.9% |
| gridness > 0.37, raw LSTM | 0/128 (0.0%) | ~0 |
| head direction > 0.47 | 109/256 (42.6%) — *not measurable, see Known gaps* | 10.2% |
| border score > 0.50 | 13/256 (5.1%) | 8.7% |
| conjunctive (both cutoffs) | 8 (40% of grid units) | 14 (11% of 129) |
| grid scale | 90–124cm, mean 108 | 28–115cm, mean 66 |
| scale clusters (BIC) | 1 (underpowered, n=16) | 3, ratios ~1.5 |
| stability, 2e5 vs 3e5 | all 0.895 / grid 0.866 / directional 0.874 | grid stable, directional **not** |

The scale rows rest on only 16 grid-like units against the paper's 129 —
underpowered, not a result on their own — but the mean itself (108cm, near
the lowest spatial frequency a 2.2m arena can express hexagonal structure at
all) is a genuine, concrete finding: our grid-like units are only the
coarsest ones, and the paper's 28–70cm population is simply absent. The
head-direction row is not a real disagreement (see Known gaps 3): the
non-negativity shift the resultant-vector formula needs collapses *any*
cosine-shaped tuning curve toward r̄≈½, above the 0.47 cutoff regardless of
modulation depth, so the number is set by an unstated convention, not by how
directional this network actually is. The stability dissociation the paper
reports (grid-like units stable, directional ones not) does not reproduce —
everything here is uniformly stable, consistent with the loss curve
flattening early (~epoch 150).

A prior sweep of the annulus inner radius (on an earlier, now-pruned
checkpoint) found the grid-like count swings several-fold on this one
unstated scoring parameter — not confirming the 7.8% shortfall is an
artefact, just lowering confidence that it reflects the network's true
periodicity rather than a scoring choice. Not yet re-run on this checkpoint;
see Known gaps 1.

### RL agent — first real run (2026-08-20), frozen at spawn

*Written from this project's own conversation record, not from saved files —
the run's checkpoint and analysis figures were deleted outside of version
control and are not recoverable. Treat the specifics below as a documented
finding to be double-checked, not as reproducible artefacts.*

The first extended RL training run (`data/checkpoints/migration_2026-08-20/`,
predating every fix listed above) ran to step 429,626,388 of the 1e9-step
budget (~43%) before being stopped. Inspecting it turned up a genuinely
degenerate policy: across recorded episodes, the agent's position was
confined to exactly 36 discrete values (confirmed by rounding to millimetre
precision, ruling out "moves a little within a small area") — it rotates in
place but essentially never translates from its spawn point. This was not an
artefact of the vision module itself: evaluating vision-module position
decoding on the same checkpoint gave a mean error of ~4.8cm, i.e. the vision
pathway can localize the agent reasonably well even though the policy never
uses that to move. Two candidate explanations were checked and ruled out
(an action-index/movement mapping bug; the agent being wall-stuck). Four
were not: goal-code bootstrapping order, advantage collapse, Hogwild gradient
dilution across 32 concurrent actors, and an uninformative grid signal this
early in joint vision+grid+policy training. None confirmed.

Several of the fixes above (vision→grid-LSTM wiring, velocity noise, u/v
velocity, spawn discretization) postdate this run and were partly motivated
by it, but a fresh run since those fixes has not yet reached a comparable
step count to say whether the frozen-policy pattern is resolved.

## Known gaps

Ranked by how likely each is to matter.

1. **The gridness shortfall's evidential status is unresolved** — see
   Results above; the annulus-radius sensitivity needs re-checking on the
   current checkpoint. Also untried: `pc_scale` (place-cell target
   resolution); the circular arena, the paper's actual 256-unit condition;
   a principled (e.g. per-unit) way to pick the inner radius.
2. **Multi-seed statistics were only measured pre-dataset-fix.** A 3-seed ×
   2-dropout sweep on the old dataset found no detectable dropout effect
   (Extended Data Fig. 4's claim) and a run-to-run noise floor of ~2.7 points
   (matching the paper's 2.8 across 100 retrainings) — those checkpoints have
   since been pruned and the sweep hasn't been repeated on the corrected
   dataset.
3. **Head-direction tuning is not measurable on a linear layer**, not just
   different from the paper. For a tuning curve `A·cos(α−φ) + B`, the
   non-negativity min-shift the resultant-vector formula (Suppl. eqs. 6–7)
   needs gives a resultant of exactly `½`, independent of both `A` and `B` —
   verified numerically (r̄ = 0.506 across amplitudes 0.001–50 and baselines
   0–1000, against 0.0001…0.49 unshifted). Since 0.506 > 0.47, any unit with
   even a faint cosine-shaped component counts as tuned; the convention
   alone spans the plausible range (per-unit min / population-wide min /
   half-wave rectification), with the paper's 10.2% lying inside it. The
   0.47 cutoff itself is the Rayleigh critical value at n=20
   (`scores.HD_BINS`); changing that bin count invalidates the threshold.
4. **Two ambiguities in the paper's own Supplementary Methods 3d**, each
   resolved here by a documented choice: the discreteness histogram's
   "10 to 36 spatial bins" range doesn't reconcile with its own reported
   28–115cm at any single bin width (`scores.discreteness` works in metres
   and lets each histogram span its own data instead); and the annulus inner
   radius, above.
5. **Square environment only** — the paper also validates a 2.2m-diameter
   circular arena (21.9%, Ext. Data Fig. 3d).
6. **`ego_vel` composition is a reconstruction**, not a confirmed fact —
   DeepMind never published it.
7. **`parameter_updates` ambiguity** — Table 1 states 300,000; the released
   flag defaults imply 1,000,000. We use 300,000.
8. **The bottleneck bias is the paper's, and untested.** Methods specifies a
   bias here that the released code lacks (`ModelConfig.bottleneck_has_bias`);
   no run has compared the two. Mirror of the LSTM-init bias difference above.
9. **RL agent: sampled hyperparameter ranges are fixed to their midpoint.**
   Supplementary Table 2's bracketed values (`learning_rate_range`,
   `baseline_cost_range`, `entropy_reg_range`) are, per its own caption,
   sampled per actor-learner — a population-based jitter this codebase
   instead fixes to `sum(range) / 2` for every actor. Still unimplemented;
   not confirmed or ruled out as a contributor to the frozen-policy pattern
   above.
