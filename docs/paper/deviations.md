# Deviations from the paper and from the released code

Every place this implementation knowingly differs from Banino et al. (2018) or from
[google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells), and every choice made
where the paper is silent. What the paper itself says is in `spec-supervised.md` and
`spec-rl-agent.md`; this file is about our side. Add an entry whenever code departs from a spec
note or fills one of its "Not specified" gaps.

## Supervised network: following the paper, not the released code

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

## Supervised dataset reconstruction

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

## RL agent: decisions where the paper is silent or ambiguous

Status as of 2026-10-08. IDs match `docs/plan/baseline-a3c.md` and plan §6. "A3C" is the frozen
legacy loop (`rl_train.py` at `55ef4ea`, later `grid_cells/rl_a3c.py`); "A2C" is the replacement.

| ID | Topic | Paper | A3C (legacy) | A2C (current) |
|---|---|---|---|---|
| A | n-step bootstrap | `γ^n V(s_{t+n})` (supp p.16) | used `V(s_t)` — bug | `V(s_{t+n})` from the next observation |
| B | reward input to policy LSTM | `r_t` is an input (main p.7) | always 0 — bug | previous step's reward |
| C | grid LSTM state at a teleport | starts at zero; must learn to reset from `y`, `z` (main p.7) | re-initialized from the pre-teleport frame's vision output — bug | zero at episode start, not reset at goal teleports |
| D | goal grid code at a new episode | zeros until the goal is reached in the episode (main p.7) | carried over from the previous episode — bug | zeroed at every episode start |
| E | goal arrival | not an episode end; agent continues (supp p.4) | policy LSTM reset and return cut (bootstrap 0) | episode continues: no reset, bootstrap through |
| F | vision masking granularity | "5% of the time" vs "zeroed the units with a probability of 95%" (main p.2, p.7) | per-unit | per-unit by default, whole-vector selectable |
| G | grid replay sequences across teleports | implied by C | cut at teleports | cut only at episode ends |
| I | goal visibility (square arena) | "an unmarked goal" (main p.6), "navigate to an unmarked goal" (supp p.4) | `levels/square_arena.lua` places the map character `G`, which `common/make_map.lua` turns into the visible `goal` pickup (`models/goal_object_02.md3`) — the goal is a visible object | option added: `--hidden-goal` / `RLEnvConfig.goal_visible=False` draws the goal with a zero-area model (same reward and collision; verified visually and by reward). Default still visible pending a decision |
| H | initial state of replayed grid sequences | grid LSTM "set to zero", not ground truth (main p.7) | ground-truth place/HD init (`train_step`) | zero state (`grid_train_zero_init`) |
| — | actor–critic RMSProp | Table 2: "gradient momentum 0.99 — momentum parameter of the shared RMSProp" (decay rate not given) | decay 0.99, momentum 0.99, eps 1e-8 | decay 0.99, **momentum 0**, eps 1e-8. Reading 0.99 as RMSProp *decay* (as in the A3C paper's shared RMSProp) rather than momentum. Evidence (2026-10-08, tiny_goal): momentum 0.99 collapsed policy entropy 1.79 → 0.001 within 6e5 frames and the agent froze; momentum 0 kept entropy ~1.4–1.6 and raised reward 17.5 → ~47 in 1.5e6 frames |
| — | vision optimizer | not specified | RMSProp lr 1e-4, decay 0.99 | same |
| — | unroll length / action repeat | Table 2: 100 / 4 (supp p.25) | 100 / 4 | 100 / 4 |
| — | hyperparameter ranges in Table 2 | sampled per replica (60 replicas, main p.7) | midpoint of each range | midpoint of each range |
| — | vision / grid update frequency | unspecified (one thread each) | as fast as each process runs | fixed counts per rollout, from the legacy run's measured ratios |
| — | replay buffer size | 20 × 10^6 steps (main p.7) | 200k frames + 2k sequences, float32 | uint8 frame store (smaller; RAM-bound) |
| — | 32 asynchronous learners | Hogwild A3C (main p.7) | as paper | synchronous batched A2C on one GPU |

## RL agent: fixes made to the legacy loop before the A2C switch

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
9. **RL agent: Table 2's bracketed ranges are fixed to their midpoint.** Corrected reading
   (2026-10-08): the caption says the values "are sampled from a categorial distribution in that
   range", and main p.7 says "60 replicas were run with hyperparameters sampled from the same
   interval and different initial random seeds" — the sampling is **per replica (training run)**,
   not per actor-learner, as an earlier version of this entry claimed. A single run using the
   midpoints is therefore one legitimate draw-like setting, not a missing per-actor mechanism.
   What remains unreplicated is the 60-replica population and the "best 30 of 60" selection used
   for the paper's curves (main p.8).
