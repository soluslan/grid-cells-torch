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
| Bottleneck width | 512 (Fig. 1) / 256 (Ext. Data Fig. 3e, RL agent) | 256 | 256 |
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

**Dropout is off at evaluation, and inverted during training.** The paper
says only "drop probability 0.5 … applied to each g⃗ₜ unit", resampled at
each time step; it never says whether ratemaps were generated with dropout
on, or whether survivors are scaled ×2 in training or activations ×0.5 at
test. Both choices match the released code (`tf.nn.dropout` under
`self.training`): `F.dropout` is applied only when `model.training`
(`grid_cells/model.py:106-110`), so analysis sees the full, unscaled g⃗ₜ.
With a linear decoder the two scaling conventions differ only by a constant
factor on g⃗ₜ, which gridness (a correlation) ignores.

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

Status as of 2026-10-08; "Ours" is the A2C loop (`grid_cells/rl_a2c.py`). Only choices and
real departures are listed here. Defects of the frozen A3C loop where the paper is explicit
(IDs A–D, H, and the visible goal of I) are recorded in
[`docs/plan/baseline-a3c.md`](../plan/baseline-a3c.md). IDs match `baseline-a3c.md` and plan §6.

| ID | Topic | Paper | Ours | Why |
|---|---|---|---|---|
| — | episode-start resets | only `g*` is specified: "zeros if the goal had not yet been reached in the episode" (main p.7); the grid LSTM is "set to zero" | grid LSTM, policy LSTM, goal grid code, previous reward and previous action all zeroed at each episode start | standard A2C practice for the unspecified ones |
| E | goal arrival | not an episode end; agent continues (supp p.4); whether the policy LSTM is reset or the return is cut is not stated | no reset, n-step return bootstrapped through the teleport | standard practice within an episode |
| F | vision masking granularity | "5% of the time" vs "zeroed the units with a probability of 95%" (main p.2, p.7) | per-unit by default, whole-vector selectable | Methods' "zeroed the units" reads per-unit; the flag keeps the other reading testable |
| G | grid replay sequences across teleports | not stated; the grid net must "learn how to reset its internal state each time it was teleported" (main p.7) | sequences cut only at episode ends, so they may span teleports | the grid net can only learn to reset from sequences that contain a teleport |
| I | goal visibility (square arena) | "an unmarked goal" (main p.6, supp p.4) | `--hidden-goal` / `RLEnvConfig.goal_visible=False` draws the goal with a zero-area model (same reward and collision; verified visually and by reward). **`goal_visible` still defaults to True** for the legacy levels: pass `--hidden-goal` to reproduce the paper setting and the 1e9-frame run | a visible goal lets the policy CNN, which sees the unmasked frame, walk to the object without the grid code, so the paper's claim could not be tested (decided 2026-10-09) |
| J | closed-door fraction (goal-doors) | doors change "randomly ... each time the agent reached the goal" (main p.6); no fraction given | `levels/goal_doors.lua`: on every restart all doors close, the doors on one random spawn→goal path open, then half of the rest open (`doorsClosed = 0.5`) | the paper names the public DMLab level `explore_obstructed_goals` as this task (main p.8); 0.5 is that level's value |
| — | maze size (goal-driven, goal-doors) | "11 × 11 grid, corresponding to 2.7 × 2.7 m" (main p.6) | the DMLab generator's 11 × 11 cells include the outer walls; at the square arena's 0.25 m per 100-unit cell that is 2.75 m. The goal is the visible default goal object, as in the paper (Fig. 3c) | 2.7 is presumably 2.75 rounded; no agent-side geometry is set up for these levels yet (`RLEnvConfig` holds square-arena values) |
| — | actor–critic RMSProp | Table 2: "gradient momentum 0.99 — momentum parameter of the shared RMSProp" (decay rate not given) | decay 0.99, **momentum 0**, eps 1e-8 | 0.99 read as RMSProp *decay* (as in the A3C paper's shared RMSProp). Evidence (2026-10-08, tiny_goal): momentum 0.99 collapsed policy entropy 1.79 → 0.001 within 6e5 frames and the agent froze; momentum 0 kept entropy ~1.4–1.6 and raised reward 17.5 → ~47 in 1.5e6 frames |
| — | vision optimizer | not specified | RMSProp lr 1e-4, decay 0.99 | kept from the legacy loop |
| — | unroll length / action repeat | Table 2: 100 / 4 (supp p.25) | 100 / 4 | Methods' "updates … every four actions (that is, action repeat of 4)" could be read as a 4-step unroll; Table 2 lists the two separately |
| — | hyperparameter ranges in Table 2 | sampled per replica (60 replicas, main p.7) | midpoint of each range | one run, not a population (see Known gaps 9) |
| — | vision / grid update frequency | unspecified (one thread each) | fixed counts per rollout | taken from the legacy run's measured ratios (`baseline-a3c.md`) |
| — | replay buffer size | 20 × 10^6 steps (main p.7) | uint8 frame store, smaller | RAM-bound |
| — | 32 asynchronous learners | Hogwild A3C (main p.7) | synchronous batched A2C on one GPU | decided 2026-10-08 (plan §1); ~4,600 frames/s vs the A3C loop's ~1,000 on the same machine |

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
   circular arena (21.9%, Ext. Data Fig. 3e).
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
