# Spec: supervised path-integration network

What the paper specifies for the supervised experiment (Fig. 1, Extended Data Figs. 1–4), with
sources. Code mapping and our own choices are in `deviations.md`, not here.

## Data: simulated rat trajectories

Source: main p.6 ("Simplified 2D environment"), supp p.24 (Table 1).

- Square (or circular) enclosure, side (or diameter) `L`. Start at a uniformly sampled location
  and facing angle. Motion model: Raudies & Hasselmo (2012) (main ref. 31).
- Table 1 motion parameters:

| Symbol | Value | Meaning |
|---|---|---|
| `T` | 15 | trajectory duration (s) |
| `L` | 2.2 | width/height, or diameter for circular (m) |
| `d` | 0.03 | perimeter region distance to walls (m) |
| `σ(v)` | 0.13 | forward velocity Rayleigh scale (m/s) |
| `μ(φ)` | 0 | rotation velocity Gaussian mean (deg/s) |
| `σ(φ)` | 330 | rotation velocity Gaussian s.d. (deg/s) |
| `ρ_RH` | 0.25 | velocity reduction factor in the perimeter |
| `Δ_RH` | 90 | change in angle in the perimeter (deg) |
| `Δt` | 0.02 | simulation step (s) |

- "trajectory length 100 — number of time steps in the trajectories used for the supervised
  learning task" (Table 1). With `T=15` and `Δt=0.02` there are 750 simulation steps; how the
  100 stored steps relate to them is not stated (see `deviations.md`).

## Targets

Source: main p.6.

- Place cells: `N` centres `μ_i` drawn uniformly in the enclosure before training, fixed.
  Activation = posterior of a mixture of isotropic 2D Gaussians with scale `σ(c)`:
  `c_i = exp(-|x-μ_i|²/2σ²) / Σ_j exp(-|x-μ_j|²/2σ²)`.
- Head-direction cells: `M` centres uniform in `[-π, π]`, von Mises with concentration `κ(h)`,
  same posterior form.
- Table 1: `N = 256`, `σ(c) = 0.01` m, `M = 12`, `κ(h) = 20`.

## Network

Source: main p.6 ("Grid cell network architecture"), Extended Data Fig. 1 (main p.9).

- Input at each step: `[v_t, sin(φ̇_t), cos(φ̇_t)]` — egocentric linear velocity and the sine and
  cosine of angular velocity (main p.6, "Supervised learning inputs").
- One LSTM, 128 hidden units, no peepholes.
- Initial state: `l_0 = W(cp) c_0 + W(cd) h_0`, `m_0 = W(hp) c_0 + W(hd) h_0` — linear
  transforms of the ground-truth place and head-direction activations at time 0, trained.
  (The equation has no bias term.)
- Linear layer `g_t`: **512** units in the main experiment ("linear layer activations
  `g_t ∈ R^512`"); 256 in the circular-arena setting (inferred: ED Fig. 3e reports 56 grid-like units = 21.9%, i.e. of 256) and in the RL agent (ED Fig. 6a).
- Dropout 0.5 on each `g_t` unit. No nonlinearity in the decoder.
- Two linear + softmax heads from `g_t`: place predictions `y_t`, head-direction predictions `z_t`.

## Loss and optimisation

Source: main p.6, supp p.24 (Table 1).

- Loss: `L = -Σ_i c_i log y_i - Σ_j h_j log z_j` at every step.
- BPTT unrolled in blocks of 100 steps. RMSProp.
- Weight decay on the **weights** projecting from `g_t` to `y_t`, `z_t`.
- Gradient clipping, element-wise to `[-gc, gc]`, on the **parameters** projecting from `g_t`
  to `y_t`, `z_t` (main p.6, "Gradient clipping").
- Weight-decay target is stated three ways: Methods (main p.6) "weights projecting from the
  dropout linear layer `g_t` to the place and head-direction predictions"; Table 1 "Regularisation
  parameter for linear layer"; Table 2 "linear layers in bottleneck". We follow Methods.
- Table 1: `gc = 1e-5`, minibatch 10, learning rate `1e-5`, momentum 0.9, L2 `1e-5`,
  300,000 parameter updates. (`1e-5` prints as `10 5` in the text extraction.)

## Reported results (the targets to compare against)

| Quantity | Paper | Source |
|---|---|---|
| Path-integration error after 15 s | 16 cm trained vs 91 cm untrained, effect size 2.83 | main p.1, Fig. 1c |
| Grid-like units (gridness > 0.37) | 129/512 (25.2%) | main p.1 |
| Over 100 retrainings | mean 23%, s.d. 2.8% | main p.1, ED Fig. 3c |
| Grid scale | 28–115 cm, mean 66 cm; 3 clusters at 47, 70, 106 cm, ratio ~1.5 | main p.1, Fig. 1e |
| Head-direction units (resultant > 0.47) | 10.2% | main p.1 |
| Border units (border score > 0.50) | 8.7% | main p.1 |
| Conjunctive grid × direction | 14 (11% of grids) | Fig. 1g caption |
| Circular arena, 256 units | 56 grid-like (21.9%), 2 scale clusters 0.58 / 0.96 m (ratio 1.66), 5 directional grids (9%); gridness threshold drawn at **0.42** | ED Fig. 3e, 3g |
| No dropout | no grid-like units | ED Fig. 4 |
| Stability (2e5 vs 3e5 steps) | grid-like units stable, directional units not | ED Fig. 3b |

## Analysis definitions

Source: main p.8 ("Neuroscience-based analyses"), supp p.16–19 (3d).

- Ratemaps: 32×32 spatial bins; directional maps: 20 bins; mean activity per bin, no smoothing.
- Gridness: spatial autocorrelogram; annulus centred at the origin, "radius of 8 bins and with
  the central peak excluded", rotated in 30° steps; interim score =
  `min(corr 60°, 120°) − max(corr 30°, 90°, 150°)` (text: "highest correlation obtained from
  rotations of 30, 90 and 150 subtracted from the lowest at 0, 60 and 120"); expand the annulus
  by 2 up to 20; gridness = highest interim score (supp p.17). The inner radius is not given.
- Grid scale: median distance from the origin of the six local maxima closest to (excluding)
  the central peak (supp p.17).
- Resultant vector length over 20 directional bins (supp p.17, eqs. 6–7); threshold 0.47 from
  a Rayleigh test at α = 0.01 (supp p.19).
- Border score: 20×20 bins, `d_b = 3`, `max_i (b_i − c)/(b_i + c)` (supp p.18).
- Thresholds 0.37 (gridness) and 0.50 (border) are means over units of per-unit field-shuffle
  95th percentiles (main p.8).
- Stability: bin-wise Pearson correlation between ratemaps at `t = 2e5` and `3e5`, unvisited
  bins excluded (main p.8).
- Thresholds were cross-checked against 500 untrained networks; the stricter (shuffle) threshold
  was used (supp p.19).
- Scale discreteness: histogram of 13 bins over scales 10–36 spatial bins, s.d. of counts, vs
  500 shuffles in which each scale is jittered uniformly by ±7 bins (supp p.19–20). Scale clusters: Gaussian mixtures with 1–8 components fit by EM (Matlab `fitgmdist`), chosen by BIC (supp p.20, ED Fig. 3a).
