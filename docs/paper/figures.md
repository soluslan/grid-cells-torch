# Figures: what only the images show

Images live in `figures/` (gitignored local copies; see `README.md`). Captions are in the text
extraction (`text/main.txt`, pages 2–4 for Figs. 1–4 and 9–18 for Extended Data). This file
records what the **picture** adds beyond the caption, and when a task should reopen the image.
Reading the caption text does not count as having looked at the figure — this project once built
the intra-arena cue as a floating pickup because only the text had been read.

## Main figures

**Fig. 1** (`fig1.png`) — supervised network.
- (a) Initial position enters the LSTM only at `t = 0` (switch symbol). Both place and HD heads
  hang off one "Linear" box.
- (b) One 15 s trajectory in the 2.2 m box, decoded path (dark) hugging the true one (light).
- (c) Error histogram: after training almost all mass below ~0.3 m; before training a broad hump
  centred near 1.1 m.
- (d) Example units: gridness 1.18, 1.15, 0.52, 0.83, 0.66 (grids), 0.17 (border-like), −0.05
  (head-direction unit with a clear polar lobe).
- (e) Scale histogram with three fitted peaks at 0.47 / 0.70 / 1.06 m, ratios 1.49 / 1.51; the
  x axis spans 0.2–1.2 m.
- (g) Scatter of gridness vs resultant length with dashed lines at 0.37 and 0.47; a dense band of
  units sits near resultant ≈ 0.75.
- Reopen when: comparing our scale distribution, ratemap appearance, or the gridness/resultant
  scatter.

**Fig. 2** (`fig2.png`) — square arena, the main RL target.
- (b) Overhead view: blue tiled floor, red-brick walls, the **intra-maze cue is a small light
  texture on the top wall** (arrow labelled "Intra-maze cue"), not an object in the arena.
- (c) First-person view: distal cues are tall grey buildings visible above the walls; HUD numbers
  at the bottom.
- (d) Architecture: velocity and the vision module both feed the grid LSTM directly; the grid
  LSTM produces the grid code, and PCP / HDP (place and head-direction predictions) hang off the
  grid code box as its output heads. The grid code branches into the policy LSTM's "goal grid
  code" and "current grid code" inputs, alongside reward, previous action and the CNN output.
- (f) **Learning curve**: grid cell agent rises steeply to ~250 by x ≈ 0.5 and plateaus around
  ~300 by x = 2.5; place cell agent ~100 at 0.5, ~200 at 2.5. x axis "Training steps (×10^8)",
  range 0–2.5. y axis is reward within a single episode (caption).
- (h) Top-down trajectories in one episode: first trajectory dashed grey, later ones solid black,
  each later trajectory starting at an "S" and running fairly directly to the goal.
- (i) Fake-goal probe: dense black loops circling an empty location.
- Reopen when: building or changing the arena level (textures, cue placement, distal cues), making
  evaluation plots in the Fig. 2h style, or reading off learning-curve milestones.

**Fig. 3** (`fig3.png`) — goal-driven / goal-doors mazes. Green-and-yellow walled rooms, red
goal object, black doors. Curves (e, f): grid cell agent at x ≈ 0.5 already ~270–280 (goal-driven) and ~230 (goal-doors); ~340 (goal-driven) and
~280 (goal-doors) at 2.5; human dashed lines at ~260 and ~240. Reopen when building these mazes.

**Fig. 4** (`fig4.png`) — sunburst maze (5 corridors with doors) and double E-maze; trajectory
bundles and corridor-usage histograms. Reopen when building the probe mazes.

## Extended Data

- **ED 1** — supervised architecture; the ground-truth `c_0`, `h_0` enter through a switch at
  `t = 0` only. Input written `[v_t, sin(φ̇_t), cos(φ̇_t)]`.
- **ED 2** — ratemaps of all 512 linear units: most are visibly periodic or multi-field; useful as
  a visual benchmark for "what a healthy layer looks like".
- **ED 3** — (a) BIC minimum at 3 components; (b) stability histograms; (c) grid-unit proportion
  across 100 seeds centred ~23%; (d–g) circular arena, threshold line at 0.42 in (g).
- **ED 4** — no dropout: ratemaps are blobby/irregular, no periodic structure.
- **ED 5** — full agent diagram. Two **STOP GRADIENT** bars: vision module → grid network, and
  grid network → actor–critic. Grid LSTM input drawn as `[u, v, sin φ̇, cos φ̇]` plus `[y, z]`.
  Reopen when touching gradient flow between modules.
- **ED 6** — (a) the agent's 256 linear units, grid/border/place-like maps; (b) curves with and
  without distal cues overlap, ~250 at 0.5, ~300 at 2.5; (c) without goal code ≈ A3C (~100–130 at
  2.5) vs intact ~330, with a "Human" line at ~260; (d) **time to reach goal: ~510 steps for the first arrival in an episode,
  ~125 for arrivals 2–3, falling to ~110 by arrival 8** (x axis: "Number of runs with goal in same position"). ED 6d is the cleanest behavioural signature of vector navigation
  and should be part of our evaluation.
- **ED 7** — (a–c) AUC robustness over 60 replicas; (d–i) other agents vs grid agent.
- **ED 8** — (a) plain A3C: policy LSTM gets `r_t`, `e'_t`, `a_{t-1}` only; (b) place cell agent
  gets ground-truth `[c, h]` and goal `[c*, h*]`.
- **ED 9** — place-cell-prediction agent and NavMemNet (key/value memory, 1,350 slots).
- **ED 10** — shortcut probes: trajectories, corridor histograms, test scores.

## The x axis of the learning curves

Fig. 2f, 3e–f, ED 6b–c and ED 7d–i all label the x axis "Training steps (×10^8)" and end at 2.5.
Methods say every experiment ran for 1e9 environment steps. Two readings:

1. "Training steps" are agent steps (actions). 2.5e8 actions × action repeat 4 = 1e9 frames, so
   the plotted range is the whole run.
2. "Training steps" are environment steps (frames), and the plots show the first quarter of the
   run.

ED 6d's latencies (~510 then ~120 "steps") fit a 5,400-step episode scoring ~289 only if those
steps are frames, so the paper uses "steps" for frames at least there. The question is open for
the learning curves. Our milestone gates are written in frames and read the curve in the more
lenient way (reading 1); see the plan.
