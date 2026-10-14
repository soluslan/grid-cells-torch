# Spec: the grid cell RL agent

What the paper specifies for the deep-RL agent (Fig. 2–4, Extended Data Figs. 5–10), with
sources. Statements the paper does **not** make are listed explicitly at the end, because the
gaps are where implementation choices (and bugs) live. Our choices are in `deviations.md`.

## Environment: square arena (Fig. 2)

Source: main p.6 ("Custom environment: square arena"), supp p.4 (1a).

- 10 × 10 grid = 2.5 × 2.5 m "assuming an agent speed of 15 cm/s".
- One coloured intra-arena cue; its position and colour change each episode, as do floor
  texture, wall texture and goal location. Distal cues (buildings rendered at infinity) give
  direction but not distance and are the same in every episode. The cue is a wall texture
  (Fig. 2b image, see `figures.md`).
- Start: random location within the central 6 × 6 grids (1.5 × 1.5 m).
- The goal is unmarked. Reaching it gives +10 and teleports the agent to a new random location;
  the episode **continues** ("Upon reaching the goal the agent was teleported to another random
  location and continued to navigate", supp p.4).
- Episode = 5,400 environment steps ≈ 90 s (supp p.4). So one environment step = 1/60 s, i.e.
  one rendered frame.
- Velocity input noise: Gaussian, μ = 0, σ = 0.01, applied throughout training and testing
  (main p.6).
- Six discrete actions with fine-grained control: rotate in small increments, accelerate
  forward/backward/sideways, rotate while moving (main p.6). The exact six are not listed.

Other environments (goal-driven, goal-doors, 11 × 11 = 2.7 × 2.7 m, procedurally generated; probe
mazes) are in main p.6–7 and supp p.13–14. Not yet implemented here.

## Architecture (Extended Data Fig. 5)

Source: main p.7 ("Agent architectures"), supp p.14 (3b), ED Fig. 5 (main p.13).

**Vision module**
- Input: RGB 64 × 64 image in `[-1, 1]` (the text also says `3×84×84`, a typo).
- CNN: 4 conv layers, 16/32/64/128 filters, 5 × 5, stride 2, padding 2, ReLU; then FC 256 (supp
  p.14). The actor–critic uses the same CNN architecture with separate (unshared) weights.
- A linear layer on the CNN embedding predicts place and head-direction activations (`y`, `z`),
  trained with the same cross-entropy loss as the supervised network (main p.7).
- Masking — three statements that do not obviously agree:
  1. "provided as input to the grid network 5% of the time on average" (main p.2, p.7; ED Fig. 5
     "only provided 5% of the time");
  2. "the output of the convolutional network `e` was then passed through a masking layer which
     zeroed the units with a probability of 95%" (main p.7);
  3. for grid training from replay, `c` and `h` "were then passed through a masking layer that
     masked them to 0 with a probability of 95%" (main p.7).
- Stop-gradient between the vision module and the grid network (ED Fig. 5).

**Grid network**
- Same as the supervised network, except the grid LSTM "was not initialized with ground truth
  place cell activations but instead set to zero" (main p.7).
- Inputs: translational velocities `u`, `v` (DMLab allows moving in a direction different from
  facing), `sin`/`cos` of angular velocity, and the vision module's `y`, `z` (main p.7).
- "the grid cell network had to use `y` and `z` to learn how to reset its internal state each
  time it was teleported to an arbitrary location in the environment (for example, after visiting
  the goal)" (main p.7).
- Place-field centres fixed for the whole of training (main p.7). Linear layer: 256 units in the
  agent (ED Fig. 6a).
- Stop-gradient between the grid network and the actor–critic (ED Fig. 5).

**Actor–critic ("policy LSTM")**
- Input: concatenation of the actor–critic CNN's FC output `e'_t`, reward `r_t`, previous action
  `a_{t-1}`, current grid code `g_t` (linear-layer activations), and goal grid code `g*` — the
  linear-layer activations "observed last time the goal was reached — or zeros if the goal had not
  yet been reached in the episode" (main p.7).
- LSTM 256. Actor: linear 6 + softmax. Critic: linear 1 (main p.7).

## Training

Source: main p.7 ("Training algorithms", "Agent training details"), supp p.16 (3c), supp p.25
(Table 2).

- Loss: `L = L_π + α L_V + β L_H` (A3C, Mnih et al. 2016, main ref. 41), with the n-step return
  `R̂_t = Σ_{i=0..n-1} γ^i r_{t+i} + γ^n V(s_{t+n})` (supp p.16).
- 32 actor–critic learner threads on one CPU machine; RMSProp with shared statistics; 1e9
  environment steps (main p.7).
- Replay: one shared buffer of per-step experience `e_t = (φ_t, u_t, v_t)`, total size 20 × 10^6,
  written by all 32 workers. Vision: minibatches of 32 frames. Grid network: minibatches of 10
  sequences of 100 consecutive observations, run through the vision module, masked (statement 3
  above), concatenated with `[u, v, sin, cos]`. One thread each for vision and grid training
  (34 threads in total). No gradient sharing among actor–critic, vision and grid (main p.7).
- Table 2 (supp p.25). Bracketed values "are sampled from a categorial distribution in that
  range"; 60 replicas per agent were run "with hyperparameters sampled from the same interval
  and different initial random seeds" (main p.7) — sampling is per replica (run).

| Parameter | Value |
|---|---|
| learning rate (actor–critic, shared RMSProp) | [1e-6, 2e-4] |
| gradient momentum (actor–critic) | 0.99 |
| baseline cost α | [0.48, 0.52] |
| entropy regularisation β | [6e-5, 1e-4] |
| discount γ | 0.99 |
| back-propagation steps, actor–critic ("used to unroll the LSTM") | 100 |
| action repeat | 4 |
| grid network learning rate (RMSProp) | 0.001 |
| place cell scale σ(c) | 40 (units not printed; read as 0.4 m) |
| M head-direction cells / κ(h) | 12 / 20 |
| back-propagation steps, supervised learner | 100 |
| L2 regularisation (linear layer) | 1e-4 |
| gradient momentum (supervised learner) | 0.9 |

**Unroll length vs. action repeat.** main p.7 says "All threads applied updates to their
gradients every four actions (that is, action repeat of 4)", which on its own could be read as
"update every four actions". Table 2 lists unroll length 100 and action repeat 4 as separate
rows, and supp 3c's return uses `V(s_{t+n})`. Reading adopted: one action every 4 frames, an
update after every 100 actions (400 frames), and "1e9 environment steps" = 1e9 frames.

## Evaluation and reported results

| Quantity | Paper | Source |
|---|---|---|
| Square arena, average score over 100 episodes | grid cell agent 289, place cell agent 238 | supp p.4 |
| Self-localization error after 15 s | 12 cm vs 88 cm untrained | main p.2, Fig. 2e |
| Grid-like units in the agent's linear layer | 21.4% | main p.3 |
| Without goal grid code | falls to A3C level (123.22 vs 112.06) | ED Fig. 6c |
| Latency to goal, first vs later visits in an episode | ~510 vs ~120 steps | ED Fig. 6d (image) |
| Goal-driven / goal-doors, 100 episodes | grid 346.5 / 284.3; place 258.76 / 90.53; A3C 137.00 / 48.69; human 261 / 240.5 | supp p.8 |
| Square arena, heterogeneous place-cell controls | 660 cells: 212; 256 cells: 225 | supp p.5 |
| Goal-code lesions (25% of units) | most grid-like silenced 126.1, random 152.5, patchy non-grid 151.7 | supp p.6–7 |
| Learning curves | see `figures.md` (Fig. 2f, 3e–f, ED 6–7) | |

- Training curves (Fig. 2f, 3e, f) show the 30 best of 60 replicas by average cumulative reward
  over 100 episodes (main p.8).
- Replica selection is stated inconsistently: main p.8 says the 30 best of 60; ED Fig. 6b–c
  captions say "average of 50% best agent replicas (n = 32)".
- Lesion training: a fresh mask silences 20% of the goal grid code every 100 training steps
  (main p.7); not needed for the main result.
- The primary place-cell control uses homogeneous 7.5 cm fields, best of 7.5 / 25 / 75 cm
  (supp p.15).
- Fake goal probe: frozen weights; after the sixth goal arrival, replace `g*` with the grid code
  from a random previously visited point (main p.8, Fig. 2i).

## Not specified by the paper

- Vision module optimizer, learning rate, and how often it is updated relative to acting.
- How often the grid network is updated relative to acting.
- Whether "5% of the time" means whole-vector masking per step or per-unit masking.
- Whether the policy LSTM state is reset at a goal teleport (only "zeros if the goal had not yet
  been reached in the episode" for `g*`).
- Whether a goal arrival cuts the n-step return (it is not an episode end, supp p.4).
- Whether replayed 100-step grid sequences may span a teleport (the grid net is expected to learn
  to reset itself after teleports, main p.7, which suggests they do).
- Number of place cells in the RL setting (Table 2 lists σ(c), M and κ but not N).
- The six actions' exact DMLab action vectors. (The grid network's own hyperparameters, e.g.
  dropout 0.5, are implied: "implemented as in the supervised learning set up" and "derived
  from the best performing network in the supervised learning experiments", main p.7.)
- The units of "Training steps" on the learning-curve x axes (`figures.md`).
