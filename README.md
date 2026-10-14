# grid-cells-torch

PyTorch reimplementation of the paper below: its **supervised** path-integration network
(Fig. 1), ported from DeepMind's TF1 release, plus a **from-scratch RL agent** (Fig. 2–4) in
DeepMind Lab — the paper's RL codebase was never released.

## Paper

> Banino, A., Barry, C., Uria, B., Blundell, C., Lillicrap, T., Mirowski, P., Pritzel, A.,
> Chadwick, M. J., Degris, T., Modayil, J., Wayne, G., Soyer, H., Viola, F., Zhang, B.,
> Goroshin, R., Rabinowitz, N., Pascanu, R., Beattie, C., Petersen, S., Sadik, A., Gaffney, S.,
> King, H., Kavukcuoglu, K., Hassabis, D., Hadsell, R. & Kumaran, D.
> **Vector-based navigation using grid-like representations in artificial agents.**
> *Nature* 557, 429–433 (2018). [doi:10.1038/s41586-018-0102-6](https://doi.org/10.1038/s41586-018-0102-6)

This repository is an independent reimplementation; we are not the authors. The paper and its
figures belong to their authors and publisher, and the original supervised-learning code is
[google-deepmind/grid-cells](https://github.com/google-deepmind/grid-cells) (Apache-2.0). Work
that builds on these results should cite the paper (BibTeX in
[`docs/paper/README.md`](docs/paper/README.md)).

## Setup

Python 3.12+, then:

```
pip install -r requirements.txt
git submodule update --init lab        # DeepMind Lab, for the RL agent only
bash scripts/setup_dmlab.sh            # patch + build + install deepmind_lab (tens of minutes)
```

`bash run_scripts.sh` prints every common command. In short:

| Step | Command | Output |
|---|---|---|
| Dataset | `python generate_data.py` | `data/datasets/<id>/shards/*.npz`, `data/latest` |
| Supervised training | `python train.py` | `data/checkpoints/baseline/` (~30 min on a GTX 1650) |
| Supervised analysis | `python analyze_supervised.py --checkpoint <ckpt>` | `results/<run>/` |
| RL training | `python train_rl_agent.py` | `data/checkpoints/<run>/` |
| RL evaluation | `python eval_rl_agent.py` | `results/rl_eval_step<N>/` videos and trajectory plots |
| Tests | `pytest -m "not dmlab and not slow"` | |

## Layout

```
config.py               all hyperparameters (dataclasses)
generate_data.py        CLI: simulated trajectories -> npz shards
train.py                CLI: supervised training
analyze_supervised.py   CLI: path integration, gridness, cell types
train_rl_agent.py       CLI: RL training
eval_rl_agent.py        CLI: RL episodes -> videos, Fig. 2h-style plots
grid_cells/             library code (no entry points)
levels/                 DeepMind Lab levels (square_arena.lua)
scripts/                setup_dmlab.sh and the DeepMind Lab build patch
tests/                  pytest suite
docs/paper/             spec notes, figure notes, deviations from the paper
docs/plan/              current plan and STATUS.md
docs/maintenance.md     conventions for working in this repo
data/                   gitignored: datasets and checkpoints
results/                committed: small figures and videos per run
```

## Following the paper

Where the paper and DeepMind's released code disagree, this port follows the paper. Every
deliberate difference, every choice made where the paper is silent, and the known gaps are in
[`docs/paper/deviations.md`](docs/paper/deviations.md). What the paper specifies, with page
citations, is in [`docs/paper/spec-supervised.md`](docs/paper/spec-supervised.md) and
[`docs/paper/spec-rl-agent.md`](docs/paper/spec-rl-agent.md).

## Results

### Supervised network

Run of 2026-08-29, at the default `nh_bottleneck=256`: 300,000 gradient steps
(29m13s on a GTX 1650), trained on the wall-redirect-corrected dataset,
paper-scoped clipping/weight decay, `seed=0`. Loss **7.679 → 3.852**,
monotone. Evaluated at `checkpoint_epoch299.pt` over 4000 trajectories
(`results/baseline/`).

**Path integration reproduces the paper.** The paper never says how it
decodes position from the place cells, so three readouts are measured; `top3`
is what `grid_cells/evaluate.py` selects:

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
| head direction > 0.47 | 109/256 (42.6%) — *not measurable, see [Known gaps](docs/paper/deviations.md#known-gaps)* | 10.2% |
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
head-direction row is not a real disagreement (see [Known gaps](docs/paper/deviations.md#known-gaps) 3): the
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
see [Known gaps](docs/paper/deviations.md#known-gaps) 1.

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

### RL agent — A3C, full 1e9-frame run (finished 2026-10-02)

Flat learning curve: mean episode reward 0.93 (goal found in 7.7% of episodes) from the first
100M frames to the last, against the paper's ~289. Evaluated episodes show the agent walking to
a wall and stopping. Seven defects in the A3C loop were found afterwards; the run, the defects
and the learner-update ratios are recorded in
[`docs/plan/baseline-a3c.md`](docs/plan/baseline-a3c.md). The A3C loop is kept frozen in
`grid_cells/rl_a3c.py` as the baseline; the replacement is a synchronous GPU (A2C) learner
(plan: [`docs/plan/2026-10-goal-finding.md`](docs/plan/2026-10-goal-finding.md)).
