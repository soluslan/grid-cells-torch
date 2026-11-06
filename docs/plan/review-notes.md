# Review notes

Inbox for things learned while reading through the code and the paper, and for changes worth
considering later. Nothing here is a decision yet. When an item is settled, move it to
`docs/paper/deviations.md` (choices and known gaps) or the relevant spec note, and delete it here.

Page numbers refer to `docs/paper/banino2018-main.pdf` (main) and
`docs/paper/banino2018-supplementary.pdf` (supp).

## Candidate changes

### README compares a square-arena result against the circular-arena number

`README.md:99-103` puts our 20/256 (7.8%) grid-like units next to the paper's 21.9%. That 21.9% is
from the circular arena (ED Fig. 3e). Our run is square arena + 256 units, which matches no
experiment in the paper. The paper's square-arena figures use 512 units: 129/512 (25.2%) in
Fig. 1, and a 23% mean over 100 seeds in ED Fig. 3c. The comparison should either change its
reference number or explain why it is used.

### Gridness ratemaps are probably 20×20, not 32×32

We score gridness and grid scale on 32×32 ratemaps (`analyze_supervised.py:32`). That value
comes from one sentence in main p.8 Methods: "Spatial bins were defined as a 32 × 32 square grid
spanning each environment".

Every other source points to 20×20 instead:

- **The paper's figures.** Ratemaps were measured from the images embedded in the main PDF.
  - Fig. 1 embeds them as raw 20×20 images, and its SACs as 41×41.
  - ED Fig. 2, ED Fig. 3d and ED Fig. 6a all show 20 bins per side.
  - ED Fig. 4 shows about 20, judged by eye because the image is blurred.
  - Fig. 2 (RL agent) shows 22×22.
  - No figure uses 32×32.
  - Fig. 1 and ED Fig. 3d print a gridness value above each 20-bin SAC, so those gridness values
    were computed at 20 bins.
- **Supp p.17.** It gives no resolution for gridness. Its annulus grows "up to a maximum of 20"
  bins, which is exactly the half-width of the SAC of a 20-bin map. For a 32-bin map, whose SAC
  half-width is 31, stopping at 20 has no stated reason.
- **The released code.** It uses `GridScorer(20, ...)` (`grid-cells/train.py:200-201`).

32 appears in only one other place: the ED Fig. 3b caption, for the stability analysis ("32
spatial bins in each map").

There are two possible readings. Neither can be settled from the paper alone.

- **A. 32 is real but belongs to other analyses.** Stability is the one stated case. The
  statistics may also have come from a separate Matlab pipeline (supp cites `fitgmdist`), with
  the figures drawn separately at 20 bins.
- **B. 32 is left over from an earlier version, or a typo.** The paper's other bin numbers also
  disagree with each other: see Known gaps 4, the discreteness histogram's "10 to 36 spatial
  bins" and "±7 bins".

Proposed check: score the same checkpoint at 20 and at 32 bins, and compare the grid-like count
and the scale distribution. Whatever the authors intended, this shows how much our
gridness-shortfall and large-scale results depend on resolution.

#### Analysis settings across sources

| | Main text | Supp | Released code | Ours |
|---|---|---|---|---|
| Gridness / scale ratemap | text: 32×32 (p.8); figures: 20×20 | not stated; consistent with 20×20 | 20×20 | 32×32 |
| Stability ratemap | 32 (ED 3b caption) | – | – | 32 |
| Border-score ratemap | – | 20×20 (p.18) | 20×20 | 20×20 |
| SAC inner radius | – | "central peak excluded", no value | 4 bins (`0.2 × 20`) | 4 bins |
| SAC outer radius | – | 8→20 bins, step 2 (7 annuli) | 8→20 bins, 10 evenly spaced values (step 1.33) | 8→20, step 2 |
| Interim gridness | – | `min(c60,c120) − max(c30,c90,c150)` | mean-based (`min_max=False` default) | min/max |

Released-code references:

- the annulus is `starts = [0.2] * 10` and `ends = np.linspace(0.4, 1.0, 10)`, multiplied by
  `nbins` (`grid-cells/train.py:198-201`, `grid-cells/scores.py:82-85`)
- the gridness formula is at `grid-cells/scores.py:87-92`

### Annulus inner radius: same bin count, different physical size

The paper gives no inner radius. Ours is 4 bins (`grid_cells/scores.py:9`), apparently copied from
the released code. The released code applies 4 bins to 20-bin maps; we apply it to 32-bin maps.
In a 2.2 m arena, the same bin counts therefore cover different distances:

| | Bin size | Inner radius (4 bins) | Outer radii (8→20 bins) |
|---|---|---|---|
| 20-bin maps (paper figures, released code) | 11 cm | ≈ 44 cm | ≈ 88–220 cm |
| 32-bin maps (ours) | 6.9 cm | ≈ 27.5 cm | ≈ 55–137.5 cm |

This feeds into Known gaps 1: the grid-like count is highly sensitive to the inner radius.

If we keep 32-bin maps, we have two ways to match the released code's physical sizes:

- set the inner radius to 6.4 bins and scale the outer radii the same way
- switch to 20-bin maps, as the previous item proposes

### Circular-arena experiment not run

`grid_cells/data_generation.py` only handles square walls (`wall_zone` / `wall_redirect`). This
is listed in Known gaps 1 and 5, but no reason is recorded.

Running it would give three things:

- a like-for-like comparison with the paper's 256-unit condition
- a test of whether our large grid scale (mean 108 cm vs the paper's 0.58 / 0.96 m clusters) comes
  from arena geometry
- the paper's own check that the result does not depend on the square arena

Required work:

- a circular boundary in the simulator
- dataset regeneration
- a circular ratemap mask
- a decision on the gridness cutoff: ED Fig. 3g draws its line at 0.42, not 0.37

## Facts learned (no action needed)

- **256 units in the circular arena is inferred, not stated.** It comes from 56 / 0.219 ≈ 256
  (ED Fig. 3e). 512 units would give 10.9%. A second check is ED Fig. 3g: "five grid units (9%)",
  and 5/56 = 8.9%. The RL agent's 256 is stated explicitly: ED Fig. 6a, main p.14.
- **Gridness omits the 0° rotation because it changes nothing.** Supp p.17 lists 0°, 60° and 120°.
  An unrotated map always correlates with itself at 1, and every correlation is ≤ 1, so
  `min(c0, c60, c120) = min(c60, c120)`.
- **The RL checkpoints differ greatly in size.** In `data/checkpoints/rl_baseline/`:

  | Files | Contents | Size |
  |---|---|---|
  | Most | weights only | ≈ 35 MB |
  | `checkpoint_latest.pt`, `checkpoint_step1000000092_final.pt` | weights + optimizer + both replay buffers (`.gitignore`) | 19.7 GB each |

  All of `data/checkpoints/` is ≈ 100 GB.
