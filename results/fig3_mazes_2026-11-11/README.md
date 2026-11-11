# Fig. 3a/b maze renders

Images of the levels [`levels/goal_driven.lua`](../../levels/goal_driven.lua) (Fig. 3a) and
[`levels/goal_doors.lua`](../../levels/goal_doors.lua) (Fig. 3b), made on 2026-11-11 so the mazes
can be compared by eye with the paper's figure. They are not training results.

## How they were made

- Seeds 0–3 per level, 512 × 512, `renderer="hardware"` (Mesa's CPU renderer, as there is no
  GPU), `goalVisible=true`.
- Each image was taken after resetting the episode and stepping one no-op frame.
- A new maze is generated every episode, so one seed is one maze. **For the same seed both
  levels have the same maze structure**; only goal-doors adds doors (e.g. `goal_driven_seed0`
  and `goal_doors_seed0` are the same maze).
- The script that made them is not kept in the repo. To regenerate, open both levels with the
  settings above and save the observations listed in the table below.

## Files

File names are `{level}_seed{N}_{kind}`.

| Kind | Contents | DMLab observation |
|---|---|---|
| `_topdown.png` | Overhead view of the maze | `DEBUG.CAMERA_INTERLEAVED.TOP_DOWN` |
| `_view.png` | The agent's first-person view at the spawn point | `RGB_INTERLEAVED` |
| `_spin.png` | Four first-person views at the spawn point, turning right by about 90° each. Left to right: 0°, 90°, 180°, 270° (±2°) | `RGB_INTERLEAVED` |
| `_layout.txt` | Text layout of the maze | `DEBUG.MAZE.LAYOUT` |

### Reading `_topdown.png`

- North is up, in the same orientation as `_layout.txt` (not mirrored).
- The camera has perspective, so walls lean towards the centre.
- **Sky-blue areas**: empty spaces enclosed by walls (the inner `*` cells of `_layout.txt`).
  The paper's figure shows these as white.
- **Black panels**: doors (goal-doors only). A panel across a passage is a closed door; a thin
  strip pushed against the wall is an open door.
- **White and blue object**: the agent (player model).
- **Small orange-red object**: the goal. It looks small from above; for a close view see the
  left edge of the first panel of `goal_doors_seed0_spin.png`.
- **Small pictures on the walls**: the wall cues (intra-maze cues).

### Reading `_view.png` and `_spin.png`

- The numbers at the bottom are DMLab's HUD (health, armour, etc.); the top right is the time
  left (episodes are 90 s, so it counts down from 1:30). Fig. 3c/d show the same HUD.
- The grey buildings above the walls are the distal cues (a skybox drawn at infinity).

### Reading `_layout.txt`

11 lines of 11 characters; each character is one cell, outer border included.

| Character | Meaning |
|---|---|
| `*` | Wall: the outer border, and empty spaces enclosed by walls |
| (space) | Walkable empty cell (corridor or room floor) |
| `P` | **Spawn candidate**. The actual spawn point is drawn at random from the `P` cells reachable from the goal, and drawn again on every teleport after reaching the goal |
| `G` | **Goal candidate**. The actual goal is drawn at random from these once per episode and stays fixed within it |
| `H` | Door passed going north–south; the door panel lies east–west (goal-doors only) |
| `I` | Door passed going east–west; the door panel lies north–south (goal-doors only) |

Notes:
- The file holds **candidates only**. It does not record which goal and spawn were chosen, or
  whether each door is open or closed; check door states in `_topdown.png`.
- `P` and `G` cells have the same floor as empty cells; the letters only mark room cells.
- Coordinates: one cell is 100 world units (0.25 m). The cell in line i from the top and
  character j from the left (both from 1) is centred at x = (j − 1)·100 + 50,
  y = (11 − i)·100 + 50, the same frame as the agent-position observation `DEBUG.POS.TRANS`.

Example (`goal_driven_seed0_layout.txt`):

```
***********
*PPP***   *
*GGP*** P *
*GGP***GPP*
* ***** PG*
*      GPG*
* ******* *
* *GPP*PPG*
* *PPP*PGP*
*  GGG GPG*
***********
```

The rooms are the four clusters of `P`/`G` (top left, top right, bottom middle, bottom right),
joined by corridors of empty cells. The `*` cells inside the border (top middle, the horizontal
band across the middle, and so on) are empty spaces enclosed by walls; they are the sky-blue
areas in `goal_driven_seed0_topdown.png`.
