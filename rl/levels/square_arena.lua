--[[ Paper's "square arena" open-field task (Fig. 2b,c; Methods "Custom environment: square
arena"). No public source -- reconstructed here from the Methods text, which the RL-agent
roadmap plan flags as a substantial task in its own right, not glue code around an existing
level.

Methods, verbatim spec this implements:
  "This comprised a 10x10 square arena, corresponding to a 2.5x2.5m arena... The arena
  contained a single, randomly changed, intra-maze cue whose position and colour changed each
  episode, as did the texture of the floor, the texture of the walls and the goal location...
  The agent always started at the central 6x6 grids (that is, 1.5x1.5m) of the environment...
  Upon reaching the goal the agent was teleported to another random location and continued to
  navigate with the aim of maximising the number of times it reached the goal before the
  episode ended."
  Episode length: Supplementary Results 1a states 5,400 steps (~90s) per episode.

GOAL POSITION IS FIXED FOR THE WHOLE EPISODE (2026-08-14 fix). "cue ... changed each episode, as
did [floor texture], [wall texture] and [the goal location]" is a list of things that all change
on the SAME cadence -- once per 90s episode, not once per goal-touch. This is the paper's Morris
water maze framing: the agent must relearn a path back to one fixed hidden goal from repeated
random starting points within an episode, not chase a freshly-relocated goal every touch. Only
the agent's own spawn point re-randomizes on every touch ("teleported to another random
location"). An earlier version of this file re-randomized the goal position (and, worse,
recompiled the entire map via q3map2) on every touch -- both a task-design bug and, with 32
concurrent RL actors, the dominant cause of a >14x training-throughput collapse (see the
RL-agent roadmap plan's "Throughput fix" section). Fixed by splitting `api:start()` (true
episode boundary: pick new goal/cue positions + a new floor/wall texture, full map rebuild) from
`api:nextMap()` (may fire many times per episode, on every goal touch: only the agent's spawn
point changes, via a `''`-mapName "fast restart" -- see
`lab/game_scripts/factories/random_goal_factory.lua` for the same pattern -- so no recompile).

Deliberately NOT `require`-ing a shared factory module: the paper's task (single unmarked goal,
no scattered reward pickups, a hard central-6x6 spawn constraint) doesn't match
`factories.random_goal_factory`'s maze-topology/scattered-fruit assumptions closely enough for a
thin wrapper, and any project-owned `rl/factories/*.lua` file couldn't be `require`d from here
anyway -- `require` inside an externally-loaded level resolves against the `lab/` submodule's
own game_scripts root, not this level's own directory (confirmed empirically via
`rl/smoke_test.py`). So this file is self-contained, `require`ing only submodule modules.

PLACEHOLDERS -- flagged rather than guessed at, per user request (2026-08-12) to ask before
inventing unspecified visuals:
  - intra-maze cue: modelled as a single `fut_obj_cylinder_01.md3` prop (an existing simple
    geometric prop already used elsewhere in game_scripts), re-positioned each episode. The
    paper doesn't say what physical object it is, and per-episode recolouring was left out here
    (would need the model's own texture name, unverifiable without opening `assets/`, which
    CLAUDE.md's own reading guide marks as binary content not worth inspecting) -- only its
    position varies episode-to-episode, not yet colour.
  - distal cues ("buildings rendered at infinity"): NOT implemented. No existing skybox asset
    with distant buildings was found referenced anywhere in game_scripts (only the generic
    `map/lab_games/sky/lg_sky_03` sky texture `make_map.lua` already defaults to). Known gap,
    not a silent omission -- revisit with reference material if/when available.
]]

local custom_observations = require 'decorators.custom_observations'
local make_map = require 'common.make_map'
local map_maker = require 'dmlab.system.map_maker'
local pickups = require 'common.pickups'
local random = require 'common.random'
local texture_sets = require 'themes.texture_sets'
local timeout = require 'decorators.timeout'
local randomMap = random(map_maker:randomGen())

-- 10x10 interior grid = the paper's "10x10 square arena"; each cell is 0.25m (2.5m / 10).
local GRID = 10
-- "central 6x6 grids": cells 3..8 (inclusive, 1-indexed) of the 10-wide interior.
local CENTER_MIN, CENTER_MAX = 3, 8
local EPISODE_LENGTH_SECONDS = 90  -- Supplementary Results 1a: 5,400 steps @ 60fps

local CUE_MODEL = 'models/fut_obj_cylinder_01.md3'  -- placeholder, see header note
-- A REWARD-type pickup with quantity=0, not `pickups.defaults.apple_reward`: the cue is meant
-- to be a purely visual landmark, and reusing an actual reward pickup would corrupt the reward
-- signal (touching it would score +1 for no reason the task defines). type=GOAL would be worse
-- -- that restarts the episode on touch. quantity=0 REWARD is a harmless touch (score +0,
-- entity despawns) until/unless a non-interactive classname is confirmed to work instead.
local CUE_PICKUP = {
    name = 'Cue', classname = 'square_arena_cue', model = CUE_MODEL,
    quantity = 0, type = pickups.type.REWARD,
}

-- Builds a bordered GRIDxGRID entity layer with a goal cell and a player-start cell (the
-- default text-level action for 'P', per docs/developers/creating_levels/text_level.md --
-- without a 'P' character, make_map never creates an info_player_start entity at all, and
-- `updateSpawnVars` below -- which only overrides existing spawnVars -- has nothing to act on),
-- plus a variation layer that's uniformly non-default so `themes.lua`'s per-call
-- `randomMap:choice` path fires (the unnamed "default" variation always picks texture index
-- [1], never randomizing -- confirmed by reading themes.lua).
local function buildLayers(goalRow, goalCol, spawnRow, spawnCol)
  local n = GRID + 2  -- +1-cell wall border on each side
  local entityRows, variationRows = {}, {}
  for row = 1, n do
    local eChars, vChars = {}, {}
    for col = 1, n do
      local isWall = (row == 1 or row == n or col == 1 or col == n)
      local isGoal = (not isWall) and (row - 1 == goalRow) and (col - 1 == goalCol)
      local isSpawn = (not isWall) and (row - 1 == spawnRow) and (col - 1 == spawnCol)
      eChars[col] = isWall and '*' or (isGoal and 'G' or (isSpawn and 'P' or ' '))
      vChars[col] = isWall and ' ' or 'A'
    end
    entityRows[row] = table.concat(eChars)
    variationRows[row] = table.concat(vChars)
  end
  return table.concat(entityRows, '\n') .. '\n', table.concat(variationRows, '\n') .. '\n'
end

-- Grid cell (1-indexed, interior-relative) -> world-unit origin string "x y z".
-- 100 world units per cell, matching make_map's own text-level convention; +1 for the border.
local function cellOrigin(row, col, z)
  return string.format('%d %d %d', col * 100 + 50, row * 100 + 50, z or 30)
end

local api = {}

-- True episode boundary (~90s, per timeout.decorate below): the goal/cue positions and the
-- floor/wall texture all change here, together, matching Methods' "changed each episode" list
-- -- and nowhere else. `nextMap()` (which also fires on every goal touch, many times per
-- episode) reads and consumes `_needFullRebuild` to know whether *this* call is that boundary
-- or just an in-episode agent-relocation.
function api:start(episode, seed)
  random:seed(seed)
  randomMap:seed(random:mapGenerationSeed())
  api._mapCount = 0
  api._goalRow = random:uniformInt(1, GRID)
  api._goalCol = random:uniformInt(1, GRID)
  api._cueRow = random:uniformInt(1, GRID)
  api._cueCol = random:uniformInt(1, GRID)
  api._needFullRebuild = true
end

function api:nextMap()
  -- The agent's own spawn point re-randomizes on every call (including every goal touch --
  -- Methods: "Upon reaching the goal the agent was teleported to another random location").
  -- Resample until it doesn't land on the fixed goal cell.
  repeat
    api._spawnRow = random:uniformInt(CENTER_MIN, CENTER_MAX)
    api._spawnCol = random:uniformInt(CENTER_MIN, CENTER_MAX)
  until api._spawnRow ~= api._goalRow or api._spawnCol ~= api._goalCol

  if not api._needFullRebuild then
    -- Fast restart (goal touch mid-episode): same compiled map, same texture, same goal --
    -- only `updateSpawnVars` below repositions the player. No q3map2 recompile. This is the
    -- fix for the throughput collapse: see this file's header note.
    return ''
  end
  api._needFullRebuild = false

  api._mapCount = api._mapCount + 1
  local entityLayer, variationLayer = buildLayers(
      api._goalRow, api._goalCol, api._spawnRow, api._spawnCol)
  -- No `kwargs.theme` passed through: make_map.makeMap builds a fresh theme object internally
  -- on every call when omitted, which is what makes the floor/wall texture re-randomize each
  -- episode.
  return make_map.makeMap{
      mapName = 'square_arena_' .. api._mapCount,
      mapEntityLayer = entityLayer,
      mapVariationsLayer = variationLayer,
      textureSet = texture_sets.MISHMASH,
      useSkybox = true,
  }
end

function api:createPickup(className)
  if className == CUE_PICKUP.classname then
    return CUE_PICKUP
  end
  return pickups.defaults[className]
end

function api:updateSpawnVars(spawnVars)
  if spawnVars.classname == 'info_player_start' then
    spawnVars.origin = cellOrigin(api._spawnRow, api._spawnCol, 40)
    spawnVars.randomAngleRange = '180'
  end
  return spawnVars
end

function api:extraEntities()
  return {
      {classname = CUE_PICKUP.classname, model = CUE_MODEL,
       origin = cellOrigin(api._cueRow, api._cueCol),
       spawnflags = '1'},  -- static (non-bobbing); see pickups.moveType.STATIC convention
  }
end

timeout.decorate(api, EPISODE_LENGTH_SECONDS)
custom_observations.decorate(api)

return api
