--[[ Learning-loop sanity gate, not a paper task. A 5x5-cell room (1.25 m) with a visible goal
object; the goal cell is re-drawn each episode, and every goal touch teleports the agent to a
random other cell, as in square_arena.lua. No intra-maze cue and a fixed texture set, so the only
thing to learn is "walk to the visible object". An agent whose training loop is wired correctly
should raise its episode reward well above the random-walk level within ~1e7 frames; if it
cannot, the bug is in the learner, not in the difficulty of the paper's task.

Same spawn/teleport mechanics as square_arena.lua (fast restart on goal touches, full rebuild
only at episode start), so the agent-facing interface is identical.
]]

local custom_observations = require 'decorators.custom_observations'
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local random = require 'common.random'
local timeout = require 'decorators.timeout'
local map_maker = require 'dmlab.system.map_maker'
local randomMap = random(map_maker:randomGen())

local GRID = 5
local EPISODE_LENGTH_SECONDS = 90

local function buildLayer(goalRow, goalCol, spawnRow, spawnCol)
  local n = GRID + 2
  local rows = {}
  for row = 1, n do
    local chars = {}
    for col = 1, n do
      local isWall = (row == 1 or row == n or col == 1 or col == n)
      local isGoal = (not isWall) and (row - 1 == goalRow) and (col - 1 == goalCol)
      local isSpawn = (not isWall) and (row - 1 == spawnRow) and (col - 1 == spawnCol)
      chars[col] = isWall and '*' or (isGoal and 'G' or (isSpawn and 'P' or ' '))
    end
    rows[row] = table.concat(chars)
  end
  return table.concat(rows, '\n') .. '\n'
end

local api = {}

function api:start(episode, seed)
  random:seed(seed)
  randomMap:seed(random:mapGenerationSeed())
  api._mapCount = 0
  api._goalRow = random:uniformInt(1, GRID)
  api._goalCol = random:uniformInt(1, GRID)
  api._needFullRebuild = true
end

function api:nextMap()
  repeat
    api._spawnRow = random:uniformInt(1, GRID)
    api._spawnCol = random:uniformInt(1, GRID)
  until api._spawnRow ~= api._goalRow or api._spawnCol ~= api._goalCol

  if not api._needFullRebuild then
    return ''
  end
  api._needFullRebuild = false
  api._mapCount = api._mapCount + 1
  return make_map.makeMap{
      mapName = 'tiny_goal_' .. api._mapCount,
      mapEntityLayer = buildLayer(api._goalRow, api._goalCol, api._spawnRow, api._spawnCol),
      useSkybox = true,
  }
end

function api:createPickup(className)
  return pickups.defaults[className]
end

function api:updateSpawnVars(spawnVars)
  if spawnVars.classname == 'info_player_start' then
    local x = api._spawnCol * 100 + random:uniformReal(0, 100)
    local y = api._spawnRow * 100 + random:uniformReal(0, 100)
    spawnVars.origin = string.format('%d %d %d', x, y, 40)
    spawnVars.randomAngleRange = '180'
  end
  return spawnVars
end

timeout.decorate(api, EPISODE_LENGTH_SECONDS)
custom_observations.decorate(api)

return api
