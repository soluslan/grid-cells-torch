--[[ Paper's "goal-driven" multi-room maze (Fig. 3a, c; Methods "DeepMind laboratory
environments: goal-driven and goal-doors", main p.6).

Methods says the task is the public DeepMind Lab level `explore_goal_locations` (main p.8:
"Both the goal driven and goal doors tasks are included as part of the latest release, named
explore_goal_locations, and explore_obstructed_goals, respectively"). So this file is a thin
wrapper around the same factory the released `contributed/dmlab30/explore_goal_locations_small`
level uses, with its defaults, which match the Methods text:

  - "Mazes were formed within an 11 x 11 grid" -> mazeHeight = mazeWidth = 11 (outer walls
    included; 100 world units per cell).
  - "procedurally generated at the beginning of each episode; thus, the layout, wall texture,
    landmarks (intra-maze cues on walls) and goal location were different for each episode but
    consistent within an episode" -> a new maze, theme and wall decals per episode.
  - "the agent starts at a random location within the maze ... After getting to the goal the
    agent received a reward of 10 points, after which it was teleported to a new random location
    within the maze" -> the goal pickup gives 10 and ends the map; quickRestart respawns the
    player at a random room cell reachable from the goal without rebuilding the map.
  - "episodes lasted a fixed duration of 5,400 environment steps (90 s)" -> episodeLengthSeconds.
  - Distal cues (buildings at infinity) -> the factory's skybox, as in square_arena.lua.
  - "The goal in both levels was always represented by the same object (Fig. 3c)" -> the goal is
    drawn as the factory's default goal object.

Level setting `goalVisible` (default true): `grid_cells/rl_common.make_dmlab_env` passes it to
every level, and the factory rejects settings it does not know, so it is declared here.
`goalVisible=false` (`--hidden-goal`) draws the goal with a zero-area model, keeping its reward
and collision box, exactly as levels/square_arena.lua does. The paper's goal here is visible.

`require` in an externally loaded level resolves against lab/game_scripts, so this file cannot
share code with the other files in levels/; the invisible-goal block is copied from
square_arena.lua.
]]

local factory = require 'factories.explore.goal_locations_factory'
local tensor = require 'dmlab.system.tensor'

local opts = {goalVisible = true}
local api = factory.createLevelApi(opts)

local INVISIBLE_MODEL = 'invisible_goal_model'

local createPickup = api.createPickup
function api:createPickup(classname)
  local pickup = createPickup(self, classname)
  if classname == 'goal' and pickup and not opts.goalVisible then
    pickup = {name = pickup.name, classname = pickup.classname, model = INVISIBLE_MODEL,
              quantity = pickup.quantity, type = pickup.type}
  end
  return pickup
end

local createModel = api.createModel
function api:createModel(modelName)
  if modelName == INVISIBLE_MODEL then
    return {
        surfaces = {
            empty = {
                vertices = tensor.FloatTensor{
                    {0, 0, -1000, 0, 0, 1, 0, 0},
                    {0, 0, -1000, 0, 0, 1, 0, 0},
                    {0, 0, -1000, 0, 0, 1, 0, 0},
                },
                indices = tensor.Int32Tensor{{1, 2, 3}},
                shaderName = 'textures/model/beam_d',
            },
        },
    }
  end
  return createModel and createModel(self, modelName)
end

return api
