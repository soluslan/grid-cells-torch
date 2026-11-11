--[[ Paper's "goal-doors" multi-room maze with stochastic doors (Fig. 3b, d; Methods
"DeepMind laboratory environments: goal-driven and goal-doors", main p.6).

Methods says the task is the public DeepMind Lab level `explore_obstructed_goals` (main p.8), so
this file is a thin wrapper around the same factory the released
`contributed/dmlab30/explore_obstructed_goals_small` level uses. Everything in goal_driven.lua's
header applies here too (11 x 11 procedural maze, visible goal worth 10, teleport on reaching
it, 90 s episodes, skybox). The difference, in the paper's words: "the state of the doors (open
or closed) changed randomly during an episode each time the agent reached the goal". The
factory does exactly that on every restart (episode start and every goal touch): it closes all
doors, opens the doors on one random path from the new spawn point to the goal, then opens a
further (1 - doorsClosed) fraction of the remaining doors. Closed doors stay shut (their
triggers are removed).

doorsClosed = 0.5 is the released level's value; the paper gives no fraction
(docs/paper/deviations.md, row J).

`goalVisible` and the invisible-goal block: see goal_driven.lua.
]]

local factory = require 'factories.explore.obstructed_goals_factory'
local tensor = require 'dmlab.system.tensor'

local opts = {goalVisible = true, doorsClosed = 0.5}
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
