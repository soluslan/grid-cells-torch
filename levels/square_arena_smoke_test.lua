--[[ M1 smoke test: does `deepmind_lab.Lab(config={'levelDirectory': ...})` actually load a
level file living outside the `lab/` git submodule, while still `require`-ing the submodule's
own Lua library code unmodified? (See the RL-agent roadmap's "Engineering design" note --
this was inferred from reading the C++ engine source, not yet runtime-verified.)

Deliberately minimal -- a single open room with one fixed goal, modelled directly on
`lab/game_scripts/levels/tests/empty_room_test.lua`. No intra-maze cue, no per-episode texture
randomization, no distal cues, no central-6x6 spawn constraint yet: those are the paper's
square-arena (Fig. 2) requirements and belong in a later, separate level file once this
smoke test confirms the plumbing works at all.
]]

local make_map = require 'common.make_map'
local pickups = require 'common.pickups'
local custom_observations = require 'decorators.custom_observations'
local timeout = require 'decorators.timeout'

-- 10x10 open interior + 1-cell wall border, one goal near the far corner from spawn.
local MAP_ENTITIES = [[
************
*P         *
*          *
*          *
*          *
*          *
*          *
*          *
*          *
*          *
*         G*
************
]]

local api = {}

function api:init(params)
  make_map.seedRng(1)
  api._map = make_map.makeMap{
      mapName = 'square_arena_smoke_test',
      mapEntityLayer = MAP_ENTITIES,
      useSkybox = true,
  }
end

function api:nextMap()
  return api._map
end

function api:createPickup(className)
  return pickups.defaults[className]
end

timeout.decorate(api, 90)
custom_observations.decorate(api)

return api
