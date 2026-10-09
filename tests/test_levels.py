import os

import numpy as np
import pytest

from config import Config

LEVELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "levels")
PLAYER_RADIUS = 16


@pytest.mark.dmlab
def test_tiny_goal_spawns_clear_of_walls():
    from grid_cells.rl_common import make_dmlab_env
    cfg = Config()
    cfg.rl.env.level_directory = LEVELS
    env = make_dmlab_env(cfg.rl.env, renderer="software", level="tiny_goal")
    try:
        for seed in range(30):
            env.reset(seed=seed)
            x, y = env.observations()["DEBUG.POS.TRANS"][:2]
            for v in (x, y):
                assert 100 + PLAYER_RADIUS < v < 600 - PLAYER_RADIUS, (seed, x, y)
    finally:
        env.close()
