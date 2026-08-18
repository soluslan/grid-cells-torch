"""M1 smoke test: does `levelDirectory` load a level living outside the `lab/` submodule,
while it still `require`s the submodule's own Lua library code (`common.make_map`,
`decorators.custom_observations`, etc.) unmodified? Random-agent loop pattern from
`lab/python/random_agent.py`'s `DiscretizedRandomAgent`.

Run from grid-cells-torch/: python rl/smoke_test.py
"""

import os

import numpy as np
import deepmind_lab

LEVEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels")


def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTIONS = [
    _action(0, 0, 0, 1, 0, 0, 0),   # forward
    _action(0, 0, 0, -1, 0, 0, 0),  # backward
    _action(-20, 0, 0, 0, 0, 0, 0),  # look left
    _action(20, 0, 0, 0, 0, 0, 0),   # look right
]


def run(n_steps: int = 200) -> None:
    env = deepmind_lab.Lab(
        "square_arena_smoke_test",
        ["RGB_INTERLEAVED", "VEL.TRANS", "VEL.ROT", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
        config={"width": "64", "height": "64", "levelDirectory": LEVEL_DIR},
    )
    env.reset()
    print("levelDirectory + external require: OK (env constructed)")

    total_reward = 0.0
    for i in range(n_steps):
        if not env.is_running():
            print(f"episode ended early at step {i}, resetting")
            env.reset()
        action = ACTIONS[np.random.randint(len(ACTIONS))]
        reward = env.step(action, num_steps=4)
        total_reward += reward
        if i % 50 == 0:
            obs = env.observations()
            print(f"step {i}: RGB shape={obs['RGB_INTERLEAVED'].shape}, "
                  f"VEL.TRANS={obs['VEL.TRANS']}, VEL.ROT={obs['VEL.ROT']}, "
                  f"POS.TRANS={obs['DEBUG.POS.TRANS']}, POS.ROT={obs['DEBUG.POS.ROT']}")

    print(f"Finished {n_steps} steps, total reward {total_reward}")


if __name__ == "__main__":
    run()
