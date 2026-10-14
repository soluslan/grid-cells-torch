"""Batched environments for the synchronous learner.

Each env auto-resets when its episode ends, so `step` always returns a valid observation for
every env: for an env whose episode just ended it is the first observation of the next episode.
Episode seeds are deterministic and never collide across envs: env i's k-th episode uses
`base_seed + i + num_envs * k`.
"""
import multiprocessing as mp

import numpy as np

from grid_cells.actor_critic import DISCRETE_ACTIONS
from grid_cells.rl_common import convert_observation


class _EnvRunner:
    def __init__(self, factory, env_cfg, index: int, num_envs: int, base_seed: int):
        self.env = factory()
        self.env_cfg = env_cfg
        self.index = index
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.episode = 0

    def _seed(self) -> int:
        return self.base_seed + self.index + self.num_envs * self.episode

    def reset(self) -> dict:
        self.env.reset(seed=self._seed())
        return convert_observation(self.env.observations(), self.env_cfg)

    def step(self, action_index: int):
        reward = float(self.env.step(DISCRETE_ACTIONS[action_index],
                                     num_steps=self.env_cfg.action_repeat))
        ended = not self.env.is_running()
        if ended:
            self.episode += 1
            self.env.reset(seed=self._seed())
        return reward, ended, convert_observation(self.env.observations(), self.env_cfg)

    def close(self):
        self.env.close()


def _stack(obs_list) -> dict:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


class SyncVecEnv:
    """All envs in this process. For tests and debugging."""

    def __init__(self, factories, env_cfg, base_seed: int = 0):
        n = len(factories)
        self.runners = [_EnvRunner(f, env_cfg, i, n, base_seed) for i, f in enumerate(factories)]
        self.num_envs = n

    def reset(self) -> dict:
        return _stack([r.reset() for r in self.runners])

    def step(self, actions):
        results = [r.step(int(a)) for r, a in zip(self.runners, actions)]
        rewards = np.array([res[0] for res in results], dtype=np.float32)
        ended = np.array([res[1] for res in results], dtype=bool)
        return _stack([res[2] for res in results]), rewards, ended

    def close(self):
        for r in self.runners:
            r.close()


def _worker(conn, factory, env_cfg, index, num_envs, base_seed):
    import torch
    torch.set_num_threads(1)
    runner = _EnvRunner(factory, env_cfg, index, num_envs, base_seed)
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                conn.send(runner.step(arg))
            elif cmd == "reset":
                conn.send(runner.reset())
            elif cmd == "close":
                runner.close()
                conn.send(None)
                return
    except (EOFError, KeyboardInterrupt):
        runner.close()


class SubprocVecEnv:
    """One subprocess per env, stepped in parallel."""

    def __init__(self, factories, env_cfg, base_seed: int = 0, start_method: str = "spawn"):
        ctx = mp.get_context(start_method)
        n = len(factories)
        self.num_envs = n
        self.conns, self.procs = [], []
        for i, factory in enumerate(factories):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, factory, env_cfg, i, n, base_seed),
                            daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def reset(self) -> dict:
        for c in self.conns:
            c.send(("reset", None))
        return _stack([c.recv() for c in self.conns])

    def step(self, actions):
        for c, a in zip(self.conns, actions):
            c.send(("step", int(a)))
        results = [c.recv() for c in self.conns]
        rewards = np.array([res[0] for res in results], dtype=np.float32)
        ended = np.array([res[1] for res in results], dtype=bool)
        return _stack([res[2] for res in results]), rewards, ended

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=10)
