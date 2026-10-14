"""Scripted fake DeepMind Lab environment plus spy modules, used to check how an RL training loop
wires observations, rewards and recurrent state, independent of what the networks learn.

Every observation's image is filled with a unique id (0-249), so a spy can tell exactly which
observation a call saw. The spy critic returns that id as its value, which makes "which state
did the bootstrap come from" directly readable.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from grid_cells.actor_critic import ActorCriticOutput, N_ACTIONS

GOAL_REWARD = 10.0


def image_id(image: torch.Tensor) -> list[int]:
    """Recover ids from a (B, 3, H, W) image batch in the agent's [-1, 1] convention."""
    return [int(round((float(v) + 1.0) * 127.5)) for v in image[:, 0, 0, 0]]


@dataclass
class ObsInfo:
    episode: int
    t: int
    after_goal: bool
    first_of_episode: bool
    env: int = 0


class ScriptedEnv:
    """Lab-like env: episodes of `episode_len` actions, +10 and a teleport at `goal_steps`."""

    def __init__(self, episode_len: int, goal_steps, registry: dict, id_offset: int = 0,
                 env_index: int = 0):
        self.env_index = env_index
        self.episode_len = episode_len
        self.goal_steps = set(goal_steps)
        self.registry = registry
        self.next_id = id_offset
        self.episode = -1
        self.t = 0
        self.x = 600.0
        self._new_obs(after_goal=False, first=True)

    def _new_obs(self, after_goal: bool, first: bool):
        self.obs_id = self.next_id
        self.next_id += 1
        assert self.obs_id < 250, "scripted run too long for unique image ids"
        self.registry[self.obs_id] = ObsInfo(self.episode, self.t, after_goal, first,
                                             self.env_index)

    def reset(self, seed=None):
        self.episode += 1
        self.t = 0
        self.x = 600.0
        self._new_obs(after_goal=False, first=True)

    def observations(self):
        img = np.full((64, 64, 3), self.obs_id, dtype=np.uint8)
        return {
            "RGB_INTERLEAVED": img,
            "DEBUG.POS.TRANS": np.array([self.x, 600.0, 0.0]),
            "DEBUG.POS.ROT": np.array([0.0, 0.0, 0.0]),
            "VEL.TRANS": np.zeros(3),
            "VEL.ROT": np.zeros(3),
        }

    def step(self, action, num_steps=1):
        reward = GOAL_REWARD if self.t in self.goal_steps else 0.0
        self.t += 1
        self.x += 10.0
        if reward > 0:
            self.x = 200.0 if self.x > 600 else 1000.0
        if self.t < self.episode_len:
            self._new_obs(after_goal=reward > 0, first=False)
        return reward

    def is_running(self):
        return self.t < self.episode_len

    def close(self):
        pass


@dataclass
class Call:
    obs_id: int
    env: int
    reward_in: float
    prev_action_in: int
    goal_in: torch.Tensor
    ac_h_in: torch.Tensor
    ac_h_out: torch.Tensor


@dataclass
class GridCall:
    obs_id: int
    env: int
    h_in: torch.Tensor
    h_out: torch.Tensor


@dataclass
class Window:
    """One env's slice of one update: the value ids it trained on, where the return was cut
    (bootstrap 0 after that step), and the bootstrap value used after the last step."""
    env: int
    value_ids: list
    cut: list
    bootstrap: float


@dataclass
class Trace:
    registry: dict
    calls: list = field(default_factory=list)
    grid_calls: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    sequences: list = field(default_factory=list)


class SpyActorCritic(nn.Module):
    def __init__(self, trace: Trace, lstm_units: int = 4):
        super().__init__()
        self.trace = trace
        self.lstm_units = lstm_units
        self.w = nn.Parameter(torch.zeros(N_ACTIONS))

    def init_hidden(self, batch_size, device=None):
        z = torch.zeros(1, batch_size, self.lstm_units, device=device)
        return z, z.clone()

    recording = True

    def step(self, image, reward, prev_action, grid_code, goal_grid_code, hidden):
        ids = image_id(image)
        h, c = hidden
        out = (h + 1.0, c + 1.0)
        for b, oid in enumerate(ids if self.recording else []):
            self.trace.calls.append(Call(
                oid, b, float(reward[b, 0]), int(prev_action[b]),
                goal_grid_code[b].detach().clone(), h[0, b].detach().clone(),
                out[0][0, b].detach().clone()))
        value = torch.tensor(ids, dtype=torch.float32, device=image.device) + 0.0 * self.w.sum()
        logits = self.w.expand(len(ids), N_ACTIONS)
        return ActorCriticOutput(action_logits=logits, value=value, hidden=out)


class SpyGrid(nn.Module):
    def __init__(self, trace: Trace, nh_lstm: int = 4, nh_bottleneck: int = 256):
        super().__init__()
        self.trace = trace
        self.nh_lstm = nh_lstm
        self.nh_bottleneck = nh_bottleneck
        self.current_ids = None

    def init_hidden_from_predictions(self, predictions):
        b = predictions[0].shape[0]
        marker = sum(float(p.abs().sum()) for p in predictions)
        z = torch.full((1, b, self.nh_lstm), marker)
        return z, z.clone()

    recording = True

    def step(self, vel_t, vision_t, hidden):
        h, c = hidden
        out = (h + 1.0, c + 1.0)
        ids = self.current_ids or [None] * h.shape[1]
        for b in range(h.shape[1] if self.recording else 0):
            self.trace.grid_calls.append(GridCall(ids[b], b, h[0, b].clone(), out[0][0, b].clone()))
        code = out[0][0].mean(-1, keepdim=True).expand(-1, self.nh_bottleneck) + 1.0
        return code, out


class IdTrackingVision(nn.Module):
    """Wraps the real vision module and tells the spy grid which observation it is on."""

    def __init__(self, vision, spy_grid):
        super().__init__()
        self.vision = vision
        self.spy_grid = spy_grid

    def forward(self, images):
        self.spy_grid.current_ids = image_id(images)
        return self.vision(images)

    def loss(self, *args, **kwargs):
        return self.vision.loss(*args, **kwargs)


def run_a3c(episode_len=30, goal_steps=(5, 17), n_episodes=2, n_step=8, seq_len=10) -> Trace:
    import torch.multiprocessing as mp

    from config import Config
    from grid_cells import rl_a3c
    from grid_cells.ensembles import build_rl_ensembles
    from grid_cells.replay_buffer import FrameReplayBuffer, SequenceReplayBuffer
    from grid_cells.vision import VisionModule

    registry = {}
    trace = Trace(registry)
    cfg = Config()
    cfg.rl.actor_critic.n_step = n_step
    cfg.rl.grid.seq_len = seq_len
    cfg.rl.total_env_steps = episode_len * n_episodes * cfg.rl.env.action_repeat
    cfg.rl.results_dir = "/tmp/rl_harness_a3c"
    torch.manual_seed(0)

    place, hd = build_rl_ensembles(cfg, torch.device("cpu"))
    spy_grid = SpyGrid(trace)
    vision = IdTrackingVision(VisionModule(place[0], hd[0]), spy_grid)
    ac = SpyActorCritic(trace)
    opt = rl_a3c.SharedRMSprop(ac.parameters(), lr=1e-4)
    frames = FrameReplayBuffer(capacity=500)
    seqs = SequenceReplayBuffer(capacity=50, seq_len=seq_len, ego_vel_dim=4)

    original = rl_a3c._apply_a3c_update

    def recording_update(actor_critic, optimizer, log_probs, values, entropies, rewards,
                         bootstrap_value, cfg):
        ids = [int(round(float(v))) for v in torch.cat(values).detach()]
        last = registry[ids[-1]]
        ended_or_goal = bootstrap_value == 0.0 and len(ids) < n_step
        cut = [False] * (len(ids) - 1) + [ended_or_goal]
        trace.windows.append(Window(0, ids, cut, float(bootstrap_value)))
        return original(actor_critic, optimizer, log_probs, values, entropies, rewards,
                        bootstrap_value, cfg)

    rl_a3c._apply_a3c_update = recording_update
    try:
        rl_a3c.actor_worker(
            0, cfg, vision, spy_grid, ac, opt, frames, seqs, place[0], hd[0],
            mp.Value("l", 0), mp.Value("b", False),
            env_factory=lambda: ScriptedEnv(episode_len, goal_steps, registry))
    finally:
        rl_a3c._apply_a3c_update = original

    images = seqs._data["image"][:len(seqs)]
    for seq in images:
        trace.sequences.append(image_id(seq))
    return trace


def run_a2c(episode_len=30, goal_steps=(5, 17), n_updates=8, n_step=8, seq_len=10,
            num_envs=2, a2c_overrides: dict | None = None) -> Trace:
    from config import Config
    from grid_cells import rl_a2c
    from grid_cells.ensembles import build_rl_ensembles
    from grid_cells.replay_buffer import TimeMajorReplay
    from grid_cells.vec_env import SyncVecEnv
    from grid_cells.vision import VisionModule

    registry = {}
    trace = Trace(registry)
    cfg = Config()
    cfg.rl.actor_critic.n_step = n_step
    cfg.rl.grid.seq_len = seq_len
    cfg.rl.a2c.vision_updates_per_rollout = 0
    cfg.rl.a2c.grid_updates_per_rollout = 0
    for key, value in (a2c_overrides or {}).items():
        setattr(cfg.rl.a2c, key, value)
    torch.manual_seed(0)
    device = torch.device("cpu")

    place, hd = build_rl_ensembles(cfg, device)
    spy_grid = SpyGrid(trace)
    vision = IdTrackingVision(VisionModule(place[0], hd[0]), spy_grid)
    ac = SpyActorCritic(trace)
    agent = rl_a2c.Agent(cfg, vision, spy_grid, ac, device)
    original_act = agent.act

    def act(state, obs, ego_vel):
        spy_grid.recording = ac.recording = torch.is_grad_enabled()
        return original_act(state, obs, ego_vel)
    agent.act = act

    factories = [lambda i=i: ScriptedEnv(episode_len, goal_steps, registry, id_offset=125 * i,
                                                 env_index=i)
                 for i in range(num_envs)]
    vec_env = SyncVecEnv(factories, cfg.rl.env)
    replay = TimeMajorReplay(1000, num_envs)
    learner = rl_a2c.A2CLearner(cfg, vec_env, agent, torch.optim.RMSprop(ac.parameters()),
                                None, None, place, hd, replay)
    original_update = learner.update_actor_critic

    def recording_update(rollout):
        values = rollout["values"].detach()
        for b in range(values.shape[1]):
            trace.windows.append(Window(
                b, [int(round(float(v))) for v in values[:, b]],
                [bool(c) for c in rollout["cuts"][:, b]], float(rollout["bootstrap"][b])))
        return original_update(rollout)
    learner.update_actor_critic = recording_update

    for _ in range(n_updates):
        learner.train_iteration()

    for _ in range(20):
        batch = replay.sample_sequences(4, seq_len)
        for seq in batch["image"]:
            trace.sequences.append([int(v) for v in seq[:, 0, 0, 0]])
    return trace
