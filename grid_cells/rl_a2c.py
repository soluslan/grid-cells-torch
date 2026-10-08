"""Synchronous batched advantage actor-critic (A2C) for the grid cell agent.

One process owns every network on one device. `num_envs` DeepMind Lab instances step in
parallel (vec_env), the policy acts on all of them in one batch, and after every
`n_step` actions per env the actor-critic takes one gradient step on the whole
(n_step x num_envs) rollout; the vision module and grid network then take a fixed number of
supervised steps on replayed experience.

Semantics follow the paper where it is explicit (docs/paper/spec-rl-agent.md) and standard A2C
where it is silent (docs/paper/deviations.md, RL table): the grid LSTM, policy LSTM, goal grid
code, previous reward and previous action are zeroed at each episode start; a goal arrival is
not an episode end (the agent is teleported and the return is bootstrapped through it); the
n-step return bootstraps from V(s_{t+n}).
"""
import csv
import os
import signal
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from grid_cells.actor_critic import ActorCriticLSTM
from grid_cells.ensembles import build_rl_ensembles, encode_targets, soft_cross_entropy
from grid_cells.replay_buffer import TimeMajorReplay
from grid_cells.rl_common import DMLabFactory, ego_velocity, image_to_input
from grid_cells.training import build_model, build_optimizer
from grid_cells.vec_env import SubprocVecEnv, SyncVecEnv
from grid_cells.vision import VisionModule


@dataclass
class AgentState:
    grid_hidden: tuple
    ac_hidden: tuple
    prev_action: torch.Tensor
    prev_reward: torch.Tensor
    goal_code: torch.Tensor

    @classmethod
    def zeros(cls, num_envs, grid_units, ac_units, code_dim, device):
        def z(units):
            return (torch.zeros(1, num_envs, units, device=device),
                    torch.zeros(1, num_envs, units, device=device))
        return cls(z(grid_units), z(ac_units),
                   torch.zeros(num_envs, dtype=torch.long, device=device),
                   torch.zeros(num_envs, 1, device=device),
                   torch.zeros(num_envs, code_dim, device=device))

    def detached(self) -> "AgentState":
        return AgentState(tuple(h.detach() for h in self.grid_hidden),
                          tuple(h.detach() for h in self.ac_hidden),
                          self.prev_action, self.prev_reward, self.goal_code.detach())


def _mask_hidden(hidden, keep):
    k = keep.view(1, -1, 1)
    return tuple(h * k for h in hidden)


def discounted_returns(rewards: torch.Tensor, cuts: torch.Tensor, bootstrap: torch.Tensor,
                       gamma: float) -> torch.Tensor:
    """rewards, cuts: (T, B); cuts[t] means nothing after step t belongs to this return (an
    episode ended). bootstrap: (B,) value of the state after the last step."""
    returns = torch.zeros_like(rewards)
    running = bootstrap
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running * (1.0 - cuts[t].float())
        returns[t] = running
    return returns


def actor_critic_loss(log_probs, values, entropies, returns, ac_cfg):
    advantage = returns - values.detach()
    policy_loss = -(log_probs * advantage).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = entropies.mean()
    alpha = sum(ac_cfg.baseline_cost_range) / 2
    beta = sum(ac_cfg.entropy_reg_range) / 2
    loss = policy_loss + alpha * value_loss - beta * entropy
    return loss, {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                  "entropy": entropy.item(), "mean_value": values.mean().item(),
                  "adv_std": advantage.std().item() if advantage.numel() > 1 else 0.0}


class Agent:
    """The three networks plus the per-env recurrent state, acting on a batch of envs."""

    def __init__(self, cfg, vision_module, grid_network, actor_critic, device):
        self.cfg = cfg
        self.vision = vision_module
        self.grid = grid_network
        self.ac = actor_critic
        self.device = device

    def initial_state(self, num_envs: int) -> AgentState:
        ac_units = self.ac.init_hidden(1)[0].shape[-1]
        return AgentState.zeros(num_envs, self.grid.nh_lstm, ac_units,
                                self.grid.nh_bottleneck, self.device)

    def ego_vel(self, obs) -> torch.Tensor:
        env_cfg = self.cfg.rl.env
        v = ego_velocity(obs["vel_trans"], obs["vel_rot"], env_cfg)
        return v + torch.randn_like(v) * env_cfg.velocity_noise_std

    def act(self, state: AgentState, obs, ego_vel):
        image = image_to_input(obs["image"])
        with torch.no_grad():
            vis = self.vision(image)
            vision_t = torch.cat([vis.place_probs, vis.hd_probs], dim=-1)
            grid_code, grid_hidden = self.grid.step(ego_vel, vision_t, state.grid_hidden)
        out = self.ac.step(image, state.prev_reward, state.prev_action, grid_code.detach(),
                           state.goal_code, state.ac_hidden)
        return out, grid_code, grid_hidden


def obs_to_device(obs: dict, device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}


class A2CLearner:
    def __init__(self, cfg, vec_env, agent: Agent, ac_optimizer, vision_optimizer,
                 grid_optimizer, place_ensembles, hd_ensembles, replay: TimeMajorReplay,
                 log_dir: str | None = None, seed: int = 0):
        self.cfg = cfg
        self.env = vec_env
        self.agent = agent
        self.ac_opt = ac_optimizer
        self.vision_opt = vision_optimizer
        self.grid_opt = grid_optimizer
        self.place_ens = place_ensembles
        self.hd_ens = hd_ensembles
        self.replay = replay
        self.log_dir = log_dir
        self.device = agent.device
        self.frames = 0
        self.updates = 0
        n = vec_env.num_envs
        self.state = agent.initial_state(n)
        self.obs_np = vec_env.reset()
        self.obs = obs_to_device(self.obs_np, self.device)
        self.episode_ids = np.arange(n, dtype=np.int64)
        self.next_episode_id = n
        self.episode_reward = np.zeros(n)
        self.episode_goals = np.zeros(n, dtype=np.int64)
        self.first_goal_frame = np.full(n, -1)
        self.episode_frames = np.zeros(n, dtype=np.int64)
        self.torch_gen = torch.Generator(device="cpu").manual_seed(seed)
        self.window_log = None

    def _log(self, name, row, header):
        if self.log_dir is None:
            return
        path = os.path.join(self.log_dir, name)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)

    def rollout(self):
        a2c = self.cfg.rl.a2c
        n_step = self.cfg.rl.actor_critic.n_step
        repeat = self.cfg.rl.env.action_repeat
        state = self.state.detached()
        log_probs, values, entropies, rewards, cuts = [], [], [], [], []
        still, moves = 0, 0
        for _ in range(n_step):
            obs = self.obs
            ego_vel = self.agent.ego_vel(obs)
            out, grid_code, grid_hidden = self.agent.act(state, obs, ego_vel)
            dist = torch.distributions.Categorical(logits=out.action_logits)
            action = dist.sample()
            next_np, reward_np, ended_np = self.env.step(action.cpu().numpy())
            self.replay.add(self.obs_np["image"], self.obs_np["pos"], self.obs_np["hd"],
                            ego_vel.cpu(), torch.from_numpy(self.episode_ids))

            reward = torch.as_tensor(reward_np, device=self.device)
            ended = torch.as_tensor(ended_np, device=self.device)
            goal = (reward > 0) & ~ended
            log_probs.append(dist.log_prob(action))
            values.append(out.value)
            entropies.append(dist.entropy())
            rewards.append(reward)
            cuts.append(ended | goal if a2c.cut_return_on_goal else ended)

            next_obs = obs_to_device(next_np, self.device)
            moved = (next_obs["pos"] - obs["pos"]).norm(dim=-1)
            valid = ~ended & ~(reward > 0)
            still += int(((moved < 0.01) & valid).sum())
            moves += int(valid.sum())

            reached = reward > 0
            goal_code = torch.where(reached.unsqueeze(-1), grid_code, state.goal_code)
            ac_hidden, grid_hidden_next = out.hidden, grid_hidden
            if a2c.reset_policy_on_goal:
                ac_hidden = _mask_hidden(ac_hidden, (~goal).float())
            if a2c.reinit_grid_on_goal:
                grid_hidden_next = _mask_hidden(grid_hidden_next, (~goal).float())
            keep = (~ended).float()
            state = AgentState(
                _mask_hidden(grid_hidden_next, keep), _mask_hidden(ac_hidden, keep),
                action * (~ended).long(), (reward * keep).unsqueeze(-1),
                goal_code * keep.unsqueeze(-1))
            self._track_episodes(reward_np, ended_np, repeat)
            self.obs, self.obs_np = next_obs, next_np
            self.frames += self.env.num_envs * repeat

        with torch.no_grad():
            boot_out, _, _ = self.agent.act(state, self.obs, self.agent.ego_vel(self.obs))
        self.state = state
        rollout = {
            "log_probs": torch.stack(log_probs), "values": torch.stack(values),
            "entropies": torch.stack(entropies), "rewards": torch.stack(rewards),
            "cuts": torch.stack(cuts), "bootstrap": boot_out.value.detach(),
            "still_frac": still / max(moves, 1),
        }
        return rollout

    def _track_episodes(self, reward_np, ended_np, repeat):
        self.episode_frames += repeat
        self.episode_reward += reward_np
        hit = reward_np > 0
        self.episode_goals += hit
        first = hit & (self.first_goal_frame < 0)
        self.first_goal_frame[first] = self.episode_frames[first]
        for i in np.flatnonzero(ended_np):
            self._log("episodes.csv",
                      [self.frames + repeat * self.env.num_envs, i, self.episode_reward[i],
                       self.episode_goals[i], self.first_goal_frame[i]],
                      ["frames", "env", "reward", "goals", "first_goal_frame"])
            self.episode_ids[i] = self.next_episode_id
            self.next_episode_id += 1
            self.episode_reward[i] = 0.0
            self.episode_goals[i] = 0
            self.first_goal_frame[i] = -1
            self.episode_frames[i] = 0

    def update_actor_critic(self, rollout) -> dict:
        ac_cfg = self.cfg.rl.actor_critic
        returns = discounted_returns(rollout["rewards"], rollout["cuts"], rollout["bootstrap"],
                                     ac_cfg.discount)
        loss, stats = actor_critic_loss(rollout["log_probs"], rollout["values"],
                                        rollout["entropies"], returns, ac_cfg)
        self.ac_opt.zero_grad()
        loss.backward()
        if self.cfg.rl.a2c.ac_grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.agent.ac.parameters(),
                                           self.cfg.rl.a2c.ac_grad_clip_norm)
        self.ac_opt.step()
        return stats

    def update_vision(self) -> float | None:
        batch_size = self.cfg.rl.vision.minibatch_size
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample_frames(batch_size)
        image = image_to_input(batch["image"].to(self.device))
        pos, hd = batch["pos"].to(self.device), batch["hd"].to(self.device)
        place_t = self.place_ens[0].posterior(pos.unsqueeze(1)).squeeze(1)
        hd_t = self.hd_ens[0].posterior(hd.unsqueeze(1)).squeeze(1)
        loss = self.agent.vision.loss(image, place_t, hd_t)
        self.vision_opt.zero_grad()
        loss.backward()
        self.vision_opt.step()
        return loss.item()

    def update_grid(self) -> float | None:
        grid_cfg = self.cfg.rl.grid
        batch = self.replay.sample_sequences(grid_cfg.minibatch_size, grid_cfg.seq_len)
        if batch is None:
            return None
        grid = self.agent.grid
        b, t = batch["image"].shape[:2]
        images = image_to_input(batch["image"].to(self.device)).reshape(b * t, 3, 64, 64)
        with torch.no_grad():
            vis = self.agent.vision(images)
            vision_seq = torch.cat([vis.place_probs, vis.hd_probs], -1).reshape(b, t, -1)
        pos, hd = batch["pos"].to(self.device), batch["hd"].to(self.device)
        vels = batch["ego_vel"].to(self.device)
        if self.cfg.rl.a2c.grid_train_zero_init:
            zero = torch.zeros(1, b, grid.nh_lstm, device=self.device)
            out = grid([], vels, vision_seq=vision_seq, initial_state=(zero, zero.clone()))
        else:
            from grid_cells.ensembles import encode_initial_conditions
            init = encode_initial_conditions(pos[:, 0], hd[:, 0], self.place_ens, self.hd_ens)
            out = grid(init, vels, vision_seq=vision_seq)
        targets = encode_targets(pos, hd, self.place_ens, self.hd_ens)
        loss = sum(soft_cross_entropy(lg, tg) for lg, tg in zip(out.logits, targets)).mean()
        self.grid_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(grid.clip_parameters(self.cfg.train.grad_clip_scope),
                                        self.cfg.train.grad_clip_value)
        self.grid_opt.step()
        return loss.item()

    def train_iteration(self) -> dict:
        start = time.time()
        rollout = self.rollout()
        stats = self.update_actor_critic(rollout)
        a2c = self.cfg.rl.a2c
        v = [self.update_vision() for _ in range(a2c.vision_updates_per_rollout)]
        g = [self.update_grid() for _ in range(a2c.grid_updates_per_rollout)]
        self.updates += 1
        v = [x for x in v if x is not None]
        g = [x for x in g if x is not None]
        stats.update(frames=self.frames, update=self.updates, still_frac=rollout["still_frac"],
                     vision_loss=float(np.mean(v)) if v else float("nan"),
                     grid_loss=float(np.mean(g)) if g else float("nan"),
                     fps=self.env.num_envs * self.cfg.rl.actor_critic.n_step
                     * self.cfg.rl.env.action_repeat / (time.time() - start))
        keys = ["frames", "update", "policy_loss", "value_loss", "entropy", "mean_value",
                "adv_std", "still_frac", "vision_loss", "grid_loss", "fps"]
        self._log("ac_diag.csv", [stats[k] for k in keys], keys)
        return stats


def build_agent(cfg, device, checkpoint: dict | None = None, reset_actor_critic: bool = False):
    place, hd = build_rl_ensembles(cfg, device)
    grid = build_model(cfg, place + hd, device, vision_dim=place[0].n_cells + hd[0].n_cells,
                       ego_vel_dim=4)
    vision = VisionModule(place[0], hd[0], embed_dim=cfg.rl.vision.embed_dim,
                          mask_prob=cfg.rl.vision.mask_prob,
                          mask_mode=cfg.rl.a2c.vision_mask_mode).to(device)
    ac = ActorCriticLSTM(grid_code_dim=cfg.model.nh_bottleneck,
                         embed_dim=cfg.rl.actor_critic.embed_dim,
                         lstm_units=cfg.rl.actor_critic.lstm_units).to(device)
    ac_cfg = cfg.rl.actor_critic
    ac_opt = torch.optim.RMSprop(ac.parameters(), lr=sum(ac_cfg.learning_rate_range) / 2,
                                 alpha=0.99, eps=1e-8, momentum=ac_cfg.gradient_momentum)
    vision_opt = torch.optim.RMSprop(vision.parameters(), lr=cfg.rl.vision.learning_rate,
                                     alpha=0.99, eps=1e-8)
    grid_opt = build_optimizer(grid, cfg, learning_rate=cfg.rl.grid.learning_rate,
                               weight_decay=cfg.rl.grid.weight_decay)
    if checkpoint is not None:
        vision.load_state_dict(checkpoint["vision_module"])
        grid.load_state_dict(checkpoint["grid_network"])
        vision_opt.load_state_dict(checkpoint["vision_optimizer"])
        grid_opt.load_state_dict(checkpoint["grid_optimizer"])
        if not reset_actor_critic:
            ac.load_state_dict(checkpoint["actor_critic"])
            ac_opt.load_state_dict(checkpoint["ac_optimizer"])
    return Agent(cfg, vision, grid, ac, device), (ac_opt, vision_opt, grid_opt), place, hd


def save_checkpoint(path, learner: A2CLearner, include_replay: bool = False):
    ckpt = {
        "algorithm": "a2c",
        "step": learner.frames,
        "updates": learner.updates,
        "vision_module": learner.agent.vision.state_dict(),
        "grid_network": learner.agent.grid.state_dict(),
        "actor_critic": learner.agent.ac.state_dict(),
        "vision_optimizer": learner.vision_opt.state_dict(),
        "grid_optimizer": learner.grid_opt.state_dict(),
        "ac_optimizer": learner.ac_opt.state_dict(),
    }
    if include_replay:
        ckpt["replay"] = learner.replay.state_dict()
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def train_a2c(cfg, env_factories=None, vec_env_cls=None, max_updates: int | None = None,
              reset_actor_critic: bool = False) -> A2CLearner:
    a2c = cfg.rl.a2c
    device = torch.device(a2c.device if torch.cuda.is_available() or a2c.device == "cpu"
                          else "cpu")
    torch.manual_seed(cfg.rl.seed)
    os.makedirs(cfg.rl.results_dir, exist_ok=True)

    checkpoint = None
    if cfg.rl.resume_from:
        checkpoint = torch.load(cfg.rl.resume_from, map_location=device, weights_only=False)
    agent, (ac_opt, vision_opt, grid_opt), place, hd = build_agent(
        cfg, device, checkpoint, reset_actor_critic)

    if env_factories is None:
        env_factories = [DMLabFactory(cfg.rl.env, renderer=a2c.renderer)
                         for _ in range(a2c.num_envs)]
    vec_env_cls = vec_env_cls or SubprocVecEnv
    vec_env = vec_env_cls(env_factories, cfg.rl.env, base_seed=cfg.rl.seed)
    replay = TimeMajorReplay(max(a2c.replay_frames // len(env_factories), cfg.rl.grid.seq_len),
                             len(env_factories), image_shape=(3, cfg.rl.env.height,
                                                              cfg.rl.env.width),
                             seed=cfg.rl.seed)
    learner = A2CLearner(cfg, vec_env, agent, ac_opt, vision_opt, grid_opt, place, hd, replay,
                         log_dir=cfg.rl.results_dir, seed=cfg.rl.seed)
    if checkpoint is not None and not reset_actor_critic:
        learner.frames = checkpoint["step"]
        learner.updates = checkpoint.get("updates", 0)
    if checkpoint is not None and "replay" in checkpoint:
        replay.load_state_dict(checkpoint["replay"])

    stop = {"flag": False}

    def handle(signum, frame):
        stop["flag"] = True
    old = {s: signal.signal(s, handle) for s in (signal.SIGTERM, signal.SIGINT)}

    last_ckpt = learner.frames
    try:
        while learner.frames < cfg.rl.total_env_steps and not stop["flag"]:
            stats = learner.train_iteration()
            if learner.updates % cfg.rl.log_every_n_updates == 0:
                print(f"frames {stats['frames']:>11,} | upd {stats['update']:>6} | "
                      f"V {stats['mean_value']:+.3f} | H {stats['entropy']:.3f} | "
                      f"still {stats['still_frac']:.2f} | vis {stats['vision_loss']:.3f} | "
                      f"grid {stats['grid_loss']:.3f} | {stats['fps']:.0f} fps", flush=True)
            if learner.frames - last_ckpt >= a2c.checkpoint_every_frames:
                save_checkpoint(os.path.join(cfg.rl.results_dir,
                                             f"checkpoint_step{learner.frames}.pt"), learner)
                last_ckpt = learner.frames
            if max_updates is not None and learner.updates >= max_updates:
                break
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        save_checkpoint(os.path.join(cfg.rl.results_dir,
                                     f"checkpoint_step{learner.frames}_final.pt"), learner)
        vec_env.close()
    return learner
