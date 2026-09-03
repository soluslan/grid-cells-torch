import os
import signal
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image, ImageDraw

from actor_critic import ActorCriticLSTM, DISCRETE_ACTIONS, N_ACTIONS
from config import Config
from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble, build_rl_ensembles
from model import GridCellsRNN
from replay_buffer import FrameReplayBuffer, SequenceAccumulator, SequenceReplayBuffer
from train import build_model, build_optimizer, train_step
from vision import VisionModule


def _cap_thread_pools() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _log_line(path: str, *values) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(",".join(str(v) for v in values) + "\n")


class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8, momentum=0.0):
        super().__init__(params, lr=lr, alpha=alpha, eps=eps, momentum=momentum)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["square_avg"] = torch.zeros_like(p.data)
                if momentum > 0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)

    def share_memory(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"].share_memory_()
                state["square_avg"].share_memory_()
                if "momentum_buffer" in state:
                    state["momentum_buffer"].share_memory_()


def world_to_meters(pos_world: torch.Tensor, env_cfg) -> torch.Tensor:
    lo, hi = env_cfg.wall_world_units
    center = (lo + hi) / 2.0
    scale = env_cfg.env_size_m / (hi - lo)
    return (pos_world - center) * scale


def actor_worker(
    rank: int, cfg: Config,
    vision_module: VisionModule, grid_network: GridCellsRNN, actor_critic: ActorCriticLSTM,
    ac_optimizer: SharedRMSprop,
    frame_buffer: FrameReplayBuffer, seq_buffer: SequenceReplayBuffer,
    place_ensemble, hd_ensemble,
    step_counter, stop_flag,
) -> None:
    _cap_thread_pools()
    import deepmind_lab

    torch.manual_seed(cfg.rl.seed + rank)
    env_cfg = cfg.rl.env
    level_dir = os.path.abspath(env_cfg.level_directory)
    env = deepmind_lab.Lab(
        env_cfg.level,
        ["RGB_INTERLEAVED", "VEL.TRANS", "VEL.ROT", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
        config={"width": str(env_cfg.width), "height": str(env_cfg.height),
                "levelDirectory": level_dir},
    )
    seq_acc = SequenceAccumulator(seq_buffer)

    def observe():
        obs = env.observations()
        image = torch.from_numpy(obs["RGB_INTERLEAVED"].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        pos_m = world_to_meters(torch.from_numpy(obs["DEBUG.POS.TRANS"][:2].copy()).float(), env_cfg)
        heading_rad = torch.tensor(
            [obs["DEBUG.POS.ROT"][1] * torch.pi / 180.0], dtype=torch.float32)
        vel_trans = torch.from_numpy(obs["VEL.TRANS"].copy()).float()
        vel_rot = torch.from_numpy(obs["VEL.ROT"].copy()).float()
        return image, pos_m, heading_rad, vel_trans, vel_rot

    def ego_vel_from(vel_trans, vel_rot):
        scale = env_cfg.env_size_m / (env_cfg.wall_world_units[1] - env_cfg.wall_world_units[0])
        u = vel_trans[0].unsqueeze(0) * scale
        v = vel_trans[1].unsqueeze(0) * scale
        dt = env_cfg.action_repeat / 60.0
        dtheta = vel_rot[1] * dt * torch.pi / 180.0
        ego_vel = torch.cat([u, v, torch.sin(dtheta).unsqueeze(0), torch.cos(dtheta).unsqueeze(0)])
        return ego_vel + torch.randn_like(ego_vel) * env_cfg.velocity_noise_std

    env.reset(seed=cfg.rl.seed + rank)
    image, pos_m, heading_rad, vel_trans, vel_rot = observe()
    grid_hidden = grid_network.init_hidden_from_predictions(
        [torch.zeros(1, place_ensemble.n_cells), torch.zeros(1, hd_ensemble.n_cells)])
    ac_hidden = actor_critic.init_hidden(1)
    seq_acc.reset(init_pos=pos_m, init_hd=heading_rad)
    prev_action = torch.zeros(1, dtype=torch.long)
    goal_grid_code = torch.zeros(1, cfg.model.nh_bottleneck)
    reward = torch.zeros(1, 1)
    episode_reward = 0.0
    episode_log_path = os.path.join(cfg.rl.results_dir, f"actor_{rank}_episodes.csv")

    log_probs, values, rewards, entropies = [], [], [], []

    while step_counter.value < cfg.rl.total_env_steps and not stop_flag.value:
        ego_vel = ego_vel_from(vel_trans, vel_rot).unsqueeze(0)

        with torch.no_grad():
            vision_out = vision_module(image.unsqueeze(0))
            vision_t = torch.cat([vision_out.place_probs, vision_out.hd_probs], dim=-1)
            grid_code, grid_hidden = grid_network.step(ego_vel, vision_t, grid_hidden)

        ac_out = actor_critic.step(
            image.unsqueeze(0), reward, prev_action, grid_code.detach(), goal_grid_code, ac_hidden)
        ac_hidden = ac_out.hidden
        dist = torch.distributions.Categorical(logits=ac_out.action_logits)
        action_idx = dist.sample()

        r = env.step(DISCRETE_ACTIONS[action_idx.item()], num_steps=env_cfg.action_repeat)
        with step_counter.get_lock():
            step_counter.value += env_cfg.action_repeat

        log_probs.append(dist.log_prob(action_idx))
        values.append(ac_out.value)
        rewards.append(float(r))
        entropies.append(dist.entropy())

        if r > 0:
            goal_grid_code = grid_code.detach()

        episode_reward += float(r)
        episode_ended = not env.is_running()
        if episode_ended:
            env.reset(seed=cfg.rl.seed + rank + int(step_counter.value))
            _log_line(episode_log_path, step_counter.value, episode_reward)
            episode_reward = 0.0

        frame_buffer.write_step(image, pos_m, heading_rad)
        step_image = image

        image, pos_m, heading_rad, vel_trans, vel_rot = observe()
        prev_action = action_idx

        teleported = r > 0 or episode_ended
        if teleported:
            grid_hidden = grid_network.init_hidden_from_predictions(
                [vision_out.place_probs.detach(), vision_out.hd_probs.detach()])
            ac_hidden = actor_critic.init_hidden(1)
            seq_acc.reset(init_pos=pos_m, init_hd=heading_rad)
        else:
            seq_acc.add_step(ego_vel.squeeze(0), pos_m, heading_rad, step_image)

        if len(rewards) == cfg.rl.actor_critic.n_step or teleported:
            _apply_a3c_update(
                actor_critic, ac_optimizer, log_probs, values, entropies, rewards,
                bootstrap_value=(0.0 if teleported else ac_out.value.item()),
                cfg=cfg.rl.actor_critic)
            log_probs, values, rewards, entropies = [], [], [], []
            ac_hidden = (ac_hidden[0].detach(), ac_hidden[1].detach())

    env.close()


def _apply_a3c_update(actor_critic, optimizer, log_probs, values, entropies, rewards,
                       bootstrap_value: float, cfg) -> None:
    R = bootstrap_value
    returns = []
    for r in reversed(rewards):
        R = r + cfg.discount * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    values_t = torch.cat(values)
    log_probs_t = torch.cat(log_probs)
    entropy_t = torch.cat(entropies)

    advantage = returns - values_t.detach()
    policy_loss = -(log_probs_t * advantage).mean()
    value_loss = F.mse_loss(values_t, returns)
    entropy_loss = -entropy_t.mean()
    alpha = sum(cfg.baseline_cost_range) / 2
    beta = sum(cfg.entropy_reg_range) / 2
    loss = policy_loss + alpha * value_loss + beta * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def vision_learner_worker(cfg: Config, vision_module: VisionModule, optimizer: SharedRMSprop,
                           frame_buffer: FrameReplayBuffer, place_ensemble, hd_ensemble,
                           step_counter, stop_flag) -> None:
    _cap_thread_pools()
    batch_size = cfg.rl.vision.minibatch_size
    log_path = os.path.join(cfg.rl.results_dir, "vision_loss.csv")
    n_updates = 0
    while len(frame_buffer) < batch_size and not stop_flag.value:
        if step_counter.value >= cfg.rl.total_env_steps:
            return
    while step_counter.value < cfg.rl.total_env_steps and not stop_flag.value:
        batch = frame_buffer.sample(batch_size)
        place_targets = place_ensemble.posterior(batch["pos"].unsqueeze(1)).squeeze(1)
        hd_targets = hd_ensemble.posterior(batch["hd"].unsqueeze(1)).squeeze(1)
        loss = vision_module.loss(batch["image"], place_targets, hd_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_updates += 1
        if n_updates % cfg.rl.log_every_n_updates == 0:
            _log_line(log_path, step_counter.value, n_updates, loss.item())


def grid_learner_worker(cfg: Config, grid_network: GridCellsRNN, vision_module: VisionModule,
                         optimizer: SharedRMSprop, seq_buffer: SequenceReplayBuffer,
                         place_ensembles, hd_ensembles, step_counter, stop_flag) -> None:
    _cap_thread_pools()
    batch_size = cfg.rl.grid.minibatch_size
    log_path = os.path.join(cfg.rl.results_dir, "grid_loss.csv")
    n_updates = 0
    while len(seq_buffer) < batch_size and not stop_flag.value:
        if step_counter.value >= cfg.rl.total_env_steps:
            return
    while step_counter.value < cfg.rl.total_env_steps and not stop_flag.value:
        batch = seq_buffer.sample(batch_size)
        loss = train_step(grid_network, place_ensembles, hd_ensembles, batch, cfg,
                          device=torch.device("cpu"), optimizer=optimizer,
                          vision_module=vision_module)
        n_updates += 1
        if n_updates % cfg.rl.log_every_n_updates == 0:
            _log_line(log_path, step_counter.value, n_updates, loss)


def _save_checkpoint(path: str, step: int, vision_module, grid_network, actor_critic,
                      vision_optimizer, grid_optimizer, ac_optimizer,
                      frame_buffer=None, seq_buffer=None) -> None:
    ckpt = {
        "step": step,
        "vision_module": vision_module.state_dict(),
        "grid_network": grid_network.state_dict(),
        "actor_critic": actor_critic.state_dict(),
        "vision_optimizer": vision_optimizer.state_dict(),
        "grid_optimizer": grid_optimizer.state_dict(),
        "ac_optimizer": ac_optimizer.state_dict(),
    }
    if frame_buffer is not None:
        ckpt["frame_buffer"] = frame_buffer.state_dict()
    if seq_buffer is not None:
        ckpt["seq_buffer"] = seq_buffer.state_dict()
    torch.save(ckpt, path)


def checkpoint_worker(cfg: Config, vision_module: VisionModule, grid_network: GridCellsRNN,
                       actor_critic: ActorCriticLSTM, vision_optimizer, grid_optimizer,
                       ac_optimizer, frame_buffer, seq_buffer, step_counter, stop_flag) -> None:
    _cap_thread_pools()
    os.makedirs(cfg.rl.results_dir, exist_ok=True)
    last_checkpoint_step = -cfg.rl.checkpoint_every_env_steps
    while step_counter.value < cfg.rl.total_env_steps and not stop_flag.value:
        step = step_counter.value
        if step - last_checkpoint_step >= cfg.rl.checkpoint_every_env_steps:
            _save_checkpoint(os.path.join(cfg.rl.results_dir, f"checkpoint_step{step}.pt"),
                              step, vision_module, grid_network, actor_critic,
                              vision_optimizer, grid_optimizer, ac_optimizer)
            _save_checkpoint(os.path.join(cfg.rl.results_dir, "checkpoint_latest.pt"),
                              step, vision_module, grid_network, actor_critic,
                              vision_optimizer, grid_optimizer, ac_optimizer,
                              frame_buffer=frame_buffer, seq_buffer=seq_buffer)
            last_checkpoint_step = step
        time.sleep(5)

    step = step_counter.value
    _save_checkpoint(os.path.join(cfg.rl.results_dir, f"checkpoint_step{step}_final.pt"),
                      step, vision_module, grid_network, actor_critic,
                      vision_optimizer, grid_optimizer, ac_optimizer,
                      frame_buffer=frame_buffer, seq_buffer=seq_buffer)


def build_shared_models(cfg: Config, checkpoint: dict | None = None):
    device = torch.device("cpu")
    place_ensembles, hd_ensembles = build_rl_ensembles(cfg, device)
    place_ensemble, hd_ensemble = place_ensembles[0], hd_ensembles[0]

    grid_network = build_model(cfg, place_ensembles + hd_ensembles, device,
                                vision_dim=place_ensemble.n_cells + hd_ensemble.n_cells,
                                ego_vel_dim=4)
    vision_module = VisionModule(place_ensemble, hd_ensemble, embed_dim=cfg.rl.vision.embed_dim,
                                  mask_prob=cfg.rl.vision.mask_prob)
    actor_critic = ActorCriticLSTM(grid_code_dim=cfg.model.nh_bottleneck,
                                    embed_dim=cfg.rl.actor_critic.embed_dim,
                                    lstm_units=cfg.rl.actor_critic.lstm_units)
    if checkpoint is not None:
        vision_module.load_state_dict(checkpoint["vision_module"])
        grid_network.load_state_dict(checkpoint["grid_network"])
        actor_critic.load_state_dict(checkpoint["actor_critic"])
    for m in (grid_network, vision_module, actor_critic):
        m.share_memory()

    grid_optimizer = build_optimizer(grid_network, cfg,
                                      learning_rate=cfg.rl.grid.learning_rate,
                                      weight_decay=cfg.rl.grid.weight_decay)
    vision_optimizer = SharedRMSprop(vision_module.parameters(), lr=cfg.rl.vision.learning_rate)
    ac_lr = sum(cfg.rl.actor_critic.learning_rate_range) / 2
    ac_optimizer = SharedRMSprop(actor_critic.parameters(), lr=ac_lr,
                                  momentum=cfg.rl.actor_critic.gradient_momentum)
    if checkpoint is not None and "vision_optimizer" in checkpoint:
        vision_optimizer.load_state_dict(checkpoint["vision_optimizer"])
    elif checkpoint is not None:
        print("resume: checkpoint has no vision_optimizer state -- starting it fresh")
    if checkpoint is not None and "ac_optimizer" in checkpoint:
        ac_optimizer.load_state_dict(checkpoint["ac_optimizer"])
    elif checkpoint is not None:
        print("resume: checkpoint has no ac_optimizer state -- starting it fresh")
    for opt in (vision_optimizer, ac_optimizer):
        opt.share_memory()
    if checkpoint is not None and "grid_optimizer" in checkpoint:
        grid_optimizer.load_state_dict(checkpoint["grid_optimizer"])
    elif checkpoint is not None:
        print("resume: checkpoint has no grid_optimizer state -- starting it fresh")
    for group in grid_optimizer.param_groups:
        for p in group["params"]:
            state = grid_optimizer.state.setdefault(p, {})
            state.setdefault("step", torch.zeros(1))
            state.setdefault("square_avg", torch.zeros_like(p.data))
            if group.get("momentum", 0) != 0:
                state.setdefault("momentum_buffer", torch.zeros_like(p.data))
            for buf in state.values():
                buf.share_memory_()

    return dict(
        vision_module=vision_module, grid_network=grid_network, actor_critic=actor_critic,
        place_ensemble=place_ensemble, hd_ensemble=hd_ensemble,
        place_ensembles=place_ensembles, hd_ensembles=hd_ensembles,
        grid_optimizer=grid_optimizer, vision_optimizer=vision_optimizer, ac_optimizer=ac_optimizer,
    )


def build_models_from_checkpoint(ckpt, cfg: Config):
    n_pc, n_hdc = cfg.task.n_pc[0], cfg.task.n_hdc[0]
    place = PlaceCellEnsemble(n_pc, stdev=cfg.rl.pc_scale[0], pos_min=-cfg.rl.env.env_size_m / 2,
                               pos_max=cfg.rl.env.env_size_m / 2, seed=cfg.task.neurons_seed)
    hd = HeadDirectionCellEnsemble(n_hdc, concentration=cfg.task.hdc_concentration[0],
                                   seed=cfg.task.neurons_seed)

    has_rl_lstm = any(k.startswith("rl_lstm.") for k in ckpt["grid_network"])
    vision_dim = (n_pc + n_hdc) if has_rl_lstm else 0
    lstm_key = "rl_lstm.weight_ih_l0" if has_rl_lstm else "lstm.weight_ih_l0"
    ego_vel_dim = ckpt["grid_network"][lstm_key].shape[1] - vision_dim
    grid_network = GridCellsRNN(target_ensembles=[place, hd], nh_lstm=cfg.model.nh_lstm,
                                nh_bottleneck=cfg.model.nh_bottleneck,
                                dropout_rates=cfg.model.dropout_rates,
                                bottleneck_has_bias=cfg.model.bottleneck_has_bias,
                                ego_vel_dim=ego_vel_dim, vision_dim=vision_dim)
    grid_network.load_state_dict(ckpt["grid_network"])
    grid_network.eval()
    place_loaded, hd_loaded = grid_network.target_ensembles[0], grid_network.target_ensembles[1]

    vision_module = VisionModule(place_loaded, hd_loaded, embed_dim=cfg.rl.vision.embed_dim,
                                 mask_prob=cfg.rl.vision.mask_prob)
    vision_module.load_state_dict(ckpt["vision_module"])
    vision_module.eval()

    actor_critic = ActorCriticLSTM(grid_code_dim=cfg.model.nh_bottleneck,
                                   embed_dim=cfg.rl.actor_critic.embed_dim,
                                   lstm_units=cfg.rl.actor_critic.lstm_units)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval()

    print(f"loaded checkpoint at step {ckpt['step']} -- grid network has vision input: "
         f"{has_rl_lstm}, ego_vel_dim: {ego_vel_dim}")
    return grid_network, vision_module, actor_critic, place_loaded, hd_loaded, has_rl_lstm, ego_vel_dim


def grid_step_velocity_only(grid_network, vel_t, hidden):
    """Mirrors the pre-2026-08-29 GridCellsRNN.step(): velocity-only, self.lstm/self.bottleneck.
    Used when the loaded checkpoint's grid network has no rl_lstm (vision_dim=0)."""
    lstm_out, hidden = grid_network.lstm(vel_t.unsqueeze(1), hidden)
    bottleneck = grid_network.bottleneck(lstm_out.squeeze(1))
    return bottleneck, hidden


def draw_hud(frame_hwc_uint8, episode_idx, t_seconds, cumulative_reward):
    img = Image.fromarray(frame_hwc_uint8)
    draw = ImageDraw.Draw(img)
    text = f"episode {episode_idx}   t={t_seconds:5.1f}s   reward={cumulative_reward:.0f}"
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(img)


def train_rl_agent(cfg: Config) -> None:
    mp.set_start_method("spawn", force=True)

    checkpoint = None
    if cfg.rl.resume_from:
        checkpoint = torch.load(cfg.rl.resume_from, map_location="cpu", weights_only=False)
        print(f"resuming from {cfg.rl.resume_from} at step {checkpoint['step']}")

    models = build_shared_models(cfg, checkpoint)
    frame_buffer = FrameReplayBuffer(capacity=cfg.rl.replay.frame_capacity,
                                      image_size=cfg.rl.env.width)
    seq_buffer = SequenceReplayBuffer(capacity=cfg.rl.replay.sequence_capacity,
                                       seq_len=cfg.rl.grid.seq_len,
                                       image_size=cfg.rl.env.width,
                                       ego_vel_dim=4)
    if checkpoint is not None and "frame_buffer" in checkpoint:
        frame_buffer.load_state_dict(checkpoint["frame_buffer"])
    elif checkpoint is not None:
        print("resume: checkpoint has no frame_buffer state -- starting it empty")
    if checkpoint is not None and "seq_buffer" in checkpoint:
        seq_buffer.load_state_dict(checkpoint["seq_buffer"])
    elif checkpoint is not None:
        print("resume: checkpoint has no seq_buffer state -- starting it empty")
    step_counter = mp.Value("l", checkpoint["step"] if checkpoint is not None else 0)
    stop_flag = mp.Value("b", False)

    def _handle_stop(signum, frame):
        stop_flag.value = True
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    processes = []
    for rank in range(cfg.rl.env.num_actors):
        p = mp.Process(target=actor_worker, args=(
            rank, cfg, models["vision_module"], models["grid_network"], models["actor_critic"],
            models["ac_optimizer"], frame_buffer, seq_buffer,
            models["place_ensemble"], models["hd_ensemble"], step_counter, stop_flag))
        p.start()
        processes.append(p)

    for target, args in (
        (vision_learner_worker, (cfg, models["vision_module"], models["vision_optimizer"],
                                  frame_buffer, models["place_ensemble"], models["hd_ensemble"],
                                  step_counter, stop_flag)),
        (grid_learner_worker, (cfg, models["grid_network"], models["vision_module"],
                                models["grid_optimizer"], seq_buffer,
                                models["place_ensembles"], models["hd_ensembles"],
                                step_counter, stop_flag)),
        (checkpoint_worker, (cfg, models["vision_module"], models["grid_network"],
                              models["actor_critic"], models["vision_optimizer"],
                              models["grid_optimizer"], models["ac_optimizer"],
                              frame_buffer, seq_buffer, step_counter, stop_flag)),
    ):
        p = mp.Process(target=target, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return models
