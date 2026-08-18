"""A3C training orchestrator for the RL agent (RL-agent roadmap plan, M4).

Four kinds of concurrent process share one set of weights and update disjoint parts of it (see
the roadmap plan's "Engineering design" note and M4 walkthrough):

1. `num_actors` actor processes, each with its own DeepMind Lab environment instance (required
   -- DMLab is "effectively not reentrant" per `lab/docs/users/issues.md`). Each does `no_grad()`
   inference through the shared vision module and grid network, `.detach()`s the grid code
   before its own *local* actor-critic forward/backward, then applies that gradient straight to
   the shared actor-critic weights via a shared, Hogwild-style optimizer (no lock). Also writes
   its experience into the two replay buffers.
2. One vision-learner process: samples single frames from `FrameReplayBuffer`, supervised loss,
   updates the shared vision module (sole writer).
3. One grid-learner process: samples 100-step segments from `SequenceReplayBuffer`, reuses
   `train.train_step()`/`build_optimizer()` completely unchanged, updates the shared grid
   network (sole writer).

Stop-gradient boundaries between the three components are automatic once they're plain
`.detach()`/`no_grad()` calls -- no cross-process autograd graph exists to leak through even
without them, but they're kept anyway as defensive, self-documenting code matching Methods'
architecture diagram.
"""

import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from actor_critic import ActorCriticLSTM, DISCRETE_ACTIONS, N_ACTIONS
from config import Config
from ensembles import build_rl_ensembles
from model import GridCellsRNN
from replay_buffer import FrameReplayBuffer, SequenceAccumulator, SequenceReplayBuffer
from train import build_model, build_optimizer, train_step
from vision import VisionModule


def _cap_thread_pools() -> None:
    """Call as the first line of every worker process (actor/learner/checkpoint). With
    `spawn`, each of the ~35 processes independently re-imports torch and, uncapped, claims up
    to the machine's full core count for its own BLAS/OMP thread pool -- up to ~560 threads
    contending for 16 cores. Confirmed absent from this codebase (grep for set_num_threads/
    OMP_NUM_THREADS/MKL_NUM_THREADS found nothing) and flagged by the roadmap plan's
    "Throughput fix" section as the single most common PyTorch-multiprocessing throughput bug.
    Each of these processes does small, single-sample or small-batch work (one frame, one
    action) -- there's no large matmul here that would benefit from intra-op parallelism
    anyway.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _log_line(path: str, *values) -> None:
    """Plain per-process CSV append -- each worker only ever writes its own file (one file per
    actor rank, one for each learner), so there's no cross-process write contention to manage.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(",".join(str(v) for v in values) + "\n")


# --------------------------------------------------------------------------
# Shared (Hogwild) optimizer
# --------------------------------------------------------------------------

class SharedRMSprop(torch.optim.RMSprop):
    """RMSprop whose per-parameter running-average state is pre-allocated and
    `share_memory_()`'d at construction time, so every process that receives this same
    optimizer object (passed as a `torch.multiprocessing.Process` arg, not rebuilt per-process)
    reads and writes the *same* physical statistics -- the standard Hogwild pattern (à la
    `ikostrikov/pytorch-a3c`). Must be constructed, and `.share_memory()` called, in the parent
    process *before* any actor process is spawned.
    """

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


# --------------------------------------------------------------------------
# Coordinate transform: DMLab world units (square_arena.lua's cellOrigin convention) -> the
# paper's meter-centered coordinates (same space PlaceCellEnsemble/dataset.py already use).
# --------------------------------------------------------------------------

def world_to_meters(pos_world: torch.Tensor, env_cfg) -> torch.Tensor:
    lo, hi = env_cfg.wall_world_units
    center = (lo + hi) / 2.0
    scale = env_cfg.env_size_m / (hi - lo)
    return (pos_world - center) * scale


# --------------------------------------------------------------------------
# Actor process
# --------------------------------------------------------------------------

def actor_worker(
    rank: int, cfg: Config,
    vision_module: VisionModule, grid_network: GridCellsRNN, actor_critic: ActorCriticLSTM,
    ac_optimizer: SharedRMSprop,
    frame_buffer: FrameReplayBuffer, seq_buffer: SequenceReplayBuffer,
    place_ensemble, hd_ensemble,
    step_counter, stop_flag,
) -> None:
    """One actor's whole life: own DMLab instance, rollout loop, local A3C updates applied to
    the shared actor-critic, experience pushed to both replay buffers. Runs until
    `step_counter` (a `torch.multiprocessing.Value`, shared and incremented here) reaches
    `cfg.rl.total_env_steps` or `stop_flag` is set.

    Deliberately importable and callable directly (not only via `mp.Process`) for testing: see
    `rl_train.py`'s own smoke test, which calls this in-process against the real
    `square_arena` level before trusting the multiprocess wiring.
    """
    _cap_thread_pools()
    import deepmind_lab  # imported here so this module stays importable without deepmind_lab
                          # installed (e.g. for unit-testing SharedRMSprop/world_to_meters alone)

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
        # [speed, sin(dtheta), cos(dtheta)] -- matches dataset.py's convention (README's "Why
        # dtheta, not rot_vel"), so the grid-learner sees exactly the feature the supervised
        # network was trained on. dtheta here is approximated from the instantaneous VEL.ROT
        # yaw rate x the wall-clock time of one action_repeat block (60fps engine, per
        # docs/users/python_api.md's `fps` default) -- an approximation of the *net* heading
        # change dataset.py derives from consecutive stored headings, not identical to it; a
        # known simplification, flagged rather than silently assumed equivalent.
        speed = vel_trans[:2].norm().unsqueeze(0) * (env_cfg.env_size_m / (env_cfg.wall_world_units[1] - env_cfg.wall_world_units[0]))
        dt = env_cfg.action_repeat / 60.0
        dtheta = vel_rot[1] * dt * torch.pi / 180.0
        return torch.cat([speed, torch.sin(dtheta).unsqueeze(0), torch.cos(dtheta).unsqueeze(0)])

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
            grid_code, grid_hidden = grid_network.step(ego_vel, grid_hidden)
            vision_out = vision_module(image.unsqueeze(0))

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
            # This level's only positive-reward source is the `goal` pickup (the cue is
            # quantity=0) -- record its grid code as g_* for the rest of the episode, per
            # Methods' "goal grid code ... observed the last time the goal was reached".
            goal_grid_code = grid_code.detach()

        episode_reward += float(r)
        episode_ended = not env.is_running()
        if episode_ended:
            env.reset(seed=cfg.rl.seed + rank + int(step_counter.value))
            _log_line(episode_log_path, step_counter.value, episode_reward)
            episode_reward = 0.0

        frame_buffer.write_step(image, pos_m, heading_rad)

        image, pos_m, heading_rad, vel_trans, vel_rot = observe()
        prev_action = action_idx

        teleported = r > 0 or episode_ended  # this level only relocates the agent on goal-reach
                                              # (mid-episode) or episode reset (timeout)
        if teleported:
            grid_hidden = grid_network.init_hidden_from_predictions(
                [vision_out.place_probs.detach(), vision_out.hd_probs.detach()])
            ac_hidden = actor_critic.init_hidden(1)
            seq_acc.reset(init_pos=pos_m, init_hd=heading_rad)
        else:
            seq_acc.add_step(ego_vel.squeeze(0), pos_m, heading_rad)

        if len(rewards) == cfg.rl.actor_critic.n_step or teleported:
            _apply_a3c_update(
                actor_critic, ac_optimizer, log_probs, values, entropies, rewards,
                bootstrap_value=(0.0 if teleported else ac_out.value.item()),
                cfg=cfg.rl.actor_critic)
            log_probs, values, rewards, entropies = [], [], [], []
            # Truncated BPTT: each n-step window's loss.backward() frees that window's
            # autograd graph, but `ac_hidden` (unlike the teleport branch's fresh
            # init_hidden()) still references it going into the next window -- without
            # detaching here, the next window's step() would try to backward through
            # already-freed graph nodes (hit this exact RuntimeError before this fix).
            ac_hidden = (ac_hidden[0].detach(), ac_hidden[1].detach())

    env.close()


def _apply_a3c_update(actor_critic, optimizer, log_probs, values, entropies, rewards,
                       bootstrap_value: float, cfg) -> None:
    """n-step A3C update (Methods: L = L_pi + alpha*L_V + beta*L_H), applied directly to the
    shared actor-critic weights via the shared optimizer -- the Hogwild step.
    """
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


# --------------------------------------------------------------------------
# Learner processes
# --------------------------------------------------------------------------

def vision_learner_worker(cfg: Config, vision_module: VisionModule, optimizer: SharedRMSprop,
                           frame_buffer: FrameReplayBuffer, place_ensemble, hd_ensemble,
                           step_counter, stop_flag) -> None:
    """Sole writer to the shared vision module. Waits for the buffer to have at least one
    minibatch's worth of frames before it starts training."""
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


def grid_learner_worker(cfg: Config, grid_network: GridCellsRNN, optimizer: SharedRMSprop,
                         seq_buffer: SequenceReplayBuffer, place_ensembles, hd_ensembles,
                         step_counter, stop_flag) -> None:
    """Sole writer to the shared grid network. Reuses train.train_step() completely
    unchanged -- see replay_buffer.py's module docstring for why SequenceReplayBuffer's
    schema was built to make this possible.

    `place_ensembles`/`hd_ensembles` are LISTS (train_step()/encode_initial_conditions()
    iterate over them, matching build_rl_ensembles()'s return shape) -- not the single ensembles
    the actor loop and vision learner use directly via .posterior(); see
    build_shared_models(), which returns both forms.
    """
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
                          device=torch.device("cpu"), optimizer=optimizer)
        n_updates += 1
        if n_updates % cfg.rl.log_every_n_updates == 0:
            _log_line(log_path, step_counter.value, n_updates, loss)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def _save_checkpoint(path: str, step: int, vision_module, grid_network, actor_critic) -> None:
    torch.save({
        "step": step,
        "vision_module": vision_module.state_dict(),
        "grid_network": grid_network.state_dict(),
        "actor_critic": actor_critic.state_dict(),
    }, path)


def checkpoint_worker(cfg: Config, vision_module: VisionModule, grid_network: GridCellsRNN,
                       actor_critic: ActorCriticLSTM, step_counter, stop_flag) -> None:
    """Sole writer of checkpoints -- periodically (polling step_counter every 5s) snapshots
    all three shared models to disk, plus always a final one when the run ends. The three
    models aren't snapshotted as one atomic instant (actor/learner processes may still be
    writing to the others mid-snapshot) -- the same eventually-consistent tradeoff
    replay_buffer.py's lock-free reads already accept; exact cross-model synchronization isn't
    needed for a checkpoint meant to resume training or run evaluation from, only "recent
    enough." Mirrors train.py's save_checkpoint pattern for the supervised pipeline.
    """
    _cap_thread_pools()
    os.makedirs(cfg.rl.results_dir, exist_ok=True)
    last_checkpoint_step = -cfg.rl.checkpoint_every_env_steps  # forces an immediate first save
    while step_counter.value < cfg.rl.total_env_steps and not stop_flag.value:
        step = step_counter.value
        if step - last_checkpoint_step >= cfg.rl.checkpoint_every_env_steps:
            _save_checkpoint(os.path.join(cfg.rl.results_dir, f"checkpoint_step{step}.pt"),
                              step, vision_module, grid_network, actor_critic)
            _save_checkpoint(os.path.join(cfg.rl.results_dir, "checkpoint_latest.pt"),
                              step, vision_module, grid_network, actor_critic)
            last_checkpoint_step = step
        time.sleep(5)  # polling, not busy-waiting -- checkpoints don't need sub-second precision

    step = step_counter.value
    _save_checkpoint(os.path.join(cfg.rl.results_dir, f"checkpoint_step{step}_final.pt"),
                      step, vision_module, grid_network, actor_critic)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_shared_models(cfg: Config):
    device = torch.device("cpu")  # shared-memory Hogwild is a CPU-multiprocess pattern here;
                                   # see roadmap plan -- GPU would need a different design
    place_ensembles, hd_ensembles = build_rl_ensembles(cfg, device)
    # Singular ensembles for direct .posterior() use (actor loop, vision learner); list forms
    # for train_step()/encode_initial_conditions(), which iterate over them (see
    # grid_learner_worker's docstring). Both refer to the same underlying ensemble objects.
    place_ensemble, hd_ensemble = place_ensembles[0], hd_ensembles[0]

    grid_network = build_model(cfg, place_ensembles + hd_ensembles, device)
    vision_module = VisionModule(place_ensemble, hd_ensemble, embed_dim=cfg.rl.vision.embed_dim,
                                  mask_prob=cfg.rl.vision.mask_prob)
    actor_critic = ActorCriticLSTM(grid_code_dim=cfg.model.nh_bottleneck,
                                    embed_dim=cfg.rl.actor_critic.embed_dim,
                                    lstm_units=cfg.rl.actor_critic.lstm_units)
    for m in (grid_network, vision_module, actor_critic):
        m.share_memory()

    grid_optimizer = build_optimizer(grid_network, cfg)  # existing supervised optimizer builder
    vision_optimizer = SharedRMSprop(vision_module.parameters(), lr=cfg.rl.vision.learning_rate)
    ac_lr = sum(cfg.rl.actor_critic.learning_rate_range) / 2
    ac_optimizer = SharedRMSprop(actor_critic.parameters(), lr=ac_lr,
                                  momentum=cfg.rl.actor_critic.gradient_momentum)
    for opt in (vision_optimizer, ac_optimizer):
        opt.share_memory()
    # grid_optimizer isn't a SharedRMSprop (train.build_optimizer returns a plain
    # torch.optim.RMSprop, reused unchanged) -- share its state manually the same way. Must
    # also pre-populate momentum_buffer when the group's momentum != 0 (cfg.train.momentum
    # defaults to 0.9, not 0) -- RMSprop.step() looks it up unconditionally in that case and
    # KeyErrors on a state dict that only has step/square_avg.
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


def train_rl_agent(cfg: Config) -> None:
    """Spawns num_actors actor processes + 2 learner processes sharing one set of weights.
    See module docstring and the roadmap plan's M4 walkthrough."""
    mp.set_start_method("spawn", force=True)  # DMLab global state + CUDA make fork unsafe

    models = build_shared_models(cfg)
    frame_buffer = FrameReplayBuffer(capacity=cfg.rl.replay.frame_capacity,
                                      image_size=cfg.rl.env.width)
    seq_buffer = SequenceReplayBuffer(capacity=cfg.rl.replay.sequence_capacity,
                                       seq_len=cfg.rl.grid.seq_len)
    step_counter = mp.Value("l", 0)
    stop_flag = mp.Value("b", False)

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
        (grid_learner_worker, (cfg, models["grid_network"], models["grid_optimizer"],
                                seq_buffer, models["place_ensembles"], models["hd_ensembles"],
                                step_counter, stop_flag)),
        (checkpoint_worker, (cfg, models["vision_module"], models["grid_network"],
                              models["actor_critic"], step_counter, stop_flag)),
    ):
        p = mp.Process(target=target, args=args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return models
