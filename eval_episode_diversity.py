"""Runs the latest RL checkpoint for 5 episodes (different seeds -> different goal/spawn/
texture/cue configurations of square_arena) and saves, per episode: a first-person mp4,
a top-down animated mp4, and a paper Fig. 2h-style static trajectory PNG.

Run from grid-cells-torch/: python3 eval_episode_diversity.py
"""
import argparse
import glob
import os
import re
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import deepmind_lab
from config import Config
from actor_critic import DISCRETE_ACTIONS
from rl_train import build_models_from_checkpoint, grid_step_velocity_only, draw_hud, world_to_meters

CHECKPOINT_DIR = "data/checkpoints/rl_baseline"
N_EPISODES = 5
VIDEO_SIZE = 336
FPS = 60
TOPDOWN_FPS = 15
MAX_SECONDS = 95  # small safety margin over the level's own 90s timeout


def find_latest_checkpoint(checkpoint_dir):
    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step*.pt"))
    candidates = [c for c in candidates if "_final" not in os.path.basename(c)]
    candidates.sort(
        key=lambda p: int(re.search(r"checkpoint_step(\d+)\.pt", p).group(1)),
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"no checkpoint_step*.pt found in {checkpoint_dir}")
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return path, ckpt
        except Exception as e:
            print(f"  failed to load {path} ({e}), trying next newest")
    raise RuntimeError("no loadable checkpoint found")


def run_episode(env, seed, cfg, grid_network, vision_module, actor_critic,
                 place, hd, has_rl_lstm, ego_vel_dim):
    env_cfg = cfg.rl.env

    def observe():
        obs = env.observations()
        big_rgb = obs["RGB_INTERLEAVED"].copy()
        small = np.asarray(Image.fromarray(big_rgb).resize((64, 64), Image.BILINEAR))
        image64 = torch.from_numpy(small).permute(2, 0, 1).float() / 127.5 - 1.0
        pos_m = world_to_meters(torch.from_numpy(obs["DEBUG.POS.TRANS"][:2].copy()).float(), env_cfg)
        vel_trans = torch.from_numpy(obs["VEL.TRANS"].copy()).float()
        vel_rot = torch.from_numpy(obs["VEL.ROT"].copy()).float()
        return image64, pos_m, vel_trans, vel_rot

    def ego_vel_from(vel_trans, vel_rot):
        scale = env_cfg.env_size_m / (env_cfg.wall_world_units[1] - env_cfg.wall_world_units[0])
        dt = env_cfg.action_repeat / 60.0
        dtheta = vel_rot[1] * dt * torch.pi / 180.0
        trig = [torch.sin(dtheta).unsqueeze(0), torch.cos(dtheta).unsqueeze(0)]
        if ego_vel_dim == 4:
            ego_vel = torch.cat([vel_trans[0].unsqueeze(0) * scale, vel_trans[1].unsqueeze(0) * scale] + trig)
        else:
            ego_vel = torch.cat([vel_trans[:2].norm().unsqueeze(0) * scale] + trig)
        return ego_vel + torch.randn_like(ego_vel) * env_cfg.velocity_noise_std

    max_frames = MAX_SECONDS * FPS
    env.reset(seed=seed)
    image64, pos_m, vel_trans, vel_rot = observe()
    start_pos = pos_m.clone()

    grid_hidden = grid_network.init_hidden_from_predictions(
        [torch.zeros(1, place.n_cells), torch.zeros(1, hd.n_cells)])
    ac_hidden = actor_critic.init_hidden(1)
    prev_action = torch.zeros(1, dtype=torch.long)
    goal_grid_code = torch.zeros(1, cfg.model.nh_bottleneck)
    reward_t = torch.zeros(1, 1)
    cumulative_reward = 0.0

    frames, positions, goal_touch_indices = [], [start_pos.clone()], []

    while env.is_running() and len(frames) < max_frames:
        ego_vel = ego_vel_from(vel_trans, vel_rot).unsqueeze(0)
        with torch.no_grad():
            vision_out = vision_module(image64.unsqueeze(0))
            if has_rl_lstm:
                vision_t = torch.cat([vision_out.place_probs, vision_out.hd_probs], dim=-1)
                grid_code, grid_hidden = grid_network.step(ego_vel, vision_t, grid_hidden)
            else:
                grid_code, grid_hidden = grid_step_velocity_only(grid_network, ego_vel, grid_hidden)
            ac_out = actor_critic.step(image64.unsqueeze(0), reward_t, prev_action,
                                        grid_code, goal_grid_code, ac_hidden)
            ac_hidden = ac_out.hidden
            action_idx = torch.distributions.Categorical(logits=ac_out.action_logits).sample()

        macro_reward = 0.0
        for _ in range(env_cfg.action_repeat):
            if not env.is_running() or len(frames) >= max_frames:
                break
            r = env.step(DISCRETE_ACTIONS[action_idx.item()], num_steps=1)
            macro_reward += float(r)
            cumulative_reward += float(r)
            if not env.is_running():
                break
            sub_obs = env.observations()
            frame = draw_hud(sub_obs["RGB_INTERLEAVED"].copy(), 0, len(frames) / FPS, cumulative_reward)
            frames.append(frame)
            pos_now = world_to_meters(torch.from_numpy(sub_obs["DEBUG.POS.TRANS"][:2].copy()).float(), env_cfg)
            positions.append(pos_now.clone())
            if r > 0:
                goal_touch_indices.append(len(positions) - 1)

        if macro_reward > 0:
            goal_grid_code = grid_code.detach()
        reward_t = torch.tensor([[macro_reward]], dtype=torch.float32)

        if not env.is_running() or len(frames) >= max_frames:
            break
        image64, pos_m, vel_trans, vel_rot = observe()
        prev_action = action_idx

    positions = torch.stack(positions).numpy()
    return frames, positions, goal_touch_indices, cumulative_reward


def save_firstperson_video(frames, path):
    writer = imageio.get_writer(path, fps=FPS, codec="libx264",
                                 ffmpeg_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()


def save_topdown_video(positions, goal_touch_indices, env_size_m, path, fps=TOPDOWN_FPS):
    half = env_size_m / 2
    goal_pos = positions[goal_touch_indices[0]] if goal_touch_indices else None
    step = max(1, round(FPS / fps))
    idxs = list(range(0, len(positions), step))

    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                 ffmpeg_params=["-pix_fmt", "yuv420p"])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    for k in idxs:
        ax.clear()
        ax.add_patch(plt.Rectangle((-half, -half), env_size_m, env_size_m,
                                    facecolor="#3a4a6b", edgecolor="#c0392b", linewidth=4, zorder=0))
        trail_start = max(0, k - 300)
        ax.plot(positions[trail_start:k + 1, 0], positions[trail_start:k + 1, 1],
                color="#f5d76e", lw=1.5, zorder=1)
        ax.scatter(*positions[k], color="white", s=80, zorder=3, edgecolor="k", linewidth=1.0)
        if goal_pos is not None:
            ax.scatter(*goal_pos, color="gold", s=140, zorder=2, edgecolor="k",
                        linewidth=1.0, marker="*")
        ax.set_xlim(-half - 0.05, half + 0.05)
        ax.set_ylim(-half - 0.05, half + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"t={k / FPS:5.1f}s", fontsize=10)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        writer.append_data(buf)
    plt.close(fig)
    writer.close()


def save_trajectory_figure(positions, goal_touch_indices, start_pos, env_size_m, step, seed, path):
    half = env_size_m / 2
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_patch(plt.Rectangle((-half, -half), env_size_m, env_size_m,
                                facecolor="#3a4a6b", edgecolor="#c0392b", linewidth=6, zorder=0))

    bounds = [0] + goal_touch_indices + [len(positions) - 1]
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        seg = positions[a:b + 1]
        is_first = (i == 0)
        ax.plot(seg[:, 0], seg[:, 1],
                color="#888888" if is_first else "black",
                linestyle="--" if is_first else "-",
                lw=1.6, zorder=1,
                label="First trajectory" if is_first else
                ("Subsequent trajectories" if i == 1 else None))
        seg_start = positions[a]
        ax.scatter(*seg_start, color="white", s=90, zorder=3, edgecolor="k", linewidth=1.0)
        ax.annotate("S", seg_start, color="black", fontsize=9, fontweight="bold",
                     ha="center", va="center", zorder=4)

    if goal_touch_indices:
        goal_pos = positions[goal_touch_indices[0]]
        ax.scatter(*goal_pos, color="gold", s=180, zorder=3, edgecolor="k", linewidth=1.2, marker="*")
        ax.annotate("Goal", goal_pos, color="white", fontsize=10, fontweight="bold",
                     xytext=(0, 12), textcoords="offset points", ha="center", zorder=4)
        title_suffix = f"{len(goal_touch_indices)}x goal reached, first at t={goal_touch_indices[0] / FPS:.1f}s"
    else:
        title_suffix = "goal not reached in 90s"

    ax.set_xlim(-half - 0.05, half + 0.05)
    ax.set_ylim(-half - 0.05, half + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"episode seed={seed}, step {step} ({title_suffix})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                         help="checkpoint_step*.pt path; omit to auto-pick the newest one")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    t_start = time.time()
    cfg = Config()

    if args.checkpoint:
        ckpt_path = args.checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        print(f"scanning {CHECKPOINT_DIR} for the newest checkpoint_step*.pt ...")
        ckpt_path, ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    step = ckpt["step"]
    print(f"using checkpoint {ckpt_path} (step {step})")

    out_dir = args.out_dir or f"results/rl_eval_step{step}"
    os.makedirs(out_dir, exist_ok=True)

    grid_network, vision_module, actor_critic, place, hd, has_rl_lstm, ego_vel_dim = \
        build_models_from_checkpoint(ckpt, cfg)

    env_cfg = cfg.rl.env
    level_dir = os.path.abspath(env_cfg.level_directory)
    env = deepmind_lab.Lab(
        env_cfg.level,
        ["RGB_INTERLEAVED", "VEL.TRANS", "VEL.ROT", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
        config={"width": str(VIDEO_SIZE), "height": str(VIDEO_SIZE), "levelDirectory": level_dir},
    )

    torch.manual_seed(0)
    for ep in range(args.episodes):
        t_ep = time.time()
        print(f"\n--- episode {ep} (seed={ep}) ---")
        frames, positions, goal_touch_indices, cumulative_reward = run_episode(
            env, ep, cfg, grid_network, vision_module, actor_critic, place, hd,
            has_rl_lstm, ego_vel_dim)
        t_run = time.time()
        print(f"  ran {len(frames)} frames ({len(frames) / FPS:.1f}s), "
              f"reward={cumulative_reward:.0f}, goal touches={len(goal_touch_indices)} "
              f"({t_run - t_ep:.1f}s)")

        fp_path = os.path.join(out_dir, f"firstperson_ep{ep}.mp4")
        save_firstperson_video(frames, fp_path)
        t_fp = time.time()
        print(f"  saved {fp_path} ({t_fp - t_run:.1f}s)")

        td_path = os.path.join(out_dir, f"topdown_ep{ep}.mp4")
        save_topdown_video(positions, goal_touch_indices, env_cfg.env_size_m, td_path)
        t_td = time.time()
        print(f"  saved {td_path} ({t_td - t_fp:.1f}s)")

        traj_path = os.path.join(out_dir, f"trajectory_ep{ep}.png")
        save_trajectory_figure(positions, goal_touch_indices, positions[0], env_cfg.env_size_m,
                                step, ep, traj_path)
        print(f"  saved {traj_path} ({time.time() - t_td:.1f}s)")

    env.close()
    print(f"\nall {args.episodes} episodes done in {time.time() - t_start:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
