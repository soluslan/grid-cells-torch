"""Simulated rat trajectories (Raudies & Hasselmo 2012 motion model, Supplementary Table 1),
written straight to the .npz shards the supervised network trains on.

Each shard is generated from the legacy global NumPy RNG seeded with `seed + shard_index`, in
exactly the call order of the original notebooks/00_make_dataset.ipynb, so a shard generated
here is identical to the one the old CSV -> npz pipeline produced from the same seed.
"""
import json
import os
import subprocess
from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import interp1d

from grid_cells.dataset import FIELDS


@dataclass(frozen=True)
class MotionParams:
    env_size: float = 2.2
    duration_s: float = 15.0
    sim_dt: float = 0.02
    perimeter_m: float = 0.03
    speed_rayleigh_scale: float = 0.13
    rot_vel_std_deg: float = 330.0
    perimeter_speed_reduction: float = 0.25
    n_steps: int = 100

    @classmethod
    def from_config(cls, cfg) -> "MotionParams":
        d = cfg.data
        return cls(env_size=cfg.task.env_size, duration_s=d.duration_s, sim_dt=d.sim_dt,
                   perimeter_m=d.perimeter_m, speed_rayleigh_scale=d.speed_rayleigh_scale,
                   rot_vel_std_deg=d.rot_vel_std_deg,
                   perimeter_speed_reduction=d.perimeter_speed_reduction, n_steps=d.n_steps)

    @property
    def n_fine_steps(self) -> int:
        return round(self.duration_s / self.sim_dt)


def wall_zone(ang, pos, min_gap, max_gap) -> bool:
    if (0 <= ang <= np.pi / 2) and np.any(pos > max_gap):
        return True
    elif (np.pi / 2 <= ang <= np.pi) and (pos[0] < min_gap or pos[1] > max_gap):
        return True
    elif (-np.pi <= ang <= -np.pi / 2) and np.any(pos < min_gap):
        return True
    elif (-np.pi / 2 <= ang <= 0) and (pos[0] > max_gap or pos[1] < min_gap):
        return True
    return False


def wall_redirect(ang, pos, min_gap, max_gap) -> float:
    half_pi = np.pi / 2
    if 0 <= ang <= np.pi / 2:
        if pos[1] > max_gap: return -half_pi
        elif pos[0] > max_gap: return half_pi
    elif np.pi / 2 <= ang <= np.pi:
        if pos[1] > max_gap: return half_pi
        elif pos[0] < min_gap: return -half_pi
    elif -np.pi <= ang <= -np.pi / 2:
        if pos[1] < min_gap: return -half_pi
        elif pos[0] < min_gap: return half_pi
    else:
        if pos[1] < min_gap: return half_pi
        elif pos[0] > max_gap: return -half_pi
    return 0.0


def simulate_rh_trajectory(p: MotionParams):
    L, dt, n_steps = p.env_size, p.sim_dt, p.n_fine_steps
    sigma_phi = np.deg2rad(p.rot_vel_std_deg)
    min_gap, max_gap = p.perimeter_m, L - p.perimeter_m
    pos = np.zeros((n_steps + 1, 2))
    ang = np.zeros(n_steps + 1)
    vel = np.zeros(n_steps + 1)
    rot_vel = np.zeros(n_steps + 1)

    pos[0] = np.random.uniform(0, L, size=2)
    ang[0] = np.random.uniform(-np.pi, np.pi)
    prev_vel = 0.0

    for t in range(1, n_steps + 1):
        cur_pos, cur_ang = pos[t - 1], ang[t - 1]
        if wall_zone(cur_ang, cur_pos, min_gap, max_gap):
            rot_sample = np.random.normal(0, sigma_phi)
            d_angle = wall_redirect(cur_ang, cur_pos, min_gap, max_gap) + rot_sample * dt
            v = prev_vel * (1 - p.perimeter_speed_reduction)
        else:
            v = np.random.rayleigh(p.speed_rayleigh_scale)
            rot_sample = np.random.normal(0, sigma_phi)
            d_angle = rot_sample * dt

        new_pos = cur_pos + np.array([np.cos(cur_ang), np.sin(cur_ang)]) * v * dt
        new_ang = cur_ang + d_angle
        if abs(new_ang) >= np.pi:
            new_ang = -np.sign(new_ang) * (np.pi - (abs(new_ang) - np.pi))

        pos[t], ang[t], vel[t], rot_vel[t] = new_pos, new_ang, v, d_angle / dt
        prev_vel = v

    return pos, ang, vel, rot_vel


def resample_trajectory(p: MotionParams, pos_fine, ang_fine, vel_fine):
    t_fine = np.linspace(0, p.duration_s, p.n_fine_steps + 1)
    t_target = np.linspace(0, p.duration_s, p.n_steps, endpoint=False)
    hd_fine = np.column_stack([np.cos(ang_fine), np.sin(ang_fine)])
    vel_fine_xy = vel_fine[:, None] * hd_fine

    pos_t = interp1d(t_fine, pos_fine, axis=0, kind="cubic")(t_target)
    vel_t = interp1d(t_fine, vel_fine_xy, axis=0, kind="linear")(t_target)
    hd_t = interp1d(t_fine, hd_fine, axis=0, kind="cubic")(t_target)
    hd_t /= np.linalg.norm(hd_t, axis=1, keepdims=True)
    return pos_t, vel_t, hd_t


def trajectories_to_arrays(pos, vel, hd, env_size: float) -> dict:
    """pos, vel, hd: float64 arrays of shape (n_traj, n_steps, 2), corner-origin positions,
    allocentric velocity, unit heading vectors. Returns the model's training fields."""
    pos_x, pos_y = pos[..., 0].astype(np.float32), pos[..., 1].astype(np.float32)
    vel_x, vel_y = vel[..., 0].astype(np.float32), vel[..., 1].astype(np.float32)
    hd_x, hd_y = hd[..., 0].astype(np.float32), hd[..., 1].astype(np.float32)
    assert np.allclose(np.sqrt(hd_x ** 2 + hd_y ** 2), 1.0, atol=1e-3)

    pos_centered = np.stack([pos_x - env_size / 2.0, pos_y - env_size / 2.0], axis=-1)
    theta = np.arctan2(hd_y, hd_x)
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    prev_hd_x = np.concatenate([hd_x[:, 0:1], hd_x[:, :-1]], axis=1)
    prev_hd_y = np.concatenate([hd_y[:, 0:1], hd_y[:, :-1]], axis=1)
    dtheta = np.arctan2(prev_hd_x * hd_y - prev_hd_y * hd_x,
                        prev_hd_x * hd_x + prev_hd_y * hd_y)
    return {
        "init_pos": pos_centered[:, 0, :].astype(np.float32),
        "init_hd": theta[:, 0:1].astype(np.float32),
        "ego_vel": np.stack([speed, np.sin(dtheta), np.cos(dtheta)], axis=-1).astype(np.float32),
        "target_pos": pos_centered.astype(np.float32),
        "target_hd": theta[..., None].astype(np.float32),
    }


def generate_shard_arrays(shard_index: int, n_traj: int, p: MotionParams,
                          seed: int = 0) -> dict:
    np.random.seed(seed + shard_index)
    pos = np.empty((n_traj, p.n_steps, 2))
    vel = np.empty((n_traj, p.n_steps, 2))
    hd = np.empty((n_traj, p.n_steps, 2))
    for i in range(n_traj):
        pos_fine, ang_fine, vel_fine, _ = simulate_rh_trajectory(p)
        pos[i], vel[i], hd[i] = resample_trajectory(p, pos_fine, ang_fine, vel_fine)
    arrays = trajectories_to_arrays(pos, vel, hd, p.env_size)
    assert set(arrays) == set(FIELDS)
    return arrays


def shard_filename(shard_index: int, n_shards: int) -> str:
    return f"{shard_index:04d}-of-{n_shards - 1:04d}.npz"


def write_shard(args) -> str:
    shard_index, n_shards, n_traj, p, seed, shard_dir = args
    path = os.path.join(shard_dir, shard_filename(shard_index, n_shards))
    np.savez(path, **generate_shard_arrays(shard_index, n_traj, p, seed))
    return path


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=os.path.dirname(__file__)).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_metadata(path: str, p: MotionParams, n_shards: int, n_traj: int, seed: int,
                   shard_indices: list[int]) -> None:
    meta = {
        "motion": asdict(p),
        "n_shards": n_shards,
        "shard_indices": shard_indices,
        "trajectories_per_shard": n_traj,
        "seed": seed,
        "shard_seed_rule": "np.random.seed(seed + shard_index)",
        "fields": list(FIELDS),
        "git_commit": git_commit(),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
