"""One-time conversion: our CSV trajectory dataset -> .npz shards ready for
PyTorch training.

Our CSVs (data/square_room_100steps_2.2m_1000000/NNNN-of-0099.csv) store, per row:
    trajectory_id, step, t, pos_x, pos_y, vel_x, vel_y, rot_vel,
    head_direction_x, head_direction_y, distance_travelled
with exactly 100 contiguous rows (step=0..99, ascending) per trajectory_id,
and 10,000 trajectories per file.

The original grid-cells TFRecord dataset instead stored, per trajectory:
    init_pos [2], init_hd [1], ego_vel [100,3], target_pos [100,2], target_hd [100,1]
using CENTER-origin coordinates (coord_range=(-1.1,1.1) for a 2.2m room).

This script derives the latter from the former. See README.md for the field
mapping and the reasoning behind the ego_vel reconstruction.

ego_vel's rotation component is `sin(dtheta), cos(dtheta)`, where `dtheta`
is the angle between consecutive `head_direction` unit vectors (i.e. the
*net* heading change actually realized over one stored 0.15s step) -- NOT
`sin(rot_vel), cos(rot_vel)` using the CSV's raw `rot_vel` column, which was
tried and empirically failed despite being closer to the paper's literal
per-step spec `[v_t, sin(phi_dot_t), cos(phi_dot_t)]` (Methods). Three
from-scratch training runs (raw `rot_vel`; clipped to +-3*sigma_phi; rescaled
by the coarse step and clipped to +-pi) all plateaued around loss 7.0-7.1,
while `dtheta` reaches ~4.9 over the same 40 epochs -- confirmed not a
dataset-seed confound (dtheta on the same reseeded CSVs also reaches ~4.9).
Root cause: `rot_vel` is an *instantaneous rate* sampled at a single fine
(0.02s) instant within each 0.15s coarse step (which contains ~7.5 fine
sub-steps), so it only reflects whatever was happening at that one moment
and misses the rest of the step -- no amount of clipping/rescaling that
single sample fixes it, since the problem is which quantity is measured,
not its scale. `dtheta` instead directly measures the *net* change across
the whole step, which is what's actually needed here.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

COLUMNS = [
    "trajectory_id", "step", "t", "pos_x", "pos_y", "vel_x", "vel_y",
    "rot_vel", "head_direction_x", "head_direction_y", "distance_travelled",
]


def load_csv_shard(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, usecols=COLUMNS)


def csv_shard_to_arrays(df: pd.DataFrame, n_traj: int, n_steps: int,
                         env_size: float = 2.2) -> dict:
    """Reshape one shard's flat CSV rows into the 5 model-ready arrays.

    Assumes rows are already ordered as n_traj contiguous blocks of n_steps
    rows each, step ascending within a block (true for our generator's
    output -- verified by the asserts below, which fail loudly otherwise).
    """
    n_rows = n_traj * n_steps
    assert len(df) == n_rows, f"expected {n_rows} rows, got {len(df)}"

    traj_id = df["trajectory_id"].to_numpy().reshape(n_traj, n_steps)
    step = df["step"].to_numpy().reshape(n_traj, n_steps)
    assert np.all(traj_id == traj_id[:, :1]), "trajectory_id not constant within a 100-row block"
    assert np.all(step == np.arange(n_steps)[None, :]), "step is not 0..99 ascending within a block"

    pos_x = df["pos_x"].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)
    pos_y = df["pos_y"].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)
    vel_x = df["vel_x"].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)
    vel_y = df["vel_y"].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)
    hd_x = df["head_direction_x"].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)
    hd_y = df["head_direction_y"].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)

    hd_norm = np.sqrt(hd_x ** 2 + hd_y ** 2)
    assert np.allclose(hd_norm, 1.0, atol=1e-3), "head_direction is not a unit vector"

    # 1. center coordinates: corner-origin [0, env_size] -> center-origin [-env_size/2, env_size/2]
    pos_cx = pos_x - env_size / 2.0
    pos_cy = pos_y - env_size / 2.0
    pos_centered = np.stack([pos_cx, pos_cy], axis=-1)  # [N,T,2]

    # 2. heading angle from the unit vector
    theta = np.arctan2(hd_y, hd_x)  # [N,T]

    # 3-6. init/target pos/hd
    init_pos = pos_centered[:, 0, :]  # [N,2]
    init_hd = theta[:, 0:1]  # [N,1]
    target_pos = pos_centered  # [N,T,2]
    target_hd = theta[..., None]  # [N,T,1]

    # 7. speed
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)  # [N,T]

    # 8. dtheta: net heading change over this 0.15s step, from consecutive
    # head_direction unit vectors (see module docstring for why this beats
    # rot_vel). Self-referential at t=0 so dtheta[:,0] == 0 exactly.
    prev_hd_x = np.concatenate([hd_x[:, 0:1], hd_x[:, :-1]], axis=1)
    prev_hd_y = np.concatenate([hd_y[:, 0:1], hd_y[:, :-1]], axis=1)
    cross = prev_hd_x * hd_y - prev_hd_y * hd_x
    dot = prev_hd_x * hd_x + prev_hd_y * hd_y
    dtheta = np.arctan2(cross, dot)  # [N,T]

    # 9. ego_vel
    ego_vel = np.stack([speed, np.sin(dtheta), np.cos(dtheta)], axis=-1)  # [N,T,3]

    return {
        "init_pos": init_pos.astype(np.float32),
        "init_hd": init_hd.astype(np.float32),
        "ego_vel": ego_vel.astype(np.float32),
        "target_pos": target_pos.astype(np.float32),
        "target_hd": target_hd.astype(np.float32),
    }


def convert_shard(csv_path: str, out_path: str, n_traj: int = 10_000,
                   n_steps: int = 100, env_size: float = 2.2) -> None:
    df = load_csv_shard(csv_path)
    arrays = csv_shard_to_arrays(df, n_traj=n_traj, n_steps=n_steps, env_size=env_size)
    np.savez(out_path, **arrays)


def convert_all(csv_dir: str, out_dir: str, shard_indices: list[int] | None = None,
                 n_traj: int = 10_000, n_steps: int = 100, env_size: float = 2.2) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if shard_indices is not None:
        csv_paths = [p for p in csv_paths if any(f"{i:04d}-of-" in os.path.basename(p) for i in shard_indices)]

    out_paths = []
    for csv_path in csv_paths:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(out_dir, basename + ".npz")
        convert_shard(csv_path, out_path, n_traj=n_traj, n_steps=n_steps, env_size=env_size)
        out_paths.append(out_path)
        print(f"converted {csv_path} -> {out_path}")
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", default="data/square_room_100steps_2.2m_1000000")
    parser.add_argument("--out_dir", default="data/shards")
    parser.add_argument("--shard_indices", type=int, nargs="*", default=None,
                         help="convert only these shard indices (e.g. 0 1); default: all")
    args = parser.parse_args()
    convert_all(args.csv_dir, args.out_dir, shard_indices=args.shard_indices)
