"""Everything between the trajectory generator's CSVs and a batch the model
can consume: CSV -> .npz shards -> DataLoader.

Replaces the original's dataset_reader.py (a TF1 queue-based TFRecord reader,
removed in TF2) and absorbs what used to be scripts/convert_csv_to_shards.py.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

FIELDS = ("init_pos", "init_hd", "ego_vel", "target_pos", "target_hd")

ENV_SIZE = 2.2  # metres; the paper's L (Supplementary Table 1)


# --------------------------------------------------------------------------
# CSV -> .npz shards
#
# Our CSVs (data/square_room_100steps_2.2m_1000000/NNNN-of-0099.csv) store, per
# row: trajectory_id, step, t, pos_x, pos_y, vel_x, vel_y, rot_vel,
# head_direction_x, head_direction_y, distance_travelled -- with exactly 100
# contiguous rows (step 0..99) per trajectory and 10,000 trajectories per file.
#
# The original TFRecord dataset instead stored, per trajectory, the five FIELDS
# above in CENTER-origin coordinates. This derives the latter from the former.
# `ego_vel`'s rotation component uses `dtheta`, the net heading change over a
# stored step, not the CSV's raw `rot_vel` rate -- see README "Why dtheta, not
# rot_vel" for the evidence behind that choice.
# --------------------------------------------------------------------------

COLUMNS = [
    "trajectory_id", "step", "t", "pos_x", "pos_y", "vel_x", "vel_y",
    "rot_vel", "head_direction_x", "head_direction_y", "distance_travelled",
]


def csv_shard_to_arrays(df: pd.DataFrame, n_traj: int, n_steps: int,
                        env_size: float = ENV_SIZE) -> dict:
    """Reshape one shard's flat CSV rows into the 5 model-ready arrays.

    Assumes rows are already ordered as n_traj contiguous blocks of n_steps
    rows each, step ascending within a block (true for our generator's output
    -- verified by the asserts below, which fail loudly otherwise).
    """
    n_rows = n_traj * n_steps
    assert len(df) == n_rows, f"expected {n_rows} rows, got {len(df)}"

    traj_id = df["trajectory_id"].to_numpy().reshape(n_traj, n_steps)
    step = df["step"].to_numpy().reshape(n_traj, n_steps)
    assert np.all(traj_id == traj_id[:, :1]), "trajectory_id not constant within a block"
    assert np.all(step == np.arange(n_steps)[None, :]), "step is not 0..99 ascending"

    def col(name):
        return df[name].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)

    pos_x, pos_y = col("pos_x"), col("pos_y")
    vel_x, vel_y = col("vel_x"), col("vel_y")
    hd_x, hd_y = col("head_direction_x"), col("head_direction_y")
    assert np.allclose(np.sqrt(hd_x ** 2 + hd_y ** 2), 1.0, atol=1e-3), \
        "head_direction is not a unit vector"

    # Corner-origin [0, env_size] -> center-origin [-env_size/2, env_size/2].
    pos_centered = np.stack([pos_x - env_size / 2.0, pos_y - env_size / 2.0], axis=-1)
    theta = np.arctan2(hd_y, hd_x)  # [N,T]
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)

    # dtheta: net heading change over this 0.15s step, from consecutive
    # head_direction unit vectors. Self-referential at t=0 so dtheta[:,0] == 0.
    prev_hd_x = np.concatenate([hd_x[:, 0:1], hd_x[:, :-1]], axis=1)
    prev_hd_y = np.concatenate([hd_y[:, 0:1], hd_y[:, :-1]], axis=1)
    dtheta = np.arctan2(prev_hd_x * hd_y - prev_hd_y * hd_x,
                        prev_hd_x * hd_x + prev_hd_y * hd_y)

    return {
        "init_pos": pos_centered[:, 0, :].astype(np.float32),
        "init_hd": theta[:, 0:1].astype(np.float32),
        "ego_vel": np.stack([speed, np.sin(dtheta), np.cos(dtheta)],
                            axis=-1).astype(np.float32),
        "target_pos": pos_centered.astype(np.float32),
        "target_hd": theta[..., None].astype(np.float32),
    }


def convert_all(csv_dir: str, out_dir: str, shard_indices: list[int] | None = None,
                n_traj: int = 10_000, n_steps: int = 100, env_size: float = ENV_SIZE,
                verbose: bool = True) -> list[str]:
    """Convert every CSV in csv_dir (or just `shard_indices`) to .npz shards."""
    os.makedirs(out_dir, exist_ok=True)
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if shard_indices is not None:
        csv_paths = [p for p in csv_paths
                     if any(f"{i:04d}-of-" in os.path.basename(p) for i in shard_indices)]

    out_paths = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=COLUMNS)
        arrays = csv_shard_to_arrays(df, n_traj=n_traj, n_steps=n_steps,
                                     env_size=env_size)
        out_path = os.path.join(
            out_dir, os.path.splitext(os.path.basename(csv_path))[0] + ".npz")
        np.savez(out_path, **arrays)
        out_paths.append(out_path)
        if verbose:
            print(f"converted {csv_path} -> {out_path}")
    return out_paths


def check_shards(paths: list[str], n_traj: int = 10_000, n_steps: int = 100) -> None:
    """Assert every shard has the shapes, dtype and finiteness the model expects."""
    expected = {"init_pos": (n_traj, 2), "init_hd": (n_traj, 1),
                "ego_vel": (n_traj, n_steps, 3), "target_pos": (n_traj, n_steps, 2),
                "target_hd": (n_traj, n_steps, 1)}
    for path in paths:
        with np.load(path) as npz:
            for k, shape in expected.items():
                arr = npz[k]
                assert arr.shape == shape, f"{path}:{k} shape {arr.shape} != {shape}"
                assert arr.dtype == np.float32, f"{path}:{k} dtype {arr.dtype}"
                assert np.isfinite(arr).all(), f"{path}:{k} has non-finite values"


# --------------------------------------------------------------------------
# .npz shards -> DataLoader
# --------------------------------------------------------------------------

class GridCellsDataset(Dataset):
    """All shards eagerly in RAM (~2.4GB together). If that stops being
    affordable, per-array .npy with mmap_mode='r' is the next step."""

    def __init__(self, shard_paths: list[str]):
        assert len(shard_paths) > 0, "no shard paths given"
        arrays = {k: [] for k in FIELDS}
        for path in shard_paths:
            with np.load(path) as npz:
                for k in FIELDS:
                    arrays[k].append(npz[k])
        self.data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
        self._len = self.data["init_pos"].shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        return {k: torch.from_numpy(self.data[k][idx]) for k in FIELDS}


def build_dataloader(shard_dir: str, shard_indices: list[int] | None, batch_size: int,
                     shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    if shard_indices is not None:
        paths = [p for p in paths
                 if any(f"{i:04d}-of-" in os.path.basename(p) for i in shard_indices)]
    return DataLoader(GridCellsDataset(paths), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=True)


def infinite_loader(loader: DataLoader):
    """Yields batches forever, reshuffling on each pass -- matching the
    original's queue-based reader, which had no epoch boundary either."""
    while True:
        yield from loader
