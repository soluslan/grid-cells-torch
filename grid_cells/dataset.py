import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FIELDS = ("init_pos", "init_hd", "ego_vel", "target_pos", "target_hd")


def check_shards(paths: list[str], n_traj: int = 10_000, n_steps: int = 100) -> None:
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


class GridCellsDataset(Dataset):
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
    while True:
        yield from loader
