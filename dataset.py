"""PyTorch Dataset/DataLoader over the .npz shards produced by
scripts/convert_csv_to_shards.py.

Replaces the original repo's dataset_reader.py (a TF1 tf.RandomShuffleQueue
fed by tf.train.string_input_producer over TFRecord files -- both removed in
TF2). All shards together are ~2.4GB, small enough to load eagerly into RAM
as a handful of big contiguous arrays; this gives O(1) random-access
__getitem__ and full compatibility with DataLoader(shuffle=True)'s
RandomSampler, with no per-item I/O and no shard-shuffling bookkeeping to get
wrong. If the dataset grows far past available RAM later, the natural next
step is per-array .npy files opened with np.load(mmap_mode='r') -- not needed
now.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FIELDS = ("init_pos", "init_hd", "ego_vel", "target_pos", "target_hd")


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
        paths = [p for p in paths if any(f"{i:04d}-of-" in os.path.basename(p) for i in shard_indices)]
    dataset = GridCellsDataset(paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, drop_last=True)


def infinite_loader(loader: DataLoader):
    """Yields batches forever, reshuffling every time the loader is exhausted
    (DataLoader(shuffle=True) draws a fresh RandomSampler permutation on each
    new iterator) -- matches the original's queue-based reader, which has no
    real "epoch" boundary either, just continuous shuffled reads."""
    while True:
        yield from loader
