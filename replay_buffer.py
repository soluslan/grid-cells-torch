"""Shared-memory replay buffers for the RL agent (RL-agent roadmap plan, M4).

Two buffers, not one, because the paper's "replay buffer" isn't one thing: A3C itself never
replays actor experience (each actor's own rollout feeds its own gradient step once and is
discarded), so the buffer exists only because the vision module and grid network are trained by
two *separate* supervised-style learner processes that need to sample random minibatches from
recent experience. The paper's literal 2x10^7-step buffer size (~245GB of raw frames) is almost
certainly an over-literal reading -- see the roadmap plan's "Engineering design" note -- so
these default to bounded sizes picked for wall-clock throughput instead.

- `FrameReplayBuffer`: single frames + ground-truth pose, for the vision learner (Methods:
  minibatch of 32 single frames).
- `SequenceReplayBuffer`: complete 100-step trajectory segments in *exactly* dataset.py's shard
  schema (`FIELDS = init_pos, init_hd, ego_vel, target_pos, target_hd`), for the grid-network
  learner (Methods: minibatch of 10 sequences of 100 steps) -- built this way specifically so
  `train.train_step()`/`build_optimizer()` can consume it completely unchanged, the same way
  they consume `dataset.py`'s shard `DataLoader`. A segment is only ever written as a whole,
  complete unit by the actor that collected it (see `SequenceAccumulator`), starting right after
  a teleport/episode reset -- sampling arbitrary 100-step windows from a flat per-step ring
  buffer would let a sampled "trajectory" straddle a teleport and be physically meaningless.

Both use fixed-capacity ring buffers backed by `torch.Tensor.share_memory_()` so every actor
process and both learner processes see the same underlying memory, with a handful of striped
locks (not one global lock) so 32 concurrent actor writers don't serialize against each other or
against the 2 learners' readers any more than necessary.
"""

import torch
import torch.multiprocessing as mp

from dataset import FIELDS  # ("init_pos", "init_hd", "ego_vel", "target_pos", "target_hd")

N_STRIPE_LOCKS = 8


class _RingBuffer:
    """Shared base: preallocated tensors, a shared write cursor/count, striped locks.

    Subclasses declare their own set of named fields (shape per item) and get write()/sample()
    for free. Not meant to be used directly.
    """

    def __init__(self, capacity: int, fields: dict):
        """fields: {name: (shape_tuple, dtype)} -- shape is per-item, no leading batch dim."""
        self.capacity = capacity
        self.fields = fields
        self._data = {
            name: torch.zeros((capacity, *shape), dtype=dtype).share_memory_()
            for name, (shape, dtype) in fields.items()
        }
        self._cursor = mp.Value('l', 0)  # next write index, mod capacity
        self._count = mp.Value('l', 0)   # number of valid (ever-written) slots, capped at capacity
        self._locks = [mp.Lock() for _ in range(N_STRIPE_LOCKS)]

    def write(self, **item: torch.Tensor) -> None:
        assert set(item) == set(self.fields), (set(item), set(self.fields))
        with self._cursor.get_lock():
            idx = self._cursor.value
            self._cursor.value = (idx + 1) % self.capacity
            if self._count.value < self.capacity:
                self._count.value += 1
        with self._locks[idx % N_STRIPE_LOCKS]:
            for name, value in item.items():
                self._data[name][idx] = value

    def __len__(self) -> int:
        return self._count.value

    def sample(self, batch_size: int) -> dict:
        """Returns {name: [batch_size, *shape]} sampled with replacement (matches the paper's
        random-minibatch-from-buffer description; no notion of epochs here)."""
        n = len(self)
        assert n > 0, "cannot sample from an empty buffer"
        idx = torch.randint(0, n, (batch_size,))
        # No per-row locking on read: a torn read (a write landing mid-copy) is, at worst, one
        # stale/mixed sample in a random minibatch out of millions of training steps -- the same
        # eventually-consistent tradeoff the roadmap plan's "Engineering design" note makes
        # explicit for this reason (avoiding it would serialize readers against every writer).
        return {name: self._data[name][idx] for name in self.fields}


class FrameReplayBuffer(_RingBuffer):
    """Single (frame, pose) pairs for the vision-learner thread."""

    def __init__(self, capacity: int = 200_000, image_size: int = 64):
        super().__init__(capacity, fields={
            "image": ((3, image_size, image_size), torch.float32),  # in [-1,1]
            "pos": ((2,), torch.float32),
            "hd": ((1,), torch.float32),
        })

    def write_step(self, image: torch.Tensor, pos: torch.Tensor, hd: torch.Tensor) -> None:
        self.write(image=image, pos=pos, hd=hd)


class SequenceReplayBuffer(_RingBuffer):
    """Complete 100-step trajectory segments, dataset.py's shard schema, for the
    grid-network-learner thread."""

    def __init__(self, capacity: int = 50_000, seq_len: int = 100):
        self.seq_len = seq_len
        super().__init__(capacity, fields={
            "init_pos": ((2,), torch.float32),
            "init_hd": ((1,), torch.float32),
            "ego_vel": ((seq_len, 3), torch.float32),
            "target_pos": ((seq_len, 2), torch.float32),
            "target_hd": ((seq_len, 1), torch.float32),
        })
        assert set(FIELDS) == set(self.fields), (
            f"SequenceReplayBuffer fields {set(self.fields)} must match dataset.FIELDS "
            f"{set(FIELDS)} for train.train_step() to consume sample() output unchanged")


class SequenceAccumulator:
    """Per-actor-process helper: buffers consecutive steps since the last reset/teleport and
    flushes a complete `seq_len`-step segment into a SequenceReplayBuffer the moment it has one
    -- never a partial segment, and always starting exactly at a reset/teleport boundary (the
    same invariant the supervised dataset's pre-cut shards have).
    """

    def __init__(self, buffer: SequenceReplayBuffer):
        self.buffer = buffer
        self.reset(init_pos=None, init_hd=None)

    def reset(self, init_pos, init_hd) -> None:
        self._init_pos = init_pos
        self._init_hd = init_hd
        self._ego_vel: list = []
        self._target_pos: list = []
        self._target_hd: list = []

    def add_step(self, ego_vel, target_pos, target_hd) -> None:
        assert self._init_pos is not None, "reset() must be called (with the post-teleport " \
            "pose) before the first add_step() of a segment"
        self._ego_vel.append(ego_vel)
        self._target_pos.append(target_pos)
        self._target_hd.append(target_hd)
        if len(self._ego_vel) == self.buffer.seq_len:
            self.buffer.write(
                init_pos=self._init_pos, init_hd=self._init_hd,
                ego_vel=torch.stack(self._ego_vel),
                target_pos=torch.stack(self._target_pos),
                target_hd=torch.stack(self._target_hd),
            )
            # Next segment starts fresh from the current pose -- NOT a reset/teleport, but the
            # ordinary "trajectory continues, just chunked into seq_len pieces" case; a real
            # teleport calls reset() directly instead, from the caller that detects it.
            self.reset(init_pos=target_pos, init_hd=target_hd)
