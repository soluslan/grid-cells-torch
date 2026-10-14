import numpy as np
import torch
import torch.multiprocessing as mp

from grid_cells.dataset import FIELDS

N_STRIPE_LOCKS = 8


class _RingBuffer:
    def __init__(self, capacity: int, fields: dict):
        self.capacity = capacity
        self.fields = fields
        self._data = {
            name: torch.zeros((capacity, *shape), dtype=dtype).share_memory_()
            for name, (shape, dtype) in fields.items()
        }
        self._cursor = mp.Value('l', 0)
        self._count = mp.Value('l', 0)
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
        n = len(self)
        assert n > 0, "cannot sample from an empty buffer"
        idx = torch.randint(0, n, (batch_size,))
        return {name: self._data[name][idx] for name in self.fields}

    def state_dict(self) -> dict:
        with self._cursor.get_lock():
            cursor, count = self._cursor.value, self._count.value
            data = {name: t.clone() for name, t in self._data.items()}
        return {"data": data, "cursor": cursor, "count": count}

    def load_state_dict(self, state: dict) -> None:
        for name, tensor in state["data"].items():
            assert tensor.shape == self._data[name].shape, (
                f"{name}: checkpoint shape {tuple(tensor.shape)} != this buffer's "
                f"{tuple(self._data[name].shape)} -- capacity/seq_len config changed since "
                f"this checkpoint was saved")
            self._data[name].copy_(tensor)
        with self._cursor.get_lock():
            self._cursor.value = state["cursor"]
        with self._count.get_lock():
            self._count.value = state["count"]


class FrameReplayBuffer(_RingBuffer):
    def __init__(self, capacity: int = 200_000, image_size: int = 64):
        super().__init__(capacity, fields={
            "image": ((3, image_size, image_size), torch.float32),
            "pos": ((2,), torch.float32),
            "hd": ((1,), torch.float32),
        })

    def write_step(self, image: torch.Tensor, pos: torch.Tensor, hd: torch.Tensor) -> None:
        self.write(image=image, pos=pos, hd=hd)


class SequenceReplayBuffer(_RingBuffer):
    def __init__(self, capacity: int = 2_000, seq_len: int = 100, image_size: int = 64,
                ego_vel_dim: int = 3):
        self.seq_len = seq_len
        super().__init__(capacity, fields={
            "init_pos": ((2,), torch.float32),
            "init_hd": ((1,), torch.float32),
            "ego_vel": ((seq_len, ego_vel_dim), torch.float32),
            "target_pos": ((seq_len, 2), torch.float32),
            "target_hd": ((seq_len, 1), torch.float32),
            "image": ((seq_len, 3, image_size, image_size), torch.float32),
        })
        assert set(FIELDS) <= set(self.fields), (
            f"SequenceReplayBuffer fields {set(self.fields)} must be a superset of "
            f"dataset.FIELDS {set(FIELDS)} for train.train_step() to consume sample() output")


class SequenceAccumulator:
    def __init__(self, buffer: SequenceReplayBuffer):
        self.buffer = buffer
        self.reset(init_pos=None, init_hd=None)

    def reset(self, init_pos, init_hd) -> None:
        self._init_pos = init_pos
        self._init_hd = init_hd
        self._ego_vel: list = []
        self._target_pos: list = []
        self._target_hd: list = []
        self._image: list = []

    def add_step(self, ego_vel, target_pos, target_hd, image) -> None:
        assert self._init_pos is not None, "reset() must be called (with the post-teleport " \
            "pose) before the first add_step() of a segment"
        self._ego_vel.append(ego_vel)
        self._target_pos.append(target_pos)
        self._target_hd.append(target_hd)
        self._image.append(image)
        if len(self._ego_vel) == self.buffer.seq_len:
            self.buffer.write(
                init_pos=self._init_pos, init_hd=self._init_hd,
                ego_vel=torch.stack(self._ego_vel),
                target_pos=torch.stack(self._target_pos),
                target_hd=torch.stack(self._target_hd),
                image=torch.stack(self._image),
            )
            self.reset(init_pos=target_pos, init_hd=target_hd)


class TimeMajorReplay:
    """Replay for the synchronous learner: all envs step together, so experience is stored as a
    (time, env) ring of per-step records with images kept as uint8. Vision samples single frames;
    the grid network samples windows of consecutive steps from one env that stay inside one
    episode (and may cross goal teleports)."""

    def __init__(self, capacity_steps: int, num_envs: int, image_shape=(3, 64, 64),
                 ego_vel_dim: int = 4, seed: int = 0):
        t, b = capacity_steps, num_envs
        self.capacity = t
        self.num_envs = b
        self.image = torch.zeros((t, b, *image_shape), dtype=torch.uint8)
        self.pos = torch.zeros((t, b, 2))
        self.hd = torch.zeros((t, b, 1))
        self.ego_vel = torch.zeros((t, b, ego_vel_dim))
        self.episode = torch.full((t, b), -1, dtype=torch.long)
        self.cursor = 0
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.count * self.num_envs

    def add(self, image, pos, hd, ego_vel, episode) -> None:
        i = self.cursor
        self.image[i] = torch.as_tensor(image)
        self.pos[i] = torch.as_tensor(pos)
        self.hd[i] = torch.as_tensor(hd)
        self.ego_vel[i] = torch.as_tensor(ego_vel)
        self.episode[i] = torch.as_tensor(episode)
        self.cursor = (i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def _physical(self, logical):
        start = (self.cursor - self.count) % self.capacity
        return (start + logical) % self.capacity

    def sample_frames(self, n: int) -> dict:
        t = self._physical(self.rng.integers(0, self.count, n))
        b = self.rng.integers(0, self.num_envs, n)
        return {"image": self.image[t, b], "pos": self.pos[t, b], "hd": self.hd[t, b]}

    def sample_sequences(self, n: int, length: int, max_tries: int = 10_000):
        if self.count < length:
            return None
        starts, envs = [], []
        for _ in range(max_tries):
            i0 = int(self.rng.integers(0, self.count - length + 1))
            b = int(self.rng.integers(0, self.num_envs))
            first, last = self._physical(i0), self._physical(i0 + length - 1)
            if self.episode[first, b] >= 0 and self.episode[first, b] == self.episode[last, b]:
                starts.append(i0)
                envs.append(b)
                if len(starts) == n:
                    break
        if len(starts) < n:
            return None
        t = self._physical(np.asarray(starts)[:, None] + np.arange(length)[None, :])
        b = np.asarray(envs)[:, None]
        return {"image": self.image[t, b], "pos": self.pos[t, b], "hd": self.hd[t, b],
                "ego_vel": self.ego_vel[t, b]}

    def state_dict(self) -> dict:
        return {"image": self.image, "pos": self.pos, "hd": self.hd, "ego_vel": self.ego_vel,
                "episode": self.episode, "cursor": self.cursor, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        for name in ("image", "pos", "hd", "ego_vel", "episode"):
            getattr(self, name).copy_(state[name])
        self.cursor, self.count = state["cursor"], state["count"]
