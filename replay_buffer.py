import torch
import torch.multiprocessing as mp

from dataset import FIELDS

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
