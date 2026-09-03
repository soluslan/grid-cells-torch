import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


class CellEnsemble(nn.Module):
    def __init__(self, n_cells: int):
        super().__init__()
        self.n_cells = n_cells

    def unnor_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def posterior(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.unnor_logpdf(x), dim=-1)


class PlaceCellEnsemble(CellEnsemble):
    def __init__(self, n_cells: int, stdev: float = 0.35, pos_min: float = -5,
                 pos_max: float = 5, seed: int | None = None):
        super().__init__(n_cells)
        rs = np.random.RandomState(seed)
        means = rs.uniform(pos_min, pos_max, size=(n_cells, 2))
        variances = np.ones_like(means) * stdev ** 2
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("variances", torch.as_tensor(variances, dtype=torch.float32))

    def unnor_logpdf(self, trajs: torch.Tensor) -> torch.Tensor:
        diff = trajs.unsqueeze(-2) - self.means
        return -0.5 * (diff ** 2 / self.variances).sum(dim=-1)

    def decode_position(self, probs: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        if mode == "argmax":
            return self.means[probs.argmax(dim=-1)]
        if mode == "weighted_mean":
            return probs @ self.means
        if mode.startswith("top"):
            weights, idx = probs.topk(int(mode[3:]), dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return (weights.unsqueeze(-1) * self.means[idx]).sum(dim=-2)
        raise ValueError(f"unknown decoder mode {mode!r}")


class HeadDirectionCellEnsemble(CellEnsemble):
    def __init__(self, n_cells: int, concentration: float = 20.0, seed: int | None = None):
        super().__init__(n_cells)
        rs = np.random.RandomState(seed)
        means = rs.uniform(-np.pi, np.pi, size=(n_cells,))
        kappa = np.ones_like(means) * concentration
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("kappa", torch.as_tensor(kappa, dtype=torch.float32))

    def unnor_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.kappa * torch.cos(x - self.means)


def build_ensembles(cfg, device):
    half = cfg.task.env_size / 2.0
    place = [PlaceCellEnsemble(n, stdev=s, pos_min=-half, pos_max=half,
                               seed=cfg.task.neurons_seed).to(device)
             for n, s in zip(cfg.task.n_pc, cfg.task.pc_scale)]
    head = [HeadDirectionCellEnsemble(n, concentration=c,
                                      seed=cfg.task.neurons_seed).to(device)
            for n, c in zip(cfg.task.n_hdc, cfg.task.hdc_concentration)]
    return place, head


def build_rl_ensembles(cfg, device):
    half = cfg.rl.env.env_size_m / 2.0
    place = [PlaceCellEnsemble(n, stdev=s, pos_min=-half, pos_max=half,
                               seed=cfg.task.neurons_seed).to(device)
             for n, s in zip(cfg.task.n_pc, cfg.rl.pc_scale)]
    head = [HeadDirectionCellEnsemble(n, concentration=c,
                                      seed=cfg.task.neurons_seed).to(device)
            for n, c in zip(cfg.task.n_hdc, cfg.task.hdc_concentration)]
    return place, head


def encode_initial_conditions(init_pos, init_hd, place_ensembles, hd_ensembles):
    return ([e.posterior(init_pos.unsqueeze(1)).squeeze(1) for e in place_ensembles]
            + [e.posterior(init_hd.unsqueeze(1)).squeeze(1) for e in hd_ensembles])


def encode_targets(target_pos, target_hd, place_ensembles, hd_ensembles):
    return ([e.posterior(target_pos) for e in place_ensembles]
            + [e.posterior(target_hd) for e in hd_ensembles])
