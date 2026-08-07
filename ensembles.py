"""Place/head-direction cell ensembles that supply the training targets, and
the helpers that encode raw position/heading into them.

Ports google-deepmind/grid-cells' ensembles.py and utils.py. Only the
"softmax" target/init mode is carried over -- the only one the original's
flag defaults ever selected.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a soft target distribution: [B,T,N] -> [B,T]."""
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)


class CellEnsemble(nn.Module):
    """Base class; subclasses implement unnor_logpdf(x) -> [B,T,n_cells]."""

    def __init__(self, n_cells: int):
        super().__init__()
        self.n_cells = n_cells

    def unnor_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def posterior(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D] -> posterior over cells [B,T,n_cells].

        Used for both targets and the t=0 LSTM initialisation, which the
        original kept as separate methods only because they could select
        different (never-used) modes.
        """
        return F.softmax(self.unnor_logpdf(x), dim=-1)


class PlaceCellEnsemble(CellEnsemble):
    """Distribution over n_cells randomly-placed, fixed-covariance 2D Gaussians."""

    def __init__(self, n_cells: int, stdev: float = 0.35, pos_min: float = -5,
                 pos_max: float = 5, seed: int | None = None):
        super().__init__(n_cells)
        rs = np.random.RandomState(seed)
        means = rs.uniform(pos_min, pos_max, size=(n_cells, 2))
        variances = np.ones_like(means) * stdev ** 2
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("variances", torch.as_tensor(variances, dtype=torch.float32))

    def unnor_logpdf(self, trajs: torch.Tensor) -> torch.Tensor:
        diff = trajs.unsqueeze(-2) - self.means  # [B,T,2] -> [B,T,n_cells,2]
        return -0.5 * (diff ** 2 / self.variances).sum(dim=-1)

    def decode_position(self, probs: torch.Tensor, mode: str = "argmax") -> torch.Tensor:
        """Read a position back out of a place-cell distribution: [..,N] -> [..,2].

        The paper decodes self-location from the place cells (Fig. 1b) without
        saying how, so all three plausible readouts are offered:

        - "argmax": the most likely cell's centre. Unbiased, but quantised to
          the N cell centres, so it cannot beat the mean spacing between them.
        - "weighted_mean": the posterior mean. Continuous, but a broad
          posterior averages towards the middle of the arena, which inflates
          error near the walls.
        - "topK" (e.g. "top3"): posterior mean over the K likeliest cells only
          -- no quantisation floor, and no pull from the far tail.
        """
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
    """Distribution over n_cells randomly-oriented, fixed-concentration Von Mises."""

    def __init__(self, n_cells: int, concentration: float = 20.0, seed: int | None = None):
        super().__init__(n_cells)
        rs = np.random.RandomState(seed)
        means = rs.uniform(-np.pi, np.pi, size=(n_cells,))
        kappa = np.ones_like(means) * concentration
        self.register_buffer("means", torch.as_tensor(means, dtype=torch.float32))
        self.register_buffer("kappa", torch.as_tensor(kappa, dtype=torch.float32))

    def unnor_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.kappa * torch.cos(x - self.means)  # [B,T,1] -> [B,T,n_cells]


def build_ensembles(cfg, device):
    """-> (place_cell_ensembles, head_direction_ensembles) for a Config."""
    half = cfg.task.env_size / 2.0
    place = [PlaceCellEnsemble(n, stdev=s, pos_min=-half, pos_max=half,
                               seed=cfg.task.neurons_seed).to(device)
             for n, s in zip(cfg.task.n_pc, cfg.task.pc_scale)]
    head = [HeadDirectionCellEnsemble(n, concentration=c,
                                      seed=cfg.task.neurons_seed).to(device)
            for n, c in zip(cfg.task.n_hdc, cfg.task.hdc_concentration)]
    return place, head


def encode_initial_conditions(init_pos, init_hd, place_ensembles, hd_ensembles):
    """init_pos [B,2], init_hd [B,1] -> list of [B, n_cells]."""
    return ([e.posterior(init_pos.unsqueeze(1)).squeeze(1) for e in place_ensembles]
            + [e.posterior(init_hd.unsqueeze(1)).squeeze(1) for e in hd_ensembles])


def encode_targets(target_pos, target_hd, place_ensembles, hd_ensembles):
    """target_pos [B,T,2], target_hd [B,T,1] -> list of [B,T,n_cells]."""
    return ([e.posterior(target_pos) for e in place_ensembles]
            + [e.posterior(target_hd) for e in hd_ensembles])
