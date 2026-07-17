"""Place cell / head-direction cell ensembles that provide training targets.

Ported from google-deepmind/grid-cells' ensembles.py (TF1 + Sonnet v1) to
PyTorch. Only the "softmax" soft_targets/soft_init branch is implemented --
that's the only mode train.py's flag defaults ever actually used. The other
original modes ("voronoi", "sample", "normalized", "zeros") are omitted rather
than half-ported.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellEnsemble(nn.Module):
    """Base class. Subclasses implement unnor_logpdf(x) -> [B,T,n_cells]."""

    def __init__(self, n_cells: int):
        super().__init__()
        self.n_cells = n_cells

    def unnor_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_posterior(self, x: torch.Tensor) -> torch.Tensor:
        logp = self.unnor_logpdf(x)
        return logp - torch.logsumexp(logp, dim=-1, keepdim=True)

    def get_targets(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,D] -> soft target distribution [B,T,n_cells]."""
        return F.softmax(self.log_posterior(x), dim=-1)

    def get_init(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,1,D] -> soft init distribution [B,1,n_cells]."""
        return F.softmax(self.log_posterior(x), dim=-1)

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Soft cross-entropy: predictions are logits, targets are a distribution.

        predictions, targets: [B,T,n_cells] -> per-(batch,time) loss [B,T]
        """
        return -(targets * F.log_softmax(predictions, dim=-1)).sum(dim=-1)


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
        # trajs: [B,T,2] -> diff: [B,T,n_cells,2]
        diff = trajs.unsqueeze(-2) - self.means
        return -0.5 * (diff ** 2 / self.variances).sum(dim=-1)


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
        # x: [B,T,1] -> broadcasts against means [n_cells] -> [B,T,n_cells]
        return self.kappa * torch.cos(x - self.means)
