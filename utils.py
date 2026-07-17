"""Helpers for encoding raw position/heading into ensemble target/init
distributions. Ported 1:1 from google-deepmind/grid-cells' utils.py
(the plotting/gradient-clipping helpers there are not needed here --
see model.py/train.py for their PyTorch equivalents)."""

import torch

from ensembles import CellEnsemble


def encode_initial_conditions(init_pos: torch.Tensor, init_hd: torch.Tensor,
                               place_cell_ensembles: list[CellEnsemble],
                               head_direction_ensembles: list[CellEnsemble]) -> list[torch.Tensor]:
    """init_pos: [B,2], init_hd: [B,1] -> list of [B, ens.n_cells] tensors."""
    initial_conds = []
    for ens in place_cell_ensembles:
        initial_conds.append(ens.get_init(init_pos.unsqueeze(1)).squeeze(1))
    for ens in head_direction_ensembles:
        initial_conds.append(ens.get_init(init_hd.unsqueeze(1)).squeeze(1))
    return initial_conds


def encode_targets(target_pos: torch.Tensor, target_hd: torch.Tensor,
                    place_cell_ensembles: list[CellEnsemble],
                    head_direction_ensembles: list[CellEnsemble]) -> list[torch.Tensor]:
    """target_pos: [B,T,2], target_hd: [B,T,1] -> list of [B,T,ens.n_cells] tensors."""
    ensembles_targets = []
    for ens in place_cell_ensembles:
        ensembles_targets.append(ens.get_targets(target_pos))
    for ens in head_direction_ensembles:
        ensembles_targets.append(ens.get_targets(target_hd))
    return ensembles_targets
