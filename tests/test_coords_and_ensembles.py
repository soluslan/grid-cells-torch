import numpy as np
import pytest
import torch

from config import Config
from grid_cells.ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from grid_cells.rl_a3c import world_to_meters


@pytest.mark.parametrize("world, metres", [(600.0, 0.0), (1100.0, 1.25), (100.0, -1.25),
                                           (850.0, 0.625)])
def test_world_to_meters(world, metres):
    env_cfg = Config().rl.env
    out = world_to_meters(torch.tensor([world, world]), env_cfg)
    assert out.tolist() == pytest.approx([metres, metres])


def test_square_arena_cells_map_into_the_arena():
    env_cfg = Config().rl.env
    cell_origins = torch.tensor([[150.0 + 100 * i, 150.0 + 100 * i] for i in range(10)])
    m = world_to_meters(cell_origins, env_cfg)
    assert m.min() > -1.25 and m.max() < 1.25


def test_place_posterior_peaks_at_each_centre():
    pc = PlaceCellEnsemble(64, stdev=0.1, pos_min=-1.1, pos_max=1.1, seed=0)
    probs = pc.posterior(pc.means.unsqueeze(0))[0]
    assert torch.equal(probs.argmax(-1), torch.arange(64))
    assert torch.allclose(probs.sum(-1), torch.ones(64))


def test_place_decoders_on_known_inputs():
    pc = PlaceCellEnsemble(4, stdev=0.1, seed=0)
    pc.means.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    probs = torch.tensor([[0.1, 0.6, 0.2, 0.1]])
    assert pc.decode_position(probs, "argmax").tolist() == [[1.0, 0.0]]
    assert pc.decode_position(probs, "weighted_mean")[0].tolist() == pytest.approx([0.7, 0.3])
    assert pc.decode_position(probs, "top2")[0].tolist() == pytest.approx([0.75, 0.25])


def test_head_direction_posterior_peaks_at_centres():
    hd = HeadDirectionCellEnsemble(12, concentration=20.0, seed=0)
    probs = hd.posterior(hd.means.unsqueeze(-1))
    assert torch.equal(probs.argmax(-1), torch.arange(12))
