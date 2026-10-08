import numpy as np
import torch

from config import Config
from golden_cases import _synthetic_batch
from grid_cells.ensembles import (build_ensembles, encode_initial_conditions, encode_targets,
                                  soft_cross_entropy)
from grid_cells.training import build_model, build_optimizer, train_step


def _setup():
    torch.manual_seed(0)
    cfg = Config()
    device = torch.device("cpu")
    place, hd = build_ensembles(cfg, device)
    model = build_model(cfg, place + hd, device)
    model.train()
    return cfg, device, place, hd, model


def test_shapes_and_initial_loss():
    cfg, device, place, hd, model = _setup()
    b = cfg.train.minibatch_size
    batch = _synthetic_batch(np.random.default_rng(0), b=b)
    init = encode_initial_conditions(batch["init_pos"], batch["init_hd"], place, hd)
    targets = encode_targets(batch["target_pos"], batch["target_hd"], place, hd)
    out = model(init, batch["ego_vel"])
    assert out.logits[0].shape == (b, 100, cfg.task.n_pc[0])
    assert out.logits[1].shape == (b, 100, cfg.task.n_hdc[0])
    assert out.bottleneck.shape == (b, 100, cfg.model.nh_bottleneck)
    assert out.lstm_output.shape == (b, 100, cfg.model.nh_lstm)
    loss = sum(soft_cross_entropy(lg, t) for lg, t in zip(out.logits, targets)).mean()
    ballpark = np.log(cfg.task.n_pc[0]) + np.log(cfg.task.n_hdc[0])
    assert torch.isfinite(loss)
    assert abs(loss.item() - ballpark) < 0.5 * ballpark

    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_train_steps_stay_finite():
    cfg, device, place, hd, model = _setup()
    opt = build_optimizer(model, cfg)
    rng = np.random.default_rng(1)
    losses = [train_step(model, place, hd, _synthetic_batch(rng), cfg, device, opt)
              for _ in range(30)]
    assert np.all(np.isfinite(losses))
