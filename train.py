"""Supervised training loop for the grid cell network.

Ports google-deepmind/grid-cells' train.py. Grid scoring is not run inline
here as it was there -- this file trains and checkpoints only; scoring is
evaluate.py's job. Library only: run it from notebooks/02_train.ipynb.
"""

import os

import numpy as np
import torch

from config import Config
from dataset import build_dataloader, infinite_loader
from ensembles import (build_ensembles, encode_initial_conditions, encode_targets,
                       soft_cross_entropy)
from model import GridCellsRNN


def build_model(cfg: Config, target_ensembles, device) -> GridCellsRNN:
    return GridCellsRNN(
        target_ensembles=target_ensembles,
        nh_lstm=cfg.model.nh_lstm,
        nh_bottleneck=cfg.model.nh_bottleneck,
        dropout_rates=cfg.model.dropout_rates,
        bottleneck_has_bias=cfg.model.bottleneck_has_bias,
        init_weight_disp=cfg.model.init_weight_disp,
        ego_vel_dim=cfg.model.ego_vel_dim,
    ).to(device)


def build_optimizer(model: GridCellsRNN, cfg: Config) -> torch.optim.Optimizer:
    # alpha=0.9 and eps=1e-10 match TF1 RMSPropOptimizer's defaults (PyTorch's
    # are 0.99 / 1e-8). Note TF computes sqrt(v + eps) and PyTorch sqrt(v) +
    # eps, a structural difference no eps can bridge; measured effect here is
    # negligible.
    scope = cfg.model.weight_decay_scope
    return torch.optim.RMSprop([
        {"params": model.decay_parameters(scope), "weight_decay": cfg.model.weight_decay},
        {"params": model.no_decay_parameters(scope), "weight_decay": 0.0},
    ], lr=cfg.train.learning_rate, momentum=cfg.train.momentum, alpha=0.9, eps=1e-10)


def train_step(model, place_cell_ensembles, head_direction_ensembles, batch, cfg,
               device, optimizer):
    init_pos = batch["init_pos"].to(device)
    init_hd = batch["init_hd"].to(device)
    ego_vel = batch["ego_vel"].to(device)
    target_pos = batch["target_pos"].to(device)
    target_hd = batch["target_hd"].to(device)

    if cfg.task.velocity_inputs:
        noise_scale = torch.tensor(cfg.task.velocity_noise, device=device, dtype=ego_vel.dtype)
        if torch.any(noise_scale != 0):
            ego_vel = ego_vel + torch.randn_like(ego_vel) * noise_scale

    init_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles,
                                            head_direction_ensembles)
    targets = encode_targets(target_pos, target_hd, place_cell_ensembles,
                              head_direction_ensembles)

    out = model(init_conds, ego_vel)
    loss = sum(soft_cross_entropy(logit, target)
               for logit, target in zip(out.logits, targets)).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(
        model.clip_parameters(cfg.train.grad_clip_scope), cfg.train.grad_clip_value)
    optimizer.step()

    return loss.item()


def train(cfg: Config, shard_dir: str) -> list[float]:
    """Train to completion; returns the per-epoch mean loss, for plotting."""
    # Seed before anything samples: weight init, dropout masks and the
    # DataLoader's shuffling all draw from torch's global generator.
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    place_cell_ensembles, head_direction_ensembles = build_ensembles(cfg, device)

    model = build_model(cfg, place_cell_ensembles + head_direction_ensembles, device)
    model.train()
    optimizer = build_optimizer(model, cfg)

    loader = build_dataloader(shard_dir, shard_indices=None,
                               batch_size=cfg.train.minibatch_size)
    train_iter = infinite_loader(loader)
    os.makedirs(cfg.train.results_dir, exist_ok=True)

    def save_checkpoint(epoch):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch},
                   os.path.join(cfg.train.results_dir, f"checkpoint_epoch{epoch}.pt"))

    last_epoch = cfg.train.epochs - 1
    epoch_losses = []
    for epoch in range(cfg.train.epochs):
        losses = [train_step(model, place_cell_ensembles, head_direction_ensembles,
                             next(train_iter), cfg, device, optimizer)
                  for _ in range(cfg.train.steps_per_epoch)]
        epoch_losses.append(float(np.mean(losses)))
        print(f"epoch {epoch}: mean_loss={np.mean(losses):.5f} std_loss={np.std(losses):.5f}")

        if epoch % cfg.train.save_every_n_epochs == 0:
            save_checkpoint(epoch)

    # Always checkpoint the final epoch even if it misses the save interval.
    if last_epoch % cfg.train.save_every_n_epochs != 0:
        save_checkpoint(last_epoch)
    return epoch_losses
