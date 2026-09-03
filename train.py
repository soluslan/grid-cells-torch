import os

import numpy as np
import torch

from config import Config
from dataset import build_dataloader, infinite_loader
from ensembles import (build_ensembles, encode_initial_conditions, encode_targets,
                       soft_cross_entropy)
from model import GridCellsRNN


def build_model(cfg: Config, target_ensembles, device, vision_dim: int = 0,
                ego_vel_dim: int | None = None) -> GridCellsRNN:
    return GridCellsRNN(
        target_ensembles=target_ensembles,
        nh_lstm=cfg.model.nh_lstm,
        nh_bottleneck=cfg.model.nh_bottleneck,
        dropout_rates=cfg.model.dropout_rates,
        bottleneck_has_bias=cfg.model.bottleneck_has_bias,
        ego_vel_dim=cfg.model.ego_vel_dim if ego_vel_dim is None else ego_vel_dim,
        vision_dim=vision_dim,
    ).to(device)


def build_optimizer(model: GridCellsRNN, cfg: Config, learning_rate: float | None = None,
                     weight_decay: float | None = None) -> torch.optim.Optimizer:
    scope = cfg.model.weight_decay_scope
    lr = cfg.train.learning_rate if learning_rate is None else learning_rate
    wd = cfg.model.weight_decay if weight_decay is None else weight_decay
    return torch.optim.RMSprop([
        {"params": model.decay_parameters(scope), "weight_decay": wd},
        {"params": model.no_decay_parameters(scope), "weight_decay": 0.0},
    ], lr=lr, momentum=cfg.train.momentum, alpha=0.9, eps=1e-10)


def train_step(model, place_cell_ensembles, head_direction_ensembles, batch, cfg,
               device, optimizer, vision_module=None):
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

    vision_seq = None
    if vision_module is not None:
        images = batch["image"].to(device)
        b, t = images.shape[:2]
        vis_out = vision_module(images.reshape(b * t, *images.shape[2:]))
        vision_seq = torch.cat([vis_out.place_probs, vis_out.hd_probs], dim=-1).reshape(b, t, -1)

    out = model(init_conds, ego_vel, vision_seq=vision_seq)
    loss = sum(soft_cross_entropy(logit, target)
               for logit, target in zip(out.logits, targets)).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(
        model.clip_parameters(cfg.train.grad_clip_scope), cfg.train.grad_clip_value)
    optimizer.step()

    return loss.item()


def train(cfg: Config, shard_dir: str) -> list[float]:
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

    if last_epoch % cfg.train.save_every_n_epochs != 0:
        save_checkpoint(last_epoch)
    return epoch_losses
