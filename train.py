"""Supervised training loop for the grid cell network.

Ported from google-deepmind/grid-cells' train.py. Key differences from a
literal 1:1 port (all deliberate, documented here and in the files they
touch):
  - PyTorch/nn.LSTM instead of TF1+Sonnet v1's custom RNNCell (model.py).
  - Weight decay is actually applied via optimizer param groups (the
    original registered Sonnet regularizers but never summed them into the
    optimized loss -- see model.py docstring). Set model.weight_decay=0.0 in
    config.py to reproduce the original's literal no-op behavior.
  - Grid-scoring evaluation (scores.py in the original) is NOT included yet;
    this trains and checkpoints only. See README.md "Phase 2".
"""

import argparse
import os
import time

import numpy as np
import torch

from config import Config
from dataset import build_dataloader, infinite_loader
from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from model import GridCellsRNN
from utils import encode_initial_conditions, encode_targets


def build_ensembles(cfg: Config, device):
    place_cell_ensembles = [
        PlaceCellEnsemble(n, stdev=s, pos_min=-cfg.task.env_size / 2.0,
                           pos_max=cfg.task.env_size / 2.0, seed=cfg.task.neurons_seed).to(device)
        for n, s in zip(cfg.task.n_pc, cfg.task.pc_scale)
    ]
    head_direction_ensembles = [
        HeadDirectionCellEnsemble(n, concentration=c, seed=cfg.task.neurons_seed).to(device)
        for n, c in zip(cfg.task.n_hdc, cfg.task.hdc_concentration)
    ]
    return place_cell_ensembles, head_direction_ensembles


def build_optimizer(model: GridCellsRNN, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.RMSprop([
        {"params": model.decay_parameters(), "weight_decay": cfg.model.weight_decay},
        {"params": model.no_decay_parameters(), "weight_decay": 0.0},
    ], lr=cfg.train.learning_rate, momentum=cfg.train.momentum)


def train_step(model, place_cell_ensembles, head_direction_ensembles, batch, cfg, device, optimizer):
    init_pos = batch["init_pos"].to(device)
    init_hd = batch["init_hd"].to(device)
    ego_vel = batch["ego_vel"].to(device)
    target_pos = batch["target_pos"].to(device)
    target_hd = batch["target_hd"].to(device)

    if cfg.task.velocity_inputs:
        noise_scale = torch.tensor(cfg.task.velocity_noise, device=device, dtype=ego_vel.dtype)
        if torch.any(noise_scale != 0):
            ego_vel = ego_vel + torch.randn_like(ego_vel) * noise_scale

    init_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    targets = encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

    out = model(init_conds, ego_vel)

    pc_loss = place_cell_ensembles[0].loss(out.logits[0], targets[0])  # [B,T]
    hd_loss = head_direction_ensembles[0].loss(out.logits[1], targets[1])  # [B,T]
    loss = (pc_loss + hd_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), cfg.train.grad_clip_value)
    optimizer.step()

    return loss.item()


def train(cfg: Config, shard_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    place_cell_ensembles, head_direction_ensembles = build_ensembles(cfg, device)
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    model = GridCellsRNN(
        target_ensembles=target_ensembles,
        nh_lstm=cfg.model.nh_lstm,
        nh_bottleneck=cfg.model.nh_bottleneck,
        dropout_rates=cfg.model.dropout_rates,
        bottleneck_has_bias=cfg.model.bottleneck_has_bias,
        init_weight_disp=cfg.model.init_weight_disp,
        ego_vel_dim=cfg.model.ego_vel_dim,
    ).to(device)
    model.train()

    optimizer = build_optimizer(model, cfg)

    loader = build_dataloader(shard_dir, shard_indices=None, batch_size=cfg.train.minibatch_size)
    train_iter = infinite_loader(loader)

    os.makedirs(cfg.train.results_dir, exist_ok=True)

    for epoch in range(cfg.train.epochs):
        losses = []
        for _ in range(cfg.train.steps_per_epoch):
            batch = next(train_iter)
            loss_val = train_step(model, place_cell_ensembles, head_direction_ensembles,
                                   batch, cfg, device, optimizer)
            losses.append(loss_val)

        print(f"epoch {epoch}: mean_loss={np.mean(losses):.5f} std_loss={np.std(losses):.5f}")

        if epoch % cfg.train.save_every_n_epochs == 0:
            ckpt_path = os.path.join(cfg.train.results_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch}, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", default="data/shards")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    cfg = Config()
    cfg.train.results_dir = args.results_dir

    t0 = time.time()
    train(cfg, args.shard_dir)
    print(f"done in {time.time() - t0:.1f}s")
