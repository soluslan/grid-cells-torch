"""Supervised training loop for the grid cell network.

Ported from google-deepmind/grid-cells' train.py. Key differences from a
literal 1:1 port (all deliberate, documented here and in the files they
touch):
  - PyTorch/nn.LSTM instead of TF1+Sonnet v1's custom RNNCell (model.py).
  - Weight decay is actually applied via optimizer param groups (the
    original registered Sonnet regularizers but never summed them into the
    optimized loss -- see model.py docstring). Set model.weight_decay=0.0 in
    config.py to reproduce the original's literal no-op behavior.
  - Grid-scoring evaluation is ported too (scores.py's GridScorer), but lives
    in scripts/evaluate.py rather than being run inline during training here
    -- this file trains and checkpoints only; run scripts/evaluate.py
    separately against a saved checkpoint to score grid-cell-like periodicity.
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
    # alpha=0.9 matches TF1's tf.train.RMSPropOptimizer default `decay` (the
    # squared-gradient moving-average smoothing constant) -- PyTorch's own
    # RMSprop default is alpha=0.99, a slower-decaying average that isn't
    # what the original code (built on tf.train.RMSPropOptimizer) used.
    #
    # eps=1e-10 matches TF1's RMSPropOptimizer default `epsilon`, vs.
    # PyTorch's own default of 1e-8 -- tried on the hypothesis that under the
    # literal grad_clip_value=1e-5 (where the squared-gradient running
    # average v settles around (1e-5)^2=1e-10), eps would matter a lot.
    # Empirically it does NOT (a 40-epoch eps=1e-8 vs eps=1e-10 comparison on
    # this dataset gave near-identical loss curves) -- because PyTorch's
    # formula is `sqrt(v) + eps` (eps *outside* the sqrt; see
    # torch.optim.rmsprop._single_tensor_rmsprop), so once sqrt(v)~1e-5, any
    # eps this small (1e-8 or 1e-10) is negligible next to it either way.
    # TF's formula instead is `sqrt(v + eps)` (eps *inside* the sqrt), where
    # it's the same order as v and actually matters -- a structural
    # difference between the two optimizers that no choice of `eps=` alone
    # can bridge; matching TF exactly would need a custom optimizer step,
    # which isn't worth it given the measured effect is negligible. Left at
    # 1e-10 anyway since it's harmless and marginally more TF-like.
    return torch.optim.RMSprop([
        {"params": model.decay_parameters(), "weight_decay": cfg.model.weight_decay},
        {"params": model.no_decay_parameters(), "weight_decay": 0.0},
    ], lr=cfg.train.learning_rate, momentum=cfg.train.momentum, alpha=0.9, eps=1e-10)


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

    def save_checkpoint(epoch):
        ckpt_path = os.path.join(cfg.train.results_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch}, ckpt_path)

    last_epoch = cfg.train.epochs - 1
    for epoch in range(cfg.train.epochs):
        losses = []
        for _ in range(cfg.train.steps_per_epoch):
            batch = next(train_iter)
            loss_val = train_step(model, place_cell_ensembles, head_direction_ensembles,
                                   batch, cfg, device, optimizer)
            losses.append(loss_val)

        print(f"epoch {epoch}: mean_loss={np.mean(losses):.5f} std_loss={np.std(losses):.5f}")

        if epoch % cfg.train.save_every_n_epochs == 0:
            save_checkpoint(epoch)

    # always checkpoint the fully-trained final epoch, even if it doesn't
    # land on a save_every_n_epochs boundary (e.g. epochs=300 -> last saved
    # by the loop above is 298, not the true final epoch 299).
    if last_epoch % cfg.train.save_every_n_epochs != 0:
        save_checkpoint(last_epoch)


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
