"""Minimal end-to-end verification: convert 2 shards (not all 100), build a
tiny DataLoader, run the model forward+backward for a handful of steps, and
confirm nothing is broken before committing to a full 100-shard conversion
and a real (long) training run.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from dataset import build_dataloader
from ensembles import HeadDirectionCellEnsemble, PlaceCellEnsemble
from model import GridCellsRNN
from scripts.convert_csv_to_shards import convert_all
from train import build_optimizer
from utils import encode_initial_conditions, encode_targets

CSV_DIR = "data/square_room_100steps_2.2m_1000000"
SHARD_DIR = "data/shards"
SHARD_INDICES = [0, 1]


def step1_convert():
    print("=== step 1: convert 2 shards ===")
    paths = convert_all(CSV_DIR, SHARD_DIR, shard_indices=SHARD_INDICES)
    assert len(paths) == 2, f"expected 2 shards converted, got {len(paths)}"
    print("OK:", paths)
    return paths


def step2_check_npz(paths):
    print("=== step 2: check .npz contents ===")
    expected_shapes = {
        "init_pos": (10000, 2), "init_hd": (10000, 1), "ego_vel": (10000, 100, 3),
        "target_pos": (10000, 100, 2), "target_hd": (10000, 100, 1),
    }
    for path in paths:
        with np.load(path) as npz:
            for k, shape in expected_shapes.items():
                arr = npz[k]
                assert arr.shape == shape, f"{path}:{k} shape {arr.shape} != {shape}"
                assert arr.dtype == np.float32, f"{path}:{k} dtype {arr.dtype} != float32"
                assert np.isfinite(arr).all(), f"{path}:{k} contains non-finite values"
    print("OK: shapes/dtypes/finite all check out")


def step3_dataloader():
    print("=== step 3: dataset + dataloader ===")
    cfg = Config()
    loader = build_dataloader(SHARD_DIR, shard_indices=SHARD_INDICES,
                               batch_size=cfg.train.minibatch_size)
    assert len(loader.dataset) == 20000, f"expected 20000 trajectories, got {len(loader.dataset)}"
    batch = next(iter(loader))
    b = cfg.train.minibatch_size
    assert batch["init_pos"].shape == (b, 2)
    assert batch["init_hd"].shape == (b, 1)
    assert batch["ego_vel"].shape == (b, 100, 3)
    assert batch["target_pos"].shape == (b, 100, 2)
    assert batch["target_hd"].shape == (b, 100, 1)
    print("OK: batch shapes match contract:", {k: tuple(v.shape) for k, v in batch.items()})
    return cfg, loader


def step4_5_model_forward(cfg, batch, device):
    print("=== step 4/5: build model, forward pass ===")
    place_cell_ensembles = [
        PlaceCellEnsemble(n, stdev=s, pos_min=-cfg.task.env_size / 2.0,
                           pos_max=cfg.task.env_size / 2.0, seed=cfg.task.neurons_seed).to(device)
        for n, s in zip(cfg.task.n_pc, cfg.task.pc_scale)
    ]
    head_direction_ensembles = [
        HeadDirectionCellEnsemble(n, concentration=c, seed=cfg.task.neurons_seed).to(device)
        for n, c in zip(cfg.task.n_hdc, cfg.task.hdc_concentration)
    ]
    target_ensembles = place_cell_ensembles + head_direction_ensembles

    model = GridCellsRNN(
        target_ensembles=target_ensembles, nh_lstm=cfg.model.nh_lstm,
        nh_bottleneck=cfg.model.nh_bottleneck, dropout_rates=cfg.model.dropout_rates,
        bottleneck_has_bias=cfg.model.bottleneck_has_bias,
        init_weight_disp=cfg.model.init_weight_disp, ego_vel_dim=cfg.model.ego_vel_dim,
    ).to(device)
    model.train()

    init_pos = batch["init_pos"].to(device)
    init_hd = batch["init_hd"].to(device)
    ego_vel = batch["ego_vel"].to(device)
    target_pos = batch["target_pos"].to(device)
    target_hd = batch["target_hd"].to(device)

    init_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
    targets = encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)
    out = model(init_conds, ego_vel)

    b = cfg.train.minibatch_size
    assert out.logits[0].shape == (b, 100, 256), out.logits[0].shape
    assert out.logits[1].shape == (b, 100, 12), out.logits[1].shape
    assert out.bottleneck.shape == (b, 100, 512), out.bottleneck.shape
    assert out.lstm_output.shape == (b, 100, 128), out.lstm_output.shape
    print("OK: output shapes match contract")
    return model, place_cell_ensembles, head_direction_ensembles, init_conds, targets, out


def step6_loss(place_cell_ensembles, head_direction_ensembles, out, targets):
    print("=== step 6: loss ===")
    pc_loss = place_cell_ensembles[0].loss(out.logits[0], targets[0])
    hd_loss = head_direction_ensembles[0].loss(out.logits[1], targets[1])
    loss = (pc_loss + hd_loss).mean()
    assert torch.isfinite(loss), "loss is not finite"
    expected_ballpark = np.log(256) + np.log(12)
    print(f"OK: loss={loss.item():.4f} (random-init ballpark ~{expected_ballpark:.2f})")
    return loss


def step7_8_backward(model, loss, cfg):
    print("=== step 7/8: backward, grad check ===")
    loss.backward()
    max_grad = 0.0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        max_grad = max(max_grad, p.grad.abs().max().item())
    print(f"OK: all params have finite gradients. max |grad| across all params = {max_grad:.6g} "
          f"(clip_value={cfg.train.grad_clip_value:g})")
    if max_grad > cfg.train.grad_clip_value * 10:
        print(f"NOTE: max grad ({max_grad:.4g}) is >>clip_value ({cfg.train.grad_clip_value:g}) "
              f"-- clip_grad_value_ will saturate essentially every step. Flagging per the plan.")


def step9_optimizer_step(model, optimizer):
    print("=== step 9: clip + optimizer.step() ===")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    torch.nn.utils.clip_grad_value_(model.parameters(), 1e-5)
    optimizer.step()
    changed = sum(not torch.equal(before[n], p) for n, p in model.named_parameters())
    print(f"OK: {changed}/{len(before)} parameter tensors changed after one optimizer step")


def step10_iterate(model, place_cell_ensembles, head_direction_ensembles, loader, cfg, device, optimizer, n_iters=30):
    print(f"=== step 10: {n_iters} iterations ===")
    model.train()
    it = iter(loader)
    losses = []
    for i in range(n_iters):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        init_pos = batch["init_pos"].to(device)
        init_hd = batch["init_hd"].to(device)
        ego_vel = batch["ego_vel"].to(device)
        target_pos = batch["target_pos"].to(device)
        target_hd = batch["target_hd"].to(device)

        init_conds = encode_initial_conditions(init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)
        targets = encode_targets(target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)
        out = model(init_conds, ego_vel)
        pc_loss = place_cell_ensembles[0].loss(out.logits[0], targets[0])
        hd_loss = head_direction_ensembles[0].loss(out.logits[1], targets[1])
        loss = (pc_loss + hd_loss).mean()
        assert torch.isfinite(loss), f"loss went non-finite at iter {i}"

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), cfg.train.grad_clip_value)
        optimizer.step()
        losses.append(loss.item())

    print("loss trajectory:", [f"{l:.4f}" for l in losses])
    print(f"OK: ran {n_iters} iterations, all losses finite")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    paths = step1_convert()
    step2_check_npz(paths)
    cfg, loader = step3_dataloader()
    batch = next(iter(loader))
    model, place_cell_ensembles, head_direction_ensembles, init_conds, targets, out = \
        step4_5_model_forward(cfg, batch, device)
    loss = step6_loss(place_cell_ensembles, head_direction_ensembles, out, targets)
    step7_8_backward(model, loss, cfg)

    optimizer = build_optimizer(model, cfg)

    step9_optimizer_step(model, optimizer)
    step10_iterate(model, place_cell_ensembles, head_direction_ensembles, loader, cfg, device, optimizer)

    print("\n=== ALL SMOKE TEST STEPS PASSED ===")


if __name__ == "__main__":
    main()
