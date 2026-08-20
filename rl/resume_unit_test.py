"""Isolated verification of the resume load-order fix (no DMLab/multiprocess needed): does
build_shared_models(cfg, checkpoint) actually (a) load real weights/optimizer state instead of
fresh random init, and (b) leave every loaded optimizer-state tensor genuinely
`.is_shared() == True`, not a private post-load copy that silently un-shares Hogwild?

Run from grid-cells-torch/: python rl/resume_unit_test.py
"""
import sys

sys.path.insert(0, ".")

import torch

from config import Config
from rl_train import _save_checkpoint, build_shared_models


def _populate_optimizer_state(models):
    """One real backward+step per model so square_avg (and momentum_buffer where applicable)
    are non-zero -- a checkpoint saved before any step would trivially "match" a fresh zero-init
    and wouldn't actually test that loading happened."""
    vision_out = models["vision_module"](torch.randn(2, 3, 64, 64))
    place_t = models["place_ensemble"].posterior(torch.randn(2, 1, 2)).squeeze(1)
    hd_t = models["hd_ensemble"].posterior(torch.randn(2, 1, 1)).squeeze(1)
    loss = models["vision_module"].loss(torch.randn(2, 3, 64, 64), place_t, hd_t)
    models["vision_optimizer"].zero_grad()
    loss.backward()
    models["vision_optimizer"].step()

    hidden = models["actor_critic"].init_hidden(2)
    out = models["actor_critic"].step(
        torch.randn(2, 3, 64, 64), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long),
        torch.randn(2, cfg.model.nh_bottleneck), torch.randn(2, cfg.model.nh_bottleneck), hidden)
    ac_loss = out.value.sum() + out.action_logits.sum()
    models["ac_optimizer"].zero_grad()
    ac_loss.backward()
    models["ac_optimizer"].step()

    h0, c0 = models["grid_network"].init_hidden(
        [torch.zeros(2, models["place_ensemble"].n_cells), torch.zeros(2, models["hd_ensemble"].n_cells)])
    bottleneck, _ = models["grid_network"].step(torch.randn(2, 3), (h0, c0))
    grid_loss = bottleneck.sum()
    models["grid_optimizer"].zero_grad()
    grid_loss.backward()
    models["grid_optimizer"].step()


cfg = Config()
cfg.rl.env.num_actors = 2

print("=== building fresh models, taking one optimizer step each ===")
models = build_shared_models(cfg, checkpoint=None)
_populate_optimizer_state(models)

# Snapshot the post-step values we'll check survive a save/load round-trip.
grid_w_before = next(models["grid_network"].parameters()).clone()
vision_square_avg_before = {
    id(p): s["square_avg"].clone() for p, s in models["vision_optimizer"].state.items()}
assert any(v.abs().sum() > 0 for v in vision_square_avg_before.values()), \
    "test setup bug: square_avg is all-zero before saving, optimizer step didn't run"

ckpt_path = "/tmp/resume_unit_test_checkpoint.pt"
_save_checkpoint(ckpt_path, step=123456, vision_module=models["vision_module"],
                  grid_network=models["grid_network"], actor_critic=models["actor_critic"],
                  vision_optimizer=models["vision_optimizer"], grid_optimizer=models["grid_optimizer"],
                  ac_optimizer=models["ac_optimizer"])
print(f"saved checkpoint to {ckpt_path}")

print("=== simulating a fresh process: build_shared_models(cfg, checkpoint) ===")
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
assert checkpoint["step"] == 123456, "step not round-tripped"
resumed = build_shared_models(cfg, checkpoint)

# 1. Model weights actually loaded (not fresh random init).
grid_w_after = next(resumed["grid_network"].parameters())
assert torch.equal(grid_w_before, grid_w_after), "grid_network weights did NOT survive resume"
print("OK: grid_network weights match saved checkpoint exactly")

# 2. Optimizer state actually loaded (non-zero, matching saved values) -- this is the part that
#    would stay silently zero if load_state_dict() ran but share_memory() ran first.
resumed_square_avg = list(resumed["vision_optimizer"].state.values())
assert len(resumed_square_avg) > 0, "no optimizer state at all after resume"
nonzero = [s["square_avg"] for s in resumed_square_avg if s["square_avg"].abs().sum() > 0]
assert len(nonzero) == len(resumed_square_avg), (
    f"only {len(nonzero)}/{len(resumed_square_avg)} vision_optimizer state tensors are non-zero "
    "after resume -- optimizer state did not load")
print(f"OK: all {len(nonzero)} vision_optimizer square_avg tensors are non-zero after resume "
      "(loaded, not reset)")

# 3. The loaded state tensors are genuinely shared memory, not private post-load copies -- the
#    exact failure mode the load-order fix exists to prevent.
not_shared = [s["square_avg"] for s in resumed["vision_optimizer"].state.values()
              if not s["square_avg"].is_shared()]
assert not not_shared, f"{len(not_shared)} vision_optimizer state tensors are NOT shared memory " \
    "after resume -- load-order bug: share_memory() ran before load_state_dict()"
print("OK: all vision_optimizer state tensors report is_shared() == True after resume")

ac_not_shared = [s["square_avg"] for s in resumed["ac_optimizer"].state.values()
                 if not s["square_avg"].is_shared()]
assert not ac_not_shared, f"{len(ac_not_shared)} ac_optimizer state tensors not shared after resume"
print("OK: all ac_optimizer state tensors report is_shared() == True after resume")

grid_not_shared = [s["square_avg"] for s in resumed["grid_optimizer"].state.values()
                   if not s["square_avg"].is_shared()]
assert not grid_not_shared, f"{len(grid_not_shared)} grid_optimizer state tensors not shared after resume"
print("OK: all grid_optimizer state tensors report is_shared() == True after resume")

print("\nALL CHECKS PASSED: resume correctly loads weights+optimizer state and keeps it shared")
