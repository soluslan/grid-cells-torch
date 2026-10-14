import glob
import os
import re
import shutil
import signal
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from config import Config
from grid_cells.rl_a3c import _save_checkpoint, build_shared_models, train_rl_agent

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _small_cfg(results_dir):
    cfg = Config()
    cfg.rl.env.num_actors = 2
    cfg.rl.replay.frame_capacity = 2_000
    cfg.rl.replay.sequence_capacity = 20
    cfg.rl.results_dir = results_dir
    cfg.rl.env.level_directory = os.path.join(REPO, "levels")
    return cfg


def _one_step_each(models, cfg):
    loss = models["vision_module"].loss(
        torch.randn(2, 3, 64, 64),
        models["place_ensemble"].posterior(torch.randn(2, 1, 2)).squeeze(1),
        models["hd_ensemble"].posterior(torch.randn(2, 1, 1)).squeeze(1))
    models["vision_optimizer"].zero_grad()
    loss.backward()
    models["vision_optimizer"].step()

    out = models["actor_critic"].step(
        torch.randn(2, 3, 64, 64), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long),
        torch.randn(2, cfg.model.nh_bottleneck), torch.randn(2, cfg.model.nh_bottleneck),
        models["actor_critic"].init_hidden(2))
    models["ac_optimizer"].zero_grad()
    (out.value.sum() + out.action_logits.sum()).backward()
    models["ac_optimizer"].step()

    n_pc, n_hd = models["place_ensemble"].n_cells, models["hd_ensemble"].n_cells
    hidden = models["grid_network"].init_hidden([torch.zeros(2, n_pc), torch.zeros(2, n_hd)])
    code, _ = models["grid_network"].step(torch.randn(2, 4), torch.randn(2, n_pc + n_hd), hidden)
    models["grid_optimizer"].zero_grad()
    code.sum().backward()
    models["grid_optimizer"].step()


def test_resume_loads_weights_and_keeps_optimizer_state_shared(tmp_path):
    cfg = _small_cfg(str(tmp_path))
    models = build_shared_models(cfg, checkpoint=None)
    _one_step_each(models, cfg)
    grid_w = next(models["grid_network"].parameters()).clone()

    path = tmp_path / "ckpt.pt"
    _save_checkpoint(str(path), step=123456, vision_module=models["vision_module"],
                     grid_network=models["grid_network"], actor_critic=models["actor_critic"],
                     vision_optimizer=models["vision_optimizer"],
                     grid_optimizer=models["grid_optimizer"], ac_optimizer=models["ac_optimizer"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 123456
    resumed = build_shared_models(cfg, checkpoint)

    assert torch.equal(grid_w, next(resumed["grid_network"].parameters()))
    for name in ("vision_optimizer", "ac_optimizer", "grid_optimizer"):
        states = list(resumed[name].state.values())
        assert states, name
        assert all(s["square_avg"].is_shared() for s in states), name
    assert all(s["square_avg"].abs().sum() > 0
               for s in resumed["vision_optimizer"].state.values())


@pytest.mark.dmlab
def test_external_level_directory_loads():
    import deepmind_lab
    env = deepmind_lab.Lab(
        "square_arena_smoke_test",
        ["RGB_INTERLEAVED", "VEL.TRANS", "VEL.ROT", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
        config={"width": "64", "height": "64", "levelDirectory": os.path.join(REPO, "levels")})
    env.reset(seed=0)
    forward = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc)
    for _ in range(20):
        env.step(forward, num_steps=4)
    obs = env.observations()
    assert obs["RGB_INTERLEAVED"].shape == (64, 64, 3)
    env.close()


@pytest.mark.dmlab
@pytest.mark.slow
def test_multiprocess_run_joins_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = _small_cfg(str(tmp_path))
    cfg.rl.total_env_steps = 2000
    cfg.rl.actor_critic.n_step = 20
    train_rl_agent(cfg)
    assert glob.glob(str(tmp_path / "checkpoint_step*_final.pt"))


def _run(cfg):
    train_rl_agent(cfg)


@pytest.mark.dmlab
@pytest.mark.slow
def test_sigterm_then_resume_continues_step_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cfg = _small_cfg(str(tmp_path))
    cfg.rl.total_env_steps = 10_000_000
    cfg.rl.checkpoint_every_env_steps = 100_000
    p = mp.get_context("spawn").Process(target=_run, args=(cfg,))
    p.start()
    time.sleep(45)
    assert p.is_alive()
    os.kill(p.pid, signal.SIGTERM)
    p.join(timeout=60)
    assert not p.is_alive()

    finals = glob.glob(str(tmp_path / "checkpoint_step*_final.pt"))
    assert len(finals) == 1
    n = int(re.search(r"checkpoint_step(\d+)_final\.pt", finals[0]).group(1))
    assert n > 0

    cfg2 = _small_cfg(str(tmp_path))
    cfg2.rl.total_env_steps = n + 40_000
    cfg2.rl.checkpoint_every_env_steps = 1_000_000_000
    cfg2.rl.resume_from = finals[0]
    train_rl_agent(cfg2)
    steps = [int(line.split(",")[0])
             for path in glob.glob(str(tmp_path / "actor_*_episodes.csv"))
             for line in open(path)]
    assert any(s > n for s in steps)
