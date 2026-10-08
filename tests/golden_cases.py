import hashlib
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(REPO, "tests", "golden", "pre_restructure.npz")
SHARD0 = os.path.join(REPO, "data", "shards", "0000-of-0099.npz")


def _modules():
    from config import Config
    try:
        from grid_cells import actor_critic, ensembles, model, scores, training, vision
    except ImportError:
        import actor_critic, ensembles, model, scores, vision
        import train as training
    return Config, actor_critic, ensembles, model, scores, training, vision


def _synthetic_batch(rng, b=10, t=100):
    init_pos = rng.uniform(-1.0, 1.0, (b, 2)).astype(np.float32)
    init_hd = rng.uniform(-np.pi, np.pi, (b, 1)).astype(np.float32)
    speed = rng.rayleigh(0.13, (b, t)).astype(np.float32)
    dtheta = rng.normal(0.0, 0.3, (b, t)).astype(np.float32)
    ego_vel = np.stack([speed, np.sin(dtheta), np.cos(dtheta)], -1).astype(np.float32)
    heading = init_hd + np.cumsum(dtheta, axis=1)
    step = np.stack([np.cos(heading), np.sin(heading)], -1) * speed[..., None] * 0.15
    target_pos = np.clip(init_pos[:, None, :] + np.cumsum(step, axis=1), -1.1, 1.1)
    return {
        "init_pos": torch.from_numpy(init_pos),
        "init_hd": torch.from_numpy(init_hd),
        "ego_vel": torch.from_numpy(ego_vel),
        "target_pos": torch.from_numpy(target_pos.astype(np.float32)),
        "target_hd": torch.from_numpy(np.arctan2(np.sin(heading), np.cos(heading))[..., None].astype(np.float32)),
    }


def compute_golden() -> dict:
    Config, actor_critic, ensembles, model, scores, training, vision = _modules()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    cfg = Config()
    out = {}

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    place, hd = ensembles.build_ensembles(cfg, device)
    net = training.build_model(cfg, place + hd, device)
    opt = training.build_optimizer(net, cfg)
    losses = [training.train_step(net, place, hd, _synthetic_batch(rng), cfg, device, opt)
              for _ in range(30)]
    out["supervised_losses"] = np.array(losses, dtype=np.float64)
    out["supervised_bottleneck_weight"] = net.bottleneck.weight.detach().numpy().copy()

    torch.manual_seed(1)
    net.eval()
    batch = _synthetic_batch(np.random.default_rng(1), b=4)
    init = ensembles.encode_initial_conditions(batch["init_pos"], batch["init_hd"], place, hd)
    with torch.no_grad():
        o = net(init, batch["ego_vel"])
    out["supervised_forward_logits0"] = o.logits[0].numpy()
    out["supervised_forward_bottleneck"] = o.bottleneck.numpy()

    torch.manual_seed(2)
    rl_place, rl_hd = ensembles.build_rl_ensembles(cfg, device)
    grid = training.build_model(cfg, rl_place + rl_hd, device,
                                vision_dim=rl_place[0].n_cells + rl_hd[0].n_cells, ego_vel_dim=4)
    vis = vision.VisionModule(rl_place[0], rl_hd[0])
    ac = actor_critic.ActorCriticLSTM(grid_code_dim=cfg.model.nh_bottleneck)
    images = torch.rand(3, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        torch.manual_seed(3)
        vo = vis(images)
        vision_t = torch.cat([vo.place_probs, vo.hd_probs], -1)
        hidden = grid.init_hidden_from_predictions(
            [torch.zeros(3, rl_place[0].n_cells), torch.zeros(3, rl_hd[0].n_cells)])
        code, hidden = grid.step(torch.randn(3, 4), vision_t, hidden)
        a = ac.step(images, torch.ones(3, 1), torch.tensor([0, 2, 5]), code, code * 0.5,
                    ac.init_hidden(3))
        vloss = vis.loss(images, rl_place[0].posterior(torch.zeros(3, 1, 2)).squeeze(1),
                         rl_hd[0].posterior(torch.zeros(3, 1, 1)).squeeze(1))
    out["vision_place_probs"] = vo.place_probs.numpy()
    out["vision_loss"] = np.array(vloss.item())
    out["grid_step_code"] = code.numpy()
    out["grid_step_h"] = hidden[0].numpy()
    out["ac_logits"] = a.action_logits.numpy()
    out["ac_value"] = a.value.numpy()

    grng = np.random.default_rng(4)
    xy = grng.uniform(-1.1, 1.1, (20000, 2))
    k = 2 * np.pi / 0.6
    angles = np.deg2rad([0, 60, 120])
    hexa = sum(np.cos(k * (xy[:, 0] * np.cos(a_) + xy[:, 1] * np.sin(a_))) for a_ in angles)
    acts = np.stack([hexa, grng.normal(size=len(xy))], 1)
    scorer = scores.GridScorer(32, ((-1.1, 1.1), (-1.1, 1.1)), scores.paper_mask_parameters())
    s60, ratemaps, sacs, _ = scores.score_units(scorer, xy, acts)
    out["scores_60"] = s60
    out["scores_ratemap0"] = np.nan_to_num(ratemaps[0])
    out["scores_sac0"] = np.nan_to_num(sacs[0])
    return out


def shard0_digest(n_traj: int = 100) -> str | None:
    if not os.path.exists(SHARD0):
        return None
    h = hashlib.sha256()
    with np.load(SHARD0) as npz:
        for key in sorted(npz.files):
            h.update(key.encode())
            h.update(np.ascontiguousarray(npz[key][:n_traj]).tobytes())
    return h.hexdigest()
