import numpy as np
import pytest
import torch

from config import Config
from grid_cells import rl_a2c
from grid_cells.actor_critic import DISCRETE_ACTIONS, ActorCriticLSTM
from grid_cells.ensembles import build_rl_ensembles
from grid_cells.replay_buffer import TimeMajorReplay
from grid_cells.rl_a3c import _apply_a3c_update
from grid_cells.training import build_model
from grid_cells.vec_env import SyncVecEnv
from grid_cells.vision import VisionModule


def test_returns_known_answer():
    r = torch.tensor([[0.0], [0.0], [10.0]])
    cuts = torch.zeros(3, 1, dtype=torch.bool)
    out = rl_a2c.discounted_returns(r, cuts, torch.tensor([2.0]), 0.99)
    assert out[:, 0].tolist() == pytest.approx([11.741598, 11.8602, 11.98], abs=1e-5)


def test_returns_respect_episode_cuts_per_env():
    r = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 2.0]])
    cuts = torch.tensor([[True, False], [False, False], [False, False]])
    out = rl_a2c.discounted_returns(r, cuts, torch.tensor([5.0, 5.0]), 0.5)
    assert out[:, 0].tolist() == pytest.approx([1.0, 1.25, 2.5])
    assert out[:, 1].tolist() == pytest.approx([2.125, 2.25, 4.5])


def test_loss_matches_a3c_update_math():
    torch.manual_seed(0)
    cfg = Config()
    rewards = [0.0, 1.0, 0.0, 10.0]
    lp = torch.randn(4, requires_grad=True)
    v = torch.randn(4, requires_grad=True)
    h = torch.rand(4, requires_grad=True)

    class NoOpt:
        def zero_grad(self): pass
        def step(self): pass

    _apply_a3c_update(None, NoOpt(), list(lp.view(4, 1)), list(v.view(4, 1)),
                      list(h.view(4, 1)), rewards, 3.0, cfg.rl.actor_critic)
    grads_a3c = [t.grad.clone() for t in (lp, v, h)]
    for t in (lp, v, h):
        t.grad = None

    returns = rl_a2c.discounted_returns(torch.tensor(rewards).view(4, 1),
                                        torch.zeros(4, 1, dtype=torch.bool),
                                        torch.tensor([3.0]), cfg.rl.actor_critic.discount)
    loss, _ = rl_a2c.actor_critic_loss(lp.view(4, 1), v.view(4, 1), h.view(4, 1), returns,
                                       cfg.rl.actor_critic)
    loss.backward()
    for a, b in zip(grads_a3c, (lp.grad, v.grad, h.grad)):
        assert torch.allclose(a, b, atol=1e-6)


def _real_agent(cfg, device=torch.device("cpu")):
    place, hd = build_rl_ensembles(cfg, device)
    grid = build_model(cfg, place + hd, device, vision_dim=place[0].n_cells + hd[0].n_cells,
                       ego_vel_dim=4)
    vision = VisionModule(place[0], hd[0], mask_prob=0.0)
    ac = ActorCriticLSTM(grid_code_dim=cfg.model.nh_bottleneck)
    return rl_a2c.Agent(cfg, vision, grid, ac, device)


def test_batched_step_equals_per_env_steps():
    torch.manual_seed(0)
    cfg = Config()
    agent = _real_agent(cfg)
    b = 3
    state = agent.initial_state(b)
    state.grid_hidden = tuple(torch.randn_like(x) for x in state.grid_hidden)
    state.ac_hidden = tuple(torch.randn_like(x) for x in state.ac_hidden)
    state.goal_code = torch.randn_like(state.goal_code)
    state.prev_action = torch.tensor([0, 3, 5])
    state.prev_reward = torch.tensor([[0.0], [10.0], [0.0]])
    obs = {"image": torch.randint(0, 255, (b, 3, 64, 64), dtype=torch.uint8)}
    ego = torch.randn(b, 4)
    with torch.no_grad():
        batched, code_b, gh_b = agent.act(state, obs, ego)
        for i in range(b):
            s = rl_a2c.AgentState(tuple(x[:, i:i + 1] for x in state.grid_hidden),
                                  tuple(x[:, i:i + 1] for x in state.ac_hidden),
                                  state.prev_action[i:i + 1], state.prev_reward[i:i + 1],
                                  state.goal_code[i:i + 1])
            single, code_s, gh_s = agent.act(s, {"image": obs["image"][i:i + 1]}, ego[i:i + 1])
            assert torch.allclose(batched.action_logits[i], single.action_logits[0], atol=1e-5)
            assert torch.allclose(batched.value[i], single.value[0], atol=1e-5)
            assert torch.allclose(batched.hidden[0][:, i], single.hidden[0][:, 0], atol=1e-5)
            assert torch.allclose(code_b[i], code_s[0], atol=1e-5)
            assert torch.allclose(gh_b[0][:, i], gh_s[0][:, 0], atol=1e-5)


def test_mask_hidden_only_touches_masked_envs():
    h = (torch.ones(1, 3, 4), torch.ones(1, 3, 4) * 2)
    out = rl_a2c._mask_hidden(h, torch.tensor([1.0, 0.0, 1.0]))
    assert out[0][0, 1].abs().sum() == 0 and out[1][0, 1].abs().sum() == 0
    assert torch.equal(out[0][0, 0], h[0][0, 0]) and torch.equal(out[1][0, 2], h[1][0, 2])


def test_replay_is_lossless_and_sequences_stay_in_one_episode():
    replay = TimeMajorReplay(capacity_steps=20, num_envs=2, seed=0)
    for t in range(30):
        img = torch.full((2, 3, 64, 64), t % 256, dtype=torch.uint8)
        replay.add(img, torch.full((2, 2), float(t)), torch.zeros(2, 1), torch.zeros(2, 4),
                   torch.tensor([t // 7, 100 + t // 11]))
    frames = replay.sample_frames(50)
    assert torch.equal(frames["image"][:, 0, 0, 0].float(), frames["pos"][:, 0])
    assert frames["pos"][:, 0].min() >= 10
    seqs = replay.sample_sequences(30, 5)
    steps = seqs["pos"][..., 0]
    assert torch.equal(steps[:, 1:] - steps[:, :-1], torch.ones(30, 4))
    for row, b in zip(steps, seqs["image"][:, 0, 0, 0, 0]):
        t0 = int(row[0])
        assert {t // 7 for t in range(t0, t0 + 5)} == {t0 // 7} or \
            {t // 11 for t in range(t0, t0 + 5)} == {t0 // 11}


class BanditEnv:
    """Constant image; +1 for choosing DISCRETE_ACTIONS[2], else 0. 50-action episodes."""

    def __init__(self):
        self.t = 0

    def reset(self, seed=None):
        self.t = 0

    def observations(self):
        return {"RGB_INTERLEAVED": np.full((64, 64, 3), 100, dtype=np.uint8),
                "DEBUG.POS.TRANS": np.array([600.0, 600.0, 0.0]),
                "DEBUG.POS.ROT": np.zeros(3), "VEL.TRANS": np.zeros(3), "VEL.ROT": np.zeros(3)}

    def step(self, action, num_steps=1):
        self.t += 1
        return 1.0 if np.array_equal(action, DISCRETE_ACTIONS[2]) else 0.0

    def is_running(self):
        return self.t < 50

    def close(self):
        pass


@pytest.mark.slow
def test_a2c_learns_a_bandit_with_paper_hyperparameters(tmp_path):
    cfg = Config()
    cfg.rl.a2c.device = "cpu"
    cfg.rl.a2c.vision_updates_per_rollout = 0
    cfg.rl.a2c.grid_updates_per_rollout = 0
    cfg.rl.actor_critic.n_step = 20
    cfg.rl.results_dir = str(tmp_path)
    cfg.rl.log_every_n_updates = 10 ** 9
    cfg.rl.total_env_steps = 10 ** 12
    torch.manual_seed(0)
    learner = rl_a2c.train_a2c(cfg, env_factories=[BanditEnv] * 8, vec_env_cls=SyncVecEnv,
                               max_updates=300)
    agent = learner.agent
    with torch.no_grad():
        out, _, _ = agent.act(learner.state, learner.obs, agent.ego_vel(learner.obs))
    p = torch.softmax(out.action_logits, -1)[:, 2].mean().item()
    assert p > 0.9, f"P(rewarded action) = {p:.3f} after 300 updates"


class SeedRecordingEnv(BanditEnv):
    seeds = []

    def reset(self, seed=None):
        super().reset(seed)
        SeedRecordingEnv.seeds.append(seed)


def test_vec_env_auto_resets_with_distinct_deterministic_seeds():
    SeedRecordingEnv.seeds = []
    cfg = Config()
    env = SyncVecEnv([SeedRecordingEnv] * 3, cfg.rl.env, base_seed=100)
    env.reset()
    ended_seen = np.zeros(3, dtype=int)
    for _ in range(120):
        _, _, ended = env.step(np.zeros(3, dtype=int))
        ended_seen += ended
    assert ended_seen.tolist() == [2, 2, 2]
    assert sorted(SeedRecordingEnv.seeds) == [100, 101, 102, 103, 104, 105, 106, 107, 108]


@pytest.mark.dmlab
def test_subprocess_vec_env_steps_real_dmlab():
    from grid_cells.rl_common import DMLabFactory
    from grid_cells.vec_env import SubprocVecEnv
    import os
    cfg = Config()
    cfg.rl.env.level_directory = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "levels")
    env = SubprocVecEnv([DMLabFactory(cfg.rl.env)] * 2, cfg.rl.env)
    try:
        obs = env.reset()
        assert obs["image"].shape == (2, 3, 64, 64) and obs["image"].dtype == np.uint8
        assert np.all(np.abs(obs["pos"]) <= 1.25)
        for _ in range(5):
            obs, rewards, ended = env.step(np.array([2, 2]))
        assert rewards.shape == (2,) and ended.dtype == bool
    finally:
        env.close()
