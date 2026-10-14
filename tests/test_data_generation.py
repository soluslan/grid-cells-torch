import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from config import Config
from grid_cells.data_generation import (MotionParams, generate_shard_arrays, resample_trajectory,
                                        simulate_rh_trajectory, trajectories_to_arrays,
                                        wall_redirect, wall_zone)
from grid_cells.dataset import check_shards

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_SHARD0 = os.path.join(REPO, "data", "shards", "0000-of-0099.npz")
PARAMS = MotionParams.from_config(Config())


def legacy_csv_shard_to_arrays(df, n_traj, n_steps, env_size):
    """The removed CSV -> npz conversion (dataset.csv_shard_to_arrays at 55ef4ea), kept as an
    oracle for the direct path."""
    def col(name):
        return df[name].to_numpy(dtype=np.float32).reshape(n_traj, n_steps)

    pos_x, pos_y = col("pos_x"), col("pos_y")
    vel_x, vel_y = col("vel_x"), col("vel_y")
    hd_x, hd_y = col("head_direction_x"), col("head_direction_y")
    pos_centered = np.stack([pos_x - env_size / 2.0, pos_y - env_size / 2.0], axis=-1)
    theta = np.arctan2(hd_y, hd_x)
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    prev_hd_x = np.concatenate([hd_x[:, 0:1], hd_x[:, :-1]], axis=1)
    prev_hd_y = np.concatenate([hd_y[:, 0:1], hd_y[:, :-1]], axis=1)
    dtheta = np.arctan2(prev_hd_x * hd_y - prev_hd_y * hd_x, prev_hd_x * hd_x + prev_hd_y * hd_y)
    return {
        "init_pos": pos_centered[:, 0, :].astype(np.float32),
        "init_hd": theta[:, 0:1].astype(np.float32),
        "ego_vel": np.stack([speed, np.sin(dtheta), np.cos(dtheta)], axis=-1).astype(np.float32),
        "target_pos": pos_centered.astype(np.float32),
        "target_hd": theta[..., None].astype(np.float32),
    }


def test_direct_path_matches_legacy_csv_roundtrip(tmp_path):
    n = 20
    np.random.seed(123)
    rows, pos, vel, hd = [], [], [], []
    for i in range(n):
        pf, af, vf, _ = simulate_rh_trajectory(PARAMS)
        p_, v_, h_ = resample_trajectory(PARAMS, pf, af, vf)
        pos.append(p_); vel.append(v_); hd.append(h_)
        for s in range(PARAMS.n_steps):
            rows.append({"trajectory_id": i, "step": s, "pos_x": p_[s, 0], "pos_y": p_[s, 1],
                         "vel_x": v_[s, 0], "vel_y": v_[s, 1],
                         "head_direction_x": h_[s, 0], "head_direction_y": h_[s, 1]})
    csv = tmp_path / "shard.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    legacy = legacy_csv_shard_to_arrays(pd.read_csv(csv), n, PARAMS.n_steps, PARAMS.env_size)
    direct = trajectories_to_arrays(np.stack(pos), np.stack(vel), np.stack(hd), PARAMS.env_size)
    for key in legacy:
        np.testing.assert_allclose(direct[key], legacy[key], rtol=0, atol=1e-6, err_msg=key)


def test_prefix_of_shard0_matches_existing_dataset():
    if not os.path.exists(LEGACY_SHARD0):
        pytest.skip("data/shards/0000-of-0099.npz not present")
    arrays = generate_shard_arrays(0, 200, PARAMS)
    with np.load(LEGACY_SHARD0) as ref:
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, ref[key][:200], err_msg=key)


@pytest.mark.slow
def test_full_shard0_matches_existing_dataset():
    if not os.path.exists(LEGACY_SHARD0):
        pytest.skip("data/shards/0000-of-0099.npz not present")
    arrays = generate_shard_arrays(0, 10_000, PARAMS)
    with np.load(LEGACY_SHARD0) as ref:
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, ref[key], err_msg=key)


def test_wall_redirect_never_turns_back_into_the_wall():
    lo, hi = PARAMS.perimeter_m, PARAMS.env_size - PARAMS.perimeter_m
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(20000):
        pos = rng.uniform(0, PARAMS.env_size, 2)
        ang = rng.uniform(-np.pi, np.pi)
        if not wall_zone(ang, pos, lo, hi):
            continue
        new = ang + wall_redirect(ang, pos, lo, hi)
        new = (new + np.pi) % (2 * np.pi) - np.pi
        step = np.array([np.cos(new), np.sin(new)])
        if pos[0] > hi and not (pos[1] < lo or pos[1] > hi):
            assert step[0] <= 1e-9
        if pos[0] < lo and not (pos[1] < lo or pos[1] > hi):
            assert step[0] >= -1e-9
        if pos[1] > hi and not (pos[0] < lo or pos[0] > hi):
            assert step[1] <= 1e-9
        if pos[1] < lo and not (pos[0] < lo or pos[0] > hi):
            assert step[1] >= -1e-9
        checked += 1
    assert checked > 100


def test_trajectories_stay_inside_the_box():
    np.random.seed(7)
    for _ in range(50):
        pos, _, _, _ = simulate_rh_trajectory(PARAMS)
        assert pos.min() > -0.05 and pos.max() < PARAMS.env_size + 0.05


def test_cli_writes_valid_shards(tmp_path):
    out = tmp_path / "ds"
    subprocess.run([sys.executable, "generate_data.py", "--shards", "3", "--trajectories-per-shard",
                    "50", "--output-dir", str(out), "--no-latest", "--workers", "1"],
                   cwd=REPO, check=True, capture_output=True)
    paths = sorted((out / "shards").glob("*.npz"))
    assert [p.name for p in paths] == ["0003-of-0099.npz"]
    check_shards([str(p) for p in paths], n_traj=50)
    assert (out / "meta.json").exists()
