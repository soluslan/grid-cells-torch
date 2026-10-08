"""Environment-facing helpers shared by the A2C learner, its vectorized environments and
evaluation: DeepMind Lab construction, observation conversion, and velocity inputs."""
import os

import numpy as np
import torch

OBSERVATIONS = ["RGB_INTERLEAVED", "VEL.TRANS", "VEL.ROT", "DEBUG.POS.TRANS", "DEBUG.POS.ROT"]
FRAMES_PER_SECOND = 60.0


def make_dmlab_env(env_cfg, renderer: str = "hardware", width: int | None = None,
                   height: int | None = None, level: str | None = None):
    import deepmind_lab
    return deepmind_lab.Lab(
        level or env_cfg.level, OBSERVATIONS,
        config={"width": str(width or env_cfg.width), "height": str(height or env_cfg.height),
                "levelDirectory": os.path.abspath(env_cfg.level_directory)},
        renderer=renderer)


class DMLabFactory:
    """Picklable zero-argument env constructor for subprocess workers."""

    def __init__(self, env_cfg, renderer: str = "hardware", level: str | None = None):
        self.env_cfg = env_cfg
        self.renderer = renderer
        self.level = level

    def __call__(self):
        return make_dmlab_env(self.env_cfg, renderer=self.renderer, level=self.level)


def world_to_meters(pos_world, env_cfg):
    lo, hi = env_cfg.wall_world_units
    return (pos_world - (lo + hi) / 2.0) * (env_cfg.env_size_m / (hi - lo))


def convert_observation(obs: dict, env_cfg) -> dict:
    """One DMLab observation dict -> numpy arrays in the agent's units (image stays uint8)."""
    return {
        "image": np.ascontiguousarray(obs["RGB_INTERLEAVED"].transpose(2, 0, 1)),
        "pos": world_to_meters(np.asarray(obs["DEBUG.POS.TRANS"][:2], dtype=np.float32),
                               env_cfg).astype(np.float32),
        "hd": np.array([np.deg2rad(obs["DEBUG.POS.ROT"][1])], dtype=np.float32),
        "vel_trans": np.asarray(obs["VEL.TRANS"], dtype=np.float32),
        "vel_rot": np.asarray(obs["VEL.ROT"], dtype=np.float32),
    }


def image_to_input(image_uint8: torch.Tensor) -> torch.Tensor:
    return image_uint8.float() / 127.5 - 1.0


def ego_velocity(vel_trans: torch.Tensor, vel_rot: torch.Tensor, env_cfg) -> torch.Tensor:
    """(B, 3) DMLab translational velocity (world units/s, agent frame) and (B, 3) angular
    velocity (deg/s) -> (B, 4) grid-network input [u, v, sin dphi, cos dphi], where dphi is the
    heading change over one action (action_repeat frames). Same convention as the A3C loop."""
    scale = env_cfg.env_size_m / (env_cfg.wall_world_units[1] - env_cfg.wall_world_units[0])
    dt = env_cfg.action_repeat / FRAMES_PER_SECOND
    dtheta = torch.deg2rad(vel_rot[:, 1] * dt)
    return torch.stack([vel_trans[:, 0] * scale, vel_trans[:, 1] * scale,
                        torch.sin(dtheta), torch.cos(dtheta)], dim=-1)
