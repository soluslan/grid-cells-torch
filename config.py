"""Configuration dataclasses, replacing the original repo's tf.flags.

Defaults mirror google-deepmind/grid-cells' train.py flag defaults exactly,
except where noted.
"""

from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    env_size: float = 2.2  # environment width/height (meters)
    n_pc: tuple = (256,)  # number of place cells per ensemble
    pc_scale: tuple = (0.01,)  # place cell stdev (meters) per ensemble
    n_hdc: tuple = (12,)  # number of head-direction cells per ensemble
    hdc_concentration: tuple = (20.0,)  # von Mises concentration per ensemble
    neurons_seed: int = 8341
    targets_type: str = "softmax"
    lstm_init_type: str = "softmax"
    velocity_inputs: bool = True
    velocity_noise: tuple = (0.0, 0.0, 0.0)  # additive N(0,1)*noise per ego_vel component


@dataclass
class ModelConfig:
    nh_lstm: int = 128
    nh_bottleneck: int = 256
    dropout_rates: tuple = (0.5,)
    # NOTE: in the original repo this weight decay was registered as a Sonnet
    # `regularizer` but never actually summed into the optimized loss (dead
    # code - see README). Here it's wired into the optimizer for real. Set to
    # 0.0 to reproduce the original's literal (no regularization) behavior.
    weight_decay: float = 1e-5
    bottleneck_has_bias: bool = False
    init_weight_disp: float = 0.0
    ego_vel_dim: int = 3


@dataclass
class TrainConfig:
    epochs: int = 1000
    steps_per_epoch: int = 1000
    minibatch_size: int = 10
    learning_rate: float = 1e-5
    momentum: float = 0.9
    # original repo used 1e-5 here, but that's ~7000x smaller than the actual
    # gradient magnitudes measured in scripts/smoke_test.py (max |grad| ~0.07
    # at init) -- with clip=1e-5 the clip saturates on essentially every step,
    # making training extremely slow. Loosened to 1.0 (user's choice) so the
    # clip only guards against extreme outliers rather than clamping every
    # step to a fixed tiny value. Set back to 1e-5 to reproduce the original's
    # literal (very slow) behavior.
    grad_clip_value: float = 1.0  # torch.nn.utils.clip_grad_value_ -- element-wise abs clamp
    save_every_n_epochs: int = 2
    results_dir: str = "results"


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
