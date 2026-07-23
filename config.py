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
    # Nature Fig. 1's main-text network (the one reporting "129/512, 25.2%
    # grid-like units") used a 512-unit bottleneck; the publicly released
    # google-deepmind/grid-cells code defaulted to 256 instead (matching the
    # RL agent's grid code width). Set to 512 to target the paper's headline
    # figure directly; set to 256 to match the released code / RL agent.
    nh_bottleneck: int = 512
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
    # 300 epochs x 1000 steps/epoch = 300,000 total gradient steps, matching
    # the paper's Supplementary Table 1 "parameter updates" figure literally
    # (the public code's train.py flag defaults imply 1,000,000 instead --
    # see README "Known caveats" #1).
    epochs: int = 300
    steps_per_epoch: int = 1000
    minibatch_size: int = 10
    learning_rate: float = 1e-5
    momentum: float = 0.9
    # Original repo's literal value. This is ~150-7000x smaller than measured
    # gradient magnitudes at every point checked so far (init: max|grad|~0.1;
    # a checkpoint trained 1e6 steps under a loosened clip=1.0: max|grad|~14-25,
    # with lstm.weight_ih_l0 having grown ~20x over that run) -- so at this
    # threshold, clip_grad_value_ saturates on essentially every step,
    # regardless of epoch. Previously loosened to 1.0 on the assumption this
    # was just an impractically slow setting; reverted back to the literal
    # value because the loosened run's LSTM weight blow-up (see memory/
    # grad_clip_investigation.md) suggests the tiny clip may be functioning as
    # an implicit stabilizer against exactly that kind of exploding-gradient
    # RNN failure mode, not merely an oversight. Not yet confirmed by an
    # actual from-scratch run at this value -- that's the next experiment.
    grad_clip_value: float = 1e-5  # torch.nn.utils.clip_grad_value_ -- element-wise abs clamp
    save_every_n_epochs: int = 2
    results_dir: str = "results"


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
