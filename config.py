"""Configuration dataclasses, replacing the original repo's tf.flags.

Defaults mirror google-deepmind/grid-cells' train.py flag defaults, except
where the paper and the released code disagree -- there we follow the paper.
Each such choice is listed in README "Following the paper, not the released
code"; the fields below only name which setting is involved.
"""

from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    env_size: float = 2.2  # environment width/height (meters)
    n_pc: tuple = (256,)  # place cells per ensemble
    pc_scale: tuple = (0.01,)  # place cell stdev (meters) per ensemble
    n_hdc: tuple = (12,)  # head-direction cells per ensemble
    hdc_concentration: tuple = (20.0,)  # von Mises concentration per ensemble
    neurons_seed: int = 8341
    velocity_inputs: bool = True
    velocity_noise: tuple = (0.0, 0.0, 0.0)  # additive N(0,1)*noise per ego_vel component


@dataclass
class ModelConfig:
    nh_lstm: int = 128
    # 256 matches the released code's default, the paper's circular-arena
    # network (Extended Data Fig. 3d: 56/256 grid-like) and the grid network
    # inside its RL agent -- which is where this project is headed. The paper's
    # headline 129/512 (25.2%) is a 512-unit figure and is NOT the comparison
    # for runs at this width; 21.9% is.
    nh_bottleneck: int = 256
    dropout_rates: tuple = (0.5,)
    weight_decay: float = 1e-5  # a no-op in the original; actually applied here
    weight_decay_scope: str = "output_heads"  # see model.decay_parameters()
    # The paper's Methods: "The linear decoder consists of three sets of weights
    # AND BIASES. The first set ... map from the LSTM hidden state m_t to the
    # linear layer activations g_t." The released code has no bias here; as
    # everywhere else in this port, the paper wins.
    bottleneck_has_bias: bool = True
    init_weight_disp: float = 0.0
    ego_vel_dim: int = 3


@dataclass
class TrainConfig:
    epochs: int = 300  # x steps_per_epoch = the paper's 300,000 parameter updates
    steps_per_epoch: int = 1000
    minibatch_size: int = 10
    learning_rate: float = 1e-5
    momentum: float = 0.9
    grad_clip_value: float = 1e-5  # element-wise abs clamp (clip_grad_value_)
    grad_clip_scope: str = "output_heads"  # see model.clip_parameters()
    # 20 keeps ~16 checkpoints per run instead of 151 (62MB vs 592MB), while
    # still landing on epoch 200 and 299 -- the paper's inter-trial stability
    # analysis compares 2e5 and 3e5 training steps.
    save_every_n_epochs: int = 20
    results_dir: str = "data/checkpoints/baseline"  # paired with results/baseline
    # Seeds weight init and batch order, so a run is reproducible and two
    # configurations can be compared without run-to-run noise confounding them.
    # (task.neurons_seed is separate: it places the place cells.)
    seed: int = 0


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
