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
class RLEnvConfig:
    """DeepMind Lab environment settings for the RL-agent phase (roadmap plan M1/M4).

    `grid_size`/`env_size_m`/`wall_world_units` must match the level script actually loaded
    (`rl/levels/square_arena.lua`'s GRID/wall geometry) -- coupled by construction, not derived
    automatically, since a level file and its Python-side config live in different languages.
    """
    level: str = "square_arena"
    level_directory: str = "rl/levels"  # relative to grid-cells-torch/, matching rl_train.py's cwd
    width: int = 64
    height: int = 64
    grid_size: int = 10
    env_size_m: float = 2.5
    # world-unit span of the interior (see square_arena.lua's cellOrigin: cell c -> c*100+50,
    # interior cells 1..grid_size, wall at c=0/grid_size+1 -> world units [100, 100+grid_size*100]).
    wall_world_units: tuple = (100.0, 1100.0)
    action_repeat: int = 4
    num_actors: int = 32


@dataclass
class VisionRLConfig:
    embed_dim: int = 256
    mask_prob: float = 0.95
    learning_rate: float = 1e-4  # not paper-specified; Adam default order of magnitude
    minibatch_size: int = 32  # Methods: vision learner minibatch of 32 single frames


@dataclass
class GridRLConfig:
    minibatch_size: int = 10  # Methods: grid learner minibatch of 10 sequences
    seq_len: int = 100
    learning_rate: float = 1e-3  # Supplementary Table 2: "Learning rate grid network"


@dataclass
class ActorCriticConfig:
    lstm_units: int = 256
    embed_dim: int = 256
    learning_rate_range: tuple = (0.000001, 0.0002)  # Supplementary Table 2
    gradient_momentum: float = 0.99  # Supplementary Table 2
    discount: float = 0.99
    entropy_reg_range: tuple = (0.00006, 0.0001)  # Supplementary Table 2, beta
    baseline_cost_range: tuple = (0.48, 0.52)  # Supplementary Table 2, alpha
    # Supplementary Table 2: "Back-propagation step in the actor-critic learner" = 100. Was 20
    # here (documented then as "not paper-specified exactly, standard A3C t_max order of
    # magnitude") -- that was wrong; the paper does specify it, found on a full re-read of Table
    # 2. See the RL-agent roadmap plan's throughput/spec-audit notes.
    n_step: int = 100


@dataclass
class ReplayBufferConfig:
    frame_capacity: int = 200_000
    sequence_capacity: int = 50_000


@dataclass
class RLConfig:
    env: RLEnvConfig = field(default_factory=RLEnvConfig)
    vision: VisionRLConfig = field(default_factory=VisionRLConfig)
    grid: GridRLConfig = field(default_factory=GridRLConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    # Supplementary Table 2: "Place cell scale" = 40, listed separately from Table 1's
    # sigma(c)=0.01 (meters, explicitly labelled) for the supervised network. Table 2 doesn't
    # restate units; 40 metres is impossible in this 2.5m arena (env.env_size_m), so read as
    # 40cm=0.4m -- a documented judgment call, not a value the paper states directly. Used by
    # ensembles.build_rl_ensembles(), not the supervised build_ensembles()/cfg.task.pc_scale.
    pc_scale: tuple = (0.4,)
    total_env_steps: int = 1_000_000_000  # Methods: 1e9 per experiment; smoke-test with far fewer
    seed: int = 0
    # Paired with results/<name>/ the same way TrainConfig.results_dir is for the supervised
    # pipeline (see README "Where things live") -- data/ is gitignored/regenerable, results/
    # tracked. A fresh checkpoint file is written here every checkpoint_every_env_steps.
    results_dir: str = "data/checkpoints/rl_baseline"
    # A real checkpoint (vision+grid+actor-critic) measures 11.85MB. At the paper's 1e9-step
    # budget, the old 50_000 default would write 1e9/50_000 = 20,000 unique files (~237GB) --
    # noticed while explaining what training produces, before ever launching the real run.
    # 500_000 gives 2,000 files (~24GB) and, at the ~548 steps/sec measured for 32 actors, still
    # lands roughly every 15 minutes -- frequent enough to resume from, far less disk churn.
    checkpoint_every_env_steps: int = 500_000
    log_every_n_updates: int = 20  # how often actor/learner processes append a log line


@dataclass
class Config:
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
