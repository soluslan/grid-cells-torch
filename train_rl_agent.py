"""Train the grid cell RL agent in DeepMind Lab (paper Fig. 2).

Default: synchronous A2C on the GPU (grid_cells/rl_a2c.py) with 32 parallel DMLab envs, the
paper's 1e9-frame budget and Supplementary Table 2 hyperparameters. `--algo a3c` runs the frozen
legacy Hogwild A3C loop (grid_cells/rl_a3c.py), kept only as the baseline; it has known defects
(docs/plan/baseline-a3c.md).

Budgets and checkpoint names count environment frames (one action = 4 frames). Logs go to
--results-dir: episodes.csv, ac_diag.csv (per update), and checkpoint_step{N}.pt files. SIGINT or
SIGTERM stops cleanly and writes checkpoint_step{N}_final.pt.

Examples:
    python train_rl_agent.py --results-dir data/checkpoints/a2c_run1
    python train_rl_agent.py --level tiny_goal --total-env-steps 10000000 --results-dir data/checkpoints/gate_tiny
    python train_rl_agent.py --resume-from data/checkpoints/a2c_run1/checkpoint_step50000000.pt
    python train_rl_agent.py --algo a3c
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--algo", choices=("a2c", "a3c"), default="a2c")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--reset-actor-critic", action="store_true",
                   help="a2c: with --resume-from, load vision/grid weights but start the "
                        "actor-critic and its optimizer fresh at frame 0")
    p.add_argument("--level", default=None, help="DMLab level in levels/ (default square_arena)")
    p.add_argument("--total-env-steps", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="actor-critic learning rate (default: midpoint of Table 2's range)")
    p.add_argument("--ac-momentum", type=float, default=None,
                   help="actor-critic RMSProp momentum (default 0; 0.99 reproduces reading Table 2's "gradient momentum" as momentum, which collapsed the policy)")
    p.add_argument("--ac-eps", type=float, default=None, help="actor-critic RMSProp epsilon")
    p.add_argument("--entropy-cost", type=float, default=None,
                   help="entropy regularisation beta (default: midpoint of Table 2's range)")
    p.add_argument("--hidden-goal", action="store_true",
                   help="square_arena: draw the goal invisibly, as the paper's unmarked goal")
    p.add_argument("--device", default=None)
    p.add_argument("--renderer", choices=("hardware", "software"), default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.algo == "a3c":
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config

    cfg = Config()
    if args.results_dir:
        cfg.rl.results_dir = args.results_dir
    cfg.rl.resume_from = args.resume_from
    if args.level:
        cfg.rl.env.level = args.level
    if args.total_env_steps:
        cfg.rl.total_env_steps = args.total_env_steps
    if args.seed is not None:
        cfg.rl.seed = args.seed
    if args.lr is not None:
        cfg.rl.actor_critic.learning_rate_range = (args.lr, args.lr)
    if args.num_envs:
        cfg.rl.a2c.num_envs = args.num_envs
        cfg.rl.env.num_actors = args.num_envs
    if args.ac_momentum is not None:
        cfg.rl.a2c.ac_rmsprop_momentum = args.ac_momentum
    if args.ac_eps is not None:
        cfg.rl.a2c.ac_rmsprop_eps = args.ac_eps
    if args.entropy_cost is not None:
        cfg.rl.actor_critic.entropy_reg_range = (args.entropy_cost, args.entropy_cost)
    if args.hidden_goal:
        cfg.rl.env.goal_visible = False
    if args.device:
        cfg.rl.a2c.device = args.device
    if args.renderer:
        cfg.rl.a2c.renderer = args.renderer

    if args.algo == "a3c":
        from grid_cells.rl_a3c import train_rl_agent
        train_rl_agent(cfg)
    else:
        from grid_cells.rl_a2c import train_a2c
        train_a2c(cfg, reset_actor_critic=args.reset_actor_critic)
    print("training run complete")


if __name__ == "__main__":
    main()
