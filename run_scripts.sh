#!/usr/bin/env bash
# Command cheat sheet. Running this file only prints the list; copy the command you need.
# All commands run from the repo root with the env1 interpreter active
# (conda activate env1, or prefix with /home/junoLee/miniconda3/envs/env1/bin/).
set -euo pipefail

cat <<'CMDS'
grid-cells-torch command cheat sheet

Setup
 1. pip install -r requirements.txt
 2. bash scripts/setup_dmlab.sh                     # build + install deepmind_lab (tens of minutes)
 3. bash scripts/setup_dmlab.sh --skip-deps --skip-build   # re-check patch + smoke test only

Supervised network (paper Fig. 1)
 4. python generate_data.py                         # 100 shards x 10,000 trajectories -> data/datasets/<id>, data/latest
 5. python generate_data.py --num-shards 2 --tag trial
 6. python train.py                                 # 300,000 updates (~30 min on a GTX 1650)
 7. python train.py --epochs 2 --steps-per-epoch 50 --results-dir /tmp/smoke
 8. python analyze_supervised.py --checkpoint data/checkpoints/baseline/checkpoint_epoch299.pt
 9. python analyze_supervised.py --checkpoint data/checkpoints/baseline/checkpoint_epoch299.pt --analysis path

RL agent (paper Fig. 2)
10. python train_rl_agent.py                        # see --help for --algo, --level, budgets, resume
11. python train_rl_agent.py --algo a3c             # frozen legacy A3C baseline (CPU, 32 processes)
12. python eval_rl_agent.py --episodes 5            # newest checkpoint: videos + Fig. 2h-style plots
13. tail -f data/checkpoints/<run>/ac_diag.csv      # actor-critic diagnostics while training

Tests
14. pytest -m "not dmlab and not slow"              # fast suite (~15 s)
15. pytest -m "not slow"                            # + DeepMind Lab checks
16. pytest                                          # everything, including multi-minute runs
CMDS
