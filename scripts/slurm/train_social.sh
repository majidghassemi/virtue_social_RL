#!/bin/bash
# Train the 12 social learners (4 conditions x 3 seeds) as a Slurm array job.
# Needs the teachers first: runs/train/prosocial_v1/lam{0,1}_seed{0,1,2}/params_final.msgpack.
# Array index = i_condition * 3 + i_seed (social_lam1: 0-2, social_lam0: 3-5,
# plain_lam1: 6-8, social_absent: 9-11).
#
#   sbatch scripts/slurm/train_social.sh                    # all 12 runs
#   sbatch --array=0 scripts/slurm/train_social.sh          # one run (social_lam1, seed 0)
#
# Before the first submission, on a login node (needs internet):
#   python -m venv .venv && .venv/bin/pip install -e ".[gpu,dev]"
#   git submodule update --init
# and run the benchmark once on a GPU node to set --time:
#   .venv/bin/python scripts/train.py --config configs/train_social_v1.yaml --bench 3   (needs teacher checkpoints)
#
# Edit the account, GPU request and time for your cluster. A run that hits the
# time limit can be resubmitted with the same array index: it resumes from
# runs/train/social_v1/<condition>_seed<seed>/runner.msgpack.
#SBATCH --job-name=social_v1
#SBATCH --array=0-11
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
##SBATCH --account=def-YOURPI      # Alliance clusters need an account

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs/slurm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name,memory.total --format=csv || true
.venv/bin/python scripts/train.py --config configs/train_social_v1.yaml \
    --array-index "${SLURM_ARRAY_TASK_ID:-0}"
