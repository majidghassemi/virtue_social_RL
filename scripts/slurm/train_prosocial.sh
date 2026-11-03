#!/bin/bash
# Train the 9 prosocial populations (3 lam x 3 seeds) as a Slurm array job.
# Array index = i_condition * 3 + i_seed (lam0: 0-2, lam0.5: 3-5, lam1: 6-8).
#
#   sbatch scripts/slurm/train_prosocial.sh                 # all 9 runs
#   sbatch --array=0 scripts/slurm/train_prosocial.sh       # one run (lam 0, seed 0)
#
# Setup and the order of runs on Alliance (Compute Canada) clusters: see
# docs/compute_canada.md. Submit from the repository root.
#
# Edit the account, GPU request and time for your cluster. A run that hits the
# time limit can be resubmitted with the same array index: it resumes from
# runs/train/prosocial_v1/<condition>_seed<seed>/runner.msgpack.
#SBATCH --job-name=prosocial_v1
#SBATCH --array=0-8
# Nibi: H100 80 GB. A plain "1" could land on an AMD MI300A node.
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
# Account: set once with  export SBATCH_ACCOUNT=def-YOURPI  (see docs/compute_canada.md)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs/slurm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Alliance clusters: the venv was built with this Python module, load it again.
if command -v module >/dev/null 2>&1; then module load python/3.11; fi
nvidia-smi --query-gpu=name,memory.total --format=csv || true
.venv/bin/python scripts/train.py --config configs/train_prosocial_v1.yaml \
    --array-index "${SLURM_ARRAY_TASK_ID:-0}"
