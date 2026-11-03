#!/bin/bash
# Pilot: the first part of two real teacher runs, lam 0 seed 0 (index 0) and
# lam 1 seed 0 (index 6). Each stops cleanly at update PILOT_UPDATES (default
# 70 of 347, about 20,000 episodes), keeps its checkpoint, and is continued
# later by scripts/slurm/train_prosocial.sh with the same index. What the pilot
# checks, fixed before it runs: docs/pilot.md.
#
#   sbatch scripts/slurm/pilot_prosocial.sh
#   PILOT_UPDATES=100 sbatch --export=ALL scripts/slurm/pilot_prosocial.sh
#
# Afterwards: .venv/bin/python scripts/plot_training.py runs/train/prosocial_v1
#
#SBATCH --job-name=pilot_prosocial_v1
#SBATCH --array=0,6
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
set +u   # Lmod's module function can trip over set -u
if command -v module >/dev/null 2>&1; then module load python/3.11; fi
set -u
nvidia-smi --query-gpu=name,memory.total --format=csv || true
.venv/bin/python scripts/train.py --config configs/train_prosocial_v1.yaml \
    --array-index "${SLURM_ARRAY_TASK_ID:-0}" --stop-after "${PILOT_UPDATES:-70}"
