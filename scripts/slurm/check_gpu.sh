#!/bin/bash
# GPU check before the real runs: JAX sees the GPU, the trainer tests pass on
# the GPU code path, and the benchmarks print the time a full run takes.
#
#   sbatch scripts/slurm/check_gpu.sh
#
#SBATCH --job-name=check_gpu
# Nibi: H100 80 GB. A plain "1" could land on an AMD MI300A node.
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# Account: set once with  export SBATCH_ACCOUNT=def-YOURPI  (see docs/compute_canada.md)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs/slurm
set +u   # Lmod's module function can trip over set -u
if command -v module >/dev/null 2>&1; then module load python/3.11; fi
set -u
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# Does JAX see the GPU as installed, or only with the cuda module loaded?
probe() { .venv/bin/python -c "import jax; print('jax', jax.__version__, 'backend', jax.default_backend(), jax.devices())" 2>&1 || true; }
out=$(probe); echo "$out"
if [[ "$out" == *"backend gpu"* ]]; then
    echo "GPU found without the cuda module"
else
    echo "no GPU without the cuda module, trying: module load cuda"
    set +u; module load cuda; set -u
    module list 2>&1 | grep -i cuda || true
    out=$(probe); echo "$out"
    if [[ "$out" != *"backend gpu"* ]]; then echo "JAX does not see a GPU"; exit 1; fi
    echo "GPU found with the cuda module"
fi
.venv/bin/python -m pytest -q tests/test_train.py tests/test_social.py
.venv/bin/python scripts/train.py --config configs/train_prosocial_v1.yaml --bench 3
