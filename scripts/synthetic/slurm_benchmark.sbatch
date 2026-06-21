#!/usr/bin/env bash
#SBATCH -A <PROJECT_ACCOUNT>
#SBATCH -p <PARTITION>
#SBATCH -C NOGPU
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 1-00:00:00
#SBATCH -J synthetic_benchmark
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Root of the checkout. Override by exporting REPO before submitting,
# otherwise it defaults to the directory you submit from.
REPO="${REPO:-$PWD}"
WORKDIR="$REPO"
VENV="$REPO/.venv"

mkdir -p "$REPO/logs" "$REPO/outputs" "$REPO/.cache/huggingface"
cd "$WORKDIR"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: venv not found at $VENV"
  exit 1
fi
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$REPO/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# CPU-only run: hide any GPUs.
export CUDA_VISIBLE_DEVICES=""

python -u -m gica.synthetic.benchmark
