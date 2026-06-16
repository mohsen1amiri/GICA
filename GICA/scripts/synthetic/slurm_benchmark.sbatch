#!/usr/bin/env bash
#SBATCH -A naiss2025-22-1453
#SBATCH -p alvis
#SBATCH -C NOGPU
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -t 1-00:00:00
#SBATCH -J plot_metrics
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_HGIFA/logs/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_HGIFA/logs/%x_%j.err

set -euo pipefail

REPO="/mimer/NOBACKUP/groups/naiss2024-22-1645/Paper_HGIFA"
WORKDIR="$REPO/Paper_HGIFA"
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

# optional to keep:
export CUDA_VISIBLE_DEVICES=""

python -u -m gica.synthetic.benchmark