#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):
#   sbatch slurm/compare.sbatch configs/pilot_bn.yaml

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/compare.sbatch <config>}"
shift
cd /raid/home/loitongbam/Project_NLP

# him2 has more cores than OpenBLAS supports (max 64): cap math-library threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate shiv; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

python -u src/eval/compare.py --config "$CONFIG" "$@"