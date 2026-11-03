#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):
#   sbatch slurm/train.sbatch configs/pilot_bn.yaml e5 trans --max_steps 20   # smoke test
#   sbatch slurm/train.sbatch configs/pilot_bn.yaml e5 both                    # full run

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/train.sbatch <config> <base> <variant> [--max_steps N]}"
BASE="${2:?base model key, e.g. e5 or labse}"
VARIANT="${3:?variant key: trans, topic or both}"
shift 3
cd /raid/home/loitongbam/Project_NLP

# him2 has more cores than OpenBLAS supports (max 64): cap math-library threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate shiv; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}, run ${BASE}_${VARIANT}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

python -u src/train/train.py --config "$CONFIG" --base "$BASE" --variant "$VARIANT" "$@"