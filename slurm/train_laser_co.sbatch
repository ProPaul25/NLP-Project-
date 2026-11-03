#!/bin/bash
#SBATCH --job-name=train_laser_co
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):
#   sbatch slurm/train_laser_co.sbatch configs/pilot_bn.yaml co 42 "--max_steps 20"   # smoke test
#   sbatch slurm/train_laser_co.sbatch configs/pilot_bn.yaml co 42                     # full run

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/train_laser_co.sbatch <config> <variant> [seed] [extra]}"
VARIANT="${2:?variant key: co, co_filter or co_topic}"
SEED="${3:-42}"
shift 3 2>/dev/null || shift $#
cd /raid/home/loitongbam/Project_NLP_2

# him2 has more cores than OpenBLAS supports (max 64): cap math-library threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate pro; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}, run laser_${VARIANT}_s${SEED}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

python -u src/laser/train_laser_co.py --config "$CONFIG" --variant "$VARIANT" --seed "$SEED" $*
