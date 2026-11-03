#!/bin/bash
#SBATCH --job-name=train_laser_seq
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Sequential LASER training runs in ONE job.
# Usage:
#   sbatch slurm/train_laser_seq.sbatch configs/pilot_bn.yaml co:42 co:43 co:44 co_filter:42 co_filter:43 co_filter:44 co_topic:42 co_topic:43 co_topic:44

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/train_laser_seq.sbatch <config> variant:seed [variant:seed ...]}"
shift
[ "$#" -ge 1 ] || { echo "no runs given (variant:seed)"; exit 1; }
cd /raid/home/loitongbam/Project_NLP_2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate pro; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}, $# runs: $*"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

for RUN in "$@"; do
  IFS=: read -r VARIANT SEED <<< "$RUN"
  echo "===== laser ${VARIANT} seed ${SEED}  start $(date)"
  python -u src/laser/train_laser_co.py --config "$CONFIG" --variant "$VARIANT" --seed "$SEED" \
    || echo "!!!!! FAILED: laser ${VARIANT} seed ${SEED}"
  echo "===== laser ${VARIANT} seed ${SEED}  end $(date)"
done
