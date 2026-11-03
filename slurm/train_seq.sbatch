#!/bin/bash
#SBATCH --job-name=train_seq
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Several training runs one after another in ONE job (the cluster allows only 2 jobs per user at a time).
# Usage (from the project root):
#   sbatch slurm/train_seq.sbatch configs/pilot_bn.yaml e5:both:43 labse:trans:43 ...
# Each run is base:variant:seed. A failed run is logged as FAILED and the loop continues.

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/train_seq.sbatch <config> base:variant:seed [base:variant:seed ...]}"
shift
[ "$#" -ge 1 ] || { echo "no runs given (base:variant:seed)"; exit 1; }
cd /raid/home/loitongbam/Project_NLP

# him2 has more cores than OpenBLAS supports (max 64): cap math-library threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate shiv; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}, $# runs: $*"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

for RUN in "$@"; do
  IFS=: read -r BASE VARIANT SEED <<< "$RUN"
  echo "===== ${BASE} ${VARIANT} seed ${SEED}  start $(date)"
  python -u src/train/train.py --config "$CONFIG" --base "$BASE" --variant "$VARIANT" --seed "$SEED" \
    || echo "!!!!! FAILED: ${BASE} ${VARIANT} seed ${SEED}"
  echo "===== ${BASE} ${VARIANT} seed ${SEED}  end $(date)"
done