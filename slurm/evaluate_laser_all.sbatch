#!/bin/bash
#SBATCH --job-name=eval_laser_all
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Score untrained LASER baseline and all 9 trained LASER3-CO checkpoints sequentially.
# Usage:
#   sbatch slurm/evaluate_laser_all.sbatch configs/pilot_bn.yaml

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/evaluate_laser_all.sbatch <config>}"
cd /raid/home/loitongbam/Project_NLP_2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate pro; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

# 1. Untrained LASER baseline
echo "===== Evaluating untrained baseline: laser ====="
python -u src/eval/evaluate_laser.py --config "$CONFIG" --name laser

# 2. Checkpoints: variants and seeds
VARIANTS=("co" "co_filter" "co_topic")
SEEDS=(42 43 44)

for VAR in "${VARIANTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    if [ "$S" -eq 42 ]; then
      RUN="laser_${VAR}"
    else
      RUN="laser_${VAR}_s${S}"
    fi
    CKPT="outputs/checkpoints/pilot_bn_laser/${RUN}/best_student.pt"
    if [ -f "$CKPT" ]; then
      echo "===== Evaluating ${RUN} (checkpoint: ${CKPT}) ====="
      python -u src/eval/evaluate_laser.py --config "$CONFIG" --name "${RUN}" --student_ckpt "${CKPT}" \
        || echo "!!!!! FAILED evaluation for ${RUN}"
    else
      echo "Skipping ${RUN}: ${CKPT} does not exist"
    fi
  done
done

echo "All evaluations finished at $(date)"
