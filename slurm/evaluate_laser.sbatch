#!/bin/bash
#SBATCH --job-name=evaluate_laser
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):
#   sbatch slurm/evaluate_laser.sbatch configs/pilot_bn.yaml
#   sbatch slurm/evaluate_laser.sbatch configs/pilot_bn.yaml --name laser_co --student_ckpt outputs/checkpoints/pilot_bn_laser/laser_co/best_student.pt

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/evaluate_laser.sbatch <config> [--name name] [--student_ckpt path]}"
shift
cd /raid/home/loitongbam/Project_NLP_2

# him2 has more cores than OpenBLAS supports (max 64): cap math-library threads.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate pro; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

python -u src/eval/evaluate_laser.py --config "$CONFIG" "$@"
