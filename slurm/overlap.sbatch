#!/bin/bash
#SBATCH --job-name=overlap
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):  sbatch slurm/overlap.sbatch configs/pilot_bn.yaml

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/overlap.sbatch <config>}"
shift
cd /raid/home/loitongbam/Project_NLP

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate shiv; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

python -u src/eval/check_overlap.py --config "$CONFIG" "$@"