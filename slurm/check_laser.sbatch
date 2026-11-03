#!/bin/bash
#SBATCH --job-name=check_laser
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

set -eo pipefail
cd /raid/home/loitongbam/Project_NLP_2

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate pro; set -u

echo "job ${SLURM_JOB_ID} on $(hostname)"; nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted - continuing"

python -u src/laser/check_laser.py
