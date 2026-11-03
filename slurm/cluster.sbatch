#!/bin/bash
#SBATCH --job-name=cluster
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):
#   sbatch slurm/cluster.sbatch configs/week02.yaml --smoke 2000   # smoke test
#   sbatch slurm/cluster.sbatch configs/week02.yaml                # full run

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/cluster.sbatch configs/week02.yaml [--smoke N]}"
shift
cd /raid/home/loitongbam/Project_NLP

# him2 has more cores than OpenBLAS supports (max 64): cap all math-library threads,
# otherwise KMeans crashes with "double free or corruption".
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Keep this source line identical to the one in slurm/split.sbatch.
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate shiv; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname), threads=${OMP_NUM_THREADS}"
nvidia-smi 2>/dev/null || echo "nvidia-smi not permitted for this user - continuing"

python -u src/data/cluster.py --config "$CONFIG" "$@"