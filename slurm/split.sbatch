#!/bin/bash
#SBATCH --job-name=split
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
#
# Usage (from the project root):
#   mkdir -p outputs/logs
#   sbatch slurm/split.sbatch configs/week02.yaml            # real run
#   sbatch slurm/split.sbatch configs/week02.yaml --dry_run  # counts only, writes nothing

set -euo pipefail
CONFIG="${1:?usage: sbatch slurm/split.sbatch configs/week02.yaml [--dry_run]}"
shift
cd /raid/home/loitongbam/Project_NLP

# Keep this source line identical to the one in slurm/align.sbatch.
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate shiv; set -u

echo "Job ${SLURM_JOB_ID} on $(hostname)"
nvidia-smi || echo "nvidia-smi not permitted for this user - continuing"

python -u src/data/split.py --config "$CONFIG" "$@"
