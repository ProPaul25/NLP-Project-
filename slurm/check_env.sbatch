#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --partition=debug          # check other partitions with: sinfo
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/logs/%x_%j.out   # %x = job name, %j = job id
#SBATCH --error=outputs/logs/%x_%j.err

set -euo pipefail

CONFIG=${1:?"Usage: sbatch slurm/check_env.sbatch <path/to/config.yaml>"}

cd /raid/home/loitongbam/Project_NLP
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shiv

echo "Job $SLURM_JOB_ID on $(hostname) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
   nvidia-smi || echo "nvidia-smi not permitted for this user - GPU is verified inside the Python script instead"
python -u src/utils/check_env.py --config "$CONFIG"