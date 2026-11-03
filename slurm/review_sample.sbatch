#!/bin/bash
#SBATCH --job-name=review_sample
#SBATCH --partition=debug          # check other partitions with: sinfo
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/raid/home/loitongbam/Project_NLP/outputs/logs/%x_%j.out   # %x = job name, %j = job id
#SBATCH --error=/raid/home/loitongbam/Project_NLP/outputs/logs/%x_%j.err

set -euo pipefail

CONFIG=${1:?"Usage: sbatch slurm/review_sample.sbatch <path/to/config.yaml>"}

cd /raid/home/loitongbam/Project_NLP
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shiv

echo "Job $SLURM_JOB_ID on $(hostname) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not permitted for this user; continuing (torch check below)"
python -u src/data/review_sample.py --config "$CONFIG"