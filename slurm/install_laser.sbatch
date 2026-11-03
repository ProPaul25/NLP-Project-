#!/bin/bash
#SBATCH --job-name=install_laser
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err

set -eo pipefail
cd /raid/home/loitongbam/Project_NLP_2
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "job $SLURM_JOB_ID"; nvidia-smi || echo "nvidia-smi not permitted - continuing"
conda env list | grep -q "^laser " || conda create -y -n laser python=3.8
conda activate pro
python -m pip install "pip<24.1" # fairseq 0.12.2's dependencies fail with newer pip
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install fairseq==0.12.2 laser_encoders==0.0.2 pandas pyyaml scikit-learn
python -c "import torch, fairseq, laser_encoders; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'fairseq', fairseq.__version__)"
