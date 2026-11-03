#!/bin/bash
#SBATCH --job-name=install
#SBATCH --partition=debug          # check other partitions with: sinfo
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/%x_%j.out   # %x = job name, %j = job id
#SBATCH --error=outputs/logs/%x_%j.err

set -euo pipefail

cd /raid/home/loitongbam/Project_NLP
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shiv

echo "Job $SLURM_JOB_ID on $(hostname) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not permitted for this user - GPU is verified inside the Python script instead"

# Use the env's own pip via "python -m pip": a broken user-level pip in ~/.local/bin
# comes earlier on PATH and must not be used.
echo "Python: $(which python)"
python -m pip --version

# Lock torch and transformers at their current versions: env 'shiv' is shared with
# other projects, so the install must not upgrade them.
PIN=$(mktemp)
python -c "import torch, transformers; print(f'torch=={torch.__version__}'); print(f'transformers=={transformers.__version__}')" > "$PIN"
echo "Pinned (must not change):"; cat "$PIN"

python -m pip freeze > "outputs/logs/pip_freeze_before_${SLURM_JOB_ID}.txt"
python -m pip install -c "$PIN" sentence-transformers
python -m pip freeze > "outputs/logs/pip_freeze_after_${SLURM_JOB_ID}.txt"

echo "Installed / changed packages:"
diff "outputs/logs/pip_freeze_before_${SLURM_JOB_ID}.txt" "outputs/logs/pip_freeze_after_${SLURM_JOB_ID}.txt" || true

python -c "import torch, transformers, sentence_transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| sentence-transformers', sentence_transformers.__version__, '| CUDA', torch.cuda.is_available())"