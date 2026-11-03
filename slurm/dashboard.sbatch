#!/bin/bash
#SBATCH --job-name=dashboard
#SBATCH --partition=debug          # check other partitions with: sinfo
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00            # dashboard stops automatically after 2 hours
#SBATCH --output=outputs/logs/%x_%j.out   # %x = job name, %j = job id
#SBATCH --error=outputs/logs/%x_%j.err
#SBATCH --cpus-per-task=8

set -euo pipefail

PORT=${1:-8765}   # optional: sbatch slurm/dashboard.sbatch 8766  (if 8765 is taken)

cd /raid/home/loitongbam/Project_NLP_2
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate pro 2>/dev/null || conda activate shiv; set -u

echo "Job $SLURM_JOB_ID on $(hostname) | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "nvidia-smi not permitted for this user - GPU is verified inside the Python script instead"

# Stop early with a clear message if someone else is already using this port.
python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1', $PORT)); s.close(); sys.exit(1 if r==0 else 0)" \
  || { echo "Port $PORT is already in use - resubmit with another port, e.g.: sbatch slurm/dashboard.sbatch 8766"; exit 1; }

echo "Dashboard starting on $(hostname), port $PORT"
echo "Open it: forward port $PORT in VS Code (Ports tab), then visit http://localhost:$PORT"
echo "Stop it when done: scancel $SLURM_JOB_ID"

python -m streamlit run dashboard/app.py \
  --server.port "$PORT" \
  --server.address 127.0.0.1 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false