#!/usr/bin/env bash
# Submit with:
#   sbatch -A bhgj-dtai-gh scripts/delta/train_ssl.sbatch configs/delta/my_config.yaml

#SBATCH --partition=ghx4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=rep-jepa
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

CFG="${1:?Usage: sbatch -A bhgj-dtai-gh scripts/delta/train_ssl.sbatch CONFIG.yaml}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

source scripts/delta/env.sh

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Config: ${CFG}"
nvidia-smi

python scripts/train_ssl.py --cfg "${CFG}"
