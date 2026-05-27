#!/usr/bin/env bash
# Submit with:
#   sbatch -A bhgj-dtai-gh scripts/delta/train_ssl_fixedcompute_vit_48h.sbatch CONFIG.yaml

#SBATCH --partition=ghx4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --job-name=vitb-fixedcompute
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

CFG="${1:?Usage: sbatch -A bhgj-dtai-gh scripts/delta/train_ssl_fixedcompute_vit_48h.sbatch CONFIG.yaml}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

source scripts/delta/env.sh

# DataLoader workers run CPU transforms; prevent each worker from spawning a
# full BLAS/OpenMP thread pool and oversubscribing the allocated CPU cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Config: ${CFG}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
nvidia-smi

python scripts/train_ssl.py --cfg "${CFG}"
