#!/usr/bin/env bash
# Submit with:
#   sbatch -A bhgj-dtai-gh scripts/delta/train_ssl_ddp4.sbatch configs/delta/smoke_lejepa_vit_b16_multicrop_ddp4.yaml

#SBATCH --partition=ghx4
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --job-name=rep-jepa-ddp4
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

CFG="${1:?Usage: sbatch -A bhgj-dtai-gh scripts/delta/train_ssl_ddp4.sbatch CONFIG.yaml}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

source scripts/delta/env.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${SLURM_NTASKS}"
export CFG

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Config: ${CFG}"
echo "Tasks: ${SLURM_NTASKS:-unknown}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
nvidia-smi

srun bash -lc '
  export RANK="${SLURM_PROCID}"
  export LOCAL_RANK="${SLURM_LOCALID}"
  python scripts/train_ssl.py --cfg "${CFG}"
'
