#!/usr/bin/env bash
# Submit with:
#   sbatch -A bhgj-dtai-gh scripts/delta/train_ssl_ddp2_48h.sbatch configs/delta/train_lejepa_vit_b16_multicrop_ddp4_b64.yaml

#SBATCH --partition=ghx4
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256g
#SBATCH --job-name=rep-jepa-ddp2
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

CFG="${1:?Usage: sbatch -A bhgj-dtai-gh scripts/delta/train_ssl_ddp2_48h.sbatch CONFIG.yaml}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/slurm

source scripts/delta/env.sh

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
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
