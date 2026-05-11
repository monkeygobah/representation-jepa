#!/usr/bin/env bash
# Source this file on DeltaAI before running repo commands:
#   source scripts/delta/env.sh

set -euo pipefail

export DELTA_PROJECT="${DELTA_PROJECT:-bhgj}"
export DELTA_ACCOUNT="${DELTA_ACCOUNT:-${DELTA_PROJECT}-dtai-gh}"
export DELTA_REPO="${DELTA_REPO:-/u/${USER}/representation-jepa}"
export DELTA_WORK="${DELTA_WORK:-/work/nvme/${DELTA_PROJECT}/${USER}/representation-jepa}"
export DELTA_PYTHONUSERBASE="${DELTA_PYTHONUSERBASE:-/work/nvme/${DELTA_PROJECT}/${USER}/python-userbase}"

export DELTA_DATA="${DELTA_DATA:-${DELTA_WORK}/data}"
export DELTA_RUNS="${DELTA_RUNS:-${DELTA_WORK}/runs}"
export DELTA_MODELS="${DELTA_MODELS:-${DELTA_WORK}/models}"
export TORCH_HOME="${TORCH_HOME:-${DELTA_WORK}/torch_cache}"

module load python/miniforge3_pytorch/2.10.0

mkdir -p \
  "${DELTA_DATA}" \
  "${DELTA_RUNS}" \
  "${DELTA_MODELS}" \
  "${DELTA_PYTHONUSERBASE}" \
  "${TORCH_HOME}"

export PYTHONUSERBASE="${DELTA_PYTHONUSERBASE}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.12/site-packages:${PYTHONPATH:-}"

echo "DeltaAI environment loaded"
echo "  DELTA_PROJECT=${DELTA_PROJECT}"
echo "  DELTA_ACCOUNT=${DELTA_ACCOUNT}"
echo "  DELTA_REPO=${DELTA_REPO}"
echo "  DELTA_WORK=${DELTA_WORK}"
