#!/usr/bin/env bash
# Start an interactive DeltaAI GPU shell.

set -euo pipefail

DELTA_PROJECT="${DELTA_PROJECT:-bhgj}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-${DELTA_PROJECT}-dtai-gh}"
DELTA_PARTITION="${DELTA_PARTITION:-ghx4-interactive}"
DELTA_TIME="${DELTA_TIME:-01:00:00}"
DELTA_GPUS="${DELTA_GPUS:-1}"
DELTA_CPUS="${DELTA_CPUS:-16}"
DELTA_MEM="${DELTA_MEM:-64g}"

exec srun \
  -A "${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --time="${DELTA_TIME}" \
  --nodes=1 \
  --gpus-per-node="${DELTA_GPUS}" \
  --cpus-per-task="${DELTA_CPUS}" \
  --mem="${DELTA_MEM}" \
  --pty /bin/bash
