#!/usr/bin/env bash
# Build WebDataset-compatible tar shards from the Delta training manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATA_ROOT="${1:-${DELTA_DATA}/subset6_minus_7_train_flat}"
MANIFEST="${2:-${DELTA_DATA}/manifests/subset6_minus_7_train_flat.txt}"
OUT_DIR="${3:-${DELTA_DATA}/shards/subset6_minus_7_train_flat}"
SHARD_SIZE="${SHARD_SIZE:-10000}"

python scripts/build_webdataset_shards.py \
  --manifest "${MANIFEST}" \
  --root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --shard-size "${SHARD_SIZE}" \
  --prefix train
