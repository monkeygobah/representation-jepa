#!/usr/bin/env bash
# Build a one-line-per-image manifest for the full training dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

DATA_ROOT="${1:-${DELTA_DATA}/subset6_minus_7_train_flat}"
MANIFEST="${2:-${DELTA_DATA}/manifests/subset6_minus_7_train_flat.txt}"

mkdir -p "$(dirname "${MANIFEST}")"

echo "Building manifest"
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  MANIFEST=${MANIFEST}"

find "${DATA_ROOT}" -type f \( \
  -iname '*.jpg' -o \
  -iname '*.jpeg' -o \
  -iname '*.png' -o \
  -iname '*.bmp' -o \
  -iname '*.tif' -o \
  -iname '*.tiff' -o \
  -iname '*.webp' \
\) > "${MANIFEST}"

wc -l "${MANIFEST}"
