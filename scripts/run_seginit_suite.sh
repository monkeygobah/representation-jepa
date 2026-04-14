#!/usr/bin/env bash
set -u

SIZE_DIR="1m"
CONFIG_DIR="configs/baselines/$SIZE_DIR"
GPU_SET="0,1,2"
NPROC_PER_NODE=3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory does not exist: $CONFIG_DIR" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="_logs/baselines/${SIZE_DIR}_seginit_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

BATCH_LOG="$LOG_DIR/batch.log"
SUMMARY_TSV="$LOG_DIR/summary.tsv"

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*seginit.yaml' | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No seginit configs found in $CONFIG_DIR" >&2
  exit 1
fi

printf "config\tstatus\texit_code\tlog_path\n" > "$SUMMARY_TSV"

echo "Starting seginit suite" | tee -a "$BATCH_LOG"
echo "  size_dir: $SIZE_DIR" | tee -a "$BATCH_LOG"
echo "  gpu_set: $GPU_SET" | tee -a "$BATCH_LOG"
echo "  nproc_per_node: $NPROC_PER_NODE" | tee -a "$BATCH_LOG"
echo "  log_dir: $LOG_DIR" | tee -a "$BATCH_LOG"

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  run_log="$LOG_DIR/${name}.log"

  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$BATCH_LOG"
  if CUDA_VISIBLE_DEVICES="$GPU_SET" torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/train_ssl.py --cfg "$cfg" > "$run_log" 2>&1; then
    status="ok"
    exit_code=0
    echo "[$(date --iso-8601=seconds)] DONE  $name" | tee -a "$BATCH_LOG"
  else
    status="failed"
    exit_code=$?
    echo "[$(date --iso-8601=seconds)] FAIL  $name (exit=$exit_code)" | tee -a "$BATCH_LOG"
  fi

  printf "%s\t%s\t%s\t%s\n" "$name" "$status" "$exit_code" "$run_log" >> "$SUMMARY_TSV"
done

echo "Finished seginit suite" | tee -a "$BATCH_LOG"
echo "Summary: $SUMMARY_TSV" | tee -a "$BATCH_LOG"
