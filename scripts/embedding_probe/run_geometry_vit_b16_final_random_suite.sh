#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_LOG_BASE="$ROOT_DIR/_logs/embedding_probe_vit_b16_final_random"
CFG="$ROOT_DIR/embedding_extract/configs/geometry_vit_b16_final_random.yaml"

LOG_BASE="$DEFAULT_LOG_BASE"
LOG_DIR=""
DETACH=1
OVERWRITE_EXTRACT=0
OVERWRITE_ANALYZE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run the final long-train ViT-B/16 random-init geometry pipeline with background
logging by default.

Stages:
  1. proj extract
  2. proj analyze
  3. proj aggregate

Options:
  --overwrite-extract    Pass --overwrite to extraction
  --overwrite-analyze    Pass --overwrite to analysis
  --log-base PATH        Parent directory for batch logs
  --log-dir PATH         Exact log directory to use for the batch
  --foreground           Run in the current shell instead of detaching
  --help                 Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite-extract)
      OVERWRITE_EXTRACT=1
      shift
      ;;
    --overwrite-analyze)
      OVERWRITE_ANALYZE=1
      shift
      ;;
    --log-base)
      LOG_BASE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --foreground)
      DETACH=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$CFG" ]]; then
  echo "Missing required config: $CFG" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Activate the correct environment first." >&2
  exit 1
fi

SUITE_NAME="embedding_probe_vit_b16_final_random"

if [[ -z "$LOG_DIR" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="$LOG_BASE/${SUITE_NAME}_$TIMESTAMP"
fi

mkdir -p "$LOG_DIR"

if [[ "$DETACH" -eq 1 ]]; then
  DETACH_ARGS=(
    --foreground
    --log-dir "$LOG_DIR"
  )
  [[ "$OVERWRITE_EXTRACT" -eq 1 ]] && DETACH_ARGS+=(--overwrite-extract)
  [[ "$OVERWRITE_ANALYZE" -eq 1 ]] && DETACH_ARGS+=(--overwrite-analyze)

  nohup bash "$0" "${DETACH_ARGS[@]}" > "$LOG_DIR/launcher.out" 2>&1 < /dev/null &
  PID=$!
  echo "Started final ViT-B/16 geometry suite in background."
  echo "PID: $PID"
  echo "Log dir: $LOG_DIR"
  echo "Launcher log: $LOG_DIR/launcher.out"
  exit 0
fi

BATCH_LOG="$LOG_DIR/batch.log"
SUMMARY_TSV="$LOG_DIR/summary.tsv"

{
  echo "suite_name"$'\t'"$SUITE_NAME"
  echo "started_at"$'\t'"$(date --iso-8601=seconds)"
  echo "overwrite_extract"$'\t'"$OVERWRITE_EXTRACT"
  echo "overwrite_analyze"$'\t'"$OVERWRITE_ANALYZE"
} >> "$BATCH_LOG"

printf "stage\tstatus\texit_code\tlog_path\n" > "$SUMMARY_TSV"
FAILURES=0

run_stage() {
  local stage_name="$1"
  local log_path="$2"
  shift 2

  echo "[$(date --iso-8601=seconds)] START $stage_name" | tee -a "$BATCH_LOG"
  echo "  log: $log_path" | tee -a "$BATCH_LOG"
  echo "  cmd: $*" | tee -a "$BATCH_LOG"

  if "$@" > "$log_path" 2>&1; then
    echo "[$(date --iso-8601=seconds)] DONE  $stage_name" | tee -a "$BATCH_LOG"
    printf "%s\t%s\t%s\t%s\n" "$stage_name" "ok" "0" "$log_path" >> "$SUMMARY_TSV"
    return 0
  fi

  local exit_code=$?
  FAILURES=$((FAILURES + 1))
  echo "[$(date --iso-8601=seconds)] FAIL  $stage_name (exit=$exit_code)" | tee -a "$BATCH_LOG"
  printf "%s\t%s\t%s\t%s\n" "$stage_name" "failed" "$exit_code" "$log_path" >> "$SUMMARY_TSV"
  return "$exit_code"
}

extract_cmd=(python "$ROOT_DIR/scripts/run_embedding_extract.py" --cfg "$CFG")
[[ "$OVERWRITE_EXTRACT" -eq 1 ]] && extract_cmd+=(--overwrite)
run_stage "geometry_vit_b16_final_random_extract" "$LOG_DIR/geometry_vit_b16_final_random_extract.log" "${extract_cmd[@]}" || exit $?

analyze_cmd=(python "$ROOT_DIR/scripts/run_embedding_analyze.py" --cfg "$CFG")
[[ "$OVERWRITE_ANALYZE" -eq 1 ]] && analyze_cmd+=(--overwrite)
run_stage "geometry_vit_b16_final_random_analyze" "$LOG_DIR/geometry_vit_b16_final_random_analyze.log" "${analyze_cmd[@]}" || exit $?

aggregate_cmd=(python "$ROOT_DIR/scripts/run_embedding_aggregate.py" --cfg "$CFG")
run_stage "geometry_vit_b16_final_random_aggregate" "$LOG_DIR/geometry_vit_b16_final_random_aggregate.log" "${aggregate_cmd[@]}" || exit $?

echo "finished_at"$'\t'"$(date --iso-8601=seconds)" >> "$BATCH_LOG"
echo "failures"$'\t'"$FAILURES" >> "$BATCH_LOG"

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Final ViT-B/16 geometry suite finished with $FAILURES failed stages. See $SUMMARY_TSV"
  exit 1
fi

echo "Final ViT-B/16 geometry suite finished successfully. See $SUMMARY_TSV"
