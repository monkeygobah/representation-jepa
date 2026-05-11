#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_EXTRACT_CFG="$ROOT_DIR/landmark_probe/configs/studies/followup_50k_extract_all_poolings.yaml"
DEFAULT_PROBE_CFG="$ROOT_DIR/landmark_probe/configs/studies/followup_50k_probe_matrix.yaml"
DEFAULT_LOG_BASE="$ROOT_DIR/_logs/landmark_probe"

EXTRACT_CFG="$DEFAULT_EXTRACT_CFG"
PROBE_CFG="$DEFAULT_PROBE_CFG"
LOG_BASE="$DEFAULT_LOG_BASE"
LOG_DIR=""
DETACH=1
RUN_PREPARE=0
RUN_EXTRACT=1
RUN_PROBE=1
RUN_AGGREGATE=1
OVERWRITE_EXTRACT=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run the periorbital landmark probe matrix with background logging by default.

Options:
  --cfg PATH             Use one study config for extraction, probe, and aggregate
  --extract-cfg PATH     Study config for embedding extraction
  --probe-cfg PATH       Study config for probe training and aggregation
  --prepare              Rebuild the prepared periorbital dataset first
  --extract-only         Run only embedding extraction
  --probe-only           Run only probe training
  --aggregate-only       Run only result aggregation
  --overwrite-extract    Recompute existing embedding artifacts
  --log-base PATH        Parent directory for batch logs
  --log-dir PATH         Exact log directory to use for the batch
  --foreground           Run in the current shell instead of detaching
  --help                 Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg)
      EXTRACT_CFG="$2"
      PROBE_CFG="$2"
      shift 2
      ;;
    --extract-cfg)
      EXTRACT_CFG="$2"
      shift 2
      ;;
    --probe-cfg)
      PROBE_CFG="$2"
      shift 2
      ;;
    --prepare)
      RUN_PREPARE=1
      shift
      ;;
    --extract-only)
      RUN_EXTRACT=1
      RUN_PROBE=0
      RUN_AGGREGATE=0
      shift
      ;;
    --probe-only)
      RUN_PREPARE=0
      RUN_EXTRACT=0
      RUN_PROBE=1
      RUN_AGGREGATE=0
      shift
      ;;
    --aggregate-only)
      RUN_PREPARE=0
      RUN_EXTRACT=0
      RUN_PROBE=0
      RUN_AGGREGATE=1
      shift
      ;;
    --overwrite-extract)
      OVERWRITE_EXTRACT=1
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

if [[ ! -f "$EXTRACT_CFG" ]]; then
  echo "Missing extraction study config: $EXTRACT_CFG" >&2
  exit 1
fi
if [[ ! -f "$PROBE_CFG" ]]; then
  echo "Missing probe study config: $PROBE_CFG" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Activate the correct environment first." >&2
  exit 1
fi

SUITE_NAME="$(basename "$PROBE_CFG" .yaml)"

if [[ -z "$LOG_DIR" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="$LOG_BASE/${SUITE_NAME}_$TIMESTAMP"
fi

mkdir -p "$LOG_DIR"

if [[ "$DETACH" -eq 1 ]]; then
  DETACH_ARGS=(--foreground --extract-cfg "$EXTRACT_CFG" --probe-cfg "$PROBE_CFG" --log-dir "$LOG_DIR")
  [[ "$RUN_PREPARE" -eq 1 ]] && DETACH_ARGS+=(--prepare)
  [[ "$RUN_EXTRACT" -eq 1 && "$RUN_PROBE" -eq 0 && "$RUN_AGGREGATE" -eq 0 ]] && DETACH_ARGS+=(--extract-only)
  [[ "$RUN_EXTRACT" -eq 0 && "$RUN_PROBE" -eq 1 && "$RUN_AGGREGATE" -eq 0 ]] && DETACH_ARGS+=(--probe-only)
  [[ "$RUN_EXTRACT" -eq 0 && "$RUN_PROBE" -eq 0 && "$RUN_AGGREGATE" -eq 1 ]] && DETACH_ARGS+=(--aggregate-only)
  [[ "$OVERWRITE_EXTRACT" -eq 1 ]] && DETACH_ARGS+=(--overwrite-extract)

  nohup bash "$0" "${DETACH_ARGS[@]}" > "$LOG_DIR/launcher.out" 2>&1 < /dev/null &
  PID=$!
  echo "Started landmark probe suite in background."
  echo "PID: $PID"
  echo "Log dir: $LOG_DIR"
  echo "Launcher log: $LOG_DIR/launcher.out"
  exit 0
fi

BATCH_LOG="$LOG_DIR/batch.log"
SUMMARY_TSV="$LOG_DIR/summary.tsv"

{
  echo "suite_name"$'\t'"$SUITE_NAME"
  echo "extract_cfg"$'\t'"$EXTRACT_CFG"
  echo "probe_cfg"$'\t'"$PROBE_CFG"
  echo "started_at"$'\t'"$(date --iso-8601=seconds)"
  echo "run_prepare"$'\t'"$RUN_PREPARE"
  echo "run_extract"$'\t'"$RUN_EXTRACT"
  echo "run_probe"$'\t'"$RUN_PROBE"
  echo "run_aggregate"$'\t'"$RUN_AGGREGATE"
  echo "overwrite_extract"$'\t'"$OVERWRITE_EXTRACT"
} >> "$BATCH_LOG"

printf "stage\tstatus\texit_code\tlog_path\n" > "$SUMMARY_TSV"

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
  echo "[$(date --iso-8601=seconds)] FAIL  $stage_name (exit=$exit_code)" | tee -a "$BATCH_LOG"
  printf "%s\t%s\t%s\t%s\n" "$stage_name" "failed" "$exit_code" "$log_path" >> "$SUMMARY_TSV"
  return "$exit_code"
}

DATASET_CFG="$(python - "$PROBE_CFG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    raw = yaml.safe_load(f) or {}
print(raw["dataset_cfg"])
PY
)"
if [[ "$DATASET_CFG" == /workspace/* ]]; then
  DATASET_CFG="$ROOT_DIR/${DATASET_CFG#/workspace/}"
fi

if [[ "$RUN_PREPARE" -eq 1 ]]; then
  run_stage "prepare" "$LOG_DIR/prepare.log" \
    python "$ROOT_DIR/scripts/run_landmark_prepare.py" --cfg "$DATASET_CFG" --overwrite || exit $?
fi

if [[ "$RUN_EXTRACT" -eq 1 ]]; then
  extract_cmd=(python "$ROOT_DIR/scripts/run_landmark_extract.py" --cfg "$EXTRACT_CFG")
  [[ "$OVERWRITE_EXTRACT" -eq 1 ]] && extract_cmd+=(--overwrite)
  run_stage "extract" "$LOG_DIR/extract.log" "${extract_cmd[@]}" || exit $?
fi

if [[ "$RUN_PROBE" -eq 1 ]]; then
  run_stage "probe" "$LOG_DIR/probe.log" \
    python "$ROOT_DIR/scripts/run_landmark_probe.py" --cfg "$PROBE_CFG" || exit $?
fi

if [[ "$RUN_AGGREGATE" -eq 1 ]]; then
  run_stage "aggregate" "$LOG_DIR/aggregate.log" \
    python "$ROOT_DIR/scripts/run_landmark_aggregate.py" --cfg "$PROBE_CFG" || exit $?
fi

echo "finished_at"$'\t'"$(date --iso-8601=seconds)" >> "$BATCH_LOG"
echo "Landmark probe suite finished successfully. See $SUMMARY_TSV"
