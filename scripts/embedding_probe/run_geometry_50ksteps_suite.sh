#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_LOG_BASE="$ROOT_DIR/_logs/embedding_probe_50ksteps"

LOG_BASE="$DEFAULT_LOG_BASE"
LOG_DIR=""
DETACH=1
RUN_PROJ=1
RUN_EMB=1
OVERWRITE_EXTRACT=0
OVERWRITE_ANALYZE=0
RUN_PLOTS=1

PROJ_CONFIGS=(
  "embedding_extract/configs/geometry_10k_50ksteps.yaml"
  "embedding_extract/configs/geometry_100k_50ksteps.yaml"
  "embedding_extract/configs/geometry_1m_50ksteps.yaml"
)

EMB_CONFIGS=(
  "embedding_extract/configs/geometry_10k_50ksteps_emb.yaml"
  "embedding_extract/configs/geometry_100k_50ksteps_emb.yaml"
  "embedding_extract/configs/geometry_1m_50ksteps_emb.yaml"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run the 50k-step embedding-probe pipeline with background logging by default.

Stages:
  1. proj extract -> analyze -> aggregate
  2. emb analyze -> aggregate
  3. plot proj and emb by-init figures

Options:
  --proj-only            Run only the projector-space pipeline
  --emb-only             Run only the embedding-space pipeline
  --overwrite-extract    Pass --overwrite to extraction
  --overwrite-analyze    Pass --overwrite to analysis
  --no-plot              Skip the plotting stage
  --log-base PATH        Parent directory for batch logs
  --log-dir PATH         Exact log directory to use for the batch
  --foreground           Run in the current shell instead of detaching
  --help                 Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proj-only)
      RUN_PROJ=1
      RUN_EMB=0
      shift
      ;;
    --emb-only)
      RUN_PROJ=0
      RUN_EMB=1
      shift
      ;;
    --overwrite-extract)
      OVERWRITE_EXTRACT=1
      shift
      ;;
    --overwrite-analyze)
      OVERWRITE_ANALYZE=1
      shift
      ;;
    --no-plot)
      RUN_PLOTS=0
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

if [[ "$RUN_PROJ" -eq 0 && "$RUN_EMB" -eq 0 ]]; then
  echo "Nothing to do: both proj and emb pipelines are disabled." >&2
  exit 1
fi

for rel_path in "${PROJ_CONFIGS[@]}" "${EMB_CONFIGS[@]}"; do
  abs_path="$ROOT_DIR/$rel_path"
  if [[ ! -f "$abs_path" ]]; then
    echo "Missing required config: $abs_path" >&2
    exit 1
  fi
done

if ! command -v python >/dev/null 2>&1; then
  echo "python not found in PATH. Activate the correct environment first." >&2
  exit 1
fi

SUITE_NAME="embedding_probe_50ksteps"

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
  [[ "$RUN_PROJ" -eq 1 && "$RUN_EMB" -eq 0 ]] && DETACH_ARGS+=(--proj-only)
  [[ "$RUN_PROJ" -eq 0 && "$RUN_EMB" -eq 1 ]] && DETACH_ARGS+=(--emb-only)
  [[ "$OVERWRITE_EXTRACT" -eq 1 ]] && DETACH_ARGS+=(--overwrite-extract)
  [[ "$OVERWRITE_ANALYZE" -eq 1 ]] && DETACH_ARGS+=(--overwrite-analyze)
  [[ "$RUN_PLOTS" -eq 0 ]] && DETACH_ARGS+=(--no-plot)

  nohup bash "$0" "${DETACH_ARGS[@]}" > "$LOG_DIR/launcher.out" 2>&1 < /dev/null &
  PID=$!
  echo "Started 50k embedding-probe suite in background."
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
  echo "run_proj"$'\t'"$RUN_PROJ"
  echo "run_emb"$'\t'"$RUN_EMB"
  echo "overwrite_extract"$'\t'"$OVERWRITE_EXTRACT"
  echo "overwrite_analyze"$'\t'"$OVERWRITE_ANALYZE"
  echo "run_plots"$'\t'"$RUN_PLOTS"
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

if [[ "$RUN_PROJ" -eq 1 ]]; then
  for cfg in "${PROJ_CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"

    extract_cmd=(python "$ROOT_DIR/scripts/run_embedding_extract.py" --cfg "$ROOT_DIR/$cfg")
    [[ "$OVERWRITE_EXTRACT" -eq 1 ]] && extract_cmd+=(--overwrite)
    run_stage "${name}_extract" "$LOG_DIR/${name}_extract.log" "${extract_cmd[@]}" || exit $?

    analyze_cmd=(python "$ROOT_DIR/scripts/run_embedding_analyze.py" --cfg "$ROOT_DIR/$cfg")
    [[ "$OVERWRITE_ANALYZE" -eq 1 ]] && analyze_cmd+=(--overwrite)
    run_stage "${name}_analyze" "$LOG_DIR/${name}_analyze.log" "${analyze_cmd[@]}" || exit $?

    aggregate_cmd=(python "$ROOT_DIR/scripts/run_embedding_aggregate.py" --cfg "$ROOT_DIR/$cfg")
    run_stage "${name}_aggregate" "$LOG_DIR/${name}_aggregate.log" "${aggregate_cmd[@]}" || exit $?
  done
fi

if [[ "$RUN_EMB" -eq 1 ]]; then
  for cfg in "${EMB_CONFIGS[@]}"; do
    name="$(basename "$cfg" .yaml)"

    analyze_cmd=(python "$ROOT_DIR/scripts/run_embedding_analyze.py" --cfg "$ROOT_DIR/$cfg")
    [[ "$OVERWRITE_ANALYZE" -eq 1 ]] && analyze_cmd+=(--overwrite)
    run_stage "${name}_analyze" "$LOG_DIR/${name}_analyze.log" "${analyze_cmd[@]}" || exit $?

    aggregate_cmd=(python "$ROOT_DIR/scripts/run_embedding_aggregate.py" --cfg "$ROOT_DIR/$cfg")
    run_stage "${name}_aggregate" "$LOG_DIR/${name}_aggregate.log" "${aggregate_cmd[@]}" || exit $?
  done
fi

if [[ "$RUN_PLOTS" -eq 1 ]]; then
  if [[ "$RUN_PROJ" -eq 1 ]]; then
    run_stage \
      "plot_proj_50ksteps" \
      "$LOG_DIR/plot_proj_50ksteps.log" \
      python "$ROOT_DIR/scripts/embedding_probe/plot_embedding_results.py" \
        --summary-kind proj \
        --embedding-key proj \
        --study-tag _50ksteps \
        --output-tag 50ksteps || exit $?
  fi

  if [[ "$RUN_EMB" -eq 1 ]]; then
    run_stage \
      "plot_emb_50ksteps" \
      "$LOG_DIR/plot_emb_50ksteps.log" \
      python "$ROOT_DIR/scripts/embedding_probe/plot_embedding_results.py" \
        --summary-kind emb \
        --embedding-key emb \
        --study-tag _50ksteps \
        --output-tag 50ksteps_emb || exit $?
  fi
fi

echo "finished_at"$'\t'"$(date --iso-8601=seconds)" >> "$BATCH_LOG"
echo "failures"$'\t'"$FAILURES" >> "$BATCH_LOG"

if [[ "$FAILURES" -gt 0 ]]; then
  echo "50k embedding-probe suite finished with $FAILURES failed stages. See $SUMMARY_TSV"
  exit 1
fi

echo "50k embedding-probe suite finished successfully. See $SUMMARY_TSV"
