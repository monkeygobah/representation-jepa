#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG_DIR="$ROOT_DIR/configs/followup_50ksteps"
DEFAULT_LOG_BASE="$ROOT_DIR/_logs/followup_50ksteps_seginit"
DEFAULT_PATTERN='*seginit-50ksteps.yaml'

CONFIG_DIR="$DEFAULT_CONFIG_DIR"
LOG_BASE="$DEFAULT_LOG_BASE"
CONFIG_PATTERN="$DEFAULT_PATTERN"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29502}"
DETACH=1
LOG_DIR=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Launch the 50k-step seginit follow-up suite sequentially with torchrun and
store one log per config. The script detaches by default so the batch keeps
running in the background.

Options:
  --config-dir PATH       Directory of YAML configs to run
  --config-pattern GLOB   Config filename pattern (default: $CONFIG_PATTERN)
  --nproc-per-node N      Number of DDP workers per run (default: $NPROC_PER_NODE)
  --master-port PORT      torchrun master port (default: $MASTER_PORT)
  --log-base PATH         Parent directory for batch logs
  --log-dir PATH          Exact log directory to use for the batch
  --foreground            Run in the current shell instead of detaching
  --help                  Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-dir)
      CONFIG_DIR="$2"
      shift 2
      ;;
    --config-pattern)
      CONFIG_PATTERN="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
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

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory does not exist: $CONFIG_DIR" >&2
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun not found in PATH. Activate the correct environment first." >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "$CONFIG_PATTERN" | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No YAML configs matching $CONFIG_PATTERN found in: $CONFIG_DIR" >&2
  exit 1
fi

SUITE_NAME="followup_50ksteps_seginit"

if [[ -z "$LOG_DIR" ]]; then
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_DIR="$LOG_BASE/${SUITE_NAME}_$TIMESTAMP"
fi

mkdir -p "$LOG_DIR"

if [[ "$DETACH" -eq 1 ]]; then
  nohup bash "$0" \
    --foreground \
    --config-dir "$CONFIG_DIR" \
    --config-pattern "$CONFIG_PATTERN" \
    --nproc-per-node "$NPROC_PER_NODE" \
    --master-port "$MASTER_PORT" \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/launcher.out" 2>&1 < /dev/null &
  PID=$!
  echo "Started seginit follow-up suite in background."
  echo "PID: $PID"
  echo "Config dir: $CONFIG_DIR"
  echo "Config pattern: $CONFIG_PATTERN"
  echo "Master port: $MASTER_PORT"
  echo "Log dir: $LOG_DIR"
  echo "Launcher log: $LOG_DIR/launcher.out"
  exit 0
fi

BATCH_LOG="$LOG_DIR/batch.log"
SUMMARY_TSV="$LOG_DIR/summary.tsv"

{
  echo "suite_name"$'\t'"$SUITE_NAME"
  echo "config_dir"$'\t'"$CONFIG_DIR"
  echo "config_pattern"$'\t'"$CONFIG_PATTERN"
  echo "nproc_per_node"$'\t'"$NPROC_PER_NODE"
  echo "master_port"$'\t'"$MASTER_PORT"
  echo "started_at"$'\t'"$(date --iso-8601=seconds)"
  echo "cuda_visible_devices"$'\t'"${CUDA_VISIBLE_DEVICES:-<unset>}"
} >> "$BATCH_LOG"

printf "config\tstatus\texit_code\tlog_path\n" > "$SUMMARY_TSV"

FAILURES=0

for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  run_log="$LOG_DIR/${name}.log"

  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$BATCH_LOG"
  echo "  cfg: $cfg" | tee -a "$BATCH_LOG"
  echo "  log: $run_log" | tee -a "$BATCH_LOG"

  if torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" scripts/train_ssl.py --cfg "$cfg" > "$run_log" 2>&1; then
    status="ok"
    exit_code=0
    echo "[$(date --iso-8601=seconds)] DONE  $name" | tee -a "$BATCH_LOG"
  else
    status="failed"
    exit_code=$?
    FAILURES=$((FAILURES + 1))
    echo "[$(date --iso-8601=seconds)] FAIL  $name (exit=$exit_code)" | tee -a "$BATCH_LOG"
  fi

  printf "%s\t%s\t%s\t%s\n" "$name" "$status" "$exit_code" "$run_log" >> "$SUMMARY_TSV"
done

echo "finished_at"$'\t'"$(date --iso-8601=seconds)" >> "$BATCH_LOG"
echo "failures"$'\t'"$FAILURES" >> "$BATCH_LOG"

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Seginit follow-up suite finished with $FAILURES failed runs. See $SUMMARY_TSV"
  exit 1
fi

echo "Seginit follow-up suite finished successfully. See $SUMMARY_TSV"
