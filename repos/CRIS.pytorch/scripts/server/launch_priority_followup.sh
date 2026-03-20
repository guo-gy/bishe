#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/bishe/repos/CRIS.pytorch}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/bishe_runs/ablation_refcoco_r50}"
CONFIG="${CONFIG:-config/refcoco/cris_r50.yaml}"
ENV_NAME="${ENV_NAME:-cris-bsd}"
WANDB_MODE="${WANDB_MODE:-disabled}"

BASE_EXP="${BASE_EXP:-refcoco_cris_r50_baseline}"
BAL_EXP="${BAL_EXP:-refcoco_cris_r50_balance}"
FOLLOW_EXP="${FOLLOW_EXP:-refcoco_cris_r50_balance_distill}"

BASE_GPU="${BASE_GPU:-0}"
BAL_GPU="${BAL_GPU:-1}"
FOLLOW_PORT_GPU0="${FOLLOW_PORT_GPU0:-29610}"
FOLLOW_PORT_GPU1="${FOLLOW_PORT_GPU1:-29611}"

BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-8}"
WORKERS_VAL="${WORKERS_VAL:-4}"
BASE_LR="${BASE_LR:-0.0000125000}"
EPOCHS="${EPOCHS:-50}"
PRINT_FREQ="${PRINT_FREQ:-50}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
KEEP_LAST_MODEL="${KEEP_LAST_MODEL:-0}"

PYTHON_CMD=(/root/miniconda3/bin/conda run --no-capture-output -n "$ENV_NAME" python)

timestamp() {
  date '+%F %T'
}

log() {
  echo "[$(timestamp)] $*"
}

exp_running() {
  local exp_name="$1"
  pgrep -af "TRAIN.exp_name ${exp_name}" >/dev/null 2>&1
}

followup_exists() {
  [[ -f "${OUTPUT_ROOT}/${FOLLOW_EXP}/best_model.pth" ]] || [[ -f "${OUTPUT_ROOT}/${FOLLOW_EXP}/last_model.pth" ]]
}

launch_followup() {
  local gpu="$1"
  local port="$2"
  local log_dir="${OUTPUT_ROOT}/logs"
  local exp_dir="${OUTPUT_ROOT}/${FOLLOW_EXP}"
  local log_file="${log_dir}/${FOLLOW_EXP}.log"

  mkdir -p "$log_dir"
  log "Launching ${FOLLOW_EXP} on GPU ${gpu}."

  (
    set -euo pipefail
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export WANDB_MODE="$WANDB_MODE"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    "${PYTHON_CMD[@]}" train.py --config "$CONFIG" --opts \
      TRAIN.exp_name "$FOLLOW_EXP" \
      TRAIN.output_folder "$OUTPUT_ROOT" \
      TRAIN.use_balance True \
      TRAIN.use_self_distill True \
      TRAIN.sync_bn False \
      TRAIN.batch_size "$BATCH_SIZE" \
      TRAIN.batch_size_val "$BATCH_SIZE" \
      TRAIN.workers "$WORKERS" \
      TRAIN.workers_val "$WORKERS_VAL" \
      TRAIN.base_lr "$BASE_LR" \
      Distributed.dist_url "tcp://127.0.0.1:${port}" \
      TRAIN.epochs "$EPOCHS" \
      TRAIN.print_freq "$PRINT_FREQ"
    if (( ! KEEP_LAST_MODEL )) && [[ -f "${exp_dir}/best_model.pth" && -f "${exp_dir}/last_model.pth" ]]; then
      rm -f "${exp_dir}/last_model.pth"
      log "Removed ${exp_dir}/last_model.pth after follow-up completion."
    fi
  ) >"$log_file" 2>&1 &

  log "Follow-up PID: $!"
}

main() {
  log "Priority follow-up watcher started."
  while true; do
    if exp_running "$FOLLOW_EXP" || followup_exists; then
      log "${FOLLOW_EXP} already exists or is running. Exiting watcher."
      exit 0
    fi

    local base_running=0
    local bal_running=0
    if exp_running "$BASE_EXP"; then
      base_running=1
    fi
    if exp_running "$BAL_EXP"; then
      bal_running=1
    fi

    if (( base_running == 0 )); then
      launch_followup "$BASE_GPU" "$FOLLOW_PORT_GPU0"
      exit 0
    fi
    if (( bal_running == 0 )); then
      launch_followup "$BAL_GPU" "$FOLLOW_PORT_GPU1"
      exit 0
    fi

    sleep "$SLEEP_SECONDS"
  done
}

main "$@"
