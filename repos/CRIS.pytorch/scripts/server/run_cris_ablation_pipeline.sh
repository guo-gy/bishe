#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="config/refcoco/cris_r50.yaml"
GPU_IDS="0,1"
GPUS_PER_JOB=1
PER_GPU_BATCH_SIZE=8
PER_GPU_BATCH_SIZE_VAL=8
WORKERS=8
WORKERS_VAL=4
OUTPUT_ROOT="exp/ablation"
ENV_BACKEND="auto"
ENV_NAME="cris-bsd"
VENV_DIR="$ROOT_DIR/.venv-cris-bsd"
PYTHON_VERSION="3.10"
HOST_PYTHON="${HOST_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2}"
BASE_PORT=29600
WANDB_MODE="${WANDB_MODE:-disabled}"
RUN_SETUP=1
RUN_EXPERIMENTS=1
RUN_TEST=0
FAIL_FAST=1
AUTO_SCALE_LR=1
DOWNLOAD_CLIP=1
VARIANTS_CSV="baseline,balance,distill,balance_distill"
declare -a EXTRA_OPTS=()
declare -a PYTHON_CMD=()
declare -a PIP_CMD=()
declare -a SLOT_GPU_CSVS=()
declare -a SLOT_PIDS=()
declare -a SLOT_VARIANTS=()
declare -a VARIANTS=()


usage() {
  cat <<'EOF'
Usage:
  bash scripts/server/run_cris_ablation_pipeline.sh [options]

Main options:
  --config PATH                 Base config file. Default: config/refcoco/cris_r50.yaml
  --gpus IDS                    GPU ids separated by commas. Default: 0,1
  --gpus-per-job N              Number of GPUs per experiment. Default: 1
  --variants CSV                baseline,balance,distill,balance_distill
  --output-root PATH            Output root for checkpoints and logs
  --run-test                    Run test.py after each training job
  --skip-setup                  Skip environment setup
  --skip-run                    Skip experiment queue
  --setup-only                  Only create the environment and install packages

Environment options:
  --env-backend auto|conda|venv
  --env-name NAME
  --venv-dir PATH
  --python-version VERSION
  --host-python BIN
  --torch-index-url URL
  --torch-packages STRING

Training options:
  --per-gpu-batch-size N
  --per-gpu-batch-size-val N
  --workers N
  --workers-val N
  --base-port PORT
  --wandb-mode online|offline|disabled
  --no-auto-scale-lr
  --continue-on-error
  --no-download-clip
  --extra-opt KEY VALUE         Can be provided multiple times
EOF
}


log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}


die() {
  log "ERROR: $*"
  exit 1
}


parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG="$2"
        shift 2
        ;;
      --gpus)
        GPU_IDS="$2"
        shift 2
        ;;
      --gpus-per-job)
        GPUS_PER_JOB="$2"
        shift 2
        ;;
      --variants)
        VARIANTS_CSV="$2"
        shift 2
        ;;
      --output-root)
        OUTPUT_ROOT="$2"
        shift 2
        ;;
      --env-backend)
        ENV_BACKEND="$2"
        shift 2
        ;;
      --env-name)
        ENV_NAME="$2"
        shift 2
        ;;
      --venv-dir)
        VENV_DIR="$2"
        shift 2
        ;;
      --python-version)
        PYTHON_VERSION="$2"
        shift 2
        ;;
      --host-python)
        HOST_PYTHON="$2"
        shift 2
        ;;
      --torch-index-url)
        TORCH_INDEX_URL="$2"
        shift 2
        ;;
      --torch-packages)
        TORCH_PACKAGES="$2"
        shift 2
        ;;
      --per-gpu-batch-size)
        PER_GPU_BATCH_SIZE="$2"
        shift 2
        ;;
      --per-gpu-batch-size-val)
        PER_GPU_BATCH_SIZE_VAL="$2"
        shift 2
        ;;
      --workers)
        WORKERS="$2"
        shift 2
        ;;
      --workers-val)
        WORKERS_VAL="$2"
        shift 2
        ;;
      --base-port)
        BASE_PORT="$2"
        shift 2
        ;;
      --wandb-mode)
        WANDB_MODE="$2"
        shift 2
        ;;
      --run-test)
        RUN_TEST=1
        shift
        ;;
      --skip-setup)
        RUN_SETUP=0
        shift
        ;;
      --skip-run)
        RUN_EXPERIMENTS=0
        shift
        ;;
      --setup-only)
        RUN_EXPERIMENTS=0
        shift
        ;;
      --no-auto-scale-lr)
        AUTO_SCALE_LR=0
        shift
        ;;
      --continue-on-error)
        FAIL_FAST=0
        shift
        ;;
      --no-download-clip)
        DOWNLOAD_CLIP=0
        shift
        ;;
      --extra-opt)
        EXTRA_OPTS+=("$2" "$3")
        shift 3
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}


setup_python_commands() {
  local allow_create="${1:-1}"
  local use_conda=0
  if [[ "$ENV_BACKEND" == "conda" ]]; then
    use_conda=1
  elif [[ "$ENV_BACKEND" == "venv" ]]; then
    use_conda=0
  elif command -v conda >/dev/null 2>&1; then
    use_conda=1
  fi

  if (( use_conda )); then
    if ! command -v conda >/dev/null 2>&1; then
      die "conda is not available but --env-backend=conda was requested."
    fi
    if ! conda env list | awk 'NF && $1 !~ /^#/' | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
      if (( allow_create )); then
        log "Creating conda environment $ENV_NAME with Python $PYTHON_VERSION"
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
      else
        die "Conda environment $ENV_NAME does not exist. Run without --skip-setup first."
      fi
    fi
    PYTHON_CMD=(conda run --no-capture-output -n "$ENV_NAME" python)
    PIP_CMD=(conda run --no-capture-output -n "$ENV_NAME" python -m pip)
  else
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
      if (( allow_create )); then
        log "Creating virtualenv at $VENV_DIR"
        "$HOST_PYTHON" -m venv "$VENV_DIR"
      else
        die "Virtualenv does not exist at $VENV_DIR. Run without --skip-setup first."
      fi
    fi
    PYTHON_CMD=("$VENV_DIR/bin/python")
    PIP_CMD=("$VENV_DIR/bin/python" -m pip)
  fi
}


install_environment() {
  setup_python_commands 1
  log "Installing runtime dependencies"
  "${PIP_CMD[@]}" install --upgrade pip setuptools wheel
  read -r -a torch_pkgs <<< "$TORCH_PACKAGES"
  "${PIP_CMD[@]}" install --index-url "$TORCH_INDEX_URL" "${torch_pkgs[@]}"
  "${PIP_CMD[@]}" install \
    cython \
    wandb \
    lmdb \
    pyarrow \
    regex \
    ftfy \
    loguru \
    pycocotools \
    matplotlib \
    tqdm \
    opencv-python-headless \
    PyYAML \
    numpy
}


cfg_query() {
  local key="$1"
  "${PYTHON_CMD[@]}" - "$CONFIG" "$key" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("cfgmod", "utils/config.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.load_cfg_from_cfg_file(sys.argv[1])
print(getattr(cfg, sys.argv[2]))
PY
}


to_abs_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT_DIR/$path"
  fi
}


ensure_clip_weight() {
  local clip_pretrain abs_path
  clip_pretrain="$(cfg_query clip_pretrain)"
  abs_path="$(to_abs_path "$clip_pretrain")"
  if [[ -f "$abs_path" ]]; then
    return
  fi
  if (( ! DOWNLOAD_CLIP )); then
    die "Missing CLIP weight: $abs_path"
  fi
  log "Downloading OpenAI CLIP RN50 weight to $abs_path"
  mkdir -p "$(dirname "$abs_path")"
  "${PIP_CMD[@]}" install git+https://github.com/openai/CLIP.git
  "${PYTHON_CMD[@]}" - "$abs_path" <<'PY'
import sys
import torch
import clip

model, _ = clip.load("RN50", device="cpu", jit=True)
torch.jit.save(model, sys.argv[1])
print(sys.argv[1])
PY
}


check_required_paths() {
  local train_lmdb val_lmdb mask_root
  train_lmdb="$(to_abs_path "$(cfg_query train_lmdb)")"
  val_lmdb="$(to_abs_path "$(cfg_query val_lmdb)")"
  mask_root="$(to_abs_path "$(cfg_query mask_root)")"

  [[ -e "$train_lmdb" ]] || die "Missing train LMDB: $train_lmdb"
  [[ -e "$val_lmdb" ]] || die "Missing val LMDB: $val_lmdb"
  [[ -d "$mask_root" ]] || die "Missing mask directory: $mask_root"

  ensure_clip_weight
}


prepare_slots() {
  IFS=',' read -r -a all_gpus <<< "$GPU_IDS"
  (( ${#all_gpus[@]} > 0 )) || die "No GPUs were provided."
  if (( ${#all_gpus[@]} % GPUS_PER_JOB != 0 )); then
    die "GPU count (${#all_gpus[@]}) must be divisible by --gpus-per-job ($GPUS_PER_JOB)."
  fi

  SLOT_GPU_CSVS=()
  for ((i = 0; i < ${#all_gpus[@]}; i += GPUS_PER_JOB)); do
    local chunk=("${all_gpus[@]:i:GPUS_PER_JOB}")
    SLOT_GPU_CSVS+=("$(IFS=,; echo "${chunk[*]}")")
  done
}


build_variant() {
  local variant="$1"
  local port="$2"
  local dataset_name base_name exp_name global_batch global_batch_val base_lr scaled_lr
  local use_balance
  local use_sd

  dataset_name="$(cfg_query dataset)"
  base_name="$(basename "${CONFIG%.yaml}")"
  case "$variant" in
    baseline)
      exp_name="${dataset_name}_${base_name}_baseline"
      use_balance="False"
      use_sd="False"
      ;;
    balance)
      exp_name="${dataset_name}_${base_name}_balance"
      use_balance="True"
      use_sd="False"
      ;;
    distill)
      exp_name="${dataset_name}_${base_name}_distill"
      use_balance="False"
      use_sd="True"
      ;;
    balance_distill)
      exp_name="${dataset_name}_${base_name}_balance_distill"
      use_balance="True"
      use_sd="True"
      ;;
    *)
      die "Unknown variant: $variant"
      ;;
  esac

  global_batch=$((PER_GPU_BATCH_SIZE * GPUS_PER_JOB))
  global_batch_val=$((PER_GPU_BATCH_SIZE_VAL * GPUS_PER_JOB))
  base_lr="$(cfg_query base_lr)"
  scaled_lr="$("${PYTHON_CMD[@]}" - "$base_lr" "$global_batch" "$AUTO_SCALE_LR" <<'PY'
import sys

base_lr = float(sys.argv[1])
global_batch = int(sys.argv[2])
auto_scale = int(sys.argv[3])
if auto_scale:
    print(f"{base_lr * global_batch / 64.0:.10f}")
else:
    print(f"{base_lr:.10f}")
PY
)"

  VARIANT_NAME="$exp_name"
  VARIANT_OPTS=(
    TRAIN.exp_name "$exp_name"
    TRAIN.output_folder "$OUTPUT_ROOT"
    TRAIN.use_balance "$use_balance"
    TRAIN.use_self_distill "$use_sd"
    TRAIN.sync_bn "False"
    TRAIN.batch_size "$global_batch"
    TRAIN.batch_size_val "$global_batch_val"
    TRAIN.workers "$WORKERS"
    TRAIN.workers_val "$WORKERS_VAL"
    TRAIN.base_lr "$scaled_lr"
    Distributed.dist_url "tcp://127.0.0.1:${port}"
  )
  if (( ${#EXTRA_OPTS[@]} > 0 )); then
    VARIANT_OPTS+=("${EXTRA_OPTS[@]}")
  fi
}


run_variant_job() {
  local variant="$1"
  local slot_idx="$2"
  local gpu_csv="$3"
  local port="$4"

  build_variant "$variant" "$port"
  local log_dir
  local log_file
  log_dir="$(to_abs_path "$OUTPUT_ROOT")/logs"
  log_file="$log_dir/${VARIANT_NAME}.log"
  mkdir -p "$log_dir"

  log "Launching $VARIANT_NAME on GPU(s) $gpu_csv"
  (
    set -euo pipefail
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu_csv"
    export WANDB_MODE="$WANDB_MODE"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    "${PYTHON_CMD[@]}" train.py --config "$CONFIG" --opts "${VARIANT_OPTS[@]}"
    if (( RUN_TEST )); then
      "${PYTHON_CMD[@]}" test.py --config "$CONFIG" --opts "${VARIANT_OPTS[@]}"
    fi
  ) >"$log_file" 2>&1 &

  SLOT_PIDS[$slot_idx]=$!
  SLOT_VARIANTS[$slot_idx]="$VARIANT_NAME"
}


cleanup_jobs() {
  local pid
  for pid in "${SLOT_PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}


run_queue() {
  prepare_slots
  IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
  (( ${#VARIANTS[@]} > 0 )) || die "No variants were provided."

  mkdir -p "$(to_abs_path "$OUTPUT_ROOT")"
  check_required_paths

  local next_variant=0
  local next_port=0
  local active_jobs=0
  local total_variants="${#VARIANTS[@]}"

  trap cleanup_jobs INT TERM

  for ((slot = 0; slot < ${#SLOT_GPU_CSVS[@]} && next_variant < total_variants; slot++)); do
    run_variant_job "${VARIANTS[$next_variant]}" "$slot" "${SLOT_GPU_CSVS[$slot]}" "$((BASE_PORT + next_port))"
    next_variant=$((next_variant + 1))
    next_port=$((next_port + 1))
    active_jobs=$((active_jobs + 1))
  done

  while (( active_jobs > 0 )); do
    sleep 10
    for ((slot = 0; slot < ${#SLOT_GPU_CSVS[@]}; slot++)); do
      local pid="${SLOT_PIDS[$slot]:-}"
      if [[ -z "${pid:-}" ]]; then
        continue
      fi
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi

      local variant_name="${SLOT_VARIANTS[$slot]:-unknown}"
      if wait "$pid"; then
        log "Finished $variant_name"
      else
        local exit_code=$?
        log "Job failed: $variant_name (exit code $exit_code)"
        SLOT_PIDS[$slot]=""
        SLOT_VARIANTS[$slot]=""
        if (( FAIL_FAST )); then
          cleanup_jobs
          die "Ablation queue stopped because $variant_name failed."
        fi
      fi

      SLOT_PIDS[$slot]=""
      SLOT_VARIANTS[$slot]=""
      active_jobs=$((active_jobs - 1))

      if (( next_variant < total_variants )); then
        run_variant_job "${VARIANTS[$next_variant]}" "$slot" "${SLOT_GPU_CSVS[$slot]}" "$((BASE_PORT + next_port))"
        next_variant=$((next_variant + 1))
        next_port=$((next_port + 1))
        active_jobs=$((active_jobs + 1))
      fi
    done
  done

  log "All requested ablation jobs finished."
}


main() {
  parse_args "$@"
  [[ -f "$CONFIG" ]] || die "Config file does not exist: $CONFIG"

  if (( RUN_SETUP )); then
    install_environment
  else
    setup_python_commands 0
  fi

  if (( RUN_EXPERIMENTS )); then
    run_queue
  else
    log "Setup finished. Experiment queue was skipped."
  fi
}


main "$@"
