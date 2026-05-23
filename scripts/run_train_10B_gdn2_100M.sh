#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/tankaifeng/gdn2_train}"
REPO="${REPO:-$ROOT/repo/GatedDeltaNet-2}"
CONDA="${CONDA:-/home/tankaifeng/ENTER/bin/conda}"
CONDA_ENV="${CONDA_ENV:-unc}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/fineweb_edu_packed_10B_from_local}"
SAVE_DIR="${SAVE_DIR:-$ROOT/runs}"
NAME="${NAME:-10B_gdn2_100M}"
MODEL="${MODEL:-gdn2_100M}"
CONFIG="${CONFIG:-tsz512x4k_10B}"
LR="${LR:-4e-4}"
SEED="${SEED:-3407}"
EVAL_ITERS="${EVAL_ITERS:-15}"
ACTUAL_TRAIN_TIME="${ACTUAL_TRAIN_TIME:-10080}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEBUG="${DEBUG:-0}"
INTERACTIVE_JOB="${INTERACTIVE_JOB:-1}"
WANDB_MODE="${WANDB_MODE:-offline}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
    IFS=',' read -r -a cuda_devices <<< "$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="${#cuda_devices[@]}"
  else
    NPROC_PER_NODE=1
  fi
fi

export CUDA_VISIBLE_DEVICES
export WANDB_MODE
export PYTHONUNBUFFERED
export SLURM_NNODES="${SLURM_NNODES:-1}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/triton/$NAME}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$SAVE_DIR" "$ROOT/logs/$NAME" "$TRITON_CACHE_DIR"

cd "$REPO"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node "$NPROC_PER_NODE")
else
  launcher=(python -u)
fi

args=(
  "${launcher[@]}" pretrain.py
  --train_data_dir "$DATA_ROOT/train_10B"
  --val_data_dir "$DATA_ROOT/val_50M"
  --output_root "$SAVE_DIR"
  --exp_name "$NAME"
  --exp_group "gdn2_100M"
  --model_name "$MODEL"
  --train_config "$CONFIG"
  --eval_iters "$EVAL_ITERS"
  --learning_rate "$LR"
  --seed "$SEED"
  --actual_train_time "$ACTUAL_TRAIN_TIME"
)

if [[ "$INTERACTIVE_JOB" == "1" ]]; then
  args+=(--interactive_job)
fi

if [[ "$DEBUG" == "1" ]]; then
  args+=(--debug)
fi

exec "$CONDA" run -n "$CONDA_ENV" "${args[@]}"
