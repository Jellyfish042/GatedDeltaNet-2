#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/tankaifeng/gdn2_train}"
REPO="${REPO:-$ROOT/repo/GatedDeltaNet-2}"
CONDA="${CONDA:-/home/tankaifeng/ENTER/bin/conda}"
CONDA_ENV="${CONDA_ENV:-unc}"
PARQUET_DIR="${PARQUET_DIR:-$ROOT/data/fineweb_edu_parquet_100BT}"
OUT="${OUT:-$ROOT/data/fineweb_edu_packed_100B}"
LOG="${LOG:-$ROOT/logs/prepare_fineweb_edu_100B_fast.log}"
PID_FILE="${PID_FILE:-$ROOT/logs/prepare_fineweb_edu_100B_fast.pid}"
TOKENIZER="${TOKENIZER:-$ROOT/tokenizers/TinyLlama_v1.1}"
PROXY="${PROXY:-http://127.0.0.1:7890}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TRAIN_TOKENS="${TRAIN_TOKENS:-100B}"
VAL_TOKENS="${VAL_TOKENS:-50M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_WORKERS="${MAX_WORKERS:-32}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
FOREGROUND=0
DOWNLOAD_ONLY=0
PACK_ONLY=0

usage() {
  cat <<EOF
Usage: bash scripts/run_prepare_fineweb_edu_100B_fast.sh [--foreground] [--download-only] [--pack-only]

This runs a faster two-stage flow:
  1. snapshot_download FineWeb-Edu sample/100BT parquet files with HF Hub concurrency.
  2. pack local parquet files into LitGPT .bin shards with resumable shard checkpoints.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --pack-only)
      PACK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DOWNLOAD_ONLY" == "1" && "$PACK_ONLY" == "1" ]]; then
  echo "--download-only and --pack-only cannot be used together" >&2
  exit 2
fi

running_jobs="$(
  pgrep -af "download_fineweb_edu_100B_parquet.py|pack_fineweb_edu_100B_from_parquet.py|run_prepare_fineweb_edu_100B_fast" \
    | awk -v self="$$" '$1 != self { print }' || true
)"
if [[ "${SKIP_RUNNING_CHECK:-0}" != "1" && -n "$running_jobs" ]]; then
  echo "A fast 100B prepare job may already be running:" >&2
  echo "$running_jobs" >&2
  exit 1
fi

if [[ ! -x "$CONDA" ]]; then
  echo "Conda executable not found: $CONDA" >&2
  exit 1
fi

if [[ ! -d "$REPO" ]]; then
  echo "Repo directory not found: $REPO" >&2
  exit 1
fi

if [[ ! -d "$TOKENIZER" ]]; then
  echo "Tokenizer directory not found: $TOKENIZER" >&2
  exit 1
fi

mkdir -p "$ROOT/hf_cache" "$ROOT/hf_cache/datasets" "$ROOT/hf_cache/xet" "$ROOT/tmp" "$ROOT/logs" "$PARQUET_DIR" "$OUT"

export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"
export http_proxy="$PROXY"
export https_proxy="$PROXY"
export HF_ENDPOINT
export HF_HOME="$ROOT/hf_cache"
export HF_DATASETS_CACHE="$ROOT/hf_cache/datasets"
export HF_XET_CACHE="$ROOT/hf_cache/xet"
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_NUM_CONCURRENT_RANGE_GETS
export HF_HUB_DOWNLOAD_TIMEOUT
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export TMPDIR="$ROOT/tmp"
unset HF_HUB_DISABLE_XET

run_job() {
  cd "$REPO"
  echo "Parquet dir: $PARQUET_DIR"
  echo "Packed out: $OUT"
  echo "Log: $LOG"
  echo "Proxy: $PROXY"
  echo "HF endpoint: $HF_ENDPOINT"
  echo "HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"
  echo "HF_XET_NUM_CONCURRENT_RANGE_GETS=$HF_XET_NUM_CONCURRENT_RANGE_GETS"
  echo "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"

  if [[ "$PACK_ONLY" != "1" ]]; then
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python "$REPO/scripts/download_fineweb_edu_100B_parquet.py" \
      --local-dir "$PARQUET_DIR" \
      --max-workers "$MAX_WORKERS"
  fi

  if [[ "$DOWNLOAD_ONLY" != "1" ]]; then
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python "$REPO/scripts/pack_fineweb_edu_100B_from_parquet.py" \
      --parquet-dir "$PARQUET_DIR" \
      --out-root "$OUT" \
      --tokenizer "$TOKENIZER" \
      --train-tokens "$TRAIN_TOKENS" \
      --val-tokens "$VAL_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --resume
  fi
}

if [[ "$FOREGROUND" == "1" ]]; then
  run_job
else
  nohup env \
    ROOT="$ROOT" \
    REPO="$REPO" \
    CONDA="$CONDA" \
    CONDA_ENV="$CONDA_ENV" \
    PARQUET_DIR="$PARQUET_DIR" \
    OUT="$OUT" \
    LOG="$LOG" \
    PID_FILE="$PID_FILE" \
    TOKENIZER="$TOKENIZER" \
    PROXY="$PROXY" \
    HF_ENDPOINT="$HF_ENDPOINT" \
    TRAIN_TOKENS="$TRAIN_TOKENS" \
    VAL_TOKENS="$VAL_TOKENS" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_WORKERS="$MAX_WORKERS" \
    HF_XET_NUM_CONCURRENT_RANGE_GETS="$HF_XET_NUM_CONCURRENT_RANGE_GETS" \
    HF_HUB_DOWNLOAD_TIMEOUT="$HF_HUB_DOWNLOAD_TIMEOUT" \
    SKIP_RUNNING_CHECK=1 \
    bash "$REPO/scripts/run_prepare_fineweb_edu_100B_fast.sh" --foreground > "$LOG" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Started fast 100B prepare job: PID $pid"
  echo "Progress:"
  echo "  tail -f $LOG"
  echo "  cat $PARQUET_DIR/download_manifest.json"
  echo "  cat $OUT/fineweb_edu_100B_manifest.json"
  echo "Stop:"
  echo "  kill $pid"
fi
