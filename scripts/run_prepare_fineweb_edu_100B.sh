#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/tankaifeng/gdn2_train}"
REPO="${REPO:-$ROOT/repo/GatedDeltaNet-2}"
CONDA="${CONDA:-/home/tankaifeng/ENTER/bin/conda}"
CONDA_ENV="${CONDA_ENV:-unc}"
OUT="${OUT:-$ROOT/data/fineweb_edu_packed_100B}"
LOG="${LOG:-$ROOT/logs/prepare_fineweb_edu_100B.log}"
PID_FILE="${PID_FILE:-$ROOT/logs/prepare_fineweb_edu_100B.pid}"
TOKENIZER="${TOKENIZER:-$ROOT/tokenizers/TinyLlama_v1.1}"
PROXY="${PROXY:-http://127.0.0.1:7890}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
TRAIN_TOKENS="${TRAIN_TOKENS:-100B}"
VAL_TOKENS="${VAL_TOKENS:-50M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CONFIG="${CONFIG:-sample-100BT}"
FOREGROUND=0
CLEAN=0
RESUME=1

usage() {
  cat <<EOF
Usage: bash scripts/run_prepare_fineweb_edu_100B.sh [--clean] [--no-resume] [--foreground]

Environment overrides:
  ROOT=$ROOT
  REPO=$REPO
  CONDA=$CONDA
  CONDA_ENV=$CONDA_ENV
  OUT=$OUT
  TOKENIZER=$TOKENIZER
  PROXY=$PROXY
  HF_ENDPOINT=$HF_ENDPOINT
  TRAIN_TOKENS=$TRAIN_TOKENS
  VAL_TOKENS=$VAL_TOKENS
  BATCH_SIZE=$BATCH_SIZE
  CONFIG=$CONFIG

Examples:
  bash scripts/run_prepare_fineweb_edu_100B.sh
  bash scripts/run_prepare_fineweb_edu_100B.sh --clean
  PROXY=http://127.0.0.1:7890 bash scripts/run_prepare_fineweb_edu_100B.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --no-resume)
      RESUME=0
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

if pgrep -af "prepare_fineweb_edu_packed.py.*$OUT" >/dev/null; then
  echo "A prepare job for this output directory is already running:" >&2
  pgrep -af "prepare_fineweb_edu_packed.py.*$OUT" >&2
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

if [[ ! -f "$REPO/scripts/prepare_fineweb_edu_packed.py" ]]; then
  echo "Prepare script not found: $REPO/scripts/prepare_fineweb_edu_packed.py" >&2
  exit 1
fi

if [[ ! -d "$TOKENIZER" ]]; then
  echo "Tokenizer directory not found: $TOKENIZER" >&2
  exit 1
fi

mkdir -p "$ROOT/hf_cache" "$ROOT/hf_cache/datasets" "$ROOT/tmp" "$ROOT/logs" "$(dirname "$OUT")"

if [[ "$CLEAN" -eq 1 ]]; then
  echo "Cleaning output directory: $OUT"
  rm -rf "$OUT"
elif [[ "$RESUME" -ne 1 && -n "$(find "$OUT" -maxdepth 2 -type f 2>/dev/null | head -1)" ]]; then
  echo "Output directory already contains files: $OUT" >&2
  echo "Use --clean to remove it or omit --no-resume." >&2
  exit 1
fi

mkdir -p "$OUT"

export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"
export http_proxy="$PROXY"
export https_proxy="$PROXY"
export HF_ENDPOINT
export HF_HOME="$ROOT/hf_cache"
export HF_DATASETS_CACHE="$ROOT/hf_cache/datasets"
export HF_HUB_DISABLE_XET=1
export TMPDIR="$ROOT/tmp"

CMD=(
  "$CONDA" run --no-capture-output -n "$CONDA_ENV" python "$REPO/scripts/prepare_fineweb_edu_packed.py"
  --out-root "$OUT"
  --dataset HuggingFaceFW/fineweb-edu
  --config "$CONFIG"
  --split train
  --tokenizer "$TOKENIZER"
  --train-tokens "$TRAIN_TOKENS"
  --val-tokens "$VAL_TOKENS"
  --train-dir-name train_100B
  --val-dir-name val_50M
  --manifest-name fineweb_edu_100B_manifest.json
  --batch-size "$BATCH_SIZE"
)

if [[ "$RESUME" -eq 1 ]]; then
  CMD+=(--resume)
fi

echo "Output: $OUT"
echo "Log: $LOG"
echo "Proxy: $PROXY"
echo "HF endpoint: $HF_ENDPOINT"
echo "Train tokens: $TRAIN_TOKENS"
echo "Val tokens: $VAL_TOKENS"
echo "Resume: $RESUME"

cd "$REPO"

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec "${CMD[@]}"
fi

nohup "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
echo "Started prepare job: PID $pid"
echo "Progress:"
echo "  cat $OUT/fineweb_edu_100B_manifest.json"
echo "  tail -f $LOG"
echo "Stop:"
echo "  kill $pid"
