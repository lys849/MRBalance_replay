#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BASE="convaiinnovations/laya-typed-decisions"
DATA_DIR="experiments/pilot-fold0/data"
LOG_DIR="experiments/pilot-fold0/logs"
mkdir -p "$LOG_DIR"

for k in 1 2 3 4; do
  FOLD_DIR="$DATA_DIR/fold$k"
  CKPT="experiments/pilot-fold0/checkpoint-fold$k"
  EVAL="experiments/pilot-fold0/eval-fold$k-test"

  if [ ! -f "$CKPT/model.safetensors" ]; then
    echo "=== train fold $k ==="
    uv run laya-train \
      --data "$FOLD_DIR/train.jsonl" \
      --eval "$DATA_DIR/dev.jsonl" \
      --base "$BASE" \
      --out "$CKPT" \
      --loss soft-ce \
      --epochs 4 \
      --micro-batch 2 \
      --grad-accum 32 \
      --max-len 128 \
      --device mps \
      --seed 209 2>&1 | tee "$LOG_DIR/train-fold$k.log"
  else
    echo "=== skip train fold $k (checkpoint exists) ==="
  fi

  if [ ! -f "$EVAL/metrics.json" ]; then
    echo "=== eval fold $k ==="
    uv run python -m eci_decision_models.finetune_eval \
      --checkpoint "$CKPT" \
      --test-jsonl "$FOLD_DIR/test.jsonl" \
      --pairs-jsonl "$FOLD_DIR/test_pairs.jsonl" \
      --output-dir "$EVAL" \
      --device mps \
      --run-id "fold$k-test" 2>&1 | tee "$LOG_DIR/eval-fold$k.log"
  else
    echo "=== skip eval fold $k (metrics exist) ==="
  fi
done

echo "=== all folds complete ==="
