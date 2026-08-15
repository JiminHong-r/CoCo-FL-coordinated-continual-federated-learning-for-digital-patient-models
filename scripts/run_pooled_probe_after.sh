#!/usr/bin/env bash
# Waits for the closeout batch (PID supplied as $1) to exit, then runs pooled
# linear probe across 3 tasks × A2_lora.
set -uo pipefail

CLOSEOUT_PID="${1:-1188540}"
MODEL_PATH=/home/jimin_hong/coco-fl/models/tmlstm/best_model.pt
TRAIN_DIR=/home/jimin_hong/coco-fl/data/tokenized/mimic_train
INPUT_DIR=/home/jimin_hong/coco-fl/data/tokenized/mimic_test

echo "Waiting for PID $CLOSEOUT_PID..."
while kill -0 "$CLOSEOUT_PID" 2>/dev/null; do
  sleep 60
done
echo "Closeout batch finished — starting pooled probe runs"

mkdir -p artifacts_repr_pooled
for TASK in hospital_mortality readmission icu_admission; do
  echo "=== pooled probe: $TASK ==="
  python -u -m cocofl.analysis.pooled_probe \
    --model-path "$MODEL_PATH" \
    --train-dir "$TRAIN_DIR" --test-dir "$INPUT_DIR" \
    --task "$TASK" --strategy A2_lora \
    --n-patients 10 --n-iters 80 \
    --out-csv "artifacts_repr_pooled/pooled_${TASK}_A2_lora.csv" \
    2>&1 | tee "artifacts_repr_pooled/pooled_${TASK}_A2_lora.log"
done

echo "=== POOLED PROBE DONE ==="
