#!/usr/bin/env bash
# Wait for the running cell, then run the cross-centre diagnostic ALONE, then
# resume the cell queue.
#
#   bash scripts/v9_after_replicate.sh <gpu> <wait_pid>
#
# WHY SEQUENTIAL, NOT ALONGSIDE
# -----------------------------
# Only one GPU is available to this project (others need the cards). Running the
# diagnostic concurrently with a cell technically stays on one card but puts two
# processes on it — which is exactly what OOM-killed `classaware readmission
# α=10.0` four hours in when another user's job expanded. The diagnostic is short
# (~1-2 h) and the queue is sequential anyway, so it costs little to slot it in
# between cells and removes the contention risk entirely.
#
# Progress bars go to stderr and are kept in a separate file, so the diagnostic's
# stdout stays a small readable matrix rather than thousands of tqdm lines.
set -u
PROJECT=/home/jimin_hong/coco-fl
GPU=${1:?gpu}; WAITPID=${2:?pid to wait for}

if [ "$WAITPID" != "0" ]; then
  echo "[$(date)] waiting for pid $WAITPID (running cell) to exit..."
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
  echo "[$(date)] cell finished — GPU is free"
  sleep 30
fi

export PATH=$PROJECT/venv/bin:$PATH
export PYTHONPATH=$PROJECT/src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$GPU

cd "$PROJECT" || exit 1

for spec in "hospital_mortality:0.5" "readmission:0.1"; do
  IFS=: read -r task alpha <<< "$spec"
  echo "[$(date)] --- cross-centre transfer: $task alpha=$alpha ---"
  python scripts/cross_centre_transfer.py \
    --test-dir   data/tokenized/mimic_train \
    --model-path models/population_reserve_nonotes_full/best_model.pt \
    --splits-dir data/federated_splits_v2_classaware \
    --task "$task" --alpha "$alpha" \
    --n-train 6000 --n-test 8000 --batch-size 8 \
    > "/tmp/xfer_${task}_a${alpha}.txt" 2> "/tmp/xfer_${task}_a${alpha}.progress" \
    && echo "[$(date)] OK -> /tmp/xfer_${task}_a${alpha}.txt" \
    || echo "[$(date)] !! FAILED (see /tmp/xfer_${task}_a${alpha}.progress)"
done

echo "[$(date)] diagnostic done — resuming cell queue"
exec bash "$PROJECT/scripts/v9_queue2.sh" "$GPU" 0 \
  "v9:classaware:readmission:10.0:EVAL_BATCH_SIZE=16" \
  "v9:random:hospital_mortality:0.1" \
  "v9:random:hospital_mortality:10.0" \
  "v9:random:readmission:0.5" \
  "v9:random:readmission:1.0" \
  "v9:random:icu_admission:0.5" \
  "v9:random:icu_admission:1.0"
