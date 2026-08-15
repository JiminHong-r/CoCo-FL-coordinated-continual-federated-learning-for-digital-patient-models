#!/usr/bin/env bash
# Overnight closeout: representation quality + slow_rate sweep.
# Sequential to keep one GPU happy.
set -uo pipefail

MODEL_PATH=/home/jimin_hong/coco-fl/models/tmlstm/best_model.pt
TRAIN_DIR=/home/jimin_hong/coco-fl/data/tokenized/mimic_train
INPUT_DIR=/home/jimin_hong/coco-fl/data/tokenized/mimic_test

mkdir -p artifacts_repr artifacts_sr02 artifacts_sr10

echo "=== [1/4] representation quality: 3 tasks × A2_lora, 5 patients ==="
for TASK in hospital_mortality readmission icu_admission; do
  echo "--- repr quality $TASK ---"
  python -u -m cocofl.analysis.representation_quality \
    --model-path "$MODEL_PATH" \
    --train-dir "$TRAIN_DIR" --test-dir "$INPUT_DIR" \
    --task "$TASK" --strategy A2_lora \
    --n-patients 5 --n-iters 80 \
    --out-csv "artifacts_repr/repr_${TASK}_A2_lora.csv" \
    2>&1 | tee "artifacts_repr/repr_${TASK}_A2_lora.log"
done

echo "=== [2/4] slow_rate=0.2 federated sim ==="
python -u -m cocofl.federated.simulate \
  model_path="$MODEL_PATH" \
  +train_dir="$TRAIN_DIR" input_dir="$INPUT_DIR" \
  output_dir=artifacts_sr02 \
  +n_centres=3 +n_rounds=6 +local_iters=40 +slow_rate=0.2 \
  +n_patients=12 +train_frac=0.5 +min_samples_per_patient=8 \
  +n_head_train=1500 +n_pop_eval=500 \
  2>&1 | tee artifacts_sr02/run.log

echo "=== [3/4] slow_rate=1.0 federated sim (vanilla FedAvg) ==="
python -u -m cocofl.federated.simulate \
  model_path="$MODEL_PATH" \
  +train_dir="$TRAIN_DIR" input_dir="$INPUT_DIR" \
  output_dir=artifacts_sr10 \
  +n_centres=3 +n_rounds=6 +local_iters=40 +slow_rate=1.0 \
  +n_patients=12 +train_frac=0.5 +min_samples_per_patient=8 \
  +n_head_train=1500 +n_pop_eval=500 \
  2>&1 | tee artifacts_sr10/run.log

echo "=== [4/4] comm efficiency (cheap, in-process) ==="
python -u -m cocofl.analysis.comm_efficiency \
  --model-path "$MODEL_PATH" \
  --rounds-csv artifacts_sr02/federated_rounds.csv \
  --out-csv artifacts_sr02/comm_efficiency.csv

echo "=== ALL DONE ==="
