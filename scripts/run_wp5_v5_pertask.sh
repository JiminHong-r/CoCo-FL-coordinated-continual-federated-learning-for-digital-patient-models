#!/bin/bash
# WP4/WP5 v5 eval, run per-task for robustness.
# Each of the 4 tasks runs in its OWN process + output dir so a single crash
# (the overnight single-process run died silently mid-cell) cannot wipe the
# others. Uses the NO-NOTES (static + T-LSTM only) reserve-trained backbone.
# Merges all 4 task CSVs into results_wp5_v5/wp5_results.csv at the end.
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1

POP_MODEL=$PROJECT/models/population_reserve_nonotes_full/best_model.pt
COHORT=$PROJECT/data/tokenized/mimic_train
SPLITS=$PROJECT/data/federated_splits_v2

for TASK in hospital_mortality readmission icu_admission; do
  OUT=$PROJECT/results_wp5_v5_$TASK
  echo "============================================================"
  echo "[$(date)] === $TASK -> $OUT  (no-notes backbone) ==="
  echo "============================================================"
  cocofl_evaluate \
    model_path=$POP_MODEL \
    +train_dir=$COHORT \
    input_dir=$COHORT \
    output_dir=$OUT \
    "+tasks=[$TASK]" \
    "+alphas=[0.1,0.5,1.0,10.0]" \
    +splits_dir=$SPLITS \
    +num_hospitals=3 \
    +max_rounds=5 +patience=999 +min_rounds=5 \
    +cohort_local_steps=15 +n_patient_iters=10 \
    +max_patients_per_hospital=5 +patients_per_round=3 \
    +n_pop_eval=200 +eval_per_hospital=150 \
    +cohort_bottleneck=64 \
    +cohort_lr=1e-3 +patient_lr=5e-3 \
    +cohort_cl_strategy=distill +cohort_cl_alpha=0.3 \
    +patient_cl_strategy=distill +patient_cl_alpha=0.3 \
    || echo "[$(date)] $TASK FAILED (continuing to next task)"
done
echo "[$(date)] all tasks done; merging"

$VENV/bin/python - <<'PY'
import pandas as pd, os
proj = "/home/jimin_hong/coco-fl"
parts = []
for t in ["hospital_mortality", "readmission", "icu_admission"]:
    fp = f"{proj}/results_wp5_v5_{t}/wp5_results.csv"
    if os.path.exists(fp):
        df = pd.read_csv(fp); parts.append(df)
        print(f"  + {fp}: {len(df)} rows")
    else:
        print(f"  ! missing {fp}")
if parts:
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(f"{proj}/results_wp5_v5", exist_ok=True)
    out.to_csv(f"{proj}/results_wp5_v5/wp5_results.csv", index=False)
    print(f"merged {len(out)} rows -> results_wp5_v5/wp5_results.csv")
    print(out.groupby(['task','alpha']).size())
PY
echo "[$(date)] ALL DONE"
