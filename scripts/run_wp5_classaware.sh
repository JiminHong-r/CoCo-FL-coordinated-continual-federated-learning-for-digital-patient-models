#!/bin/bash
# WP4/WP5 v5 eval, run per-task for robustness.
# Each of the 4 tasks runs in its OWN process + output dir so a single crash
# (the overnight single-process run died silently mid-cell) cannot wipe the
# others. Uses the NO-NOTES (static + T-LSTM only) reserve-trained backbone.
# Merges all 4 task CSVs into results_wp5_classaware/wp5_results.csv at the end.
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1

# Code tree — see run_wp5_v7_full50.sh. CODE_SRC lets a run use new code without
# overwriting the src/ that in-flight jobs import from.
export PYTHONPATH=${CODE_SRC:-$PROJECT/src}
echo "code tree: $PYTHONPATH"

POP_MODEL=$PROJECT/models/population_reserve_nonotes_full/best_model.pt
COHORT=$PROJECT/data/tokenized/mimic_train
SPLITS=${SPLITS:-$PROJECT/data/federated_splits_v2_classaware}

# Shared canonical config — see scripts/wp5_common.sh. This script previously
# carried its own knob list and had drifted badly: cohort_local_steps=15,
# n_patient_iters=10, 5 patients per centre, eval CAPPED at 150 windows, and — the
# fatal one — NO population_head_dir, so the classifier head was randomly
# initialised and population_only scored at chance. An A/B against the random
# split run under those settings compared nothing. Sourcing the shared config
# makes the two arms of the A/B identical in every knob except --splits-dir.
# shellcheck source=scripts/wp5_common.sh
source "$(dirname "$0")/wp5_common.sh"

FAILED_TASKS=()
for TASK in ${TASKS:-hospital_mortality readmission icu_admission}; do
  OUT=$PROJECT/results_wp5_classaware_$TASK
  wp5_args
  echo "============================================================"
  echo "[$(date)] === $TASK -> $OUT  (class-aware split, no-notes backbone) ==="
  echo "============================================================"
  cocofl_evaluate "${WP5_COMMON[@]}" output_dir=$OUT 2>&1 | tee "$OUT.log" \
    || echo "[$(date)] $TASK FAILED (continuing to next task)"
  if [ ! -s "$OUT/wp5_results.csv" ]; then
    echo "[$(date)] !! $TASK produced NO wp5_results.csv — see $OUT.log"
    FAILED_TASKS+=("$TASK")
  fi
done
echo "[$(date)] all tasks done; merging"

$VENV/bin/python - <<'PY'
import pandas as pd, os
proj = "/home/jimin_hong/coco-fl"
parts = []
for t in ["hospital_mortality", "readmission", "icu_admission"]:
    fp = f"{proj}/results_wp5_classaware_{t}/wp5_results.csv"
    if os.path.exists(fp):
        df = pd.read_csv(fp); parts.append(df)
        print(f"  + {fp}: {len(df)} rows")
    else:
        print(f"  ! missing {fp}")
if parts:
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(f"{proj}/results_wp5_classaware", exist_ok=True)
    out.to_csv(f"{proj}/results_wp5_classaware/wp5_results.csv", index=False)
    print(f"merged {len(out)} rows -> results_wp5_classaware/wp5_results.csv")
    print(out.groupby(['task','alpha']).size())
PY
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
  echo "[$(date)] INCOMPLETE — no results for: ${FAILED_TASKS[*]}"
  echo "Re-run with: TASKS='${FAILED_TASKS[*]}' GPU=\$GPU bash scripts/run_wp5_classaware.sh"
  exit 1
fi
echo "[$(date)] ALL DONE"
