#!/bin/bash
# WP4/WP5 v7 — the whole 50% federated pool, on a partition that actually works.
#
# WHY v7 EXISTS (this is not "v6 with bigger numbers")
# ----------------------------------------------------
# v6 set eval_per_hospital=0 intending "score the whole test partition", and
# budgeted a day of GPU for it. It would not have done that. `bind_to_dataset`
# looked up each window's patient with `patient_id_at_idx[sample_index]`, but
# that accessor is keyed by TOKEN index. The task datasets hold ~490 K samples
# against ~247 M tokens, so range(len(dataset)) only ever reached the first
# 490 K tokens — owned by 385 of 328,164 patients. Every window was attributed
# to whoever owned token i, which is unrelated to that window's own patient.
#
# Measured on the alpha=1.0 hospital_mortality test partition:
#   patients per centre   30.9% / 56.0% / 13.1%
#   windows, FIXED        11,065 / 20,105 / 5,309   (30.3 / 55.1 / 14.6 %)
#   windows, OLD           5,547 / 42,981 / 1,455   (11.1 / 86.0 /  2.9 %)
# The fixed binding tracks patient proportions; the old one gave one centre 86%
# of the data. The partition was wrong in COMPOSITION, so alpha was not actually
# controlling heterogeneity. eval_per_hospital=0 was therefore uncapping a pool
# that held the wrong windows.
#
# Fixed in src/cocofl/federated/hospital_split.py (keys on start_indices, as
# personalisation/loop.py already did). Regression tests:
# tests/test_hospital_split_binding.py. With the fix a centre holds ~57,300
# train and ~12,300 test windows instead of ~190 total.
#
# WHAT CHANGES vs v6
# ------------------
#   cohort_batch_size=32       NEW. Cohort training could only run at batch 1 —
#                              downstream windows are variable-length, so
#                              collate's torch.stack failed above 1. v7 uses the
#                              shuffled length-bucketed sampler
#                              (evaluation/batching.make_train_loader), so
#                              batches stay exact-length homogeneous and nothing
#                              is padded into the T-LSTM recurrence.
#   val_per_hospital=2000      was 150. Validation now drives MODEL SELECTION
#                              (best round is restored, round 0 = the un-adapted
#                              population model is a candidate), not just early
#                              stopping. At 2.09 % prevalence 150 windows is ~3
#                              positives per hospital, and it showed: fedavg
#                              selected round 5 on the highest val macro of any
#                              method (0.9228) and came in BELOW baseline on
#                              test. 2000 gives ~42 positives, which is the
#                              minimum for the macro to mean anything.
#   cohort_local_steps=100     was 15. local_steps counts OPTIMISER STEPS, so at
#                              batch 32 a centre now sees 3,200 windows/round
#                              (16,000 over 5 rounds, ~28 % of its partition)
#                              against 15 windows/round before. This is the
#                              "use the whole dataset" change on the training
#                              side; eval_per_hospital=0 is the one on the
#                              scoring side.
#
# THIS IS A NEW EXPERIMENT, NOT A PRECISION RE-RUN. v6 was careful to hold the
# optimisation fixed so differences were attributable to sample size. v7 changes
# both the partition (bug fix) and the amount of cohort training, so v7-vs-v5
# differences are NOT attributable to any single cause. Report it as a rerun on
# a corrected partition with a larger training budget.
#
# COST at the measured ~69 windows/s (length-bucketed, uncontended A6000):
#   eval  ~2.9 M window-passes           -> ~12 h
#   train 3 hosp x 5 rounds x 3,200 x 4 alpha x 5 methods x 4 tasks, with
#         backward, at ~25 windows/s     -> ~11 h
#   Budget ~1 day. Runs per-task so a crash cannot take everything.
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1

# Code tree. CODE_SRC pins an alternative tree so a run is isolated from
# in-flight edits — and, more importantly, so launching a run on new code does
# not require overwriting the src/ that already-running jobs import from.
export PYTHONPATH=${CODE_SRC:-$PROJECT/src}
echo "code tree: $PYTHONPATH"

POP_MODEL=$PROJECT/models/population_reserve_nonotes_full/best_model.pt
COHORT=$PROJECT/data/tokenized/mimic_train
SPLITS=${SPLITS:-$PROJECT/data/federated_splits_v2}
TAG=${TAG:-v7}

# Shared canonical config — see scripts/wp5_common.sh. Sourced rather than
# copied so the ablations, the class-aware run and this one cannot drift apart.
# shellcheck source=scripts/wp5_common.sh
source "$(dirname "$0")/wp5_common.sh"

FAILED_TASKS=()
for TASK in ${TASKS:-hospital_mortality readmission icu_admission}; do
  OUT=$PROJECT/results_wp5_${TAG}_$TASK
  wp5_args
  echo "============================================================"
  echo "[$(date)] === $TASK -> $OUT  (full 50% pool) ==="
  echo "============================================================"
  cocofl_evaluate "${WP5_COMMON[@]}" output_dir=$OUT 2>&1 | tee "$OUT.log" \
    || echo "[$(date)] $TASK FAILED (continuing to next task)"
  # A task that dies mid-cell leaves an EMPTY output dir and a short log, and the
  # old loop said nothing further — which is how readmission came to be missing
  # from the WP5 tables while the run looked like it had covered three tasks.
  # Check for the artefact, not the exit status: the pipe to tee masks it.
  if [ ! -s "$OUT/wp5_results.csv" ]; then
    echo "[$(date)] !! $TASK produced NO wp5_results.csv — see $OUT.log"
    FAILED_TASKS+=("$TASK")
  fi
done
echo "[$(date)] all tasks done; merging"

$VENV/bin/python - <<PY
import pandas as pd, os
proj, tag = "$PROJECT", "$TAG"
parts = []
for t in ["hospital_mortality", "readmission", "icu_admission"]:
    fp = f"{proj}/results_wp5_{tag}_{t}/wp5_results.csv"
    if os.path.exists(fp):
        df = pd.read_csv(fp); parts.append(df)
        print(f"  + {fp}: {len(df)} rows")
    else:
        print(f"  ! missing {fp}")
if parts:
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(f"{proj}/results_wp5_{tag}", exist_ok=True)
    out.to_csv(f"{proj}/results_wp5_{tag}/wp5_results.csv", index=False)
    print(f"merged {len(out)} rows -> results_wp5_{tag}/wp5_results.csv")
    per = out[out.hospital != "MACRO"]
    if "n_eval_windows" in per:
        # The load-bearing check: under the bug this was ~190. If it still is,
        # the partition fix did not take effect and nothing below is meaningful.
        print("\neval windows per hospital-row (expect thousands, NOT ~190):")
        print(per.groupby("task")["n_eval_windows"].describe()[["count", "min", "mean", "max"]])
PY
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
  echo "[$(date)] INCOMPLETE — no results for: ${FAILED_TASKS[*]}"
  echo "Do NOT report this run as covering all tasks. Re-run the missing ones with"
  echo "  TASKS='${FAILED_TASKS[*]}' GPU=\$GPU bash scripts/run_wp5_v7_full50.sh"
  exit 1
fi
echo "[$(date)] ALL DONE"
