#!/usr/bin/env bash
# Population baseline STABILITY check (WP3 tasks 3-4).
#
# Re-runs the leakage-free head-probe on the FULL 10% global test set N times and
# reports mean +/- std per task. Purpose: show that scoring the entire global
# test (instead of a 3,000-sample subset) removes the mortality-AUROC swing that
# came from having only ~50 positives in a capped draw.
#
# Each rep runs under seed=$rep. It previously ran every rep at the fixed default
# seed, so the reps drew the SAME training windows and the reported spread
# measured nothing but run-to-run nondeterminism.
#
# Run this ON THE SERVER (needs the tokenized cohort, the population checkpoint
# and a GPU). Override paths via env vars or edit the defaults below.
#
#   PROJECT=/path/to/coco-fl N_REPS=3 bash scripts/run_population_stability.sh
#
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
COHORT_DIR="${COHORT_DIR:-$PROJECT/data/tokenized/mimic_train}"
POP_OUT="${POP_OUT:-$PROJECT/models/population_reserve_nonotes}"
GLOBAL_TEST="${GLOBAL_TEST:-$PROJECT/data/federated_splits_v2/global_test.json}"
POP_RESERVE="${POP_RESERVE:-$PROJECT/data/federated_splits_v2/population_reserve.json}"
N_REPS="${N_REPS:-3}"
# TASKS: comma-separated subset to evaluate (default: just mortality, the only
# low-prevalence / unstable task). Set TASKS=all to run all four.
TASKS="${TASKS:-hospital_mortality}"
OUT_ROOT="${OUT_ROOT:-$PROJECT/results_population_stability}"

echo "[$(date)] population baseline stability: $N_REPS reps on the FULL global test"
echo "  checkpoint : $POP_OUT/best_model.pt"
echo "  global_test: $GLOBAL_TEST"

for rep in $(seq 1 "$N_REPS"); do
  out="$OUT_ROOT/rep_${rep}"
  echo "---- rep $rep/$N_REPS -> $out ----"
  task_arg=()
  if [ "$TASKS" != "all" ]; then task_arg=(+tasks="$TASKS"); fi
  cocofl_finetune \
    model_path="$POP_OUT/best_model.pt" \
    +train_dir="$COHORT_DIR" \
    input_dir="$COHORT_DIR" \
    +global_test_path="$GLOBAL_TEST" \
    +train_restrict_path="$POP_RESERVE" \
    +probe=balanced \
    +n_train_samples="${N_TRAIN:-20000}" +n_eval_samples=0 +batch_size="${BATCH_SIZE:-16}" \
    +val_frac=0.1 +seed="$rep" \
    "${task_arg[@]}" \
    output_dir="$out"
done

echo "============================================================"
echo "AGGREGATE (mean +/- std across $N_REPS reps, FULL global test)"
python3 - "$OUT_ROOT" "$N_REPS" <<'PY'
import sys, glob, csv, statistics, os
out_root, n = sys.argv[1], int(sys.argv[2])
by_task = {}
for f in sorted(glob.glob(os.path.join(out_root, "rep_*", "downstream_auc_backbone.csv"))):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            by_task.setdefault(r["task"], []).append(float(r["auc"]))
if not by_task:
    print("no result CSVs found under", out_root); sys.exit(1)
w = max(len(t) for t in by_task)
print(f"{'task':<{w}}  mean    std     min     max     reps")
for t, v in by_task.items():
    sd = statistics.pstdev(v) if len(v) > 1 else 0.0
    print(f"{t:<{w}}  {statistics.mean(v):.4f}  {sd:.4f}  {min(v):.4f}  {max(v):.4f}  {len(v)}")
print("\nstd <= 0.03 => stable / reportable as a single number.")
print("std still high on mortality => raise +n_train_samples (head trains on ~60 positives).")
PY
