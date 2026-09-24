#!/usr/bin/env bash
# Population baselines (WP3 tasks 3-4) with the class-imbalance-aware probe.
#
# Replaces the head-probe invocation inside run_wp2_pipeline_v2.sh. Same
# leakage-free protocol — head fit on the 40% population reserve only
# (train_restrict_path), scored on the disjoint 10% global test (global_test_path,
# n_eval_samples=0 -> every window) — but the head itself is now:
#
#   * fit on a CLASS-STRATIFIED draw (every positive kept), not a uniform one,
#   * fit with class_weight="balanced" (the rare class carries equal total
#     weight), not unweighted BCE,
#   * selected on validation AUROC, not validation loss.
#
# The old protocol reported mortality at 0.589 / readmission at 0.666; at ~2%
# prevalence its selection rule rewarded predicting the base rate. See
# src/cocofl/downstream/probe.py for the full argument.
#
# Run this ON THE SERVER (needs the tokenized cohort, the population checkpoint
# and a GPU). Override paths via env vars or edit the defaults below.
#
#   PROJECT=/path/to/coco-fl bash scripts/run_population_baseline.sh
#
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Code tree. MUST be set to the tree carrying the elapsed-times fix: without it
# the probe feeds the T-LSTM elapsed=1.0 for every token, which pins the time
# decay at a constant 0.7615 and fits the head on features the backbone does not
# actually produce. That is how the heads in models/population_heads/ were built.
# The console scripts live in the project venv; this script used to rely on the
# caller's PATH, so a detached launch failed with "cocofl_finetune: not found".
export PATH="${VENV:-$PROJECT/venv}/bin:$PATH"
export PYTHONPATH="${CODE_SRC:-$PROJECT/src}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
echo "code tree: $PYTHONPATH   gpu: $CUDA_VISIBLE_DEVICES"
COHORT_DIR="${COHORT_DIR:-$PROJECT/data/tokenized/mimic_train}"
POP_OUT="${POP_OUT:-$PROJECT/models/population_reserve_nonotes_full}"
GLOBAL_TEST="${GLOBAL_TEST:-$PROJECT/data/federated_splits_v2/global_test.json}"
POP_RESERVE="${POP_RESERVE:-$PROJECT/data/federated_splits_v2/population_reserve.json}"
# Training windows drawn from the reserve. Positives are kept first, so this is
# a cap on the NEGATIVES more than anything; raise it if a task is still short.
N_TRAIN="${N_TRAIN:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASKS="${TASKS:-all}"
OUT="${OUT:-$PROJECT/results_population_balanced}"
# Where the per-task torch heads are written. These are what WP4/WP5 load via
# +population_head_dir, so refitting the baseline and refitting the heads must be
# the SAME run — otherwise the heads and the reported baseline disagree.
HEAD_DIR="${HEAD_DIR:-$PROJECT/models/population_heads}"

echo "[$(date)] population baselines — balanced probe"
echo "  checkpoint : $POP_OUT/best_model.pt"
echo "  reserve    : $POP_RESERVE (head training)"
echo "  global_test: $GLOBAL_TEST (evaluation, FULL set)"
echo "  n_train    : $N_TRAIN   batch_size: $BATCH_SIZE"
echo "  output     : $OUT"
echo "  heads      : $HEAD_DIR"

task_arg=()
if [ "$TASKS" != "all" ]; then
  # Hydra needs list syntax for multiple values: +tasks=[a,b,c]. A bare
  # comma-separated string is "ambiguous" and aborts the run.
  case "$TASKS" in
    \[*\]) task_arg=("+tasks=$TASKS") ;;
    *)      task_arg=("+tasks=[$TASKS]") ;;
  esac
fi

cocofl_finetune \
  model_path="$POP_OUT/best_model.pt" \
  +train_dir="$COHORT_DIR" \
  input_dir="$COHORT_DIR" \
  +global_test_path="$GLOBAL_TEST" \
  +train_restrict_path="$POP_RESERVE" \
  +probe=balanced \
  +n_train_samples="$N_TRAIN" \
  +n_eval_samples=0 \
  +batch_size="$BATCH_SIZE" \
  +val_frac=0.1 \
  +save_head_dir="$HEAD_DIR" \
  "${task_arg[@]}" \
  output_dir="$OUT"

echo "============================================================"
echo "[$(date)] DONE -> $OUT/downstream_auc_backbone.csv"
python3 - "$OUT/downstream_auc_backbone.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
w = max(len(r["task"]) for r in rows)
below = []
for r in rows:
    auc = float(r["auc"])
    flag = "" if auc >= 0.7 else "   <-- BELOW 0.70"
    if auc < 0.7:
        below.append(r["task"])
    print(f"  {r['task']:<{w}}  AUROC={auc:.4f}  AUPRC={float(r['auprc']):.4f}"
          f"  prevalence={float(r['prevalence']):.3%}{flag}")
print()
print("all tasks >= 0.70" if not below else f"still below 0.70: {', '.join(below)}")
PY
echo "============================================================"
