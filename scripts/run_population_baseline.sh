#!/usr/bin/env bash
# Population baselines (WP3 tasks 3-4) and the per-task heads that WP4/WP5 load.
#
# The head is fitted on the 40 % population reserve only and scored on every window
# of the separate 10 % global test set. To cope with the rare positive class it is:
#   * fitted on a class-stratified draw (every positive kept),
#   * fitted with class_weight="balanced",
#   * chosen by validation AUROC, not validation loss.
# An unweighted fit chosen by loss mostly learns to predict the base rate at ~2 %
# prevalence (see src/cocofl/downstream/probe.py).
#
# Run on the server (it needs the tokenised cohort, the backbone and a GPU):
#   PROJECT=/path/to/coco-fl bash scripts/run_population_baseline.sh
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Use the project venv's console scripts and the given code tree. The code must
# include the elapsed-time fix, or the probe feeds the T-LSTM a constant time gap
# and the heads are fitted on the wrong features.
export PATH="${VENV:-$PROJECT/venv}/bin:$PATH"
export PYTHONPATH="${CODE_SRC:-$PROJECT/src}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
echo "code tree: $PYTHONPATH   gpu: $CUDA_VISIBLE_DEVICES"
COHORT_DIR="${COHORT_DIR:-$PROJECT/data/tokenized/mimic_train}"
POP_OUT="${POP_OUT:-$PROJECT/models/population_reserve_nonotes_full}"
GLOBAL_TEST="${GLOBAL_TEST:-$PROJECT/data/federated_splits_v2/global_test.json}"
POP_RESERVE="${POP_RESERVE:-$PROJECT/data/federated_splits_v2/population_reserve.json}"
# Training windows drawn from the reserve. Positives are kept first, so this mostly
# caps the negatives.
N_TRAIN="${N_TRAIN:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TASKS="${TASKS:-all}"
OUT="${OUT:-$PROJECT/results_population_balanced}"
# Where the per-task heads go. WP4/WP5 load them via +population_head_dir, so the
# heads and the reported baseline must come from the same run.
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
  # Hydra needs list syntax for several values: +tasks=[a,b,c]. A plain
  # comma-separated string aborts the run.
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
