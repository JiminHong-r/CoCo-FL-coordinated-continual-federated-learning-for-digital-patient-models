#!/bin/bash
# WP2 pipeline (TD Sec. 2.2, tasks 1-6) on one tokenised cohort: the 10/40/50 split,
# the per-centre characterisation, and a population backbone trained only on the 40
# % reserve.
#
#   task 1  preprocessing. ETHOS tokenisation already handles irregular time gaps
#           (passed to the T-LSTM), missing values (quantile and special tokens)
#           and variable-length histories (sliding windows). Step 0 runs only if
#           the cohort folder does not exist yet.
#   task 2  take the 10 % global test set out first
#   task 3  keep 40 % for population pre-training only (no centre sees it)
#   task 4  split the remaining 50 % into 3 centres with Dirichlet(alpha),
#           alpha in {0.1, 0.5, 1.0, 10.0}
#   task 5  describe each centre (class balance, sequence length, temporal density)
#   task 6  train/val/test (70/15/15) inside each centre, with no leakage
#
# This runs the quantity-only partition with a 10,000-window backbone, the WP2
# characterisation setting. The reported WP4/WP5 grid uses the README's steps 3-5
# instead. All splitting and leakage checks live in scripts/build_split_v2.py.
set -e
set -o pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$PROJECT/venv}"
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# One cohort folder. The global test set, the reserve and the centres are all cut
# from it, because the loaders read a single tokenised folder.
COHORT_DIR=$PROJECT/data/tokenized/mimic_train

SPLITS=$PROJECT/data/federated_splits_v2
WP2_ART=$PROJECT/wp2_artifacts_v2
# Static features + T-LSTM only (no note encoder).
POP_OUT=$PROJECT/models/population_reserve_nonotes
GLOBAL_TEST=$SPLITS/global_test.json
POP_RESERVE=$SPLITS/population_reserve.json
ALPHAS=0.1,0.5,1.0,10.0

echo "============================================================"
echo "[$(date)] WP2 pipeline v2 — 10/40/50 single-cohort split"
echo "  cohort dir: $COHORT_DIR"
echo "  splits:     $SPLITS"
echo "============================================================"

# Step 0 (task 1): tokenise the cohort if it is not there yet.
if [[ ! -f $COHORT_DIR/static_data.pickle ]]; then
  echo "[task 1] tokenising cohort -> $COHORT_DIR"
  cocofl_tokenize \
    input_dir=$PROJECT/data/mimic_meds/data/train \
    output_dir=$COHORT_DIR \
    worker=4
else
  echo "[task 1] cohort already tokenised (irregular intervals / missing / "
  echo "         variable-length handled by ETHOS) — skipping STEP 0"
fi

# Step 1 (tasks 2-4, 6): the 10/40/50 split and the Dirichlet centres.
echo "[tasks 2-4,6] building 10/40/50 split"
python scripts/build_split_v2.py \
  --tokenized-dir $COHORT_DIR \
  --out-dir $SPLITS \
  --global-test-frac 0.10 \
  --population-frac 0.40 \
  --federated-frac 0.50 \
  --alphas $ALPHAS \
  --num-hospitals 3 \
  --frac-train 0.70 --frac-val 0.15 --frac-test 0.15 \
  --seed 42

# Step 2 (task 5): describe the actual centre assignments.
echo "[task 5] characterising per-centre distributions"
python scripts/wp2_alpha_sweep.py \
  --tokenized-dir $COHORT_DIR \
  --splits-dir $SPLITS \
  --alphas $ALPHAS \
  --num-hospitals 3 \
  --out-dir $WP2_ART

# Step 3 (task 3): pre-train the backbone on the 40 % reserve only.
# population_reserve_path keeps every centre and global-test patient out of it.
echo "[task 3] pre-training population backbone (static + T-LSTM) on the 40% reserve"
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
torchrun --no_python --standalone --nproc_per_node=${NUM_GPUS:-1} cocofl_train \
  --config-name=population_training \
  data_fp=$COHORT_DIR \
  val_size=0.05 \
  population_reserve_path=$POP_RESERVE \
  n_train_windows=10000 \
  out_dir=$POP_OUT \
  device=cuda

# Step 4 (task 2): population baseline. The head is fitted on the reserve only and
# scored on the separate 10 % global test set.
echo "[task 2] population baseline on the 10% global held-out test"
# The probe handles class imbalance (stratified draw, class_weight="balanced",
# selection on validation AUROC). See src/cocofl/downstream/probe.py.
cocofl_finetune \
  model_path=$POP_OUT/best_model.pt \
  +train_dir=$COHORT_DIR \
  input_dir=$COHORT_DIR \
  +global_test_path=$GLOBAL_TEST \
  +train_restrict_path=$POP_RESERVE \
  +probe=balanced \
  +n_train_samples=20000 +n_eval_samples=0 +batch_size=16 \
  +val_frac=0.1 \
  output_dir=$PROJECT/results_population_v2_global_test_nonotes

echo "============================================================"
echo "[$(date)] WP2 pipeline v2 COMPLETE"
echo "  split:            $SPLITS/{global_test,population_reserve,alpha_*}.json"
echo "  characterisation: $WP2_ART/wp2_alpha_report.md"
echo "  backbone:         $POP_OUT/best_model.pt"
echo "  population AUROC: $PROJECT/results_population_v2_global_test_nonotes/"
echo "============================================================"
