#!/bin/bash
# WP2 spec-compliant pipeline (TD 2026-06-08, Sec. 2.2 tasks 1-6).
#
# Runs the whole WP2 flow on ONE cohort tokenized dir, producing the corrected
# 10/40/50 three-way split, the per-centre characterisation, and a population
# backbone pre-trained ONLY on the exclusive 40% reserve.
#
#   task 1  preprocess: ETHOS tokenisation already handles irregular time
#           intervals (elapsed_times deltas -> T-LSTM), missing values
#           (quantile / special tokens), and variable-length sequences
#           (sliding windows of n_positions). Re-run STEP 0 only if the
#           cohort dir does not yet exist.
#   task 2  extract 10% global held-out test BEFORE any other split
#   task 3  reserve 40% exclusively for population pre-training (no centre access)
#   task 4  Dirichlet(alpha) partition the remaining 50% into 3 centres,
#           alpha in {0.1, 0.5, 1.0, 10.0}
#   task 5  characterise per-centre distributions (class balance, sequence
#           length, temporal density) on the ACTUAL centre assignments
#   task 6  within-centre train/val/test splits (70/15/15), no leakage across
#           centres or into the population reserve / global test
#
# All splitting + leakage assertions live in scripts/build_split_v2.py.
set -e
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ---- Single cohort dir (the "longitudinal clinical dataset"). All three
# sets (global test / population reserve / federated centres) are carved from
# this one dir so the loaders, which read a single tokenized dir, stay valid.
COHORT_DIR=$PROJECT/data/tokenized/mimic_train

SPLITS=$PROJECT/data/federated_splits_v2
WP2_ART=$PROJECT/wp2_artifacts_v2
# Static + T-LSTM only (no NoteEncoder, per the WP2 design decision).
POP_OUT=$PROJECT/models/population_reserve_nonotes
GLOBAL_TEST=$SPLITS/global_test.json
POP_RESERVE=$SPLITS/population_reserve.json
ALPHAS=0.1,0.5,1.0,10.0

echo "============================================================"
echo "[$(date)] WP2 pipeline v2 — 10/40/50 single-cohort split"
echo "  cohort dir: $COHORT_DIR"
echo "  splits:     $SPLITS"
echo "============================================================"

# ---- STEP 0 (task 1): preprocess/tokenise the cohort if not already present.
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

# ---- STEP 1 (tasks 2-4, 6): build the 10/40/50 split + Dirichlet centres.
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

# ---- STEP 2 (task 5): characterise the ACTUAL centre assignments.
echo "[task 5] characterising per-centre distributions"
python scripts/wp2_alpha_sweep.py \
  --tokenized-dir $COHORT_DIR \
  --splits-dir $SPLITS \
  --alphas $ALPHAS \
  --num-hospitals 3 \
  --out-dir $WP2_ART

# ---- STEP 3 (task 3): pre-train the population backbone on the 40% reserve.
# population_reserve_path restricts pre-training to the reserve patients only;
# no centre or global-test patient is ever seen by the backbone.
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

# ---- STEP 4 (task 2): honest population baseline on the 10% global test.
# Head fit on the reserve only (train_restrict_path) and evaluated on the
# disjoint 10% global test (global_test_path) — leakage-free.
echo "[task 2] population baseline on the 10% global held-out test"
# The probe is class-imbalance-aware (stratified draw + class_weight="balanced"
# + selection on val AUROC); at ~2% mortality prevalence the previous unweighted
# uniform-draw variant selected for the base-rate predictor. See
# src/cocofl/downstream/probe.py and scripts/run_population_baseline.sh.
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
echo "  population AUROC:  $PROJECT/results_population_v2_global_test/"
echo "============================================================"
