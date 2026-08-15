#!/usr/bin/env bash
# Personalisation ablation runner.
# Required:
#   MODEL_PATH   — WP2 best_model.pt
#   TRAIN_DIR    — population train tokenized data (for offline head finetune)
#   INPUT_DIR    — population test tokenized data
# Optional:
#   MODE=per_patient        | cohort
#   OUTPUT_DIR=artifacts
#   N_ITERS=200
#   N_HEAD_TRAIN=1500
#   N_PATIENTS=20           (per_patient mode)
#   TRAIN_FRAC=0.5          (per_patient mode)
#   MIN_SAMPLES=4           (per_patient mode)
#   POP_PROBE_PATIENTS=3    (per_patient mode)
#   N_PERSONAL_TRAIN=400    (cohort mode)
#   N_PERSONAL_EVAL=400     (cohort mode)
#   N_POP_EVAL=2000
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${TRAIN_DIR:?TRAIN_DIR must be set}"
: "${INPUT_DIR:?INPUT_DIR must be set}"

MODE="${MODE:-per_patient}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
N_ITERS="${N_ITERS:-200}"
N_HEAD_TRAIN="${N_HEAD_TRAIN:-1500}"
N_PATIENTS="${N_PATIENTS:-20}"
TRAIN_FRAC="${TRAIN_FRAC:-0.5}"
MIN_SAMPLES="${MIN_SAMPLES:-4}"
POP_PROBE_PATIENTS="${POP_PROBE_PATIENTS:-3}"
N_PERSONAL_TRAIN="${N_PERSONAL_TRAIN:-400}"
N_PERSONAL_EVAL="${N_PERSONAL_EVAL:-400}"
N_POP_EVAL="${N_POP_EVAL:-2000}"

python -u -m cocofl.personalize.run_ablation \
    model_path="$MODEL_PATH" \
    +train_dir="$TRAIN_DIR" \
    input_dir="$INPUT_DIR" \
    output_dir="$OUTPUT_DIR" \
    +mode="$MODE" \
    +n_iters="$N_ITERS" \
    +n_head_train="$N_HEAD_TRAIN" \
    +n_patients="$N_PATIENTS" \
    +train_frac="$TRAIN_FRAC" \
    +min_samples_per_patient="$MIN_SAMPLES" \
    +pop_probe_patients="$POP_PROBE_PATIENTS" \
    +n_personal_train="$N_PERSONAL_TRAIN" \
    +n_personal_eval="$N_PERSONAL_EVAL" \
    +n_pop_eval="$N_POP_EVAL"
