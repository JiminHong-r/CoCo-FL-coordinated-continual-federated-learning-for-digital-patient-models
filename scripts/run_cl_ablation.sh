#!/bin/bash
# CL ablation: same config, 4 CL strategies (none, distill, ewc, replay)
# Tests Spec WP4 task 4: which catastrophic-forgetting mitigation works best?
#
# 1 task (hospital_mortality) × 4 α × 5 methods × 4 CL strategies
# Each strategy = separate output dir for clean comparison.
set -e
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

POP_MODEL=$PROJECT/models/population_with_notes_full/best_model.pt
SPLITS=$PROJECT/data/federated_splits

for STRATEGY in none distill ewc replay; do
    OUT=$PROJECT/results_wp5_cl_${STRATEGY}
    echo "============================================================"
    echo "[$(date)] CL ablation: strategy=${STRATEGY}"
    echo "============================================================"
    cocofl_evaluate \
      model_path=$POP_MODEL \
      +train_dir=$PROJECT/data/tokenized/mimic_train \
      input_dir=$PROJECT/data/tokenized/mimic_test \
      output_dir=$OUT \
      "+tasks=[hospital_mortality]" \
      "+alphas=[0.1,0.5,1.0,10.0]" \
      +splits_dir=$SPLITS \
      +num_hospitals=3 \
      +max_rounds=5 +patience=999 +min_rounds=5 \
      +cohort_local_steps=15 +n_patient_iters=10 \
      +max_patients_per_hospital=5 +patients_per_round=3 \
      +n_pop_eval=200 +eval_per_hospital=150 \
      +cohort_bottleneck=64 \
      +cohort_cl_strategy=${STRATEGY} +cohort_cl_alpha=0.3 \
      +patient_cl_strategy=${STRATEGY} +patient_cl_alpha=0.3
    python scripts/summarize_wp5.py --csv $OUT/wp5_results.csv --out $OUT/wp5_summary.md
done

echo "[$(date)] CL ablation DONE — 4 strategy folders generated"
ls -d $PROJECT/results_wp5_cl_*/
