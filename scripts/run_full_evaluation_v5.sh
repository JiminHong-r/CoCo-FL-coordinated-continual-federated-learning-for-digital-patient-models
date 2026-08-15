#!/bin/bash
# WP5 v5: comprehensive evaluation on the corrected single-cohort split.
#   - One cohort dir (mimic_train) split 10/40/50 into global_test /
#     population reserve / federated pool (data/federated_splits_v2/, rebuilt
#     by scripts/build_split_v2.py).
#   - Backbone pre-trained ONLY on the 40% reserve
#     (models/population_with_notes_reserve, see run_wp2_pipeline_v2.sh).
#   - Hospitals = the 50% federated pool, Dirichlet-split per α.
#
# IMPORTANT vs v4: because the split is now over the mimic_train cohort, the
# hospital patient IDs are mimic_train patients — so input_dir AND train_dir
# both point at mimic_train. With v4's input_dir=mimic_test every hospital
# loader resolved to zero patients.
#
# Budget estimate: ~14-18 h.
set -e
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Reserve-trained backbone (40% population reserve only).
# No-notes (static + T-LSTM only) per the WP2 decision to drop the NoteEncoder.
POP_MODEL=$PROJECT/models/population_reserve_nonotes/best_model.pt
COHORT=$PROJECT/data/tokenized/mimic_train
SPLITS=$PROJECT/data/federated_splits_v2
OUT=$PROJECT/results_wp5_v5

echo "============================================================"
echo "[$(date)] WP5 v5 — single-cohort 10/40/50 split"
echo "  cohort:  $COHORT"
echo "  splits:  $SPLITS"
echo "  backbone:$POP_MODEL"
echo "============================================================"

cocofl_evaluate \
  model_path=$POP_MODEL \
  +train_dir=$COHORT \
  input_dir=$COHORT \
  output_dir=$OUT \
  "+tasks=[hospital_mortality,readmission,icu_admission]" \
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
  +patient_cl_strategy=distill +patient_cl_alpha=0.3

echo "[$(date)] WP5 v5 DONE"
python scripts/summarize_wp5.py --csv $OUT/wp5_results.csv --out $OUT/wp5_summary.md
python scripts/analyse_thresholds.py --csv $OUT/wp5_results.csv --out $OUT/threshold_analysis.md
python scripts/subgroup_analysis.py \
  --csv $OUT/wp5_results.csv \
  --static-pickle $COHORT/static_data.pickle \
  --splits $SPLITS \
  --out $OUT/subgroup_analysis.md
echo "Markdown reports written"
ls -la $OUT/
