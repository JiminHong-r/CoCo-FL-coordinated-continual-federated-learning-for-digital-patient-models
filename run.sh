#!/bin/bash
# Single entry point. Each stage is a thin wrapper over the script that already
# does the work, in the order the README documents.
#
#   bash run.sh tests            # unit tests, no data and no GPU needed
#   bash run.sh smoke            # whole WP5 path at a tiny budget (~20 min)
#   bash run.sh splits           # both partitions, seed 42
#   bash run.sh backbone         # population pre-training on the 40% reserve
#   bash run.sh heads            # task heads + the population baseline table
#   bash run.sh grid             # the 12 reported cells
#   bash run.sh analysis         # cells -> reported tables
#   bash run.sh all              # splits -> backbone -> heads -> grid -> analysis
#
# There is no single blocking "reproduce everything" command because the full
# grid is roughly a week of GPU time; `all` runs the stages in order, and any
# stage can be re-run on its own. Override PROJECT, VENV, GPU or COHORT from the
# environment.
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV="${VENV:-$PROJECT/venv}"
[ -d "$VENV/bin" ] && export PATH="$VENV/bin:$PATH"
GPU="${GPU:-0}"
COHORT="${COHORT:-$PROJECT/data/tokenized/mimic_train}"
SPLITS="${SPLITS:-$PROJECT/data/federated_splits_v2}"
SPLITS_CA="${SPLITS_CA:-$PROJECT/data/federated_splits_v2_classaware}"
POP_OUT="${POP_OUT:-$PROJECT/models/population_reserve_nonotes_full}"
TAG="${TAG:-v10flw}"
ALPHAS="${ALPHAS:-0.1 0.5 1.0 10.0}"
TASKS="${TASKS:-hospital_mortality readmission icu_admission}"

need_cohort () {
  [ -f "$COHORT/static_data.pickle" ] || {
    echo "No tokenised cohort at $COHORT."
    echo "Run scripts/meds/run_mimic.sh then scripts/run_tokenization.sh first —"
    echo "see the 'Data' section of the README; MIMIC-IV is not redistributable."
    exit 2; }
}

stage_tests ()  { cd "$PROJECT" && python -m pytest tests -q; }

stage_smoke ()  { need_cohort; GPU="$GPU" bash "$PROJECT/scripts/smoke_wp5.sh"; }

stage_splits () {
  need_cohort
  for spec in "$SPLITS:" "$SPLITS_CA:--class-aware"; do
    out="${spec%%:*}"; flag="${spec#*:}"
    echo "[splits] $out ${flag:-(quantity-only)}"
    python "$PROJECT/scripts/build_split_v2.py" \
      --tokenized-dir "$COHORT" --out-dir "$out" \
      --alphas 0.1,0.5,1.0,10.0 --num-hospitals 3 --seed 42 ${flag:+$flag}
  done
}

stage_backbone () {
  need_cohort
  torchrun --no_python --standalone --nproc_per_node="${NUM_GPUS:-1}" cocofl_train \
    --config-name=population_training \
    data_fp="$COHORT" \
    population_reserve_path="$SPLITS/population_reserve.json" \
    val_size=0.05 num_epochs="${EPOCHS:-5}" n_train_windows="${TRAIN_WINDOWS:-0}" \
    out_dir="$POP_OUT" device=cuda
}

stage_heads () { GPU="$GPU" POP_OUT="$POP_OUT" bash "$PROJECT/scripts/run_population_baseline.sh"; }

stage_grid () {
  need_cohort
  local cells=()
  for task in $TASKS; do
    for a in $ALPHAS; do
      cells+=("$TAG:classaware:$task:$a:USE_FLOWER=true,SEED=42,RUN_SEED=42")
    done
  done
  echo "[grid] ${#cells[@]} cells on gpu $GPU"
  GPU="$GPU" bash "$PROJECT/scripts/v9_queue2.sh" "$GPU" 0 "${cells[@]}"
}

stage_analysis () {
  cd "$PROJECT"
  python scripts/summarise_paired.py --glob "results_${TAG}*"
  python scripts/heterogeneity_analysis.py results_${TAG}_*
  python scripts/diagnose_patient_streams.py
}

case "${1:-}" in
  tests|smoke|splits|backbone|heads|grid|analysis) "stage_$1" ;;
  all) stage_splits && stage_backbone && stage_heads && stage_grid && stage_analysis ;;
  *) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
