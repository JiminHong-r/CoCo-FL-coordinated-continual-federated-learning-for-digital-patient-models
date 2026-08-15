#!/bin/bash
# WP4 + WP5 launch on the corrected pipeline (real Δt, refitted heads).
#
# GATED on the pre-flight verification. If that did not report a clear pipeline
# this refuses to start: committing ~85 GPU-hours to an unverified harness is how
# this project lost a day twice already.
#
# STREAM PRIORITY. The question is whether `hierarchical` beats `population_only`,
# so the two cells that answer it most decisively start first, in parallel:
#   * class-aware mortality α=0.5  — the primary task
#   * class-aware readmission α=0.5 — ~410 val positives so selection actually
#     works, 7,376 test positives so the CI is ≈±0.011 rather than ±0.016, and
#     the patient layer is evaluable (37/40 val slices vs 0/40 for mortality)
# The WP4 grid runs alongside on the third card since it is required regardless.
set -u
PROJECT=/home/jimin_hong/coco-fl
export CODE_SRC=$PROJECT/src_v8
PRE=/home/jimin_hong/preflight.log

if ! grep -aq "pipeline is clear" "$PRE" 2>/dev/null; then
  echo "[$(date)] REFUSING TO LAUNCH — pre-flight has not reported a clear pipeline."
  echo "  see $PRE"
  exit 1
fi
echo "[$(date)] pre-flight green; launching"

# Print the FULL argument expansion one cell will receive, and fail if a knob we
# depend on is missing. The probe slice was silently absent from an entire
# 85-GPU-hour campaign because the launcher set VAL_TARGET_POS and left
# VAL_PROBE_FRAC to its default of 0 — while the pre-flight that "verified the
# pipeline" had been run with the probe passed explicitly. A pre-flight that runs
# a different configuration from the campaign verifies the wrong thing.
_show_expansion () {
  ( PROJECT=$PROJECT POP_MODEL=$PROJECT/models/population_reserve_nonotes_full/best_model.pt \
    COHORT=$PROJECT/data/tokenized/mimic_train SPLITS=$PROJECT/data/federated_splits_v2 \
    TASK=hospital_mortality VAL_TARGET_POS=$VAL_TARGET_POS \
    bash -c "source $PROJECT/scripts/wp5_common.sh; wp5_args; printf '%s\n' \"\${WP5_COMMON[@]}\"" )
}
echo "--- effective config for every cell in this launch ---"
_show_expansion | sed "s/^/    /"
for req in val_target_pos val_probe_frac min_eval_pos cohort_balanced_sampling; do
  _show_expansion | grep -q "^+$req=" || { echo "ABORT: $req missing from expansion"; exit 1; }
done
_pf=$(_show_expansion | grep "^+val_probe_frac=" | cut -d= -f2)
[ "$_pf" = "0" ] && echo "  WARNING: val_probe_frac=0 — no achievability bound will be computed"
echo "------------------------------------------------------"

# Validation sized by POSITIVES, not windows. At mortality's 2.09 % prevalence a
# 2,000-window val set holds ~42 positives and "best round" is near-arbitrary —
# the verification showed a better round existing (+0.0059 on the probe) while
# selection took round 0 on nine positives.
export VAL_TARGET_POS=200
export TAG=v9

cell () {  # split task alpha [extra env assignments...]
  local split=$1 task=$2 alpha=$3; shift 3
  env "$@" GPU=$GPU SPLIT=$split TASK=$task ALPHA=$alpha \
      TAG=${CELLTAG:-$TAG} CODE_SRC=$CODE_SRC VAL_TARGET_POS=$VAL_TARGET_POS \
      bash "$PROJECT/scripts/run_wp5_cell.sh" \
    || echo "[$(date)] CELL FAILED: $split/$task/a$alpha (continuing)"
}

stream0 () { GPU=0
  cell classaware hospital_mortality 0.5
  cell random     hospital_mortality 0.5
  cell classaware hospital_mortality 1.0
  cell random     hospital_mortality 1.0
}
stream1 () { GPU=1
  cell classaware readmission 0.5
  # The arm most likely to move AUROC: the intercept is freed so a per-centre
  # base-rate shift is corrected where it belongs instead of by distorting
  # features through a frozen head. Separate TAG so it cannot overwrite the
  # baseline cell it is compared against.
  CELLTAG=v9arm cell classaware hospital_mortality 0.5 \
      HEAD_BIAS_ONLY=true PER_HOSPITAL_SELECTION=true
  cell classaware icu_admission 0.5
}
stream2 () { GPU=2
  env GPU=2 CODE_SRC=$CODE_SRC TASK=hospital_mortality SPLIT=classaware \
      ALPHAS=0.5 TAG=v9abl VAL_TARGET_POS=$VAL_TARGET_POS \
      bash "$PROJECT/scripts/run_wp4_ablations.sh"
}

cd "$PROJECT" || exit 1
setsid nohup bash -c "$(declare -f cell stream0); TAG=$TAG CODE_SRC=$CODE_SRC PROJECT=$PROJECT VAL_TARGET_POS=$VAL_TARGET_POS stream0" > /home/jimin_hong/v9_gpu0.log 2>&1 &
echo "  gpu0 pid $!  (mortality: classaware 0.5 -> random 0.5 -> a=1.0 both)"
setsid nohup bash -c "$(declare -f cell stream1); TAG=$TAG CODE_SRC=$CODE_SRC PROJECT=$PROJECT VAL_TARGET_POS=$VAL_TARGET_POS stream1" > /home/jimin_hong/v9_gpu1.log 2>&1 &
echo "  gpu1 pid $!  (readmission 0.5 -> bias-only arm -> icu 0.5)"
setsid nohup bash -c "$(declare -f stream2); CODE_SRC=$CODE_SRC PROJECT=$PROJECT VAL_TARGET_POS=$VAL_TARGET_POS stream2" > /home/jimin_hong/v9_gpu2.log 2>&1 &
echo "  gpu2 pid $!  (WP4 grid: 14 arms, a=0.5)"
echo "[$(date)] all three streams launched"
