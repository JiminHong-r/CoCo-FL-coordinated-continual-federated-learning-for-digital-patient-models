#!/bin/bash
# Smoke-test every WP4 knob that has never run on real data, at a tiny budget.
#
# WHY: the full grid is ~42 GPU-hours. The task-3/4/5 plumbing added on
# 2026-08-11 (per-level weighting, time-based recency, "after N patient updates",
# per-level CL strength) has only ever been exercised by unit tests against
# fakes. This project's history is of configurations that run for hours and then
# turn out to have been no-ops or crashes — `uncertainty` died on a missing kwarg
# after 25 h, `recency` decayed by list position for weeks. Spend ten minutes
# proving each knob reaches the coordinator and completes a round.
#
#   GPU=1 bash scripts/smoke_wp4_knobs.sh
set -o pipefail
PROJECT=/home/jimin_hong/coco-fl
export CODE_SRC=${CODE_SRC:-$PROJECT/src_v8}
export METHODS=hierarchical
# Deliberately tiny: 2 rounds, 2 optimiser steps, 2 patients, capped eval.
export MAX_ROUNDS=2 COHORT_STEPS=2 PATIENT_ITERS=2 PATIENTS_PER_ROUND=2
export MAX_PATIENTS=2 PATIENT_EVAL_PATIENTS=2 VAL_PER_HOSP=100
export TRAJ_PER_HOSP=20 N_POP_EVAL=50 EVAL_PER_HOSP=50 ALPHAS=0.5
export SPLIT=${SPLIT:-classaware} TASK=${TASK:-hospital_mortality} TAG=smoke

pass=0; fail=0; failed=()
probe () {
  local name="$1"; shift
  echo "──────── SMOKE: $name ────────"
  if env "$@" GPU=${GPU:-1} SPLIT=$SPLIT TASK=$TASK ALPHA=0.5 TAG=smoke_$name \
       CODE_SRC=$CODE_SRC bash "$PROJECT/scripts/run_wp5_cell.sh" \
       > /tmp/smoke_$name.log 2>&1; then
    echo "   PASS  $name"; pass=$((pass+1))
  else
    echo "   FAIL  $name  (tail below)"; tail -12 /tmp/smoke_$name.log | sed 's/^/      /'
    fail=$((fail+1)); failed+=("$name")
  fi
}

probe recency_time      WEIGHTING=recency
probe uncertainty       WEIGHTING=uncertainty
probe lvl_split         PATIENT_WEIGHTING=recency COHORT_WEIGHTING=uncertainty
probe after_n_patients  POP_UPDATE_FREQ=999 AFTER_N_PATIENTS=6
probe divergence        POP_UPDATE_FREQ=999 DIVERGENCE_THRESHOLD=0.001
probe cl_percentage     COHORT_CL=distill COHORT_CL_ALPHA=0.1 PATIENT_CL=distill PATIENT_CL_ALPHA=0.5
probe cl_ewc            CL=ewc
probe cl_replay         CL=replay

echo
echo "════ SMOKE SUMMARY: $pass passed, $fail failed ════"
[ $fail -gt 0 ] && { echo "FAILED: ${failed[*]}"; exit 1; }
echo "All WP4 knobs reach the coordinator and complete rounds."
