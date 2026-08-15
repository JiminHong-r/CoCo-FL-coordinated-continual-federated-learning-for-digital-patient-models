#!/bin/bash
# Candidate ladder: can any legitimate configuration make adaptation beat the
# frozen population model?
#
# BUDGET DESIGN. Training is reduced (3 rounds, 50 steps, 10 patients) so each
# rung takes ~1.5 h instead of ~10 h. Evaluation is NOT reduced — eval_per_hospital
# stays 0 (every window). That asymmetry is deliberate: the smoke config's 100
# eval windows hold ~2 positives at 2 % prevalence, where AUROC is noise and
# "adaptation won" is a coin flip.
#
# READ THE RESULT ASYMMETRICALLY:
#   a WIN here is meaningful and would likely be LARGER at full budget (more
#     training, more adaptation);
#   a NULL here is weak evidence — reduced training means reduced adaptation, so
#     it cannot rule out a win at full budget.
# So this screens FOR wins; it does not establish a null.
#
# Rungs are cumulative: each keeps what the previous one turned on.
#   GPU=2 bash scripts/probe_ladder.sh
set -o pipefail
PROJECT=/home/jimin_hong/coco-fl
export CODE_SRC=${CODE_SRC:-$PROJECT/src_v8}
export METHODS=population_only,hierarchical
export MAX_ROUNDS=${MAX_ROUNDS:-3} COHORT_STEPS=${COHORT_STEPS:-50}
export PATIENTS_PER_ROUND=${PATIENTS_PER_ROUND:-10} PATIENT_EVAL_PATIENTS=10
export VAL_PROBE_FRAC=${VAL_PROBE_FRAC:-0.3}
SPLIT=${SPLIT:-classaware}; TASK=${TASK:-hospital_mortality}; ALPHA=${ALPHA:-0.5}

rung () {
  local name="$1"; shift
  local out=$PROJECT/results_rung_${name}_${SPLIT}_${TASK}_a${ALPHA}
  echo "════════════════════════════════════════════════════════"
  echo "[$(date)] RUNG: $name    extra: $*"
  echo "════════════════════════════════════════════════════════"
  env "$@" GPU=${GPU:-2} SPLIT=$SPLIT TASK=$TASK ALPHA=$ALPHA \
      TAG=rung_${name} CODE_SRC=$CODE_SRC \
      bash "$PROJECT/scripts/run_wp5_cell.sh" || echo "  RUNG $name FAILED (continuing)"
  $PROJECT/venv/bin/python - "$out/wp5_results.csv" "$name" <<'PY'
import csv, sys, os
path, name = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print(f"  {name}: NO CSV"); raise SystemExit
m = [r for r in csv.DictReader(open(path)) if r["hospital"] == "MACRO"][0]
def f(k):
    try: return float(m[k])
    except Exception: return None
pop, hier = f("auc_population_only"), f("auc_hierarchical_cohort")
ppop, phier = f("probe_auroc_population_only"), f("probe_auroc_hierarchical")
gain = None if (pop is None or hier is None) else hier - pop
print(f"\n  ── {name} ──")
print(f"     TEST  population_only={pop}  hierarchical={hier}  gain={gain:+.4f}"
      if gain is not None else f"     TEST  pop={pop} hier={hier}")
if ppop is not None and phier is not None:
    print(f"     PROBE population_only={ppop:.4f}  hierarchical={phier:.4f}  "
          f"gain={phier-ppop:+.4f}   <- decides the config")
print(f"     selected_round={m.get('selected_round_hierarchical')}  "
      f"|delta|={m.get('selected_delta_norm_hierarchical')}")
print(f"     achievability: best_round={m.get('probe_best_round_hierarchical')} "
      f"gain_vs_round0={m.get('probe_best_gain_vs_round0')}")
print(f"     val positives/hospital={m.get('val_positives_per_hospital')}")
print(f"     ECE pop={f('ece_population_only')} hier={f('ece_hierarchical_cohort')}\n")
PY
}

# 1. Refresh-ordering fix alone. The single most likely cause: the reported
#    hierarchical adapter was being attenuated ~27 % immediately before eval.
rung refreshfix
# 2. + validation sized by positives (selection was ranking rounds on ~42).
rung valsize            VAL_TARGET_POS=200
# 3. + each centre keeps its own best round.
rung perhosp            VAL_TARGET_POS=200 PER_HOSPITAL_SELECTION=true
# 4. + intercept free to absorb the per-centre base-rate shift.
rung biasonly           VAL_TARGET_POS=200 PER_HOSPITAL_SELECTION=true HEAD_BIAS_ONLY=true
# 5. train at the natural prevalence instead of a 3:1 resample.
rung nobalance          VAL_TARGET_POS=200 PER_HOSPITAL_SELECTION=true HEAD_BIAS_ONLY=true BALANCED=false
# 6. more rounds — fedavg picked the FINAL round in both completed cells, so the
#    budget was binding.
rung morerounds         VAL_TARGET_POS=200 PER_HOSPITAL_SELECTION=true HEAD_BIAS_ONLY=true MAX_ROUNDS=6

echo "[$(date)] LADDER DONE"
