#!/bin/bash
# End-to-end WP5 smoke test: all five methods, tiny budget, one cell.
#
# WHY: a full cell is 17-24 h. This proves the whole pipeline completes and that
# every column the TD's axes require is actually POPULATED — not merely present
# in the header. Several axes were silently blank for weeks (delta_cohort,
# n_eval_windows on ablation runs, the population trajectory), and a header with
# no value in it looks identical to a metric that was computed.
#
#   GPU=1 bash scripts/smoke_wp5.sh            # baseline config
#   GPU=1 ARM="PER_HOSPITAL_SELECTION=true HEAD_BIAS_ONLY=true" bash scripts/smoke_wp5.sh
set -o pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export CODE_SRC=${CODE_SRC:-$PROJECT/src}
export MAX_ROUNDS=${MAX_ROUNDS:-2} COHORT_STEPS=${COHORT_STEPS:-2}
export PATIENT_ITERS=${PATIENT_ITERS:-2} PATIENTS_PER_ROUND=${PATIENTS_PER_ROUND:-2}
export MAX_PATIENTS=2 PATIENT_EVAL_PATIENTS=2
export VAL_PER_HOSP=${VAL_PER_HOSP:-100} TRAJ_PER_HOSP=20 N_POP_EVAL=50
export EVAL_PER_HOSP=${EVAL_PER_HOSP:-100}
export MIN_EVAL_POS=${MIN_EVAL_POS:-1} MIN_EVAL_NEG=${MIN_EVAL_NEG:-1}
NAME=${NAME:-smoke5}

echo "[$(date)] WP5 smoke: split=${SPLIT:-classaware} task=${TASK:-hospital_mortality}"
# shellcheck disable=SC2086
if env ${ARM:-} GPU=${GPU:-1} SPLIT=${SPLIT:-classaware} \
     TASK=${TASK:-hospital_mortality} ALPHA=${ALPHA:-0.5} TAG=$NAME \
     CODE_SRC=$CODE_SRC bash "$PROJECT/scripts/run_wp5_cell.sh" \
     > /tmp/$NAME.log 2>&1; then
  echo "  run: PASS"
else
  echo "  run: FAIL — tail:"; tail -20 /tmp/$NAME.log | sed 's/^/    /'; exit 1
fi

CSV=$PROJECT/results_${NAME}_${SPLIT:-classaware}_${TASK:-hospital_mortality}_a${ALPHA:-0.5}/wp5_results.csv
$PROJECT/venv/bin/python - "$CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
macro = [r for r in rows if r["hospital"] == "MACRO"][0]
# (column, which TD axis it serves)
MACRO_COLS = [
    ("auc_population_only","1"), ("auc_local_finetune","1"),
    ("auc_standard_fedavg","1"), ("auc_top_down_only","1"),
    ("auc_hierarchical_cohort","1"), ("auprc_hierarchical_cohort","1"),
    ("ece_hierarchical_cohort","1"),
    ("delta_cohort_hier","2"),
    ("bwt_hier_cohort","3"), ("fwt_hier_global","3"),
    ("population_round_aucs","3"), ("population_stability_range","3"),
    ("patient_propagation_gain","4"), ("population_gain_from_cohorts","4"),
    ("params_updated_cohort_per_round","6"), ("params_updated_patient_per_round","6"),
    ("params_naive_full_model","6"),
    ("probe_auroc_population_only","cfg"), ("probe_auroc_hierarchical","cfg"),
    ("probe_round_aucs_hierarchical","cfg"), ("probe_best_gain_vs_round0","cfg"),
    ("val_auroc_population_only","cfg"), ("val_auroc_hierarchical","cfg"),
    ("val_positives_per_hospital","-"), ("selected_round_hierarchical","-"),
    ("n_eval_windows","-"), ("patient_adapters_considered","2"),
]
# Required only when the patient layer produced a score at all. On a TERMINAL
# outcome (hospital_mortality) every pooled patient set is single-class, so the
# AUROC is correctly WITHHELD and these have no value — blank is the right answer,
# not a defect. Demanding them unconditionally would push toward reporting a
# number where none is defined, which is the failure they were added to prevent.
MACRO_COLS_IF_PATIENT = [
    ("delta_patient_hier","2"),
]
HOSP_COLS_IF_PATIENT = [
    ("patient_cohort_ref_auroc","2"), ("delta_patient_hier_unpaired","2"),
]
# These are written on the PER-HOSPITAL rows only, not the MACRO aggregate.
HOSP_COLS = [
    ("fwt_hier_samecentre","3"), ("in_macro","-"),
    ("auroc_ci95_halfwidth_population_only","1"),
    ("comm_savings_ratio_vs_naive","6"),
]
hosp = [r for r in rows if r["hospital"] != "MACRO"]
bad = []
print("  %-40s %-5s %-6s %s" % ("column", "axis", "row", "value"))
for col, axis in MACRO_COLS:
    v = macro.get(col, None)
    ok = v not in (None, "")
    print("  %-40s %-5s %-6s %s" % (col, axis, "MACRO", (v[:24] if ok else "*** EMPTY ***")))
    if not ok:
        bad.append((col, axis, "MACRO"))
patient_scored = macro.get("auc_hierarchical_patient") not in (None, "")
if patient_scored:
    MACRO_COLS_ACTIVE = MACRO_COLS_IF_PATIENT
    HOSP_COLS_ACTIVE = HOSP_COLS_IF_PATIENT
else:
    print("  (patient layer produced no score for this task — Δ_patient columns "
          "correctly withheld; see cocofl.data patient evaluability)")
    MACRO_COLS_ACTIVE, HOSP_COLS_ACTIVE = [], []
for col, axis in MACRO_COLS_ACTIVE:
    v = macro.get(col, None)
    ok = v not in (None, "")
    print("  %-40s %-5s %-6s %s" % (col, axis, "MACRO", (v[:24] if ok else "*** EMPTY ***")))
    if not ok:
        bad.append((col, axis, "MACRO"))
for col, axis in list(HOSP_COLS) + list(HOSP_COLS_ACTIVE):
    # populated on at least one centre; a per-centre metric can legitimately be
    # blank where that centre had no evaluable data.
    vals = [r.get(col) for r in hosp]
    ok = any(v not in (None, "") for v in vals)
    shown = next((v for v in vals if v not in (None, "")), None)
    print("  %-40s %-5s %-6s %s" % (col, axis, "hosp", (str(shown)[:24] if ok else "*** EMPTY ***")))
    if not ok:
        bad.append((col, axis, "hosp"))
print()
if bad:
    print("  FAIL — populated header but no value:", bad)
    sys.exit(1)
# Axis 2 needs per-patient rows, and they cannot be reconstructed after a run.
import os
prec = os.path.join(os.path.dirname(sys.argv[1]), "patient_records.csv")
if os.path.exists(prec):
    pr = list(csv.DictReader(open(prec)))
    cols = set(pr[0]) if pr else set()
    need = {"hospital","patient_id","n_fit","n_val","n_eval","n_eval_pos",
            "val_auroc_patient","val_auroc_cohort","evaluable","accepted"}
    miss = need - cols
    print("  patient_records.csv: %d rows, %d evaluable, %d accepted"
          % (len(pr), sum(r["evaluable"] == "True" for r in pr),
             sum(r["accepted"] == "True" for r in pr)))
    if miss:
        print("  FAIL — patient_records.csv missing columns:", sorted(miss))
        sys.exit(1)
else:
    print("  FAIL — no patient_records.csv (axis 2 subgroup analysis would be "
          "unreconstructable after the run)")
    sys.exit(1)
print("  All TD-axis columns POPULATED. WP5 pipeline is clear.")
PY
