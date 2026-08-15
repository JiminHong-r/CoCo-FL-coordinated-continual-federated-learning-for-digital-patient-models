#!/bin/bash
# WP4 tasks 3, 4, 5 — the ablations the TD asks for, on the full 50% pool.
#
# WHY THIS EXISTS
# ---------------
# The TD asks WP4 to *design and compare*:
#   task 3  weighting schemes for upward propagation (volume / uncertainty /
#           recency), and
#   task 5  triggering conditions for population sync (after N updates, or on a
#           divergence threshold).
# Both were implemented and neither was ever compared. `run_evaluation.py` did
# not read `weighting_scheme` or `divergence_threshold`, so every reported run
# used volume + frequency-only. `uncertainty` was additionally non-functional:
# coordinator.py passed a hardcoded loss of 1.0 for every patient, so
# inverse-loss weights were all equal and it silently behaved as `uniform`.
# Both are fixed; `train_patient_adapter` now returns its mean task BCE.
#
# Task 4 (catastrophic forgetting) has always been runnable — `cohort_cl_strategy`
# was exposed — but its only ablation is from the legacy notes/old-split run,
# which also predates the hospital-partition fix. It is re-run here.
#
# DATASET USE (per supervisor: use the whole 50% pool). Uncapped where it is
# reported, documented subsets where it is not — full cohort training is ~3
# weeks of GPU, so it is a subset by necessity, not an oversight:
#   eval_per_hospital=0     FULL — every window of a centre's test partition
#   val/traj = 150          capped; runs every round, used only to rank rounds
#   cohort training         3,200 windows/centre/round (~28% of partition over 5)
#   eval_per_hospital=0     score every window of a centre's test partition
#   cohort_batch_size=32    length-bucketed training batches
#   cohort_local_steps=100  3,200 windows/centre/round (v5/v6 used 15 windows)
#   patient_min_samples=20  ~10 eval windows/patient, 20 patients pooled
#
# SCOPE: only the `hierarchical` method is affected by these knobs — fedavg,
# local_finetune and population_only have no upward propagation — so each arm
# runs one method, not five. Run on ONE task to keep the comparison legible;
# hospital_mortality is the default since it is the most α-sensitive.
#
#   GPU=2 TASK=hospital_mortality bash scripts/run_wp4_ablations.sh
#
# Each arm is a separate output dir so a crash cannot lose the others.
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1

# Code tree. The main checkout now carries every fix, so this defaults there;
# CODE_SRC only exists to pin an alternative tree if a run must be isolated from
# in-flight edits.
export PYTHONPATH=${CODE_SRC:-$PROJECT/src}
echo "code tree: $PYTHONPATH"

POP_MODEL=$PROJECT/models/population_reserve_nonotes_full/best_model.pt
COHORT=$PROJECT/data/tokenized/mimic_train
# Which split the ablations run on. This was hardcoded to the RANDOM split,
# which is the wrong default now that class-aware is the primary: every knob
# these arms vary (weighting scheme, CL strategy, sync trigger) acts on the
# hierarchical method's adaptation, and on the random split adaptation is barely
# selected at all — at α=1.0 and α=10.0 every method falls back to round 0, so
# the arms would be comparing no-ops. Ablate on the split where adaptation is
# actually engaged.
SPLIT=${SPLIT:-classaware}
case "$SPLIT" in
  random)     SPLITS=$PROJECT/data/federated_splits_v2 ;;
  classaware) SPLITS=$PROJECT/data/federated_splits_v2_classaware ;;
  /*)         SPLITS=$SPLIT ;;
  *)          echo "SPLIT must be random|classaware|/abs/path, got '$SPLIT'"; exit 2 ;;
esac
[ -d "$SPLITS" ] || { echo "no such splits dir: $SPLITS"; exit 2; }
TASK=${TASK:-hospital_mortality}
# Split goes in the tag so a random-split arm can never be merged with a
# class-aware one by filename alone.
TAG=${TAG:-wp4abl}_${SPLIT}

# Shared canonical config — see scripts/wp5_common.sh. Every arm here uses the
# SAME knobs as the main WP5 run; arms differ only in the variable they set
# before calling ``wp5_args``. Arms launched before 2026-08-10 predate this and
# are not comparable to these (different patient budget, no balanced sampling).
# shellcheck source=scripts/wp5_common.sh
source "$(dirname "$0")/wp5_common.sh"

# Only the hierarchical method responds to these knobs. Running the other four
# per arm would triple the cost and produce bit-identical baseline columns.
export METHODS=hierarchical

run_arm () {
  local name="$1"; shift
  local out=$PROJECT/results_${TAG}_${TASK}_${name}
  wp5_args   # rebuild with whatever CL / WEIGHTING / POP_UPDATE_FREQ is set now
  echo "============================================================"
  echo "[$(date)] === ARM: $name -> $out ==="
  echo "============================================================"
  cocofl_evaluate "${WP5_COMMON[@]}" "$@" output_dir="$out" 2>&1 | tee "$out.log" \
    || echo "[$(date)] ARM $name FAILED (continuing)"
}

# ---- Shared reference config -------------------------------------------------
# volume weighting + distill CL + population sync every 5 rounds. This is the
# default, and it is simultaneously the `volume` arm of task 3, the `distill` arm
# of task 4, and the `freq5` arm of task 5. Run once, referenced three times.
WEIGHTING=volume CL=distill POP_UPDATE_FREQ=5 run_arm "ref_volume_distill_freq5"

# ---- Task 3: upward-propagation weighting -----------------------------------
# "including weighting schemes based on data volume, uncertainty, or update
# recency" — and the TD defines TWO mechanisms (patient->cohort, and cohort->
# population), so the scheme is varied per level, not once for both.
#
# `recency` now decays by ELAPSED TIME (half-life in days) using each update's
# real timestamps. The previous implementation decayed by position in the
# within-round patient list, where every update is equally recent, so that arm
# measured arbitrary ordering rather than recency.
#
# CL held at distill so the only thing moving is the weighting.
#
# NOTE: `volume` is NOT run here. volume+distill+freq5 is the default
# configuration, which means the would-be arms `w_volume`, `cl_distill` and
# `trig_freq5` are byte-identical runs of the same config. It is run ONCE below
# as `ref_volume_distill_freq5` and referenced by all three tasks — running it
# three times would burn ~6 GPU-hours to recompute the same numbers.
for scheme in uniform recency uncertainty; do
  WEIGHTING=$scheme CL=distill run_arm "w_${scheme}"
done

# Per-level contrast: does it matter WHERE the scheme is applied? Both arms hold
# the other level at volume, so the difference isolates one mechanism.
PATIENT_WEIGHTING=recency COHORT_WEIGHTING=volume CL=distill \
  run_arm "w_lvl_patient_recency"
PATIENT_WEIGHTING=volume COHORT_WEIGHTING=recency CL=distill \
  run_arm "w_lvl_cohort_recency"

# ---- Task 4: catastrophic-forgetting mitigation ------------------------------
# Re-run on the CORRECTED partition; the existing table is legacy old-split.
# Read BWT_cohort here, not just AUROC — forgetting is the quantity of interest.
#
# NOTE on reading this arm: in any (α) cell where model selection lands on
# round 0 the CL strategy is a no-op by construction — no adaptation happened for
# it to regularise — so all four arms return bit-identical numbers there. That is
# not four strategies agreeing. Check ``selected_round_hierarchical`` in the CSV
# and drop those cells from the comparison.
# `distill` omitted — it is the shared reference config (see above).
for cl in none ewc replay; do
  WEIGHTING=volume CL=$cl run_arm "cl_${cl}"
done

# "...calibrated to the update frequency at each hierarchy level." The arms above
# apply ONE strategy at ONE strength to both levels, which does not vary the
# calibration the TD asks about. The patient level updates fastest (60 iters per
# patient per round) and drifts most, so it should carry the stronger anchor;
# the cohort level updates once per round and needs less. These two arms make
# that contrast explicit — same strategy, per-level strength swapped.
COHORT_CL=distill COHORT_CL_ALPHA=0.1 PATIENT_CL=distill PATIENT_CL_ALPHA=0.5 \
  WEIGHTING=volume run_arm "cl_calib_fast_strong"
COHORT_CL=distill COHORT_CL_ALPHA=0.5 PATIENT_CL=distill PATIENT_CL_ALPHA=0.1 \
  WEIGHTING=volume run_arm "cl_calib_fast_weak"

# ---- Task 5: coordination protocol ------------------------------------------
# freq5  = one sync on the final round (what every reported run did)
# freq2  = sync every 2 rounds — the "after N updates" arm
# diverge= divergence-triggered on top of a slow frequency
#
# The divergence threshold is a GUESS until the observed divergences are known.
# ``max_cohort_divergence`` is logged per round: if it never reaches the
# threshold the arm is silently identical to freq5 again, which is exactly the
# failure the 2026-08-08 fix was supposed to end. Read that column before
# reporting this arm.
# freq5 is the shared reference config (see task 3) — not re-run here.
WEIGHTING=volume CL=distill POP_UPDATE_FREQ=2 run_arm "trig_freq2"
# "after N patient updates" — the TD's actual first condition. POP_UPDATE_FREQ is
# pushed out of reach so this trigger is the only one that can fire; otherwise a
# round-frequency sync would pre-empt it and the arm would duplicate freq5.
# 20 patients/round × 3 centres = 60 patient updates per round, so N=90 fires
# roughly every other round and is distinguishable from both freq arms.
WEIGHTING=volume CL=distill POP_UPDATE_FREQ=999 \
  AFTER_N_PATIENTS=${AFTER_N_PATIENTS:-90} run_arm "trig_after_n_patients"
WEIGHTING=volume CL=distill POP_UPDATE_FREQ=999 \
  DIVERGENCE_THRESHOLD=${DIVERGENCE_THRESHOLD:-0.5} run_arm "trig_diverge"

echo "[$(date)] ALL ARMS DONE"
$VENV/bin/python - <<PY
import pandas as pd, glob, os, re
proj, tag, task = "$PROJECT", "$TAG", "$TASK"
rows = []
for d in sorted(glob.glob(f"{proj}/results_{tag}_{task}_*")):
    if not os.path.isdir(d):
        continue
    fp = os.path.join(d, "wp5_results.csv")
    if not os.path.exists(fp):
        print(f"  ! missing {fp}"); continue
    df = pd.read_csv(fp)
    df["arm"] = re.sub(rf"^results_{tag}_{task}_", "", os.path.basename(d))
    rows.append(df)
if rows:
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(f"{proj}/results_{tag}_{task}", exist_ok=True)
    out.to_csv(f"{proj}/results_{tag}_{task}/ablations.csv", index=False)
    # wp5_results.csv is WIDE: one column per method, plus a MACRO hospital row.
    macro = out[out.hospital.astype(str) == "MACRO"]
    cols = [c for c in ("auc_hierarchical_cohort", "auc_hierarchical_patient",
                        "bwt_hier", "patient_propagation_gain") if c in macro]
    if len(macro) and cols:
        print("\\nMACRO by arm (hierarchical):")
        print(macro.groupby("arm")[cols].mean().sort_values(
            cols[0], ascending=False).to_string())
    else:
        print("\\n! no MACRO rows or expected columns found; inspect ablations.csv")
    print(f"\\nmerged {len(out)} rows -> results_{tag}_{task}/ablations.csv")
PY
