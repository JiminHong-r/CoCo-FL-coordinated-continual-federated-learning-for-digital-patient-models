#!/bin/bash
# Canonical WP4/WP5 evaluation configuration — ONE definition, sourced by every
# runner, so numbers are comparable across methods, tasks, α, splits and arms.
#
# WHY THIS FILE EXISTS
# --------------------
# The WP5 main run, the class-aware run and the WP4 ablations each carried their
# own copy of the knob list, and the copies drifted apart. The damaging drift was
# ``cohort_balanced_sampling``: it defaults to FALSE in run_evaluation.py and was
# set only on the final A/B command line, so the published v7 tables and every
# ablation arm trained on the natural ~2 %-prevalence draw while the A/B trained
# on a 3:1 resampled one. Those numbers were never comparable, and neither script
# said so — RESULTS.md's WP4 config table listed balanced sampling as if it had
# been on throughout.
#
# Sourcing this file means a knob change lands in every runner at once. Override
# any value from the environment; do not fork the list.
#
# USAGE
#   PROJECT/POP_MODEL/COHORT/SPLITS/TASK must be set by the caller, then:
#     source "$(dirname "$0")/wp5_common.sh"
#     wp5_args                       # (re)builds the WP5_COMMON array
#     cocofl_evaluate "${WP5_COMMON[@]}" output_dir=... [arm-specific knobs]
#
#   ``wp5_args`` is a function, not a static array, so an ablation arm can set
#   CL / WEIGHTING / POP_UPDATE_FREQ and rebuild. Passing the same Hydra ``+key``
#   twice on one command line is an error, so arms must override through these
#   variables rather than by appending a second copy of the key.

wp5_args () {
  WP5_COMMON=(
    model_path=$POP_MODEL
    +train_dir=$COHORT
    input_dir=$COHORT
    "+tasks=[$TASK]"
    "+alphas=[${ALPHAS:-0.1,0.5,1.0,10.0}]"
    +splits_dir=$SPLITS
    # Overridable so a pipeline-verification run can point at a STABLE copy
    # while a refit is rewriting the live directory.
    +population_head_dir=${HEAD_DIR:-$PROJECT/models/population_heads}
    +num_hospitals=${NUM_HOSPITALS:-3}

    # --- rounds + model selection ---------------------------------------------
    # patience=999 never fires; selection is by best validation round, with
    # round 0 (the un-adapted population model) as a candidate.
    +max_rounds=${MAX_ROUNDS:-5} +patience=999 +min_rounds=${MAX_ROUNDS:-5}

    # --- cohort level ---------------------------------------------------------
    +cohort_local_steps=${COHORT_STEPS:-100}
    +cohort_batch_size=${COHORT_BATCH:-32}
    +cohort_bottleneck=${COHORT_BOTTLENECK:-64}
    +cohort_lr=${COHORT_LR:-1e-3}
    # Class rebalancing: keep all positives, cap negatives at 3:1, and recompute
    # pos_weight from the resampled draw. OFF by default in the harness, ON here
    # for every run — at ~2 % prevalence the un-resampled draw trains the adapter
    # to un-learn a class-balanced population head.
    +cohort_balanced_sampling=${BALANCED:-true}
    +cohort_max_neg_per_pos=${MAX_NEG_PER_POS:-3.0}

    # --- patient level --------------------------------------------------------
    # 60 iters × batch 8 = 480 windows per adapter. The pre-fix setting (10 iters
    # at batch 1) fitted 66 K parameters on ten windows, which memorises; every
    # ablation launched before 2026-08-10 used it and is NOT comparable to runs
    # made with this file.
    +n_patient_iters=${PATIENT_ITERS:-60}
    +patient_batch_size=${PATIENT_BATCH:-8}
    +patient_min_samples=${PATIENT_MIN:-60}
    +patient_lr=${PATIENT_LR:-5e-3}
    +max_patients_per_hospital=${MAX_PATIENTS:-50}
    +patients_per_round=${PATIENTS_PER_ROUND:-20}
    +patient_eval_patients=${PATIENT_EVAL_PATIENTS:-20}

    # --- evaluation -----------------------------------------------------------
    # 0 = every window of the partition. Overridable ONLY so smoke tests can cap
    # it; any reported run must leave this at 0.
    +eval_per_hospital=${EVAL_PER_HOSP:-0}
    +val_per_hospital=${VAL_PER_HOSP:-2000}
    +traj_eval_per_hospital=${TRAJ_PER_HOSP:-150}
    +eval_batch_size=${EVAL_BATCH_SIZE:-32}
    +n_pop_eval=${N_POP_EVAL:-200}
    # Evidence floor for the MACRO average: a centre needs this many positives
    # AND negatives to enter it. Without it a 69-patient centre at α=0.1 moves
    # MACRO as much as a 134,000-patient one, which is what made MACRO
    # incomparable across α. Per-hospital rows still record every centre.
    # Size the VAL set by positives, not by a flat window count: at 2 %
    # prevalence 2,000 windows hold ~42 positives, so "best round" is close to a
    # coin toss. 0 = flat val_per_hospital (previous behaviour).
    +val_target_pos=${VAL_TARGET_POS:-0}
    # Fraction of val held back as a probe slice that NEVER drives round
    # selection. Enabled by default from 2026-08-12 because it is the only surface
    # that answers "was a better round available at all?" — the achievability
    # bound (probe_best_gain_vs_round0). Without it, a method that selects round 0
    # is indistinguishable from a method for which no better round existed, and
    # those two call for opposite conclusions.
    #
    # DEVIATION TO RECORD: the probe is carved OUT of validation, so a cell run
    # with it has ~30 % less data for round selection than one run without. The
    # already-completed readmission α=0.5 cell and the mortality α=0.5 cell (plus
    # its bias-only arm) ran WITHOUT it, so those three are mutually comparable
    # and later cells are not strictly comparable to them on selection power.
    # Judged acceptable because readmission had 1,143 validation positives and the
    # best validation gain of any method was +0.00056 — losing 30 % of that
    # evidence cannot change a conclusion at that effect size.
    +val_probe_frac=${VAL_PROBE_FRAC:-0.3}
    +min_eval_pos=${MIN_EVAL_POS:-10}
    +min_eval_neg=${MIN_EVAL_NEG:-10}

    # --- coordination + continual learning (ablation arms override via env) ---
    # Task 3: the scheme is settable per LEVEL. WEIGHTING sets both (the old
    # single-knob behaviour); PATIENT_WEIGHTING / COHORT_WEIGHTING override one
    # level each, which is what "two mechanisms" in the TD requires.
    +weighting_scheme=${WEIGHTING:-volume}
    +recency_half_life_days=${RECENCY_HALF_LIFE_DAYS:-180}
    # Task 5: round frequency and/or "after N patient updates" and/or divergence.
    +population_update_freq=${POP_UPDATE_FREQ:-5}
    # Task 4: CL strategy AND strength per level, so "calibrated to the update
    # frequency at each hierarchy level" can actually be varied. CL/CL_ALPHA set
    # both levels; the per-level vars override.
    +cohort_cl_strategy=${COHORT_CL:-${CL:-distill}}
    +cohort_cl_alpha=${COHORT_CL_ALPHA:-${CL_ALPHA:-0.3}}
    +patient_cl_strategy=${PATIENT_CL:-${CL:-distill}}
    +patient_cl_alpha=${PATIENT_CL_ALPHA:-${CL_ALPHA:-0.3}}
  )
  [ -n "${PATIENT_WEIGHTING:-}" ] && WP5_COMMON+=(+patient_weighting_scheme=$PATIENT_WEIGHTING)
  [ -n "${COHORT_WEIGHTING:-}" ] && WP5_COMMON+=(+cohort_weighting_scheme=$COHORT_WEIGHTING)
  [ -n "${AFTER_N_PATIENTS:-}" ] && WP5_COMMON+=(+population_update_after_n_patients=$AFTER_N_PATIENTS)
  # --- method changes, OFF unless explicitly asked for ----------------------
  # PER_HOSPITAL_SELECTION=true : each centre keeps its own best round instead of
  #   one globally-chosen round imposed on all centres.
  # HEAD_BIAS_ONLY=true        : unfreeze only the classifier head's intercepts,
  #   so a per-centre base-rate shift is corrected where it belongs rather than
  #   by distorting features through a frozen boundary.
  # Both change the METHOD, not a defect, so they must never be on by default —
  # a run with either enabled is not comparable to the campaign's other cells.
  # --- seeds: DATA vs TRAINING, deliberately separate -----------------------
  # SEED     : Dirichlet partition, balanced-sampling draw, dataloader order.
  #            Hold this FIXED across a campaign — varying it redraws which
  #            patients belong to which centre.
  # RUN_SEED : torch global RNG, i.e. adapter initialisation. Vary this to
  #            measure run-to-run variance on the SAME partition.
  # Before this existed the global RNG was unseeded, so no cell was reproducible
  # and a same-config replicate could select a different round purely by chance.
  [ -n "${SEED:-}" ] && WP5_COMMON+=(+seed=$SEED)
  [ -n "${RUN_SEED:-}" ] && WP5_COMMON+=(+run_seed=$RUN_SEED)
  [ -n "${PER_HOSPITAL_SELECTION:-}" ] && WP5_COMMON+=(+per_hospital_selection=$PER_HOSPITAL_SELECTION)
  [ -n "${HEAD_BIAS_ONLY:-}" ] && WP5_COMMON+=(+train_head_bias_only=$HEAD_BIAS_ONLY)
  # LOCAL_FT_PATIENTS=true : make `local_finetune` the TD's actual baseline —
  #   per-centre AND per-patient fine-tuning, patients starting from the
  #   population model. Roughly triples that method's runtime and changes its
  #   numbers, so completed cells are not comparable to cells run with it on.
  [ -n "${LOCAL_FT_PATIENTS:-}" ] && WP5_COMMON+=(+local_finetune_patients=$LOCAL_FT_PATIENTS)
  # Optional: only added when set, since the harness treats absence as "off".
  if [ -n "${DIVERGENCE_THRESHOLD:-}" ]; then
    WP5_COMMON+=(+divergence_threshold=$DIVERGENCE_THRESHOLD)
  fi
  if [ -n "${METHODS:-}" ]; then
    WP5_COMMON+=("+methods=[$METHODS]")
  fi
}
