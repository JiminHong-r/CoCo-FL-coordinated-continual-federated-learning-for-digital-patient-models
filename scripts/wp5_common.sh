#!/bin/bash
# The one WP4/WP5 evaluation configuration. Every runner sources this file, so all
# methods, tasks, alphas, splits and ablation arms run with the same settings.
# (Separate copies of this list drifted apart once, which made results
# incomparable.) Override any value from the environment instead of copying the
# list.
#
# Usage (the caller sets PROJECT, POP_MODEL, COHORT, SPLITS and TASK):
#     source "$(dirname "$0")/wp5_common.sh"
#     wp5_args                     # builds the WP5_COMMON array
#     cocofl_evaluate "${WP5_COMMON[@]}" output_dir=... [extra settings]
#
# wp5_args is a function so an ablation arm can change CL, WEIGHTING,
# POP_UPDATE_FREQ and so on, then rebuild. Hydra rejects the same +key twice on one
# command line, so override through these variables, not by appending a second copy.

wp5_args () {
  WP5_COMMON=(
    model_path=$POP_MODEL
    +train_dir=$COHORT
    input_dir=$COHORT
    "+tasks=[$TASK]"
    "+alphas=[${ALPHAS:-0.1,0.5,1.0,10.0}]"
    +splits_dir=$SPLITS
    # Overridable so a check run can use a stable copy of the heads while a refit
    # rewrites the live folder.
    +population_head_dir=${HEAD_DIR:-$PROJECT/models/population_heads}
    +num_hospitals=${NUM_HOSPITALS:-3}

    # Rounds and model selection. patience=999 means early stopping never fires; the
    # best validation round is kept, with round 0 (the un-adapted population model)
    # as a candidate.
    +max_rounds=${MAX_ROUNDS:-5} +patience=999 +min_rounds=${MAX_ROUNDS:-5}

    # Cohort level. 875 steps x batch 32 = 28,000 windows per centre per round.
    +cohort_local_steps=${COHORT_STEPS:-875}
    +cohort_batch_size=${COHORT_BATCH:-32}
    +cohort_bottleneck=${COHORT_BOTTLENECK:-64}
    +cohort_lr=${COHORT_LR:-1e-3}
    # Class rebalancing: keep every positive, cap negatives at 3 per positive, and
    # recompute pos_weight from the resampled draw. Off by default in the harness,
    # on here for every run: at ~2 % prevalence the natural draw trains the adapter
    # away from the class-balanced population head.
    +cohort_balanced_sampling=${BALANCED:-true}
    +cohort_max_neg_per_pos=${MAX_NEG_PER_POS:-3.0}

    # Patient level. 60 iterations x batch 8 = 480 windows per adapter; far fewer
    # windows let the 66 K-parameter adapter memorise them.
    +n_patient_iters=${PATIENT_ITERS:-60}
    +patient_batch_size=${PATIENT_BATCH:-8}
    +patient_min_samples=${PATIENT_MIN:-60}
    +patient_lr=${PATIENT_LR:-5e-3}
    +max_patients_per_hospital=${MAX_PATIENTS:-50}
    +patients_per_round=${PATIENTS_PER_ROUND:-20}
    +patient_eval_patients=${PATIENT_EVAL_PATIENTS:-20}

    # Evaluation. 0 = every window in the partition. Only smoke tests should lower
    # it; a reported run must leave it at 0.
    +eval_per_hospital=${EVAL_PER_HOSP:-0}
    +val_per_hospital=${VAL_PER_HOSP:-2000}
    +traj_eval_per_hospital=${TRAJ_PER_HOSP:-150}
    +eval_batch_size=${EVAL_BATCH_SIZE:-32}
    +n_pop_eval=${N_POP_EVAL:-200}
    # Optionally size the validation set by positives instead of windows (at 2 %
    # prevalence 2,000 windows hold only ~42). 0 = use val_per_hospital.
    +val_target_pos=${VAL_TARGET_POS:-0}
    # Share of validation held back as a probe slice that never drives round
    # selection. It answers "was a better round available at all?"
    # (probe_best_gain_vs_round0), which tells "selection missed it" apart from
    # "there was nothing to find". It leaves ~30 % less data for round selection.
    +val_probe_frac=${VAL_PROBE_FRAC:-0.3}
    # Evidence floor for the MACRO average: a centre needs this many positives and
    # negatives to count, so a tiny centre cannot move MACRO as much as a large one.
    # Per-hospital rows still include every centre.
    +min_eval_pos=${MIN_EVAL_POS:-10}
    +min_eval_neg=${MIN_EVAL_NEG:-10}

    # Coordination and continual learning (ablation arms override these). WEIGHTING
    # sets the upward weighting at both levels; PATIENT_WEIGHTING and
    # COHORT_WEIGHTING override one level each (WP4 task 3).
    +weighting_scheme=${WEIGHTING:-volume}
    +recency_half_life_days=${RECENCY_HALF_LIFE_DAYS:-180}
    # Population update every N rounds (WP4 task 5). AFTER_N_PATIENTS and
    # DIVERGENCE_THRESHOLD below add the other two triggers.
    +population_update_freq=${POP_UPDATE_FREQ:-5}
    # Continual-learning strategy and strength per level (WP4 task 4). CL and
    # CL_ALPHA set both levels; the per-level variables override.
    +cohort_cl_strategy=${COHORT_CL:-${CL:-distill}}
    +cohort_cl_alpha=${COHORT_CL_ALPHA:-${CL_ALPHA:-0.3}}
    +patient_cl_strategy=${PATIENT_CL:-${CL:-distill}}
    +patient_cl_alpha=${PATIENT_CL_ALPHA:-${CL_ALPHA:-0.3}}
  )
  [ -n "${PATIENT_WEIGHTING:-}" ] && WP5_COMMON+=(+patient_weighting_scheme=$PATIENT_WEIGHTING)
  [ -n "${COHORT_WEIGHTING:-}" ] && WP5_COMMON+=(+cohort_weighting_scheme=$COHORT_WEIGHTING)
  [ -n "${AFTER_N_PATIENTS:-}" ] && WP5_COMMON+=(+population_update_after_n_patients=$AFTER_N_PATIENTS)
  # Method changes, off unless asked for. A run with any of them on is not
  # comparable to the reported grid:
  #   PER_HOSPITAL_SELECTION=true  each centre keeps its own best round.
  #   HEAD_BIAS_ONLY=true          also train the classifier head's intercepts, so a
  #                                per-centre base-rate shift is fixed in the head.
  #
  # Seeds, deliberately separate:
  #   SEED      the data: Dirichlet partition, balanced-sampling draw, dataloader order.
  #             Keep it fixed across a campaign.
  #   RUN_SEED  training: torch's global RNG, i.e. adapter initialisation. Vary it to
  #             measure run-to-run variance on the same partition.
  # They use ++ rather than +: `seed` already exists in the base config, and Hydra
  # rejects +seed with "An item is already at 'seed'". ++ adds or overrides.
  [ -n "${SEED:-}" ] && WP5_COMMON+=(++seed=$SEED)
  [ -n "${RUN_SEED:-}" ] && WP5_COMMON+=(++run_seed=$RUN_SEED)
  [ -n "${PER_HOSPITAL_SELECTION:-}" ] && WP5_COMMON+=(+per_hospital_selection=$PER_HOSPITAL_SELECTION)
  [ -n "${HEAD_BIAS_ONLY:-}" ] && WP5_COMMON+=(+train_head_bias_only=$HEAD_BIAS_ONLY)
  # TRAIN_HEAD=true: train the whole classifier head per centre, to test whether the
  # limit is where the capacity sits. Changes what a cohort model is.
  [ -n "${TRAIN_HEAD:-}" ] && WP5_COMMON+=(+train_head=$TRAIN_HEAD)
  # USE_FLOWER=true: run the federated steps through Flower instead of the
  # in-process loop. Same maths, different order, so results agree within about
  # +/-0.0008 AUROC rather than exactly. Do not mix the two in one grid.
  [ -n "${USE_FLOWER:-}" ] && WP5_COMMON+=(+use_flower=$USE_FLOWER)
  # LOCAL_FT_PATIENTS: local_finetune's patient level (per-centre and per-patient
  # fine-tuning, the TD baseline). Already on by default in the harness; set false
  # only to reproduce older per-centre-only runs.
  [ -n "${LOCAL_FT_PATIENTS:-}" ] && WP5_COMMON+=(+local_finetune_patients=$LOCAL_FT_PATIENTS)
  # Added only when set, because the harness treats a missing key as "off".
  if [ -n "${DIVERGENCE_THRESHOLD:-}" ]; then
    WP5_COMMON+=(+divergence_threshold=$DIVERGENCE_THRESHOLD)
  fi
  if [ -n "${METHODS:-}" ]; then
    WP5_COMMON+=("+methods=[$METHODS]")
  fi
}
