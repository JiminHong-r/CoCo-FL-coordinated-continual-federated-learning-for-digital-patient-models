#!/bin/bash
# ONE WP5 cell = one (split, task, α). Its own process, its own GPU, its own
# output dir.
#
# WHY PER-CELL
# ------------
# The per-task runners loop α inside a single process, which has cost us twice:
# a task that died mid-α took the whole task's output with it (readmission left
# an empty dir after 5 h), and a 3-α sweep is forced to run sequentially even
# when GPUs are idle. One cell per process fixes both — a crash loses one cell,
# and the α sweep fans out across cards.
#
# Config comes from wp5_common.sh, so a cell is comparable to every other run by
# construction. The only thing that varies is which split/task/α it points at.
#
# USAGE
#   GPU=0 SPLIT=classaware TASK=hospital_mortality ALPHA=1.0 \
#     CODE_SRC=/home/jimin_hong/coco-fl/src_v8 bash scripts/run_wp5_cell.sh
#
#   SPLIT is `random` or `classaware` (or an absolute path to a splits dir).
#   Output: results_v8_<split>_<task>_a<alpha>/wp5_results.csv
set -o pipefail

VENV=/home/jimin_hong/coco-fl/venv
PROJECT=/home/jimin_hong/coco-fl
export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1
export PYTHONPATH=${CODE_SRC:-$PROJECT/src}

TASK=${TASK:?set TASK}
ALPHA=${ALPHA:?set ALPHA}
SPLIT=${SPLIT:-random}
TAG=${TAG:-v8}

case "$SPLIT" in
  random)     SPLITS=$PROJECT/data/federated_splits_v2 ;;
  classaware) SPLITS=$PROJECT/data/federated_splits_v2_classaware ;;
  /*)         SPLITS=$SPLIT ;;
  *)          echo "SPLIT must be random|classaware|/abs/path, got '$SPLIT'"; exit 2 ;;
esac
[ -d "$SPLITS" ] || { echo "no such splits dir: $SPLITS"; exit 2; }

POP_MODEL=$PROJECT/models/population_reserve_nonotes_full/best_model.pt
COHORT=$PROJECT/data/tokenized/mimic_train

# --- pre-flight VRAM gate ---------------------------------------------------
# The box is SHARED. A class-aware α=1.0 cell ran 11.7 h and then died of CUDA
# OOM because another user's notebook kernel had taken 43 of the card's 47 GiB;
# our process never got above 4.4 GiB. Nothing was wrong with the run — it was
# simply started on a card that could not hold it.
#
# A cell needs ~6 GiB. Wait for that much to be free before starting, rather
# than launching into a starved card and finding out half a day later. This
# cannot prevent an OOM caused by someone allocating *after* we start; it only
# stops us from beginning a doomed run.
NEED_MIB=${NEED_MIB:-8000}
GATE_WAIT_MIN=${GATE_WAIT_MIN:-180}
gate_deadline=$(( $(date +%s) + GATE_WAIT_MIN * 60 ))
while :; do
  free_mib=$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" \
      --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -n "$free_mib" ] || { echo "cannot read GPU ${CUDA_VISIBLE_DEVICES} memory; proceeding"; break; }
  [ "$free_mib" -ge "$NEED_MIB" ] && { echo "[$(date)] gpu${CUDA_VISIBLE_DEVICES}: ${free_mib} MiB free ≥ ${NEED_MIB} — starting"; break; }
  if [ "$(date +%s)" -gt "$gate_deadline" ]; then
    echo "[$(date)] gpu${CUDA_VISIBLE_DEVICES}: only ${free_mib} MiB free after ${GATE_WAIT_MIN}min — GIVING UP on this cell"
    exit 3
  fi
  echo "[$(date)] gpu${CUDA_VISIBLE_DEVICES}: ${free_mib} MiB free < ${NEED_MIB} needed — waiting (other users on the card)"
  sleep 300
done
OUT=$PROJECT/results_${TAG}_${SPLIT}_${TASK}_a${ALPHA}

# Never start on top of an existing result. A stale wp5_results.csv left in the
# target dir is indistinguishable from this run's own output until the run
# finishes, and anything that polls the dir for progress will read the OLD file
# and believe the new run has already produced results. That is not
# hypothetical: a June-25 CSV sitting in a reused output dir caused a live α=0.5
# run to be killed 1.5 h in, because a "has α=0.5 finished?" check matched the
# stale rows. Archive with a timestamp rather than delete — the old numbers are
# still evidence of what was run when.
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  ARCHIVE="$OUT.stale_$(date +%Y%m%d_%H%M%S)"
  mv "$OUT" "$ARCHIVE"
  echo "[$(date)] target dir was NOT empty — archived prior contents to $(basename "$ARCHIVE")"
fi
mkdir -p "$OUT"

# shellcheck source=scripts/wp5_common.sh
source "$(dirname "$0")/wp5_common.sh"
ALPHAS=$ALPHA wp5_args

echo "============================================================"
echo "[$(date)] CELL split=$SPLIT task=$TASK α=$ALPHA gpu=$CUDA_VISIBLE_DEVICES"
echo "  code tree: $PYTHONPATH"
echo "  splits:    $SPLITS"
echo "  out:       $OUT"
echo "============================================================"

cocofl_evaluate "${WP5_COMMON[@]}" output_dir="$OUT" 2>&1 | tee "$OUT.log"

# Check the artefact, not the exit status — the pipe to tee masks it, which is
# how a dead task previously reported success.
if [ -s "$OUT/wp5_results.csv" ]; then
  echo "[$(date)] CELL OK: $OUT/wp5_results.csv ($(wc -l < "$OUT/wp5_results.csv") lines)"
else
  echo "[$(date)] CELL FAILED — no wp5_results.csv in $OUT; see $OUT.log"
  exit 1
fi
