#!/bin/bash
# Run one WP5 cell, i.e. one (split, task, alpha), in its own process with its own
# output folder. If a run crashes you lose only that cell, and cells can run on
# different GPUs at the same time. All settings come from wp5_common.sh, so every
# cell is comparable; only the split, task and alpha change.
#
# Usage:
#   GPU=0 SPLIT=classaware TASK=hospital_mortality ALPHA=1.0 \
#     bash scripts/run_wp5_cell.sh
#
# SPLIT is `random`, `classaware`, or an absolute path to a splits folder. Output:
# results_<TAG>_<split>_<task>_a<alpha>/wp5_results.csv
set -o pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${VENV:-$PROJECT/venv}"
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

POP_MODEL=${POP_MODEL:-$PROJECT/models/population_reserve_nonotes_full/best_model.pt}
COHORT=${COHORT:-$PROJECT/data/tokenized/mimic_train}

# Wait for enough free GPU memory before starting. The server is shared: a cell once
# died of out-of-memory after 11.7 h because another user had taken most of the
# card. A cell uses about 6 GiB; we wait for NEED_MIB to be free. This cannot stop
# someone else allocating memory after we start.
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

# Never write on top of an old result. A leftover wp5_results.csv looks exactly like
# this run's output, and a progress check once killed a live run because of one.
# Move the old folder aside with a timestamp instead of deleting it.
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

# Check for the output file, not the exit status: the pipe to tee hides the exit
# status.
if [ -s "$OUT/wp5_results.csv" ]; then
  echo "[$(date)] CELL OK: $OUT/wp5_results.csv ($(wc -l < "$OUT/wp5_results.csv") lines)"
else
  echo "[$(date)] CELL FAILED — no wp5_results.csv in $OUT; see $OUT.log"
  exit 1
fi
