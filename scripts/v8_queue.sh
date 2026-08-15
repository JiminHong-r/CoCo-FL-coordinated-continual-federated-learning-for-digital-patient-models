#!/bin/bash
# Sequential cell queue pinned to one GPU.
#
#   bash scripts/v8_queue.sh 0 classaware:hospital_mortality:1.0 random:hospital_mortality:0.5
#
# Each argument is a cell spec "split:task:alpha", run one after another on the
# given GPU via run_wp5_cell.sh. A failing cell does NOT stop the queue — a cell
# is independent, and losing the whole queue to one bad cell is the failure mode
# the per-cell split exists to avoid.
set -u

PROJECT=/home/jimin_hong/coco-fl
GPU=${1:?usage: v8_queue.sh <gpu> <split:task:alpha> ...}
shift
CODE_SRC=${CODE_SRC:-$PROJECT/src_v8}

run_cell () {
  local cell="$1" split task alpha
  IFS=: read -r split task alpha <<< "$cell"
  echo "[$(date)] --- queue gpu=$GPU starting $split/$task/α=$alpha ---"
  GPU="$GPU" SPLIT="$split" TASK="$task" ALPHA="$alpha" CODE_SRC="$CODE_SRC" \
    bash "$PROJECT/scripts/run_wp5_cell.sh"
}

echo "[$(date)] QUEUE gpu=$GPU cells=$*"
ok=0; failed=()
for cell in "$@"; do
  if run_cell "$cell"; then ok=$((ok + 1)); else
    failed+=("$cell")
    echo "[$(date)] !! cell FAILED: $cell (continuing queue, will retry at the end)"
  fi
done

# One retry pass. A cell most often fails for a reason that is TRANSIENT on a
# shared box — another user holding the card — so a cell dropped on the first
# attempt is usually recoverable an hour later. Without this, a failed cell just
# vanished from the campaign and had to be noticed and re-queued by hand.
if [ ${#failed[@]} -gt 0 ]; then
  echo "[$(date)] QUEUE gpu=$GPU retry pass for: ${failed[*]}"
  still=()
  for cell in "${failed[@]}"; do
    if run_cell "$cell"; then ok=$((ok + 1)); else still+=("$cell"); fi
  done
  failed=("${still[@]+"${still[@]}"}")
fi

echo "[$(date)] QUEUE gpu=$GPU DONE — $ok ok, ${#failed[@]} still failing${failed[*]+: ${failed[*]}}"
