#!/bin/bash
# Wait for a running pid, then run cells on one GPU — with a retry pass.
#
#   bash scripts/v9_queue2.sh <gpu> <wait_pid> <tag:split:task:alpha[:VAR=val,...]> ...
#
# WHY THIS EXISTS
# ---------------------------------------
# The GPU is shared. `class-aware readmission α=10.0` ran for four hours and died
# with a CUDA OOM when another user's job expanded onto the card — and because the
# previous runner had no retry, the queue simply moved on and that cell went
# missing from the grid. A transient failure caused by someone else's allocation is
# the most likely failure mode here and the most recoverable, so it gets one retry
# at the end rather than silently costing a cell.
#
# The pre-flight VRAM gate in run_wp5_cell.sh only checks memory at START; nothing
# can protect a 19-hour cell from a mid-run spike. Lowering EVAL_BATCH_SIZE for the
# retry reduces peak footprint. Eval batching changes memory and speed only, never
# per-window predictions, so a retried cell stays comparable to the rest.
set -u
PROJECT=/home/jimin_hong/coco-fl
GPU=${1:?gpu}; WAITPID=${2:?pid to wait for}; shift 2
CODE_SRC=${CODE_SRC:-$PROJECT/src}

if [ "$WAITPID" != "0" ]; then
  echo "[$(date)] gpu=$GPU waiting for pid $WAITPID to exit..."
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
  echo "[$(date)] gpu=$GPU pid $WAITPID gone — starting"
  sleep 30
fi

run_one () {
  local spec="$1" extra_env="${2:-}"
  local tag split task alpha extra
  IFS=: read -r tag split task alpha extra <<< "$spec"
  echo "[$(date)] --- gpu=$GPU $tag $split/$task/a=$alpha ${extra:-} ${extra_env:-} ---"
  local env_args=()
  if [ -n "${extra:-}" ]; then
    IFS=, read -ra kvs <<< "$extra"
    for kv in "${kvs[@]}"; do env_args+=("$kv"); done
  fi
  if [ -n "$extra_env" ]; then
    IFS=, read -ra kvs2 <<< "$extra_env"
    for kv in "${kvs2[@]}"; do env_args+=("$kv"); done
  fi
  env "${env_args[@]+"${env_args[@]}"}" \
      GPU="$GPU" SPLIT="$split" TASK="$task" ALPHA="$alpha" \
      TAG="$tag" CODE_SRC="$CODE_SRC" \
      bash "$PROJECT/scripts/run_wp5_cell.sh"
}

failed=()
for spec in "$@"; do
  if run_one "$spec"; then
    echo "[$(date)] OK $spec"
  else
    echo "[$(date)] !! FAILED $spec (will retry at the end)"
    failed+=("$spec")
  fi
done

# One retry pass, at a smaller eval batch. A cell that lost to a transient
# allocation is usually recoverable hours later on a quieter card.
if [ ${#failed[@]} -gt 0 ]; then
  echo "[$(date)] gpu=$GPU RETRY PASS for: ${failed[*]}"
  for spec in "${failed[@]}"; do
    if run_one "$spec" "EVAL_BATCH_SIZE=8"; then
      echo "[$(date)] OK on retry $spec"
    else
      echo "[$(date)] !! STILL FAILING $spec"
    fi
  done
fi
echo "[$(date)] gpu=$GPU queue done"
