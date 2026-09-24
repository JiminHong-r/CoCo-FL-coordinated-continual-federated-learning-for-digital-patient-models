#!/bin/bash
# Wait for a running process to finish, then run cells one after another on one GPU,
# retrying failed cells once at the end.
#
# bash scripts/v9_queue2.sh <gpu> <wait_pid> <tag:split:task:alpha[:VAR=val,...]>
# ...
#
# The GPU is shared, so the most likely failure is an out-of-memory error caused by
# someone else's job. That is usually fine a few hours later, so each failed cell
# gets one retry with a smaller eval batch. The eval batch changes only memory use
# and speed, not predictions, so a retried cell stays comparable.
set -u
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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

# Retry pass at a smaller eval batch.
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
