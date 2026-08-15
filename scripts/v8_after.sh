#!/bin/bash
# Wait for a pid to exit, then run a cell queue on a GPU.
#
#   WAIT_PID=1610240 GPU=1 bash scripts/v8_after.sh classaware:readmission:0.5 ...
#
# Deliberately dumb, like v8_then_ablations.sh: it waits on process liveness and
# then launches. It kills nothing and reads no result files, so it cannot
# misfire on stale output the way the result-inspecting handoffs did.
set -u

PROJECT=/home/jimin_hong/coco-fl
WAIT_PID=${WAIT_PID:?set WAIT_PID}
GPU=${GPU:?set GPU}
POLL=${POLL:-120}
[ "$#" -gt 0 ] || { echo "no cells given"; exit 2; }

echo "[$(date)] gpu$GPU queue waiting for pid $WAIT_PID; then: $*"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep "$POLL"; done
echo "[$(date)] pid $WAIT_PID gone — starting queue on gpu$GPU"
sleep 30   # let the driver release memory

cd "$PROJECT" || exit 1
exec bash scripts/v8_queue.sh "$GPU" "$@"
