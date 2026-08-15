#!/bin/bash
# When the GPU-1 cell finishes, put GPU 1 on the WP4 ablations.
#
# Deliberately dumb: it waits for a pid to EXIT and then starts something. It
# kills nothing and inspects no result files, so it has no way to misfire on
# stale output — unlike the two handoffs that did exactly that earlier today
# (one matched a June-25 CSV and killed a live run 1.5 h in).
#
#   WAIT_PID=<pid of the cell> bash scripts/v8_then_ablations.sh
set -u

PROJECT=/home/jimin_hong/coco-fl
WAIT_PID=${WAIT_PID:?set WAIT_PID}
POLL=${POLL:-120}

echo "[$(date)] waiting for pid $WAIT_PID (GPU 1 cell) to finish"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep "$POLL"; done
echo "[$(date)] pid $WAIT_PID gone — GPU 1 free"

# Give the driver a moment to release memory before the next job grabs it.
sleep 30

cd "$PROJECT" || exit 1
echo "[$(date)] GPU 1 -> WP4 ablations: 11 arms, α=0.5, hierarchical only, v8 code"
exec env GPU=1 CODE_SRC=$PROJECT/src_v8 TASK=hospital_mortality SPLIT=${SPLIT:-classaware} \
  TAG=v8abl ALPHAS=0.5 bash scripts/run_wp4_ablations.sh
