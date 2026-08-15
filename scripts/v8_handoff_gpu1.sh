#!/bin/bash
# GPU 1 handoff: the first v8 class-aware job runs α ∈ {0.5, 1.0, 10.0} inside a
# single process, but α=1.0 and α=10.0 are now assigned to the GPU 0 and GPU 2
# cell queues. Left alone it would recompute both — ~12 GPU-hours of duplicate
# work. So once its α=0.5 is banked on disk, stop it and put GPU 1 on the WP4
# ablations, which are the priority-2 block and have to be redone from scratch
# (all eleven arms; five have old-code CSVs, one crashed, two never ran, three
# never started).
#
# Same safety rule as v8_handoff.sh: kill only after the α=0.5 rows are verified
# present in wp5_results.csv. Parent bash is killed BEFORE its python child, or
# the loop simply starts the next thing — the mistake that cost 80 minutes of
# contention earlier today.
set -u

PROJECT=/home/jimin_hong/coco-fl
POLL=${POLL:-180}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-480}
CELL_DIR=$PROJECT/results_wp5_classaware_hospital_mortality
PY_PID=${PY_PID:-1529718}
SH_PID=${SH_PID:-1529714}

alive () { kill -0 "$1" 2>/dev/null; }
banked () {
  local csv="$CELL_DIR/wp5_results.csv"
  [ -s "$csv" ] || return 1
  awk -F, 'NR>1 && $2 == "0.5" {f=1} END {exit !f}' "$csv"
}

echo "[$(date)] gpu1 handoff waiting on α=0.5 in $CELL_DIR"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while :; do
  if banked; then echo "[$(date)] α=0.5 banked"; break; fi
  if ! alive "$PY_PID"; then
    echo "[$(date)] job already gone; proceeding (α=0.5 may be incomplete — check the CSV)"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[$(date)] !! timed out after ${MAX_WAIT_MIN}min — NOT killing, exiting"
    exit 1
  fi
  echo "[$(date)] not yet"
  sleep "$POLL"
done

echo "[$(date)] stopping the α-sweep job (parent shell first)"
alive "$SH_PID" && { kill "$SH_PID"; echo "  TERM -> parent $SH_PID"; }
sleep 2
alive "$PY_PID" && { kill "$PY_PID"; echo "  TERM -> python $PY_PID"; }
sleep 15
for p in "$SH_PID" "$PY_PID"; do alive "$p" && { kill -9 "$p"; echo "  KILL -> $p"; }; done
sleep 5
for p in "$SH_PID" "$PY_PID"; do alive "$p" && echo "  !! $p STILL ALIVE" || echo "  $p stopped"; done

# Rename so the dir says what it actually holds: one α, produced by v8 code.
if [ -d "$CELL_DIR" ]; then
  mv "$CELL_DIR" "$PROJECT/results_v8_classaware_hospital_mortality_a0.5" \
    && echo "  -> results_v8_classaware_hospital_mortality_a0.5"
fi

echo "[$(date)] GPU 1 -> WP4 ablations, all 11 arms at α=0.5, hierarchical only"
cd "$PROJECT" || exit 1
setsid nohup env GPU=1 CODE_SRC=$PROJECT/src_v8 TASK=hospital_mortality \
  TAG=v8abl ALPHAS=0.5 bash scripts/run_wp4_ablations.sh \
  > "$PROJECT/v8_ablations_gpu1.log" 2>&1 &
echo "  ablation queue pid $!"
echo "[$(date)] gpu1 handoff complete"
