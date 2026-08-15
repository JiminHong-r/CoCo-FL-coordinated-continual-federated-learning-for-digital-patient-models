#!/bin/bash
# Hand GPUs 0 and 2 over from the old-code A/B runs to the v8 cell queues, the
# moment the A/B runs have banked their α=0.5 results and not one second before.
#
# WHY THIS IS SCRIPTED RATHER THAN DONE BY HAND
# ---------------------------------------------
# The α=0.5 A/B pair is the only internally-consistent evidence currently on
# disk (both arms share one config), so it is the fallback if the v8 re-run
# stalls. It must not be killed early. But it is also on superseded code, so
# every minute past α=0.5 is wasted GPU. The window is narrow and lands at an
# unpredictable hour, so waiting for it is scripted.
#
# THE SAFETY RULE: a run is killed only after its α=0.5 rows are verified
# PRESENT ON DISK in wp5_results.csv. The log header for α=1.0 is treated as a
# hint to go looking, never as proof — the CSV is the artefact, and checking the
# log alone is how a dead task previously looked finished.
set -u

PROJECT=/home/jimin_hong/coco-fl
POLL=${POLL:-180}
MAX_WAIT_MIN=${MAX_WAIT_MIN:-300}

banked () {   # $1 = results dir, $2 = pid whose output it must be
  local csv="$1/wp5_results.csv"
  [ -s "$csv" ] || return 1
  # The CSV must be NEWER than the process that is supposed to have written it.
  # Without this the check happily matches a stale CSV left in a reused output
  # dir and reports the run finished before it started — which is exactly how a
  # live α=0.5 run got killed 1.5 h in earlier today, against a June-25 file.
  local etimes start mtime
  etimes=$(ps -o etimes= -p "$2" 2>/dev/null | tr -d ' ')
  if [ -n "$etimes" ]; then
    start=$(( $(date +%s) - etimes ))
    mtime=$(stat -c %Y "$csv" 2>/dev/null || echo 0)
    [ "$mtime" -ge "$start" ] || {
      echo "    (ignoring $csv: older than pid $2 — stale from a previous run)"
      return 1
    }
  fi
  awk -F, 'NR>1 && $2 == "0.5" {found=1} END {exit !found}' "$csv"
}

alive () { kill -0 "$1" 2>/dev/null; }

RAND_PID=${RAND_PID:-1239347}
CA_PID=${CA_PID:-1239586}
RAND_DIR=$PROJECT/results_final_random
CA_DIR=$PROJECT/results_final_classaware

echo "[$(date)] handoff waiting: random(pid $RAND_PID) classaware(pid $CA_PID)"
deadline=$(( $(date +%s) + MAX_WAIT_MIN * 60 ))
while :; do
  r_ok=0; c_ok=0
  banked "$RAND_DIR" "$RAND_PID" && r_ok=1
  banked "$CA_DIR"   "$CA_PID"   && c_ok=1
  alive "$RAND_PID" || r_ok=1     # already gone: nothing left to wait for
  alive "$CA_PID"   || c_ok=1
  echo "[$(date)] banked? random=$r_ok classaware=$c_ok"
  [ "$r_ok" = 1 ] && [ "$c_ok" = 1 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[$(date)] !! timed out after ${MAX_WAIT_MIN}min — NOT killing anything, exiting"
    exit 1
  fi
  sleep "$POLL"
done

echo "[$(date)] α=0.5 banked on both. Stopping the superseded A/B runs."
for p in "$RAND_PID" "$CA_PID"; do
  alive "$p" && { kill "$p"; echo "  TERM -> $p"; }
done
sleep 15
for p in "$RAND_PID" "$CA_PID"; do
  alive "$p" && { kill -9 "$p"; echo "  KILL -> $p"; }
done
sleep 5
for p in "$RAND_PID" "$CA_PID"; do
  alive "$p" && echo "  !! $p STILL ALIVE" || echo "  $p stopped"
done

# Park the old-code outputs under a name that says so, so no later merge can
# silently mix them with v8 cells.
for d in "$RAND_DIR" "$CA_DIR"; do
  [ -d "$d" ] && mv "$d" "${d}.OLDCODE_alpha0.5only" && echo "  parked $(basename "$d")"
done

echo "[$(date)] launching v8 queues on the freed GPUs"
cd "$PROJECT" || exit 1
# Ordering rationale: mortality is a partly VACUOUS experiment. At 2 % prevalence
# a 2,000-window validation set holds ~42 positives, too noisy to detect a real
# improvement, so selection falls back to round 0 — at α≥1.0 it did so for every
# method, and those cells measure nothing. readmission (~20 %) and icu (~16 %)
# carry ~10× the positive signal, resolve model selection properly (the dead
# readmission run selected rounds 4/4/3/1, never 0), and cost less per method.
# So the α=0.5 A/B pair comes first, then the extra α values, and the near-IID
# α=10.0 control comes last.
setsid nohup bash scripts/v8_queue.sh 0 \
  classaware:hospital_mortality:1.0 \
  random:hospital_mortality:0.5 \
  > "$PROJECT/v8_queue_gpu0.log" 2>&1 &
echo "  gpu0 queue pid $!"
setsid nohup bash scripts/v8_queue.sh 2 \
  random:readmission:0.5 \
  random:hospital_mortality:1.0 \
  classaware:hospital_mortality:10.0 \
  > "$PROJECT/v8_queue_gpu2.log" 2>&1 &
echo "  gpu2 queue pid $!"
echo "[$(date)] handoff complete"
