#!/usr/bin/env bash
# Push the code tree to the GPU server and verify it landed intact.
#
# WHY THIS EXISTS
# ---------------
# Two ways an ad-hoc `rsync -az --delete` has already cost real GPU hours:
#
# 1. It copied `__pycache__`. Python embeds the SOURCE PATH in a .pyc and
#    validates it against the .py's mtime+size — both of which rsync -a
#    preserves. The server silently executed bytecode compiled on a laptop, and
#    tracebacks pointed at /Users/... paths that do not exist there. Identical
#    bytecode, so nothing was miscomputed; the hazard is the NEXT sync, where a
#    .py can change without its size/mtime advancing past the cached .pyc and
#    Python then reuses stale bytecode. On a 15-hour cell that is an expensive
#    way to discover the server is not running the code you think it is.
#
# 2. It would clobber RUNNING shell scripts. bash reads a script incrementally
#    as it executes, so rewriting `run_wp5_cell.sh` underneath a live cell makes
#    it resume at a byte offset into different text. That has already killed one
#    run. `scripts/` is therefore opt-in, never deleted, and refused outright
#    while any cell is executing.
#
# src/ and tests/ are always safe: Python reads them at import, so a running
# process is unaffected by a later write.
#
#   scripts/sync_to_server.sh                 # src + tests, verified
#   scripts/sync_to_server.sh --tests         # ... and run the suite remotely
#   scripts/sync_to_server.sh --scripts       # ... also scripts/ (refuses if busy)
set -euo pipefail

HOST=${HOST:-coco-fl-server}
REMOTE=${REMOTE:-/home/jimin_hong/coco-fl}
LOCAL=$(cd "$(dirname "$0")/.." && pwd)

WITH_TESTS=0; WITH_SCRIPTS=0
for a in "$@"; do
  case "$a" in
    --tests) WITH_TESTS=1 ;;
    --scripts) WITH_SCRIPTS=1 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache')

echo "==> syncing $LOCAL -> $HOST:$REMOTE"
for d in src tests; do
  rsync -az --delete "${EXCLUDES[@]}" "$LOCAL/$d/" "$HOST:$REMOTE/$d/"
  echo "    $d/"
done

if [ "$WITH_SCRIPTS" = 1 ]; then
  running=$(ssh "$HOST" 'pgrep -cf cocofl_evaluate || true')
  if [ "${running:-0}" -gt 0 ]; then
    echo "    REFUSED: $running cell(s) executing — bash reads scripts" >&2
    echo "    incrementally, so overwriting one mid-run corrupts it." >&2
    exit 1
  fi
  # No --delete: the server legitimately holds runners this tree does not.
  rsync -az "${EXCLUDES[@]}" "$LOCAL/scripts/" "$HOST:$REMOTE/scripts/"
  echo "    scripts/ (no --delete)"
fi

# A checksum comparison, not a "rsync exited 0" — the question being answered is
# "is the server running this exact tree?", which an exit code does not settle.
echo "==> verifying src/ is byte-identical"
# LC_ALL=C on BOTH sides is load-bearing: macOS and Linux `sort` disagree on the
# order of the same filenames under different locales, so a digest-of-digests
# built from an unsorted-consistently list reports a mismatch on identical
# content. A verifier that cries wolf is worse than none — it trains you to
# ignore it on the one run where the trees really have diverged.
_SUM="LC_ALL=C find . -name '*.py' -type f | LC_ALL=C sort | xargs shasum | shasum | cut -d' ' -f1"
sum_local=$(cd "$LOCAL/src" && eval "$_SUM")
sum_remote=$(ssh "$HOST" "cd $REMOTE/src && $_SUM")
if [ "$sum_local" != "$sum_remote" ]; then
  echo "    MISMATCH: local $sum_local != remote $sum_remote" >&2
  echo "    per-file diff:" >&2
  diff <(cd "$LOCAL/src" && LC_ALL=C find . -name '*.py' -type f | LC_ALL=C sort | xargs shasum) \
       <(ssh "$HOST" "cd $REMOTE/src && LC_ALL=C find . -name '*.py' -type f | LC_ALL=C sort | xargs shasum") >&2 || true
  exit 1
fi
echo "    match ($sum_local, $(printf '%s\n' "$sum_local" >/dev/null; cd "$LOCAL/src" && LC_ALL=C find . -name '*.py' -type f | wc -l | tr -d ' ') files)"

# Which tree are the RUNNING cells importing? `v8_queue.sh` defaults CODE_SRC to
# $PROJECT/src_v8, a second copy of the package — so a verified sync to src/ can
# leave every live cell executing different code, silently. Syncing is not the
# same as taking effect; say which tree is actually loaded.
echo "==> code tree in use by running cells"
ssh "$HOST" '
  running=0
  for p in $(pgrep -f cocofl_evaluate 2>/dev/null); do
    # NOTE: this block is single-quoted for ssh, so it must contain no
    # apostrophes — one would close the quote and the rest would run locally.
    # The redirect is evaluated by the shell before tr runs, so a pid that exits
    # mid-loop errors even with 2>/dev/null on tr. Guard on the file instead.
    [ -r /proc/$p/environ ] || continue
    pp=$(cat /proc/$p/environ 2>/dev/null | tr "\0" "\n" | sed -n "s/^PYTHONPATH=//p")
    [ -n "$pp" ] && { echo "    pid $p -> $pp"; running=1; }
  done
  [ "$running" = 0 ] && echo "    (no cells running)"
  for alt in '"$REMOTE"'/src_*; do
    [ -d "$alt" ] || continue
    # Archived trees are meant to differ; warning about them trains you to
    # ignore the warning, which defeats it.
    case "$alt" in *.bak-*|*.retired-*) continue ;; esac
    if diff -rq --exclude=__pycache__ '"$REMOTE"'/src "$alt" >/dev/null 2>&1; then
      echo "    $alt: identical to src/"
    else
      echo "    $alt: DIVERGED from src/ — cells using it run different code"
    fi
  done'

if [ "$WITH_TESTS" = 1 ]; then
  echo "==> remote test suite"
  ssh "$HOST" "cd $REMOTE && PATH=$REMOTE/venv/bin:\$PATH PYTHONPATH=$REMOTE/src \
    python -m pytest tests/ -q 2>&1 | tail -3"
fi
