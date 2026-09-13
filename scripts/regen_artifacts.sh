#!/usr/bin/env bash
# Regenerate the three artifacts report.py refuses to build without.
#
# SEQUENTIALLY, and that is the whole point. Running vol_matrix,
# direction_bench and econ_voltarget concurrently killed all three on
# 2026-09-13: each holds the full 476,359 x 42 episode table plus torch, and
# three at once exceeded the 15 GB box. The tell was that none left a
# traceback, all three truncated mid-output, and memory was fully reclaimed --
# a process that dies without a traceback was KILLED, not crashed, and the
# difference matters because a crash is a bug in your code and a kill is a bug
# in your scheduling.
#
# CPU load was 0.58 when they were launched. Load is not the constraint here;
# resident memory is. Check the one that binds.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="${1:-/tmp/regen}"
mkdir -p "$LOG"

run() {
  local name="$1" mod="$2" out="$3"
  if [ -f "$out" ]; then
    echo "[skip] $name -- $out already exists"
    return 0
  fi
  echo "[run ] $name  -> $out"
  local t0=$SECONDS
  if OMP_NUM_THREADS=2 python -m "$mod" > "$LOG/$name.log" 2>&1; then
    echo "[ok  ] $name  ($((SECONDS-t0))s)"
  else
    echo "[FAIL] $name  ($((SECONDS-t0))s) -- tail:"
    tail -15 "$LOG/$name.log" | sed 's/^/       /'
    return 1
  fi
  free -g | sed -n '2p' | sed 's/^/       mem: /'
}

run direction_bench model.eval.direction_bench model/artifacts/direction_bench.json
run econ_voltarget  model.eval.econ_voltarget  model/artifacts/econ_voltarget.json
run vol_matrix      model.eval.vol_matrix      model/artifacts/vol_matrix.json
echo
echo "all three present; report.py can now rebuild RESEARCH_REPORT.md"
