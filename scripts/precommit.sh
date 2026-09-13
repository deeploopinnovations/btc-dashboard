#!/usr/bin/env bash
# Gate a commit on the checks CI will run anyway.
#
# WHY THIS EXISTS
#
# On 2026-09-13 a commit was pushed that failed CI on `ledger --validate`, and
# the validator had ALREADY reported the problem locally in the same command
# that did the push. The cause was not carelessness, it was a shell detail:
#
#     python -m model.research.ledger --validate 2>&1 | tail -1 && git commit ...
#
# A pipeline reports the exit status of its LAST command, so `tail` returning 0
# masked the validator returning 1 and the `&&` happily proceeded. Piping a
# gate's output through `tail` to keep it short silently disarms the gate.
#
# This script sets `pipefail`, so that cannot happen, and runs the same checks
# CI runs. Use it instead of chaining a validator into a commit by hand.
set -euo pipefail

cd "$(dirname "$0")/.."
fail=0

run() {
  local name="$1"; shift
  if "$@" >/tmp/precommit.$$ 2>&1; then
    printf '  [ok  ] %s\n' "$name"
  else
    printf '  [FAIL] %s\n' "$name"
    tail -20 /tmp/precommit.$$ | sed 's/^/         /'
    fail=1
  fi
  rm -f /tmp/precommit.$$
}

echo "precommit gates (the same ones CI runs)"
run "ledger --validate"      python -m model.research.ledger --validate
run "test_serving"           python model/tests/test_serving.py
run "test_history"           python model/tests/test_history.py
run "test_adaptive"          python model/tests/test_adaptive.py
run "test_features"          python model/tests/test_features.py
run "test_selfimprove"       python model/tests/test_selfimprove.py
run "test_level_report"      python model/tests/test_level_report.py

if [ "$fail" -ne 0 ]; then
  echo
  echo "REFUSING: fix the above before committing. CI runs these too, and a"
  echo "pushed failure costs a round trip plus a wake-up."
  exit 1
fi
echo
echo "all gates pass"
