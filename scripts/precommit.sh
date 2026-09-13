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

# F821 FIRST, and it is not decoration. On 2026-09-13 ruff found two live
# NameErrors in the eval suite: `direction.block_bootstrap_ci` referenced a
# `block_len` that was added to a NEIGHBOURING function's signature and not its
# own -- broken for 106 commits across 14 call sites -- and
# `anchor_freshness._verdict` read a `spike` mask off `main`'s scope. Neither
# was caught by a single test, because neither code path is reached by one: the
# first sits behind modules nothing had rerun, the second behind a guard the
# stored-fold path never satisfies. Unreached code is exactly what a test suite
# cannot audit and a static check can, so the static check runs unconditionally
# and runs first -- it costs under a second and it is the only gate here that
# looks at code nothing executes.
run "ruff F821 (undefined names)" ruff check --no-cache --select F821 model/ scripts/

echo "precommit gates (the same ones CI runs)"
run "ledger --validate"      python -m model.research.ledger --validate
run "test_serving"           python model/tests/test_serving.py
run "test_history"           python model/tests/test_history.py
run "test_adaptive"          python model/tests/test_adaptive.py
run "test_features"          python model/tests/test_features.py
run "test_selfimprove"       python model/tests/test_selfimprove.py
run "test_level_report"      python model/tests/test_level_report.py
# These two were in CI and NOT here, so the header's claim to run "the same
# ones CI runs" was aspirational. Both check FILES written by scripts rather
# than a write path, which is the version of the check that binds.
run "pitfalls --self-test"   python -m model.research.pitfalls --self-test
run "teacher_ledger"         python -m model.research.teacher_ledger --validate

if [ "$fail" -ne 0 ]; then
  echo
  echo "REFUSING: fix the above before committing. CI runs these too, and a"
  echo "pushed failure costs a round trip plus a wake-up."
  exit 1
fi
echo
echo "all gates pass"
