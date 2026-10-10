"""
eval/forward_clock_dow_anchor.py
=====================================================================
The forward holdout for the COMBINATION that would be served if both anchor
candidates were switched on -- the clock-aware anchor (HOUR_ANCHOR) and the
day-of-week increment fitted on its residual (DOW_ANCHOR) -- against the
shipped anchor, as frozen in research/DATA_USE.md ("Fifth: the clock-aware +
day-of-week combination, frozen 2026-09-29"). Mechanics shared with
eval/forward_weekend_anchor.py: each night is forecast with BOTH flags off and
BOTH on through serve/predict.forecast.

  PRIMARY    per-episode Brier of the served barrier curves, one evaluation.
  REPORTED   log score, far-barrier Brier, QLIKE, ex-ante weekday classes.

N_MIN = 300, fixed now: the two walk-forward Brier effects are +0.000698
(clock-aware, P4-hour-anchor-pe-result) and +0.000444 (day-of-week), measured
as additive on the no-network law (audit E, P4-weekend-fix); even at half the
sum the walk-forward noise (95% half-width ~0.00014 at 2,046 nights) is
cleared at ~150 nights, and 300 matches the clock-aware holdout.

    python -m model.eval.forward_clock_dow_anchor            # live history
    python -m model.eval.forward_clock_dow_anchor --offline
    python -m model.eval.forward_clock_dow_anchor --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.forward_weekend_anchor import CAL_FLAGS, run as _run         # noqa: E402

FREEZE = "2026-09-29"
N_MIN = 300
FLAG = ("HOUR_ANCHOR", "DOW_ANCHOR")
LOCK = Path(__file__).resolve().parents[1] / "research" / "forward_clock_dow_anchor_result.json"


def run(hours, model, lock: Path = LOCK, n_min: int = N_MIN, freeze: str = FREEZE) -> dict:
    return _run(hours, model, lock=lock, n_min=n_min, freeze=freeze, flag=FLAG,
                member="forward_clock_dow_anchor")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward holdout, clock-aware + day-of-week anchor")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    from serve.history import get_hours, load_bundle
    from serve.runtime import load_model
    if a.offline:
        hours = load_bundle()
    else:
        from serve.fetch import fetch_bars
        hours, _ = get_hours(fetch_bars)
    run(hours, load_model())
    return 0


def selftest() -> int:
    """Mechanics on ALREADY-SEEN data before the freeze, temporary lock."""
    from serve import predict as P
    from serve.history import load_bundle
    from serve.runtime import load_model
    ok = []
    hours, model = load_bundle(), load_model()
    last = pd.to_datetime(hours["hour_ts"].iloc[-1], unit="s", utc=True)
    fake = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    with tempfile.TemporaryDirectory() as td:
        lock = Path(td) / "lock.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r0 = run(hours, model, lock=lock, n_min=10_000, freeze=fake)
        ok.append(("below N_MIN: count only", r0["scored"] is False
                   and "brier" not in buf.getvalue().lower()))
        with contextlib.redirect_stdout(io.StringIO()):
            r1 = run(hours, model, lock=lock, n_min=3, freeze=fake)
        ok.append(("at N_MIN: scored, locked, arms differ", r1.get("scored") is True
                   and lock.exists() and r1["primary_brier"]["diff"] != 0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = run(hours, model, lock=lock, n_min=3, freeze=fake)
        ok.append(("second run returns the locked result", r2 == json.loads(lock.read_text())))
        from eval.forward_family import lock_problems
        probs = lock_problems(r1, "forward_clock_dow_anchor", {"brier": r1["primary_brier"]})
        ok.append((f"lock keeps per-night deltas and the family interval {probs or ''}", not probs))
        ok.append(("flags restored", all(getattr(P, f) is False for f in CAL_FLAGS)))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
