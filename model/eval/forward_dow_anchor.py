"""
eval/forward_dow_anchor.py
=====================================================================
The forward holdout for the day-of-week anchor increment, as frozen in
research/DATA_USE.md ("Fourth candidate: the day-of-week anchor, frozen
2026-09-29"). Same mechanics as eval/forward_weekend_anchor.py (shared code):
every production night after the freeze is forecast with DOW_ANCHOR off and on
through serve/predict.forecast, every other anchor flag off.

  PRIMARY    per-episode Brier of the served barrier curves (shipped minus
             candidate), block bootstrap, one evaluation. Walk-forward effect
             +0.000444 per night.
  REPORTED   log score, far-barrier Brier, QLIKE, Brier on ex-ante weekday
             classes. No outcome-selected subset.

N_MIN = 450, fixed before any forward night exists. At the walk-forward noise
Brier needs ~200 nights, but the increment carries coefficients frozen on the
artifact's split, which kept 0.62-0.82 of a yearly refit's gain
(eval/dow_staleness.py); at 0.67 of the effect the requirement is ~450.

    python -m model.eval.forward_dow_anchor            # live history
    python -m model.eval.forward_dow_anchor --offline
    python -m model.eval.forward_dow_anchor --selftest
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
N_MIN = 450
FLAG = "DOW_ANCHOR"
LOCK = Path(__file__).resolve().parents[1] / "research" / "forward_dow_anchor_result.json"


def run(hours, model, lock: Path = LOCK, n_min: int = N_MIN, freeze: str = FREEZE) -> dict:
    return _run(hours, model, lock=lock, n_min=n_min, freeze=freeze, flag=FLAG)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward holdout, day-of-week anchor")
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
        ok.append(("flags restored", all(getattr(P, f) is False for f in CAL_FLAGS)))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
