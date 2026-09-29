"""
eval/forward_weekend_anchor.py
=====================================================================
The forward holdout for the true-weekend anchor increment, as frozen in
research/DATA_USE.md ("Third candidate: the true-weekend anchor, frozen
2026-09-29").

For every production night (17:00 UTC anchor, H = 19) anchored STRICTLY AFTER
the freeze whose window has closed, the SERVED forecast is computed twice from
the same artifact -- WEEKEND_ANCHOR off and on, HOUR_ANCHOR off (the served
base at the freeze) -- through serve/predict.forecast itself (its `raw` hook),
and scored against noctua/episodes.build_episodes' outcome.

  PRIMARY    per-episode Brier of the served barrier curves (off minus on,
             positive = the candidate is better), moving-block bootstrap, one
             evaluation. The walk-forward result it tests: +0.000301 per night.
  REPORTED   per-episode log score, pinball-free far-barrier Brier (+/-5%),
             QLIKE on sigma_mean (labelled underpowered: ~1,400 nights needed),
             and Brier by EX-ANTE weekday class (Fri/Sat/Sun anchors vs the
             rest) -- the walk-forward said the gain is on weekend nights and
             Mon-Thu pay a little. No outcome-selected subset is scored (R89).

ONE EVALUATION, AND NO PEEKING. Below N_MIN nights the script prints the COUNT
and nothing else. At or above N_MIN it scores once and writes LOCK; later runs
print the locked result. N_MIN = 450, fixed before any forward night exists:
at the walk-forward noise (per-night Brier delta, 95% half-width 0.000137 at
2,046 nights, block 38) the interval narrows below the effect at ~425 nights.

    python -m model.eval.forward_weekend_anchor            # live history
    python -m model.eval.forward_weekend_anchor --offline
    python -m model.eval.forward_weekend_anchor --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.forward_hour_anchor import (FAR_PCT, PROD_A, PROD_H,          # noqa: E402
                                      brier_night, forward_nights, qlike)

FREEZE = "2026-09-29"            # research/DATA_USE.md; anchors strictly after
N_MIN = 450
LOCK = Path(__file__).resolve().parents[1] / "research" / "forward_weekend_anchor_result.json"
ALPHA = 0.05


def logs_night(curves: dict, e) -> float:
    errs = []
    for side, M in (("up", e["M_up"]), ("dn", -e["M_dn"])):
        for rung in curves[side]:
            u = np.log1p(abs(rung["pct"]) / 100.0)
            o = float(M >= u)
            p = float(np.clip(rung["touch_prob"], 1e-6, 1 - 1e-6))
            errs.append(-(o * np.log(p) + (1 - o) * np.log(1 - p)))
    return float(np.mean(errs))


CAL_FLAGS = ("HOUR_ANCHOR", "WEEKEND_ANCHOR", "DOW_ANCHOR")

# The arrays each candidate was FROZEN with (research/DATA_USE.md): first 16
# hex of sha256 over the array bytes. A holdout scored against different
# arrays would test a different candidate under the frozen one's name, so
# scoring REFUSES on any mismatch (counting does not need the check).
FROZEN_SHA16 = {
    "HOUR_ANCHOR": {"season_profile": "836813fb13ec4769",
                    "har_beta_season": "27daa2be75a28cb6"},
    "WEEKEND_ANCHOR": {"har_beta_weekend": "03ca1419345447ef",
                       "har_beta_weekend_season": "4b59300962e77815"},
    "DOW_ANCHOR": {"har_beta_dow": "a10cfe4816204035",
                   "har_beta_dow_season": "e1053f4c9aff7f38"},
}


def check_frozen(model, flags) -> None:
    import hashlib
    flags = (flags,) if isinstance(flags, str) else tuple(flags)
    w = getattr(model, "w", {})
    for fl in flags:
        for name, want in FROZEN_SHA16[fl].items():
            got = (hashlib.sha256(np.asarray(w[name]).tobytes()).hexdigest()[:16]
                   if name in w else "missing")
            if got != want:
                raise SystemExit(f"REFUSING to score: {name} is {got}, frozen as {want} "
                                 f"(research/DATA_USE.md). This is not the frozen candidate.")


def score(model, hours: pd.DataFrame, nights: pd.DataFrame, flag="WEEKEND_ANCHOR"):
    """Forecast each night with `flag` off and on, every OTHER anchor flag off
    (the served base at the freeze). `flag` may be a tuple of flags switched
    together. Shared by forward_dow_anchor.py and forward_clock_dow_anchor.py."""
    flags = (flag,) if isinstance(flag, str) else tuple(flag)
    from serve import predict as P
    rows = []
    saved = {f: getattr(P, f) for f in CAL_FLAGS}
    try:
        for f in CAL_FLAGS:
            setattr(P, f, False)
        for _, e in nights.iterrows():
            out = {}
            for on in (False, True):
                for fl in flags:
                    setattr(P, fl, on)
                raw: dict = {}
                pay = P.forecast(model, hours, H=PROD_H,
                                 anchor_ts=int(e["anchor_ts"]), raw=raw)
                out[on] = {"mean": float(raw["pred"]["sigma_mean"][0]),
                           "curves": pay["barrier_curves"]}
            rows.append((e, out))
    finally:
        for f, v in saved.items():
            setattr(P, f, v)
    return rows


def evaluate(rows) -> dict:
    from eval.ci import mean_ci
    L = max(2 * PROD_H // 24 + 1, int(round(len(rows) ** (1 / 3))))

    def diff(fn):
        return np.array([fn(o[False], e) - fn(o[True], e) for e, o in rows])

    def ci(d, block=L):
        lo, hi = mean_ci(d, alpha=ALPHA, block_len=block)["ci95"]
        return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)]}

    b = diff(lambda o, e: brier_night(o["curves"], e))
    lg = diff(lambda o, e: logs_night(o["curves"], e))
    bf = diff(lambda o, e: brier_night(o["curves"], e, FAR_PCT))
    rv = np.array([e["RV"] for e, _ in rows])
    q = {f: qlike(rv, np.array([o[f]["mean"] for _, o in rows])) for f in (False, True)}
    dow = np.array([pd.Timestamp(int(e["anchor_ts"]), unit="s", tz="UTC").dayofweek
                    for e, _ in rows])
    classes = {}
    for name, msk in (("Fri/Sat/Sun", dow >= 4), ("Mon-Thu", dow <= 3)):
        classes[name] = {"n": int(msk.sum()),
                         **({"brier": ci(b[msk], 1)} if msk.sum() >= 10 else {})}
    return {"n_nights": len(rows), "block_len": L,
            "primary_brier": ci(b),
            "logs": ci(lg), "brier_far": ci(bf),
            "qlike": {**ci(q[False] - q[True]),
                      "note": "underpowered: ~1,400 nights needed"},
            "exante_weekday_classes": classes}


def run(hours, model, lock: Path = LOCK, n_min: int = N_MIN,
        freeze: str = FREEZE, flag="WEEKEND_ANCHOR") -> dict:
    if lock.exists():
        res = json.loads(lock.read_text())
        print(f"LOCKED: scored once on {res['scored_on']}; printing that result "
              f"and refusing to rescore.")
        print(json.dumps(res, indent=2))
        return res
    nights = forward_nights(hours, freeze)
    n = len(nights)
    if n < n_min:
        print(f"{n} forward production nights after {freeze} with closed windows; "
              f"the holdout is scored once at {n_min}. Nothing else is printed "
              f"before then.")
        return {"n_nights": n, "scored": False}
    check_frozen(model, flag)
    res = evaluate(score(model, hours, nights, flag))
    res.update(scored=True, freeze=freeze, n_min=n_min,
               scored_on=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    lock.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"wrote {lock} -- this holdout is now spent (R26)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward holdout, true-weekend anchor")
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
    """Mechanics only, on ALREADY-SEEN data before the freeze: a fake freeze
    inside the committed bundle, a tiny N_MIN, a temporary lock."""
    import contextlib
    import io
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
        ok.append(("below N_MIN: count only, no metric printed",
                   r0["scored"] is False and "brier" not in buf.getvalue().lower()))
        with contextlib.redirect_stdout(io.StringIO()):
            r1 = run(hours, model, lock=lock, n_min=3, freeze=fake)
        ok.append(("at N_MIN: scored and locked",
                   r1.get("scored") is True and lock.exists() and r1["n_nights"] >= 3))
        ok.append(("both arms scored on the same nights, finite, and they DIFFER",
                   np.isfinite(r1["primary_brier"]["diff"])
                   and r1["primary_brier"]["diff"] != 0.0))
        ok.append(("ex-ante weekday classes only, no outcome-selected subset",
                   set(r1["exante_weekday_classes"]) == {"Fri/Sat/Sun", "Mon-Thu"}
                   and "spike" not in json.dumps(r1).lower()))
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = run(hours, model, lock=lock, n_min=3, freeze=fake)
        ok.append(("second run returns the locked result", r2 == json.loads(lock.read_text())))
        import copy
        bad = copy.copy(model)
        bad.w = dict(model.w)
        bad.w["har_beta_weekend"] = model.w["har_beta_weekend"] + 1e-9
        refused = False
        try:
            check_frozen(bad, "WEEKEND_ANCHOR")
        except SystemExit:
            refused = True
        ok.append(("scoring refuses a tampered frozen array", refused))
        check_frozen(model, CAL_FLAGS)
        from serve import predict as P
        ok.append(("flags restored after scoring",
                   all(getattr(P, f) is False for f in CAL_FLAGS)))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
