"""
eval/forward_hour_anchor.py
=====================================================================
The forward holdout for the clock-aware anchor, as frozen in
research/DATA_USE.md ("Second candidate: the clock-aware anchor, frozen
2026-09-27").

For every production night (17:00 UTC anchor, H = 19) anchored STRICTLY AFTER
the freeze whose window has closed, the SERVED forecast is computed twice from
the same artifact -- HOUR_ANCHOR off and on -- through serve/predict.forecast
itself (unrounded via its `raw` hook, so nothing here re-implements serving),
and scored against the outcome built by noctua/episodes.build_episodes, the
same labels every walk-forward result used.

  PRIMARY    per-episode QLIKE difference on sigma_mean (clock-blind minus
             clock-aware, positive = the candidate is better), moving-block
             bootstrap, one evaluation.
  SECONDARY  median log(RV / sigma_med) under each -- the mechanism claim is
             that the candidate moves it toward zero.
  REPORTED   per-episode Brier of the served barrier curves, labelled
             underpowered; and, AMENDED 2026-09-28 BEFORE ANY FORWARD NIGHT
             EXISTED (P4-hour-anchor-exante-result, R89): the all-night
             far-barrier Brier (the +/-5% rungs, the benchmark's far barrier),
             and Brier / far-barrier Brier on nights flagged EX ANTE as HOT
             (top decile of har_6h - har_22d) or HIGH (top decile of har_1d).
             No outcome-selected subset is ever scored.

ONE EVALUATION, AND NO PEEKING. Below N_MIN nights the script prints the COUNT
and nothing else -- no metric, no sign. At or above N_MIN it scores once and
writes the result to LOCK; every later run prints the locked result and refuses
to rescore. N_MIN = 300 is fixed here before any forward night exists: at the
walk-forward noise level (per-episode QLIKE half-width 0.0047 at 2,046 nights,
99.2%), a 95% interval narrows below the walk-forward effect (0.0094) at about
290 nights.

    python -m model.eval.forward_hour_anchor            # live history (network)
    python -m model.eval.forward_hour_anchor --offline  # committed bundle only
    python -m model.eval.forward_hour_anchor --selftest
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

FREEZE = "2026-09-27"            # research/DATA_USE.md; anchors strictly after
N_MIN = 300
PROD_H, PROD_A = 19, 17
LOCK = Path(__file__).resolve().parents[1] / "research" / "forward_hour_anchor_result.json"
ALPHA = 0.05


def forward_nights(hours: pd.DataFrame, freeze: str = FREEZE) -> pd.DataFrame:
    from noctua.episodes import build_episodes
    ep = build_episodes(hours, horizons=[PROD_H])
    t0 = int(pd.Timestamp(freeze, tz="UTC").timestamp())
    keep = (ep["anchor_hour"] == PROD_A) & (ep["anchor_ts"] > t0)
    return ep[keep].reset_index(drop=True)


def score(model, hours: pd.DataFrame, nights: pd.DataFrame) -> dict:
    from serve import predict as P
    rows = []
    orig = P.HOUR_ANCHOR
    try:
        for _, e in nights.iterrows():
            out = {}
            for flag in (False, True):
                P.HOUR_ANCHOR = flag
                raw: dict = {}
                pay = P.forecast(model, hours, H=PROD_H,
                                 anchor_ts=int(e["anchor_ts"]), raw=raw)
                pred = raw["pred"]
                out[flag] = {"mean": float(pred["sigma_mean"][0]),
                             "med": float(pred["sigma_med"][0]),
                             "curves": pay["barrier_curves"]}
            rows.append((e, out))
    finally:
        P.HOUR_ANCHOR = orig
    return rows


def qlike(rv, s):
    r = rv ** 2 / s ** 2
    return r - np.log(r) - 1.0


FAR_PCT = 5.0


def brier_night(curves: dict, e, only_pct: float | None = None) -> float:
    errs = []
    for side, M in (("up", e["M_up"]), ("dn", -e["M_dn"])):
        for rung in curves[side]:
            if only_pct is not None and abs(abs(rung["pct"]) - only_pct) > 1e-9:
                continue
            u = np.log1p(abs(rung["pct"]) / 100.0)
            errs.append((rung["touch_prob"] - float(M >= u)) ** 2)
    return float(np.mean(errs))


def exante_flags(hours, nights) -> dict:
    """HOT / HIGH flags from features at each anchor (known in advance)."""
    from noctua.features import build_features
    X = build_features(hours, nights)
    shock = X["har_6h"].to_numpy(np.float64) - X["har_22d"].to_numpy(np.float64)
    lvl = X["har_1d"].to_numpy(np.float64)
    return {"hot": shock >= np.nanquantile(shock, 0.9),
            "high": lvl >= np.nanquantile(lvl, 0.9)}


def evaluate(rows, exante: dict | None = None) -> dict:
    from eval.direction import mean_ci
    rv = np.array([e["RV"] for e, _ in rows])
    q = {f: qlike(rv, np.array([o[f]["mean"] for _, o in rows])) for f in (False, True)}
    d = q[False] - q[True]
    L = max(2 * PROD_H // 24 + 1, int(round(len(d) ** (1 / 3))))
    lo, hi = mean_ci(d, alpha=ALPHA, block_len=L)["ci95"]
    bias = {f: float(np.median(np.log(rv / np.array([o[f]["med"] for _, o in rows]))))
            for f in (False, True)}
    b = {f: np.array([brier_night(o[f]["curves"], e) for e, o in rows]) for f in (False, True)}
    db = b[False] - b[True]
    blo, bhi = mean_ci(db, alpha=ALPHA, block_len=L)["ci95"]
    bf = {f: np.array([brier_night(o[f]["curves"], e, FAR_PCT) for e, o in rows])
          for f in (False, True)}
    dbf = bf[False] - bf[True]
    flo, fhi = mean_ci(dbf, alpha=ALPHA, block_len=L)["ci95"]
    # ex-ante subsets, from features known at each anchor (R89)
    sub = {}
    if exante is not None:
        for name, msk in (("HOT", exante["hot"]), ("HIGH", exante["high"])):
            cell = {"n": int(msk.sum())}
            for key, arr in (("brier", db), ("brier_far", dbf)):
                if msk.sum() >= 10:
                    lo_, hi_ = mean_ci(arr[msk], alpha=ALPHA, block_len=1)["ci95"]
                    cell[key] = {"diff": float(arr[msk].mean()), "ci95": [float(lo_), float(hi_)]}
            sub[name] = cell
    return {"n_nights": len(rows), "block_len": L,
            "qlike_clock_blind": float(q[False].mean()),
            "qlike_clock_aware": float(q[True].mean()),
            "qlike_diff": float(d.mean()), "qlike_ci95": [float(lo), float(hi)],
            "qlike_pct": float(100 * d.mean() / q[False].mean()),
            "median_log_rv_over_sigma_med": {"clock_blind": bias[False],
                                             "clock_aware": bias[True]},
            "brier_diff": float(db.mean()), "brier_ci95": [float(blo), float(bhi)],
            "brier_far_diff": float(dbf.mean()), "brier_far_ci95": [float(flo), float(fhi)],
            "exante_subsets": sub,
            "brier_note": "reported, underpowered at any horizon under a year"}


def run(hours, model, lock: Path = LOCK, n_min: int = N_MIN,
        freeze: str = FREEZE) -> dict:
    if lock.exists():
        res = json.loads(lock.read_text())
        print(f"LOCKED: the holdout was scored once on {res['scored_on']}; "
              f"printing that result and refusing to rescore.")
        print(json.dumps(res, indent=2))
        return res
    nights = forward_nights(hours, freeze)
    n = len(nights)
    if n < n_min:
        print(f"{n} forward production nights after {freeze} with closed "
              f"windows; the holdout is scored once at {n_min}. Nothing else "
              f"is printed before then.")
        return {"n_nights": n, "scored": False}
    res = evaluate(score(model, hours, nights), exante_flags(hours, nights))
    res.update(scored=True, freeze=freeze, n_min=n_min,
               scored_on=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    lock.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"wrote {lock} -- this holdout is now spent (R26)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward holdout, clock-aware anchor")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    from serve.runtime import load_model
    from serve.history import get_hours, load_bundle
    if a.offline:
        hours = load_bundle()
    else:
        from serve.fetch import fetch_bars
        hours, _ = get_hours(fetch_bars)
    run(hours, load_model())
    return 0


def selftest() -> int:
    """Mechanics only, on ALREADY-SEEN data before the freeze (never the
    holdout): a fake freeze inside the committed bundle, a tiny N_MIN, a
    temporary lock. Prints pass/fail, not the numbers."""
    import contextlib
    import io
    from serve.history import load_bundle
    from serve.runtime import load_model
    ok = []
    hours, model = load_bundle(), load_model()
    last = pd.to_datetime(hours["hour_ts"].iloc[-1], unit="s", utc=True)
    fake = (last - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    with tempfile.TemporaryDirectory() as td:
        lock = Path(td) / "lock.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r0 = run(hours, model, lock=lock, n_min=10_000, freeze=fake)
        ok.append(("below N_MIN: count only, no metric printed",
                   r0["scored"] is False and "qlike" not in buf.getvalue().lower()))
        with contextlib.redirect_stdout(io.StringIO()):
            r1 = run(hours, model, lock=lock, n_min=3, freeze=fake)
        ok.append(("at N_MIN: scored and locked", r1.get("scored") is True
                   and lock.exists() and r1["n_nights"] >= 3))
        ok.append(("both arms scored on the same nights, finite",
                   np.isfinite(r1["qlike_diff"]) and np.isfinite(r1["brier_diff"])
                   and np.isfinite(r1["brier_far_diff"])))
        ok.append(("ex-ante subsets present, no outcome-selected subset",
                   set(r1["exante_subsets"]) == {"HOT", "HIGH"}
                   and "spike" not in json.dumps(r1).lower()))
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = run(hours, model, lock=lock, n_min=3, freeze=fake)
        ok.append(("second run returns the locked result, does not rescore",
                   r2 == json.loads(lock.read_text())))
        nights = forward_nights(hours, fake)
        ok.append(("only 17:00 anchors strictly after the freeze",
                   bool((nights["anchor_hour"] == PROD_A).all()) and
                   int(nights["anchor_ts"].min()) >
                   int(pd.Timestamp(fake, tz="UTC").timestamp())))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
