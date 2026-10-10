"""
eval/forward_iv1d_anchor.py
=====================================================================
The forward holdout for the next-day implied-vol increment, as frozen in
research/DATA_USE.md ("Sixth candidate: the next-day implied-vol anchor,
frozen 2026-09-29"). The walk-forward evidence is POST HOC
(P4-iv1d-anchor-result REJECT; P4-iv1d-posthoc; P4-iv1d-proxy-control), so this
is the first test the candidate has not shaped.

For every production night (17:00 UTC, H = 19) after the freeze whose window
has closed: the night's next-day ATM IV is rebuilt from Deribit's trade
history with the SAME code serving uses (noctua/iv1d.py), and the served
forecast is computed three ways through serve/predict.forecast, every other
anchor flag off:
  M0s  IV_ANCHOR off (shipped)
  Is   IV_ANCHOR on with the night's own IV
  Ip   IV_ANCHOR on with the IV of another forward night (a seeded
       permutation of the forward nights' IVs): same values, no night --
       the control whose absence sank the walk-forward test

  PRIMARY    PASS iff the per-episode Brier difference is positive with its
             95% block-bootstrap interval above zero for BOTH Is vs M0s AND
             Is vs Ip. One evaluation.
  REPORTED   log score, far-barrier Brier, QLIKE on sigma_mean, and the same
             contrasts on ex-ante classes x < 0 / x >= 0 (x = log hourly IV
             minus the shipped anchor at the night). Nights without an IV are
             scored with Is = M0s (as deployed) and counted.

N_MIN = 200, fixed before any forward night exists: at the walk-forward noise
the Is-vs-Ip Brier gap (+0.00108 per night) needs ~105 nights for its 95%
interval to clear zero; 200 allows for a smaller forward effect.

    python -m model.eval.forward_iv1d_anchor            # live history
    python -m model.eval.forward_iv1d_anchor --offline
    python -m model.eval.forward_iv1d_anchor --selftest
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.forward_hour_anchor import (FAR_PCT, PROD_H, brier_night,     # noqa: E402
                                      forward_nights, qlike)
from eval.forward_weekend_anchor import CAL_FLAGS, check_frozen, logs_night  # noqa: E402

FREEZE = "2026-09-29"
N_MIN = 200
SEED = 29
LOCK = Path(__file__).resolve().parents[1] / "research" / "forward_iv1d_anchor_result.json"
ALPHA = 0.05


def night_ivs(nights: pd.DataFrame, fetch=None) -> np.ndarray:
    from noctua.iv1d import iv_nextday_for
    fetch = fetch or iv_nextday_for
    out = []
    for t in nights["anchor_ts"].to_numpy(np.int64):
        day = pd.Timestamp(int(t), unit="s", tz="UTC").normalize().to_pydatetime()
        # None = no qualifying next-day trade: a real outcome, scored as shipped.
        # An EXCEPTION (timeout, rate limit, bad response) propagates: run()
        # then fails before writing the lock and the daily workflow retries.
        # Swallowing it would score an outage as "no trade" on the one
        # evaluation day and spend the holdout (Codex review, PR #14).
        v = fetch(day.replace(tzinfo=timezone.utc))
        out.append(np.nan if v is None else float(v))
    return np.array(out)


def score(model, hours, nights, ivs) -> list:
    from serve import predict as P
    rng = np.random.default_rng(SEED)
    have = np.flatnonzero(np.isfinite(ivs))
    placebo = ivs.copy()
    placebo[have] = ivs[rng.permutation(have)]
    saved = {f: getattr(P, f) for f in CAL_FLAGS}
    rows = []
    try:
        for f in CAL_FLAGS:
            setattr(P, f, False)
        for i, (_, e) in enumerate(nights.iterrows()):
            out = {}
            for arm, flag, iv in (("M0s", False, None), ("Is", True, ivs[i]), ("Ip", True, placebo[i])):
                P.IV_ANCHOR = flag
                raw: dict = {}
                pay = P.forecast(model, hours, H=PROD_H, anchor_ts=int(e["anchor_ts"]), raw=raw,
                                 iv_pct=None if not np.isfinite(iv if iv is not None else np.nan) else iv,
                                 fetch_iv=False)
                out[arm] = {"mean": float(raw["pred"]["sigma_mean"][0]),
                            "curves": pay["barrier_curves"], "anchor": raw["anchor_logvol"]}
            rows.append((e, out))
    finally:
        for f, v in saved.items():
            setattr(P, f, v)
    return rows


def evaluate(rows, ivs) -> dict:
    from eval.ci import mean_ci
    from noctua.iv1d import log_hourly
    L = max(2 * PROD_H // 24 + 1, int(round(len(rows) ** (1 / 3))))

    def diff(fn, c, o):
        return np.array([fn(out[o], e) - fn(out[c], e) for e, out in rows])

    def ci(d, block=L):
        lo, hi = mean_ci(d, alpha=ALPHA, block_len=block)["ci95"]
        return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)]}

    br = lambda o, e: brier_night(o["curves"], e)                        # noqa: E731
    res = {"n_nights": len(rows), "n_with_iv": int(np.isfinite(ivs).sum()), "block_len": L}
    for c, o in (("Is", "M0s"), ("Is", "Ip")):
        res[f"brier_{c}_vs_{o}"] = ci(diff(br, c, o))
        res[f"logs_{c}_vs_{o}"] = ci(diff(lambda o_, e: logs_night(o_["curves"], e), c, o))
        res[f"brier_far_{c}_vs_{o}"] = ci(diff(lambda o_, e: brier_night(o_["curves"], e, FAR_PCT), c, o))
        rv = np.array([e["RV"] for e, _ in rows])
        qo = qlike(rv, np.array([out[o]["mean"] for _, out in rows]))
        qc = qlike(rv, np.array([out[c]["mean"] for _, out in rows]))
        res[f"qlike_{c}_vs_{o}"] = ci(qo - qc)
    x = np.array([log_hourly(v) - out["M0s"]["anchor"] if np.isfinite(v) else np.nan
                  for v, (_, out) in zip(ivs, rows)])
    cls = {}
    for name, m in (("x<0", x < 0), ("x>=0", x >= 0)):
        cell = {"n": int(m.sum())}
        if m.sum() >= 10:
            for c, o in (("Is", "M0s"), ("Is", "Ip")):
                cell[f"brier_{c}_vs_{o}"] = ci(diff(br, c, o)[m], 1)
        cls[name] = cell
    res["exante_x_classes"] = cls
    res["passed"] = bool(res["brier_Is_vs_M0s"]["ci95"][0] > 0 and res["brier_Is_vs_Ip"]["ci95"][0] > 0)
    return res


def run(hours, model, lock: Path = LOCK, n_min: int = N_MIN, freeze: str = FREEZE,
        fetch=None) -> dict:
    if lock.exists():
        res = json.loads(lock.read_text())
        print(f"LOCKED: scored once on {res['scored_on']}; printing that result.")
        print(json.dumps(res, indent=2))
        return res
    nights = forward_nights(hours, freeze)
    if len(nights) < n_min:
        print(f"{len(nights)} forward production nights after {freeze} with closed windows; "
              f"the holdout is scored once at {n_min}. Nothing else is printed before then.")
        return {"n_nights": len(nights), "scored": False}
    check_frozen(model, "IV_ANCHOR")
    ivs = night_ivs(nights, fetch)
    res = evaluate(score(model, hours, nights, ivs), ivs)
    res.update(scored=True, freeze=freeze, n_min=n_min,
               scored_on=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    lock.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"wrote {lock} -- this holdout is now spent (R26)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="forward holdout, next-day implied-vol anchor")
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
    """Mechanics on ALREADY-SEEN data before the freeze, a synthetic IV feed
    (no network), a temporary lock."""
    from serve import predict as P
    from serve.history import load_bundle
    from serve.runtime import load_model
    ok = []
    hours, model = load_bundle(), load_model()
    last = pd.to_datetime(hours["hour_ts"].iloc[-1], unit="s", utc=True)
    fake = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    feed = lambda day: 40.0 + (day.day % 7) * 5.0                            # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        lock = Path(td) / "lock.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r0 = run(hours, model, lock=lock, n_min=10_000, freeze=fake, fetch=feed)
        ok.append(("below N_MIN: count only", r0["scored"] is False and "brier" not in buf.getvalue().lower()))
        # Codex review, PR #14: a fetch ERROR on the one scoring day must not be
        # scored as a no-trade night and locked -- it must abort, lock unwritten,
        # so the daily workflow retries. A genuine no-trade night (None) is scored.
        lock_err = Path(td) / "lock_err.json"

        def flaky(day):
            if day.day % 3 == 0:
                raise TimeoutError("Deribit timeout (planted)")
            return feed(day)
        raised = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run(hours, model, lock=lock_err, n_min=3, freeze=fake, fetch=flaky)
        except Exception:
            raised = True
        ok.append(("a fetch error aborts the scoring and writes no lock", raised and not lock_err.exists()))
        lock_none = Path(td) / "lock_none.json"
        with contextlib.redirect_stdout(io.StringIO()):
            rn = run(hours, model, lock=lock_none, n_min=3, freeze=fake,
                     fetch=lambda day: None if day.day % 2 else feed(day))
        ok.append(("a no-trade night is scored as shipped and counted",
                   lock_none.exists() and 0 < rn["n_with_iv"] < rn["n_nights"]))
        with contextlib.redirect_stdout(io.StringIO()):
            r1 = run(hours, model, lock=lock, n_min=3, freeze=fake, fetch=feed)
        ok.append(("at N_MIN: scored, locked, both contrasts present",
                   r1.get("scored") is True and lock.exists()
                   and "brier_Is_vs_Ip" in r1 and "brier_Is_vs_M0s" in r1))
        ok.append(("the IV moves the forecast (Is differs from M0s)",
                   r1["brier_Is_vs_M0s"]["diff"] != 0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            r2 = run(hours, model, lock=lock, n_min=3, freeze=fake, fetch=feed)
        ok.append(("second run returns the locked result", r2 == json.loads(lock.read_text())))
        ok.append(("flags restored", all(getattr(P, f) is False for f in CAL_FLAGS)))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
