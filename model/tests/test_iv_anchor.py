"""
tests/test_iv_anchor.py
=====================================================================
The serving gate for the next-day implied-vol increment (P4-iv1d-*). Never
touches the network: the implied vol is passed in, and the fetch is disabled.

THE CONTRACT
  OFF (the shipped default) IS THE SHIPPED ANCHOR, BIT FOR BIT -- against the
  artifact with har_beta_iv stripped, every leaf except `iv_anchor`.
  ON, AT 17:00 / H=19, WITH AN IV, the served median moves by
      (1 - blend_w) * (a + b * (log hourly IV - anchor))
  through the seed ensemble, to 1e-9.
  ON ELSEWHERE, OR WITHOUT AN IV, NOTHING MOVES, and the payload says why.
  ON WITH ANY OTHER ANCHOR INCREMENT IS AN ERROR (it is fitted on the shipped
  anchor's residual). ON WITHOUT THE ARRAY IS AN ERROR.
  THE GRID NEVER MOVES. AND THE GATE CAN FAIL (R2).

    python model/tests/test_iv_anchor.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua.features import build_features                         # noqa: E402
from noctua.iv1d import log_hourly                                 # noqa: E402
from serve import predict as P                                     # noqa: E402
from serve.history import load_bundle                              # noqa: E402
from serve.runtime import load_model                               # noqa: E402

FROZEN = ("/pct", "/price", "/alpha")
PROD_A, PROD_H = 17, 19
IV = 48.0                    # a plausible next-day ATM IV, percent


def flatten(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k in ("iv_anchor", "source"):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, f"{pre}{k}/"))
        elif isinstance(v, list):
            for i, r in enumerate(v):
                if isinstance(r, dict):
                    out.update(flatten(r, f"{pre}{k}/{i}/"))
                else:
                    out[f"{pre}{k}/{i}"] = r
        else:
            out[f"{pre}{k}"] = v
    return out


def stripped(model):
    m = copy.copy(model)
    m.w = {k: v for k, v in model.w.items() if k != "har_beta_iv"}
    return m


def main() -> int:
    ok = []

    def check(name, good, detail=""):
        ok.append((name, bool(good), detail))

    model = load_model()
    hours = load_bundle()
    if hours is None or len(hours) < 24 * 400:
        print("no committed history bundle; cannot run the serving gate")
        return 1
    hts = hours["hour_ts"].to_numpy(np.int64)
    dts = pd.to_datetime(hts, unit="s", utc=True)
    a17 = int(hts[np.flatnonzero(dts.hour == PROD_A)[-1]])
    a09 = int(hts[np.flatnonzero(dts.hour == 9)[-1]])

    check("artifact-carries-the-array", model.has_iv_anchor, "har_beta_iv")
    check("default-is-off", P.IV_ANCHOR is False, f"IV_ANCHOR = {P.IV_ANCHOR}")

    def run(m, flag, anchor, iv=IV, **other):
        saved = {f: getattr(P, f) for f in ("IV_ANCHOR", "HOUR_ANCHOR", "WEEKEND_ANCHOR", "DOW_ANCHOR")}
        P.IV_ANCHOR = flag
        for f, v in other.items():
            setattr(P, f, v)
        try:
            return P.forecast(m, hours, H=PROD_H, anchor_ts=anchor, iv_pct=iv, fetch_iv=False)
        finally:
            for f, v in saved.items():
                setattr(P, f, v)

    off = run(model, False, a17)
    base = run(stripped(model), False, a17)
    f_off, f_base = flatten(off), flatten(base)
    diff = sorted(k for k in f_off if f_off[k] != f_base.get(k))
    check("off-is-the-shipped-payload", not diff,
          f"differs: {diff}" if diff else f"all {len(f_off)} leaves identical")
    on = run(model, True, a17)
    f_on = flatten(on)
    moved = sorted(k for k in f_off if f_off[k] != f_on.get(k))
    check("on-at-17-does-something", len(moved) > 0 and on["iv_anchor"]["applied"] is True,
          f"{len(moved)} leaves move; iv_pct {on['iv_anchor']['iv_pct']}")
    grid = [k for k in moved if any(t in k for t in FROZEN)]
    check("on-never-moves-the-grid", not grid,
          f"moved: {grid}" if grid else "barrier pct/price and alphas held")
    at9 = run(model, True, a09)
    f9, f9off = flatten(at9), flatten(run(model, False, a09))
    check("on-at-another-hour-moves-nothing", f9 == f9off and at9["iv_anchor"]["applied"] is False,
          at9["iv_anchor"]["reason"])
    noiv = run(model, True, a17, iv=None)
    check("on-without-iv-moves-nothing", flatten(noiv) == f_off and noiv["iv_anchor"]["applied"] is False,
          noiv["iv_anchor"]["reason"])

    # ---- the algebra, through the seed ensemble -----------------------------
    row = int(np.searchsorted(hts, a17))
    dt = pd.to_datetime(hts[row], unit="s", utc=True)
    X = build_features(hours, pd.DataFrame({"anchor_ts": [hts[row]], "H": [PROD_H], "row": [row],
                                            "dt": [dt], "anchor_hour": [dt.hour], "dow": [dt.dayofweek]}))
    Hh = np.array([float(PROD_H)])

    def med(flag, w=None):
        model.iv_anchor = flag
        keep = model.w
        if w is not None:
            model.w = w
        try:
            d = model.prepare(X, Hh, iv_log_hourly=[log_hourly(IV)] if flag else None)
            return float(np.log(model.predict(d)["sigma_med"][0])), d
        finally:
            model.w = keep
            model.iv_anchor = False

    lo, d0 = med(False)
    ln, _ = med(True)
    hb, (ca, cb) = model.w["har_beta"], model.w["har_beta_iv"]
    anc = hb[0] + d0["Xb"][0] @ hb[1:]
    want = (1 - model.blend_w) * (ca + cb * (log_hourly(IV) - anc))
    check("median-moves-by-the-algebra", abs((ln - lo) - want) < 1e-9,
          f"log sigma_med moved {ln - lo:+.6f}, algebra {want:+.6f}")
    w_bad = dict(model.w)
    w_bad["har_beta_iv"] = np.array([ca, cb + 0.5])
    lx, _ = med(True, w_bad)
    check("gate-can-fail", abs((lx - lo) - want) > 1e-6,
          f"b + 0.5 moves the median by {lx - lo:+.6f}, not {want:+.6f}")

    for other in ("HOUR_ANCHOR", "DOW_ANCHOR", "WEEKEND_ANCHOR"):
        raised = False
        try:
            run(model, True, a17, **{other: True})
        except RuntimeError:
            raised = True
        check(f"with-{other.lower()}-raises", raised, "fitted on the shipped anchor only")
    raised = False
    try:
        run(stripped(model), True, a17)
    except RuntimeError:
        raised = True
    check("on-without-array-raises", raised, "no silent fallback")

    print("iv-anchor serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
