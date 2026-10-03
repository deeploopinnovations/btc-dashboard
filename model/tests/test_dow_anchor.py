"""
tests/test_dow_anchor.py
=====================================================================
The serving gate for the day-of-week anchor increment (P4-dow-anchor-result).

THE CONTRACT
  OFF (the shipped default) IS THE SHIPPED ANCHOR, BIT FOR BIT -- against the
  same artifact with the dow arrays stripped, every leaf except `dow_anchor`.
  ON MOVES EXACTLY WHAT THE ALGEBRA SAYS on every weekday's 17:00 night:
      (1 - blend_w) * (inc[0] + fracs(Mon..Sat) . inc[1:])
  through the seed ensemble, to 1e-9; with HOUR_ANCHOR on, the increment
  fitted on the clock-aware anchor's residual is used.
  THE FRACTIONS ARE THE TRUE CALENDAR (pandas), never cal_weekend_frac.
  DOW_ANCHOR AND WEEKEND_ANCHOR ARE ALTERNATIVES: both on is an error.
  ON NEVER MOVES THE GRID; ON WITHOUT THE ARRAYS IS AN ERROR; THE TRAILING
  FACTOR IS COMPUTED WITH THE FLAG ON. AND THE GATE CAN FAIL (R2).

    python model/tests/test_dow_anchor.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua import calendar as CAL                                  # noqa: E402
from noctua.features import build_features                         # noqa: E402
from serve import predict as P                                     # noqa: E402
from serve.history import load_bundle                              # noqa: E402
from serve.runtime import load_model                               # noqa: E402

FROZEN = ("/pct", "/price", "/alpha")
PROD_A, PROD_H = 17, 19
NEW = ("har_beta_dow", "har_beta_dow_season")
NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def flatten(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k in ("dow_anchor", "source"):
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
    m.w = {k: v for k, v in model.w.items() if k not in NEW}
    return m


def pandas_fracs(t: int, H: int) -> np.ndarray:
    hrs = pd.to_datetime(int(t) + 3600 * np.arange(H), unit="s", utc=True)
    return np.array([np.mean(hrs.dayofweek == d) for d in range(7)])


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
    at17 = np.flatnonzero(dts.hour == PROD_A)
    night = {d: int(hts[at17[dts[at17].dayofweek == d][-1]]) for d in range(7)}
    thu = night[3]

    check("artifact-carries-the-arrays", model.has_dow_anchor, " + ".join(NEW))
    check("default-is-off", P.DOW_ANCHOR is False, f"DOW_ANCHOR = {P.DOW_ANCHOR}")

    def run(m, flag, anchor, hour_flag=False, weekend_flag=False):
        o = (P.DOW_ANCHOR, P.HOUR_ANCHOR, P.WEEKEND_ANCHOR)
        P.DOW_ANCHOR, P.HOUR_ANCHOR, P.WEEKEND_ANCHOR = flag, hour_flag, weekend_flag
        try:
            return P.forecast(m, hours, H=PROD_H, anchor_ts=anchor)
        finally:
            P.DOW_ANCHOR, P.HOUR_ANCHOR, P.WEEKEND_ANCHOR = o

    off = run(model, False, thu)
    base = run(stripped(model), False, thu)
    f_off, f_base = flatten(off), flatten(base)
    diff = sorted(k for k in f_off if f_off[k] != f_base.get(k))
    check("off-is-the-shipped-payload", not diff,
          f"differs: {diff}" if diff else f"all {len(f_off)} leaves identical")
    on = run(model, True, thu)
    f_on = flatten(on)
    moved = sorted(k for k in f_off if f_off[k] != f_on.get(k))
    check("on-does-something", len(moved) > 0, f"{len(moved)} leaves move")
    grid = [k for k in moved if any(t in k for t in FROZEN)]
    check("on-never-moves-the-grid", not grid,
          f"moved: {grid}" if grid else "barrier pct/price and alphas held")
    check("payload-reports-the-switch",
          on["dow_anchor"]["enabled"] is True and off["dow_anchor"]["enabled"] is False
          and on["dow_anchor"]["day_fracs_mon_sat"] is not None,
          f"Thursday 17:00 fractions Mon..Sat {on['dow_anchor']['day_fracs_mon_sat']}")
    fac_on = on.get("vol_calibration", {}).get("factor")
    fac_off = off.get("vol_calibration", {}).get("factor")
    check("trailing-factor-sees-the-flag", fac_on is not None and fac_on != fac_off,
          f"factor off {fac_off} -> on {fac_on}")

    def feats(anchor):
        row = int(np.searchsorted(hts, anchor))
        dt = pd.to_datetime(hts[row], unit="s", utc=True)
        ep = pd.DataFrame({"anchor_ts": [hts[row]], "H": [PROD_H], "row": [row],
                           "dt": [dt], "anchor_hour": [dt.hour], "dow": [dt.dayofweek]})
        return build_features(hours, ep)

    Hh = np.array([float(PROD_H)])

    def med(m, anchor, flag, hr=False, w=None):
        m.dow_anchor, m.hour_anchor = flag, hr
        keep = m.w
        if w is not None:
            m.w = w
        try:
            d = m.prepare(feats(anchor), Hh)
            return float(np.log(m.predict(d)["sigma_med"][0])), d
        finally:
            m.w = keep
            m.dow_anchor = m.hour_anchor = False

    wb = 1 - model.blend_w
    inc0, incs = model.w["har_beta_dow"], model.w["har_beta_dow_season"]
    cols = list(CAL.DOW_COLS)
    alg_ok, frac_ok, worst = True, True, 0.0
    for d in range(7):
        lo, _ = med(model, night[d], False)
        ln, d1 = med(model, night[d], True)
        fr = d1["dow_fracs"][0]
        ref = pandas_fracs(night[d], PROD_H)[cols]
        frac_ok &= np.allclose(fr, ref, atol=1e-12)
        want = wb * (inc0[0] + fr @ inc0[1:])
        worst = max(worst, abs((ln - lo) - want))
    check("fractions-are-the-true-calendar-every-night", frac_ok,
          "Mon..Sat window fractions equal pandas' count on all 7 weekdays")
    check("median-moves-by-the-algebra-every-night", worst < 1e-9,
          f"max |moved - algebra| over 7 nights = {worst:.1e}")
    lh, _ = med(model, night[5], False, hr=True)
    lb, d5 = med(model, night[5], True, hr=True)
    want_s = wb * (incs[0] + d5["dow_fracs"][0] @ incs[1:])
    check("with-hour-anchor-uses-the-season-increment", abs((lb - lh) - want_s) < 1e-9,
          f"Saturday: moved {lb - lh:+.6f} on the clock-aware anchor, algebra {want_s:+.6f}")

    w_bad = dict(model.w)
    w_bad["har_beta_dow"] = inc0.copy()
    w_bad["har_beta_dow"][5] += 0.5            # the Friday coefficient
    lo, _ = med(model, thu, False)
    lx, dx = med(model, thu, True, w=w_bad)
    check("gate-can-fail", abs((lx - lo) - wb * (inc0[0] + dx["dow_fracs"][0] @ inc0[1:])) > 1e-6,
          f"Friday coefficient +0.5 moves Thursday night by {lx - lo:+.6f}")

    raised = False
    try:
        run(model, True, thu, weekend_flag=True)
    except RuntimeError:
        raised = True
    check("dow-and-weekend-together-raise", raised, "the week contains the weekend")
    raised = False
    try:
        run(stripped(model), True, thu)
    except RuntimeError:
        raised = True
    check("on-without-arrays-raises", raised, "no silent fallback to the shipped anchor")

    print("day-of-week anchor serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
