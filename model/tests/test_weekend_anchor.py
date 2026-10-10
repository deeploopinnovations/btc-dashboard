"""
tests/test_weekend_anchor.py
=====================================================================
The serving gate for the true-weekend anchor increment (P4-weekend-fix-result).

THE CONTRACT

  OFF (the shipped default) IS THE SHIPPED ANCHOR, BIT FOR BIT
    With WEEKEND_ANCHOR off, an artifact that CARRIES the increment publishes
    exactly the payload of the same artifact with the arrays stripped -- every
    leaf except the `weekend_anchor` block.

  ON MOVES EXACTLY WHAT THE ALGEBRA SAYS, THROUGH THE SEED ENSEMBLE
    At a production anchor (17:00 UTC, H = 19) the served median moves by
        (1 - blend_w) * (inc[0] + inc[1] * weekend_frac)
    in log space, to 1e-9 -- on a Saturday night (fraction 1) and a Monday
    night (fraction 0). With HOUR_ANCHOR on too, the increment fitted on the
    CLOCK-AWARE anchor's residual is the one used.

  THE FRACTION IS THE TRUE CALENDAR
    Serving recovers it from the anchor's weekday/hour features; it must equal
    noctua/calendar.weekend_frac and an independent pandas count -- never the
    Fri+Sat cal_weekend_frac (P4-weekend-bug).

  ON NEVER MOVES THE GRID; ON WITHOUT THE ARRAYS IS AN ERROR; THE TRAILING
  FACTOR IS COMPUTED WITH THE FLAG ON (from the same anchor it corrects).

  AND THE GATE CAN FAIL (R2): a perturbed coefficient must break the algebra.

    python model/tests/test_weekend_anchor.py
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
NEW = ("har_beta_weekend", "har_beta_weekend_season")


def flatten(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k in ("weekend_anchor", "source"):
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


def pandas_frac(t: int, H: int) -> float:
    hrs = pd.to_datetime(int(t) + 3600 * np.arange(H), unit="s", utc=True)
    return float(np.isin(hrs.dayofweek, (5, 6)).mean())


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
    sat = int(hts[at17[dts[at17].dayofweek == 5][-1]])
    mon = int(hts[at17[dts[at17].dayofweek == 0][-1]])

    check("artifact-carries-the-arrays", model.has_weekend_anchor, " + ".join(NEW))
    check("default-is-off", P.WEEKEND_ANCHOR is False,
          f"WEEKEND_ANCHOR = {P.WEEKEND_ANCHOR}")

    def run(m, flag, anchor, hour_flag=False):
        o1, o2 = P.WEEKEND_ANCHOR, P.HOUR_ANCHOR
        P.WEEKEND_ANCHOR, P.HOUR_ANCHOR = flag, hour_flag
        try:
            return P.forecast(m, hours, H=PROD_H, anchor_ts=anchor)
        finally:
            P.WEEKEND_ANCHOR, P.HOUR_ANCHOR = o1, o2

    off = run(model, False, sat)
    base = run(stripped(model), False, sat)
    f_off, f_base = flatten(off), flatten(base)
    diff = sorted(k for k in f_off if f_off[k] != f_base.get(k))
    check("off-is-the-shipped-payload", not diff,
          f"differs: {diff}" if diff else f"all {len(f_off)} leaves identical")

    on = run(model, True, sat)
    f_on = flatten(on)
    moved = sorted(k for k in f_off if f_off[k] != f_on.get(k))
    check("on-does-something", len(moved) > 0, f"{len(moved)} leaves move")
    grid = [k for k in moved if any(t in k for t in FROZEN)]
    check("on-never-moves-the-grid", not grid,
          f"moved: {grid}" if grid else "barrier pct/price and alphas held")
    check("payload-reports-the-switch",
          on["weekend_anchor"]["enabled"] is True
          and off["weekend_anchor"]["enabled"] is False
          and on["weekend_anchor"]["weekend_frac"] == 1.0,
          f"Saturday 17:00 weekend_frac {on['weekend_anchor']['weekend_frac']}, "
          f"coef {on['weekend_anchor']['weekend_coef']}")
    fac_on = on.get("vol_calibration", {}).get("factor")
    fac_off = off.get("vol_calibration", {}).get("factor")
    check("trailing-factor-sees-the-flag", fac_on is not None and fac_on != fac_off,
          f"factor off {fac_off} -> on {fac_on}")

    # ---- the algebra, at the model level, through the seed ensemble -------
    def feats(anchor):
        row = int(np.searchsorted(hts, anchor))
        dt = pd.to_datetime(hts[row], unit="s", utc=True)
        ep = pd.DataFrame({"anchor_ts": [hts[row]], "H": [PROD_H], "row": [row],
                           "dt": [dt], "anchor_hour": [dt.hour],
                           "dow": [dt.dayofweek]})
        return build_features(hours, ep)

    Hh = np.array([float(PROD_H)])

    def med(m, anchor, wk, hr=False, w=None):
        m.weekend_anchor, m.hour_anchor = wk, hr
        keep = m.w
        if w is not None:
            m.w = w
        try:
            d = m.prepare(feats(anchor), Hh)
            return float(np.log(m.predict(d)["sigma_med"][0])), d
        finally:
            m.w = keep
            m.weekend_anchor = m.hour_anchor = False

    wb = 1 - model.blend_w
    inc0, incs = model.w["har_beta_weekend"], model.w["har_beta_weekend_season"]
    for name, anchor, want_wf in (("saturday", sat, 1.0), ("monday", mon, 0.0)):
        lo, _ = med(model, anchor, False)
        ln, d1 = med(model, anchor, True)
        wf = float(d1["weekend_frac"][0])
        check(f"{name}-fraction-is-the-true-calendar",
              wf == want_wf == pandas_frac(anchor, PROD_H)
              == float(CAL.weekend_frac(np.array([anchor]), np.array([PROD_H]))[0]),
              f"serving {wf}, calendar module and pandas agree")
        want = wb * (inc0[0] + inc0[1] * wf)
        check(f"{name}-median-moves-by-the-algebra", abs((ln - lo) - want) < 1e-9,
              f"log sigma_med moved {ln - lo:+.6f}, algebra {want:+.6f}")

    lh, _ = med(model, sat, False, hr=True)
    lb, _ = med(model, sat, True, hr=True)
    want_s = wb * (incs[0] + incs[1] * 1.0)
    check("with-hour-anchor-uses-the-season-increment", abs((lb - lh) - want_s) < 1e-9,
          f"moved {lb - lh:+.6f} on top of the clock-aware anchor, algebra {want_s:+.6f}")

    w_bad = dict(model.w)
    w_bad["har_beta_weekend"] = inc0.copy()
    w_bad["har_beta_weekend"][1] += 0.5
    lo, _ = med(model, sat, False)
    lx, _ = med(model, sat, True, w=w_bad)
    check("gate-can-fail", abs((lx - lo) - wb * (inc0[0] + inc0[1])) > 1e-6,
          f"coefficient +0.5 moves the median by {lx - lo:+.6f}")

    raised = False
    try:
        run(stripped(model), True, sat)
    except RuntimeError:
        raised = True
    check("on-without-arrays-raises", raised, "no silent fallback to the shipped anchor")

    # ---- the calendar, against pandas, and its two forms --------------------
    rng = np.random.default_rng(5)
    t = (1_450_000_000 + rng.integers(0, 3 * 10**8, 400)) // 3600 * 3600
    H = rng.choice([1, 6, 12, 19, 24], len(t))
    ref = np.array([pandas_frac(a, h) for a, h in zip(t, H)])
    dt = pd.to_datetime(t, unit="s", utc=True)
    clk = CAL.weekend_frac_from_clock(dt.dayofweek.to_numpy(), dt.hour.to_numpy(), H)
    check("calendar-agrees-with-pandas",
          np.allclose(CAL.weekend_frac(t, H), ref, atol=1e-12)
          and np.allclose(clk, ref, atol=1e-12),
          "timestamp and clock forms, 400 random windows")
    dd = np.arange(7)
    check("weekday-recovered-from-cal-features",
          np.array_equal(CAL.dow_from_cal(np.sin(2 * np.pi * dd / 7),
                                          np.cos(2 * np.pi * dd / 7)), dd),
          "all 7 weekdays round-trip")

    print("weekend-anchor serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
