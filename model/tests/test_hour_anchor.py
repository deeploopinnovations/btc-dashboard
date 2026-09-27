"""
tests/test_hour_anchor.py
=====================================================================
The serving gate for the clock-aware anchor (P4-hour-anchor-result).

THE CONTRACT

  OFF (the shipped default) IS THE CLOCK-BLIND MODEL, BIT FOR BIT
    With HOUR_ANCHOR off, an artifact that CARRIES the new arrays publishes
    exactly the payload of the same artifact with the arrays stripped -- every
    leaf except the `hour_anchor` block, which reports the arrays' presence.
    Adding the arrays to the shipped file is therefore not a product change.

  ON MOVES EXACTLY WHAT THE ALGEBRA SAYS, THROUGH THE SEED ENSEMBLE
    At the production anchor (17:00 UTC, H = 19), the served median moves by
        (1 - blend_w) * (har_beta_season . [1, Xb, season_fwd] - har_beta . [1, Xb])
    in log space, to 1e-9, via NoctuaV2.predict's per-seed loop -- which
    KeyErrors if the seed scope drops the arrays. season_fwd at 17:00/H=19 is
    the value noctua/season.py computes from the stored profile.

  ON NEVER MOVES THE GRID
    Barrier `pct`/`price` rungs and safe-level alphas are fixed; the anchor
    moves probabilities ON the grid.

  ON WITHOUT THE ARRAYS IS AN ERROR, NOT A SILENT FALLBACK.

  ONE IMPLEMENTATION
    eval/hour_anchor.py (which produced the evidence) and the runtime import
    the same season functions from noctua/season.py.

  AND THE GATE CAN FAIL (R2): a perturbed season coefficient must break the
  algebra check.

    python model/tests/test_hour_anchor.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua import season as SE                                   # noqa: E402
from noctua.features import build_features                        # noqa: E402
from serve import predict as P                                    # noqa: E402
from serve.history import load_bundle                             # noqa: E402
from serve.runtime import load_model                              # noqa: E402

FROZEN = ("/pct", "/price", "/alpha")
PROD_A, PROD_H = 17, 19


def flatten(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if k in ("hour_anchor", "source"):
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
    m.w = {k: v for k, v in model.w.items()
           if k not in ("season_profile", "har_beta_season")}
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
    hr = pd.to_datetime(hts, unit="s", utc=True).hour
    anchor = int(hts[np.flatnonzero(hr == PROD_A)[-1]])

    check("artifact-carries-the-arrays", model.has_hour_anchor,
          "season_profile + har_beta_season present")
    check("default-is-off", P.HOUR_ANCHOR is False,
          f"HOUR_ANCHOR = {P.HOUR_ANCHOR}; switching it on is the owner's call")

    def run(m, flag):
        orig, P.HOUR_ANCHOR = P.HOUR_ANCHOR, flag
        try:
            return P.forecast(m, hours, H=PROD_H, anchor_ts=anchor)
        finally:
            P.HOUR_ANCHOR = orig

    off = run(model, False)
    blind = run(stripped(model), False)
    f_off, f_blind = flatten(off), flatten(blind)
    diff = sorted(k for k in f_off if f_off[k] != f_blind.get(k))
    check("off-is-the-clock-blind-payload", not diff,
          f"differs: {diff}" if diff else f"all {len(f_off)} leaves identical")

    on = run(model, True)
    f_on = flatten(on)
    moved = sorted(k for k in f_off if f_off[k] != f_on.get(k))
    check("on-does-something", len(moved) > 0, f"{len(moved)} leaves move")
    grid = [k for k in moved if any(t in k for t in FROZEN)]
    check("on-never-moves-the-grid", not grid,
          f"moved: {grid}" if grid else "barrier pct/price and alphas held")
    check("payload-reports-the-switch",
          on["hour_anchor"]["enabled"] is True
          and off["hour_anchor"]["enabled"] is False
          and on["hour_anchor"]["season_fwd"] is not None,
          f"season_fwd {on['hour_anchor']['season_fwd']}, "
          f"coef {on['hour_anchor']['season_coef']}")

    # ---- the algebra, at the model level, through the seed ensemble -------
    row = int(np.searchsorted(hts, anchor))
    dt = pd.to_datetime(hts[row], unit="s", utc=True)
    ep = pd.DataFrame({"anchor_ts": [hts[row]], "H": [PROD_H], "row": [row],
                       "dt": [dt], "anchor_hour": [dt.hour],
                       "dow": [dt.dayofweek]})
    X = build_features(hours, ep)
    Hh = np.array([float(PROD_H)])

    def med(m, flag, w=None):
        m.hour_anchor = flag
        keep = m.w
        if w is not None:
            m.w = w
        try:
            d = m.prepare(X, Hh)
            return float(np.log(m.predict(d)["sigma_med"][0])), d
        finally:
            m.w = keep
            m.hour_anchor = False

    lo, d0 = med(model, False)
    ln, d1 = med(model, True)
    sf = SE.season_fwd(model.w["season_profile"], np.array([PROD_A]), Hh)[0]
    hb, bs = model.w["har_beta"], model.w["har_beta_season"]
    xb = d0["Xb"][0]
    want = (1 - model.blend_w) * ((bs[0] + xb @ bs[1:-1] + bs[-1] * sf)
                                  - (hb[0] + xb @ hb[1:]))
    check("season-at-the-production-anchor",
          abs(d1["season_fwd"][0] - sf) < 1e-12,
          f"season_fwd(17:00, H=19) = {sf:+.5f}")
    check("median-moves-by-the-algebra", abs((ln - lo) - want) < 1e-9,
          f"log sigma_med moved {ln - lo:+.6f}, algebra {want:+.6f}")

    # R2: a perturbed coefficient must break the same check
    w_bad = dict(model.w)
    w_bad["har_beta_season"] = bs.copy()
    w_bad["har_beta_season"][-1] += 0.5
    lb, _ = med(model, True, w_bad)
    check("gate-can-fail", abs((lb - lo) - want) > 1e-6,
          f"coefficient +0.5 moves the median by {lb - lo:+.6f}, not {want:+.6f}")

    # ---- on without arrays is an error ------------------------------------
    raised = False
    try:
        run(stripped(model), True)
    except RuntimeError:
        raised = True
    check("on-without-arrays-raises", raised, "no silent clock-blind fallback")

    # ---- one implementation, and the hour inversion ------------------------
    from eval import hour_anchor as EH
    check("eval-and-runtime-share-the-math",
          EH.season_fwd is SE.season_fwd and EH.hour_profile is SE.hour_profile,
          "eval/hour_anchor imports noctua/season")
    hh = np.arange(24)
    back = SE.anchor_hour_from_cal(np.sin(2 * np.pi * hh / 24),
                                   np.cos(2 * np.pi * hh / 24))
    check("hour-recovered-from-cal-features", np.array_equal(back, hh),
          "all 24 anchor hours round-trip")

    print("hour-anchor serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
