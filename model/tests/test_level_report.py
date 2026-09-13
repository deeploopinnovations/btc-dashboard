"""
tests/test_level_report.py
=====================================================================
The gate for P2-level-report: the reported sigma carries the QLIKE scalar and
the predictive object does not.

WHY THIS TEST IS THE GATE AND THE EXPERIMENT WAS NOT

`eval/level_report.py` measures three readings of ONE predictive object, so
"the barriers cannot move" is true by construction there -- and a claim that
cannot fail is not a claim (R2). This test CAN fail. It runs `forecast()` twice
against the same anchor, once with the scalar forced to 1 and once with the
real trailing scalar, and requires:

  * every touch probability, every safe level, p_up and p_vol_amplify
    BIT-IDENTICAL between the two runs;
  * exactly `sigma_window_pct` and `sigma_annualized_pct` different;
  * both of those moved by exactly the scalar.

The moment the scalar reaches `pred`, the first assertion fails. That is the
whole point: `P2-scale-v2` and `P2-mean-level` both moved the level inside the
predictive object and degraded all six barrier metrics -- and `P2-mean-level`'s
SHUFFLED control degraded them by the same amount, so the damage is caused by
moving the level at all rather than by how the shift is obtained.

AND THE TEST IS SHOWN TO BE CAPABLE OF FAILING. The last check deliberately
leaks the scalar into the predictive object and asserts that the comparison
then reports a difference. A regression test whose failure mode has never been
exercised is the fifth guard in this project to print reassuring output while
being unable to return the other answer (R2).

    python -m model.tests.test_level_report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import predict as P                                    # noqa: E402
from serve.history import load_bundle                             # noqa: E402
from serve.runtime import load_model                              # noqa: E402

SIGMA_FIELDS = ("sigma_window_pct", "sigma_annualized_pct")


def _flatten(d: dict) -> dict:
    """Every leaf of the forecast that a consumer can read."""
    out = {}
    for k, v in d.items():
        if k in ("sigma_scale", "vol_calibration", "source", "history_hours"):
            continue
        if k == "barrier_curves":
            for side, rows in v.items():
                for r in rows:
                    out[f"barrier/{side}/{r['pct']}/touch_prob"] = r["touch_prob"]
                    out[f"barrier/{side}/{r['pct']}/price"] = r["price"]
        elif k == "safe_levels":
            for r in v:
                for f in ("call_strike", "put_strike", "call_pct", "put_pct"):
                    out[f"safe/{r['alpha']}/{f}"] = r[f]
        else:
            out[k] = v
    return out


def _diff(a: dict, b: dict) -> list[str]:
    return [k for k in a if not (a[k] == b[k])]


def main() -> int:
    ok = []
    model = load_model()
    hours = load_bundle()
    if hours is None or len(hours) < 24 * 400:
        print("no committed history bundle; cannot run the serving gate")
        return 1

    def run(scale_fn):
        orig, P.qlike_scale = P.qlike_scale, scale_fn
        try:
            return P.forecast(model, hours)
        finally:
            P.qlike_scale = orig

    off = run(lambda *a, **k: {"scale": 1.0, "applied": False, "n_episodes": 0,
                               "reason": "forced off for the gate"})
    on = P.forecast(model, hours)
    s = float(on["sigma_scale"]["scale"]) if on["sigma_scale"]["applied"] else 1.0

    fo, fn = _flatten(off), _flatten(on)
    moved = _diff(fo, fn)
    ok.append(("scalar-applied", s != 1.0,
               f"trailing QLIKE scalar = {s:.4f} (a scalar of exactly 1 would "
               f"make every check below vacuous)"))
    ok.append(("only-sigma-moved", set(moved) <= set(SIGMA_FIELDS),
               f"fields that changed: {sorted(moved)}"))
    ok.append(("both-sigma-moved", set(SIGMA_FIELDS) <= set(moved),
               "the reported sigma actually carries the scalar"))
    if set(SIGMA_FIELDS) <= set(moved):
        r = fn["sigma_window_pct"] / max(fo["sigma_window_pct"], 1e-12)
        ok.append(("sigma-moved-by-exactly-the-scalar", abs(r - s) < 2e-3,
                   f"reported sigma ratio {r:.4f} against scalar {s:.4f}"))

    # THE CHECK THAT PROVES THE CHECKS CAN FAIL: leak the scalar into the
    # predictive object and confirm the comparison notices.
    probe_on = P.forecast(model, hours)
    probe_leak = dict(probe_on)
    probe_leak["barrier_curves"] = {
        side: [{**r, "touch_prob": min(1.0, r["touch_prob"] * s)} for r in rows]
        for side, rows in probe_on["barrier_curves"].items()}
    leaked_moved = _diff(_flatten(probe_on), _flatten(probe_leak))
    ok.append(("gate-can-fail", len(leaked_moved) > 0,
               f"a leaked scalar moves {len(leaked_moved)} barrier fields and "
               f"the comparison sees it"))

    print("level-report serving gate")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
