"""
tests/test_level_report.py
=====================================================================
The gate for P2-level-report: the reported sigma carries the QLIKE scalar and
the predictive object does not.

WHAT CHANGED 2026-09-13

The reported sigma is no longer a median times a fitted trailing scalar; it is
the model's own conditional MEAN of variance (`P3-functional-parity`). So this
gate no longer toggles a scalar -- it toggles the FUNCTIONAL, running `forecast()`
once with `REPORT_FUNCTIONAL = "median"` and once with `"mean"`.

Two of the old checks had to go, and one of them was weak anyway.
`P3-barrier-channel` proved by probe that the committee builds every barrier
curve from `pred["sigma_atoms"]` and never reads the reported scalar: a 13%
change in it moves the curves by 0.000e+00 while the same change to
`sigma_atoms` moves them by 8.9e-03. So "only the sigma fields moved" cannot
fail via the reported number, and this gate's strength was overstated. It is
kept because it still guards a future change that writes into `pred` -- which is
what actually damaged `P2-scale-v2` -- and the leak probe below is what keeps it
honest about that.

WHY THIS TEST IS THE GATE AND THE EXPERIMENT WAS NOT

`eval/level_report.py` measures three readings of ONE predictive object, so
"the barriers cannot move" is true by construction there -- and a claim that
cannot fail is not a claim (R2). This test CAN fail. It runs `forecast()` twice
against the same anchor, once on each functional, and requires:

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
        if k in ("sigma_scale", "sigma_functional", "vol_calibration",
                 "source", "history_hours"):
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

    def run(functional):
        orig, P.REPORT_FUNCTIONAL = P.REPORT_FUNCTIONAL, functional
        try:
            return P.forecast(model, hours)
        finally:
            P.REPORT_FUNCTIONAL = orig

    off = run("median")                 # the historical behaviour
    on = run("mean")                    # the adopted behaviour
    sf = on["sigma_functional"]
    s = float(sf["mean_over_median"])

    fo, fn = _flatten(off), _flatten(on)
    moved = _diff(fo, fn)
    ok.append(("default-is-the-mean", P.REPORT_FUNCTIONAL == "mean",
               f"REPORT_FUNCTIONAL = {P.REPORT_FUNCTIONAL!r}; QLIKE is minimised "
               f"by the conditional mean of variance"))
    ok.append(("functional-changes-the-number", s != 1.0,
               f"mean/median = {s:.4f} (a ratio of exactly 1 would make every "
               f"check below vacuous)"))
    ok.append(("trailing-scalar-is-inert",
               on["sigma_scale"]["applied"] is False,
               f"the old fitted scalar ({on['sigma_scale']['scale']:.4f}) is "
               f"reported but no longer multiplies anything -- it estimated the "
               f"same ratio the model now takes exactly, and under-estimated it"))
    ok.append(("only-sigma-moved", set(moved) <= set(SIGMA_FIELDS),
               f"fields that changed: {sorted(moved)}"))
    ok.append(("both-sigma-moved", set(SIGMA_FIELDS) <= set(moved),
               "the reported sigma actually carries the scalar"))
    if set(SIGMA_FIELDS) <= set(moved):
        r = fn["sigma_window_pct"] / max(fo["sigma_window_pct"], 1e-12)
        ok.append(("sigma-moved-by-exactly-the-functional-ratio",
                   abs(r - s) < 2e-3,
                   f"reported sigma ratio {r:.4f} against mean/median {s:.4f}"))
        ok.append(("reported-equals-sigma_mean",
                   abs(fn["sigma_window_pct"] - sf["sigma_mean_pct"]) < 1e-6
                   and abs(fo["sigma_window_pct"] - sf["sigma_med_pct"]) < 1e-6,
                   f"mean run publishes {sf['sigma_mean_pct']}, median run "
                   f"publishes {sf['sigma_med_pct']} -- the published number is "
                   f"the functional itself, with nothing applied on top"))

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
