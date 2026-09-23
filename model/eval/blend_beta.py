"""
eval/blend_beta.py
=====================================================================
P3-blend-beta: does the Log-HAR blend CAUSE the under-reaction it is blamed
for sitting in training?

THE TWO LINES THAT NEVER MET

Every measurement of the MZ slope in this project -- `P3-beta-is-affine`,
`P3-beta-sample-size`, `eval/beta_stability.py`, the whole shrinkage line --
was made on `teacher_oof.npz`, where `noctua_v1` is the RAW network.
`P3-functional-adopt-scope` says so in as many words. Serving BLENDS that
network with Log-HAR at `blend_w = 0.25`.

The blend weight has been tested twice (`E-blend`, `E-blend-1state`) and both
times the readout was QLIKE. Nobody looked at beta.

WHY THE ANSWER IS NOT NEUTRAL, AND WHY IT COSTS NOTHING

The blend is affine in log space, so with `r = log RV`, `x = log sigma_noctua`
and `y = log sigma_loghar`,

    beta(w) = [w cov(r,x) + (1-w) cov(r,y)]  /  var(w x + (1-w) y)

The numerator is LINEAR in w. The denominator is QUADRATIC, and

    var(wx + (1-w)y) = w^2 vx + (1-w)^2 vy + 2w(1-w) rho sqrt(vx vy)
                     <  w vx + (1-w) vy        whenever rho < 1

so averaging two imperfectly correlated forecasts compresses the dynamic range
by more than it averages it. A smaller denominator raises the slope. beta(w)
can therefore EXCEED BOTH ENDPOINTS in the interior -- an under-reaction
manufactured by averaging, out of two components that may each under-react
less.

Nothing here is retrained. One pass over an artifact that already exists.

WHAT IT WOULD MEAN. If beta peaks in the interior and the shipped weight sits
near the peak, `beta > 1` is not a training defect: it is arithmetic, and the
price of a robustness trade this project made knowingly -- pure NOCTUA lost
+72.3% in the 2023 volatility collapse and the blend bounded that fold at
+6.7%.

THE APPROXIMATION, NAMED BEFORE THE RUN. `teacher_oof`'s `log_har` is the
teacher-zoo fit; serving's anchor is `log_har_cal` through `BASE_COLS`. Same
family, different fit, so `beta(0.25)` here is the served blend's slope up to
that difference rather than the served number itself. The SHAPE of `beta(w)` is
what the prediction is about and it does not depend on which Log-HAR is used --
which is checked by running the sweep against both.

    python -m model.eval.blend_beta --selftest
    python -m model.eval.blend_beta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mz_recalibration import YEARS, fit_mz                          # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                    # noqa: E402
from eval.teacher_zoo import FoldScopedFit                               # noqa: E402

NEURAL = "noctua_v1"
ANCHORS = ("log_har", "log_har_cal")
SHIPPED_W = 0.25
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 3)


def blend_log(x: np.ndarray, y: np.ndarray, w: float) -> np.ndarray:
    """log sigma of the blend: w * neural + (1 - w) * anchor.

    `infer.predict` shifts the whole predictive distribution so its median
    lands on `qa_med + (1 - w)(har - qa_med)`, which is exactly this. w = 1 is
    the raw network and w = 0 the pure anchor, so the two endpoints of the
    sweep are objects this project has already measured separately.
    """
    return float(w) * x + (1.0 - float(w)) * y


def beta_curve(rv, sig_n, sig_a, grid=GRID) -> dict:
    """MZ slope of the blended forecast at each weight, one slice."""
    ok = (np.isfinite(rv) & np.isfinite(sig_n) & np.isfinite(sig_a)
          & (rv > 0) & (sig_n > 0) & (sig_a > 0))
    if ok.sum() < 200:
        return {}
    r, x, y = np.log(rv[ok]), np.log(sig_n[ok]), np.log(sig_a[ok])
    out = {}
    for w in grid:
        s = np.exp(blend_log(x, y, w))
        p = fit_mz(np.exp(r), s)
        if p is not None:
            out[float(w)] = float(p[1])
    return out


def analytic_beta(r, x, y, w: float) -> float:
    """The closed form, used only to check the regression agrees with it.

    A sweep that recomputed an OLS twenty-one times and a formula that says
    what the sweep must produce are two implementations of one quantity; if
    they disagree, one of them is wrong and the selftest says which.
    """
    z = blend_log(np.asarray(x, np.float64), np.asarray(y, np.float64), w)
    v = float(np.var(z, ddof=1))
    if v < 1e-12:
        return float("nan")
    return float(np.cov(np.asarray(r, np.float64), z, ddof=1)[0, 1] / v)


def run_horizon(z, H: int, anchor: str) -> dict:
    """Pooled over folds, plus each fold, for one horizon and one anchor."""
    rvs, ns, as_ = [], [], []
    folds = {}
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kt = f"{y}/{H}/test"
            if (f"{kt}/sigma/{NEURAL}" not in z
                    or f"{kt}/sigma/{anchor}" not in z):
                continue
            sn = np.asarray(sc.test(z, H, NEURAL), np.float64)
            sa = np.asarray(sc.test(z, H, anchor), np.float64)
            rv = np.asarray(z[f"{kt}/rv"], np.float64)
        c = beta_curve(rv, sn, sa)
        if c:
            folds[y] = c
        rvs.append(rv); ns.append(sn); as_.append(sa)
    if not rvs:
        return {}
    pooled = beta_curve(np.concatenate(rvs), np.concatenate(ns),
                        np.concatenate(as_))
    return {"pooled": pooled, "folds": folds}


def summarise(pooled: dict) -> dict:
    """Where the curve peaks, and whether the peak is interior."""
    if not pooled:
        return {}
    ws = sorted(pooled)
    bs = [pooled[w] for w in ws]
    i = int(np.argmax(bs))
    return {"w_peak": ws[i], "beta_peak": bs[i],
            "beta_raw": pooled.get(1.0), "beta_anchor": pooled.get(0.0),
            "beta_shipped": pooled.get(SHIPPED_W),
            "interior_peak": bool(0 < i < len(ws) - 1),
            "peak_above_both": bool(bs[i] > max(bs[0], bs[-1]) + 1e-9)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MZ slope against the blend weight")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/blend_beta.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    print("P3-blend-beta   DIAGNOSTIC, no decision rule, nothing retrained")
    print("beta(w) for log sigma = w * noctua_v1 + (1 - w) * anchor\n")
    out = {"neural": NEURAL, "shipped_w": SHIPPED_W, "anchors": {}}
    for anchor in ANCHORS:
        print(f"=== anchor {anchor}")
        head = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
        print(f"{'H':>5} " + " ".join(f"w={w:<4.2f}" for w in head)
              + f" {'peak w':>8} {'peak b':>8}")
        hh = {}
        for H in HORIZONS:
            r = run_horizon(z, H, anchor)
            if not r:
                continue
            s = summarise(r["pooled"])
            hh[str(H)] = {"pooled": r["pooled"], "summary": s,
                          "folds": {str(k): v for k, v in r["folds"].items()}}
            row = " ".join(f"{r['pooled'].get(w, float('nan')):6.3f} "
                           for w in head)
            print(f"{H:>5} {row} {s['w_peak']:>8.2f} {s['beta_peak']:>8.3f}"
                  + ("  INTERIOR" if s["peak_above_both"] else ""))
        out["anchors"][anchor] = hh
        print()

    print("the prediction was: beta(w) exceeds BOTH endpoints somewhere in the "
          "interior,\nand the shipped w = 0.25 sits nearer that peak than the "
          "raw network does.")
    for anchor, hh in out["anchors"].items():
        for H, d in hh.items():
            s = d["summary"]
            if s.get("beta_raw") is None or s.get("beta_shipped") is None:
                continue
            closer = abs(s["beta_shipped"] - s["beta_peak"]) < abs(
                s["beta_raw"] - s["beta_peak"])
            print(f"  {anchor:>12} H={H:>3}  peak above both endpoints: "
                  f"{str(s['peak_above_both']):>5}   shipped nearer the peak "
                  f"than raw: {str(closer):>5}   "
                  f"raw {s['beta_raw']:.3f} -> shipped {s['beta_shipped']:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(17)
    n = 20000

    # 1-2. The blend is the affine map serving applies, at both endpoints.
    x = rng.normal(size=n); y = rng.normal(size=n)
    ok.append(("w=1 is the raw network", np.array_equal(blend_log(x, y, 1.0), x)))
    ok.append(("w=0 is the pure anchor", np.array_equal(blend_log(x, y, 0.0), y)))

    # 3. THE MECHANISM, PLANTED. Two imperfectly correlated forecasts, each
    #    UNBIASED in slope against the same target. Averaging them must push
    #    the slope ABOVE 1 even though neither component does.
    t = rng.normal(0.0, 1.0, n)                 # the true log level
    xa = t + rng.normal(0.0, 0.6, n)            # forecast A, independent noise
    ya = t + rng.normal(0.0, 0.6, n)            # forecast B
    r = t + rng.normal(0.0, 0.3, n)             # the outcome
    b_a = analytic_beta(r, xa, ya, 1.0)
    b_b = analytic_beta(r, xa, ya, 0.0)
    b_m = analytic_beta(r, xa, ya, 0.5)
    ok.append(("averaging raises the slope above both components",
               b_m > max(b_a, b_b) + 1e-6))
    ok.append(("and the components are the ones planted",
               abs(b_a - b_b) < 0.05))

    # 4. The closed form and the regression are the same quantity. If they
    #    disagree, the sweep is measuring something the algebra does not
    #    describe and the whole argument above is decoration.
    rv = np.exp(r); sn = np.exp(xa); sa = np.exp(ya)
    c = beta_curve(rv, sn, sa, grid=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    worst = max(abs(c[w] - analytic_beta(r, xa, ya, w)) for w in c)
    ok.append(("the OLS sweep agrees with the closed form", worst < 1e-8))

    # 5. PERFECTLY correlated forecasts give a MONOTONE curve -- no
    #    compression, no interior peak. The mechanism needs rho < 1 and this
    #    is what shows the planted result is not an artifact of the fixture.
    yb = xa.copy()
    flat = [analytic_beta(r, xa, yb, w) for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ok.append(("identical forecasts give a flat curve",
               max(flat) - min(flat) < 1e-9))

    # 6. Too few points refuses rather than fitting.
    ok.append(("a short slice returns nothing",
               beta_curve(rv[:50], sn[:50], sa[:50]) == {}))

    # 7. The summary reports an interior peak as interior.
    s = summarise({0.0: 1.0, 0.5: 1.4, 1.0: 1.1})
    ok.append(("an interior peak is flagged",
               s["peak_above_both"] and s["w_peak"] == 0.5))
    s2 = summarise({0.0: 1.0, 0.5: 1.1, 1.0: 1.4})
    ok.append(("a monotone curve is not",
               not s2["peak_above_both"] and s2["w_peak"] == 1.0))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
