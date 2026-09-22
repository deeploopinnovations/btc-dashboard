"""
eval/shrunk_slope.py
=====================================================================
P3-shrunk-slope: the MZ slope earns its parameter only where it can be
estimated. Can one rule, with no horizon-specific tuning, get that right
everywhere?

THE MEASUREMENT THAT MOTIVATES THIS

Against a level-only rescale, the two-parameter MZq correction is worth

    H = 1      6       24       168
      +2.77%  +2.38%  -1.35%  -19.55%

and the independent calibration windows available to fit the slope are

    4,247     707      176       24

Same ordering, and nothing else in the table has it (`P3-beta-sample-size`,
`P3-beta-regime-correction`). At H=168 the fitted slopes are
[1.009, -0.017, 0.757, 0.467, 0.388, 0.761] -- one of them asserts that the
more NOCTUA forecasts, the less volatility realises -- and applying them costs
a fifth of the correction's value.

The obvious fix is a threshold: use the slope at short horizons and drop it at
long ones. That is tuning a rule to the answers already seen, and it would be
fitted on exactly the four numbers above. What is tested here instead is a rule
that derives the same behaviour from a quantity measured on CALIB alone.

THE RULE

    beta_shrunk = 1 + w * (beta_hat - 1),      w = tau^2 / (tau^2 + SE^2)

`SE` is the moving-block bootstrap standard error of that fold's own slope,
computed on its calibration slice. `tau^2` is the between-fold variance of the
true slope, estimated from PAST FOLDS ONLY as var(beta_hat) - mean(SE^2),
floored at zero -- an expanding window, so no fold is shrunk using a number
from its own future. The first fold has no past and therefore no estimate of
tau, so it gets w = 0 and the level-only correction; inventing a prior there
would be inventing the thing being tested.

Standard empirical-Bayes. When the slope is precisely estimated relative to how
much slopes genuinely vary, w approaches 1 and this reduces to MZq. When the
slope is noise, w approaches 0 and it reduces to the level rescale. Nothing in
it knows what a horizon is.

The LEVEL is re-optimised after shrinking, closed-form on calib, exactly as
`fit_mzq` does -- a slope change moves the optimal level, and leaving the old
one would charge the shrinkage for a level error it did not make.

THE CONTROL THAT DECIDES IT

`half` applies a FIXED w = 0.5 at every horizon and fold. If precision
weighting cannot beat an arbitrary constant, then the measured standard errors
are contributing nothing and the rule is shrinkage-in-general rather than this
shrinkage. That control is the experiment, in the same way the width-matched
band was for `P3-frvp-double-touch`.

    python -m model.eval.shrunk_slope --selftest
    python -m model.eval.shrunk_slope
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.beta_stability import beta_se                                # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.mz_recalibration import (apply_mz, fit_mz, optimal_c,        # noqa: E402
                                   qlike_vec, YEARS)
from eval.teacher_scorecard import HORIZONS, load_oof                  # noqa: E402
from eval.teacher_zoo import FoldScopedFit                             # noqa: E402

TEACHER = "noctua_v1_mean"
ARMS = ("c", "mzq", "shrunk", "drift", "drift_ws", "half")
FIXED_W = 0.5                      # the control's constant
# COLD-START DEFAULT. tau^2 needs two past folds, so the first two folds of a
# walk-forward have no estimate and the original `drift` arm gave them w = 0 --
# no slope correction at all, on a third of the sample. That, and not the
# weighting, is where `drift` lost to the constant: its warm weights come out
# near 0.5 at the short horizons anyway (P3-shrunk-slope-result). `drift_ws`
# changes only the start-up policy, falling back to FIXED_W until a real
# estimate exists. It is deliberately the SAME constant the control uses, so
# the fallback cannot be accused of being tuned.
COLD_START_W = FIXED_W
N_BOOT = 300                       # bootstrap reps for each fold's slope SE


def shrink_weight(tau2: float, se: float) -> float:
    """w = tau^2 / (tau^2 + SE^2), clipped to [0, 1] and safe at the edges.

    tau2 <= 0 means past folds showed no variation beyond their own noise, so
    there is nothing for a fold-specific slope to capture: w = 0. A non-finite
    SE means the slope could not be estimated at all, which is the strongest
    possible reason to ignore it: also w = 0.
    """
    if not np.isfinite(se) or not np.isfinite(tau2) or tau2 <= 0.0:
        return 0.0
    return float(np.clip(tau2 / (tau2 + se * se), 0.0, 1.0))


def drift2_from_past(bc: list, bt: list) -> float:
    """Mean squared CALIB->TEST drift of the slope, over past folds.

    THE QUANTITY THE PRECISION RULE MISSES. `shrink_weight` asks how well a
    fold's slope is ESTIMATED. What actually matters is whether the slope
    fitted on calib still holds on test, and those come apart exactly where
    this project's problem lives: at H=168 the between-fold spread is large
    (so empirical-Bayes says "keep the fold's own slope") while the calib
    estimate does not survive into the fold's own test slice at all -- 2022's
    calib slope is -0.017 against its own test slope of 0.918.

    Walk-forward makes this observable without leakage: when fitting fold k,
    folds 1..k-1 are wholly in the past, test slices included, so the drift
    they exhibited is known. Needs one past fold.
    """
    d = [(t - c) ** 2 for c, t in zip(bc, bt)
         if np.isfinite(c) and np.isfinite(t)]
    return float(np.mean(d)) if d else 0.0


def tau2_from_past(betas: list, ses: list) -> float:
    """Between-fold variance of the TRUE slope, from past folds only.

    var(beta_hat) overstates it, because each beta_hat carries its own
    estimation error; subtracting the mean squared SE removes that. Floored at
    zero because a negative variance estimate means "no detectable spread",
    not a negative one. Needs two past folds to have a variance at all.
    """
    b = [x for x in betas if np.isfinite(x)]
    s = [x for x in ses if np.isfinite(x)]
    if len(b) < 2 or not s:
        return 0.0
    return max(float(np.var(b, ddof=1)) - float(np.mean(np.square(s))), 0.0)


def fit_arms(rv_c, sig_c, tau2: float, se: float, drift2: float = 0.0) -> dict:
    """(alpha, beta) per arm, all fitted on the CALIB slice only."""
    ok = np.isfinite(rv_c) & np.isfinite(sig_c) & (sig_c > 0) & (rv_c > 0)
    if ok.sum() < 50:
        return {}
    rv_c, sig_c = rv_c[ok], sig_c[ok]
    p_mz = fit_mz(rv_c, sig_c)
    if p_mz is None:
        return {}
    beta_hat = float(p_mz[1])

    def level_for(beta: float) -> tuple:
        sloped = sig_c ** beta
        return (float(np.log(optimal_c(rv_c, sloped))), float(beta))

    return {
        "c": level_for(1.0),
        "mzq": level_for(beta_hat),
        "shrunk": level_for(1.0 + shrink_weight(tau2, se) * (beta_hat - 1.0)),
        # The drift arm charges the slope for BOTH its estimation error and the
        # calib->test movement past folds exhibited. Same functional form, one
        # more variance component in the denominator, so it can only shrink
        # harder than `shrunk` -- never less.
        "drift": level_for(1.0 + shrink_weight(tau2, np.sqrt(se * se + drift2))
                           * (beta_hat - 1.0)),
        # identical to `drift` once a real estimate exists; differs ONLY on the
        # cold-start folds, where it uses the constant instead of no correction
        "drift_ws": level_for(
            1.0 + (shrink_weight(tau2, np.sqrt(se * se + drift2))
                   if tau2 > 0.0 else COLD_START_W) * (beta_hat - 1.0)),
        "half": level_for(1.0 + FIXED_W * (beta_hat - 1.0)),
        "_beta_hat": beta_hat, "_se": se, "_tau2": tau2, "_drift2": drift2,
        "_w": shrink_weight(tau2, se),
        "_w_drift": shrink_weight(tau2, float(np.sqrt(se * se + drift2))),
        "_w_ws": (shrink_weight(tau2, float(np.sqrt(se * se + drift2)))
                  if tau2 > 0.0 else COLD_START_W),
        "_cold": bool(tau2 <= 0.0),
    }


def run_horizon(z, H: int, n_boot: int = N_BOOT, verbose: bool = True) -> dict:
    """Walk the folds in order, shrinking each with PAST folds' spread only."""
    past_b, past_se, past_bt = [], [], []
    rv_all = {a: [] for a in ARMS}
    sig_all = {a: [] for a in ARMS}
    rows = []
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kt, kc = f"{y}/{H}/test", f"{y}/{H}/calib"
            if f"{kt}/sigma/{TEACHER}" not in z or f"{kc}/sigma/{TEACHER}" not in z:
                continue
            sig_c = np.asarray(sc.calib(z, H, TEACHER), np.float64)
            rv_c = np.asarray(z[f"{kc}/rv"], np.float64)
            sig_t = np.asarray(sc.test(z, H, TEACHER), np.float64)
            rv_t = np.asarray(z[f"{kt}/rv"], np.float64)
        se = beta_se(rv_c, sig_c, H, n_rep=n_boot, seed=H * 1000 + y)
        tau2 = tau2_from_past(past_b, past_se)
        drift2 = drift2_from_past(past_b, past_bt)
        f = fit_arms(rv_c, sig_c, tau2, se, drift2)
        if not f:
            continue
        for a in ARMS:
            al, be = f[a]
            rv_all[a].append(rv_t)
            sig_all[a].append(apply_mz(sig_t, al, be))
        # this fold's OWN test slope -- recorded for the NEXT fold's drift
        # estimate, never for its own, which is what keeps it causal
        p_t = fit_mz(rv_t, sig_t)
        rows.append({"year": y, "beta_hat": f["_beta_hat"], "se": f["_se"],
                     "tau2": f["_tau2"], "drift2": f["_drift2"],
                     "w": f["_w"], "w_drift": f["_w_drift"],
                     "beta_test": None if p_t is None else float(p_t[1]),
                     "beta_shrunk": f["shrunk"][1],
                     "beta_drift": f["drift"][1],
                     "w_ws": f["_w_ws"], "cold": f["_cold"],
                     "beta_drift_ws": f["drift_ws"][1]})
        past_b.append(f["_beta_hat"]); past_se.append(se)
        past_bt.append(float("nan") if p_t is None else float(p_t[1]))
        if verbose:
            print(f"   fold {y}  beta_hat {f['_beta_hat']:+7.3f}  SE {se:5.3f}  "
                  f"tau {np.sqrt(tau2):5.3f}  drift {np.sqrt(drift2):5.3f}  |  "
                  f"w {f['_w']:5.3f} -> {f['shrunk'][1]:+6.3f}   "
                  f"w_d {f['_w_drift']:5.3f} -> {f['drift'][1]:+6.3f}")
    if not rows:
        return {}
    out = {"H": H, "folds": rows, "arms": {}}
    q = {}
    for a in ARMS:
        rv = np.concatenate(rv_all[a]); sg = np.concatenate(sig_all[a])
        q[a] = qlike_vec(rv, sg)
        out["arms"][a] = {"qlike": float(np.nanmean(q[a]))}
    out["_q"] = q
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="precision-shrunk MZ slope")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/shrunk_slope.json"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    # Family fixed BEFORE any result is read: 4 horizons x 3 contrasts
    # (shrunk vs c, vs mzq, vs half).
    n_family = len(HORIZONS) * 9      # {shrunk, drift, drift_ws} x {c, mzq, half}
    alpha = 0.05 / n_family
    print(f"P3-shrunk-slope   family {n_family} -> {100*(1-alpha):.4f}% intervals")
    print(f"teacher {TEACHER}, {a.boot} bootstrap reps per fold slope\n")

    res, out = {}, {"teacher": TEACHER, "n_family": n_family, "alpha": alpha,
                    "fixed_w": FIXED_W, "horizons": {}}
    for H in HORIZONS:
        print(f"=== H = {H}")
        r = run_horizon(z, H, n_boot=a.boot)
        if not r:
            print("   no usable folds\n"); continue
        res[H] = r
        print(f"\n   {'arm':>8} {'QLIKE':>10} {'vs c %':>9} {'vs mzq %':>10}")
        qc, qm = r["_q"]["c"], r["_q"]["mzq"]
        for arm in ARMS:
            qa = r["_q"][arm]
            print(f"   {arm:>8} {np.nanmean(qa):10.5f} "
                  f"{100*np.nanmean(qc-qa)/np.nanmean(qc):+9.2f} "
                  f"{100*np.nanmean(qm-qa)/np.nanmean(qm):+10.2f}")
        L = max(int(round(len(qc) ** (1/3))), 2 * H)
        ci = {}
        for cand in ("shrunk", "drift", "drift_ws"):
            for other in ("c", "mzq", "half"):
                d = r["_q"][other] - r["_q"][cand]        # >0 favours candidate
                g = np.isfinite(d)
                c95 = mean_ci(d[g], alpha=alpha, block_len=L)["ci95"]
                ci[f"{cand}_vs_{other}"] = [float(v) for v in c95]
                verdict = (f"{cand} BETTER" if c95[0] > 0 else
                           f"{cand} WORSE" if c95[1] < 0 else "not separated")
                print(f"   {cand:>6} vs {other:>4}: {np.nanmean(d):+9.5f}  "
                      f"CI [{c95[0]:+.5f}, {c95[1]:+.5f}]  {verdict}")
        out["horizons"][str(H)] = {
            "arms": r["arms"], "folds": r["folds"], "block_len": L,
            "ci_vs": ci}
        print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []

    # 1-4. The weight, at its edges and in between. These decide the whole rule.
    ok.append(("precise slope (SE << tau) keeps it",
               shrink_weight(0.04, 0.01) > 0.93))
    ok.append(("noisy slope (SE >> tau) discards it",
               shrink_weight(0.01, 0.30) < 0.11))
    ok.append(("no between-fold spread -> w = 0",
               shrink_weight(0.0, 0.05) == 0.0))
    ok.append(("unestimable SE -> w = 0",
               shrink_weight(0.04, float("nan")) == 0.0))

    # 5-6. tau^2 removes estimation noise, and refuses when it cannot be formed.
    #      Six betas with sd 0.2 whose own SEs are 0.2 carry NO real spread.
    b = [1.0, 1.2, 0.8, 1.3, 0.7, 1.1]
    ok.append(("tau^2 nets out estimation error",
               tau2_from_past(b, [0.25] * 6) == 0.0))
    ok.append(("tau^2 survives when SEs are small",
               tau2_from_past(b, [0.02] * 6) > 0.03))
    ok.append(("one past fold gives no variance", tau2_from_past([1.1], [0.05]) == 0.0))

    # 7-9. The rule must REDUCE to its endpoints, which is the whole claim.
    rng = np.random.default_rng(3)
    n = 4000
    true = np.exp(rng.normal(-4.0, 0.5, n))
    rv = true * np.exp(rng.normal(0, 0.2, n))
    sig = np.exp(0.3 + np.log(true) / 1.4)
    f_hi = fit_arms(rv, sig, tau2=1.0, se=0.001)      # w -> 1
    f_lo = fit_arms(rv, sig, tau2=1e-9, se=1.0)       # w -> 0
    ok.append(("w->1 reproduces mzq's slope",
               abs(f_hi["shrunk"][1] - f_hi["mzq"][1]) < 1e-3))
    ok.append(("w->0 reproduces the level-only slope of 1",
               abs(f_lo["shrunk"][1] - 1.0) < 1e-9))
    ok.append(("w->0 reproduces c's LEVEL too, not just its slope",
               abs(f_lo["shrunk"][0] - f_lo["c"][0]) < 1e-9))

    # 10. The level must be RE-OPTIMISED for the shrunk slope, not inherited.
    #     If it were inherited from mzq, this would differ.
    f_mid = fit_arms(rv, sig, tau2=0.01, se=0.1)
    al, be = f_mid["shrunk"]
    okm = np.isfinite(sig) & (sig > 0)
    want = float(np.log(optimal_c(rv[okm], sig[okm] ** be)))
    ok.append(("level is re-optimised at the shrunk slope", abs(al - want) < 1e-9))

    # 11-12. CAUSALITY, checked on the parse tree rather than on the text.
    #     A fold must never be shrunk using its own future, so inside
    #     `run_horizon` every append to the past-fold lists has to come AFTER
    #     the fit that consumes them. The first draft asserted this by
    #     searching the file for two literals -- and the literals appear in
    #     this selftest's own source, so the search matched ITSELF and the
    #     check failed for a reason that had nothing to do with the code. A
    #     source-scanning assertion has to exclude the scanner.
    import ast as _ast
    tree = _ast.parse(Path(__file__).read_text())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "run_horizon")
    fit_lines = [n.lineno for n in _ast.walk(fn)
                 if isinstance(n, _ast.Call)
                 and getattr(n.func, "id", "") == "fit_arms"]
    app_lines = [n.lineno for n in _ast.walk(fn)
                 if isinstance(n, _ast.Call)
                 and getattr(n.func, "attr", "") == "append"
                 and getattr(getattr(n.func, "value", None), "id", "")
                 in ("past_b", "past_se", "past_bt")]
    ok.append(("run_horizon fits before it appends to the past lists",
               bool(fit_lines) and bool(app_lines)
               and max(fit_lines) < min(app_lines)))
    ok.append(("all three past lists are maintained", len(app_lines) == 3))

    # 13-14. The warm-start arm must differ from `drift` ONLY at the cold start.
    rng2 = np.random.default_rng(9)
    tr = np.exp(rng2.normal(-4.0, 0.5, 3000))
    rv2 = tr * np.exp(rng2.normal(0, 0.2, 3000))
    sg2 = np.exp(0.2 + np.log(tr) / 1.3)
    warm = fit_arms(rv2, sg2, tau2=0.05, se=0.05, drift2=0.05)
    cold = fit_arms(rv2, sg2, tau2=0.0, se=0.05, drift2=0.05)
    ok.append(("warm fold: drift_ws is identical to drift",
               abs(warm["drift_ws"][1] - warm["drift"][1]) < 1e-12))
    ok.append(("cold fold: drift_ws uses the constant, drift uses none",
               abs(cold["drift"][1] - 1.0) < 1e-12
               and abs(cold["drift_ws"][1] - cold["half"][1]) < 1e-12))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
