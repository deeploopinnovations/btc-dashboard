"""
eval/shock_dispersion.py
=====================================================================
P4-shock-dispersion: make the dispersion correction SPIKE-AWARE.

WHAT IS COMBINED, AND WHY NEITHER HALF IS ENOUGH ALONE

1. The width correction works on average. The predictive distribution is
   over-dispersed by ~13% on the production slice; narrowing it improves four
   of six barrier metrics while its mirror damages the same four
   (P3-dispersion-barriers-result, P3-dispersion-deployable-result).

2. Spike nights are predictable from what is known at the anchor, and they are
   where the loss is. BENCHMARK 12: a causal spike classifier reaches AUC 0.78
   (onset 0.73); 7.7% of nights carry 25.8% of all loss.

A UNIFORM narrowing takes width away from every night, including the ones
about to break -- exactly the reallocation P3-dispersion-conditional could not
measure. This conditions the correction on a causal SHOCK SCORE, so the
distribution narrows where the night looks calm and keeps its width where a
spike is brewing.

THE SHOCK SCORE is `har_6h - har_22d`: the last six hours' log vol rate
against the 22-day one. Both are existing causal features; nothing new is
fitted to produce it. Positive means volatility is running above its own norm.

THE RULE, all fitted on each fold's CALIB slice only:
  * tercile cut points of the shock score on calib;
  * one lambda per tercile, `fit_lambda` on that tercile's calib episodes --
    the same estimator the uniform correction uses, applied three times;
  * every test episode takes its tercile's lambda.

Three parameters in place of one. Arms: M0 shipped, M1 the uniform per-fold
lambda (the control, and the thing this has to beat), MS the shock-conditioned
lambda, and MR the REVERSAL control -- the same three lambdas handed to the
wrong nights (calm gets the shock tercile's lambda and vice versa). The
decisive contrast is MS against M1; MR is what separates "the shock score
carries information" from "three knobs beat one knob by perturbation".

    python -m model.eval.shock_dispersion --selftest
    python -m model.eval.shock_dispersion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                    # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.dispersion_barriers import fit_lambda                        # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                   # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

PROD_H = 19
METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
HIGHER_BETTER = {"DSC"}
ARMS = ("M0", "M1", "MS", "MR")
CONTRASTS = (("MS", "M0"), ("M1", "M0"), ("MS", "M1"), ("MR", "M1"))


def shock_score(X) -> np.ndarray:
    """Recent against long-run log vol rate. NaN where either is missing."""
    return (X["har_6h"].to_numpy(np.float64)
            - X["har_22d"].to_numpy(np.float64))


def tercile_lambdas(shock_cal, rv_cal, sig_cal, sigm_cal, fallback: float):
    """Cut points and one lambda per tercile, all from CALIB.

    A tercile too thin for fit_lambda (it refuses below 50 rows) inherits the
    uniform fallback rather than a number fitted on a handful of nights.
    """
    ok = np.isfinite(shock_cal)
    q1, q2 = np.quantile(shock_cal[ok], [1 / 3, 2 / 3])
    bins = np.digitize(shock_cal, [q1, q2])          # 0, 1, 2
    lams = []
    for k in range(3):
        m = ok & (bins == k)
        lam = fit_lambda(rv_cal[m], sig_cal[m], sigm_cal[m])
        lams.append(float(lam) if np.isfinite(lam) else float(fallback))
    return (float(q1), float(q2)), lams


def lambda_array(shock_all, cuts, lams, fallback: float) -> np.ndarray:
    """One lambda per episode in the table, by its shock tercile."""
    bins = np.digitize(shock_all, list(cuts))
    out = np.asarray(lams, np.float64)[np.clip(bins, 0, 2)]
    return np.where(np.isfinite(shock_all), out, fallback)


def better(met, a, b):
    return a > b if met in HIGHER_BETTER else a < b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="shock-conditioned dispersion")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/shock_dispersion.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    shock = shock_score(X)
    n_family = len(CONTRASTS) * len(METRICS)
    alpha = 0.05 / n_family
    print(f"P4-shock-dispersion   production slice H={PROD_H}   seeds={a.seeds}")
    print(f"family {n_family} -> {100*(1-alpha):.3f}% intervals\n")

    rows = []
    for f in folds:
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if r0 is None:
            continue
        pe0 = r0["per_episode"]
        lam_u = fit_lambda(pe0["rv_cal"], pe0["sigma_cal"], pe0["sigma_mean_cal"])
        if not np.isfinite(lam_u):
            continue
        sc = shock[pe0["cal_idx"]]
        cuts, lams = tercile_lambdas(sc, pe0["rv_cal"], pe0["sigma_cal"],
                                     pe0["sigma_mean_cal"], lam_u)
        lam_arr = lambda_array(shock, cuts, lams, lam_u)
        lam_rev = lambda_array(shock, cuts, lams[::-1], lam_u)
        row = {"year": f["year"], "lam_uniform": float(lam_u),
               "lam_terciles": lams, "cuts": list(cuts),
               "q_M0": qlike_vec(pe0["rv"], pe0["sigma_mean"]),
               "bar_M0": barrier_cols(r0["rows"])}
        okf = True
        for arm, lam in (("M1", float(lam_u)), ("MS", lam_arr), ("MR", lam_rev)):
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, disp_lambda=lam)
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], pe0["test_idx"]):
                raise SystemExit("REFUSING: arms scored different episodes")
            if float(np.max(np.abs(pe1["sigma_med"] - pe0["sigma_med"]))) > 0:
                raise SystemExit("REFUSING: the width hook moved sigma_med")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
        if okf:
            rows.append(row)
            print(f"  fold {f['year']}: uniform {lam_u:.3f}   terciles "
                  f"calm {lams[0]:.3f} / mid {lams[1]:.3f} / shock {lams[2]:.3f}")

    if not rows:
        print("no complete folds"); return 1

    per_fold = {m: {k: [r[f"bar_{k}"].get(m) for r in rows]
                    for k in ARMS} for m in METRICS}
    print(f"\n{'metric':>9} " + " ".join(f"{k:>10}" for k in ARMS))
    for m in METRICS:
        print(f"{m:>9} " + " ".join(f"{np.mean(per_fold[m][k]):10.6f}"
                                    for k in ARMS))

    ci_out, res = {}, {c: {"better": [], "worse": []} for c in CONTRASTS}
    print(f"\n{'metric':>9} {'contrast':>9} {'delta':>12} {'CI (corrected)':>28} {'folds':>6}")
    for m in METRICS:
        sgn = 1.0 if m in HIGHER_BETTER else -1.0
        for c, o in CONTRASTS:
            d = sgn * (np.asarray(per_fold[m][c], float)
                       - np.asarray(per_fold[m][o], float))
            lo, hi = mean_ci(d, alpha=alpha)["ci95"]
            key = f"{m}_{c}_vs_{o}"
            ci_out[key] = {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
                           "n_better": int((d > 0).sum())}
            if lo > 0:
                res[(c, o)]["better"].append(m)
            if hi < 0:
                res[(c, o)]["worse"].append(m)
            print(f"{m:>9} {c+'-'+o:>9} {d.mean():+12.6f} "
                  f"[{lo:+12.6f}, {hi:+12.6f}] {int((d>0).sum()):>3}/{len(d)}")

    L = block_len_for(PROD_H, sum(len(r["q_M0"]) for r in rows))
    q0 = np.concatenate([r["q_M0"] for r in rows])
    print("\nper-episode QLIKE vs M0:")
    qci = {}
    for arm in ("M1", "MS", "MR"):
        qa = np.concatenate([r[f"q_{arm}"] for r in rows])
        dd = q0 - qa; g = np.isfinite(dd)
        lo, hi = mean_ci(dd[g], alpha=alpha, block_len=L)["ci95"]
        qci[arm] = {"pct": float(100 * np.nanmean(dd) / np.nanmean(q0)),
                    "ci95": [float(lo), float(hi)]}
        print(f"   {arm}: {qci[arm]['pct']:+6.2f}%  CI [{lo:+.5f}, {hi:+.5f}]")

    ms_m1, mr_m1 = res[("MS", "M1")], res[("MR", "M1")]
    shared = sorted(set(ms_m1["better"]) & set(mr_m1["better"]))
    met = len(ms_m1["better"]) >= 2 and not ms_m1["worse"] and not shared
    print("\n--- pre-registered rule: MS beats M1 on >= 2 metrics, loses on none,")
    print("    and the reversal MR beats M1 on none of the metrics MS wins")
    print(f"   MS vs M1 better on {ms_m1['better'] or 'nothing'}, "
          f"worse on {ms_m1['worse'] or 'nothing'}")
    print(f"   MR vs M1 better on {mr_m1['better'] or 'nothing'}, "
          f"worse on {mr_m1['worse'] or 'nothing'}")
    print(f"   -> {'MET' if met else 'NOT MET'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_family": n_family, "alpha": alpha,
        "folds": [{k: v for k, v in r.items() if not k.startswith(("q_", "bar_"))}
                  for r in rows],
        "barriers": {m: {k: float(np.mean(v)) for k, v in d.items()}
                     for m, d in per_fold.items()},
        "ci": ci_out, "qlike": qci,
        "ms_vs_m1": ms_m1, "mr_vs_m1": mr_m1, "rule_met": bool(met)}, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(1)
    n = 6000
    shock = rng.normal(0, 1, n)
    # planted: calm nights over-dispersed (true lambda 0.7), shock nights
    # correctly dispersed (true lambda 1.0). The tercile fit must recover the
    # gradient; the uniform fit must land in between.
    med = np.exp(np.full(n, -4.0))
    s_imp = 0.4
    true_s = np.where(shock < np.quantile(shock, 1/3), 0.7 * s_imp,
                      np.where(shock > np.quantile(shock, 2/3), 1.0 * s_imp,
                               0.85 * s_imp))
    rv = med * np.exp(rng.normal(0, 1, n) * true_s)
    sigm = med * np.exp(s_imp ** 2)
    cuts, lams = tercile_lambdas(shock, rv, med, sigm, fallback=1.0)
    ok.append(("calm tercile narrows most", lams[0] < lams[1] < lams[2]))
    ok.append(("calm lambda near the planted 0.7", abs(lams[0] - 0.7) < 0.06))
    ok.append(("shock lambda near the planted 1.0", abs(lams[2] - 1.0) < 0.06))
    arr = lambda_array(shock, cuts, lams, 1.0)
    ok.append(("each episode gets its tercile's lambda",
               np.allclose(arr[shock < cuts[0]], lams[0])
               and np.allclose(arr[shock > cuts[1]], lams[2])))
    arr2 = lambda_array(np.array([np.nan, 0.0]), cuts, lams, 0.9)
    ok.append(("a missing shock score takes the uniform fallback", arr2[0] == 0.9))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
