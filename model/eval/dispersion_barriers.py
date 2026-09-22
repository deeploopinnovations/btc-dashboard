"""
eval/dispersion_barriers.py
=====================================================================
P3-dispersion-barriers-v2: the model is over-dispersed by 11-28%. Correcting
that costs QLIKE. Does it help the thing the dispersion actually builds?

THE SPLIT THAT MAKES THIS WORTH RUNNING AFTER A NEGATIVE SCREEN

QLIKE asks that sigma^2 equal E[RV^2]. `P3-dispersion` showed that at H=6 and
H=24 it is satisfied by a dispersion error and a level error CANCELLING --
0.8719 x 1.1371 = 0.9915 at H=24 -- and `P3-dispersion-screen` showed that
removing either one alone therefore makes QLIKE worse, by up to 5.5%.

The barrier battery asks something different: that the DISTRIBUTION be right.
A cancellation in a point forecast does nothing for a probability curve. The
committee builds every curve from `sigma_atoms`, so an 11-28% width error is an
error in the object the product is made of. That is the open question, and a
negative QLIKE screen cannot answer it.

THE ARMS, AND WHY M3 IS THE EXPERIMENT

    M0  shipped, disp_lambda = 1.0
    M1  disp_lambda fitted per fold on CALIB as s_realised / s_implied
    M2  disp_lambda fixed at the sample-median fitted value -- the control for
        "does fitting it per fold do anything a constant does not" (R82)
    M3  disp_lambda = 2 - lambda_fitted: the MIRROR, same magnitude of change
        in the opposite direction

M3 is what licenses the inference. `P2-mean-level` moved the LEVEL and found
all six barrier metrics degraded by about 20% -- with its SHUFFLED arm
degrading them identically, which proved the damage belonged to moving the
level at all rather than to the correction's content. If M1 and M3 improve or
degrade together here, the same conclusion follows for the width and the atoms
are not safe to move in any direction.

An earlier registration specified M3 as "disp_lambda permuted across episodes".
That is incoherent for a per-fold SCALAR -- permuting a constant leaves it
unchanged, so M3 would have been bit-identical to M1 while looking like a
placebo. The control's form was copied from a prior experiment instead of being
derived from what this arm varies. Amended before any arm ran.

    python -m model.eval.dispersion_barriers --selftest
    python -m model.eval.dispersion_barriers
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
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                   # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

PROD_H = 19
ARMS = ("M1", "M2", "M3")
METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
# DSC is a discrimination score: higher is better. Every other metric here is
# a loss. Getting this backwards would report every result inverted for one
# column, so the direction lives in one place.
HIGHER_BETTER = {"DSC"}
N_FAMILY = len(ARMS) * len(METRICS)


def fit_lambda(rv_cal, sig_cal, sig_mean_cal) -> float:
    """s_realised / s_implied on the CALIB slice.

    s_implied comes from the model's own two scalars (sigma_mean/sigma_med =
    exp(s^2)); s_realised is the spread of outcomes about the predicted median.
    Both are calib-only, so the fold's test slice is untouched.
    """
    rv = np.asarray(rv_cal, np.float64)
    med = np.asarray(sig_cal, np.float64)
    mn = np.asarray(sig_mean_cal, np.float64)
    ok = (np.isfinite(rv) & np.isfinite(med) & np.isfinite(mn)
          & (rv > 0) & (med > 0) & (mn > 0))
    if ok.sum() < 50:
        return float("nan")
    s_imp = float(np.nanmean(np.sqrt(np.maximum(
        np.log(mn[ok] / med[ok]), 0.0))))
    s_real = float(np.nanstd(np.log(rv[ok] / med[ok]), ddof=1))
    if not (s_imp > 0):
        return float("nan")
    return s_real / s_imp


def better(metric: str, arm_val: float, ref_val: float) -> bool:
    return arm_val > ref_val if metric in HIGHER_BETTER else arm_val < ref_val


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(5)
    n = 20000

    # 1-3. fit_lambda on planted data: a correctly-dispersed model returns 1,
    #      an over-dispersed one returns < 1 (narrow it), under-dispersed > 1.
    mu, s = -4.0, 0.40
    med = np.full(n, np.exp(mu))
    rv = np.exp(rng.normal(mu, s, n))
    ok.append(("correctly dispersed -> lambda ~ 1",
               abs(fit_lambda(rv, med, med * np.exp(s * s)) - 1.0) < 0.03))
    ok.append(("over-dispersed -> lambda < 1",
               fit_lambda(rv, med, med * np.exp((1.5 * s) ** 2)) < 0.75))
    ok.append(("under-dispersed -> lambda > 1",
               fit_lambda(rv, med, med * np.exp((0.6 * s) ** 2)) > 1.4))

    # 4. Refuses rather than inventing a lambda from nothing.
    ok.append(("refuses below 50 rows",
               not np.isfinite(fit_lambda(rv[:10], med[:10], med[:10]))))

    # 5-6. The mirror must be symmetric about 1 and preserve the magnitude.
    for lam in (0.8, 1.2):
        mirror = 2.0 - lam
        ok.append((f"mirror of {lam} is equidistant from 1",
                   abs((mirror - 1.0) + (lam - 1.0)) < 1e-12))

    # 7-8. Metric direction, which is the one thing that silently inverts a
    #      whole column if wrong.
    ok.append(("DSC higher is better", better("DSC", 0.9, 0.8)))
    ok.append(("brier lower is better", better("brier", 0.1, 0.2)))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dispersion vs the barrier battery")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/dispersion_barriers.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    print(f"P3-dispersion-barriers-v2   production slice H={PROD_H}  "
          f"seeds={a.seeds}")
    print(f"family {N_FAMILY} ({len(ARMS)} arms x {len(METRICS)} metrics) -> "
          f"{100*(1-alpha):.3f}% intervals\n")

    # PASS ONE: the reference, which also supplies each fold's calib slice and
    # therefore its lambda. lambda cannot be known before the model runs, so
    # two passes is exact rather than approximate.
    acc, lams = [], []
    for f in folds:
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if r0 is None:
            continue
        pe0 = r0["per_episode"]
        lam = fit_lambda(pe0["rv_cal"], pe0["sigma_cal"], pe0["sigma_mean_cal"])
        if not np.isfinite(lam):
            print(f"  fold {f['year']}: no lambda, skipped"); continue
        acc.append({"year": f["year"], "lam": float(lam), "r0": r0})
        lams.append(float(lam))
        print(f"  fold {f['year']}  lambda {lam:.4f}  "
              f"(implied spread scaled to {100*lam:.1f}% of itself)")
    if not acc:
        print("no usable folds"); return 1
    k_const = float(np.median(lams))
    print(f"\nconstant arm uses the sample-median lambda = {k_const:.4f}\n")

    # PASS TWO
    rows = []
    for rec in acc:
        f = next(x for x in folds if x["year"] == rec["year"])
        r0, lam = rec["r0"], rec["lam"]
        pe0 = r0["per_episode"]
        row = {"year": rec["year"], "lam": lam,
               "q_M0": qlike_vec(pe0["rv"], pe0["sigma_mean"]),
               "bar_M0": barrier_cols(r0["rows"])}
        lam_of = {"M1": lam, "M2": k_const, "M3": 2.0 - lam}
        okf = True
        for arm in ARMS:
            r1 = run_fold(ep, X, f, a.hidden, a.seeds,
                          disp_lambda=lam_of[arm])
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], pe0["test_idx"]):
                raise SystemExit(
                    f"REFUSING: fold {rec['year']} arm {arm} scored different "
                    f"episodes than the reference. The contrast is not paired.")
            # The hook must leave the median EXACTLY alone, or a level change
            # is being scored as a dispersion effect.
            d = float(np.max(np.abs(pe1["sigma_med"] - pe0["sigma_med"])))
            if d > 0.0:
                raise SystemExit(
                    f"REFUSING: fold {rec['year']} arm {arm} moved sigma_med "
                    f"by {d:.3e}. disp_lambda must scale width only.")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
        if okf:
            rows.append(row)
            print(f"  fold {rec['year']}: all arms scored")

    if not rows:
        print("no complete folds"); return 1

    print(f"\n{'metric':>9} {'M0':>10} " + " ".join(f"{m:>10}" for m in ARMS))
    bar = {}
    for met in METRICS:
        vals = {}
        for arm in ("M0",) + ARMS:
            v = [r[f"bar_{arm}"].get(met) for r in rows
                 if met in r.get(f"bar_{arm}", {})]
            vals[arm] = float(np.mean(v)) if v else float("nan")
        bar[met] = vals
        print(f"{met:>9} " + " ".join(f"{vals[k]:10.6f}" for k in ("M0",) + ARMS))

    print("\npaired per-episode QLIKE (reported, NOT a pass condition):")
    L = block_len_for(PROD_H, sum(len(r["q_M0"]) for r in rows))
    q0 = np.concatenate([r["q_M0"] for r in rows])
    qci = {}
    for arm in ARMS:
        qa = np.concatenate([r[f"q_{arm}"] for r in rows])
        d = q0 - qa
        g = np.isfinite(d)
        ci = mean_ci(d[g], alpha=alpha, block_len=L)["ci95"]
        qci[arm] = [float(v) for v in ci]
        print(f"   {arm}: {100*np.nanmean(d)/np.nanmean(q0):+6.2f}%  "
              f"CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")

    print(f"\n--- pre-registered rule ---")
    verdicts = {}
    for arm in ARMS:
        wins = [m for m in METRICS
                if np.isfinite(bar[m][arm]) and better(m, bar[m][arm], bar[m]["M0"])]
        verdicts[arm] = {"metrics_better": wins, "n_better": len(wins)}
        print(f"   {arm}: {len(wins)}/6 metrics better than shipped  {wins}")
    m1, m3 = verdicts["M1"]["n_better"], verdicts["M3"]["n_better"]
    print(f"\n   M1 majority-better: {m1 >= 4}   "
          f"MIRROR M3 majority-better: {m3 >= 4}")
    if m1 >= 4 and m3 >= 4:
        print("   -> BOTH DIRECTIONS HELP: the gain is perturbing the atoms at "
              "all,\n      not correcting the dispersion. Same shape as "
              "P2-mean-level's level result.")
    elif m1 >= 4 and m3 < 4:
        print("   -> the correction's DIRECTION matters; width is a different "
              "intervention\n      class from level. Candidate, pending the "
              "M1-vs-M2 separation below.")
    else:
        print("   -> the dispersion correction does not improve the battery.")
    print(f"   M1 vs M2 (is the per-fold fit worth anything over a constant?): "
          f"{verdicts['M1']['n_better']} vs {verdicts['M2']['n_better']} metrics better")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"n_family": N_FAMILY, "alpha": alpha, "prod_H": PROD_H,
         "k_const": k_const, "lambdas": {str(r["year"]): r["lam"] for r in rows},
         "barriers": bar, "qlike_ci": qci, "verdicts": verdicts}, indent=2,
        default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
