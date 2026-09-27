"""
eval/trailing_dispersion.py
=====================================================================
P4-trailing-dispersion: the dispersion correction in the only form serving
can actually run.

WHERE THIS STANDS

P3-dispersion-deployable-result: the per-fold lambda (M1), fitted on each
fold's six-month calib slice, clears four of six barrier metrics AND improves
discrimination (DSC), while its mirror damages the same four. The causal
CONSTANT (M4, the median of past folds' lambdas) keeps the calibration-side
gains and loses the DSC gain. So the thing worth shipping is M1 -- and M1 is a
number fitted once a year on a slice serving never sees as such. Serving has
no "calib slice"; it has the model's own settled forecasts.

WHAT IS BUILT

For every production test episode anchored at t (17:00 UTC, H = 19):

    window  = production episodes (17:00, H = 19) of this fold's calib and
              test slices whose 19-hour window CLOSED in (t - 182d, t]
    lambda_t = fit_lambda(window)     -- the estimator M1 uses, on the model's
                                         own M0 forecasts and realised RV

182 days is the calib slice's own length, so the only change against M1 is
that the window SLIDES and holds only what serving itself would have logged
(one 17:00 forecast per day). A window thinner than fit_lambda's 50-row floor
falls back to the fold's M1 lambda, and the count is reported.

Arms: M0 shipped, M1 per-fold calib lambda, MT trailing lambda, MTm its MIRROR
(2 - lambda_t: same magnitude of change, opposite direction). If MT and MTm
move the battery the same way the effect is perturbation, not correction.

    python -m model.eval.trailing_dispersion --selftest
    python -m model.eval.trailing_dispersion
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

HOUR = 3600
PROD_H = 19
WINDOW_H = 182 * 24
METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
HIGHER_BETTER = {"DSC"}
ARMS = ("M0", "M1", "MT", "MTm")
CONTRASTS = (("MT", "M0"), ("MTm", "M0"), ("MT", "M1"))


def trailing_lambdas(ts_hist, rv_hist, med_hist, mean_hist, ts_query,
                     H: int, fallback: float, window_h: int = WINDOW_H):
    """One lambda per query anchor from history rows settled in the window.

    A history row is usable at t only once its window has closed:
    anchor + H hours <= t. Returns (lambdas, n_rows_used, n_fallback).
    """
    settle = np.asarray(ts_hist, np.int64) + int(H) * HOUR
    order = np.argsort(settle, kind="stable")
    settle = settle[order]
    rv, med, mn = (np.asarray(a, np.float64)[order]
                   for a in (rv_hist, med_hist, mean_hist))
    out = np.empty(len(ts_query)); used = np.empty(len(ts_query), np.int64)
    n_fb = 0
    for i, t in enumerate(np.asarray(ts_query, np.int64)):
        hi = np.searchsorted(settle, t, side="right")          # settle <= t
        lo = np.searchsorted(settle, t - window_h * HOUR, side="right")
        lam = fit_lambda(rv[lo:hi], med[lo:hi], mn[lo:hi])
        used[i] = hi - lo
        if not np.isfinite(lam):
            lam = fallback; n_fb += 1
        out[i] = lam
    return out, used, n_fb


def better(met, a, b):
    return a > b if met in HIGHER_BETTER else a < b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="trailing-window dispersion")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/trailing_dispersion.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    prod = S.production_mask(ep)
    n_family = len(CONTRASTS) * len(METRICS)
    alpha = 0.05 / n_family
    print(f"P4-trailing-dispersion   production slice H={PROD_H}   "
          f"seeds={a.seeds}   window {WINDOW_H // 24}d")
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
        # history = the model's own M0 forecasts at the served configuration
        cp = prod[pe0["cal_idx"]]
        ti = pe0["test_idx"]
        ts_hist = np.concatenate([ts_all[pe0["cal_idx"]][cp], ts_all[ti]])
        rv_hist = np.concatenate([pe0["rv_cal"][cp], pe0["rv"]])
        md_hist = np.concatenate([pe0["sigma_cal"][cp], pe0["sigma_med"]])
        mn_hist = np.concatenate([pe0["sigma_mean_cal"][cp], pe0["sigma_mean"]])
        lam_t, used, n_fb = trailing_lambdas(ts_hist, rv_hist, md_hist, mn_hist,
                                             ts_all[ti], PROD_H, float(lam_u))
        lam_arr = np.ones(len(ep)); lam_arr[ti] = lam_t
        mir_arr = np.ones(len(ep)); mir_arr[ti] = 2.0 - lam_t
        row = {"year": f["year"], "lam_m1": float(lam_u),
               "lam_t_mean": float(lam_t.mean()), "lam_t_min": float(lam_t.min()),
               "lam_t_max": float(lam_t.max()), "window_rows_min": int(used.min()),
               "n_fallback": int(n_fb),
               "q_M0": qlike_vec(pe0["rv"], pe0["sigma_mean"]),
               "bar_M0": barrier_cols(r0["rows"])}
        okf = True
        for arm, lam in (("M1", float(lam_u)), ("MT", lam_arr), ("MTm", mir_arr)):
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, disp_lambda=lam)
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            if float(np.max(np.abs(pe1["sigma_med"] - pe0["sigma_med"]))) > 0:
                raise SystemExit("REFUSING: the width hook moved sigma_med")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
        if okf:
            rows.append(row)
            print(f"  fold {f['year']}: M1 {lam_u:.3f}   trailing mean "
                  f"{lam_t.mean():.3f} [{lam_t.min():.3f}, {lam_t.max():.3f}]   "
                  f"min window {used.min()} rows, {n_fb} fallbacks", flush=True)

    if not rows:
        print("no complete folds"); return 1

    per_fold = {m: {k: [r[f"bar_{k}"].get(m) for r in rows] for k in ARMS}
                for m in METRICS}
    print(f"\n{'metric':>9} " + " ".join(f"{k:>10}" for k in ARMS))
    for m in METRICS:
        print(f"{m:>9} " + " ".join(f"{np.mean(per_fold[m][k]):10.6f}"
                                    for k in ARMS))

    ci_out, res = {}, {c: {"better": [], "worse": []} for c in CONTRASTS}
    print(f"\n{'metric':>9} {'contrast':>9} {'delta':>12} "
          f"{'CI (corrected)':>28} {'folds':>6}")
    for m in METRICS:
        sgn = 1.0 if m in HIGHER_BETTER else -1.0
        for c, o in CONTRASTS:
            d = sgn * (np.asarray(per_fold[m][c], float)
                       - np.asarray(per_fold[m][o], float))
            lo, hi = mean_ci(d, alpha=alpha)["ci95"]
            ci_out[f"{m}_{c}_vs_{o}"] = {"delta": float(d.mean()),
                                         "ci95": [float(lo), float(hi)],
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
    for arm in ("M1", "MT", "MTm"):
        qa = np.concatenate([r[f"q_{arm}"] for r in rows])
        dd = q0 - qa; g = np.isfinite(dd)
        lo, hi = mean_ci(dd[g], alpha=alpha, block_len=L)["ci95"]
        qci[arm] = {"pct": float(100 * np.nanmean(dd) / np.nanmean(q0)),
                    "ci95": [float(lo), float(hi)]}
        print(f"   {arm:>3}: {qci[arm]['pct']:+6.2f}%  CI [{lo:+.5f}, {hi:+.5f}]")

    mt, mm, mt1 = res[("MT", "M0")], res[("MTm", "M0")], res[("MT", "M1")]
    dsc_pt = ci_out["DSC_MT_vs_M0"]["delta"]
    met = (len(mt["better"]) >= 4 and len(mm["better"]) < 4
           and not mt["worse"] and not mt1["worse"] and dsc_pt >= 0)
    print("\n--- pre-registered rule: MT clears >= 4 of 6 vs M0, worse on none;")
    print("    mirror MTm clears < 4; MT significantly worse than M1 on none;")
    print("    MT's DSC point estimate vs M0 is not negative")
    print(f"   MT  vs M0: better {mt['better'] or 'nothing'}, worse {mt['worse'] or 'nothing'}")
    print(f"   MTm vs M0: better {mm['better'] or 'nothing'}, worse {mm['worse'] or 'nothing'}")
    print(f"   MT  vs M1: better {mt1['better'] or 'nothing'}, worse {mt1['worse'] or 'nothing'}")
    print(f"   MT DSC vs M0 point estimate {dsc_pt:+.6f}")
    print(f"   -> {'MET' if met else 'NOT MET'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_family": n_family, "alpha": alpha, "window_h": WINDOW_H,
        "folds": [{k: v for k, v in r.items() if not k.startswith(("q_", "bar_"))}
                  for r in rows],
        "barriers": {m: {k: float(np.mean(v)) for k, v in d.items()}
                     for m, d in per_fold.items()},
        "ci": ci_out, "qlike": qci,
        "results": {f"{c}_vs_{o}": v for (c, o), v in res.items()},
        "rule_met": bool(met)}, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(5)
    n = 900
    ts = np.arange(n, dtype=np.int64) * 24 * HOUR        # one anchor a day
    med = np.full(n, np.exp(-4.0))
    s_imp = 0.4
    mean = med * np.exp(s_imp ** 2)
    # planted: over-dispersed by 0.7 for the first 450 days, correct after
    true = np.where(np.arange(n) < 450, 0.7, 1.0) * s_imp
    rv = med * np.exp(rng.normal(0, 1, n) * true)
    q = ts[200:]
    lam, used, n_fb = trailing_lambdas(ts, rv, med, mean, q, H=19, fallback=0.5)
    early = lam[(q >= 300 * 24 * HOUR) & (q < 440 * 24 * HOUR)]
    late = lam[q >= 700 * 24 * HOUR]
    ok.append(("tracks the planted 0.7 regime", abs(early.mean() - 0.7) < 0.08))
    ok.append(("tracks the switch to 1.0 once the window has turned over",
               abs(late.mean() - 1.0) < 0.08))
    ok.append(("window holds about 182 daily rows",
               int(np.median(used)) in range(179, 184)))
    # causality: a query at t never uses a row whose window closes after t
    lam1, used1, _ = trailing_lambdas(ts, rv, med, mean, np.array([100 * 24 * HOUR]),
                                      H=19, fallback=0.5)
    # rows 0..99 have anchor + 19h <= day 100 -> rows 0..99 only (row 100 not)
    ok.append(("a label is used only after its window closes", used1[0] == 100))
    lam2, _, fb2 = trailing_lambdas(ts, rv, med, mean, np.array([10 * 24 * HOUR]),
                                    H=19, fallback=0.5)
    ok.append(("a thin window falls back and is counted",
               lam2[0] == 0.5 and fb2 == 1))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
