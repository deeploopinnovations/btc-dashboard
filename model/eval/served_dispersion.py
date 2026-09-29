"""
eval/served_dispersion.py
=====================================================================
P4-served-dispersion: the dispersion correction, tested on the object that
would actually be served.

WHY THE EARLIER TESTS ARE NOT ENOUGH

Every dispersion result in this project (P3-dispersion-barriers-result,
P3-dispersion-deployable-result) was scored against the benchmark's M0, which
has NEITHER serving's trailing level factor NOR the clock-aware anchor.
P4-hour-anchor-result showed that M0 over-forecasts at the 17:00 product
anchor, that serving's factor does not remove it, and that any arm LOWERING the
17:00 level wins on the calibration side for that reason alone. Narrowing the
predictive distribution cuts far-barrier touch probabilities much as a level
cut does -- so part of the dispersion correction's win may have been the same
effect, and would vanish on a correctly levelled base.

THE BASE

    B = clock-aware anchor (P4-hour-anchor-result) + serving's trailing factor
        computed from B's own forecasts, exactly as eval/hour_anchor.py builds
        its As arm.

THE ARMS, all on top of B

    M1   per-fold lambda, fit_lambda on B's own CALIB forecasts (all hours, H=19)
    MT   the DEPLOYABLE lambda: per night, fit_lambda on B's own settled 17:00
         forecasts over the trailing 182 days (eval/trailing_dispersion.py)
    MTm  MT's mirror, 2 - lambda_t
    MS   shock-conditioned: one lambda per calib tercile of har_6h - har_22d
         (eval/shock_dispersion.py), on B's forecasts

B's forecasts at any episode are M0's times exp(B's log shift) -- exact, because
run_fold asserts post_shift_fn moves the median by what it asks, and a level
shift leaves the atoms' spread about the median unchanged. So lambda is fitted
on the base it will be applied to without an extra run.

    python -m model.eval.served_dispersion --selftest
    python -m model.eval.served_dispersion
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
from eval.hour_anchor import (FAC_STRIDE_H, PROD_A, PROD_H,            # noqa: E402
                              fold_anchors, fold_tests, served_log_factor)
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.shock_dispersion import (lambda_array, shock_score,          # noqa: E402
                                   tercile_lambdas)
from eval.trailing_dispersion import trailing_lambdas                  # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                   # noqa: E402
from noctua import infer as I                                          # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
HIGHER_BETTER = {"DSC"}
ARMS = ("B", "M1", "MT", "MTm", "MS")
CONTRASTS = (("M1", "B"), ("MT", "B"), ("MTm", "B"), ("MS", "M1"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dispersion on the served base")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/served_dispersion.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    prod = S.production_mask(ep)
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = {(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}
    fac_mask = at19 & np.isin(ah, sorted(fac_hours))
    shock = shock_score(X)
    w = I.BLEND_W
    n_family = len(CONTRASTS) * len(METRICS)
    alpha = 0.05 / n_family
    print(f"P4-served-dispersion   base = clock-aware anchor + served factor   "
          f"seeds={a.seeds}")
    print(f"family {n_family} -> {100*(1-alpha):.3f}% intervals\n")

    rows = []
    for f in folds:
        anc = fold_anchors(ep, X, f)
        sh_A = np.nan_to_num((1 - w) * (anc["A"] - anc["lhc"]))
        rF = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=fac_mask)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if rF is None or r0 is None:
            continue
        peF, pe0 = rF["per_episode"], r0["per_episode"]
        ci, ti = pe0["cal_idx"], pe0["test_idx"]
        # B's served factor, from B's own uncorrected forecasts
        h_idx = np.concatenate([ci, peF["test_idx"]])
        h_rv = np.concatenate([pe0["rv_cal"], peF["rv"]])
        h_sig = np.concatenate([pe0["sigma_cal"], peF["sigma_med"]]) * np.exp(sh_A[h_idx])
        lf = served_log_factor(ts_all[ti], ts_all[h_idx], h_rv, h_sig)
        base = sh_A.copy(); base[ti] += lf
        fn = (lambda mask, _mt, _s=base: _s[mask])

        # B's forecasts (median and mean move together under a level shift)
        g_cal = np.exp(base[ci]); g_te = np.exp(base[ti])
        med_c, mn_c = pe0["sigma_cal"] * g_cal, pe0["sigma_mean_cal"] * g_cal
        med_t, mn_t = pe0["sigma_med"] * g_te, pe0["sigma_mean"] * g_te
        lam1 = fit_lambda(pe0["rv_cal"], med_c, mn_c)
        if not np.isfinite(lam1):
            continue
        cp = prod[ci]
        lam_t, used, n_fb = trailing_lambdas(
            np.concatenate([ts_all[ci][cp], ts_all[ti]]),
            np.concatenate([pe0["rv_cal"][cp], pe0["rv"]]),
            np.concatenate([med_c[cp], med_t]), np.concatenate([mn_c[cp], mn_t]),
            ts_all[ti], PROD_H, float(lam1))
        mt = np.ones(len(ep)); mt[ti] = lam_t
        mtm = np.ones(len(ep)); mtm[ti] = 2.0 - lam_t
        cuts, lams = tercile_lambdas(shock[ci], pe0["rv_cal"], med_c, mn_c, lam1)
        ms = lambda_array(shock, cuts, lams, lam1)

        row = {"year": f["year"], "lam_m1": float(lam1), "lam_terciles": lams,
               "lam_t_mean": float(lam_t.mean()), "lam_t_range":
               [float(lam_t.min()), float(lam_t.max())], "n_fallback": int(n_fb),
               "logfac_mean": float(lf.mean())}
        okf = True
        for arm, lam in (("B", 1.0), ("M1", float(lam1)), ("MT", mt),
                         ("MTm", mtm), ("MS", ms)):
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, post_shift_fn=fn,
                          disp_lambda=lam)
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
            row[f"medchk_{arm}"] = float(np.max(np.abs(pe1["sigma_med"] - med_t)
                                                / med_t))
        if not okf:
            continue
        # every arm must serve B's median: the width hook cannot move it, and
        # B's median must be M0's times exp(shift) -- the algebra lambda used
        worst = max(row[f"medchk_{k}"] for k in ARMS)
        if worst > 1e-9:
            raise SystemExit(f"REFUSING: served median differs from B's by "
                             f"{worst:.2e} -- lambda was fitted on another base")
        rows.append(row)
        print(f"  fold {f['year']}: M1 {lam1:.3f}  trailing {lam_t.mean():.3f} "
              f"[{lam_t.min():.3f}, {lam_t.max():.3f}]  terciles "
              f"{lams[0]:.3f}/{lams[1]:.3f}/{lams[2]:.3f}  "
              f"mean log factor {lf.mean():+.3f}", flush=True)

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
            ft = fold_tests(d, alpha)
            ci_out[f"{m}_{c}_vs_{o}"] = {"delta": float(d.mean()),
                                         "ci95": [float(lo), float(hi)],
                                         "n_better": int((d > 0).sum()), **ft}
            if lo > 0:
                res[(c, o)]["better"].append(m)
            if hi < 0:
                res[(c, o)]["worse"].append(m)
            print(f"{m:>9} {c+'-'+o:>9} {d.mean():+12.6f} "
                  f"[{lo:+12.6f}, {hi:+12.6f}] {int((d>0).sum()):>3}/{len(d)}"
                  f"   t [{ft['t_ci'][0]:+.6f}, {ft['t_ci'][1]:+.6f}]"
                  f"  p_flip {ft['p_flip']:.3f}")

    L = block_len_for(PROD_H, sum(len(r["q_B"]) for r in rows))
    print("\nper-episode QLIKE (sigma_mean):")
    qci = {}
    for c, o in CONTRASTS:
        qo = np.concatenate([r[f"q_{o}"] for r in rows])
        qc = np.concatenate([r[f"q_{c}"] for r in rows])
        dd = qo - qc; g = np.isfinite(dd)
        lo, hi = mean_ci(dd[g], alpha=alpha, block_len=L)["ci95"]
        qci[f"{c}_vs_{o}"] = {"pct": float(100 * np.nanmean(dd) / np.nanmean(qo)),
                              "ci95": [float(lo), float(hi)]}
        print(f"   {c:>3} vs {o:<3}: {qci[f'{c}_vs_{o}']['pct']:+6.2f}%  "
              f"CI [{lo:+.5f}, {hi:+.5f}]")

    mt_, mtm_, ms_ = res[("MT", "B")], res[("MTm", "B")], res[("MS", "M1")]
    dsc_pt = ci_out["DSC_MT_vs_B"]["delta"]
    q_ok = qci["MT_vs_B"]["ci95"][0] > 0
    met_mt = (len(mt_["better"]) >= 4 and not mt_["worse"]
              and len(mtm_["better"]) < 4 and dsc_pt >= 0 and q_ok)
    met_ms = len(ms_["better"]) >= 2 and not ms_["worse"]
    print("\n--- rule 1 (deployable dispersion on the served base): MT clears >= 4 of 6")
    print("    vs B, worse on none; MTm clears < 4; MT's DSC point estimate >= 0;")
    print("    MT's per-episode QLIKE interval vs B excludes zero favourably")
    print(f"   MT  vs B: better {mt_['better'] or 'nothing'}, worse {mt_['worse'] or 'nothing'}")
    print(f"   MTm vs B: better {mtm_['better'] or 'nothing'}, worse {mtm_['worse'] or 'nothing'}")
    print(f"   DSC point {dsc_pt:+.6f}   QLIKE clears {q_ok}   ->  "
          f"{'MET' if met_mt else 'NOT MET'}")
    print("--- rule 2 (shock conditioning): MS beats M1 on >= 2 metrics, worse on none")
    print(f"   MS vs M1: better {ms_['better'] or 'nothing'}, worse {ms_['worse'] or 'nothing'}"
          f"  ->  {'MET' if met_ms else 'NOT MET'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_family": n_family, "alpha": alpha,
        "folds": [{**{k: v for k, v in r.items()
                      if not k.startswith(("q_", "bar_", "medchk_"))},
                   "bar": {k[4:]: v for k, v in r.items() if k.startswith("bar_")},
                   "qlike": {k[2:]: float(np.nanmean(v)) for k, v in r.items()
                             if k.startswith("q_")}} for r in rows],
        "ci": ci_out, "qlike": qci,
        "results": {f"{c}_vs_{o}": v for (c, o), v in res.items()},
        "rule_mt_met": bool(met_mt), "rule_ms_met": bool(met_ms)},
        indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    """The one identity this file adds: under a level shift g, fit_lambda on
    (rv, med*g, mean*g) equals fit_lambda on (rv / g, med, mean) -- lambda is
    fitted on the base it is applied to, not on M0."""
    ok = []
    rng = np.random.default_rng(11)
    n = 3000
    med = np.exp(rng.normal(-4, 0.3, n))
    mean = med * np.exp(0.4 ** 2)
    rv = med * np.exp(rng.normal(0, 0.3, n))
    g = np.exp(rng.normal(-0.05, 0.03, n))            # a varying level shift
    a1 = fit_lambda(rv, med * g, mean * g)
    a2 = fit_lambda(rv / g, med, mean)
    ok.append(("lambda on the shifted base = lambda on de-shifted outcomes",
               abs(a1 - a2) < 1e-12))
    a0 = fit_lambda(rv, med, mean)
    ok.append(("a VARYING shift changes lambda (so fitting on M0 would be wrong)",
               abs(a1 - a0) > 1e-4))
    gc = np.full(n, 0.93)
    ok.append(("a CONSTANT shift leaves lambda unchanged",
               abs(fit_lambda(rv, med * gc, mean * gc) - a0) < 1e-12))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g_ for _, g_ in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
