"""
eval/simple_vs_noctua.py
=====================================================================
P4-simple-vs-noctua: does the neural network earn its place in the PRODUCT,
once the simple model it is compared with is given the same clock and the same
serving factor?

The standing brief: "If the evidence shows that a simple HAR/GARCH/teacher
ensemble is superior to a neural NOCTUA V2, report that result and do not
preserve NOCTUA complexity for branding reasons." BENCHMARK 2 already found the
plain Log-HAR + Gaussian first-passage baseline AHEAD of NOCTUA on barrier
discrimination (DSC 0.008707 vs 0.008178), behind on calibration. Since then
the anchor has been given the clock (P4-hour-anchor-result) -- and that anchor
is 75% of NOCTUA's own level. The fair comparison now is

    N_As   NOCTUA as it would be served with HOUR_ANCHOR on: network + blend
           + committee, clock-aware anchor, serving's trailing factor
    G_As   NO NETWORK: sigma = the clock-aware Log-HAR anchor times serving's
           trailing factor computed from ITS OWN forecasts, barrier curves from
           the Gaussian first-passage law (the benchmark's log_har_gauss)
    G_M0s  the same without the clock (reference)

G arms need no training: their curves are built offline by eval/product_score
through the SAME arithmetic run_fold uses, and the script REFUSES to run unless
(1) product_score reproduces N_As's battery from its saved curves and
(2) it reproduces run_fold's own log_har_gauss row from the M0 run, both to
1e-10.

Brier, log score, pinball and CRPS are scored PER EPISODE with a moving-block
bootstrap over the pooled production nights (R88); DSC and MCB are fold-level
with the t-interval and sign-flip p.

    python -m model.eval.simple_vs_noctua
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
from eval.hour_anchor import (FAC_STRIDE_H, PROD_A, PROD_H,            # noqa: E402
                              fold_anchors, fold_tests, served_log_factor)
from eval.product_score import battery, gaussian_curves, per_episode   # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for                              # noqa: E402
from noctua import infer as I                                          # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

EP_METRICS = ("brier", "logs", "pinball", "crps")        # per-episode
FOLD_METRICS = ("DSC", "MCB")                            # fold-level only
HIGHER_BETTER = {"DSC"}
ARMS = ("N_As", "G_As", "G_M0s")
CONTRASTS = (("N_As", "G_As"), ("N_As", "G_M0s"), ("G_As", "G_M0s"))
TOL = 1e-10


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="network vs no network, product")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/simple_vs_noctua.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = {(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}
    fac_mask = at19 & np.isin(ah, sorted(fac_hours))
    w = I.BLEND_W
    n_family = len(CONTRASTS) * (len(EP_METRICS) + len(FOLD_METRICS))
    alpha = 0.05 / n_family
    rv_all = ep.RV.to_numpy(np.float64)
    print(f"P4-simple-vs-noctua   production slice H={PROD_H} @ {PROD_A}:00   "
          f"family {n_family} -> {100*(1-alpha):.3f}%\n")

    fold_bat = {k: [] for k in ARMS}
    pe = {k: {m: [] for m in EP_METRICS} for k in ARMS}
    years = []
    for f in folds:
        anc = fold_anchors(ep, X, f)
        sh_A = np.nan_to_num((1 - w) * (anc["A"] - anc["lhc"]))
        rF = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=fac_mask)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if rF is None or r0 is None:
            continue
        peF, pe0 = rF["per_episode"], r0["per_episode"]
        ci, ti = pe0["cal_idx"], pe0["test_idx"]
        M = pe0["M_abs"]
        h_idx = np.concatenate([ci, peF["test_idx"]])
        h_rv = np.concatenate([pe0["rv_cal"], peF["rv"]])
        sq = np.sqrt(PROD_H)

        # guard 2: product_score reproduces run_fold's own Gaussian competitor
        g_ref = battery(gaussian_curves(np.exp(anc["lhc"][ti]) * sq), M)
        g_run = barrier_cols(r0["rows"], "log_har_gauss")
        err = max(abs(g_ref[k] - g_run[k]) for k in g_run)
        if not err < TOL:
            raise SystemExit(f"REFUSING: offline Gaussian differs from run_fold's "
                             f"log_har_gauss by {err:.2e}")

        # N_As: the network, clock-aware anchor, served factor (as hour_anchor)
        h_sig = np.concatenate([pe0["sigma_cal"], peF["sigma_med"]]) * np.exp(sh_A[h_idx])
        lf = served_log_factor(ts_all[ti], ts_all[h_idx], h_rv, h_sig)
        full = sh_A.copy(); full[ti] += lf
        rA = run_fold(ep, X, f, a.hidden, a.seeds,
                      post_shift_fn=lambda mask, _mt, _s=full: _s[mask])
        if rA is None:
            continue
        peA = rA["per_episode"]
        if not np.array_equal(peA["test_idx"], ti):
            raise SystemExit("REFUSING: arms scored different episodes")
        curA = peA["curves"]["noctua_v2"]
        # guard 1: product_score reproduces the NOCTUA battery from its curves
        bA, bA_run = battery(curA, M), barrier_cols(rA["rows"])
        err = max(abs(bA[k] - bA_run[k]) for k in bA_run)
        if not err < TOL:
            raise SystemExit(f"REFUSING: offline NOCTUA battery differs from "
                             f"run_fold's by {err:.2e}")

        # G arms: no network, own served factor, Gaussian first passage
        def g_arm(anchor):
            hs = np.exp(anchor[h_idx]) * sq
            lfg = served_log_factor(ts_all[ti], ts_all[h_idx], h_rv, hs)
            return gaussian_curves(np.exp(anchor[ti] + lfg) * sq), float(lfg.mean())
        curG, lfA = g_arm(anc["A"])
        curG0, lf0 = g_arm(anc["lhc"])

        for arm, cur in (("N_As", curA), ("G_As", curG), ("G_M0s", curG0)):
            fold_bat[arm].append(battery(cur, M))
            p = per_episode(cur, M)
            for m in EP_METRICS:
                pe[arm][m].append(p[m])
        years.append(f["year"])
        print(f"  fold {f['year']}: guards ok   DSC N_As {fold_bat['N_As'][-1]['DSC']:.6f} "
              f"G_As {fold_bat['G_As'][-1]['DSC']:.6f} G_M0s {fold_bat['G_M0s'][-1]['DSC']:.6f}"
              f"   mean log factor N {lf.mean():+.3f} G {lfA:+.3f}", flush=True)

    if not years:
        print("no complete folds"); return 1

    print(f"\n{'metric':>8} " + " ".join(f"{k:>10}" for k in ARMS))
    for m in FOLD_METRICS + EP_METRICS:
        print(f"{m:>8} " + " ".join(f"{np.mean([b[m] for b in fold_bat[k]]):10.6f}"
                                    for k in ARMS))

    n_ep = sum(len(x) for x in pe["N_As"]["brier"])
    L = block_len_for(PROD_H, n_ep)
    res, out_ci = {c: {"better": [], "worse": []} for c in CONTRASTS}, {}
    print(f"\n{'metric':>8} {'contrast':>12} {'delta (+ = first better)':>26} {'interval':>30}  note")
    for c, o in CONTRASTS:
        for m in EP_METRICS:                       # per episode, block bootstrap
            d = np.concatenate(pe[o][m]) - np.concatenate(pe[c][m])
            lo, hi = mean_ci(d, alpha=alpha, block_len=L)["ci95"]
            out_ci[f"{m}_{c}_vs_{o}"] = {"delta": float(d.mean()), "ci95": [lo, hi],
                                         "estimator": "per-episode block bootstrap"}
            if lo > 0: res[(c, o)]["better"].append(m)
            if hi < 0: res[(c, o)]["worse"].append(m)
            print(f"{m:>8} {c+'-'+o:>12} {d.mean():+26.6f} [{lo:+.6f}, {hi:+.6f}]  per-episode")
        for m in FOLD_METRICS:                     # fold-level, t-interval decides
            sgn = 1.0 if m in HIGHER_BETTER else -1.0
            d = sgn * (np.array([b[m] for b in fold_bat[c]])
                       - np.array([b[m] for b in fold_bat[o]]))
            ft = fold_tests(d, alpha)
            lo, hi = ft["t_ci"]
            out_ci[f"{m}_{c}_vs_{o}"] = {"delta": float(d.mean()), "t_ci": [lo, hi],
                                         "p_flip": ft["p_flip"],
                                         "n_better": int((d > 0).sum())}
            if lo > 0: res[(c, o)]["better"].append(m)
            if hi < 0: res[(c, o)]["worse"].append(m)
            print(f"{m:>8} {c+'-'+o:>12} {d.mean():+26.6f} [{lo:+.6f}, {hi:+.6f}]  fold t, "
                  f"{int((d>0).sum())}/{len(d)} folds, p_flip {ft['p_flip']:.3f}")

    r = res[("N_As", "G_As")]
    ep_better = [m for m in r["better"] if m in EP_METRICS]
    dsc_g_ge = out_ci["DSC_N_As_vs_G_As"]["delta"] <= 0
    if len(ep_better) >= 2 and not r["worse"]:
        verdict = "NOCTUA EARNS ITS PLACE"
    elif not [m for m in r["better"] if m in EP_METRICS] and dsc_g_ge:
        verdict = "SIMPLE IS SUFFICIENT"
    else:
        verdict = "MIXED"
    print(f"\n--- pre-registered reading, N_As vs G_As: better {r['better'] or 'nothing'}, "
          f"worse {r['worse'] or 'nothing'}  ->  {verdict}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "years": years, "alpha": alpha, "n_family": n_family,
        "fold_battery": fold_bat, "ci": out_ci,
        "results": {f"{c}_vs_{o}": v for (c, o), v in res.items()},
        "verdict": verdict}, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
