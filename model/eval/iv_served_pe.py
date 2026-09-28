"""
eval/iv_served_pe.py
=====================================================================
P4-iv-served-pe: E2c, the implied-volatility correction, re-scored PER EPISODE
on the SERVED base.

E2c produced the largest forecast effect in this project (-11.6% pooled QLIKE
on production, -6.2% on the wide slice) and was downgraded to NOT PROVEN
(E2-audit-downgrade) because its fold-level interval could not fail at n = 5
and "the paired per-episode estimator does not exist". It exists now
(eval/product_score, validated against run_fold to 1e-10). And P4-hour-anchor-
result showed that the benchmark's factor-free M0 rewards any arm that lowers
the 17:00 level -- a confound every earlier E2c product test carried.

So: the same E2c correction (five dynamics features, no intercept, SIGMA_B
shrinkage, beta fitted on EARLIER folds' cached out-of-sample forecasts --
eval/iv_confirm.py's construction, imported, not re-derived), applied through
run_fold's post_shift_fn, with SERVING'S TRAILING FACTOR computed from each
arm's own forecasts (eval/hour_anchor.py's emulation). The factor absorbs E2c's
average level; what survives is whether implied volatility says WHICH NIGHTS
are riskier.

Arms: M0s (served baseline), E2s (E2c + factor), P2s (the placebo: the same
machinery on IV features rotated about a year inside the covered era, + factor).
Scored on IV-covered production test nights, identical across arms.

This touches only walk-forward data. E2c's forward holdout (DATA_USE.md,
frozen 2026-08-28) is not read or scored.

    python -m model.eval.iv_served_pe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import BARRIER_U, run_fold                         # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.hour_anchor import (FAC_STRIDE_H, PROD_A, PROD_H,            # noqa: E402
                              served_log_factor)
from eval.iv_confirm import COLS, _rotation_rows, fit_history_beta     # noqa: E402
from eval.iv_correction import EXP_BOUND, standardise                  # noqa: E402
from eval.product_score import battery, per_episode, touch             # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import qlike_vec                                  # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

METRICS = ("qlike", "brier", "logs", "pinball", "crps", "brier_far")
CONTRASTS = (("E2s", "M0s"), ("P2s", "M0s"), ("E2s", "P2s"))
SUBSETS = ("HOT", "HIGH")
TOL = 1e-10


def far_brier(cur, M) -> np.ndarray:
    k = len(BARRIER_U) - 1
    out = []
    for side in ("up", "dn"):
        p = np.clip(touch(cur[side])[:, k], 1e-6, 1 - 1e-6)
        out.append((p - (M[side] >= BARRIER_U[k]).astype(float)) ** 2)
    return np.mean(out, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2c per-episode on the served base")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/iv_served_pe.json"))
    a = ap.parse_args(argv)

    npz = np.load(a.artifacts / "blend_ceiling.npz")
    years = sorted({int(k.split("_")[-1]) for k in npz.files})
    ep, X = load_all(a.artifacts)
    iv = pd.read_parquet(a.artifacts / "iv_features.parquet")
    if not np.array_equal(iv["anchor_ts"].to_numpy(np.int64), ep["anchor_ts"].to_numpy(np.int64)):
        raise SystemExit("REFUSING: iv_features not aligned with episodes")
    Z_all = iv[COLS].to_numpy(np.float64)
    covered = np.flatnonzero(np.isfinite(Z_all).all(axis=1))
    roll = _rotation_rows(covered, iv["anchor_ts"].to_numpy(np.int64))
    pos_of = {int(r): i for i, r in enumerate(covered)}

    def rotated(rows):
        out = np.full((len(rows), len(COLS)), np.nan)
        for j, r in enumerate(rows):
            i = pos_of.get(int(r))
            if i is not None:
                out[j] = Z_all[covered[(i + roll) % len(covered)]]
        return out

    def shift_vec(fit, rot: bool) -> np.ndarray:
        """E2c log shift for every episode; exactly zero where IV is absent."""
        z = rotated(np.arange(len(ep))) if rot else Z_all
        cov = np.isfinite(z).all(axis=1)
        out = np.zeros(len(ep))
        zz = standardise(z[cov], fit["mu"], fit["sd"], False)
        out[cov] = np.clip(zz @ fit["beta"], -EXP_BOUND, EXP_BOUND)
        return out

    Hall = ep["H"].to_numpy(np.float64)
    raw_har = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(m):                                   # as iv_confirm / benchmark main
        lo, hi = np.quantile(raw_har[m], [0.005, 0.995])
        return np.maximum(np.clip(raw_har, lo, hi), 1e-12)

    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = {(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}
    fac_mask = at19 & np.isin(ah, sorted(fac_hours))
    shock = X["har_6h"].to_numpy(np.float64) - X["har_22d"].to_numpy(np.float64)
    lvl = X["har_1d"].to_numpy(np.float64)

    L = {arm: {m: [] for m in METRICS} for arm in ("M0s", "E2s", "P2s")}
    keep = {k: [] for k in ("idx", "year", "hot", "high")}
    for f in S.walk_forward_folds(ep):
        y = f["year"]
        fit = fit_history_beta(npz, years, y, Z_all)
        fit_p = fit_history_beta(npz, years, y, Z_all, rotate=roll, covered_rows=covered)
        if fit is None or fit_p is None:
            print(f"  {y}: no prior fold -- excluded", flush=True)
            continue
        sh = {"M0s": np.zeros(len(ep)), "E2s": shift_vec(fit, False),
              "P2s": shift_vec(fit_p, True)}
        rF = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn, prod_override=fac_mask)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn)
        if rF is None or r0 is None:
            continue
        peF, pe0 = rF["per_episode"], r0["per_episode"]
        ci, ti = pe0["cal_idx"], pe0["test_idx"]
        h_idx = np.concatenate([ci, peF["test_idx"]])
        h_rv = np.concatenate([pe0["rv_cal"], peF["rv"]])
        h_sig = np.concatenate([pe0["sigma_cal"], peF["sigma_med"]])
        cov_te = np.isfinite(Z_all[ti]).all(axis=1)
        got = {}
        for arm in ("M0s", "E2s", "P2s"):
            lf = served_log_factor(ts_all[ti], ts_all[h_idx], h_rv,
                                   h_sig * np.exp(sh[arm][h_idx]))
            full = sh[arm].copy(); full[ti] += lf
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn,
                          post_shift_fn=lambda mask, _mt, _s=full: _s[mask])
            p1 = r1["per_episode"]
            if not np.array_equal(p1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            cur, M = p1["curves"]["noctua_v2"], p1["M_abs"]
            b, b_run = battery(cur, M), barrier_cols(r1["rows"])
            if not max(abs(b[k] - b_run[k]) for k in b_run) < TOL:
                raise SystemExit("REFUSING: offline battery differs from run_fold")
            pe = per_episode(cur, M)
            pe["brier_far"] = far_brier(cur, M)
            pe["qlike"] = qlike_vec(p1["rv"], p1["sigma_mean"])
            got[arm] = pe
        for arm in got:
            for m in METRICS:
                L[arm][m].append(np.asarray(got[arm][m])[cov_te])
        tc = ti[cov_te]
        keep["idx"].append(tc); keep["year"].append(np.full(len(tc), y))
        keep["hot"].append(shock[tc] >= np.nanquantile(shock[tc], 0.9))
        keep["high"].append(lvl[tc] >= np.nanquantile(lvl[tc], 0.9))
        print(f"  {y}: guards ok, {cov_te.sum()} IV-covered of {len(ti)} nights, "
              f"beta {np.round(fit['beta'], 3).tolist()}", flush=True)

    cat = {k: np.concatenate(v) for k, v in keep.items()}
    LL = {arm: {m: np.concatenate(v) for m, v in d.items()} for arm, d in L.items()}
    np.savez_compressed(a.out.with_suffix(".npz"), **cat,
                        **{f"{arm}__{m}": LL[arm][m] for arm in LL for m in METRICS})
    n_family = len(CONTRASTS) * len(METRICS) + len(SUBSETS) * len(METRICS)
    alpha = 0.05 / n_family
    out = {"alpha": alpha, "n_family": n_family, "n_nights": int(len(cat["idx"])), "ci": {}}
    res = {c: {"better": [], "worse": []} for c in CONTRASTS}
    print(f"\n{len(cat['idx'])} IV-covered production nights, family {n_family} -> "
          f"{100*(1-alpha):.2f}%, block 38 (subsets block 1)")
    for c, o in CONTRASTS:
        for m in METRICS:
            d = LL[o][m] - LL[c][m]
            lo, hi = mean_ci(d[np.isfinite(d)], alpha=alpha, block_len=38)["ci95"]
            pct = 100 * np.nanmean(d) / np.nanmean(LL[o][m])
            out["ci"][f"{m}_{c}_vs_{o}"] = {"pct": float(pct), "ci": [float(lo), float(hi)]}
            if lo > 0: res[(c, o)]["better"].append(m)
            if hi < 0: res[(c, o)]["worse"].append(m)
            print(f"  {m:>9} {c+'-'+o:>9} {pct:+7.2f}%  [{lo:+.6f}, {hi:+.6f}]")
    tail_damage = []
    for name, key in (("HOT", "hot"), ("HIGH", "high")):
        msk = cat[key]
        for m in METRICS:
            d = LL["M0s"][m][msk] - LL["E2s"][m][msk]
            lo, hi = mean_ci(d[np.isfinite(d)], alpha=alpha, block_len=1)["ci95"]
            pct = 100 * np.nanmean(d) / np.nanmean(LL["M0s"][m][msk])
            out["ci"][f"{m}_E2s_vs_M0s_{name}"] = {"pct": float(pct), "ci": [float(lo), float(hi)],
                                                   "n": int(msk.sum())}
            if hi < 0: tail_damage.append(f"{name}:{m}")
            print(f"  {m:>9} E2s-M0s {name:>4} (n={int(msk.sum())}) {pct:+7.2f}%  [{lo:+.6f}, {hi:+.6f}]")
    e, p, ep_ = res[("E2s", "M0s")], res[("P2s", "M0s")], res[("E2s", "P2s")]
    bar_better = [m for m in e["better"] if m not in ("qlike",)]
    met = (len(bar_better) >= 2 and "qlike" in e["better"] and not e["worse"]
           and not [m for m in e["better"] if m in p["better"]]
           and not tail_damage)
    print(f"\n--- registered rule: E2s beats M0s on >= 2 barrier metrics AND QLIKE, worse on none;")
    print(f"    the placebo beats M0s on none of the metrics E2s wins; no ex-ante tail damage")
    print(f"   E2s vs M0s: better {e['better'] or 'nothing'}, worse {e['worse'] or 'nothing'}")
    print(f"   P2s vs M0s: better {p['better'] or 'nothing'}, worse {p['worse'] or 'nothing'}")
    print(f"   E2s vs P2s: better {ep_['better'] or 'nothing'}, worse {ep_['worse'] or 'nothing'}")
    print(f"   ex-ante tail damage: {tail_damage or 'none'}   ->  {'MET' if met else 'NOT MET'}")
    out.update(results={f"{c}_vs_{o}": v for (c, o), v in res.items()},
               tail_damage=tail_damage, rule_met=bool(met))
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
