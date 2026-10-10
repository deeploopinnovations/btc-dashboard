"""
eval/iv1d_anchor.py
=====================================================================
P4-iv1d-anchor: does the market's own forecast for the product night improve
the served barrier product? (Pre-registered in research/ledger.json before
any IV-vs-RV relationship was looked at.)

FEATURE (eval/harvest_iv1d.py): iv = the ATM implied vol of the Deribit option
expiring 08:00 UTC next morning, from trades in [16:00, 17:00) UTC -- strictly
before the 17:00 anchor. x = log(iv/100/sqrt(8760)) - lhc: the gap between the
market's hourly vol and the fold's Log-HAR anchor (both log hourly vol).

ARMS (all through run_fold, every one with serving's trailing factor from its
own forecasts; folds 2022-2026):
  M0s  shipped
  Is   increment a + b*x at 17:00/H=19 nights WITH iv, OLS of the anchor
       residual on [1, x] over the fold's TRAIN 17:00 nights with iv;
       nights without iv keep the served anchor (as deployed)
  Ip   PLACEBO: the same, each night's iv replaced by a random other night's
       iv from the same calendar year -- same distribution, no night

    python -m model.eval.iv1d_anchor --selftest
    python -m model.eval.iv1d_anchor
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IV_PATH = Path("model/artifacts/iv1d.parquet")
FIRST_FOLD = 2022
ARMS = ("M0s", "Is", "Ip")
EP_METRICS = ("brier", "logs", "pinball", "crps", "qlike")
BAR = ("brier", "logs", "pinball", "crps")
CONTRASTS = (("Is", "M0s"), ("Ip", "M0s"))
TOL_BAT = 1e-10
SEED = 7


def iv_per_episode(ep: pd.DataFrame, iv: pd.DataFrame, prod19: np.ndarray) -> np.ndarray:
    """log hourly implied vol at each 17:00/H=19 episode with a next-day IV."""
    day = pd.to_datetime(ep["anchor_ts"].to_numpy(np.int64), unit="s", utc=True).strftime("%Y-%m-%d")
    lut = dict(zip(iv["day"], iv["iv_nextday"]))
    v = np.array([lut.get(d, np.nan) for d in day], np.float64)
    out = np.full(len(ep), np.nan)
    ok = prod19 & np.isfinite(v) & (v > 0)
    out[ok] = np.log(v[ok] / 100.0 / np.sqrt(8760.0))
    return out


def shuffle_within_year(liv: np.ndarray, ts: np.ndarray, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yr = pd.to_datetime(ts, unit="s", utc=True).year.to_numpy()
    out = liv.copy()
    for y in np.unique(yr):
        idx = np.flatnonzero((yr == y) & np.isfinite(liv))
        out[idx] = liv[rng.permutation(idx)]
    return out


def fit_increment(y, lhc, liv, m_fit, w) -> tuple[float, float]:
    from noctua import baselines as B
    m = m_fit & np.isfinite(liv) & np.isfinite(lhc)
    X = pd.DataFrame({"x": (liv - lhc)[m]})
    beta = B.OLS(["x"]).fit(X, (y - lhc)[m], w[m] if w is not None else None).beta
    return float(beta[0]), float(beta[1])


def main(argv=None) -> int:
    from eval.benchmark import run_fold
    from eval.ci import mean_ci
    from eval.hour_anchor import FAC_STRIDE_H, PROD_A, PROD_H, fold_tests, served_log_factor
    from eval.product_score import battery, per_episode
    from eval.scale_adopt import barrier_cols
    from eval.vol_matrix import block_len_for, qlike_vec
    from noctua import baselines as B
    from noctua import infer as I
    from noctua import splits as S
    from noctua.train import load_all

    ap = argparse.ArgumentParser(description="next-day implied vol in the anchor")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/iv1d_anchor.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    iv = pd.read_parquet(IV_PATH)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    prod = np.asarray(S.production_mask(ep), bool)
    fac_mask = at19 & np.isin(ah, sorted({(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}))
    fin = np.isfinite(X.to_numpy()).all(1)
    y = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))
    liv = {"Is": iv_per_episode(ep, iv, prod)}
    liv["Ip"] = shuffle_within_year(liv["Is"], ts)
    w_bl = I.BLEND_W
    print(f"P4-iv1d-anchor   folds {FIRST_FOLD}-2026, production 17:00/H=19; nights with "
          f"iv: {int(np.isfinite(liv['Is']).sum())}", flush=True)

    pe = {k: {m: [] for m in EP_METRICS} for k in ARMS}
    dsc = {k: [] for k in ARMS}
    keep = {"idx": [], "year": [], "has_iv": [], "x": []}
    fits, years = [], []
    for f in S.walk_forward_folds(ep):
        if f["year"] < FIRST_FOLD:
            continue
        t0 = time.time()
        m_tr = np.asarray(f["train"], bool) & fin
        wtr = S.sample_weights(ep, m_tr)
        cols = B.VOL_BASELINES["log_har_cal"]
        lhc = np.full(len(ep), np.nan)
        lhc[fin] = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], wtr).predict(X[cols][fin])
        wfull = np.zeros(len(ep)); wfull[m_tr] = wtr
        sh = {"M0s": np.zeros(len(ep))}
        fit = {"year": f["year"]}
        for arm in ("Is", "Ip"):
            ca, cb = fit_increment(y, lhc, liv[arm], m_tr & prod, wfull)
            s_ = np.zeros(len(ep))
            ok = np.isfinite(liv[arm]) & np.isfinite(lhc)
            s_[ok] = (1 - w_bl) * (ca + cb * (liv[arm][ok] - lhc[ok]))
            sh[arm] = s_
            fit[arm] = {"a": ca, "b": cb,
                        "n_train_nights": int((m_tr & prod & np.isfinite(liv[arm])).sum())}
        rB = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=prod | fac_mask)
        if rB is None:
            continue
        peB = rB["per_episode"]
        te = peB["test_idx"]
        ti = te[np.flatnonzero(prod[te])]
        f_rows = np.flatnonzero(fac_mask[te])
        h_idx = np.concatenate([peB["cal_idx"], te[f_rows]])
        h_rv = np.concatenate([peB["rv_cal"], peB["rv"][f_rows]])
        h_sig = np.concatenate([peB["sigma_cal"], peB["sigma_med"][f_rows]])
        for arm in ARMS:
            lf = served_log_factor(ts[ti], ts[h_idx], h_rv, h_sig * np.exp(sh[arm][h_idx]))
            full = sh[arm].copy()
            full[ti] += lf
            r1 = run_fold(ep, X, f, a.hidden, a.seeds,
                          post_shift_fn=lambda mask, _mt, _s=full: _s[mask])
            p1 = r1["per_episode"]
            if not np.array_equal(p1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            cur, M = p1["curves"]["noctua_v2"], p1["M_abs"]
            b, b_run = battery(cur, M), barrier_cols(r1["rows"])
            if not max(abs(b[k] - b_run[k]) for k in b_run) < TOL_BAT:
                raise SystemExit("REFUSING: offline battery differs from run_fold")
            p = per_episode(cur, M)
            for m in BAR:
                pe[arm][m].append(p[m])
            pe[arm]["qlike"].append(qlike_vec(p1["rv"], p1["sigma_mean"]))
            dsc[arm].append(b["DSC"])
        keep["idx"].append(ti); keep["year"].append(np.full(len(ti), f["year"]))
        keep["has_iv"].append(np.isfinite(liv["Is"][ti]))
        keep["x"].append(liv["Is"][ti] - lhc[ti])
        fits.append(fit); years.append(f["year"])
        cov = float(np.isfinite(liv["Is"][ti]).mean())
        print(f"  fold {f['year']}: Is a {fit['Is']['a']:+.3f} b {fit['Is']['b']:+.3f} "
              f"(n {fit['Is']['n_train_nights']})  Ip b {fit['Ip']['b']:+.3f}  test coverage "
              f"{cov:.0%}  DSC " + " ".join(f"{k} {dsc[k][-1]:.6f}" for k in ARMS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    cat = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in pe.items()}
    has = np.concatenate(keep["has_iv"])
    xx = np.concatenate(keep["x"])
    n = len(has)
    L = block_len_for(PROD_H, n)
    alpha = 0.05 / (len(CONTRASTS) * len(EP_METRICS))
    out = {"years": years, "n_nights": n, "coverage": float(has.mean()), "fits": fits,
           "alpha": alpha, "ci": {}, "dsc": {}, "subsets": {}}
    res = {}
    print(f"\n{n} production nights ({100 * has.mean():.1f}% with iv), block {L}, "
          f"family 10 -> {100 * (1 - alpha):.2f}%")
    for c, o in CONTRASTS:
        better, worse = [], []
        for m in EP_METRICS:
            d = cat[o][m] - cat[c][m]
            lo, hi = mean_ci(d, alpha=alpha, block_len=L)["ci95"]
            pct = 100 * d.mean() / cat[o][m].mean()
            out["ci"][f"{m}_{c}_vs_{o}"] = {"pct": float(pct), "ci": [float(lo), float(hi)]}
            if lo > 0: better.append(m)
            if hi < 0: worse.append(m)
            print(f"  {m:>8} {c + '-' + o:>7} {pct:+7.3f}%  [{lo:+.6f}, {hi:+.6f}]")
        dd = np.array(dsc[c]) - np.array(dsc[o])
        ft = fold_tests(dd, alpha)
        out["dsc"][f"{c}_vs_{o}"] = {"delta": float(dd.mean()), "t": ft["t_ci"],
                                     "folds_better": int((dd > 0).sum())}
        print(f"       DSC {c + '-' + o:>7} {dd.mean():+.6f}  {int((dd > 0).sum())}/{len(dd)} folds")
        res[f"{c}_vs_{o}"] = {"better": better, "worse": worse}
    thr = np.nanquantile(np.abs(xx[has]), 0.9) if has.any() else np.nan
    print("\nreported, no rule (Is - M0s):")
    for name, msk in (("nights with iv", has),
                      ("ex-ante top decile |x|", has & (np.abs(xx) >= thr))):
        cell = {"n": int(msk.sum())}
        for m in ("brier", "logs", "qlike"):
            d = cat["M0s"][m][msk] - cat["Is"][m][msk]
            lo, hi = mean_ci(d, alpha=0.05, block_len=1 if "decile" in name else L)["ci95"]
            cell[m] = {"pct": float(100 * d.mean() / cat["M0s"][m][msk].mean()),
                       "ci95": [float(lo), float(hi)]}
        out["subsets"][name] = cell
        print(f"   {name:>24} n={cell['n']:4d}  " + "  ".join(
            f"{m} {cell[m]['pct']:+.2f}% [{cell[m]['ci95'][0]:+.5f}, {cell[m]['ci95'][1]:+.5f}]"
            for m in ("brier", "logs", "qlike")))
    ri, rp = res["Is_vs_M0s"], res["Ip_vs_M0s"]
    met = len([m for m in ri["better"] if m in BAR]) >= 2 and not ri["worse"] and not rp["better"]
    print("\n--- registered rule: Is beats M0s on >= 2 of the 4 barrier metrics, worse on")
    print("    none of the five, and the shuffled-IV placebo better on none")
    print(f"   Is vs M0s: better {ri['better'] or 'nothing'}, worse {ri['worse'] or 'nothing'}")
    print(f"   Ip vs M0s: better {rp['better'] or 'nothing'}, worse {rp['worse'] or 'nothing'}")
    print(f"   -> {'MET (ADVANCE)' if met else 'NOT MET (REJECT)'}")
    out.update(results=res, rule_met=bool(met))
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(a.out.with_suffix(".npz"),
                        **{k: np.concatenate(v) for k, v in keep.items()},
                        **{f"{k}__{m}": cat[k][m] for k in ARMS for m in EP_METRICS})
    print(f"wrote {a.out} and {a.out.with_suffix('.npz')}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(1)
    ts = (1_600_000_000 + np.arange(1000) * 86400).astype(np.int64)
    liv = rng.normal(-5, 0.3, 1000); liv[::7] = np.nan
    sh = shuffle_within_year(liv, ts)
    yr = pd.to_datetime(ts, unit="s", utc=True).year.to_numpy()
    same = all(np.allclose(np.sort(sh[(yr == y) & np.isfinite(sh)]),
                           np.sort(liv[(yr == y) & np.isfinite(liv)])) for y in np.unique(yr))
    ok.append(("placebo keeps each year's IV values, and the missing nights", same
               and np.array_equal(np.isfinite(sh), np.isfinite(liv))))
    ok.append(("placebo actually moves nights", np.nanmean(np.abs(sh - liv)) > 0.1))
    lhc = rng.normal(-5, 0.3, 1000)
    yv = lhc + 0.1 + 0.4 * (liv - lhc) + rng.normal(0, 0.05, 1000)
    ca, cb = fit_increment(yv, lhc, liv, np.ones(1000, bool), np.ones(1000))
    ok.append(("increment recovers a planted a=0.1, b=0.4", abs(ca - 0.1) < 0.02 and abs(cb - 0.4) < 0.02))
    for name, g in ok:
        print(f"  [{'ok' if g else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
