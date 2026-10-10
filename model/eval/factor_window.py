"""
eval/factor_window.py
=====================================================================
P4-factor-window (registered in research/ledger.json before this ran): does
the TRUNCATION of serving's trailing factor window change the served product?

serve/adaptive.volatility_correction means a 60-day median of realised/forecast
over settled anchors at a 6-hour stride -- and eval/hour_anchor.served_log_factor,
which every walk-forward result used, computes exactly that. In serving, the
400-day history bundle cannot give the oldest ~26 days of that window their
365-day reg_rv_vs_year lookback, so those anchors are dropped and the served
window is ~34 days just after the weekly bundle rewrite, ~41 just before it.

No-network Gaussian law (eval/product_score), folds 2022-2026, production
nights, fold Log-HAR anchor. The arms differ ONLY in the window length:
  W60  the intended window (what research scored)
  W34  the served window just after a bundle rewrite
  W41  the served window just before one

    python -m model.eval.factor_window
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ci import mean_ci                                            # noqa: E402
from eval.hour_anchor import (FAC_HI, FAC_LO, FAC_MIN, FAC_STRIDE_H,   # noqa: E402
                              FAC_WINDOW_D, HOUR, PROD_H, served_log_factor)
from eval.iv1d_anchor import FIRST_FOLD                                # noqa: E402
from eval.product_score import gaussian_curves, per_episode            # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

ARMS = {"W60": 60, "W34": 34, "W41": 41}
CONTRASTS = (("W34", "W60"), ("W41", "W60"))


def log_factor(t_query, ts_hist, rv_hist, sig_hist, window_d: int) -> np.ndarray:
    """eval/hour_anchor.served_log_factor with the window as a parameter:
    median of rv/sigma over anchors [t - H - window, t - H) at the stride,
    >= FAC_MIN ratios or 1.0, clipped to [FAC_LO, FAC_HI]."""
    lut = dict(zip(np.asarray(ts_hist, np.int64).tolist(),
                   zip(np.asarray(rv_hist, float), np.asarray(sig_hist, float))))
    out = np.zeros(len(t_query))
    for i, t in enumerate(np.asarray(t_query, np.int64)):
        last = int(t) - PROD_H * HOUR
        grid = np.arange(last - window_d * 24 * HOUR, last, FAC_STRIDE_H * HOUR, dtype=np.int64)
        r = np.array([p[0] / p[1] for p in (lut[a] for a in grid.tolist() if a in lut)
                      if np.isfinite(p[0]) and np.isfinite(p[1]) and p[0] > 0 and p[1] > 0])
        if len(r) < FAC_MIN:
            continue
        out[i] = np.log(np.clip(float(np.median(r)), FAC_LO, FAC_HI))
    return out


def selftest() -> int:
    """The parametrised factor at 60 days IS research's served_log_factor."""
    rng = np.random.default_rng(0)
    ts = np.arange(0, 400 * 24 * HOUR, HOUR, dtype=np.int64)
    rv, sg = np.exp(rng.normal(0, 0.3, len(ts))), np.exp(rng.normal(0, 0.1, len(ts)))
    q = ts[-200::7]
    a = log_factor(q, ts, rv, sg, FAC_WINDOW_D)
    b = served_log_factor(q, ts, rv, sg)
    ok = [("W60 equals research's served_log_factor, bit for bit", np.array_equal(a, b)),
          ("a shorter window gives a different factor", not np.array_equal(a, log_factor(q, ts, rv, sg, 34)))]
    for n, g in ok:
        print(f"  [{'ok' if g else 'FAIL'}] {n}")
    return 0 if all(g for _, g in ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="served factor window: 60 days vs the truncated served window")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/factor_window.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    rv = ep["RV"].to_numpy(np.float64)
    fin = np.isfinite(X.to_numpy()).all(1)
    at19 = (ep.H == PROD_H).to_numpy()
    prod = np.asarray(S.production_mask(ep), bool)
    y = B.har_target(rv, ep.H.to_numpy(np.float64))
    sq = np.sqrt(PROD_H)
    M_up, M_dn = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy())
    cols = B.VOL_BASELINES["log_har_cal"]
    loss = {k: {"brier": [], "logs": []} for k in ARMS}
    lfs = {k: [] for k in ARMS}
    for f in S.walk_forward_folds(ep):
        if f["year"] < FIRST_FOLD:
            continue
        m_tr = np.asarray(f["train"], bool) & fin
        w = S.sample_weights(ep, m_tr)
        lhc = np.full(len(ep), np.nan)
        lhc[fin] = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], w).predict(X[cols][fin])
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & prod & np.isfinite(rv) & (rv > 0))
        hist = np.flatnonzero(at19 & fin & np.isfinite(rv) & (rv > 0)
                              & (ts >= ts[np.asarray(f["calib"], bool)].min()) & (ts <= ts[te].max()))
        M = {"up": M_up[te], "dn": M_dn[te]}
        for k, wd in ARMS.items():
            lf = log_factor(ts[te], ts[hist], rv[hist], np.exp(lhc[hist]) * sq, wd)
            lfs[k].append(lf)
            p = per_episode(gaussian_curves(np.exp(lhc[te] + lf) * sq), M)
            loss[k]["brier"].append(p["brier"]); loss[k]["logs"].append(p["logs"])
        print(f"  fold {f['year']}: {len(te)} nights; mean log factor "
              + "  ".join(f"{k} {lfs[k][-1].mean():+.4f}" for k in ARMS), flush=True)
    cat = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in loss.items()}
    lf = {k: np.concatenate(v) for k, v in lfs.items()}
    out = {"n_nights": int(len(lf["W60"])), "log_factor": {}, "contrasts": {}}
    print(f"\n{out['n_nights']} nights")
    for k in ARMS:
        out["log_factor"][k] = {"mean": float(lf[k].mean()), "sd": float(lf[k].std()),
                                "sd_of_night_change": float(np.diff(lf[k]).std())}
        print(f"  {k}: log factor mean {lf[k].mean():+.4f} sd {lf[k].std():.4f} "
              f"night-to-night sd {np.diff(lf[k]).std():.4f}")
    print("\ncontrasts (99.5%, block 38; positive = the first arm is better):")
    res = {}
    for c, o in CONTRASTS:
        res[f"{c}_vs_{o}"] = {"better": [], "worse": []}
        for m in ("brier", "logs"):
            d = cat[o][m] - cat[c][m]
            lo, hi = mean_ci(d, alpha=0.005, block_len=38)["ci95"]
            pct = float(100 * d.mean() / cat[o][m].mean())
            out["contrasts"][f"{m}_{c}_vs_{o}"] = {"pct": pct, "ci": [float(lo), float(hi)]}
            if lo > 0:
                res[f"{c}_vs_{o}"]["better"].append(m)
            if hi < 0:
                res[f"{c}_vs_{o}"]["worse"].append(m)
            print(f"   {c} vs {o} {m:>6} {pct:+.3f}%  [{lo:+.6f}, {hi:+.6f}]")
    short_better = any(set(v["better"]) >= {"brier", "logs"} for v in res.values())
    w60_better = (any(v["worse"] for v in res.values())
                  and not any(v["better"] for v in res.values()))
    tie = not any(v["better"] or v["worse"] for v in res.values())
    decision = ("(c) a short window is better on both: report it, keep 60, register it separately"
                if short_better else
                "(b) W60 is better: restore 60 days in serving" if w60_better else
                "(a) no contrast separates: restore 60 days in serving (R18 tie-break)" if tie else
                "mixed: not covered by (a)-(c) -- report, change nothing")
    print(f"\n--- registered decision rule -> {decision}")
    out.update(results=res, decision=decision)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
