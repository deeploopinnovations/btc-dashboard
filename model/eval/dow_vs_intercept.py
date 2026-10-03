"""
eval/dow_vs_intercept.py -- audit M re-run (P4-dow-level-control): does the
day-of-week anchor gain survive a 17:00-ONLY intercept control? Gaussian law,
all folds, served factor from each arm; the intercept is applied at 17:00/H=19
nights only (an all-hours constant is absorbed by the median factor --
pitfall arm-not-degenerate, checked inline).

    python -m model.eval.dow_vs_intercept
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "model")
from eval.ci import mean_ci
from eval.hour_anchor import PROD_H, served_log_factor
from eval.product_score import gaussian_curves, per_episode
from eval.dow_anchor import extra_cols
from noctua import baselines as B, splits as S
from noctua.train import load_all
from research.pitfalls import check_arm_not_degenerate
ep, X = load_all(Path("model/artifacts"))
ts = ep.anchor_ts.to_numpy(np.int64); H = ep.H.to_numpy(np.int64); rv = ep.RV.to_numpy(float)
fin = np.isfinite(X.to_numpy()).all(1); at19 = (ep.H == PROD_H).to_numpy(); prod = np.asarray(S.production_mask(ep), bool)
y = B.har_target(rv, H.astype(float)); sq = np.sqrt(PROD_H)
Mu, Md = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy()); cols = B.VOL_BASELINES["log_har_cal"]
D = pd.DataFrame(extra_cols(ts, H)["Ds"]); dc = list(D.columns)
L = {k: {"brier": [], "logs": []} for k in ("G0", "GD", "GI", "GDI")}; lfs = {"G0": [], "GI": []}
for f in S.walk_forward_folds(ep):
    m_tr = np.asarray(f["train"], bool) & fin; w = S.sample_weights(ep, m_tr); wf = np.zeros(len(ep)); wf[m_tr] = w
    te = np.flatnonzero(np.asarray(f["test"], bool) & fin & prod & (rv > 0))
    hist = np.flatnonzero(at19 & fin & (rv > 0) & (ts >= ts[np.asarray(f["calib"], bool)].min()) & (ts <= ts[te].max()))
    base = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], w); XJ = pd.concat([X[cols], D], axis=1)
    dow = B.OLS(cols + dc).fit(XJ[m_tr], y[m_tr], w)
    a0 = np.full(len(ep), np.nan); a0[fin] = base.predict(X[cols][fin])
    aD = np.full(len(ep), np.nan); aD[fin] = dow.predict(XJ[fin])
    M = {"up": Mu[te], "dn": Md[te]}; p17 = m_tr & prod
    for k, an in (("G0", a0), ("GD", aD), ("GI", a0), ("GDI", aD)):
        an = an.copy()
        if k in ("GI", "GDI"):                       # intercept ONLY at 17:00/H=19 nights
            c = np.average((y - an)[p17], weights=wf[p17]); an[prod] += c
        lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(an[hist]) * sq)
        if k in ("G0", "GI"): lfs[k].append(an[te] + lf)
        p = per_episode(gaussian_curves(np.exp(an[te] + lf) * sq), M)
        L[k]["brier"].append(p["brier"]); L[k]["logs"].append(p["logs"])
print(check_arm_not_degenerate(np.concatenate(lfs["GI"]), np.concatenate(lfs["G0"]), "GI vs G0 served level"))
c = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in L.items()}
for x, o in (("GI", "G0"), ("GD", "G0"), ("GDI", "GI")):
    for m in ("brier", "logs"):
        d = c[o][m] - c[x][m]; lo, hi = mean_ci(d, alpha=0.005, block_len=38)["ci95"]
        print(f"{x:>3} vs {o:<3} {m:>5} {100*d.mean()/c[o][m].mean():+.3f}%  [{lo:+.6f}, {hi:+.6f}]")
