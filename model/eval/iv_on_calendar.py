"""
eval/iv_on_calendar.py -- P4-iv-on-calendar: does next-day implied vol add
anything on top of the day-of-week calendar plus a 17:00-only intercept?
Gaussian law, folds 2022-2026, served factor from each arm; refute-only.

    python -m model.eval.iv_on_calendar
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "model")
from eval.ci import mean_ci
from eval.hour_anchor import PROD_H, served_log_factor
from eval.product_score import gaussian_curves, per_episode
from eval.dow_anchor import extra_cols
from eval.iv1d_anchor import iv_per_episode, IV_PATH, FIRST_FOLD
from noctua import baselines as B, splits as S
from noctua.train import load_all
ep, X = load_all(Path("model/artifacts"))
ts = ep.anchor_ts.to_numpy(np.int64); H = ep.H.to_numpy(np.int64); rv = ep.RV.to_numpy(float)
fin = np.isfinite(X.to_numpy()).all(1); at19 = (ep.H == PROD_H).to_numpy(); prod = np.asarray(S.production_mask(ep), bool)
y = B.har_target(rv, H.astype(float)); sq = np.sqrt(PROD_H)
Mu, Md = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy()); cols = B.VOL_BASELINES["log_har_cal"]
D = pd.DataFrame(extra_cols(ts, H)["Ds"]); dc = list(D.columns)
liv = iv_per_episode(ep, pd.read_parquet(IV_PATH), prod)
L = {k: {"brier": [], "logs": []} for k in ("GDI", "GDIV")}
for f in S.walk_forward_folds(ep):
    if f["year"] < FIRST_FOLD: continue
    m_tr = np.asarray(f["train"], bool) & fin; w = S.sample_weights(ep, m_tr); wf = np.zeros(len(ep)); wf[m_tr] = w
    te = np.flatnonzero(np.asarray(f["test"], bool) & fin & prod & (rv > 0))
    hist = np.flatnonzero(at19 & fin & (rv > 0) & (ts >= ts[np.asarray(f["calib"], bool)].min()) & (ts <= ts[te].max()))
    XJ = pd.concat([X[cols], D], axis=1); dow = B.OLS(cols + dc).fit(XJ[m_tr], y[m_tr], w)
    aD = np.full(len(ep), np.nan); aD[fin] = dow.predict(XJ[fin])
    p17 = m_tr & prod
    c = np.average((y - aD)[p17], weights=wf[p17]); aDI = aD.copy(); aDI[prod] += c      # calendar + 17:00 intercept
    m = p17 & np.isfinite(liv)
    beta = B.OLS(["x"]).fit(pd.DataFrame({"x": (liv - aDI)[m]}), (y - aDI)[m], wf[m]).beta
    aV = aDI.copy(); ok = prod & np.isfinite(liv); aV[ok] += beta[0] + beta[1] * (liv[ok] - aDI[ok])
    print(f"  fold {f['year']}: IV increment on calendar+intercept: a {beta[0]:+.3f} b {beta[1]:+.3f}")
    M = {"up": Mu[te], "dn": Md[te]}
    for k, an in (("GDI", aDI), ("GDIV", aV)):
        lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(an[hist]) * sq)
        p = per_episode(gaussian_curves(np.exp(an[te] + lf) * sq), M)
        L[k]["brier"].append(p["brier"]); L[k]["logs"].append(p["logs"])
for m in ("brier", "logs"):
    a_, b_ = np.concatenate(L["GDI"][m]), np.concatenate(L["GDIV"][m])
    lo, hi = mean_ci(a_ - b_, alpha=0.005, block_len=38)["ci95"]
    print(f"IV on top of calendar+17:00 intercept, {m}: {100*(a_-b_).mean()/a_.mean():+.3f}%  [{lo:+.6f}, {hi:+.6f}]")
