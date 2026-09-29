"""
eval/iv1d_proxy_control.py
=====================================================================
Attack on P4-iv1d-posthoc: is next-day implied vol just a FAST ESTIMATE OF
RECENT REALISED VOL? At the 17:00 anchor the model already has har_1h (the
16:00-17:00 hour -- the same hour the option trades come from) and har_6h.
If an increment built from those reproduces the implied-vol gain, the market
adds nothing the model could not compute itself.

No-network Gaussian law (eval/product_score), folds 2022-2026, production
nights; every arm is an INCREMENT on the fold's Log-HAR anchor at 17:00/H=19
nights, a + sum_k b_k x_k with x_k = (signal_k - lhc), OLS on the fold's train
17:00 nights, serving's trailing factor from each arm's own forecasts.

  G0     lhc                      Gint   + intercept only (the 17:00 level)
  Grv    + har_1h and har_6h      Giv    + implied vol
  Gall   + har_1h, har_6h and implied vol

    python -m model.eval.iv1d_proxy_control
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ci import mean_ci                                            # noqa: E402
from eval.hour_anchor import PROD_A, PROD_H, served_log_factor         # noqa: E402
from eval.iv1d_anchor import FIRST_FOLD, IV_PATH, iv_per_episode       # noqa: E402
from eval.product_score import gaussian_curves, per_episode            # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

ARMS = {"G0": None, "Gint": [], "Grv": ["har_1h", "har_6h"], "Giv": ["iv"],
        "Gall": ["har_1h", "har_6h", "iv"]}
CONTRASTS = (("Giv", "Grv"), ("Gall", "Grv"), ("Grv", "Gint"), ("Giv", "Gint"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IV vs fast realised-vol proxy")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/iv1d_proxy_control.json"))
    a = ap.parse_args(argv)
    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    rv = ep["RV"].to_numpy(np.float64)
    fin = np.isfinite(X.to_numpy()).all(1)
    at19 = (ep.H == PROD_H).to_numpy()
    prod = np.asarray(S.production_mask(ep), bool)
    y = B.har_target(rv, ep.H.to_numpy(np.float64))
    sq = np.sqrt(PROD_H)
    M_up, M_dn = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy())
    sig = {"iv": iv_per_episode(ep, pd.read_parquet(IV_PATH), prod),
           "har_1h": X["har_1h"].to_numpy(np.float64),
           "har_6h": X["har_6h"].to_numpy(np.float64)}
    cols = B.VOL_BASELINES["log_har_cal"]
    loss = {k: {"brier": [], "logs": []} for k in ARMS}
    coefs = []
    for f in S.walk_forward_folds(ep):
        if f["year"] < FIRST_FOLD:
            continue
        m_tr = np.asarray(f["train"], bool) & fin
        w = S.sample_weights(ep, m_tr)
        wfull = np.zeros(len(ep)); wfull[m_tr] = w
        lhc = np.full(len(ep), np.nan)
        lhc[fin] = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], w).predict(X[cols][fin])
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & prod & np.isfinite(rv) & (rv > 0))
        hist = np.flatnonzero(at19 & fin & np.isfinite(rv) & (rv > 0)
                              & (ts >= ts[np.asarray(f["calib"], bool)].min()) & (ts <= ts[te].max()))
        ok_all = prod & np.isfinite(lhc) & np.all([np.isfinite(v) for v in sig.values()], axis=0)
        fit_m = m_tr & ok_all
        M = {"up": M_up[te], "dn": M_dn[te]}
        row = {"year": f["year"]}
        for k, feats in ARMS.items():
            an = lhc.copy()
            if feats is not None:
                Xi = pd.DataFrame({s: (sig[s] - lhc) for s in feats})
                if feats:
                    beta = B.OLS(feats).fit(Xi[fit_m], (y - lhc)[fit_m], wfull[fit_m]).beta
                else:
                    beta = np.array([np.average((y - lhc)[fit_m], weights=wfull[fit_m])])
                row[k] = [float(b) for b in beta]
                app = ok_all
                inc = beta[0] + (Xi[app].to_numpy() @ beta[1:] if feats else 0.0)
                an[app] = lhc[app] + inc
            lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(an[hist]) * sq)
            p = per_episode(gaussian_curves(np.exp(an[te] + lf) * sq), M)
            loss[k]["brier"].append(p["brier"]); loss[k]["logs"].append(p["logs"])
        coefs.append(row)
        print(f"  fold {f['year']}: Grv b {np.round(row['Grv'][1:], 3).tolist()}  Giv b "
              f"{np.round(row['Giv'][1:], 3).tolist()}  Gall b {np.round(row['Gall'][1:], 3).tolist()}",
              flush=True)
    cat = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in loss.items()}
    out = {"coefs": coefs, "vs_G0": {}, "contrasts": {}}
    print(f"\n{len(cat['G0']['brier'])} nights; gain vs G0 (per-night mean, %):")
    for k in ARMS:
        if k == "G0":
            continue
        g = {m: float(100 * (cat["G0"][m] - cat[k][m]).mean() / cat["G0"][m].mean()) for m in ("brier", "logs")}
        out["vs_G0"][k] = g
        print(f"   {k:>5}: brier {g['brier']:+.3f}%  logs {g['logs']:+.3f}%")
    print("\ncontrasts (99.5%, block 38):")
    for c, o in CONTRASTS:
        for m in ("brier", "logs"):
            d = cat[o][m] - cat[c][m]
            lo, hi = mean_ci(d, alpha=0.005, block_len=38)["ci95"]
            out["contrasts"][f"{m}_{c}_vs_{o}"] = {"pct": float(100 * d.mean() / cat[o][m].mean()),
                                                  "ci": [float(lo), float(hi)]}
            print(f"   {c:>5} vs {o:<5} {m:>6} {100 * d.mean() / cat[o][m].mean():+.3f}%  [{lo:+.6f}, {hi:+.6f}]")
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
