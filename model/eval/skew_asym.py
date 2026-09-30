"""
eval/skew_asym.py
=====================================================================
P4-skew-asym (screen): does the next-day option SKEW predict the asymmetry of
the night's excursions? Registered in research/ledger.json before any
skew-vs-outcome look.

z = (median IV of next-day OTM puts, log K/S in [-0.10,-0.03]
     - median IV of OTM calls, log K/S in [+0.03,+0.10]) / next-day ATM IV,
from trades in [16:00, 17:00) UTC (model/artifacts/iv1d_trades.parquet),
nights with >= 2 of each; other nights stay symmetric.

Base: fold Log-HAR anchor + serving's factor, one sigma both sides.
Sk:   sigma_dn = s*exp(+g/2), sigma_up = s*exp(-g/2), g = a + c*z, with
      (a, c) = OLS of log(M_dn/M_up) on [1, z] over TRAIN 17:00 nights.
Sp:   the same with z shuffled within each year (same freedom, no night).

    python -m model.eval.skew_asym
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
from eval.hour_anchor import PROD_H, served_log_factor                 # noqa: E402
from eval.iv1d_anchor import FIRST_FOLD, shuffle_within_year           # noqa: E402
from eval.product_score import gaussian_curves, per_episode            # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

TRADES = Path("model/artifacts/iv1d_trades.parquet")
IV = Path("model/artifacts/iv1d.parquet")


def skew_by_day() -> dict:
    t = pd.read_parquet(TRADES)
    nd = t[(t.hours_to_expiry > 0) & (t.hours_to_expiry <= 24)]
    p = nd[(nd.logm >= -0.10) & (nd.logm <= -0.03) & (nd.cp == "P")].groupby("day")["iv"]
    c = nd[(nd.logm >= 0.03) & (nd.logm <= 0.10) & (nd.cp == "C")].groupby("day")["iv"]
    atm = dict(zip(*pd.read_parquet(IV)[["day", "iv_nextday"]].T.values))
    g = pd.DataFrame({"pm": p.median(), "pn": p.size(), "cm": c.median(), "cn": c.size()}).dropna()
    g = g[(g.pn >= 2) & (g.cn >= 2)]
    return {d: (r.pm - r.cm) / atm[d] for d, r in g.iterrows()
            if d in atm and np.isfinite(atm[d]) and atm[d] > 0}


def sided(s, g):
    """Gaussian curves with the down side scaled up by exp(g/2), up side down."""
    up = gaussian_curves(s * np.exp(-g / 2))["up"]
    dn = gaussian_curves(s * np.exp(+g / 2))["dn"]
    return {"up": up, "dn": dn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="skew -> excursion asymmetry screen")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/skew_asym.json"))
    a = ap.parse_args(argv)
    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    rv = ep["RV"].to_numpy(np.float64)
    fin = np.isfinite(X.to_numpy()).all(1)
    at19 = (ep.H == PROD_H).to_numpy()
    prod = np.asarray(S.production_mask(ep), bool)
    y = B.har_target(rv, ep.H.to_numpy(np.float64))
    sq = np.sqrt(PROD_H)
    Mu, Md = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy())
    asym = np.log(np.maximum(Md, 1e-6) / np.maximum(Mu, 1e-6))
    sk = skew_by_day()
    day = pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d")
    z = np.array([sk.get(d, np.nan) if p else np.nan for d, p in zip(day, prod)])
    zs = {"Sk": z, "Sp": shuffle_within_year(z, ts, seed=11)}
    cols = B.VOL_BASELINES["log_har_cal"]
    loss = {k: {"brier": [], "logs": []} for k in ("B", "Sk", "Sp")}
    side = {k: {"up": [], "dn": []} for k in ("B", "Sk", "Sp")}
    fits = []
    for f in S.walk_forward_folds(ep):
        if f["year"] < FIRST_FOLD:
            continue
        m_tr = np.asarray(f["train"], bool) & fin
        w = S.sample_weights(ep, m_tr)
        wf = np.zeros(len(ep)); wf[m_tr] = w
        lhc = np.full(len(ep), np.nan)
        lhc[fin] = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], w).predict(X[cols][fin])
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & prod & (rv > 0))
        hist = np.flatnonzero(at19 & fin & (rv > 0) & (ts >= ts[np.asarray(f["calib"], bool)].min())
                              & (ts <= ts[te].max()))
        lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(lhc[hist]) * sq)
        s = np.exp(lhc[te] + lf) * sq
        M = {"up": Mu[te], "dn": Md[te]}
        row = {"year": f["year"]}
        for k in ("B", "Sk", "Sp"):
            g = np.zeros(len(te))
            if k != "B":
                zz = zs[k]
                fm = m_tr & np.isfinite(zz) & np.isfinite(asym)
                beta = B.OLS(["z"]).fit(pd.DataFrame({"z": zz[fm]}), asym[fm], wf[fm]).beta
                row[k] = [float(beta[0]), float(beta[1]), int(fm.sum())]
                ok = np.isfinite(zz[te])
                g[ok] = beta[0] + beta[1] * zz[te][ok]
            cur = sided(s, g)
            p = per_episode(cur, M)
            loss[k]["brier"].append(p["brier"]); loss[k]["logs"].append(p["logs"])
            for sd in ("up", "dn"):
                side[k][sd].append(per_episode({"up": cur[sd], "dn": cur[sd]},
                                               {"up": M[sd], "dn": M[sd]})["brier"])
        fits.append(row)
        print(f"  fold {f['year']}: Sk (a, c, n) {np.round(row['Sk'][:2], 3).tolist()} n {row['Sk'][2]}"
              f"   Sp c {row['Sp'][1]:+.3f}", flush=True)
    cat = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in loss.items()}
    out = {"fits": fits, "ci": {}}
    res = {}
    print(f"\n{len(cat['B']['brier'])} nights; per-episode, 99.5%, block 38")
    for c_, o in (("Sk", "Sp"), ("Sk", "B"), ("Sp", "B")):
        res[f"{c_}_vs_{o}"] = {"better": [], "worse": []}
        for m in ("brier", "logs"):
            d = cat[o][m] - cat[c_][m]
            lo, hi = mean_ci(d, alpha=0.005, block_len=38)["ci95"]
            out["ci"][f"{m}_{c_}_vs_{o}"] = {"pct": float(100 * d.mean() / cat[o][m].mean()),
                                            "ci": [float(lo), float(hi)]}
            if lo > 0: res[f"{c_}_vs_{o}"]["better"].append(m)
            if hi < 0: res[f"{c_}_vs_{o}"]["worse"].append(m)
            print(f"  {m:>6} {c_}-{o}: {100 * d.mean() / cat[o][m].mean():+.3f}%  [{lo:+.6f}, {hi:+.6f}]")
    for k in ("Sk", "Sp"):
        for sd in ("up", "dn"):
            d = np.concatenate(side["B"][sd]) - np.concatenate(side[k][sd])
            print(f"   side {sd}: {k}-B brier {100 * d.mean() / np.concatenate(side['B'][sd]).mean():+.3f}%")
    met = (set(res["Sk_vs_Sp"]["better"]) >= {"brier", "logs"} and not res["Sk_vs_B"]["worse"])
    print(f"\n--- registered screen: Sk beats Sp on brier AND logs, worse than base on neither"
          f" -> {'PASSED' if met else 'NOT PASSED'}")
    out.update(results=res, screen_passed=bool(met))
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
