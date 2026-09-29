"""
eval/short_anchor.py
=====================================================================
P4-short-anchor: a REACTIVE anchor that knows the clock -- aimed at the one
thing NOCTUA does not do better than a simple model: RANK nights.

WHAT THE AUDITS SAID

  * P4-simple-vs-noctua-result: NOCTUA earns its place on the SHAPE of the
    barrier distribution (log score, pinball) and not on discrimination -- a
    clock-aware Log-HAR + Gaussian ranks nights as well (DSC level or ahead).
    Ranking is a question about the LEVEL forecast moving with the night.
  * P4-zoo-stack-v2-result: the most reactive teacher, har_short (last 1h and
    6h of vol), carries weight at short horizons; P4-stack-anchor-result: at
    the 17:00 product anchor it points the WRONG way, because its trailing
    hours are the busy US session and the window ahead is the quiet night.

So the reactive information is there and its clock is wrong. Remove the clock
from it: DESEASONALISE the short windows by the training-slice intraday
profile (noctua/season.py), keep the forward-window season term, and fit the
anchor on train as before.

    har_xh_ds = har_xh - season_back(a, x)      x in {1, 6} hours
    season_back(a, x) = 0.5 log mean v(clock hours a-x .. a-1)   (profile, train)

ARMS (no network -- Gaussian first-passage through eval/product_score,
validated against run_fold; serving's trailing factor from each arm's own
forecasts at the factor hours):
  G_As   clock-aware Log-HAR anchor (P4-hour-anchor) -- the reference
  G_SD   + deseasonalised har_1h, har_6h      -- the candidate
  G_SR   + RAW har_1h, har_6h                 -- the control: same reactivity,
                                                 clock left in

    python -m model.eval.short_anchor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                     # noqa: E402
from eval.hour_anchor import (PROD_A, PROD_H, fold_anchors,            # noqa: E402
                              fold_tests, served_log_factor)
from eval.product_score import battery, gaussian_curves, per_episode   # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.season import season_fwd                                   # noqa: E402
from noctua.train import load_all                                      # noqa: E402

EP_METRICS = ("brier", "logs", "pinball", "crps")
ARMS = ("G_As", "G_SD", "G_SR")
CONTRASTS = (("G_SD", "G_As"), ("G_SR", "G_As"), ("G_SD", "G_SR"))


def season_back(profile, anchor_hour, x: int) -> np.ndarray:
    """0.5 log mean seasonal variance over the x clock hours ending at the
    anchor, [a - x, a), centred like season_fwd."""
    a = (np.asarray(anchor_hour, np.int64) - x) % 24
    return season_fwd(profile, a, np.full(len(a), x))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="deseasonalised reactive anchor")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/short_anchor.json"))
    a = ap.parse_args(argv)
    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    rv = ep["RV"].to_numpy(np.float64)
    H = ep.H.to_numpy(np.float64)
    fin = np.isfinite(X.to_numpy()).all(1)
    at19 = (ep.H == PROD_H).to_numpy()
    y = B.har_target(rv, H)
    sq = np.sqrt(PROD_H)
    M_up, M_dn = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy())
    base_cols = B.VOL_BASELINES["log_har_cal"]

    pe = {k: {m: [] for m in EP_METRICS} for k in ARMS}
    fold_bat = {k: [] for k in ARMS}
    coefs = []
    for f in S.walk_forward_folds(ep):
        anc = fold_anchors(ep, X, f)
        prof = anc["profile"]
        m_tr = np.asarray(f["train"], bool) & fin
        wtr = S.sample_weights(ep, m_tr)
        sf = season_fwd(prof, ah, H)
        cols_sd = base_cols + ["season_fwd", "h1_ds", "h6_ds"]
        cols_sr = base_cols + ["season_fwd", "h1", "h6"]
        X2 = X[base_cols].copy()
        X2["season_fwd"] = sf
        X2["h1"] = X["har_1h"].to_numpy(np.float64)
        X2["h6"] = X["har_6h"].to_numpy(np.float64)
        X2["h1_ds"] = X2["h1"] - season_back(prof, ah, 1)
        X2["h6_ds"] = X2["h6"] - season_back(prof, ah, 6)
        anchors = {"G_As": anc["A"]}
        for arm, cols in (("G_SD", cols_sd), ("G_SR", cols_sr)):
            ols = B.OLS(cols).fit(X2[m_tr], y[m_tr], wtr)
            v = np.full(len(ep), np.nan); v[fin] = ols.predict(X2[fin])
            anchors[arm] = v
            coefs.append({"year": f["year"], "arm": arm,
                          "short_coefs": [float(c) for c in ols.beta[-2:]]})
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & at19 & (ah == PROD_A)
                            & np.isfinite(rv) & (rv > 0))
        hist = np.flatnonzero(at19 & fin & np.isfinite(rv) & (rv > 0)
                              & (ts >= ts[np.asarray(f["calib"], bool)].min())
                              & (ts <= ts[te].max()))
        M = {"up": M_up[te], "dn": M_dn[te]}
        for arm in ARMS:
            an = anchors[arm]
            lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(an[hist]) * sq)
            cur = gaussian_curves(np.exp(an[te] + lf) * sq)
            fold_bat[arm].append(battery(cur, M))
            p = per_episode(cur, M)
            for m in EP_METRICS:
                pe[arm][m].append(p[m])
        print(f"  fold {f['year']}: DSC " + "  ".join(
            f"{k} {fold_bat[k][-1]['DSC']:.6f}" for k in ARMS)
            + "   short coefs SD " + str(np.round(coefs[-2]["short_coefs"], 3).tolist())
            + " SR " + str(np.round(coefs[-1]["short_coefs"], 3).tolist()), flush=True)

    n_family = len(CONTRASTS) * (len(EP_METRICS) + 1)
    alpha = 0.05 / n_family
    out = {"alpha": alpha, "coefs": coefs, "ci": {},
           "fold_dsc": {k: [b["DSC"] for b in v] for k, v in fold_bat.items()}}
    print(f"\nfamily {n_family} -> {100*(1-alpha):.2f}%; per-episode block 38; DSC fold t")
    res = {c: {"better": [], "worse": []} for c in CONTRASTS}
    for c, o in CONTRASTS:
        for m in EP_METRICS:
            d = np.concatenate(pe[o][m]) - np.concatenate(pe[c][m])
            lo, hi = mean_ci(d, alpha=alpha, block_len=38)["ci95"]
            pct = 100 * d.mean() / np.concatenate(pe[o][m]).mean()
            out["ci"][f"{m}_{c}_vs_{o}"] = [float(pct), float(lo), float(hi)]
            if lo > 0: res[(c, o)]["better"].append(m)
            if hi < 0: res[(c, o)]["worse"].append(m)
            print(f"  {m:>8} {c+'-'+o:>11} {pct:+6.2f}%  [{lo:+.6f}, {hi:+.6f}]")
        d = np.array([b["DSC"] for b in fold_bat[c]]) - np.array([b["DSC"] for b in fold_bat[o]])
        ft = fold_tests(d, alpha)
        n_up = int((d > 0).sum())
        out["ci"][f"DSC_{c}_vs_{o}"] = {"delta": float(d.mean()), "t": ft["t_ci"],
                                        "p_flip": ft["p_flip"], "folds_better": n_up}
        if ft["t_ci"][0] > 0: res[(c, o)]["better"].append("DSC")
        if ft["t_ci"][1] < 0: res[(c, o)]["worse"].append("DSC")
        print(f"       DSC {c+'-'+o:>11} {d.mean():+.6f}  t [{ft['t_ci'][0]:+.6f}, "
              f"{ft['t_ci'][1]:+.6f}]  {n_up}/{len(d)} folds  p_flip {ft['p_flip']:.3f}")
    sd, sr, dr = res[("G_SD", "G_As")], res[("G_SR", "G_As")], res[("G_SD", "G_SR")]
    dsc_folds = out["ci"]["DSC_G_SD_vs_G_As"]["folds_better"]
    ep_better = [m for m in sd["better"] if m in EP_METRICS]
    met = (len(ep_better) >= 2 and not sd["worse"] and dsc_folds >= 5
           and len(dr["better"]) >= 1)
    print("\n--- registered rule: G_SD beats G_As on >= 2 of 4 per-episode metrics, worse on")
    print("    none, DSC better in >= 5 of 6 folds, and beats the raw control G_SR on >= 1 metric")
    print(f"   G_SD vs G_As: better {sd['better'] or 'nothing'}, worse {sd['worse'] or 'nothing'}, "
          f"DSC {dsc_folds}/6 folds")
    print(f"   G_SR vs G_As: better {sr['better'] or 'nothing'}, worse {sr['worse'] or 'nothing'}")
    print(f"   G_SD vs G_SR: better {dr['better'] or 'nothing'}, worse {dr['worse'] or 'nothing'}")
    print(f"   -> {'MET' if met else 'NOT MET'}")
    out.update(results={f"{c}_vs_{o}": v for (c, o), v in res.items()}, rule_met=bool(met))
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
