"""
eval/dow_staleness.py
=====================================================================
Before the day-of-week increment is built: does it still work with STALE
coefficients?

P4-dow-anchor-result's day coefficients grow every fold (Friday +0.16 in the
2021 fold to +0.48 in 2026). An artifact increment fitted on the artifact's
training split (through 2023-01-01) carries roughly the 2023-era size. This
measures, on the no-network Gaussian law (eval/product_score) with serving's
trailing factor from each arm's own forecasts, over the test years that are
all AFTER that split (2023-2026):

  G0      the fold's own Log-HAR anchor
  Gfresh  + day increment refitted on the fold's train slice (what the
            walk-forward evaluated, in increment form)
  Gstale  + day increment fitted ONCE on the artifact's split and frozen
  Gjoint  the joint refit (the evaluated arm's form)

If Gstale keeps most of Gfresh's gain, the increment can be fitted on the
artifact's split like the others; if not, the fit window must be recent.

    python -m model.eval.dow_staleness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dow_anchor import extra_cols                                 # noqa: E402
from eval.weekend_fix import day_frac                                   # noqa: E402
from eval.hour_anchor import PROD_A, PROD_H, served_log_factor         # noqa: E402
from eval.product_score import gaussian_curves, per_episode            # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

ARMS = ("G0", "Gfresh", "Gstale", "Gjoint")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="day-of-week increment staleness")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/dow_staleness.json"))
    a = ap.parse_args(argv)
    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    Hs = ep.H.to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    rv = ep["RV"].to_numpy(np.float64)
    fin = np.isfinite(X.to_numpy()).all(1)
    at19 = (ep.H == PROD_H).to_numpy()
    y = B.har_target(rv, Hs.astype(float))
    sq = np.sqrt(PROD_H)
    M_up, M_dn = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy())
    cols = B.VOL_BASELINES["log_har_cal"]
    D = pd.DataFrame(extra_cols(ts, Hs)["Ds"])
    dcols = list(D.columns)
    # INCREMENT columns: the residual regression has no Fri+Sat column, so
    # Mon..Fri plus its intercept would force Saturday and Sunday to share a
    # level. Mon..Sat (Sunday the reference) spans the week, like the joint fit.
    # (First run used Mon..Fri: 2026 increment +0.110% vs joint +0.448%.)
    D["d5"] = day_frac(ts, Hs, (5,))
    icols = dcols + ["d5"]

    art = S.time_splits(ep)["train"] & fin
    w_art = S.sample_weights(ep, art)
    lhc_art = B.OLS(cols).fit(X[cols][art], y[art], w_art)
    r_art = y - lhc_art.predict(X[cols])
    inc_art = B.OLS(icols).fit(D[icols][art], r_art[art], w_art)
    print("stale increment (artifact split, through "
          f"{S.TRAIN_END}): day coefs {np.round(inc_art.beta[1:], 3).tolist()}")

    out = {"stale_coefs": inc_art.beta[1:].tolist(), "years": {}}
    for f in S.walk_forward_folds(ep):
        if f["year"] < 2023:
            continue
        m_tr = np.asarray(f["train"], bool) & fin
        w = S.sample_weights(ep, m_tr)
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & at19 & (ah == PROD_A)
                            & np.isfinite(rv) & (rv > 0))
        hist = np.flatnonzero(at19 & fin & np.isfinite(rv) & (rv > 0)
                              & (ts >= ts[np.asarray(f["calib"], bool)].min())
                              & (ts <= ts[te].max()))
        idx = np.concatenate([hist, te])
        lhc = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], w)
        base = np.full(len(ep), np.nan)
        base[idx] = lhc.predict(X[cols].iloc[idx])
        r = y - np.nan_to_num(lhc.predict(X[cols]))
        inc_f = B.OLS(icols).fit(D[icols][m_tr], r[m_tr], w)
        XJ = pd.concat([X[cols], D[dcols]], axis=1)
        joint = B.OLS(cols + dcols).fit(XJ[m_tr], y[m_tr], w)
        an = {"G0": base.copy(), "Gfresh": base.copy(), "Gstale": base.copy(),
              "Gjoint": np.full(len(ep), np.nan)}
        an["Gfresh"][idx] += inc_f.predict(D[icols].iloc[idx])
        an["Gstale"][idx] += inc_art.predict(D[icols].iloc[idx])
        an["Gjoint"][idx] = joint.predict(XJ.iloc[idx])
        M = {"up": M_up[te], "dn": M_dn[te]}
        sc = {}
        for k in ARMS:
            lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(an[k][hist]) * sq)
            p = per_episode(gaussian_curves(np.exp(an[k][te] + lf) * sq), M)
            sc[k] = {m: p[m] for m in ("brier", "logs")}
        row = {"fresh_coefs": inc_f.beta[1:].tolist()}
        for k in ARMS[1:]:
            for m in ("brier", "logs"):
                row[f"{k}_{m}_gain_pct"] = float(100 * (sc["G0"][m] - sc[k][m]).mean()
                                                 / sc["G0"][m].mean())
        row["stale_share_of_fresh_brier"] = (row["Gstale_brier_gain_pct"]
                                             / row["Gfresh_brier_gain_pct"])
        out["years"][f["year"]] = row
        print(f"  {f['year']}: Brier gain  fresh {row['Gfresh_brier_gain_pct']:+.3f}%  "
              f"stale {row['Gstale_brier_gain_pct']:+.3f}%  joint "
              f"{row['Gjoint_brier_gain_pct']:+.3f}%   (stale/fresh "
              f"{row['stale_share_of_fresh_brier']:.2f})   fresh coefs "
              f"{np.round(inc_f.beta[1:], 3).tolist()}", flush=True)
    yrs = out["years"].values()
    out["mean_stale_share"] = float(np.mean([r["stale_share_of_fresh_brier"] for r in yrs]))
    print(f"\nmean stale/fresh Brier-gain share over 2023-2026: {out['mean_stale_share']:.2f}")
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
