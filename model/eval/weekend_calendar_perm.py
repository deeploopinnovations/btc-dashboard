"""
eval/weekend_calendar_perm.py
=====================================================================
Audit check for P4-weekend-fix-result: IS IT THE CALENDAR?

The registered placebo was one pair (Tue+Wed). This runs the full permutation:
every unordered pair of weekdays (21) and every single day (7) enters the
Log-HAR anchor as one extra column, fitted on each fold's train slice exactly
like the candidate, with serving's trailing factor from the arm's OWN
forecasts, and is scored on the 2,046 production nights through the
no-network Gaussian first-passage law (eval/product_score, validated for
relative comparisons). The calendar claim predicts Sat+Sun at or near the top;
the audit's disqualifier is "a non-calendar control reproduces >= half the
gain".

    python -m model.eval.weekend_calendar_perm
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.hour_anchor import PROD_A, PROD_H, served_log_factor         # noqa: E402
from eval.product_score import gaussian_curves, per_episode            # noqa: E402
from eval.weekend_fix import day_frac                                  # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="calendar permutation of the weekend column")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/weekend_calendar_perm.json"))
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
    sets = {"base": None}
    sets.update({"+".join(DAYS[d] for d in p): p for p in itertools.combinations(range(7), 2)})
    sets.update({DAYS[d]: (d,) for d in range(7)})
    extra = {k: (None if p is None else day_frac(ts, Hs, p)) for k, p in sets.items()}

    loss = {k: {"brier": [], "logs": []} for k in sets}
    coef = {k: [] for k in sets if k != "base"}
    for f in S.walk_forward_folds(ep):
        m_tr = np.asarray(f["train"], bool) & fin
        w = S.sample_weights(ep, m_tr)
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & at19 & (ah == PROD_A)
                            & np.isfinite(rv) & (rv > 0))
        hist = np.flatnonzero(at19 & fin & np.isfinite(rv) & (rv > 0)
                              & (ts >= ts[np.asarray(f["calib"], bool)].min())
                              & (ts <= ts[te].max()))
        M = {"up": M_up[te], "dn": M_dn[te]}
        for k, col in extra.items():
            X2 = X[cols].copy()
            cc = list(cols)
            if col is not None:
                X2["extra"] = col
                cc = cc + ["extra"]
            ols = B.OLS(cc).fit(X2[m_tr], y[m_tr], w)
            an = np.full(len(ep), np.nan)
            idx = np.concatenate([hist, te])
            an[idx] = ols.predict(X2.iloc[idx])
            if col is not None:
                coef[k].append(float(ols.beta[-1]))
            lf = served_log_factor(ts[te], ts[hist], rv[hist], np.exp(an[hist]) * sq)
            p = per_episode(gaussian_curves(np.exp(an[te] + lf) * sq), M)
            loss[k]["brier"].append(p["brier"])
            loss[k]["logs"].append(p["logs"])
        print(f"  fold {f['year']} done", flush=True)

    base = {m: np.concatenate(loss["base"][m]) for m in ("brier", "logs")}
    rows = []
    for k in sets:
        if k == "base":
            continue
        g = {m: float(np.mean(base[m] - np.concatenate(loss[k][m]))) for m in ("brier", "logs")}
        rows.append({"set": k, "n_days": len(sets[k]), **g,
                     "brier_pct": 100 * g["brier"] / float(base["brier"].mean()),
                     "coef_mean": float(np.mean(coef[k]))})
    out = {"n_nights": int(len(base["brier"])), "rows": rows}
    for nd in (2, 1):
        rr = sorted([r for r in rows if r["n_days"] == nd], key=lambda r: -r["brier"])
        print(f"\n{'pairs' if nd == 2 else 'single days'} ranked by per-episode Brier gain "
              f"over the base (Gaussian law, served factor), {out['n_nights']} nights:")
        for i, r in enumerate(rr, 1):
            print(f"  {i:2d}. {r['set']:>8}  brier {r['brier']:+.6f} ({r['brier_pct']:+.3f}%)  "
                  f"logs {r['logs']:+.6f}  mean coef {r['coef_mean']:+.3f}")
        if nd == 2:
            ss = next(r for r in rr if r["set"] == "Sat+Sun")
            others = [r for r in rr if r["set"] != "Sat+Sun"]
            best_other = max(others, key=lambda r: r["brier"])
            out["sat_sun_rank"] = 1 + [r["set"] for r in rr].index("Sat+Sun")
            out["best_non_weekend_pair"] = best_other["set"]
            out["best_non_weekend_share_of_sat_sun"] = best_other["brier"] / ss["brier"]
            print(f"\n  Sat+Sun rank {out['sat_sun_rank']}/21; best other pair "
                  f"{best_other['set']} reaches {100 * out['best_non_weekend_share_of_sat_sun']:.0f}% "
                  f"of Sat+Sun's Brier gain")
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
