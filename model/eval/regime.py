"""
eval/regime.py
=====================================================================
The model was fitted on a market that no longer exists, and one of its
"regime" features is an untrained random offset switched on in production.

WHAT BTC ACTUALLY DID

Measured on the production anchors (H = 19, 17:00 UTC), median 19-hour
realized volatility by year:

    2013  5.28%      2021  3.32%      2024  2.05%
    2017  4.01%      2022  2.55%      2025  1.52%
    2019  2.41%      2023  1.71%      2026  1.61%

A 60-70% decline, and the fat tails went with it: return kurtosis falls from
15.04 in 2013 to 0.41 in 2026, and the 95th percentile of realized vol from
21.3% to 3.3%. Split at the spot-ETF launch (2024-01-11):

    pre   n=4,384   median RV 2.975%   p95 9.188%
    post  n=  941   median RV 1.736%   p95 3.827%
    ratio 0.584     Mann-Whitney p = 3.1e-156     KS = 0.4344

This is not a subtle drift. The deployed model is asked about a market
roughly 42% quieter than the one its training median describes, with tails
that have all but disappeared -- and it shows: `eval/either.py` finds every
model-based arm over-forecasting in the 2025 and 2026 folds (realized P(either
touched at 2%) 0.397 and 0.418 against predictions of 0.614 and 0.636), and
`serve/adaptive.py` has been quietly applying a 0.93-0.96 shrink for months.

THE DEFECT THIS EXPOSED

`reg_post_etf` is `hour_ts >= 2024-01-11`. In the SHIPPED split (train ends
2023-01-01) it is:

    train   189,831 rows, identically 0, sd 0 -> standardizes to exactly 0.0
    test     73,867 rows, identically 1       -> standardizes to exactly 1.0

A constant input receives zero gradient, so its first-layer weight never moves
from random initialization. Measured on the shipped artifact, that weight has
entirely typical magnitude -- |contribution| mean 0.088-0.114 per seed against
a mean |W| over all inputs of 0.096-0.116. So **at serve time every hidden
unit of both stages receives a full-strength, never-trained random offset that
was identically absent during training.**

It is not a weak feature. It is noise, injected only in production, and it
cannot be anything else: a flag that never varies in training carries no
information by construction, so there is nothing to lose by removing it and an
untrained perturbation to gain by doing so.

WHY THE WALK-FORWARD MOSTLY HIDES IT

Fold 2021 trains to 2020-07 and tests 2021 -- the flag is 0 on both sides, so
nothing flips. Only fold 2024 (train all pre-ETF, test partly post) and fold
2025 expose it at all. The SHIPPED split exposes it fully, which is why this
file scores that split directly rather than relying on the walk-forward alone.

DECISION RULE, fixed before running: `reg_post_etf` is dropped from the model
inputs if the shipped-split test scores are no worse. This is deliberately a
weaker bar than the >= 5/6 used for genuine improvements, and the reason is
that the feature is *provably* uninformative in training -- the question is
not "does removing it help" but "does removing an untrained random offset
hurt", and the answer should be no.

    python -m model.eval.regime
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

from eval.benchmark import run_fold                                       # noqa: E402
from eval.efficiency import summarise                                     # noqa: E402
from noctua import splits as S                                           # noqa: E402
from noctua.spec import SHAPE_COLS                                       # noqa: E402
from noctua.train import load_all                                        # noqa: E402

DROP = "reg_post_etf"


def causal_sigma_fn(ep, X):
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def era_table(ep, X) -> list:
    prod = S.production_mask(ep)
    e = ep[prod].copy()
    e["year"] = pd.to_datetime(e["anchor_ts"], unit="s", utc=True).dt.year
    rows = []
    for y, g in e.groupby("year"):
        r = g["RV"].to_numpy(np.float64)
        mx = np.maximum(np.abs(g["M_up"].to_numpy()), np.abs(g["M_dn"].to_numpy()))
        rows.append({"year": int(y), "n": int(len(g)),
                     "median_rv_pct": float(100 * np.median(r)),
                     "p95_rv_pct": float(100 * np.quantile(r, 0.95)),
                     "p_move_gt_2pct": float((mx >= np.log1p(0.02)).mean()),
                     "kurtosis": float(pd.Series(g["R"].to_numpy()).kurtosis())})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regime shift, and the untrained flag")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/regime.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    eras = era_table(ep, X)
    print("median 19h realized vol by year (production anchors):")
    for r in eras:
        print(f"  {r['year']}  n={r['n']:4}  median {r['median_rv_pct']:6.3f}%  "
              f"p95 {r['p95_rv_pct']:6.3f}%  P(|move|>2%) "
              f"{100*r['p_move_gt_2pct']:5.1f}%  kurt {r['kurtosis']:6.2f}")

    sp = S.time_splits(ep)
    v_tr = X.loc[sp["train"], DROP].to_numpy()
    v_te = X.loc[sp["test"], DROP].to_numpy()
    print(f"\n{DROP} in the SHIPPED split: train sd {np.nanstd(v_tr):.6f} "
          f"(n={sp['train'].sum():,}), test sd {np.nanstd(v_te):.6f} "
          f"(n={sp['test'].sum():,})")
    if np.nanstd(v_tr) > 1e-9:
        print("  NOTE: not constant in training here -- the defect does not apply")

    # ---- walk-forward: with and without the flag --------------------------
    keep_shape = [c for c in SHAPE_COLS if c != DROP]
    folds = S.walk_forward_folds(ep)
    sig_fn = causal_sigma_fn(ep, X)
    X_drop = X.drop(columns=[DROP])
    sig_fn_drop = causal_sigma_fn(ep, X_drop)

    recs = []
    for f in folds:
        line = {"year": f["year"]}
        tr_sd = float(np.nanstd(X.loc[f["train"], DROP].to_numpy()))
        line["train_sd_of_flag"] = tr_sd
        for name, Xa, sc, sf in (("with_flag", X, None, sig_fn),
                                 ("without_flag", X_drop, keep_shape, sig_fn_drop)):
            t0 = time.time()
            out = run_fold(ep, Xa, f, hidden=a.hidden, seeds=a.seeds,
                           shape_cols=sc, sigma_ref_fn=sf)
            if out is None:
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            line[name] = s
            print(f"  {f['year']}  {name:13} (flag sd in train {tr_sd:.3f})  "
                  f"DSC/UNC {s['DSS']:.5f}  QLIKE {s['qlike']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if "with_flag" in line and "without_flag" in line:
            recs.append(line)

    print(f"\n{'metric':>10} {'with flag':>11} {'without':>11} {'delta':>11} {'wins':>7}")
    report = {}
    for key, sgn in (("DSS", +1), ("pinball", -1), ("crps", -1), ("qlike", -1)):
        a0 = np.array([r["with_flag"][key] for r in recs])
        a1 = np.array([r["without_flag"][key] for r in recs])
        d = sgn * (a1 - a0)
        report[key] = {"with": float(a0.mean()), "without": float(a1.mean()),
                       "wins": int((d > 0).sum()), "n": len(recs)}
        print(f"{key:>10} {a0.mean():11.6f} {a1.mean():11.6f} "
              f"{a1.mean()-a0.mean():+11.6f} {int((d>0).sum()):>4}/{len(recs)}")
    print("\n  Folds where the flag is CONSTANT in training are the ones that")
    print("  expose the defect; folds where it varies could learn a response.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"eras": eras, "folds": recs, "summary": report,
                                 "dropped": DROP}, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
