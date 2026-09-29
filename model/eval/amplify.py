"""
eval/amplify.py
=====================================================================
The one number the option seller actually trades on, finally scored.

WHAT THIS IS

The dashboard publishes `p_vol_amplify`: the model's probability that realized
volatility over the forward window will EXCEED the trailing realized
volatility over a window of the same length. In the seller's language that is

    "will it get wilder than it has been?"

and it is the sell-premium-versus-buy-the-straddle decision. If forward vol
lands below trailing, a short strangle collects; if it lands well above, the
strike breaks and the straddle would have paid.

**It had never been scored.** `BENCHMARK.md` measures the volatility LEVEL
(QLIKE) and the barrier touch probabilities (CORP, Christoffersen), both
thoroughly. The binary amplification call is derived from Stage A's quantile
curve by `serve/predict.py:155`, published on every refresh, and appears in no
evaluation anywhere in the repository. That is the same asymmetry of attention
that left direction unmeasured for months.

WHY IT IS NOT AUTOMATICALLY GOOD JUST BECAUSE QLIKE IS GOOD

A well-calibrated volatility LEVEL does not imply a well-calibrated
EXCEEDANCE probability. The level forecast can be right on average while the
predictive quantile SPREAD is too narrow or too wide, and the amplification
call reads the spread rather than the centre: it is `1 - F(trailing)`
evaluated on Stage A's own quantile curve. A model whose median is perfect but
whose distribution is 20% too tight will systematically understate the chance
of a wild night, which is precisely the error that bankrupts a seller.

There is also a mechanical trap worth naming. Trailing RV appears on BOTH
sides of this question -- it is the threshold, and it is (through `har_1d` and
its siblings) the dominant input to the forecast. Volatility mean-reverts, so
when trailing RV is high the forward is likely lower and vice versa. A
forecaster could therefore look skilled by doing nothing but tracking the
trailing level. The reference set below exists to strip that out.

THE COMPETITORS, chosen so the naive routes are all represented

  climatology       the training-era base rate of amplification. Constant, so
                    DSC = 0 by construction -- the thing to beat.
  har_gaussian      the calibrated Log-HAR level, with a lognormal spread
                    fitted on the training residuals. This is the "you did not
                    need a neural network" competitor.
  persistence       forward RV = trailing RV exactly, so P = 0.5 always.
  noctua            Stage A's own quantile curve.

DSC is judged against a SHUFFLED null, not against zero, because in-sample
isotonic regression manufactures positive DSC out of noise. Confidence
intervals use a moving-block bootstrap; consecutive episodes overlap.

    python -m model.eval.amplify
    python -m model.eval.amplify --assets eth,ltc,xrp     # unseen instruments
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

from eval.benchmark import brier, corp_decomposition, log_score              # noqa: E402
from eval.direction import block_bootstrap_ci, shuffled_dsc_null             # noqa: E402
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import load_all, prepare, train_model                      # noqa: E402

EPS = 1e-6


def amplification_label(ep: pd.DataFrame, X: pd.DataFrame) -> tuple:
    """(threshold, outcome) for the amplification question.

    The threshold is the trailing realized vol over a window of the SAME
    length as the forecast horizon, built from the causal feature `har_1d`
    (log hourly vol rate over the trailing day) rather than from any forward
    quantity. `episodes.RV1` would be the tempting choice and is wrong: it is
    built from fwd_rv1 and looks forward from the anchor.
    """
    H = ep["H"].to_numpy(np.float64)
    trail = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)
    fwd = ep["RV"].to_numpy(np.float64)
    return trail, (fwd > trail).astype(np.float64)


def noctua_p_amplify(pred: dict, thresh: np.ndarray) -> np.ndarray:
    """P(window vol > threshold) read off Stage A's predictive quantiles.

    Vectorised twin of `serve/predict.py:model_prob_rv_above`, which handles
    one episode at a time.
    """
    qa = pred["qa"]
    H = np.asarray(pred["H"], np.float64)
    tot = np.exp(qa) * np.sqrt(H)[:, None]          # (n, K) window vol per level
    out = np.empty(len(thresh))
    for i in range(len(thresh)):
        out[i] = 1.0 - np.interp(thresh[i], tot[i], I.LEVELS, left=0.0, right=1.0)
    return np.clip(out, 0.0, 1.0)


def gaussian_p_amplify(mu_log: np.ndarray, sd_log: float,
                       thresh: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Lognormal competitor: log window vol ~ N(mu, sd), sd fitted on train."""
    from scipy.stats import norm
    mu_win = mu_log + 0.5 * np.log(H)               # log(exp(mu)*sqrt(H))
    z = (np.log(np.maximum(thresh, 1e-12)) - mu_win) / max(sd_log, 1e-9)
    return np.clip(1.0 - norm.cdf(z), 0.0, 1.0)


def score(name, p, y, seed=0, null_reps=200):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    y = np.asarray(y, np.float64)
    c = corp_decomposition(p, y)
    null = shuffled_dsc_null(p, y, n_rep=null_reps, seed=seed)
    return {"model": name, "n": int(len(y)), "log_loss": log_score(p, y),
            "brier": c["brier"], "MCB": c["MCB"], "DSC": c["DSC"], "UNC": c["UNC"],
            "DSS": c["DSC"] / max(c["UNC"], 1e-12),
            "DSC_null_p95": float(np.quantile(null, 0.95)),
            "clears_null": bool(c["DSC"] > np.quantile(null, 0.95)),
            "mean_p": float(p.mean()), "base_rate": c["base_rate"]}


def run_fold(ep, X, fold, trail, y_amp, hidden=32, seeds=3):
    fin = np.isfinite(X.to_numpy()).all(1)
    prod = S.production_mask(ep)
    m_tr = fold["train"] & fin
    m_te = fold["test"] & fin & prod
    if m_tr.sum() < 5000 or m_te.sum() < 30:
        return None

    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)
    lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
    sref = np.maximum(np.clip(raw, lo, hi), 1e-12)

    tr, stds = prepare(ep, X, m_tr, sigma_ref=sref[m_tr])
    wtr = S.sample_weights(ep, m_tr)
    yall = B.har_target(ep.RV.to_numpy(), H)
    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), wtr)
    bl = B.fit_vol_baselines(X[m_tr], yall[m_tr], wtr)
    models = [train_model(tr, wtr, tr, hidden=hidden, epochs=40, seed=s,
                          ols_beta=ols.beta)[0] for s in range(seeds)]

    d, _ = prepare(ep, X, m_te, *stds)
    lp = bl["log_har_cal"].predict(X[m_te])
    preds = [I.predict(m, d, har_logvol=lp) for m in models]
    avg = dict(preds[0])
    for k in ("qa", "sigma_atoms", "sigma_med", "q_r", "q_up", "q_dn", "q_mx"):
        if all(k in p for p in preds):
            avg[k] = np.mean([p[k] for p in preds], axis=0)

    # lognormal spread fitted on the TRAINING residuals only
    lp_tr = bl["log_har_cal"].predict(X[m_tr])
    resid = yall[m_tr] - lp_tr
    sd_log = float(np.std(resid[np.isfinite(resid)]))

    t_te, y_te = trail[m_te], y_amp[m_te]
    base = float(np.average(y_amp[m_tr], weights=wtr))
    out = {
        "noctua":      noctua_p_amplify(avg, t_te),
        "har_gaussian": gaussian_p_amplify(lp, sd_log, t_te, H[m_te]),
        "persistence": np.full(int(m_te.sum()), 0.5),
        "climatology": np.full(int(m_te.sum()), base),
    }
    return {"year": fold["year"], "preds": out, "y": y_te,
            "n": int(m_te.sum()), "sd_log": sd_log}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the sell-vs-straddle call")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/amplify.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    trail, y_amp = amplification_label(ep, X)
    prod = S.production_mask(ep)
    print(f"episodes {len(ep):,}; production slice {prod.sum():,}")
    print(f"unconditional amplification rate (production): "
          f"{y_amp[prod].mean():.4f}\n")

    folds = S.walk_forward_folds(ep)
    recs = []
    for f in folds:
        t0 = time.time()
        r = run_fold(ep, X, f, trail, y_amp, hidden=a.hidden, seeds=a.seeds)
        if r is None:
            print(f"  {f['year']}: SKIPPED")
            continue
        recs.append(r)
        print(f"  {f['year']}: n={r['n']:4,}  amp_rate={r['y'].mean():.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not recs:
        print("no fold produced a result")
        return 1

    names = list(recs[0]["preds"])
    y_pool = np.concatenate([r["y"] for r in recs])
    p_clim = np.concatenate([r["preds"]["climatology"] for r in recs])
    ll_clim = -(y_pool * np.log(np.clip(p_clim, EPS, 1 - EPS))
                + (1 - y_pool) * np.log(np.clip(1 - p_clim, EPS, 1 - EPS)))

    rows = []
    for nm in names:
        p = np.concatenate([r["preds"][nm] for r in recs])
        s = score(nm, p, y_pool, seed=11, null_reps=a.null_reps)
        ll = -(y_pool * np.log(np.clip(p, EPS, 1 - EPS))
               + (1 - y_pool) * np.log(np.clip(1 - p, EPS, 1 - EPS)))
        lo, hi = block_bootstrap_ci(ll_clim - ll, seed=13)
        s["vs_clim_gain"] = float((ll_clim - ll).mean())
        s["vs_clim_ci95"] = [lo, hi]
        s["beats_clim"] = bool(lo > 0.0)
        s["folds_won"] = int(sum(
            1 for r in recs
            if log_score(np.clip(r["preds"][nm], EPS, 1 - EPS), r["y"])
            < log_score(np.clip(r["preds"]["climatology"], EPS, 1 - EPS), r["y"])))
        rows.append(s)

    print(f"\n{'model':>14} {'logloss':>9} {'brier':>8} {'MCB':>9} {'DSC/UNC':>9} "
          f"{'clears':>7} {'gain':>9} {'CI95':>22} {'folds':>6}")
    for r in rows:
        print(f"{r['model']:>14} {r['log_loss']:9.5f} {r['brier']:8.5f} "
              f"{r['MCB']:9.5f} {100*r['DSS']:8.3f}% {str(r['clears_null']):>7} "
              f"{r['vs_clim_gain']:+9.5f} "
              f"[{r['vs_clim_ci95'][0]:+.5f},{r['vs_clim_ci95'][1]:+.5f}] "
              f"{r['folds_won']:>4}/{len(recs)}")
    print("\n  'gain' is mean log-loss improvement over the causal climatology, "
          "in nats.\n  DSC/UNC is the Brier skill score; 'clears' is DSC above "
          "the shuffled null's p95.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"n_folds": len(recs), "pooled": rows,
         "per_fold": [{"year": r["year"], "n": r["n"],
                       "amp_rate": float(r["y"].mean())} for r in recs]},
        indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
