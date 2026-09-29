"""
eval/either.py
=====================================================================
Does modelling the max excursion directly beat assuming independence?

THE QUESTION THE PRODUCT ACTUALLY ASKS

A seller short both wings of a strangle is not exposed to the up barrier or
to the down barrier. They are exposed to **either**. That quantity --
P(max(M_up, M_dn) >= u) -- is the one the position lives or dies on, and
until `q_mx` the model did not produce it. Anyone needing it had to build it
from the two marginals as `1 - (1-p_up)(1-p_dn)`, which assumes the sides are
independent.

They are not, and the dependence is large and mechanical: a path with a fixed
realized-variance budget cannot spend it travelling up AND down. Measured on
5,324 production episodes, Spearman(M_up/RV, M_dn/RV) = **-0.687**. A
driftless Brownian control simulated at the same 5-minute resolution gives
**-0.812** -- so the structure is, if anything, STRONGER in the textbook than
in Bitcoin. Independence is the wrong assumption for any path model.

Its cost runs in the direction that hurts a seller. Realized against the
independent construction, on the same episodes:

    barrier   P(either) realized   assuming independence
      1.0%          0.8922                0.8368
      2.0%          0.6362                0.5995
      3.0%          0.4401                0.4231

Independence UNDERSTATES the chance of being broken, by up to 5.5 pp. A
strangle looks safer than it is.

WHAT IS COMPARED

  q_mx_head       the new dedicated head, mixed over the 32 sigma atoms
  independence    1 - (1-p_up)(1-p_dn) from the existing marginals
  gaussian        the textbook two-sided first-passage law under the model's
                  own sigma, for reference
  climatology     the training-era base rate at each barrier

scored with the same machinery as everything else: CORP decomposition, DSC
against a SHUFFLED null rather than against zero, moving-block bootstrap.

DECISION RULE, fixed before the run: adopt `q_mx` only if it beats the
independence construction on log loss in at least 5 of 6 folds, at the
barriers a seller actually writes (1%-3%). A win at one barrier or 4/6 is
noise and gets reported as null.

    python -m model.eval.either
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import corp_decomposition, log_score                     # noqa: E402
from eval.direction import block_bootstrap_ci, shuffled_dsc_null             # noqa: E402
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import load_all, prepare, train_model                      # noqa: E402

EPS = 1e-6
BARRIERS = (1.0, 2.0, 3.0)


def run_fold(ep, X, fold, hidden=32, seeds=3):
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
        avg[k] = np.mean([p[k] for p in preds], axis=0)

    e = ep[m_te]
    mu = np.abs(e["M_up"].to_numpy(np.float64))
    md = np.abs(e["M_dn"].to_numpy(np.float64))
    sig = avg["sigma_med"]
    n = int(m_te.sum())

    out = {"year": fold["year"], "n": n, "barriers": {}}
    for pct in BARRIERS:
        u = np.full(n, np.log1p(pct / 100.0))
        y = ((mu >= u) | (md >= u)).astype(np.float64)
        p_up = I.touch_prob(avg, u, True)
        p_dn = I.touch_prob(avg, u, False)
        # base rate from the TRAINING episodes only -- causal climatology
        etr = ep[m_tr]
        y_tr = ((np.abs(etr["M_up"].to_numpy()) >= u[0])
                | (np.abs(etr["M_dn"].to_numpy()) >= u[0])).astype(np.float64)
        out["barriers"][str(pct)] = {
            "y": y,
            "q_mx_head": I.touch_prob_either(avg, u),
            "independence": np.clip(1.0 - (1.0 - p_up) * (1.0 - p_dn), 0.0, 1.0),
            # two-sided Brownian, leading reflection term, on the model's sigma
            "gaussian": np.clip(4.0 * norm.cdf(-u / np.maximum(sig, 1e-12)),
                                0.0, 1.0),
            "climatology": np.full(n, float(np.average(y_tr,
                                                       weights=S.sample_weights(ep, m_tr)))),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Model the max, or assume independence?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/either.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    recs = []
    for f in folds:
        t0 = time.time()
        r = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds)
        if r is None:
            print(f"  {f['year']}: SKIPPED")
            continue
        recs.append(r)
        b = r["barriers"]["2.0"]
        print(f"  {f['year']}: n={r['n']:4,}  realized P(either|2%)="
              f"{b['y'].mean():.4f}  head={b['q_mx_head'].mean():.4f}  "
              f"indep={b['independence'].mean():.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

    if not recs:
        return 1

    names = ["q_mx_head", "independence", "gaussian", "climatology"]
    report = {}
    print(f"\n{'barrier':>8} {'model':>14} {'logloss':>9} {'brier':>8} "
          f"{'DSC/UNC':>9} {'clears':>7} {'vs indep':>10} {'CI95':>22} {'folds':>6}")
    for pct in BARRIERS:
        k = str(pct)
        y = np.concatenate([r["barriers"][k]["y"] for r in recs])
        p_ind = np.clip(np.concatenate(
            [r["barriers"][k]["independence"] for r in recs]), EPS, 1 - EPS)
        ll_ind = -(y * np.log(p_ind) + (1 - y) * np.log(1 - p_ind))
        report[k] = {}
        for nm in names:
            p = np.clip(np.concatenate(
                [r["barriers"][k][nm] for r in recs]), EPS, 1 - EPS)
            c = corp_decomposition(p, y)
            null = shuffled_dsc_null(p, y, n_rep=a.null_reps, seed=17)
            ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            lo, hi = block_bootstrap_ci(ll_ind - ll, seed=19)
            won = sum(1 for r in recs
                      if log_score(np.clip(r["barriers"][k][nm], EPS, 1 - EPS),
                                   r["barriers"][k]["y"])
                      < log_score(np.clip(r["barriers"][k]["independence"],
                                          EPS, 1 - EPS), r["barriers"][k]["y"]))
            report[k][nm] = {
                "log_loss": log_score(p, y), "brier": c["brier"],
                "MCB": c["MCB"], "DSC": c["DSC"], "UNC": c["UNC"],
                "DSS": c["DSC"] / max(c["UNC"], 1e-12),
                "clears_null": bool(c["DSC"] > np.quantile(null, 0.95)),
                "vs_indep_gain": float((ll_ind - ll).mean()),
                "vs_indep_ci95": [lo, hi], "folds_won": int(won),
                "mean_p": float(p.mean()), "base_rate": c["base_rate"]}
            r_ = report[k][nm]
            print(f"{pct:7.1f}% {nm:>14} {r_['log_loss']:9.5f} {r_['brier']:8.5f} "
                  f"{100*r_['DSS']:8.3f}% {str(r_['clears_null']):>7} "
                  f"{r_['vs_indep_gain']:+10.5f} "
                  f"[{lo:+.5f},{hi:+.5f}] {won:>4}/{len(recs)}")
        print()

    wins = sum(report[str(p)]["q_mx_head"]["folds_won"] for p in BARRIERS)
    tot = len(BARRIERS) * len(recs)
    print(f"q_mx head beats independence on {wins} of {tot} (barrier, fold) cells.")
    print("Decision rule fixed before the run: adopt only at >= 5 of 6 folds "
          "at each of the 1-3% barriers.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n_folds": len(recs), "barriers": report,
                                 "seeds": a.seeds, "hidden": a.hidden},
                                indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
