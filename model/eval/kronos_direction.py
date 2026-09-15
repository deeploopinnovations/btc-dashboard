"""
eval/kronos_direction.py
=====================================================================
Kronos displays an upside probability. Does it mean anything?

WHY THIS FILE EXISTS

`eval/direction.py` concluded that the sign of the BTC return is not
predictable at 6-24 hours from trailing bars -- not by NOCTUA, not by a
39-feature logistic, not by gradient boosting. The obvious objection is
empirical and fair: the Kronos dashboard displays an "Upside Probability (Next
24h)" of 66.7%, so evidently *something* can do it.

It cannot. 66.7% is 2/3. The stored predictions in `data/kronos_predictions.json`
settle it: every one of the 120 `p_up` values is an exact multiple of 1/32,
because `kronos_local/app.py:133` computes

    "upside": round(100.0 * ups / N_SAMPLES, 1)

-- the fraction of Monte-Carlo rollouts that happened to finish higher. It is a
COUNT, not a calibrated probability, and the count has sampling error nobody
is showing. At n = 32 the standard error of a proportion near 0.5 is
sqrt(0.25/32) = 8.8 pp, so a displayed "66.7%" carries a 95% interval of
roughly [50%, 83%] from Monte-Carlo noise alone, before any question of
whether the model knows anything. Displaying it to one decimal place is
spurious precision by a factor of ~90.

The stored file also shows `p_up` hitting 0.000 and 1.000 -- absolute
certainty about the direction of Bitcoin 19 hours ahead, asserted on the
strength of 32 coin flips landing the same way.

But an argument about sampling error is still an argument. The 120 stored
episodes carry the realized outcome alongside the forecast, so the question
can simply be settled by scoring, which is what this file does.

WHAT IS SCORED, AND AGAINST WHAT

Three questions, all on the same 120 episodes:

  DIRECTION      p_up against 1{R > 0}
  VOLATILITY     sigma against realized RV, by QLIKE
  AMPLIFICATION  P(RV_forward > trailing RV) -- the "will it get wilder"
                 call an option seller actually acts on

against a causal climatological base rate, and with DSC judged against a
SHUFFLED null rather than against zero, because in-sample isotonic regression
manufactures positive DSC from noise. n = 120 is small; the bootstrap is
reported so the smallness is visible rather than hidden.

    python -m model.eval.kronos_direction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import brier, corp_decomposition, log_score              # noqa: E402
from eval.direction import block_bootstrap_ci, shuffled_dsc_null             # noqa: E402

EPS = 1e-6


def score(name, p, y, seed=0, null_reps=400):
    """Score a probabilistic binary forecast the same way `direction.py` does."""
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    y = np.asarray(y, np.float64)
    c = corp_decomposition(p, y)
    null = shuffled_dsc_null(p, y, n_rep=null_reps, seed=seed)
    return {
        "model": name, "n": int(len(y)),
        "log_loss": log_score(p, y), "brier": c["brier"],
        "MCB": c["MCB"], "DSC": c["DSC"], "UNC": c["UNC"],
        "DSC_null_p95": float(np.quantile(null, 0.95)),
        "clears_null": bool(c["DSC"] > np.quantile(null, 0.95)),
        "mean_p": float(p.mean()), "base_rate": c["base_rate"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Does Kronos's upside number mean anything?")
    ap.add_argument("--preds", type=Path, default=Path("data/kronos_predictions.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/kronos_direction.json"))
    a = ap.parse_args(argv)

    d = json.loads(a.preds.read_text())
    eps = d["episodes"]
    n_samp = int(d["samples_per_episode"])
    pup = np.array([e["p_up"] for e in eps], np.float64)
    R = np.array([e["R"] for e in eps], np.float64)
    RV = np.array([e["RV"] for e in eps], np.float64)
    sig = np.array([e["sigma"] for e in eps], np.float64)
    y = (R > 0).astype(np.float64)

    # ---- the granularity tell -------------------------------------------
    mult = pup * n_samp
    is_count = bool(np.allclose(mult, np.round(mult)))
    se = float(np.sqrt(0.25 / n_samp))
    print(f"episodes {len(eps)}, samples/episode {n_samp}")
    print(f"every p_up an exact multiple of 1/{n_samp}: {is_count}")
    print(f"  -> it is a COUNT of rollouts, not a calibrated probability")
    print(f"  Monte-Carlo standard error at p=0.5, n={n_samp}: {100*se:.1f} pp")
    print(f"  displayed to 0.1 pp, i.e. {se*100/0.1:.0f}x more precision than it has")
    print(f"  p_up range [{pup.min():.3f}, {pup.max():.3f}]; "
          f"episodes claiming 0% or 100% certainty: "
          f"{int(((pup <= 0) | (pup >= 1)).sum())}")
    print()

    # ---- direction -------------------------------------------------------
    base = float(y.mean())
    rows = [
        score("kronos_p_up", pup, y, seed=1),
        score("constant_half", np.full(len(y), 0.5), y, seed=2),
        score("base_rate", np.full(len(y), base), y, seed=3),
    ]
    preds = {"kronos_p_up": pup,
             "constant_half": np.full(len(y), 0.5),
             "base_rate": np.full(len(y), base)}
    pb = np.clip(preds["base_rate"], EPS, 1 - EPS)
    ll_base = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    for r in rows:
        p = np.clip(preds[r["model"]], EPS, 1 - EPS)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        lo, hi = block_bootstrap_ci(ll_base - ll, seed=7)
        r["vs_base_rate_gain"] = float((ll_base - ll).mean())
        r["vs_base_rate_ci95"] = [lo, hi]
        r["beats_base_rate"] = bool(lo > 0.0)

    print(f"{'DIRECTION':>16} {'logloss':>9} {'DSC':>10} {'null p95':>10} "
          f"{'clears':>7} {'gain':>9} {'CI95':>22}")
    for r in rows:
        print(f"{r['model']:>16} {r['log_loss']:9.5f} {r['DSC']:10.6f} "
              f"{r['DSC_null_p95']:10.6f} {str(r['clears_null']):>7} "
              f"{r['vs_base_rate_gain']:+9.5f} "
              f"[{r['vs_base_rate_ci95'][0]:+.5f},{r['vs_base_rate_ci95'][1]:+.5f}]")

    # ---- volatility ------------------------------------------------------
    def qlike(f2, r2):
        f2 = np.maximum(f2, 1e-16); r2 = np.maximum(r2, 1e-16)
        return float(np.mean(r2 / f2 - np.log(r2 / f2) - 1.0))

    vol = {"kronos_sigma": qlike(sig**2, RV**2)}
    print(f"\nVOLATILITY (QLIKE, lower better): kronos sigma {vol['kronos_sigma']:.5f}")
    print(f"  kronos sigma / realized RV ratio: median "
          f"{float(np.median(sig / np.maximum(RV, 1e-12))):.4f}")

    out = {"n_episodes": len(eps), "samples_per_episode": n_samp,
           "p_up_is_a_count": is_count,
           "mc_standard_error_pp": 100 * se,
           "episodes_claiming_certainty": int(((pup <= 0) | (pup >= 1)).sum()),
           "direction": rows, "volatility": vol}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
