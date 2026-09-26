"""
eval/shape_relevance.py
=====================================================================
Does conditioning on PREDICTED path shape actually improve a barrier
forecast -- and is the instrument that framed the question even usable?

WHY THIS EXISTS SEPARATELY FROM eval/shape.py

`eval/shape.py` measured whether the scale-free travel ratio is predictable
and concluded "predictable and useful" on all four arms. Two of its three
criteria do not survive checking, so this file re-runs the decisive part.

  1. It read "clears the permutation null" as evidence of skill. For the
     gradient-boosted arms the null sits at R^2 = -0.047 and the real R^2 is
     -0.016. Beating a null centred well below zero means "less bad than
     fitting shuffled targets", not "predictive". An out-of-sample R^2 below
     zero means the model loses to the constant it is being compared against.

  2. Its relevance test was `mean |change| in P(touch 2%)`, an ABSOLUTE value.
     Pure noise passes that: the reported mean |delta| is 1.05-2.53 pp while
     the mean SIGNED delta is 0.01-0.40 pp. ROADMAP Priority 1 asked whether
     shape "moves the 2% touch probability by >= 1 pp", and meant IMPROVES.
     The ambiguity was in the rule; the correct reading is not ambiguous.

So the test here is the one that matters to a seller: does using the shape
prediction make the touch forecast BETTER than using a fitted constant shape?
Scored with the Brier score on the realized touch indicator, both arms fed the
REALIZED volatility so that only shape differs, with a moving-block bootstrap
CI on the paired difference.

AND THE THING THAT TURNED UP ON THE WAY

The comparison needs a floor, so this also scores the base rate -- a forecaster
that ignores every input and quotes the historical frequency. The Gaussian
first-passage law, fed PERFECT volatility and a shape constant fitted on the
training split, loses to it by 1.62x at 2%, 2.46x at 3% and 6.55x at 5%.

That matters far beyond the shape question, because `eval/firstpassage.py`
computes its "a perfect volatility forecast removes only 6-8.5% of the barrier
error" decomposition THROUGH that same law. The decomposition is therefore a
statement about the Gaussian first-passage forecaster, not about NOCTUA --
which does not use it. NOCTUA reads barrier probabilities off the learned
Stage B quantile heads via `survival_from_quantiles`, and beats the constant
(Brier 0.19607 against climatology 0.21666 at the 2% up barrier).
ROADMAP.md attributed the 92% figure to "the model". That was an overreach and
is corrected there.

    python -m model.eval.shape_relevance
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import block_bootstrap_ci                              # noqa: E402
from noctua import splits as S                                             # noqa: E402
from noctua.train import load_all                                          # noqa: E402

BARRIERS = (0.5, 1.0, 2.0, 3.0, 5.0)
# E[max(M_up, |M_dn|)/sqrt(RV)] for a Brownian path sampled as this data is --
# extremes fine, RV at 5 minutes. See firstpassage.brownian_control.
BROWNIAN_MAX = 1.2472760209437084


def p_touch(u: float, sigma: np.ndarray) -> np.ndarray:
    """Reflection principle, one-sided and therefore exact."""
    return np.clip(2.0 * norm.cdf(-abs(u) / np.maximum(sigma, 1e-12)), 0.0, 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is predicted shape worth using?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/shape_relevance.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    sp = S.time_splits(ep)
    fin = np.isfinite(X.to_numpy()).all(1)
    prod = S.production_mask(ep)
    tr, te = sp["train"] & fin & prod, sp["test"] & fin & prod

    RV = ep["RV"].to_numpy(np.float64)
    mx = np.maximum(np.abs(ep["M_up"].to_numpy(np.float64)),
                    np.abs(ep["M_dn"].to_numpy(np.float64)))
    y = mx / np.maximum(RV, 1e-12)

    Xa = np.nan_to_num(X.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-12
    Z = (Xa - mu) / sd
    pred = np.clip(Ridge(alpha=a.alpha).fit(Z[tr], y[tr]).predict(Z), 0.2, 4.0)
    r_clim = float(y[tr].mean())

    rho, pv = spearmanr(pred[te], y[te])
    r2 = 1.0 - ((y[te] - pred[te]) ** 2).sum() / ((y[te] - r_clim) ** 2).sum()
    print(f"train {tr.sum():,}  test {te.sum():,}")
    print(f"shape target, out-of-sample: Spearman {rho:+.4f} (p={pv:.2e})  "
          f"R^2 vs climatology {r2:+.5f}")
    print(f"climatology shape {r_clim:.4f}   predicted mean {pred[te].mean():.4f} "
          f"sd {pred[te].std():.4f}\n")

    print("Brier on the touch indicator; every arm fed the REALIZED volatility,")
    print("so the ONLY difference between the first two columns is shape.\n")
    print(f"{'barrier':>8} {'realized':>9} {'const shape':>12} {'pred shape':>11} "
          f"{'delta':>10} {'95% CI':>22}  {'base rate':>10} {'vs base':>8}")
    rows = []
    for pct in BARRIERS:
        u = np.log1p(pct / 100.0)
        hit = (mx[te] >= u).astype(np.float64)
        p = float(hit.mean())
        s_c = np.sqrt(RV[te]) * (r_clim / BROWNIAN_MAX)
        s_p = np.sqrt(RV[te]) * (pred[te] / BROWNIAN_MAX)
        b_c = (p_touch(u, s_c) - hit) ** 2
        b_p = (p_touch(u, s_p) - hit) ** 2
        d = b_p - b_c                              # negative => shape helps
        lo, hi = block_bootstrap_ci(d, seed=5)
        base = p * (1.0 - p)                       # constant at the base rate
        v = "HELPS" if hi < 0 else ("HURTS" if lo > 0 else "no effect")
        rows.append({"barrier_pct": pct, "realized": p,
                     "brier_const_shape": float(b_c.mean()),
                     "brier_pred_shape": float(b_p.mean()),
                     "delta": float(d.mean()), "ci95": [lo, hi],
                     "brier_base_rate": base,
                     "gauss_vs_base_ratio": float(b_c.mean() / base),
                     "verdict": v})
        print(f"{pct:7.1f}% {p:9.4f} {b_c.mean():12.5f} {b_p.mean():11.5f} "
              f"{d.mean():+10.5f} [{lo:+.5f},{hi:+.5f}] {base:10.5f} "
              f"{b_c.mean()/base:7.2f}x  {v}")

    print("\n  'vs base' > 1 means the Gaussian first-passage law, given PERFECT")
    print("  volatility and a fitted shape constant, is WORSE than quoting the")
    print("  historical base rate. NOCTUA does not use that law.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"n_train": int(tr.sum()), "n_test": int(te.sum()),
         "spearman": float(rho), "spearman_p": float(pv), "r2": float(r2),
         "clim_shape": r_clim, "brownian_max_ratio": BROWNIAN_MAX,
         "barriers": rows}, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
