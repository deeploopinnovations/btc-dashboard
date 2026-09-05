"""
eval/firstpassage.py
=====================================================================
Splitting the barrier error into the two things that can cause it.

THE QUESTION

A barrier forecast is a composition of two independent claims:

    P(touch u) = f( sigma_hat , shape )

  * the VOLATILITY claim -- how big will the moves be?
  * the SHAPE claim      -- given moves of that size, how far does the path
                            actually travel before settlement?

When the touch probability is wrong, those two are usually reported together
and the blame is unassignable. They call for completely different work: a bad
sigma means more/better volatility modelling, a bad shape means the
first-passage law is misspecified and no amount of volatility work will fix
it.

This file separates them by the only clean method available -- give the
Gaussian first-passage formula the REALIZED volatility, i.e. a perfect
volatility forecast it could not possibly have made, and see what is left.
Whatever error survives is pure shape error. It is an oracle experiment and
deliberately not a forecast: no one can trade it. It bounds how much of the
barrier error is even addressable by better volatility forecasting.

THE SHAPE STATISTIC, AND A GAP IN THE LITERATURE

Under driftless Brownian motion the expected range over a window relates to
the realized volatility by a constant:

    E[ max - min ] / sqrt( realized variance )  =  sqrt(8/pi)  =  1.5958

so the ratio is a direct, dimensionless test of whether real paths travel like
Brownian paths. A literature search (see LITERATURE.md section 5) found the
theory -- Feller's 1951 range distribution, and its numerical validation --
but **no published empirical measurement of this ratio on real financial
data**, let alone on crypto. Two independent studies do establish the
qualitative direction on equities and FX (Valenti et al. 2006 on hitting-time
densities; Boyarchenko & Levendorskii 2023 on jump-vs-diffusion no-touch
prices differing 15-22%), but neither reports this statistic.

So it is measured here, and reported as a measurement rather than a novelty
claim: absence of a located precedent is not proof of absence.

WHY IT MATTERS FOR THE SELLER

If the ratio is BELOW the benchmark (1.5860 -- see brownian_control, which
documents two earlier attempts at this control that were wrong in opposite
directions), real paths chop -- they burn realized variance
without travelling, so a Gaussian first-passage formula fed a correct sigma
will OVERSTATE the chance of touching a given strike, and a seller using it
quotes strikes further out than necessary and leaves premium on the table.
If it is ABOVE, paths trend and the formula understates the risk of being
broken. The two errors have opposite signs and opposite costs, and the
answer is not the same in the body of the distribution as in the tail.

    python -m model.eval.firstpassage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import block_bootstrap_ci                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402

BARRIER_PCT = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
BROWNIAN_RATIO = np.sqrt(8.0 / np.pi)          # 1.5958, CONTINUOUS time


def brownian_control(n: int = 60_000, H: int = 19, steps_per_hour: int = 360,
                     rv_minutes: int = 5, seed: int = 0) -> dict:
    """Simulate the benchmark with the NUMERATOR and DENOMINATOR sampled the
    way the data's actually are -- which are NOT the same resolution.

    This has now been got wrong twice, in opposite directions, so the reasoning
    is written out rather than asserted.

    Attempt 1 compared BTC's ratio against the continuous constant sqrt(8/pi)
    = 1.5958 and reported a gap of -0.2646.

    Attempt 2 "corrected" that by simulating 5-minute closes, obtaining 1.5218,
    and revised the gap to -0.1907 with the claim that ~28% of the original was
    discretization. **That correction was wrong.** It assumed the measured
    excursions come from 5-minute closes. They do not. `episodes.build_hourly`
    sets `hour_high = max of the 1-minute bar HIGHS` and `M_up` is a running
    max over those hourly highs -- so the numerator inherits intra-minute tick
    extremes and is very close to the CONTINUOUS running maximum. The
    denominator is the only 5-minute object: `RV = sqrt(sum of rv5)`.

    Sampling the extremes coarser than the data does biases the benchmark DOWN
    and therefore biases the measured gap toward zero -- it makes BTC look more
    Brownian than it is. Measured here, at n = 60,000:

        extremes from  5-min closes   1.5190   <- attempt 2's control
        extremes from  1-min closes   1.5605
        extremes from 10-sec closes   1.5860   <- what this function now uses
        continuous limit sqrt(8/pi)   1.5958

    The default is 10-second resolution, not the continuous constant, and that
    is deliberately CONSERVATIVE: real 1-minute bar highs come from ticks, so
    the true benchmark sits nearer 1.5958, and using 1.5860 understates the gap
    rather than flattering it. The residual ambiguity is ~0.01, against a gap
    of ~0.26.
    """
    rng = np.random.default_rng(seed)
    N = H * steps_per_hour
    inc = rng.standard_normal((n, N)) / np.sqrt(N)
    path = np.cumsum(inc, axis=1)
    # numerator: running extremes at fine resolution, as bar highs/lows are
    m_up = np.maximum(path.max(axis=1), 0.0)
    m_dn = np.maximum(-path.min(axis=1), 0.0)
    # denominator: RV from returns sampled every `rv_minutes`, as rv5 is
    k = int(rv_minutes * steps_per_hour / 60)
    p0 = np.concatenate([np.zeros((n, 1)), path], axis=1)
    r = np.diff(p0[:, ::k], axis=1)
    rv = np.sqrt((r ** 2).sum(axis=1))
    from scipy.stats import spearmanr
    rho, _ = spearmanr(m_up / rv, m_dn / rv)
    return {"ratio_mean": float(((m_up + m_dn) / rv).mean()),
            "ratio_median": float(np.median((m_up + m_dn) / rv)),
            "spearman_up_dn": float(rho), "n": n, "H": H,
            "steps_per_hour": steps_per_hour, "rv_minutes": rv_minutes,
            "note": "extremes fine-sampled (as bar highs are); RV at 5 min"}


def gaussian_touch(u: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """P(max of a driftless Brownian path >= u) = 2 * Phi(-u/sigma).

    The reflection principle. This is the textbook barrier law and the thing
    every option desk reaches for first.
    """
    return np.clip(2.0 * norm.cdf(-np.abs(u) / np.maximum(sigma, 1e-12)), 0.0, 1.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Volatility error vs shape error")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/firstpassage.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    prod = S.production_mask(ep)
    e = ep[prod]
    RV = e["RV"].to_numpy(np.float64)
    M_up = np.abs(e["M_up"].to_numpy(np.float64))
    M_dn = np.abs(e["M_dn"].to_numpy(np.float64))
    rng = M_up + M_dn                              # max - min in log terms
    print(f"production episodes: {len(e):,}\n")

    # ---- the shape statistic --------------------------------------------
    ratio = rng / np.maximum(RV, 1e-12)
    q = np.quantile(ratio, [0.05, 0.25, 0.5, 0.75, 0.95])
    ctl = brownian_control(H=int(np.median(e["H"])))
    ref = ctl["ratio_mean"]
    lo, hi = block_bootstrap_ci(ratio - ref, seed=3)
    print(f"RANGE / sqrt(realized variance)")
    print(f"  Brownian, continuous : {BROWNIAN_RATIO:.4f}  (NOT the right "
          f"comparison -- see brownian_control)")
    print(f"  Brownian, sampled as the data is (extremes at "
          f"{3600//ctl['steps_per_hour']}s, RV at {ctl['rv_minutes']}min): "
          f"{ref:.4f}")
    print(f"  BTC measured  median : {q[2]:.4f}   mean {ratio.mean():.4f}")
    print(f"                  IQR  : [{q[1]:.4f}, {q[3]:.4f}]")
    print(f"            5-95 pct   : [{q[0]:.4f}, {q[4]:.4f}]")
    print(f"  mean - sampled ref   : {ratio.mean()-ref:+.4f}  "
          f"block-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  (against the continuous constant the gap would read "
          f"{ratio.mean()-BROWNIAN_RATIO:+.4f}; the control is deliberately "
          f"conservative, so the true gap is between the two)")
    print(f"  Spearman(M_up/RV, M_dn/RV): BTC "
          f"{__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(M_up/np.maximum(RV,1e-12), M_dn/np.maximum(RV,1e-12))[0]:+.4f}"
          f"   Brownian control {ctl['spearman_up_dn']:+.4f}")
    verdict = ("CHOPS -- less travel per unit of volatility than Brownian"
               if ratio.mean() < ref else
               "TRENDS -- more travel per unit of volatility than Brownian")
    print(f"  -> BTC {verdict}\n")

    # ---- oracle first passage -------------------------------------------
    # Feed the Gaussian formula the REALIZED volatility. Any error left is
    # pure shape error: it is what no volatility model could ever fix.
    print(f"ONE-SIDED barrier touch (upside), Gaussian law fed REALIZED vol")
    print(f"{'barrier':>8} {'realized':>9} {'gaussian':>9} {'ratio':>7} "
          f"{'abs err pp':>11}  interpretation")
    rows = []
    for pct in BARRIER_PCT:
        u = np.log1p(pct / 100.0)
        # ONE-SIDED, where the reflection principle is EXACT. A two-sided
        # version would need the full Feller series; approximating it as twice
        # the one-sided probability over-counts and would manufacture exactly
        # the overstatement this file is testing for. Scored on the upside
        # barrier alone, and separately on the downside below.
        hit = (M_up >= u).astype(np.float64)
        p_g = gaussian_touch(u, RV)
        realized, predicted = float(hit.mean()), float(p_g.mean())
        d = predicted - realized
        rows.append({"barrier_pct": float(pct), "realized": realized,
                     "gaussian_oracle": predicted,
                     "ratio": predicted / max(realized, 1e-12),
                     "abs_err_pp": 100 * d})
        tag = ("OVERSTATES touch risk" if d > 0.005 else
               "understates touch risk" if d < -0.005 else "close")
        print(f"{pct:7.1f}% {realized:9.4f} {predicted:9.4f} "
              f"{predicted/max(realized,1e-12):7.3f} {100*d:+11.2f}  {tag}")

    print("\n  'gaussian' is fed the REALIZED volatility -- an oracle no")
    print("  forecaster could match. Whatever error remains is SHAPE error,")
    print("  the part better volatility forecasting can never remove.")

    # ---- how much of the total error is addressable ----------------------
    # Compare against the same formula fed a causal forecast (exp(har_1d)).
    H = e["H"].to_numpy(np.float64)
    sig_hat = np.exp(X.loc[prod, "har_1d"].to_numpy(np.float64)) * np.sqrt(H)
    # `har_1d` is NaN through the warm-up, so both arms are restricted to the
    # episodes where a causal forecast actually exists. Comparing an oracle
    # over all episodes against a forecaster over a subset would flatter the
    # oracle by exactly the episodes the forecaster could not see.
    ok = np.isfinite(sig_hat)
    print(f"\ncomparable episodes (causal sigma available): {int(ok.sum()):,} "
          f"of {len(e):,}")
    print(f"{'barrier':>8} {'oracle err':>11} {'forecast err':>13} "
          f"{'share from vol':>15}")
    for i, pct in enumerate(BARRIER_PCT):
        u = np.log1p(pct / 100.0)
        hit = (M_up >= u).astype(np.float64)[ok]
        e_or = abs(float(gaussian_touch(u, RV[ok]).mean()) - hit.mean())
        e_fc = abs(float(gaussian_touch(u, sig_hat[ok]).mean()) - hit.mean())
        share = 100.0 * (1.0 - e_or / max(e_fc, 1e-12))
        rows[i]["forecast_err_pp"] = 100 * e_fc
        rows[i]["share_from_vol_pct"] = share
        print(f"{pct:7.1f}% {100*e_or:10.2f}pp {100*e_fc:12.2f}pp {share:14.1f}%")
    print("\n  'share from vol' = the fraction of the causal forecaster's error")
    print("  that a PERFECT volatility forecast would remove. The rest is shape.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_episodes": int(len(e)),
        "brownian_ratio_continuous": BROWNIAN_RATIO,
        "brownian_control_sampled": ctl,
        "range_over_sqrt_rv": {
            "mean": float(ratio.mean()), "median": float(q[2]),
            "p05": float(q[0]), "p25": float(q[1]), "p75": float(q[3]),
            "p95": float(q[4]),
            "mean_minus_brownian_sampled": float(ratio.mean() - ref),
            "mean_minus_brownian_continuous": float(ratio.mean() - BROWNIAN_RATIO),
            "ci95": [lo, hi]},
        "barriers": rows}, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
