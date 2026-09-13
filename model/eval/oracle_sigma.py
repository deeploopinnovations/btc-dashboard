"""
eval/oracle_sigma.py
=====================================================================
How much of NOCTUA's barrier error would a PERFECT volatility forecast remove?

WHY THIS REPLACES THE NUMBER §5b WAS QUOTING

`eval/firstpassage.py` answered this by feeding the **Gaussian first-passage
law** the realized volatility and comparing against the same law fed a causal
forecast, concluding that a perfect volatility forecast removes only 6.0-8.5%
of the barrier error -- i.e. that ~92% is "shape".

§8 showed that instrument is broken: given PERFECT volatility and a fitted
shape constant, the Gaussian law is 1.62x/2.46x/6.55x WORSE than quoting the
historical base rate at the 2/3/5% barriers. Both arms of that decomposition
are the broken instrument, so the 92% describes the Gaussian forecaster and
says nothing about NOCTUA -- which does not use it. NOCTUA reads barrier
probabilities off its learned Stage B quantile heads via
`infer.survival_from_quantiles`.

So the decomposition is redone here through the mapping that actually ships.

THE DESIGN, AND THE CONFOUND IT HAS TO AVOID

Stage B is conditioned on `log_sigma` and integrated over 32 sigma ATOMS drawn
from Stage A's predictive distribution. The naive oracle -- swap the 32 atoms
for a point mass at the realized RV -- changes two things at once: the LOCATION
of sigma (accuracy, what we want to measure) and the SPREAD over atoms (the
model's uncertainty about sigma, which perfect knowledge also removes). Those
are different quantities and reporting their sum as "what better volatility
forecasting would buy" overstates it, because no forecaster ever gets the
second for free.

Three arms, so the two can be separated:

    committee_32atom   the shipped object: 32 atoms from Stage A     (reference)
    point_forecast     ONE atom at Stage A's median sigma            (accuracy floor)
    point_oracle       ONE atom at the REALIZED RV                   (the oracle)

`point_forecast` vs `point_oracle` is the apples-to-apples comparison: both are
single atoms, so the only difference is whether that atom is right. The
addressable share of barrier error is
    1 - err(point_oracle) / err(point_forecast)
and `committee_32atom` is carried alongside to show what the atom integration
is worth, rather than letting it hide inside the oracle's margin.

SCORING

Brier score on the realized touch indicator -- strictly proper -- with the
historical BASE RATE scored as a floor, because §8 established that a barrier
forecaster can be worse than a constant and nobody had been checking. Also
reported: mean absolute calibration error in percentage points, so the numbers
are directly comparable to the ones `firstpassage.py` prints.

WHAT A NEGATIVE RESULT LOOKS LIKE

If `point_oracle` barely beats `point_forecast`, then NOCTUA's barrier error
really is dominated by something other than the volatility level, and the
conclusion §5b reached for the wrong reason is right for a different one. If
the oracle removes a large share, then better volatility forecasting IS the
lever for barriers, and the withdrawn "stop investing in volatility" advice was
not merely unsupported but backwards.

    python -m model.eval.oracle_sigma
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import block_bootstrap_ci                              # noqa: E402
from noctua import infer as I                                              # noqa: E402
from noctua import splits as S                                             # noqa: E402
from noctua.train import load_all                                          # noqa: E402
from serve import runtime as R                                             # noqa: E402

BARRIERS = (0.5, 1.0, 2.0, 3.0, 5.0)


def stage_b_at_sigma(model, d: dict, sigma: np.ndarray) -> dict:
    """A predictive object whose sigma is EXACTLY `sigma`, one atom, no spread.

    Averaged over the seed ensemble the same way `predict` does, so the only
    difference from the shipped path is the sigma being conditioned on.
    """
    sig = np.maximum(np.asarray(sigma, np.float64), 1e-12)[:, None]
    ls = np.log(sig)
    full = model.w
    ups, dns = [], []
    try:
        for s in range(model.n_seeds):
            model.w = {**model._seed_scope(full, s), "har_beta": full["har_beta"]}
            _, u_, d_, _ = R.NumpyNoctua.stage_b(model, d["Xs"], ls)
            ups.append(u_); dns.append(d_)
    finally:
        model.w = full
    return {"sigma_atoms": sig,
            "q_up": np.mean(ups, axis=0)[:, None, :],
            "q_dn": np.mean(dns, axis=0)[:, None, :],
            "H": d["H"]}


def score(pred: dict, u: float, hit_up: np.ndarray, hit_dn: np.ndarray) -> dict:
    p_up = I.touch_prob(pred, u, up=True)
    p_dn = I.touch_prob(pred, u, up=False)
    b = 0.5 * ((p_up - hit_up) ** 2 + (p_dn - hit_dn) ** 2)
    cal = 0.5 * (abs(p_up.mean() - hit_up.mean()) + abs(p_dn.mean() - hit_dn.mean()))
    return {"brier": b, "cal_pp": 100.0 * cal,
            "mean_p": 0.5 * (p_up.mean() + p_dn.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Oracle sigma through NOCTUA's own map")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--model", type=Path, default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/oracle_sigma.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    sp = S.time_splits(ep)
    fin = np.isfinite(X.to_numpy()).all(1)
    te = sp["test"] & fin & S.production_mask(ep)
    m = R.load_model(str(a.model))

    Xte = X[te]
    H = ep["H"].to_numpy(np.float64)[te]
    d = m.prepare(Xte, H)
    RV = ep["RV"].to_numpy(np.float64)[te]
    M_up = np.abs(ep["M_up"].to_numpy(np.float64))[te]
    M_dn = np.abs(ep["M_dn"].to_numpy(np.float64))[te]

    full = m.predict(d)                              # the shipped 32-atom object
    sig_med = full["sigma_med"]
    arms = {
        "committee_32atom": full,
        "point_forecast": stage_b_at_sigma(m, d, sig_med),
        "point_oracle": stage_b_at_sigma(m, d, RV),
    }
    print(f"test production episodes: {int(te.sum()):,}   model {a.model.name}")
    print(f"median forecast sigma {np.median(sig_med):.5f}   "
          f"median realized RV {np.median(RV):.5f}\n")

    print(f"{'barrier':>8} {'realized':>9} {'32atom':>9} {'pt-fcast':>9} "
          f"{'pt-ORACLE':>10} {'base':>8} | {'addressable':>12} {'95% CI':>22}")
    rows = []
    for pct in BARRIERS:
        u = np.log1p(pct / 100.0)
        hu = (M_up >= u).astype(np.float64)
        hd = (M_dn >= u).astype(np.float64)
        s = {k: score(v, u, hu, hd) for k, v in arms.items()}
        base = 0.5 * (hu.mean() * (1 - hu.mean()) + hd.mean() * (1 - hd.mean()))
        bf, bo = s["point_forecast"]["brier"], s["point_oracle"]["brier"]
        share = 100.0 * (1.0 - bo.mean() / max(bf.mean(), 1e-12))
        lo, hi = block_bootstrap_ci(bo - bf, seed=9)
        rows.append({
            "barrier_pct": pct, "realized_up": float(hu.mean()),
            "brier_committee": float(s["committee_32atom"]["brier"].mean()),
            "brier_point_forecast": float(bf.mean()),
            "brier_point_oracle": float(bo.mean()),
            "brier_base_rate": float(base),
            "addressable_share_pct": float(share),
            "oracle_minus_forecast_ci95": [lo, hi],
            "cal_pp_committee": float(s["committee_32atom"]["cal_pp"]),
            "cal_pp_point_forecast": float(s["point_forecast"]["cal_pp"]),
            "cal_pp_point_oracle": float(s["point_oracle"]["cal_pp"])})
        print(f"{pct:7.1f}% {hu.mean():9.4f} "
              f"{s['committee_32atom']['brier'].mean():9.5f} {bf.mean():9.5f} "
              f"{bo.mean():10.5f} {base:8.5f} | {share:11.1f}% "
              f"[{lo:+.5f},{hi:+.5f}]")

    print("\n  'addressable' = the share of the SINGLE-ATOM forecaster's Brier score")
    print("  that perfect knowledge of volatility removes, through NOCTUA's own")
    print("  Stage B mapping. Both arms are one atom, so only sigma's accuracy")
    print("  differs -- the atom spread is held out and shown separately as the")
    print("  gap between 'pt-fcast' and '32atom'.")
    print("\n  calibration error, percentage points (comparable to firstpassage.py):")
    print(f"  {'barrier':>8} {'32atom':>9} {'pt-fcast':>9} {'pt-ORACLE':>10}")
    for r in rows:
        print(f"  {r['barrier_pct']:7.1f}% {r['cal_pp_committee']:9.2f} "
              f"{r['cal_pp_point_forecast']:9.2f} {r['cal_pp_point_oracle']:10.2f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"n_test": int(te.sum()),
                                 "model": str(a.model), "barriers": rows},
                                indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
