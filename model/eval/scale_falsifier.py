"""
eval/scale_falsifier.py
=====================================================================
Two mechanism falsifiers from the teacher scorecard, both scalar, both cheap.

WHAT IS BEING TESTED, AND WHY A SCALAR IS THE RIGHT INSTRUMENT

The scorecard produced two claims that look like mechanisms and might only be
LEVEL BIAS. QLIKE is asymmetric -- at a factor-2 error it penalises
under-forecasting about 1.60x more than over-forecasting -- so a model that is
simply biased high buys spike performance and pays in calm, with no
episode-specific knowledge at all. A single multiplicative constant, fitted on
CALIB, separates "this model knows something about tails" from "this model
runs hot".

    M-garchnormal-tails
        garch_normal wins spike AND deep-tail at all four horizons while never
        winning pooled, with calibration ratio mean(RV^2/sigma^2) BELOW 1
        everywhere (0.845 / 0.861 / 0.778 / 0.416) -- i.e. it over-forecasts.
        If rescaling another teacher upward reproduces its tail advantage, the
        mechanism is scale and there is nothing to transfer but a number.

    M-noctua-calm-spike-split
        NOCTUA V1 is the BEST calm forecaster in the zoo at H = 1 and H = 6 and
        among the worst on tails, with ratio 1.166-1.464 -- it under-forecasts.
        This is the signature `E-scale` has had open since Phase 0: the model
        reports a conditional MEDIAN while QLIKE is minimised by the conditional
        MEAN of variance. If one scalar closes most of the pooled gap AND lands
        the ratio near 1, the deficit is scale, not information.

THE SCALAR IS FITTED ON CALIB AND APPLIED TO TEST. Fitting it on test would be
the whole experiment answering itself: any model can be made to look calibrated
on data used to calibrate it.

CLOSED FORM, NOT A SEARCH. For sigma' = c*sigma the QLIKE mean is

    (1/c^2)*mean(RV^2/sigma^2) - 2*log(c) - mean(log(RV^2/sigma^2)) - 1

whose derivative in c vanishes at c^2 = mean(RV^2/sigma^2). So the
QLIKE-optimal rescale IS the square root of the calibration ratio. That is
worth stating because it means the "calibration ratio" column in the scorecard
was already reporting the optimal correction, squared -- and it makes the test
exact rather than approximate.

    python -m model.eval.scale_falsifier
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.teacher_scorecard import gather, load_oof                          # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402

HORIZONS = (1, 6, 24, 168)
N_FAMILY = 8            # 2 mechanisms x 4 horizons, fixed before results


def optimal_c(rv: np.ndarray, sig: np.ndarray) -> float:
    """The QLIKE-minimising multiplicative rescale, in closed form."""
    r = np.nanmean(rv ** 2 / np.maximum(sig, 1e-12) ** 2)
    return float(np.sqrt(max(r, 1e-12)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="scalar mechanism falsifiers")
    ap.add_argument("--oof", type=Path, default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/scale_falsifier.json"))
    a = ap.parse_args(argv)

    z, teachers = load_oof(a.oof)
    alpha = 0.05 / N_FAMILY
    print(f"scalar rescale fitted on CALIB, applied to TEST. "
          f"Bonferroni family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("the optimal c is sqrt(mean(RV^2/sigma^2)) in closed form, not a search\n")

    out = {"family_size": N_FAMILY, "alpha": alpha, "horizons": {}}
    for H in HORIZONS:
        rows = {}
        for t in teachers:
            dc, dt = gather(z, H, "calib", t), gather(z, H, "test", t)
            if dc is None or dt is None:
                continue
            c = optimal_c(dc["rv"], dc["sigma"])           # CALIB only
            q0 = dt["q"]
            q1 = qlike_vec(dt["rv"], c * dt["sigma"])
            good = np.isfinite(q0) & np.isfinite(q1)
            L = block_len_for(H, int(good.sum()))
            ci = mean_ci((q0 - q1)[good], alpha=alpha, block_len=L)
            hi = dt["rv"] >= np.quantile(dt["rv"], 0.95)
            rows[t] = {
                "c": c,
                "pooled_before": float(np.nanmean(q0)),
                "pooled_after": float(np.nanmean(q1)),
                "gain": float(np.nanmean(q0 - q1)),
                "gain_ci": list(ci["ci95"]),
                "spike_before": float(np.nanmean(q0[hi])),
                "spike_after": float(np.nanmean(q1[hi])),
                "calm_before": float(np.nanmean(q0[~hi])),
                "calm_after": float(np.nanmean(q1[~hi])),
                "ratio_after": float(np.nanmean(
                    dt["rv"] ** 2 / np.maximum(c * dt["sigma"], 1e-12) ** 2)),
            }
        if not rows:
            continue
        best_raw = min(rows, key=lambda k: rows[k]["pooled_before"])
        best_scaled = min(rows, key=lambda k: rows[k]["pooled_after"])
        print("=" * 100)
        print(f"H = {H}h    best raw: {best_raw}    best after rescale: {best_scaled}")
        print("=" * 100)
        print(f"{'teacher':>14} {'c':>7} {'pooled→':>18} {'gain':>9} "
              f"{'spike→':>16} {'calm→':>16} {'ratio':>7}")
        for t in sorted(rows, key=lambda k: rows[k]["pooled_after"]):
            r = rows[t]
            print(f"{t:>14} {r['c']:7.3f} "
                  f"{r['pooled_before']:8.5f}→{r['pooled_after']:<9.5f} "
                  f"{r['gain']:+9.5f} "
                  f"{r['spike_before']:7.3f}→{r['spike_after']:<8.3f} "
                  f"{r['calm_before']:7.4f}→{r['calm_after']:<8.4f} "
                  f"{r['ratio_after']:7.3f}")
        out["horizons"][str(H)] = {"best_raw": best_raw, "best_scaled": best_scaled,
                                   "arms": rows}
        print()

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
