"""
eval/hour_anchor_exante.py
=====================================================================
P4-hour-anchor-exante: the tail check conditioned on what is known AT THE
ANCHOR, not on the outcome (the forecaster's dilemma, Lerch et al. 2017, is why
P4-hour-anchor-cond's outcome-selected spike subset cannot be read as proper).

Reads the per-episode losses eval/hour_anchor_cond.py saved; no retraining.

    python -m model.eval.hour_anchor_exante
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                     # noqa: E402
from noctua.train import load_all                                      # noqa: E402

METRICS = ("brier", "logs", "pinball", "crps", "brier_far")
Q = 0.90


def main() -> int:
    z = np.load("model/artifacts/hour_anchor_cond.npz")
    _, X = load_all(Path("model/artifacts"))
    idx, year = z["idx"], z["year"]
    shock = (X["har_6h"].to_numpy(np.float64) - X["har_22d"].to_numpy(np.float64))[idx]
    level = X["har_1d"].to_numpy(np.float64)[idx]
    hot = np.zeros(len(idx), bool); high = np.zeros(len(idx), bool)
    for y in np.unique(year):
        m = year == y
        hot[m] = shock[m] >= np.nanquantile(shock[m], Q)
        high[m] = level[m] >= np.nanquantile(level[m], Q)
    subsets = {"HOT": hot, "not HOT": ~hot, "HIGH": high, "not HIGH": ~high}
    alpha = 0.05 / (len(METRICS) * len(subsets))
    out = {"alpha": alpha, "rows": {}}
    print(f"ex-ante subsets (top {100*(1-Q):.0f}% per fold), As minus M0s (+ = As better), "
          f"family {len(METRICS)*len(subsets)} -> {100*(1-alpha):.2f}%")
    print(f"{'subset':>9} {'n':>5} " + " ".join(f"{m:>28}" for m in METRICS))
    damage = []
    for name, msk in subsets.items():
        cells = []
        for m in METRICS:
            d = z[f"M0s__{m}"][msk] - z[f"As__{m}"][msk]
            bl = 38 if msk.sum() > 600 else 1
            lo, hi = mean_ci(d, alpha=alpha, block_len=bl)["ci95"]
            pct = 100 * d.mean() / z[f"M0s__{m}"][msk].mean()
            out["rows"].setdefault(name, {})[m] = {"n": int(msk.sum()), "pct": float(pct),
                                                   "ci": [float(lo), float(hi)]}
            tag = "+" if lo > 0 else "-" if hi < 0 else " "
            cells.append(f"{pct:+6.2f}% [{lo:+.5f},{hi:+.5f}]{tag}")
            if name in ("HOT", "HIGH") and hi < 0:
                damage.append(f"{name}:{m}")
        print(f"{name:>9} {int(msk.sum()):>5} " + " ".join(f"{c:>28}" for c in cells))
    # overlap with the outcome-selected spikes, for the record
    sp = z["spike"]
    print(f"\nspike nights that were HOT ex ante: {int((sp & hot).sum())}/{int(sp.sum())}; "
          f"HIGH: {int((sp & high).sum())}/{int(sp.sum())}")
    verdict = "TAIL DAMAGE CONFIRMED on " + str(damage) if damage else "NOT CONFIRMED"
    print(f"--- registered reading: {verdict}")
    out.update(damage=damage, verdict=verdict,
               spike_hot=int((sp & hot).sum()), spike_high=int((sp & high).sum()),
               n_spike=int(sp.sum()))
    Path("model/artifacts/hour_anchor_exante.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
