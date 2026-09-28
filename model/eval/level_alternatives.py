"""
eval/level_alternatives.py
=====================================================================
P4-level-alternatives: is the clock-aware anchor doing work a simpler LEVEL
fix would not?

On the product slice (17:00, H=19) the clock term is a constant shift. Two
simpler ways to lower 17:00:

  G_C    a constant from the fold's CALIB outcomes: the 17:00 differential
         the serving factor cannot see (median log RV/sigma at 17:00 minus at
         the factor's hours), applied at 17:00 only, on top of the factor;
  G_F17  serving's own trailing factor, read at the SAME hour it corrects:
         settled 17:00/H=19 episodes over the last 60 days (one per day),
         median RV/sigma, clip [0.70, 1.40], >= 20 -- a one-line change to
         serve/adaptive.py's anchor stride, versus the 22/04/10/16 it reads.

All arms are the NO-NETWORK Gaussian first-passage law through
eval/product_score (validated against run_fold to 1e-10 in
P4-simple-vs-noctua-result), so this costs no training. The level question is
shared with NOCTUA; the barrier SHAPE is not, which is why this is a diagnostic.

    python -m model.eval.level_alternatives
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                     # noqa: E402
from eval.hour_anchor import (FAC_LO, FAC_HI, FAC_MIN, PROD_A,         # noqa: E402
                              PROD_H, fold_anchors, served_log_factor)
from eval.product_score import battery, gaussian_curves, per_episode   # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

HOUR = 3600
EP_METRICS = ("brier", "logs", "pinball", "crps")
ARMS = ("G_M0s", "G_As", "G_C", "G_F17")
CONTRASTS = (("G_C", "G_As"), ("G_F17", "G_As"))


def hour_matched_log_factor(t_query, ts_hist, rv_hist, sig_hist,
                            days: int = 60, H: int = PROD_H) -> np.ndarray:
    """Trailing median RV/sigma over settled episodes at the query's own
    clock hour: anchors t - 24h*k, k = 1..days, each settled by t (k >= 1 and
    H < 24 guarantee it)."""
    lut = dict(zip(np.asarray(ts_hist, np.int64).tolist(),
                   zip(np.asarray(rv_hist, float), np.asarray(sig_hist, float))))
    out = np.zeros(len(t_query))
    for i, t in enumerate(np.asarray(t_query, np.int64)):
        pairs = [lut[a] for a in (t - 24 * HOUR * np.arange(1, days + 1)).tolist()
                 if a in lut]
        r = np.array([p[0] / p[1] for p in pairs
                      if np.isfinite(p[0]) and np.isfinite(p[1]) and p[0] > 0 and p[1] > 0])
        if len(r) >= FAC_MIN:
            out[i] = np.log(np.clip(float(np.median(r)), FAC_LO, FAC_HI))
    assert H < 24
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="simpler level fixes vs the clock")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/level_alternatives.json"))
    a = ap.parse_args(argv)
    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    rv = ep["RV"].to_numpy(np.float64)
    fin = np.isfinite(X.to_numpy()).all(1)
    at19 = (ep.H == PROD_H).to_numpy()
    sq = np.sqrt(PROD_H)
    M_up, M_dn = np.abs(ep.M_up.to_numpy()), np.abs(ep.M_dn.to_numpy())

    pe = {k: {m: [] for m in EP_METRICS} for k in ARMS}
    dsc = {k: [] for k in ARMS}
    rows = []
    for f in folds:
        anc = fold_anchors(ep, X, f)
        te = np.flatnonzero(np.asarray(f["test"], bool) & fin & at19 & (ah == PROD_A)
                            & np.isfinite(rv) & (rv > 0))
        # causal history: every finite H=19 episode from calib start to test end
        hist = np.flatnonzero(at19 & fin & np.isfinite(rv) & (rv > 0)
                              & (ts >= ts[np.asarray(f["calib"], bool)].min())
                              & (ts <= ts[te].max()))
        cal17 = np.flatnonzero(np.asarray(f["calib"], bool) & fin & at19
                               & (ah == PROD_A) & np.isfinite(rv) & (rv > 0))
        lhc, A = anc["lhc"], anc["A"]
        # G_C, AMENDED (see the ledger): a constant applied at ALL hours is
        # absorbed exactly by serving's median factor, so the registered
        # version was degenerate (G_C == G_M0s bit for bit). The meaningful
        # constant is the 17:00 DIFFERENTIAL the factor cannot see: calib
        # median log(RV/sigma) at 17:00 minus the same at the factor hours
        # (04/10/16/22), applied at 17:00 only, on top of the served factor.
        r_all = np.log(rv) - (lhc + 0.5 * np.log(PROD_H))
        calib = np.asarray(f["calib"], bool) & fin & at19 & np.isfinite(rv) & (rv > 0)
        fh = np.isin(ah, [4, 10, 16, 22])
        c = float(np.median(r_all[calib & (ah == PROD_A)]) - np.median(r_all[calib & fh]))

        def served(anchor):
            return served_log_factor(ts[te], ts[hist], rv[hist], np.exp(anchor[hist]) * sq)

        sig = {"G_M0s": np.exp(lhc[te] + served(lhc)) * sq,
               "G_As": np.exp(A[te] + served(A)) * sq}
        sig["G_C"] = np.exp(lhc[te] + c + served(lhc)) * sq
        f17 = hour_matched_log_factor(ts[te], ts[hist], rv[hist], np.exp(lhc[hist]) * sq)
        sig["G_F17"] = np.exp(lhc[te] + f17) * sq
        M = {"up": M_up[te], "dn": M_dn[te]}
        for arm in ARMS:
            cur = gaussian_curves(sig[arm])
            dsc[arm].append(battery(cur, M)["DSC"])
            p = per_episode(cur, M)
            for m in EP_METRICS:
                pe[arm][m].append(p[m])
        shift17 = float(np.nanmean(A[te] - lhc[te]))
        rows.append({"year": f["year"], "calib_const": float(c), "clock_shift_17": shift17,
                     "f17_mean": float(f17.mean())})
        print(f"  fold {f['year']}: calib 17:00 constant {c:+.3f}   clock shift {shift17:+.3f}"
              f"   hour-matched factor mean {f17.mean():+.3f}", flush=True)

    alpha = 0.05 / (len(CONTRASTS) * len(EP_METRICS))
    out = {"alpha": alpha, "folds": rows, "ci": {},
           "dsc": {k: [float(x) for x in v] for k, v in dsc.items()}}
    print(f"\nmean per-episode loss  (DSC fold mean)")
    for arm in ARMS:
        print(f"  {arm:>6}: " + "  ".join(f"{m} {np.concatenate(pe[arm][m]).mean():.6f}"
                                         for m in EP_METRICS)
              + f"   DSC {np.mean(dsc[arm]):.6f}")
    print(f"\ncontrasts (+ = first arm better), family {len(CONTRASTS)*len(EP_METRICS)}, block 38")
    for c_, o in CONTRASTS:
        for m in EP_METRICS:
            d = np.concatenate(pe[o][m]) - np.concatenate(pe[c_][m])
            lo, hi = mean_ci(d, alpha=alpha, block_len=38)["ci95"]
            out["ci"][f"{m}_{c_}_vs_{o}"] = [float(d.mean()), float(lo), float(hi)]
            tag = "BETTER" if lo > 0 else "WORSE" if hi < 0 else "tie"
            print(f"  {m:>8} {c_+'-'+o:>12} {d.mean():+.6f} [{lo:+.6f}, {hi:+.6f}] {tag}")
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
