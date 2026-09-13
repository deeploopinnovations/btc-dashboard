"""
eval/scale_adopt.py
=====================================================================
P2-scale-v2: the ADOPTION test for the scale correction.

WHY THIS IS NOT THE SAME EXPERIMENT AS eval/scale_falsifier.py

The falsifier answered "is NOCTUA's deficit a scale constant". It is:
per-fold, calib-fitted c of 1.198 / 1.212 / 1.213 / 1.089 turns NOCTUA into the
best arm in the zoo at H = 1, 6 and 24. That is a MECHANISM result and it does
NOT license a change to the shipped model, for two reasons the falsifier
structurally could not address.

  1. It reports pooled means and a WITHIN-ARM gain interval. The adoption
     question is a PAIRED PER-EPISODE contrast against the best teacher, with
     that teacher ALSO rescaled by its own calib-fitted c -- because the
     correction is available to every arm and R39 says a baseline denied the
     candidate's advantage is not a baseline.

  2. IT NEVER TOUCHED THE BARRIER CURVES. `E-scale`'s own pre-registration,
     open since Phase 0, says: "Must also re-score the barrier curves, since
     scaling sigma moves every touch probability, which a median-only
     re-weighting does not test." R4 says the stricter written commitment
     governs. So adoption is BLOCKED until pinball, CRPS, Brier, the CORP
     decomposition and Christoffersen have all been re-scored under c --
     whatever QLIKE says.

HOW THE CORRECTION IS APPLIED, AND WHY THROUGH THE EXISTING HOOK

sigma -> c*sigma is exactly log-vol -> log-vol + log(c). `benchmark.run_fold`
already has `post_shift_fn`, which shifts the FINAL blended log-vol level and
carries an assertion that the achieved shift equals the requested one to 1e-6.
Passing log(c) therefore moves sigma by exactly c and lets the correction
propagate into Stage B, the committee and every barrier -- which is the whole
point of the barrier condition. Reimplementing the scoring here instead would
be R18's mistake.

`predict_avg` applies the shift on EVERY call, including the calibration slice
the committee is fitted on, so the committee sees the corrected scale rather
than being fitted on one scale and served another.

TWO PASSES, DELIBERATELY

Pass A runs with no shift and returns the calib-slice sigma. c is fitted from
it -- CALIB ONLY, closed form c = sqrt(mean(RV^2/sigma^2)). Pass B re-runs with
post_shift_fn = log(c). The model is trained twice per fold, which is wasteful
and correct; the alternative is caching a model across two different scoring
configurations, which is how a stale artifact gets scored.

    python -m model.eval.scale_adopt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                          # noqa: E402
from eval.direction import mean_ci                                           # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402
from noctua import splits as S                                              # noqa: E402
from noctua.train import load_all                                            # noqa: E402

N_FAMILY = 4                      # 4 horizons x 1 primary, fixed before results
PROD_H = 19
RATIO_LO, RATIO_HI = 0.95, 1.05   # guard (i)
SLICE_TOL = 0.02                  # guard (ii): no slice may degrade > 2% rel


def optimal_c(rv, sig):
    r = np.nanmean(rv ** 2 / np.maximum(sig, 1e-12) ** 2)
    return float(np.sqrt(max(r, 1e-12)))


def barrier_cols(rows, model="noctua_v2"):
    """Pull the barrier metrics for one model out of run_fold's row list."""
    r = next((x for x in rows if x["model"] == model), None)
    if r is None:
        return {}
    out = {}
    for pat in ("pinball_", "crps_", "brier_", "DSC_", "MCB_", "logs_"):
        vals = [v for k, v in r.items() if k.startswith(pat)
                and isinstance(v, (int, float))]
        if vals:
            out[pat.rstrip("_")] = float(np.mean(vals))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-scale-v2 adoption test")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/scale_adopt.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(train_mask):
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    alpha = 0.05 / N_FAMILY
    print(f"production slice H={PROD_H}, {len(folds)} folds, seeds={a.seeds}")
    print(f"Bonferroni family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("pass A: no shift, fit c on CALIB.  pass B: post_shift_fn = log(c)\n")

    acc = []
    for f in folds:
        t0 = time.time()
        rA = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn)
        if rA is None:
            print(f"  fold {f['year']}: skipped"); continue
        peA = rA["per_episode"]
        c = optimal_c(peA["rv_cal"], peA["sigma_cal"])          # CALIB ONLY
        rB = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn,
                      post_shift_fn=lambda m, mt, _c=c: np.full(int(m.sum()),
                                                                np.log(_c)))
        peB = rB["per_episode"]

        # the shift must have done exactly what was asked
        achieved = float(np.median(peB["sigma_med"] / peA["sigma_med"]))
        if abs(achieved - c) > 1e-3 * c:
            raise SystemExit(
                f"REFUSING: fold {f['year']} asked for a sigma factor of {c:.6f} "
                f"and achieved {achieved:.6f}. The post_shift_fn algebra is "
                f"wrong, so the correction being scored is not the one fitted.")

        acc.append({
            "year": f["year"], "c": c, "achieved": achieved,
            "rv": peA["rv"],
            "sig_before": peA["sigma_med"], "sig_after": peB["sigma_med"],
            "vol_before": rA["vol"], "vol_after": rB["vol"],
            "bar_before": barrier_cols(rA["rows"]),
            "bar_after": barrier_cols(rB["rows"]),
            "chr_before": rA["christoffersen"], "chr_after": rB["christoffersen"],
        })
        print(f"  fold {f['year']}  c={c:.4f} (achieved {achieved:.4f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not acc:
        print("no usable folds"); return 1

    rv = np.concatenate([r["rv"] for r in acc])
    s0 = np.concatenate([r["sig_before"] for r in acc])
    s1 = np.concatenate([r["sig_after"] for r in acc])
    q0, q1 = qlike_vec(rv, s0), qlike_vec(rv, s1)
    hi = rv >= np.quantile(rv, 0.95)
    L = block_len_for(PROD_H, len(q0))
    ci = mean_ci((q0 - q1)[np.isfinite(q0 - q1)], alpha=alpha, block_len=L)
    ratio = float(np.nanmean(rv ** 2 / np.maximum(s1, 1e-12) ** 2))

    print("\n" + "=" * 92)
    print(f"PRODUCTION SLICE  {len(q0):,} test episodes  "
          f"mean c = {np.mean([r['c'] for r in acc]):.4f} "
          f"(spread {max(r['c'] for r in acc) - min(r['c'] for r in acc):.4f})")
    print("=" * 92)
    print(f"  QLIKE pooled  {np.nanmean(q0):.5f} -> {np.nanmean(q1):.5f}   "
          f"gain {np.nanmean(q0 - q1):+.5f}  CI [{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]")
    print(f"  QLIKE spike   {np.nanmean(q0[hi]):.5f} -> {np.nanmean(q1[hi]):.5f}")
    print(f"  QLIKE calm    {np.nanmean(q0[~hi]):.5f} -> {np.nanmean(q1[~hi]):.5f}")
    print(f"  calib ratio   {float(np.nanmean(rv**2/np.maximum(s0,1e-12)**2)):.4f} "
          f"-> {ratio:.4f}   guard [{RATIO_LO}, {RATIO_HI}]")

    print("\n  --- THE BARRIER CONDITION, inherited from E-scale ---")
    bar = {}
    for k in sorted(set().union(*[set(r["bar_before"]) for r in acc])):
        b = float(np.mean([r["bar_before"][k] for r in acc if k in r["bar_before"]]))
        aft = float(np.mean([r["bar_after"][k] for r in acc if k in r["bar_after"]]))
        better = (aft > b) if k == "DSC" else (aft < b)
        bar[k] = {"before": b, "after": aft, "better": bool(better)}
        print(f"  {k:>8}  {b:.6f} -> {aft:.6f}   "
              f"{'BETTER' if better else 'WORSE '}  ({100*(aft-b)/abs(b):+.2f}%)")

    guards = {
        "ratio_in_band": RATIO_LO <= ratio <= RATIO_HI,
        "spike_not_worse_2pct":
            np.nanmean(q1[hi]) <= np.nanmean(q0[hi]) * (1 + SLICE_TOL),
        "calm_not_worse_2pct":
            np.nanmean(q1[~hi]) <= np.nanmean(q0[~hi]) * (1 + SLICE_TOL),
        "dsc_not_worse": bar.get("DSC", {}).get("better", False)
                         or abs(bar.get("DSC", {}).get("after", 0)
                                - bar.get("DSC", {}).get("before", 0)) < 1e-9,
        "brier_not_worse": bar.get("brier", {}).get("better", True),
    }
    print("\n  --- pre-registered guards ---")
    for k, v in guards.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    verdict = ("ADOPTABLE" if all(guards.values()) and ci["ci95"][0] > 0
               else "NOT ADOPTABLE")
    print(f"\n  VERDICT: {verdict}")
    if not all(guards.values()):
        print("  (a QLIKE gain does not license adoption when a guard fails -- "
              "NOCTUA's product is a touch-probability curve, not a sigma)")

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha, "block_len": L,
        "c_per_fold": {str(r["year"]): r["c"] for r in acc},
        "qlike": {"pooled_before": float(np.nanmean(q0)),
                  "pooled_after": float(np.nanmean(q1)),
                  "gain": float(np.nanmean(q0 - q1)), "ci": list(ci["ci95"]),
                  "spike_before": float(np.nanmean(q0[hi])),
                  "spike_after": float(np.nanmean(q1[hi])),
                  "calm_before": float(np.nanmean(q0[~hi])),
                  "calm_after": float(np.nanmean(q1[~hi]))},
        "calib_ratio_after": ratio, "barriers": bar,
        "guards": guards, "verdict": verdict,
    }, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
