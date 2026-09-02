"""
eval/level_report.py
=====================================================================
P2-level-report: sigma and the barrier curve are two products with two losses.

THE ARGUMENT, IN ONE PARAGRAPH

QLIKE on a variance point forecast is minimised by the conditional MEAN of
variance. The barrier curve is scored by strictly proper rules on the
predictive DISTRIBUTION, and `P2-scale-v2` measured what happens when that
distribution's level is moved: QLIKE improved 9.7 % and ALL SIX barrier metrics
degraded. So the correction belongs to the REPORTED FUNCTIONAL, not to the
distribution -- and `serve/predict.py` already reads the two from different
places, `sigma_window_pct` from `pred["sigma_med"]` and the curves from
`model.touch_prob(pred, ...)`.

ONE PREDICTIVE OBJECT PER FOLD, THREE READINGS OF IT

    R0  sigma_med          the shipped reading, and the reference
    R1  sigma_mean         infer.py's own atom-grid mean. PER EPISODE, ZERO
                           fitted parameters
    R2  c * sigma_med      c = sqrt(mean(RV^2/sigma^2)) fitted on that fold's
                           CALIB slice only (R44). One parameter, and the
                           ceiling R1 is trying to reach without fitting

No arm runs a second pipeline, so no arm can touch a barrier. That is the point
and it is also why the barrier claim is NOT verified here: "the barriers are
unchanged" is true by construction, and a claim that cannot fail is not a claim
(R2). The pass condition for adoption is a SERVING REGRESSION TEST asserting
every touch_prob, safe_level, p_up and p_vol_amplify is bit-identical and that
exactly the two sigma fields moved. That test would fail if the change leaked
into the predictive object; this file cannot.

WHAT THIS IS NOT

It is not a model improvement. Nothing the model knows changes, no barrier gets
sharper, and no teacher is beaten that the rescaled ranking did not already
beat. It is a correction to WHICH NUMBER IS PUBLISHED under a loss that names a
different one. If a write-up of this cannot resist calling it a 9 % gain, it
should not ship.

    python -m model.eval.level_report
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
from eval.scale_adopt import optimal_c                                       # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402

PROD_H = 19
N_FAMILY = 2                      # R1 and R2 on one slice, fixed a priori
RATIO_LO, RATIO_HI = 0.95, 1.05


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-level-report")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/level_report.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(train_mask):
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    alpha = 0.05 / N_FAMILY
    print(f"P2-level-report   production slice H={PROD_H}   seeds={a.seeds}")
    print(f"family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("ONE predictive object per fold. The arms are readings of it, not "
          "second pipelines.\n")

    acc = []
    for f in folds:
        t0 = time.time()
        r = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn)
        if r is None:
            print(f"  fold {f['year']}: skipped"); continue
        pe = r["per_episode"]
        c = optimal_c(pe["rv_cal"], pe["sigma_cal"])          # CALIB only
        c_mean = optimal_c(pe["rv_cal"], pe["sigma_mean_cal"])
        acc.append({"year": f["year"], "rv": pe["rv"],
                    "R0": pe["sigma_med"], "R1": pe["sigma_mean"],
                    "R2": c * pe["sigma_med"], "c": c, "c_mean": c_mean,
                    "mm": float(np.mean(pe["sigma_mean"] / pe["sigma_med"]))})
        print(f"  fold {f['year']}  c={c:.4f}  mean/median={acc[-1]['mm']:.4f}  "
              f"(c the mean reading would still need: {c_mean:.4f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not acc:
        print("no usable folds"); return 1

    rv = np.concatenate([r["rv"] for r in acc])
    L = block_len_for(PROD_H, len(rv))
    q = {k: qlike_vec(rv, np.concatenate([r[k] for r in acc]))
         for k in ("R0", "R1", "R2")}
    ratio = {k: float(np.nanmean(rv ** 2 / np.maximum(
        np.concatenate([r[k] for r in acc]), 1e-12) ** 2))
        for k in ("R0", "R1", "R2")}

    print("\n" + "=" * 92)
    print(f"PRODUCTION SLICE  {len(rv):,} test episodes  blocks of {L}")
    print(f"mean c {np.mean([r['c'] for r in acc]):.4f}   "
          f"mean model mean/median {np.mean([r['mm'] for r in acc]):.4f}")
    print("=" * 92)
    print(f"  {'reading':>26} {'QLIKE':>9} {'vs R0':>10} {'rel %':>8} "
          f"{'ratio':>7}   paired CI")
    print(f"  {'R0  sigma_med (shipped)':>26} {np.nanmean(q['R0']):9.5f} "
          f"{'-':>10} {'-':>8} {ratio['R0']:7.4f}   (reference)")
    out = {"family_size": N_FAMILY, "alpha": alpha, "block_len": L,
           "n_test": int(len(rv)), "horizon": PROD_H,
           "c_per_fold": [float(r["c"]) for r in acc],
           "mean_over_median_per_fold": [float(r["mm"]) for r in acc],
           "readings": {"R0": {"qlike": float(np.nanmean(q["R0"])),
                               "calib_ratio": ratio["R0"]}}}
    for k, label in (("R1", "R1  sigma_mean (no fit)"),
                     ("R2", "R2  c * sigma_med (1 fit)")):
        d = q["R0"] - q[k]
        g = np.isfinite(d)
        ci = mean_ci(d[g], alpha=alpha, block_len=L)
        rel = 100 * float(np.nanmean(d)) / float(np.nanmean(q["R0"]))
        clears = bool(ci["ci95"][0] > 0)
        inband = bool(RATIO_LO <= ratio[k] <= RATIO_HI)
        print(f"  {label:>26} {np.nanmean(q[k]):9.5f} {np.nanmean(d):+10.5f} "
              f"{rel:+8.2f} {ratio[k]:7.4f}   "
              f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
              f"  {'clears' if clears else 'does not clear'}"
              f", ratio {'in band' if inband else 'OUT OF BAND'}")
        out["readings"][k] = {"qlike": float(np.nanmean(q[k])),
                              "vs_r0": float(np.nanmean(d)), "rel_pct": rel,
                              "ci": list(ci["ci95"]), "clears": clears,
                              "calib_ratio": ratio[k], "ratio_in_band": inband}

    ok = [k for k in ("R1", "R2")
          if out["readings"][k]["clears"] and out["readings"][k]["ratio_in_band"]]
    print(f"\n  guard band [{RATIO_LO}, {RATIO_HI}]")
    print(f"  readings that clear the primary AND the ratio guard: "
          f"{', '.join(ok) if ok else 'none'}")
    print("\n  ADOPTION STILL REQUIRES the serving regression test: every "
          "touch_prob,\n  safe_level, p_up and p_vol_amplify bit-identical, "
          "exactly two sigma fields moved.\n  Nothing in THIS file can fail "
          "that test, which is why it is not the gate.")
    out["candidates"] = ok
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
