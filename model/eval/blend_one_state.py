"""
eval/blend_one_state.py
=====================================================================
The strictly simpler hypothesis the two-state test left untouched.

WHAT HAPPENED FIRST

`eval/blend_ceiling.py` tested a TWO-state ensemble weight -- one w for
spike-risk episodes, one for calm -- and returned NULL: shrunk pooled QLIKE
+0.00307, CI [-0.01008, +0.02603]. Its constant-w sweep, though, is a separate
and much simpler object, and it is not flat:

    w      0.00    0.15    0.25    0.35    0.45    0.55    0.75    1.00
    pooled 0.3057  0.2952  0.2904  0.2875  0.2863  0.2870  0.2940  0.3138
                           ^shipped        ^minimum

The six-fold mean is minimised at w = 0.45, 1.41% below the shipped 0.25. That
number is NOT a result and must not be quoted as one: it is chosen after seeing
every fold's test score, which is the definition of tuning on test. It is a
reason to run an honest version, and nothing more.

The honest version is this file: pick the constant on folds STRICTLY BEFORE the
scored fold, apply it out of sample. One estimated parameter instead of two.
The combination-puzzle literature cited in `blend_ceiling.py` -- Smith & Wallis
(OBES 2009), Claeskens et al. (IJF 2016), and Clements & Vasnev's HAR-specific
result (J. Forecasting 2024) -- all say the penalty for estimating a weight
scales with how many you estimate, so the one-parameter version is the version
with the best prior odds, and it was never tested.

THIS IS THE SECOND LOOK AT THIS DATA, AND THAT IS STATED UP FRONT

The two-state test on these same six folds returned NULL. Running a second
hypothesis against the same forecasts is a multiple comparison, and the nominal
95% interval below is therefore optimistic -- the true family-wise error rate
across the two tests is higher than 5%. Genre, Kenny, Meyler & Timmermann
(IJF 2013) found most combination schemes stop beating a simple average once
exactly this correction is applied.

Two things follow, both adopted rather than merely noted:

  * this experiment cannot ADOPT on its own under any outcome. A favourable
    result earns a full-pipeline confirmation run, which is a fresh test;
  * the rule below requires the interval to clear zero with room -- the
    Bonferroni-adjusted 97.5% interval, not the nominal 95% one -- so the
    second look is paid for rather than ignored.

No model is retrained. This reads `blend_ceiling.npz`, the cached out-of-sample
components, so the forecasts being re-weighted are byte-identical to the ones
the two-state test used.

PRE-REGISTERED RULE, fixed before any score is read

  ARMS, both walk-forward, both fitted on prior folds only:
    RAW     the pooled-history grid argmin, adopted as-is.
    SHRUNK  the same argmin pulled toward the shipped w0 = 0.25 by
            lam = m / (m + (sd_f w_argmin[f] / 0.25) ** 2), the identical
            estimator blend_ceiling applies -- constants unchanged, so this is
            not a second tuning opportunity.

  PRIMARY: pooled QLIKE of the SHRUNK arm versus the shipped constant, over
  folds with at least one prior fold, with a bootstrap CI at the
  BONFERRONI-ADJUSTED level (97.5%, two tests) that must exclude zero on the
  favourable side.

  GUARDS, all of which must hold:
    - spike-episode QLIKE must not worsen (own CI, same adjusted level);
    - calm-episode QLIKE must not worsen by more than 1%;
    - WORST-FOLD BOUND: no fold may be worse than the shipped constant by more
      than 5% of that fold's baseline. This guard exists because w = 0.25 was
      not chosen to minimise the mean -- `infer.BLEND_W`'s own note records
      that it was chosen to bound the worst fold at +6.7% where pure NOCTUA
      suffered +72.3% in the 2023 volatility collapse. A rule that improves the
      mean by giving that back has not improved anything, and the two-state
      test already showed 2023 is exactly where a raised weight is punished
      (+19.20% raw, +10.53% shrunk).
    - CANNOT ADOPT ALONE, per the second-look argument above.

  REJECTION: if the shrunk arm's adjusted CI contains zero, the ensemble weight
  is not a lever at this sample size in either its one- or two-state form, and
  the 0.45 minimum in the sweep is an artifact of choosing after the fact.

    python -m model.eval.blend_one_state
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.blend_ceiling import W0, W_GRID, shrink, sigma_at, snap          # noqa: E402
from eval.direction import ci_excludes_zero, mean_ci                       # noqa: E402
from eval.levers import qlike                                              # noqa: E402
from research import pitfalls as P                                         # noqa: E402

N_TESTS = 2          # this one and blend_ceiling's two-state test
WORST_FOLD_BAR = 5.0  # percent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One-state walk-forward blend weight")
    ap.add_argument("--components", type=Path,
                    default=Path("model/artifacts/blend_ceiling.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/blend_one_state.json"))
    a = ap.parse_args(argv)

    if not a.components.exists():
        raise SystemExit(
            f"REFUSING: {a.components} not found. Run "
            f"`python -m model.eval.blend_ceiling` first -- this file re-weights "
            f"ITS cached out-of-sample forecasts and never trains a model.")
    npz = np.load(a.components)
    years = sorted({int(k.split("_")[-1]) for k in npz.files})

    folds = []
    for y in years:
        rv, raw, har, H = (npz[f"{k}_{y}"] for k in ("rv", "raw", "har", "H"))
        sp = npz[f"spike_{y}"].astype(bool)
        Q = np.stack([qlike(rv, sigma_at(w, raw, har, H)) for w in W_GRID])
        folds.append({"year": y, "Q": Q, "spike": sp, "n": len(rv),
                      "mean": Q.mean(axis=1),
                      "argmin": float(W_GRID[int(np.argmin(Q.mean(axis=1)))])})
    j0 = int(np.argmin(np.abs(W_GRID - W0)))

    print(f"{'year':>6} {'n':>5} {'own argmin':>11} {'raw':>6} {'lam':>6} "
          f"{'shrunk':>7} {'base':>9} {'raw%':>8} {'shrunk%':>9}")
    dlt = {k: {"pooled": [], "spike": [], "calm": []} for k in ("raw", "shrunk")}
    rows, worst = [], {"raw": -1e9, "shrunk": -1e9}
    for i, f in enumerate(folds):
        hist = folds[:i]
        if not hist:
            print(f"{f['year']:>6} {f['n']:>5}  {f['argmin']:>10.2f}"
                  f"   (no prior fold -- excluded from the test)")
            continue
        # RAW: argmin of the mean-over-history QLIKE curve.
        curve = np.mean([h["mean"] for h in hist], axis=0)
        w_raw = float(W_GRID[int(np.argmin(curve))])
        w_sh, lam, _, sd = shrink([h["argmin"] for h in hist])
        w_sh = snap(w_sh)
        base = float(f["mean"][j0])
        row = {"year": f["year"], "n": f["n"], "own_argmin": f["argmin"],
               "w_raw": w_raw, "w_shrunk": w_sh, "lam": lam, "sd_hist": sd,
               "base": base, "hist_argmin": [h["argmin"] for h in hist]}
        for tag, w in (("raw", w_raw), ("shrunk", w_sh)):
            j = int(np.argmin(np.abs(W_GRID - w)))
            q = f["Q"][j]
            qb = f["Q"][j0]
            dlt[tag]["pooled"].append(float(q.mean() - qb.mean()))
            dlt[tag]["calm"].append(float(q[~f["spike"]].mean() - qb[~f["spike"]].mean()))
            if f["spike"].any():
                dlt[tag]["spike"].append(float(q[f["spike"]].mean() - qb[f["spike"]].mean()))
            row[f"pooled_{tag}"] = float(q.mean())
            worst[tag] = max(worst[tag], 100.0 * (q.mean() - qb.mean()) / base)
        rows.append(row)
        print(f"{f['year']:>6} {f['n']:>5}  {f['argmin']:>10.2f} {w_raw:>6.2f} "
              f"{lam:>6.2f} {w_sh:>7.2f} {base:>9.5f} "
              f"{100*dlt['raw']['pooled'][-1]/base:>+8.2f} "
              f"{100*dlt['shrunk']['pooled'][-1]/base:>+9.2f}")

    if len(rows) < 2:
        print("fewer than 2 scored folds -- no verdict"); return 1

    alpha = 0.05 / N_TESTS          # Bonferroni over the two blend-weight tests
    out = {"n_tests": N_TESTS, "alpha": alpha, "worst_fold_pct": worst,
           "folds": rows, "w0": W0}
    print(f"\nCIs at the Bonferroni-adjusted level "
          f"({100*(1-alpha):.1f}%, {N_TESTS} tests on these forecasts)")
    print(f"{'arm':>8} {'quantity':>8} {'delta':>11} {'CI':>26} {'signs':>8}")
    for tag in ("raw", "shrunk"):
        out[tag] = {}
        for nm in ("pooled", "spike", "calm"):
            arr = np.asarray(dlt[tag][nm], np.float64)
            ci = mean_ci(arr, seed=43, alpha=alpha)
            out[tag][nm] = {"delta": float(arr.mean()), "ci": ci["ci95"],
                            "level": ci["level"],
                            "n_negative": ci["n_negative"],
                            "n_positive": ci["n_positive"]}
            print(f"{tag:>8} {nm:>8} {arr.mean():+11.5f}   "
                  f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]   "
                  f"{ci['n_negative']}-/{ci['n_positive']}+")

    base_calm = float(np.mean([np.mean(f["Q"][j0][~f["spike"]]) for f in folds[1:]]))
    calm_pct = 100.0 * np.mean(dlt["shrunk"]["calm"]) / max(base_calm, 1e-12)
    primary = ci_excludes_zero(out["shrunk"]["pooled"]["ci"], -1)
    g_spike = not ci_excludes_zero(out["shrunk"]["spike"]["ci"], +1)
    g_calm = calm_pct <= 1.0
    g_worst = worst["shrunk"] <= WORST_FOLD_BAR
    ok = primary and g_spike and g_calm and g_worst
    out["verdict"] = "ADVANCE" if ok else "REJECT"
    print(f"\n--- pre-registered rule ---")
    print(f"  PRIMARY shrunk pooled CI clears zero favourably (adj) : {primary}")
    print(f"  GUARD   spike QLIKE not worse                         : {g_spike}")
    print(f"  GUARD   calm QLIKE within 1%  ({calm_pct:+.2f}%)        : {g_calm}")
    print(f"  GUARD   worst fold within {WORST_FOLD_BAR:.0f}%  "
          f"({worst['shrunk']:+.2f}%)          : {g_worst}")
    print(f"  -> {out['verdict']}"
          + ("  (NOT an adoption -- second look, needs a fresh confirmation run)"
             if ok else ""))

    rep = P.Report()
    rep.add(P.check_ci_is_defined(out["shrunk"]["pooled"]["ci"], "shrunk pooled"))
    rep.add(P.check_not_a_coin_flip(dlt["shrunk"]["pooled"], "shrunk pooled delta"))
    rep.add(P.check_rule_satisfiable(2, len(rows), "folds"))
    print("\n--- research/pitfalls on this experiment ---")
    print(rep.render())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
