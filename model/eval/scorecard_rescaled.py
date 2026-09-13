"""
eval/scorecard_rescaled.py
=====================================================================
P2-scorecard-rescaled: does the teacher ranking survive giving every teacher
its own calib-fitted level scalar?

WHY THIS EXISTS

Three separate Phase 2 findings have now reduced to forecast LEVEL rather than
conditional information:

  * garch_normal's spike and deep-tail wins at all four horizons -- rescaling
    destroyed them, and at H=168 its spike went 0.239 -> 1.620 (P2-mechanism);
  * NOCTUA's whole deficit against the zoo, which is one stable constant per
    horizon (the scale falsifier);
  * har_short's fitting PANEL -- pooling horizons looked worth 8.11% at H=6
    raw, CI [+0.02559, +0.04341], and is worth +0.40% CI [-0.00265, +0.00580]
    once each panel carries its own scalar (P2-pool-composition).

`teacher_scorecard.json` ranks teachers by RAW pooled QLIKE and prints each
one's calibration ratio in the next column. At H=6 those ratios run from 0.8606
to 1.4642. A ratio of 1.05 against a ratio of 1.39 is a level advantage of
roughly 13% handed over before any conditional skill is compared -- and garch_t,
the declared winner at H=1 and H=6, is the teacher whose ratio is already near
1. The same arithmetic that explained the panel effect is available to explain
the ranking, and it has not been ruled out.

WHAT IS AND IS NOT BEING CLAIMED

This is a RANKING question. It does not licence adoption of anything: P2-scale-
v2 established that applying a level correction to the shipped model improves
QLIKE 9.7% and degrades all six barrier metrics. A rescaled ranking says which
teacher carries more CONDITIONAL information, which is what a teacher is for.

THE CONSTANT IS FITTED ON CALIB, PER FOLD, AND THE RUN VOIDS ITSELF IF IT WAS NOT

c fitted on test would drive every calibration ratio to exactly 1 by
construction and the comparison would measure nothing. The post-rescale ratio
is therefore computed on TEST and printed; a value of exactly 1.0000 is
evidence the fit leaked. Per fold rather than pooled because a constant fitted
on calib pooled across all six folds had to be withdrawn once already (R44),
and every read goes through FoldScopedFit.

The leak check itself is PER FOLD for the same reason the fit is. Checking only
the POOLED ratio catches a run where every fold leaked -- that ratio is exactly
1 by construction -- but is blind to a run where some folds leaked and others
did not, because the clean folds pull the pooled average off 1. `--selftest`
holds a 2-of-6-fold leak whose pooled ratio (1.00217) sits FURTHER from 1 than
the honest run's (1.00152), so no tolerance on the pooled number can separate
them: the fold-level check is what has teeth, and the selftest is what proves
it still does.

    python -m model.eval.scorecard_rescaled
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.teacher_scorecard import HORIZONS, YEARS, load_oof, metrics        # noqa: E402
from eval.teacher_zoo import FoldScopedFit                                   # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402

N_FAMILY = 4                       # one rank-change contrast per horizon
LEAK_TOL = 1e-9                    # a post-rescale TEST ratio of exactly 1


def optimal_c(rv, sig):
    return float(np.sqrt(max(np.nanmean(rv ** 2 / np.maximum(sig, 1e-12) ** 2),
                             1e-12)))


def test_ratio(rv, sig) -> float:
    """The post-rescale calibration ratio on one fold's test slice."""
    return float(np.nanmean(rv ** 2 / np.maximum(sig, 1e-12) ** 2))


def leak_folds(d: dict) -> list:
    """Folds whose post-rescale TEST ratio is 1 to within LEAK_TOL.

    A ratio of exactly 1 on the slice being scored means the constant was
    fitted on that slice. Returned per fold, because the pooled ratio of a
    partially-leaking run is not near 1 and says nothing (see module docstring).
    """
    return [y for y, r in zip(d["years"], d["fold_ratio"])
            if abs(r - 1.0) < LEAK_TOL]


def gather_rescaled(z, H: int, teacher: str, rescale: bool):
    """Test-slice predictions, optionally each fold rescaled by its own calib c.

    Mirrors teacher_scorecard.gather so the raw column here reproduces the
    published scorecard exactly -- which is the check that this file is
    measuring the same object and not a re-derivation of it.
    """
    rv, sig, fold_q, years, cs, fold_ratio = [], [], [], [], [], []
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kt, kc = f"{y}/{H}/test", f"{y}/{H}/calib"
            if f"{kt}/sigma/{teacher}" not in z:
                continue
            s = np.asarray(sc.test(z, H, teacher), np.float64)
            r = np.asarray(z[f"{kt}/rv"], np.float64)
            ok = np.isfinite(s) & (s > 0)
            if ok.mean() < 0.95:
                continue
            s = np.where(ok, s, np.nan)
            c = 1.0
            if rescale:
                if f"{kc}/sigma/{teacher}" not in z:
                    continue
                sc_ = np.asarray(sc.calib(z, H, teacher), np.float64)
                rc = np.asarray(z[f"{kc}/rv"], np.float64)
                okc = np.isfinite(sc_) & (sc_ > 0)
                c = optimal_c(rc[okc], sc_[okc])
        s = c * s
        rv.append(r); sig.append(s); cs.append(c)
        fold_q.append(float(np.nanmean(qlike_vec(r, s))))
        fold_ratio.append(test_ratio(r, s))
        years.append(y)
    if not rv:
        return None
    rv = np.concatenate(rv); sig = np.concatenate(sig)
    return {"rv": rv, "sigma": sig, "q": qlike_vec(rv, sig),
            "per_fold": fold_q, "years": years, "c": cs,
            "fold_ratio": fold_ratio}


def _synth(H: int, teacher: str, seed: int = 7) -> dict:
    """A shape-only oof archive. No research number is read off it.

    Holes are punched at DIFFERENT rates in calib and test so the two valid
    index sets genuinely differ -- the condition under which the per-fold
    check was claimed to be defeated.
    """
    rng = np.random.default_rng(seed)
    z = {}
    for y in YEARS:
        for sl, n, hole in (("test", 900, 0.03), ("calib", 400, 0.12)):
            sig = np.exp(rng.normal(-4.0, 0.35, n))
            rv = sig * 1.22 * np.exp(rng.normal(0, 0.5, n))
            sig[rng.random(n) < hole] = np.nan
            rv[rng.random(n) < 0.02] = np.nan
            z[f"{y}/{H}/{sl}/sigma/{teacher}"] = sig
            z[f"{y}/{H}/{sl}/rv"] = rv
    return z


def _leaky(z, H: int, teacher: str, n_leak: int) -> dict:
    """gather_rescaled with the first `n_leak` folds' c fitted on TEST."""
    rv, sig, fold_q, years, cs, fr = [], [], [], [], [], []
    for i, y in enumerate(YEARS):
        kt, kc = f"{y}/{H}/test", f"{y}/{H}/calib"
        s = np.asarray(z[f"{kt}/sigma/{teacher}"], np.float64)
        r = np.asarray(z[f"{kt}/rv"], np.float64)
        ok = np.isfinite(s) & (s > 0)
        s = np.where(ok, s, np.nan)
        if i < n_leak:
            c = optimal_c(r[ok], s[ok])
        else:
            sc_ = np.asarray(z[f"{kc}/sigma/{teacher}"], np.float64)
            rc = np.asarray(z[f"{kc}/rv"], np.float64)
            okc = np.isfinite(sc_) & (sc_ > 0)
            c = optimal_c(rc[okc], sc_[okc])
        s = c * s
        rv.append(r); sig.append(s); cs.append(c)
        fold_q.append(float(np.nanmean(qlike_vec(r, s))))
        fr.append(test_ratio(r, s)); years.append(y)
    rv = np.concatenate(rv); sig = np.concatenate(sig)
    return {"rv": rv, "sigma": sig, "q": qlike_vec(rv, sig),
            "per_fold": fold_q, "years": years, "c": cs, "fold_ratio": fr}


def selftest() -> int:
    """The guard must fire on a leak in ANY fold, not only in all of them."""
    H, T = 24, "probe"
    z = _synth(H, T)
    checks = []

    clean = gather_rescaled(z, H, T, rescale=True)
    checks.append(("honest-run-passes", not leak_folds(clean),
                   f"leak folds {leak_folds(clean)}, pooled ratio "
                   f"{test_ratio(clean['rv'], clean['sigma']):.9f}"))

    for k in range(1, len(YEARS) + 1):
        d = _leaky(z, H, T, k)
        bad = leak_folds(d)
        checks.append((f"leak-in-{k}-of-{len(YEARS)}-folds-caught",
                       len(bad) == k,
                       f"named {len(bad)} of {k}: {bad}"))

    # the point of the per-fold check: the pooled number cannot do this job
    two = _leaky(z, H, T, 2)
    p_clean = abs(test_ratio(clean["rv"], clean["sigma"]) - 1.0)
    p_leak = abs(test_ratio(two["rv"], two["sigma"]) - 1.0)
    checks.append(("pooled-ratio-cannot-separate", p_leak > p_clean,
                   f"|pooled-1| honest {p_clean:.6f} vs 2-fold leak "
                   f"{p_leak:.6f} -- the leaking run looks MORE honest"))

    # and a fold-level check on an all-fold leak still fires
    allf = _leaky(z, H, T, len(YEARS))
    checks.append(("all-fold-leak-also-pooled-detectable",
                   abs(test_ratio(allf["rv"], allf["sigma"]) - 1.0) < LEAK_TOL,
                   f"pooled ratio {test_ratio(allf['rv'], allf['sigma']):.12f}"))

    print("scorecard_rescaled leak-guard selftest")
    bad = 0
    for name, ok, detail in checks:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-scorecard-rescaled")
    ap.add_argument("--selftest", action="store_true",
                    help="prove the leak guard fires on a partial-fold leak")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/scorecard_rescaled.json"))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, teachers = load_oof(a.oof)
    alpha = 0.05 / N_FAMILY
    print(f"P2-scorecard-rescaled   {len(teachers)} teachers   "
          f"family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("c is fitted on CALIB, per fold. The post-rescale ratio below is "
          "computed on TEST\nand must NOT be exactly 1.0000 -- that would mean "
          "the constant was fitted on test.\n")

    out = {"family_size": N_FAMILY, "alpha": alpha, "horizons": {}}
    for H in HORIZONS:
        raw, resc = {}, {}
        for t in teachers:
            d0 = gather_rescaled(z, H, t, rescale=False)
            d1 = gather_rescaled(z, H, t, rescale=True)
            if d0 is None or d1 is None:
                continue
            raw[t], resc[t] = metrics(d0), metrics(d1)
            resc[t]["c_per_fold"] = [float(c) for c in d1["c"]]
            resc[t]["fold_ratio"] = [float(r) for r in d1["fold_ratio"]]
            bad = leak_folds(d1)
            if bad or abs(resc[t]["calib_ratio"] - 1.0) < LEAK_TOL:
                where = (f"fold(s) {bad}" if bad else "the pooled slice")
                raise SystemExit(
                    f"REFUSING: {t} at H={H} has a post-rescale TEST "
                    f"calibration ratio of exactly 1 on {where}. The constant "
                    f"was fitted on the slice it is being checked against, so "
                    f"this run measures nothing.")
        if not raw:
            continue
        n = raw[next(iter(raw))]["n"]
        print("=" * 100)
        print(f"H = {H}h   {n:,} test episodes")
        print("=" * 100)
        print(f"{'teacher':>14} | {'RAW pooled':>10} {'ratio':>7} | "
              f"{'RESCALED':>10} {'ratio':>7} {'gain %':>8} | {'c (mean)':>9}")
        for t in sorted(raw, key=lambda k: raw[k]["pooled"]):
            g = 100 * (raw[t]["pooled"] - resc[t]["pooled"]) / raw[t]["pooled"]
            cm = float(np.mean(resc[t]["c_per_fold"]))
            print(f"{t:>14} | {raw[t]['pooled']:10.5f} "
                  f"{raw[t]['calib_ratio']:7.4f} | {resc[t]['pooled']:10.5f} "
                  f"{resc[t]['calib_ratio']:7.4f} {g:+8.2f} | {cm:9.4f}")

        w_raw = min(raw, key=lambda k: raw[k]["pooled"])
        w_res = min(resc, key=lambda k: resc[k]["pooled"])
        changed = w_raw != w_res
        print(f"\n  best raw: {w_raw}    best rescaled: {w_res}    "
              f"{'RANK CHANGED' if changed else 'rank held'}")
        row = {"raw": raw, "rescaled": resc, "best_raw": w_raw,
               "best_rescaled": w_res, "rank_changed": bool(changed)}
        if changed:
            a_ = gather_rescaled(z, H, w_res, True)
            b_ = gather_rescaled(z, H, w_raw, True)
            d = qlike_vec(b_["rv"], b_["sigma"]) - qlike_vec(a_["rv"], a_["sigma"])
            g = np.isfinite(d)
            L = block_len_for(H, int(g.sum()))
            ci = mean_ci(d[g], alpha=alpha, block_len=L)
            print(f"  paired, both rescaled: {w_res} beats {w_raw} by "
                  f"{np.nanmean(d):+.5f}  CI [{ci['ci95'][0]:+.5f}, "
                  f"{ci['ci95'][1]:+.5f}]  blocks of {L}"
                  + ("  CLEARS" if ci["ci95"][0] > 0 else "  does not clear"))
            row["rank_change_ci"] = list(ci["ci95"])
            row["rank_change_gain"] = float(np.nanmean(d))
            row["rank_change_clears"] = bool(ci["ci95"][0] > 0)
        out["horizons"][str(H)] = row
        print()

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
