"""
eval/dispersion.py
=====================================================================
P3-dispersion: the mean functional's calibration is two errors cancelling, not
two errors being small -- and where they stop cancelling is H=168.

WHAT THIS EXPLAINS

`P3-functional-audited` reported calibration ratios of 0.963 / 0.988 / 0.992 /
0.899 for the adopted mean functional and read the first three as near-perfect
with nothing fitted. They are not. Under lognormal log-vol the ratio factors
exactly:

    E[RV^2] / sigma_mean^2 = exp(2*(mu_true - mu)) * exp(2*(s_true^2 - s_imp^2))
                             \_____ LEVEL _____/     \____ DISPERSION ____/

and the two parts run in opposite directions at the short horizons:

    H        ratio     dispersion      level
    1       0.9630       1.0879       0.8852
    6       0.9880       0.9313       1.0609
    24      0.9915       0.8719       1.1371
    168     0.8992       0.9380       0.9587

At H=24 a 13% dispersion error and a 14% level error cancel to 0.8%. At H=168
a 6% dispersion error and a 4% level error COMPOUND to 10%. So H=168 is not a
separate defect needing its own explanation -- it is the same two errors
failing to cancel, and the three "good" horizons are good by coincidence.

HOW THE IMPLIED DISPERSION IS RECOVERED WITHOUT RE-RUNNING THE MODEL

For lognormal log-vol with sd s, sigma_mean / sigma_med = exp(s^2). Both
scalars are already emitted per episode and stored in the out-of-fold archive,
so the model's own implied dispersion is available by inversion. The realised
dispersion is the sd of log(RV / sigma_med) -- the spread of outcomes about the
predicted median, which is what the implied figure should equal.

TWO CAVEATS, BOTH OF WHICH CUT AGAINST THE READING HERE

1. LOGNORMALITY. The factorisation is exact only under it. The atoms are
   quantiles of a learned distribution with no such guarantee, so the split
   between level and dispersion is an approximation even though the total is
   measured directly.

2. RV IS A NOISY ESTIMATOR of integrated variance, and its noise inflates the
   realised sd. That biases every row toward "under-dispersed", worst at H=1
   where a window contains fewest observations. So the H=1 under-dispersion
   reading is unreliable and may be measurement noise entirely, while the
   over-dispersion at H=6/24/168 survives a bias pointing the other way --
   the conservative direction, which is why it is the part worth acting on.

WHY IT MATTERS BEYOND A REPORTED SCALAR

The committee builds every barrier curve from `sigma_atoms`. A dispersion error
of 11-28% is an error in the object those curves are built from, so it reaches
the product rather than only the published number.

    python -m model.eval.dispersion --selftest
    python -m model.eval.dispersion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mz_recalibration import YEARS                                # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                  # noqa: E402

MED, MEAN = "noctua_v1", "noctua_v1_mean"


def implied_s(sig_med, sig_mean):
    """s from the model's own two scalars: sigma_mean/sigma_med = exp(s^2)."""
    med = np.asarray(sig_med, np.float64); mn = np.asarray(sig_mean, np.float64)
    ok = np.isfinite(med) & np.isfinite(mn) & (med > 0) & (mn > 0)
    out = np.full(med.shape, np.nan)
    out[ok] = np.sqrt(np.maximum(np.log(mn[ok] / med[ok]), 0.0))
    return out


def decompose(rv, sig_med, sig_mean) -> dict:
    """Split the calibration ratio into level and dispersion contributions."""
    rv = np.asarray(rv, np.float64)
    med = np.asarray(sig_med, np.float64); mn = np.asarray(sig_mean, np.float64)
    ok = (np.isfinite(rv) & np.isfinite(med) & np.isfinite(mn)
          & (rv > 0) & (med > 0) & (mn > 0))
    if ok.sum() < 50:
        return {}
    s_imp = float(np.nanmean(implied_s(med[ok], mn[ok])))
    s_real = float(np.nanstd(np.log(rv[ok] / med[ok]), ddof=1))
    ratio = float(np.nanmean(rv[ok] ** 2 / mn[ok] ** 2))
    disp = float(np.exp(2.0 * (s_real ** 2 - s_imp ** 2)))
    return {"n": int(ok.sum()), "s_implied": s_imp, "s_realised": s_real,
            "disp_ratio": s_imp / s_real if s_real > 0 else float("nan"),
            "calib_ratio": ratio, "part_dispersion": disp,
            "part_level": ratio / disp if disp > 0 else float("nan")}


def gather_h(z, H: int) -> dict:
    rv, med, mn = [], [], []
    for y in YEARS:
        k = f"{y}/{H}/test"
        if f"{k}/sigma/{MED}" not in z or f"{k}/sigma/{MEAN}" not in z:
            continue
        rv.append(np.asarray(z[f"{k}/rv"], np.float64))
        med.append(np.asarray(z[f"{k}/sigma/{MED}"], np.float64))
        mn.append(np.asarray(z[f"{k}/sigma/{MEAN}"], np.float64))
    if not rv:
        return {}
    return decompose(np.concatenate(rv), np.concatenate(med), np.concatenate(mn))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="level/dispersion decomposition")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/dispersion.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    rows = {H: gather_h(z, H) for H in HORIZONS}
    rows = {H: r for H, r in rows.items() if r}
    print("IMPLIED vs REALISED predictive dispersion, and what the calibration")
    print("ratio is actually made of.\n")
    print(f"{'H':>5} {'n':>8} {'s implied':>10} {'s realised':>11} {'imp/real':>9}"
          f" | {'ratio':>8} {'= disp':>8} {'x level':>9}")
    for H, r in rows.items():
        print(f"{H:>5} {r['n']:>8,} {r['s_implied']:10.4f} {r['s_realised']:11.4f} "
              f"{r['disp_ratio']:9.3f} | {r['calib_ratio']:8.4f} "
              f"{r['part_dispersion']:8.4f} {r['part_level']:9.4f}")
    print("\nCANCELLATION, not calibration: at H=6 and H=24 the two parts run in")
    print("OPPOSITE directions and nearly cancel, so a ratio of 0.988/0.992 hides")
    print("errors of 7-13% in each. At H=168 they COMPOUND, which is the whole")
    print("of why that horizon is the outlier -- not a separate defect.")
    print("\nCaveats: the split assumes lognormality (the TOTAL is measured, the")
    print("SPLIT is not); and RV's estimation noise inflates `s realised`,")
    print("biasing every row toward under-dispersion, worst at H=1 -- so the")
    print("H=1 reading is unreliable and the others are conservative.")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"teacher_med": MED, "teacher_mean": MEAN,
         "rows": {str(H): r for H, r in rows.items()}}, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(17)
    n = 60000

    # 1-2. Planted lognormal: implied_s must invert the ratio exactly, and a
    #      CORRECTLY specified model must decompose to 1.0 on both parts.
    mu, s = -4.0, 0.40
    med = np.exp(mu) * np.ones(n)
    mean = med * np.exp(s * s)                      # sigma_mean/sigma_med = e^{s^2}
    ok.append(("implied_s inverts the ratio",
               abs(float(np.nanmean(implied_s(med, mean))) - s) < 1e-9))
    rv = np.exp(rng.normal(mu, s, n))               # realised matches implied
    d = decompose(rv, med, mean)
    ok.append(("a correctly specified model decomposes to ~1 on both parts",
               abs(d["part_dispersion"] - 1.0) < 0.05
               and abs(d["part_level"] - 1.0) < 0.05))

    # 3-4. OVER-dispersed: the model claims more spread than reality delivers.
    #      The dispersion part must fall BELOW 1 and the sign must be right.
    mean_over = med * np.exp((s * 1.5) ** 2)
    d_over = decompose(rv, med, mean_over)
    ok.append(("over-dispersion pushes the dispersion part below 1",
               d_over["part_dispersion"] < 0.9))
    ok.append(("over-dispersion is detected as imp/real > 1",
               d_over["disp_ratio"] > 1.2))

    # 5. UNDER-dispersed must push the other way -- a one-sided test would pass
    #    on a function that always returned "over".
    mean_under = med * np.exp((s * 0.6) ** 2)
    ok.append(("under-dispersion pushes the dispersion part above 1",
               decompose(rv, med, mean_under)["part_dispersion"] > 1.1))

    # 6. A pure LEVEL error must leave the dispersion part alone. This is the
    #    check that the split actually separates the two, rather than smearing
    #    one into the other.
    d_lvl = decompose(rv * 1.2, med, mean)
    ok.append(("a pure level shift moves only the level part",
               abs(d_lvl["part_dispersion"] - d["part_dispersion"]) < 0.02
               and d_lvl["part_level"] > d["part_level"] * 1.3))

    # 7. The parts must MULTIPLY back to the measured total, which is the one
    #    quantity here that rests on no distributional assumption.
    ok.append(("parts reconstruct the measured ratio",
               abs(d_over["part_dispersion"] * d_over["part_level"]
                   - d_over["calib_ratio"]) < 1e-9))

    # 8. Refuses on too little data rather than reporting a decomposition of it.
    ok.append(("refuses below 50 usable rows",
               decompose(rv[:10], med[:10], mean[:10]) == {}))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
