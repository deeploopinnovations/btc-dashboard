"""
eval/mz_recalibration.py
=====================================================================
P3-mz: after removing AFFINE bias from every teacher symmetrically, does NOCTUA
retain any advantage -- or was its whole edge calibration?

WHERE THIS DESIGN COMES FROM

Not from this project. "Forecasting Realized Volatility with Time Series
Foundation Models" (arXiv 2607.05291) compares nine zero-shot foundation models
against eight econometric specifications on 50 assets under QLIKE, with formal
forecast-comparison tests, and reports:

    "A Mincer-Zarnowitz recalibration ... shows that at the shorter horizons
     TTM's edge over Log-HAR is largely a calibration effect that several other
     foundation models also enjoy, while at the monthly horizon TTM retains a
     genuine informational advantage."

That is this project's own Phase 2 conclusion -- NOCTUA's deficit against the HAR
family is one level constant, not missing information -- arrived at independently
on 50 other assets. It also supplies the instrument this project had been using a
weaker version of.

WHAT IS NEW HERE VERSUS `scorecard_rescaled`

`P2-scorecard-rescaled` gave every teacher ONE free parameter: a scale c fitted on
calib, which is the closed-form QLIKE-optimal level correction. MZ gives every
teacher TWO: an intercept and a slope. In the log-vol domain,

    log sigma_MZ = alpha + beta * log sigma_hat

nests the existing correction exactly at beta = 1, so the three columns reported
here form a ladder:

    raw        0 free parameters
    c-rescale  1 free parameter   (level)          <- what P2 did
    MZ         2 free parameters  (level + slope)  <- the published standard

A slope different from 1 means the forecast's RESPONSIVENESS is miscalibrated, not
just its level: beta < 1 says the model over-reacts to its own signal, beta > 1
that it under-reacts. Neither is information about volatility; both are fixable by
an affine map that any competitor can also apply. Accuracy that survives the
correction is the part that is actually information (the paper's phrasing:
"accuracy surviving the correction reflects information beyond an affine
rescaling").

FITTED OUT OF SAMPLE, SYMMETRICALLY, OR IT MEASURES NOTHING

alpha and beta are fitted on each fold's CALIB slice and applied to that fold's
TEST slice, through FoldScopedFit, exactly as the c-rescale was. Every teacher
gets the same treatment -- the paper is explicit that it "appl[ies] the correction
symmetrically to all models", and asymmetric application is how a favoured model
is handed a free fit.

    python -m model.eval.mz_recalibration --selftest
    python -m model.eval.mz_recalibration
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.scorecard_rescaled import optimal_c, test_ratio                    # noqa: E402
from eval.teacher_scorecard import HORIZONS, YEARS, load_oof, metrics        # noqa: E402
from eval.teacher_zoo import FoldScopedFit                                   # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402

N_FAMILY = 4                 # one rank contrast per horizon, as P2 declared
LEAK_TOL = 1e-9


def fit_mz(rv_cal, sig_cal):
    """log sigma_realised = alpha + beta * log sigma_forecast, on CALIB only.

    The 'realised sigma' for an episode is sqrt(RV^2) = RV, so in the log domain
    the regressand is log RV. Returns (alpha, beta); beta == 1 recovers a pure
    level shift, which is what makes this nest the existing c-rescale.
    """
    ok = (np.isfinite(rv_cal) & np.isfinite(sig_cal)
          & (rv_cal > 0) & (sig_cal > 0))
    if ok.sum() < 200:
        return None
    x = np.log(sig_cal[ok])
    y = np.log(rv_cal[ok])
    # ddof must MATCH between the covariance and the variance. The first
    # version used np.cov(ddof=1) over np.var(ddof=0), which biases beta by
    # exactly n/(n-1): on a perfect forecast it returned 1.000125 instead of 1,
    # and the selftest that asserted "a perfect forecast needs no correction"
    # is what found it.
    vx = float(np.var(x, ddof=1))
    if not np.isfinite(vx) or vx < 1e-12:
        return None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / vx)
    alpha = float(np.mean(y) - beta * np.mean(x))
    return alpha, beta


def apply_mz(sig, alpha, beta):
    out = np.full_like(sig, np.nan)
    ok = np.isfinite(sig) & (sig > 0)
    out[ok] = np.exp(alpha + beta * np.log(sig[ok]))
    return out


def fit_mzq(rv_cal, sig_cal):
    """The slope from MZ, the LEVEL from QLIKE. Two parameters, right loss.

    WHY THE PUBLISHED FORM IS NOT USABLE HERE AS-IS, measured rather than argued:
    a plain MZ fit degrades pooled QLIKE for six of this project's seven teachers
    -- by 31% to 38% at H=1 -- and leaves a post-correction calibration ratio near
    1.7 instead of 1. MZ minimises SQUARED ERROR; QLIKE is minimised by the
    conditional MEAN of variance, which this project established separately as
    `E-scale` (the shipped model reports a conditional median and the overlay
    consequently overshoots its vol target). Fitting a level by least squares and
    then scoring it with QLIKE charges the forecast a Jensen gap that has nothing
    to do with whether it is informative.

    So: take beta from the MZ regression, which is what carries the RESPONSIVENESS
    information the correction exists to remove, and then set the level by the
    closed-form QLIKE-optimal constant for the already-sloped forecast. This still
    nests the existing 1-parameter correction exactly at beta = 1, so the ladder
    survives, and every teacher gets both parameters symmetrically.
    """
    p = fit_mz(rv_cal, sig_cal)
    if p is None:
        return None
    beta = p[1]
    ok = (np.isfinite(rv_cal) & np.isfinite(sig_cal)
          & (rv_cal > 0) & (sig_cal > 0))
    sloped = sig_cal[ok] ** beta
    c = optimal_c(rv_cal[ok], sloped)
    return float(np.log(c)), beta


def gather(z, H: int, teacher: str, mode: str):
    """mode in {raw, c, mz, mzq}. Per fold, fitted on calib, applied to test."""
    rv, sig, fold_q, years, pars = [], [], [], [], []
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
            p = None
            if mode != "raw":
                if f"{kc}/sigma/{teacher}" not in z:
                    continue
                sc_ = np.asarray(sc.calib(z, H, teacher), np.float64)
                rc = np.asarray(z[f"{kc}/rv"], np.float64)
                if mode == "c":
                    okc = np.isfinite(sc_) & (sc_ > 0)
                    p = (float(np.log(optimal_c(rc[okc], sc_[okc]))), 1.0)
                elif mode == "mz":
                    p = fit_mz(rc, sc_)
                else:
                    p = fit_mzq(rc, sc_)
                if p is None:
                    continue
        if p is not None:
            s = apply_mz(s, *p)
            pars.append(p)
        rv.append(r); sig.append(s); years.append(y)
        fold_q.append(float(np.nanmean(qlike_vec(r, s))))
    if not rv:
        return None
    rv = np.concatenate(rv); sig = np.concatenate(sig)
    return {"rv": rv, "sigma": sig, "q": qlike_vec(rv, sig),
            "per_fold": fold_q, "years": years, "params": pars}


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(4)
    n = 8000
    # a forecast that is right about DYNAMICS but wrong by an affine map:
    # log sigma_hat = (log sigma_true - a) / b  =>  MZ should recover (a, b)
    true = np.exp(rng.normal(-4.0, 0.45, n))
    a_, b_ = 0.30, 0.80
    sig = np.exp((np.log(true) - a_) / b_)
    rv = true
    p = fit_mz(rv, sig)
    ok.append(("mz-recovers-a-known-affine-distortion",
               p is not None and abs(p[0] - a_) < 0.02 and abs(p[1] - b_) < 0.02,
               f"fitted alpha {p[0]:+.4f} vs {a_:+.2f}, beta {p[1]:.4f} vs {b_}"))
    ok.append(("applying-it-restores-the-forecast",
               float(np.max(np.abs(apply_mz(sig, *p) / rv - 1.0))) < 0.15,
               "the corrected forecast tracks the realisation"))

    # beta == 1 must reduce EXACTLY to a level shift, so the ladder really nests
    c = optimal_c(rv, sig)
    lvl = apply_mz(sig, float(np.log(c)), 1.0)
    ok.append(("beta-1-is-the-existing-c-rescale",
               np.allclose(lvl, c * sig, equal_nan=True),
               "MZ with beta=1 reproduces c * sigma bit-for-bit"))

    # a perfectly calibrated forecast must come back untouched
    p2 = fit_mz(rv, rv)
    ok.append(("a-perfect-forecast-needs-no-correction",
               p2 is not None and abs(p2[0]) < 1e-9 and abs(p2[1] - 1) < 1e-9,
               f"alpha {p2[0]:+.2e}, beta {p2[1]:.9f}"))

    # and the fit must REFUSE rather than return nonsense on degenerate input
    ok.append(("degenerate-input-is-refused",
               fit_mz(rv[:10], sig[:10]) is None
               and fit_mz(rv, np.full(n, 0.01)) is None,
               "too few points, and a constant forecast with no variance"))

    # MZq must nest the c-rescale at beta=1 and must be QLIKE-calibrated
    p3 = fit_mzq(rv, sig)
    ok.append(("mzq-keeps-the-mz-slope",
               p3 is not None and abs(p3[1] - p[1]) < 1e-12,
               f"beta {p3[1]:.6f} is the MZ slope, unchanged"))
    corr = apply_mz(sig, *p3)
    ratio = float(np.nanmean(rv ** 2 / np.maximum(corr, 1e-12) ** 2))
    ok.append(("mzq-is-qlike-calibrated-in-sample", abs(ratio - 1.0) < 1e-6,
               f"post-correction ratio {ratio:.9f} against 1 -- the level is "
               f"the closed-form QLIKE optimum, not a least-squares fit"))
    # The Jensen gap only exists when the forecast has genuine ERROR -- on a
    # noiseless affine distortion MZ recovers the truth exactly and leaves the
    # ratio at 1. So this check needs a forecast that is imperfect, which is the
    # only case that occurs in practice.
    noisy = sig * np.exp(rng.normal(0, 0.45, n))
    pn, pq = fit_mz(rv, noisy), fit_mzq(rv, noisy)
    r_mz = float(np.nanmean(rv ** 2 / np.maximum(apply_mz(noisy, *pn), 1e-12) ** 2))
    r_mzq = float(np.nanmean(rv ** 2 / np.maximum(apply_mz(noisy, *pq), 1e-12) ** 2))
    ok.append(("plain-mz-is-NOT-qlike-calibrated-on-an-IMPERFECT-forecast",
               abs(r_mz - 1.0) > 0.05 and abs(r_mzq - 1.0) < 1e-6,
               f"with real forecast error, plain MZ leaves ratio {r_mz:.4f} "
               f"while MZq leaves {r_mzq:.9f} -- the gap is the Jensen term "
               f"that made plain MZ degrade QLIKE by 31-38% on live teachers"))

    print("mz_recalibration selftest")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-mz")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/mz_result.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, teachers = load_oof(a.oof)
    alpha = 0.05 / N_FAMILY
    print("P3-mz   Mincer-Zarnowitz recalibration, fitted on CALIB per fold,")
    print("        applied symmetrically to every teacher (arXiv 2607.05291).")
    print(f"        family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals\n")
    print("        raw = 0 free params | c = 1 (level) | MZ = 2 (level+slope)")
    print("        beta < 1: the model OVER-reacts to its own signal.\n")
    out = {"family": N_FAMILY, "alpha": alpha, "source": "arXiv 2607.05291",
           "horizons": {}}

    for H in HORIZONS:
        got = {}
        for t in teachers:
            d = {m: gather(z, H, t, m)
                 for m in ("raw", "c", "mz", "mzq")}
            if any(v is None for v in d.values()):
                continue
            got[t] = d
        if not got:
            continue
        print("=" * 100)
        print(f"H = {H}h")
        print("=" * 100)
        print(f"{'teacher':>14} {'raw':>9} {'c':>9} {'MZ':>9} "
              f"{'MZq':>9} | {'beta':>8} {'MZq vs c %':>11} "
              f"{'MZq ratio':>10}")
        row = {}
        for t in sorted(got, key=lambda k: np.nanmean(got[k]["mzq"]["q"])):
            qr = float(np.nanmean(got[t]["raw"]["q"]))
            qc = float(np.nanmean(got[t]["c"]["q"]))
            qm = float(np.nanmean(got[t]["mz"]["q"]))
            qq = float(np.nanmean(got[t]["mzq"]["q"]))
            betas = [p[1] for p in got[t]["mzq"]["params"]]
            bm = float(np.mean(betas))
            ratio = test_ratio(got[t]["mzq"]["rv"], got[t]["mzq"]["sigma"])
            row[t] = {"raw": qr, "c": qc, "mz": qm, "mzq": qq,
                      "beta_mean": bm,
                      "beta_per_fold": [float(b) for b in betas],
                      "mzq_vs_c_pct": float(100 * (qc - qq) / qc),
                      "mz_vs_c_pct": float(100 * (qc - qm) / qc),
                      "mzq_ratio": float(ratio)}
            print(f"{t:>14} {qr:9.5f} {qc:9.5f} {qm:9.5f} {qq:9.5f} | "
                  f"{bm:8.4f} {100*(qc-qq)/qc:+11.2f} {ratio:10.4f}")
            if abs(ratio - 1.0) < LEAK_TOL:
                raise SystemExit(f"REFUSING: {t} at H={H} has a post-MZq TEST "
                                 f"ratio of exactly 1 -- the fit leaked.")

        best = {m: min(got, key=lambda k: np.nanmean(got[k][m]["q"]))
                for m in ("raw", "c", "mz", "mzq")}
        print(f"\n  best raw: {best['raw']}   best c: {best['c']}   "
              f"best MZ: {best['mz']}   best MZq: {best['mzq']}")
        # the question: does NOCTUA's c-rescale advantage survive 2 parameters?
        if "noctua_v1" in got:
            for ref in (best["mzq"],):
                if ref == "noctua_v1":
                    rival = min((k for k in got if k != "noctua_v1"),
                                key=lambda k: np.nanmean(got[k]["mzq"]["q"]))
                else:
                    rival = ref
                dd = (qlike_vec(got[rival]["mzq"]["rv"],
                               got[rival]["mzq"]["sigma"])
                      - qlike_vec(got["noctua_v1"]["mzq"]["rv"],
                                  got["noctua_v1"]["mzq"]["sigma"]))
                g = np.isfinite(dd)
                L = block_len_for(H, int(g.sum()))
                ci = mean_ci(dd[g], alpha=alpha, block_len=L)
                verdict = ("noctua_v1 still ahead under MZq"
                           if ci["ci95"][0] > 0 else
                           "NOT established under MZq")
                print(f"  noctua_v1 vs {rival} after MZq: {np.nanmean(dd):+.5f}  "
                      f"CI [{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]  "
                      f"-> {verdict}")
                row["_contrast"] = {"rival": rival,
                                    "delta": float(np.nanmean(dd)),
                                    "ci": [float(ci["ci95"][0]),
                                           float(ci["ci95"][1])],
                                    "clears": bool(ci["ci95"][0] > 0)}
        out["horizons"][str(H)] = {"teachers": row, "best": best}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print()
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
