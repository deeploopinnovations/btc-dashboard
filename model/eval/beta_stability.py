"""
eval/beta_stability.py
=====================================================================
P3-beta-is-affine / P3-beta-noise: is NOCTUA's MZ slope defect an INFORMATION
defect, an affine miscalibration, or an estimator that cannot be fitted?

THE QUESTION THIS EXISTS TO SETTLE

`P3-functional-audited` left one substantive defect after the mean functional
was adopted: the Mincer-Zarnowitz slope beta, measured on the raw forecast, is
1.303 / 1.236 / 1.116 / 0.915 at H = 1 / 6 / 24 / 168. beta > 1 says the
forecast UNDER-REACTS to its own signal; beta < 1 that it over-reacts. The
written plan named a CRPS-trained variant as the remedy.

That plan assumed the defect is in the network. There are three candidates and
they imply completely different work:

  (a) INFORMATION      the forecast's dynamics are wrong       -> retrain
  (b) AFFINE           a stable two-parameter map removes it   -> recalibrate
  (c) UNESTIMABLE      the correction itself will not fit      -> more windows

(a) and (b) are separated by fitting MZq on each fold's CALIB slice, applying
it to that fold's TEST slice, and RE-MEASURING the slope on rows the correction
never saw. Re-fitting on test and reporting beta = 1.000 is arithmetic, not
evidence -- MZq sets the slope to 1 by construction, so an in-sample version of
this test confirms itself whatever the data says. The out-of-fold version can
fail, and at one horizon it does.

(b) and (c) are separated by asking whether the per-fold slopes the correction
fitted are consistent with a single common value plus estimation noise. That is
what `--se` measures, with a moving-block bootstrap at 2H, because consecutive
episodes are anchored hourly against an H-hour forward window and share H-1 of
their H hours: at H = 168, 8,760 rows carry on the order of 52 independent
windows, and an estimator handed 52 observations is entitled to scatter.

WHAT IT FOUND

Short horizons are (b) and the long horizon is (c). Neither is (a), so no part
of this result argues for a retrain.

    python -m model.eval.beta_stability --selftest
    python -m model.eval.beta_stability
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mz_recalibration import fit_mz, fit_mzq, gather, optimal_c   # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                  # noqa: E402

TEACHER = "noctua_v1_mean"
NEUTRAL = 0.03          # |beta - 1| within this is called neutral


def beta_se(rv, sig, H: int, n_rep: int = 400, seed: int = 0) -> float:
    """Moving-block bootstrap standard error of the fitted MZ slope.

    Block length 2H, not n^(1/3). The dependence range here is set by the
    forward window, not by the sample size: neighbouring episodes share H-1 of
    their H hours however many of them there are. n^(1/3) at H=168 would be 37
    against an overlap of 168 and would report a standard error roughly a
    factor of two too small -- which is the difference between calling the
    cross-fold scatter noise and calling it regime variation.

    IMPORTANT, AND NOT OBVIOUS: block length changes a SLOPE's standard error
    only when the regression RESIDUAL is serially dependent, not merely when the
    regressor is. With independent errors the slope's variance comes from the
    errors and blocking them buys nothing -- measured here at a ratio of 0.79
    from block 4 to block 300, i.e. slightly the wrong way. With an overlapped
    residual the ratio is 4.49. So the 2H choice is only justified if the MZ
    residual on the real table is itself autocorrelated, which `residual_acf`
    below measures rather than assumes.
    """
    rv = np.asarray(rv, np.float64); sig = np.asarray(sig, np.float64)
    ok = np.isfinite(rv) & np.isfinite(sig) & (sig > 0) & (rv > 0)
    rv, sig = rv[ok], sig[ok]
    n = len(rv)
    L = min(max(2 * H, 2), n)
    if n < 3 * L:
        return float("nan")
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for i in range(n_rep):
        st = rng.integers(0, n - L + 1, size=nb)
        idx = (st[:, None] + np.arange(L)[None, :]).reshape(-1)[:n]
        p = fit_mz(rv[idx], sig[idx])
        out[i] = np.nan if p is None else p[1]
    return float(np.nanstd(out, ddof=1))


def residual_acf(rv, sig, lags) -> dict:
    """Autocorrelation of the MZ regression RESIDUAL, which is what decides
    whether the block length matters at all.

    `beta_se` blocks at 2H on the argument that consecutive episodes share H-1
    of their H hours. That argument is about the DATA's overlap; what the
    bootstrap actually needs is dependence in the residual of
    log(realised) on log(forecast). The two are not the same claim: a forecast
    that tracked the shared component perfectly would leave a white residual
    however much its windows overlap, and then 2H would only inflate the
    interval. So measure it. `decay_lag` is the first lag at which |acf| falls
    under 0.1 and stays there; if that is on the order of H, the 2H block is
    justified by the data rather than by the argument.
    """
    rv = np.asarray(rv, np.float64); sig = np.asarray(sig, np.float64)
    ok = np.isfinite(rv) & np.isfinite(sig) & (sig > 0) & (rv > 0)
    x, y = np.log(sig[ok]), np.log(rv[ok])
    if len(x) < 100:
        return {"acf": {}, "decay_lag": None}
    b = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    r = y - (float(np.mean(y) - b * np.mean(x)) + b * x)
    r = r - r.mean()
    v = float(r @ r)
    acf = {int(k): (float(r[:-k] @ r[k:] / v) if 0 < k < len(r) else float("nan"))
           for k in lags}
    decay = next((k for k in sorted(acf) if abs(acf[k]) < 0.1), None)
    return {"acf": acf, "decay_lag": decay}


def oof_row(z, H: int) -> dict | None:
    """beta measured on test rows, raw and after each correction fitted on calib."""
    row = {"H": H}
    raw = gather(z, H, TEACHER, "raw")
    if raw is None:
        return None
    row["n_test"] = int(np.isfinite(raw["sigma"]).sum())
    row["n_independent"] = int(row["n_test"] // max(H, 1))

    # The CALIB slice is what the correction is fitted on, so its independent
    # count -- not the test one -- is the number that decides whether a
    # two-parameter correction can be estimated at all. Episodes are anchored
    # hourly against an H-hour forward window, so n/H is the order of the
    # independent-window count. At H=168 this is 24 PER FOLD.
    nc = [len(z[f"{y}/{H}/calib/rv"]) for y in raw["years"]
          if f"{y}/{H}/calib/rv" in z]
    row["n_calib_per_fold"] = int(np.median(nc)) if nc else 0
    row["n_calib_independent_per_fold"] = int(row["n_calib_per_fold"] // max(H, 1))
    row["beta_raw"] = fit_mz(raw["rv"], raw["sigma"])[1]

    # Per-fold slopes measured on the SAME rows the pooled number uses, so the
    # two are comparable. The pooled beta is a mixture across folds; if the
    # folds disagree, the mixture is not a slope anybody's forecast has.
    per = []
    for y in raw["years"]:
        kt = f"{y}/{H}/test"
        if f"{kt}/sigma/{TEACHER}" not in z:
            continue
        p = fit_mz(np.asarray(z[f"{kt}/rv"], np.float64),
                   np.asarray(z[f"{kt}/sigma/{TEACHER}"], np.float64))
        if p is not None:
            per.append(float(p[1]))
    row["beta_raw_per_fold"] = per
    row["beta_raw_median_fold"] = float(np.median(per)) if per else float("nan")
    for mode in ("c", "mz", "mzq"):
        g = gather(z, H, TEACHER, mode)
        if g is None:
            row[f"beta_{mode}"] = float("nan"); continue
        row[f"beta_{mode}"] = fit_mz(g["rv"], g["sigma"])[1]
        if mode == "mzq":
            row["fold_betas"] = [float(p[1]) for p in g["params"]]
            row["fold_years"] = list(g["years"])
    return row


def verdict_of(row: dict) -> str:
    b = row["beta_mzq"]
    if not np.isfinite(b):
        return "NO CORRECTION"
    moved = abs(row["beta_raw"] - 1.0) - abs(b - 1.0)
    if abs(b - 1.0) <= NEUTRAL:
        return "AFFINE (correction transfers)"
    if moved <= 0.02:
        return "UNESTIMABLE (correction does not transfer)"
    return "PARTIAL"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MZ slope: affine, or unestimable?")
    ap.add_argument("--oof", type=Path, default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/beta_stability.json"))
    ap.add_argument("--se", action="store_true",
                    help="block-bootstrap a standard error for each fold's slope "
                         "(slow; this is what separates noise from regime)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    rows = [r for H in HORIZONS if (r := oof_row(z, H)) is not None]

    print(f"MZ slope for {TEACHER}, measured on TEST rows only")
    print("corrections fitted on each fold's calib slice and applied to its test\n")
    print(f"{'H':>5} {'calib indep/fold':>17} {'raw':>8} {'+c':>8} {'+MZ':>8} "
          f"{'+MZq':>8}   reading")
    for r in rows:
        print(f"{r['H']:>5} {r['n_calib_independent_per_fold']:>17,} "
              f"{r['beta_raw']:>8.3f} {r['beta_c']:>8.3f} {r['beta_mz']:>8.3f} "
              f"{r['beta_mzq']:>8.3f}   {verdict_of(r)}")
    print("\n  Read the first column against the last. Whether a two-parameter")
    print("  correction transfers out of fold tracks the number of INDEPENDENT")
    print("  windows it was fitted on, monotonically, and nothing else here does.")
    print("\n  '+c' does NOT equal 'raw', and the reason matters. A single level")
    print("  rescale cannot move a slope -- log(c*sigma) shifts log sigma by a")
    print("  constant and leaves cov/var untouched. But c is fitted PER FOLD, so")
    print("  each fold's forecasts shift by a DIFFERENT constant, and the slope")
    print("  of the POOLED scatter moves even though no single fold's slope does.")
    print("  This column was first shipped with a comment asserting the equality")
    print("  that its own numbers refute (1.334 vs 1.303 at H=1). The per-fold")
    print("  slopes below are the honest unit; every pooled beta here is a")
    print("  mixture across six differently-levelled folds.")

    print(f"\nthe pooled RAW beta against the per-fold raw betas on the same rows:")
    print(f"{'H':>5} {'pooled':>8} {'median fold':>12}   per-fold")
    for r in rows:
        pf = r.get("beta_raw_per_fold") or []
        print(f"{r['H']:>5} {r['beta_raw']:>8.3f} {r['beta_raw_median_fold']:>12.3f}"
              f"   {[round(b, 3) for b in pf]}")
    print("  Where these two disagree, the pooled number is an artifact of")
    print("  stacking folds at different levels, not a property of the forecast.")

    print(f"\nper-fold slopes the MZq correction FITTED on calib:")
    for r in rows:
        fb = r.get("fold_betas") or []
        if not fb:
            continue
        spread = max(fb) - min(fb)
        neg = [y for y, b in zip(r["fold_years"], fb) if b < 0]
        print(f"  H={r['H']:>3}  {[round(b, 3) for b in fb]}")
        print(f"          spread {spread:.3f}   sd {np.std(fb, ddof=1):.3f}"
              + (f"   NEGATIVE SLOPE in {neg}" if neg else ""))

    if a.se:
        print("\nis the cross-fold scatter estimation noise, or regime variation?")
        print("moving-block bootstrap at 2H on each fold's CALIB slice\n")
        print("first, does the block length even matter -- is the MZ residual")
        print("serially dependent on the real table?\n")
        print(f"{'H':>5} {'acf(1)':>8} {'acf(H/2)':>9} {'acf(H)':>8} {'acf(2H)':>9} "
              f"{'decay<0.1':>10}  block 2H justified?")
        for r in rows:
            H = r["H"]
            g = gather(z, H, TEACHER, "raw")
            lags = sorted({1, max(1, H // 2), H, 2 * H})
            d = residual_acf(g["rv"], g["sigma"], lags)
            r["residual_acf"] = d
            a = d["acf"]
            f = lambda k: f"{a[k]:8.3f}" if k in a else "       -"
            dl = d["decay_lag"]
            just = "YES" if (dl is None or dl >= H) else f"NO -- decays by lag {dl}"
            print(f"{H:>5} {f(1)} {f(max(1, H // 2)):>9} {f(H)} {f(2 * H):>9} "
                  f"{str(dl):>10}  {just}")
        print()
        print(f"{'H':>5} {'sd across folds':>16} {'mean within-fold SE':>20}  reading")
        for r in rows:
            H = r["H"]
            ses = []
            for y in r.get("fold_years", []):
                kc = f"{y}/{H}/calib"
                if f"{kc}/sigma/{TEACHER}" not in z:
                    continue
                ses.append(beta_se(np.asarray(z[f"{kc}/rv"], np.float64),
                                   np.asarray(z[f"{kc}/sigma/{TEACHER}"], np.float64),
                                   H, seed=H))
            if not ses:
                continue
            sd = float(np.std(r["fold_betas"], ddof=1))
            se = float(np.nanmean(ses))
            r["sd_across_folds"], r["mean_within_fold_se"] = sd, se
            read = ("NOISE -- scatter is what one fold's own SE predicts"
                    if np.isfinite(se) and sd <= 1.5 * se else
                    "REGIME -- scatter exceeds estimation error")
            print(f"{H:>5} {sd:>16.3f} {se:>20.3f}  {read}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"teacher": TEACHER, "oof": str(a.oof), "neutral_band": NEUTRAL,
         "rows": [{k: v for k, v in r.items()} for r in rows],
         "verdicts": {str(r["H"]): verdict_of(r) for r in rows}},
        indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(11)
    n = 6000

    # 1-2. A forecast wrong by a PURE AFFINE map in log-vol: fit_mz must recover
    #      the map, and applying it must return a slope of 1.
    true = np.exp(rng.normal(-4.0, 0.5, n))
    rv = true * np.exp(rng.normal(0, 0.15, n))
    sig = np.exp(0.3 + np.log(true) / 1.4)            # alpha=0.3, beta=1/1.4
    al, be = fit_mz(rv, sig)
    ok.append(("fit_mz recovers a planted slope", abs(be - 1.4) < 0.06))
    ok.append(("corrected slope is 1",
               abs(fit_mz(rv, np.exp(al) * sig ** be)[1] - 1.0) < 1e-6))

    # 3. A PERFECT forecast needs no correction: the slope is already 1. This is
    #    the check that caught a ddof mismatch between np.cov and np.var.
    ok.append(("perfect forecast has slope 1",
               abs(fit_mz(rv, true)[1] - 1.0) < 0.05))

    # 4. A level rescale CANNOT move a slope. verdict_of leans on this.
    # A SINGLE level rescale leaves a slope alone. Note the scope: this does
    # NOT extend to the pooled slope across folds that each got their own c,
    # where differing shifts do move it -- the report prints that distinction
    # because an earlier version of it asserted the stronger, false claim.
    ok.append(("one level rescale leaves the slope alone",
               abs(fit_mz(rv, optimal_c(rv, sig) * sig)[1] - be) < 1e-9))
    c1, c2 = 0.7, 1.9
    mixed = np.concatenate([c1 * sig[: n // 2], c2 * sig[n // 2:]])
    ok.append(("but two different level shifts DO move the pooled slope",
               abs(fit_mz(rv, mixed)[1] - be) > 0.01))

    # 5-8. Block length matters for a SLOPE only when the RESIDUAL is serially
    #      dependent -- not merely when the regressor is. Two earlier drafts of
    #      this check got that wrong. The first asserted a wider SE on the IID
    #      fixture above, where a longer block absorbs nothing. The second built
    #      a dependent REGRESSOR with IID errors and measured the SE going the
    #      wrong way (ratio 0.79): for OLS with independent errors the slope's
    #      variance comes from the errors, so blocking them buys nothing, and a
    #      long block merely narrows the effective x-range per resample. Only
    #      when the residual itself overlaps does the block do its job (ratio
    #      4.49). Both directions are asserted here, because a test that only
    #      checks the favourable one would pass on a bootstrap that ignored its
    #      block argument entirely.
    def smooth(x, w):
        return np.convolve(x, np.ones(w) / w, mode="valid")[:n]

    W = 120
    dep = smooth(rng.normal(0, 1.0, n + W), W)
    sig_d = np.exp(-4.0 + dep * 3.0 / 1.4)
    rv_iid = np.exp(-4.0 + dep * 3.0) * np.exp(rng.normal(0, 0.30, n))
    rv_ovl = np.exp(-4.0 + dep * 3.0) * np.exp(
        smooth(rng.normal(0, 0.30 * np.sqrt(W), n + W), W))
    se_i2, se_i300 = (beta_se(rv_iid, sig_d, 2, n_rep=250, seed=3),
                      beta_se(rv_iid, sig_d, 150, n_rep=250, seed=3))
    se_o2, se_o300 = (beta_se(rv_ovl, sig_d, 2, n_rep=250, seed=3),
                      beta_se(rv_ovl, sig_d, 150, n_rep=250, seed=3))
    ok.append(("block SE is finite and positive",
               all(np.isfinite(v) and v > 0 for v in (se_i2, se_o2, se_o300))))
    ok.append(("dependent residual: long block widens the SE", se_o300 > 2 * se_o2))
    ok.append(("iid residual: long block does NOT widen the SE", se_i300 < 1.2 * se_i2))
    ok.append(("the bootstrap actually reads its block argument",
               abs(se_o300 - se_o2) > 0.01))

    # 9-11. residual_acf must distinguish the two cases above, because the whole
    #       justification for the 2H block rests on it. A white residual must
    #       decay at lag 1; an overlapped one must stay correlated out to roughly
    #       its window; and too little data must return None rather than a
    #       number, since a decay lag invented from 50 points would silently
    #       justify whatever block was already chosen.
    acf_i = residual_acf(rv_iid, sig_d, [1, 10, W])
    acf_o = residual_acf(rv_ovl, sig_d, [1, 10, W, 2 * W])
    ok.append(("white residual decays at lag 1", acf_i["decay_lag"] == 1))
    ok.append(("overlapped residual stays correlated to ~its window",
               acf_o["decay_lag"] is None or acf_o["decay_lag"] >= W))
    ok.append(("too little data returns no decay lag",
               residual_acf(rv_iid[:50], sig_d[:50], [1])["decay_lag"] is None))

    ok.append(("refuses when n < 3 blocks",
               not np.isfinite(beta_se(rv[:100], sig[:100], 60, n_rep=10))))

    # 8-9. verdict_of must call a transferring correction AFFINE and a
    #      non-transferring one UNESTIMABLE -- including when raw is already
    #      near 1, where "did not move" must not be read as success.
    ok.append(("transfers -> AFFINE",
               verdict_of({"beta_raw": 1.30, "beta_mzq": 1.006})
               == "AFFINE (correction transfers)"))
    ok.append(("does not transfer -> UNESTIMABLE",
               verdict_of({"beta_raw": 0.915, "beta_mzq": 0.916})
               == "UNESTIMABLE (correction does not transfer)"))

    # 10. A missing correction is not a passing one.
    ok.append(("nan correction -> NO CORRECTION",
               verdict_of({"beta_raw": 1.2, "beta_mzq": float("nan")})
               == "NO CORRECTION"))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
