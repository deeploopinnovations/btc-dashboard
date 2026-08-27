"""
eval/iv_correction.py
=====================================================================
Implied volatility as a RESIDUAL CORRECTION -- the only admissible design.

WHY NOT A FEATURE COLUMN

Deribit's DVOL index is the first candidate feature in this project that is
NOT a statistic of BTC's own past bars. Every existing input -- har_1h through
har_22d, the range and calendar terms -- describes what already happened.
DVOL is the price at which option sellers write insurance against what happens
next, which is the information class BENCHMARK.md 12's onset problem needs and
has never had.

It also cannot be used the obvious way. Measured, not assumed (ledger entry
`iv-coverage-2`, re-derived through `noctua.splits.time_splits`):

    split    episodes   with a causal IV observation at anchor - 1h
    train     189,831   62,051   32.7%
    calib      52,359   52,359  100.0%
    test       73,867   73,867  100.0%

DVOL begins 2021-03-24; the training window begins 2017-08-01. Two thirds of
training episodes have no implied vol and every calibration and test episode
has one. Adding `iv_level` as a column with a fill value would let the trainer
learn from the fill-versus-real distinction -- a signal with zero variance at
test time. That is not a data-quality problem to be patched; it is a property
of when the instrument started trading.

Two designs survive:

  (a) refit the whole model on the covered era only, paying ~67% of the
      training data for the feature;
  (b) leave NOCTUA exactly as it is and fit an IV-conditioned CORRECTION on
      top of its out-of-sample forecasts.

(b) is what this file does, and it is strictly safer: its failure mode is a
correction of zero. It also costs nothing to run. `eval/blend_ceiling.py`
caches each fold's out-of-sample components -- the recovered neural median,
the Log-HAR forecast, realized RV, H, the causal spike flag, and the episode
indices -- so the base forecast this correction sits on top of already exists
and no model is retrained here.

THE CORRECTION, AND WHY IT IS CONVEX

The shipped point forecast is a geometric blend in log space,

    log sigma_hat = w0 * neural_median + (1 - w0) * har_logvol,   w0 = 0.25

and the correction is a multiplicative adjustment to it:

    log sigma_corrected = log sigma_hat + z . beta

where `z` is the standardised IV feature vector with an intercept. Writing
r_hat = rv^2 / sigma_hat^2, QLIKE at the corrected forecast is

    QLIKE(beta) = r_hat * exp(-2 z.beta) + 2 z.beta - log r_hat - 1

whose Hessian is 4 z z' r_hat exp(-2 z.beta), positive semi-definite for every
beta. **The fit is globally convex**, so the optimiser cannot land in a local
minimum and "the fit did not converge" cannot masquerade as "IV does not
help". This is deliberate: BENCHMARK.md 20 records a GARCH baseline whose fit
silently never ran and returned its library's defaults in four of six folds,
which inflated NOCTUA's margin from 12.2% to 38%.

THREE ARMS, AND THE THIRD IS THE ONE THAT MATTERS

  RAW     beta from the convex fit on the history folds, used as-is.
  SHRUNK  the same beta pulled toward ZERO -- no correction -- in proportion
          to its fold-to-fold instability, by the identical estimator
          `blend_ceiling.shrink` applies to the blend weight, for the reasons
          in that file's citations (Smith & Wallis OBES 2009; Claeskens et al.
          IJF 2016; Blanc & Setzer, Management Science 2020). This is primary.
  PLACEBO the same fit on IV features circularly rotated a year within the
          covered era, destroying their alignment with the episode while
          preserving every marginal distribution, the autocorrelation
          everywhere but the wrap seam, the exact set of scored episodes, and
          every bit of the correction's flexibility.

The placebo is the arm that decides whether a gain means anything. A
free-parameter correction fitted by minimising the evaluation metric will
improve that metric somewhat on any input, and the only way to know how much
of an improvement is "IV" rather than "six more parameters" is to run the same
machinery on an input that cannot carry the signal. If the placebo moves
pooled QLIKE nearly as much as the real features, the real result is
flexibility, not information -- and that is the finding.

PRE-REGISTERED RULE, fixed before any score is read

  POPULATION: test episodes with a causal IV observation. Note the scale:
  `benchmark.run_fold` scores the PRODUCTION slice, so a fold is ~365 episodes
  (one 19-hour window per day) of which ~20 carry the spike flag. Seven
  coefficients fitted on at most five prior folds is a small-sample problem by
  construction, which is why the shrunk arm is primary and the placebo arm
  exists at all. Both arms are scored
  on the SAME episodes -- the uncorrected base is re-scored on the covered
  subset rather than quoted from the full-population number, because 6l's veto
  turned out to be entirely a difference in slice size (pitfalls check 4).

  PRIMARY: pooled QLIKE of the SHRUNK arm versus the uncorrected base, over
  folds with at least one prior fold to fit on, bootstrap CI must exclude zero
  on the favourable side.

  Pooled is primary because the correction changes EVERY covered episode; its
  target population is the pooled one. (Contrast 19, where the treatment aimed
  at 6% of episodes and the slice had to be primary.)

  GUARDS, all of which must hold:
    - PLACEBO: the shrunk real-feature gain must exceed the placebo's gain,
      and the difference must itself have a CI excluding zero. A real gain
      that a shifted series matches is not a gain.
    - spike-episode QLIKE must not worsen (own CI);
    - calm-episode QLIKE must not worsen by more than 1%;
    - the RAW beta signs must not oscillate across folds for any coefficient
      that the shrunk arm leaves materially non-zero -- shrinkage must not
      launder instability into a small, steady-looking number;
    - CANNOT ADOPT ALONE. This re-weights a recorded median and does not
      rebuild the barrier curves or the committee, so a favourable result
      earns a full-pipeline run, not a place in the artifact.

  REJECTION: if the shrunk arm's CI contains zero, or the placebo matches it,
  DVOL carries no incremental information over what the existing cascade
  already has -- the (a) hypothesis in `eval/ivfeatures.py`'s failure taxonomy,
  implied vol being persistence in a different costume.

    python -m model.eval.iv_correction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.blend_ceiling import W0, shrink                                 # noqa: E402
from eval.direction import ci_excludes_zero, mean_ci                      # noqa: E402
from eval.levers import qlike                                             # noqa: E402
from research import pitfalls as P                                        # noqa: E402

IV_COLS = ["iv_level", "iv_chg_1h", "iv_chg_6h", "iv_chg_24h",
           "iv_z_20d", "ivrv_ratio"]
SIGMA_B = 0.10          # scale of a "large" coefficient; fixed a priori
EXP_BOUND = 1.0         # |z . beta| bound: the correction may scale sigma by at
                        # most e^1 = 2.72x either way. A multiplicative
                        # adjustment larger than that is not a correction to a
                        # forecast, it is a different forecast.
SPIKE_POINT_BAR = 2.0   # percent; see the spike guard's note
PLACEBO_SHIFT_H = 365 * 24      # rotation distance, in covered-episode rows


def fit_beta(z: np.ndarray, r_hat: np.ndarray, iters: int = 60,
             tol: float = 1e-10) -> np.ndarray:
    """Newton on the convex QLIKE objective.

        f(b)  = mean[ r_hat * exp(-2 z.b) + 2 z.b ]
        g(b)  = 2 * mean[ z * (1 - r_hat * exp(-2 z.b)) ]
        H(b)  = 4 * mean[ z z' * r_hat * exp(-2 z.b) ]

    H is PSD everywhere, so this converges to the global minimum from any
    start. Convergence is ASSERTED, not hoped for: 20 records a baseline whose
    fit silently returned its library's defaults, and a correction that
    quietly failed to fit would be indistinguishable from one that found
    nothing.
    """
    b = np.zeros(z.shape[1])
    for _ in range(iters):
        e = np.exp(np.clip(-2.0 * (z @ b), -30, 30))
        w = r_hat * e
        g = 2.0 * (z * (1.0 - w)[:, None]).mean(axis=0)
        H = 4.0 * (z * w[:, None]).T @ z / len(z)
        H[np.diag_indices_from(H)] += 1e-8          # numerical floor only
        step = np.linalg.solve(H, g)
        b -= step
        if np.max(np.abs(step)) < tol:
            # The gradient and Hessian above assume exp(-2 z.b) is smooth. Where
            # the clip binds it is not -- its derivative there is zero, which
            # the formulas do not model, so a fit that "converged" against a
            # locally wrong gradient would pass the step test while being wrong.
            reach = float(np.max(np.abs(2.0 * (z @ b))))
            if reach > 25.0:
                raise RuntimeError(
                    f"Newton converged with |2 z.b| reaching {reach:.1f}, inside "
                    f"the clip's saturated region (bound 30). The gradient and "
                    f"Hessian are not valid there, so this fit is not the "
                    f"minimum it reports being.")
            return b
    raise RuntimeError(
        f"IV correction did not converge in {iters} Newton steps "
        f"(last step {np.max(np.abs(step)):.3e}); the objective is convex, so "
        f"this is a numerical fault, not an absent signal -- do not report the "
        f"unconverged fit as a null result")


def standardise(z: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                intercept: bool = True) -> np.ndarray:
    """Standardise with moments from the HISTORY folds, optionally with an
    intercept.

    E2b (pre-registered in the ledger before it ran) drops the intercept in BOTH
    arms. E2's decomposition showed a bare scalar -- no IV features at all --
    delivers 72.0% of the spike gain, with a fitted e^a stable in [1.139, 1.202]
    across five independently fitted folds. That is the point-forecast
    functional, not implied volatility, and the placebo collects it too because
    the placebo also has an intercept. Without the intercept the correction can
    only RESHAPE sigma, never rescale it, so what is left is incremental content
    or nothing.
    """
    zs = (z - mu) / np.maximum(sd, 1e-9)
    return np.column_stack([np.ones(len(zs)), zs]) if intercept else zs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IV as a residual correction")
    ap.add_argument("--components", type=Path,
                    default=Path("model/artifacts/blend_ceiling.npz"))
    ap.add_argument("--iv", type=Path,
                    default=Path("model/artifacts/iv_features.parquet"))
    ap.add_argument("--no-intercept", action="store_true",
                    help="E2b: hold beta_0 at zero in BOTH arms, so the "
                         "correction can only reshape sigma and never rescale "
                         "it. E2's 72%% intercept confound is then gone and the "
                         "IV features are tested on incremental content alone.")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/iv_correction.json"))
    a = ap.parse_args(argv)
    icept = not a.no_intercept
    if a.no_intercept and a.out == Path("model/artifacts/iv_correction.json"):
        a.out = Path("model/artifacts/iv_correction_nointercept.json")
    print(f"intercept: {'ON (E2)' if icept else 'OFF (E2b)'}\n")

    if not a.components.exists():
        raise SystemExit(
            f"REFUSING: {a.components} not found. Run "
            f"`python -m model.eval.blend_ceiling` first -- this file scores a "
            f"correction on top of ITS cached out-of-sample forecasts and never "
            f"retrains a model of its own.")
    npz = np.load(a.components)
    years = sorted({int(k.split("_")[-1]) for k in npz.files})
    iv = pd.read_parquet(a.iv)

    # The IV table is POSITIONALLY aligned with episodes.parquet. Verify that
    # against the anchor timestamps cached beside the forecasts rather than
    # trusting it -- a silent off-by-one row shift would join every episode to
    # a neighbouring hour's implied vol and still produce plausible numbers.
    iv_ts = iv["anchor_ts"].to_numpy(np.int64)
    Z_all = iv[IV_COLS].to_numpy(np.float64)

    folds = []
    for y in years:
        idx = npz[f"test_idx_{y}"]
        ats = npz[f"anchor_ts_{y}"]
        if not np.array_equal(iv_ts[idx], ats):
            raise SystemExit(
                f"REFUSING: fold {y}'s cached anchor timestamps do not match "
                f"iv_features.parquet at the same row indices. The positional "
                f"alignment this join relies on is broken.")
        z = Z_all[idx]
        cov = np.isfinite(z).all(axis=1)
        rv, raw, har, H = (npz[f"{k}_{y}"] for k in ("rv", "raw", "har", "H"))
        log_sig = W0 * raw + (1.0 - W0) * har
        sig = np.exp(log_sig) * np.sqrt(H)
        folds.append({"year": y, "z": z, "cov": cov, "rv": rv, "sig": sig,
                      "spike": npz[f"spike_{y}"].astype(bool), "ats": ats,
                      "n": int(len(rv)), "n_cov": int(cov.sum())})
        print(f"  {y}  n={len(rv):6d}  IV-covered {int(cov.sum()):6d} "
              f"({100*cov.mean():5.1f}%)  spike-covered "
              f"{int((npz[f'spike_{y}'].astype(bool) & cov).sum()):4d}")

    # PLACEBO: the same features CIRCULARLY ROTATED one year within the
    # DVOL-covered era.
    #
    # A plain backward shift was the first design and it is wrong here: DVOL
    # begins 2021-03 and the folds begin 2021, so reading a year earlier leaves
    # the early folds with no placebo data at all. The placebo would then be
    # scored on a smaller and later set of episodes than the real arm, and the
    # margin between them would be partly a difference in slice -- which is
    # exactly the defect that made 6l's veto uninterpretable (pitfalls check 4).
    #
    # A circular rotation inside the covered window maps every covered episode
    # to another covered episode, so coverage is IDENTICAL by construction and
    # the two arms are scored on the same rows. It preserves each feature's
    # marginal distribution exactly and its autocorrelation everywhere except
    # at the single wrap seam. What it destroys is the one thing under test:
    # alignment between the implied vol and the episode it is supposed to
    # inform.
    covered_rows = np.flatnonzero(np.isfinite(Z_all).all(axis=1))
    if len(covered_rows) == 0:
        raise SystemExit("REFUSING: no episode has a complete IV feature vector")
    pos_of_row = {int(r): i for i, r in enumerate(covered_rows)}
    if len(covered_rows) < 3 * PLACEBO_SHIFT_H:
        raise SystemExit(
            f"REFUSING: only {len(covered_rows):,} covered episodes for a "
            f"{PLACEBO_SHIFT_H:,}-row rotation. The modulo would wrap to a small "
            f"NEAR-NEIGHBOUR shift, which decorrelates nothing while still "
            f"passing the coverage-preservation check -- a placebo that is not a "
            f"placebo. Shorten PLACEBO_SHIFT_H deliberately or do not run this.")
    roll = PLACEBO_SHIFT_H % len(covered_rows)
    for f in folds:
        zp = np.full_like(f["z"], np.nan)
        idx = npz[f"test_idx_{f['year']}"]
        for j, r in enumerate(idx):
            i = pos_of_row.get(int(r))
            if i is not None:
                zp[j] = Z_all[covered_rows[(i + roll) % len(covered_rows)]]
        f["z_placebo"] = zp
        f["cov_placebo"] = np.isfinite(zp).all(axis=1)
        if not np.array_equal(f["cov_placebo"], f["cov"]):
            raise SystemExit(
                f"REFUSING: fold {f['year']} placebo coverage differs from the "
                f"real arm's ({int(f['cov_placebo'].sum())} vs "
                f"{int(f['cov'].sum())}). The rotation is meant to be "
                f"coverage-preserving; if it is not, the margin between the arms "
                f"is partly a difference in which episodes were scored.")

    def score(f, key_z, key_cov, beta, mu, sd):
        """Corrected and uncorrected QLIKE on the SAME episodes.

        The exponent is bounded. `beta` is fitted on history and applied to a
        later fold standardised with HISTORY's moments, so a regime shift can
        put `z . beta` far outside anything the fit saw -- and QLIKE is bounded
        below by 0 but unbounded above, so one extrapolated episode can dominate
        a fold's mean. The bias that introduces is one-directional (QLIKE's
        floor means a wild forecast can only make the delta worse, never
        better), so an unbounded exponent buys spurious REJECTs, not spurious
        wins. It is still a wrong number. EXCEEDANCES ARE COUNTED AND RETURNED
        rather than silently clipped, because a fold where the bound binds is a
        fold whose correction was extrapolating, which is worth knowing.
        """
        m = f[key_cov]
        zz = standardise(f[key_z][m], mu, sd, icept)
        raw_exp = zz @ beta
        n_clip = int((np.abs(raw_exp) > EXP_BOUND).sum())
        sig_c = f["sig"][m] * np.exp(np.clip(raw_exp, -EXP_BOUND, EXP_BOUND))
        if not np.all(np.isfinite(sig_c)):
            raise RuntimeError(
                "corrected sigma is not finite even after bounding the exponent")
        q_c = qlike(f["rv"][m], sig_c)
        q_b = qlike(f["rv"][m], f["sig"][m])
        assert len(q_c) == len(q_b) == int(m.sum())
        return q_c, q_b, f["spike"][m], n_clip

    print(f"\n{'year':>6} {'arm':>8} {'beta (intercept + 6)':>10}")
    dlt = {k: {"pooled": [], "spike": [], "calm": []}
           for k in ("raw", "shrunk", "placebo")}
    rows_out = []
    for i, f in enumerate(folds):
        hist = folds[:i]
        if not hist:
            print(f"{f['year']:>6}    (no prior fold -- excluded from the test)")
            continue
        # moments and fit from HISTORY only
        zh = np.vstack([h["z"][h["cov"]] for h in hist])
        mu, sd = zh.mean(axis=0), zh.std(axis=0)
        per_fold_beta = []
        for h in hist:
            m = h["cov"]
            zz = standardise(h["z"][m], mu, sd, icept)
            r_hat = np.maximum(h["rv"][m] ** 2, 1e-18) / np.maximum(h["sig"][m] ** 2, 1e-18)
            per_fold_beta.append(fit_beta(zz, r_hat))
        pooled_z = standardise(zh, mu, sd, icept)
        pooled_r = np.concatenate([
            np.maximum(h["rv"][h["cov"]] ** 2, 1e-18)
            / np.maximum(h["sig"][h["cov"]] ** 2, 1e-18) for h in hist])
        b_raw = fit_beta(pooled_z, pooled_r)
        B = np.array(per_fold_beta)                     # (m, p)
        shr = [shrink(B[:, j], w0=0.0, sigma_w=SIGMA_B) for j in range(B.shape[1])]
        b_shrunk = np.array([r[0] for r in shr])
        lam = np.array([r[1] for r in shr])

        # placebo fitted with its OWN moments and its own history slices
        zhp = np.vstack([h["z_placebo"][h["cov_placebo"]] for h in hist])
        mup, sdp = zhp.mean(axis=0), zhp.std(axis=0)
        rp = np.concatenate([
            np.maximum(h["rv"][h["cov_placebo"]] ** 2, 1e-18)
            / np.maximum(h["sig"][h["cov_placebo"]] ** 2, 1e-18) for h in hist])
        b_plac = fit_beta(standardise(zhp, mup, sdp, icept), rp)

        row = {"year": f["year"], "n_cov": f["n_cov"],
               "beta_raw": b_raw.tolist(), "beta_shrunk": b_shrunk.tolist(),
               "lam": lam.tolist(), "beta_placebo": b_plac.tolist(),
               "beta_per_fold": B.tolist()}
        for tag, (kz, kc, b, m_, s_) in {
                "raw":     ("z", "cov", b_raw, mu, sd),
                "shrunk":  ("z", "cov", b_shrunk, mu, sd),
                "placebo": ("z_placebo", "cov_placebo", b_plac, mup, sdp)}.items():
            q_c, q_b, sp, n_clip = score(f, kz, kc, b, m_, s_)
            dlt[tag]["pooled"].append(float(q_c.mean() - q_b.mean()))
            dlt[tag]["calm"].append(float(q_c[~sp].mean() - q_b[~sp].mean()))
            if sp.any():
                dlt[tag]["spike"].append(float(q_c[sp].mean() - q_b[sp].mean()))
            row[f"pooled_{tag}"] = float(q_c.mean())
            row[f"base_{tag}"] = float(q_b.mean())
            row[f"n_clipped_{tag}"] = n_clip
            row[f"n_scored_{tag}"] = int(len(q_c))
            # the calm-slice base, which the calm guard needs as its denominator
            row[f"base_calm_{tag}"] = float(q_b[~sp].mean())
            if sp.any():
                row[f"base_spike_{tag}"] = float(q_b[sp].mean())
        rows_out.append(row)
        print(f"{f['year']:>6}  raw     " +
              " ".join(f"{v:+.3f}" for v in b_raw))
        print(f"{'':>6}  shrunk  " +
              " ".join(f"{v:+.3f}" for v in b_shrunk) +
              "   lam " + " ".join(f"{v:.2f}" for v in lam))
        print(f"{'':>6}  pooled QLIKE base {row['base_raw']:.5f}  "
              f"raw {row['pooled_raw']:.5f}  shrunk {row['pooled_shrunk']:.5f}  "
              f"placebo {row['pooled_placebo']:.5f}")

    out = {"folds": rows_out, "iv_cols": IV_COLS, "sigma_b": SIGMA_B,
           "intercept": icept,
           "placebo_shift_hours": PLACEBO_SHIFT_H, "w0": W0}
    if len(dlt["shrunk"]["pooled"]) < 2:
        print("\nfewer than 2 scored folds -- no verdict"); return 1

    print(f"\n{'arm':>10} {'quantity':>8} {'delta':>11} {'95% CI':>26} {'signs':>8}")
    for tag in ("raw", "shrunk", "placebo"):
        out[tag] = {}
        for nm in ("pooled", "spike", "calm"):
            arr = np.asarray(dlt[tag][nm], np.float64)
            if len(arr) < 2:
                out[tag][nm] = {"delta": float(arr.mean()) if len(arr) else None,
                                "ci95": None, "n": int(len(arr)),
                                "note": "fewer than 2 folds carried this slice"}
                print(f"{tag:>10} {nm:>8}   only {len(arr)} fold(s) -- no interval")
                continue
            ci = mean_ci(arr, seed=37)
            # `ci["mean"]` is the mean of the SAME filtered array the interval
            # came from; `arr.mean()` would report a nan beside a finite CI.
            out[tag][nm] = {"delta": ci["mean"], "ci95": ci["ci95"],
                            "n_negative": ci["n_negative"],
                            "n_positive": ci["n_positive"]}
            print(f"{tag:>10} {nm:>8} {arr.mean():+11.5f}   "
                  f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]   "
                  f"{ci['n_negative']}-/{ci['n_positive']}+")

    # placebo margin: real minus placebo, fold by fold
    margin = (np.asarray(dlt["shrunk"]["pooled"]) -
              np.asarray(dlt["placebo"]["pooled"]))
    ci_m = mean_ci(margin, seed=41)
    out["placebo_margin"] = {"delta": float(margin.mean()), "ci95": ci_m["ci95"]}
    print(f"\n  placebo margin (shrunk - placebo, pooled): {margin.mean():+.5f}   "
          f"[{ci_m['ci95'][0]:+.5f}, {ci_m['ci95'][1]:+.5f}]")

    # The calm guard divides by the CALM base, not the pooled one. The first
    # version used `base_raw`, which is pooled over calm AND spike episodes --
    # and spike QLIKE is roughly ten times calm QLIKE, so the pooled base is
    # several times the calm base and the guard was silently several times too
    # lenient. A calm degradation of 1.2% would have been reported as 0.78% and
    # passed.
    base_calm = float(np.mean([r["base_calm_shrunk"] for r in rows_out]))
    calm_pct = 100.0 * np.mean(dlt["shrunk"]["calm"]) / max(base_calm, 1e-12)
    base_spike = float(np.mean([r["base_spike_shrunk"] for r in rows_out
                                if "base_spike_shrunk" in r]))
    spike_pct = 100.0 * np.mean(dlt["shrunk"]["spike"]) / max(base_spike, 1e-12)

    primary = ci_excludes_zero(out["shrunk"]["pooled"]["ci95"], -1)
    guard_placebo = ci_excludes_zero(ci_m["ci95"], -1)

    # THE SPIKE GUARD IS TWO CONDITIONS, AND THE LOOSER CI BAR IS DELIBERATE.
    # A fold carries ~20 spike episodes, so the spike CI is very wide and a bar
    # of "the interval must not touch zero on the bad side" would fail almost
    # regardless of the data -- an unsatisfiable condition, which pitfalls
    # check 9 exists to catch. The CI bar is therefore the loose one (fail only
    # if the interval lies ENTIRELY above zero), and it is paired with a POINT
    # bar so the guard still has teeth: the spike delta may not exceed 2% of
    # the spike baseline even when the interval is inconclusive.
    guard_spike_ci = not ci_excludes_zero(out["shrunk"]["spike"]["ci95"], +1)
    guard_spike_pt = spike_pct <= SPIKE_POINT_BAR
    guard_spike = guard_spike_ci and guard_spike_pt
    guard_calm = calm_pct <= 1.0

    # (1) SIGN STABILITY of the RAW coefficients, which the rule requires and
    # the first version documented without implementing. A coefficient that is
    # +0.30 in one fold and -0.25 in the next is not a signal; shrinkage would
    # turn that pair into a small steady-looking number, which is exactly what
    # the rule forbids. Only coefficients the shrunk arm leaves materially
    # non-zero are tested -- one shrunk to ~0 is already saying "no signal".
    MATERIAL = 0.01
    unstable = []
    n_coef = len(rows_out[0]["beta_raw"])
    for j in range(n_coef):
        if max(abs(r["beta_shrunk"][j]) for r in rows_out) < MATERIAL:
            continue
        signs = {np.sign(r["beta_raw"][j]) for r in rows_out
                 if abs(r["beta_raw"][j]) >= MATERIAL}
        if {-1.0, 1.0} <= signs:
            unstable.append(("intercept" if (icept and j == 0)
                             else IV_COLS[j - 1 if icept else j]))
    guard_stable = not unstable

    ok = (primary and guard_placebo and guard_spike and guard_calm
          and guard_stable)
    out["verdict"] = "ADVANCE" if ok else "REJECT"
    print(f"\n--- pre-registered rule ---")
    print(f"  PRIMARY shrunk pooled QLIKE CI excludes zero favourably : {primary}")
    print(f"  GUARD   beats the placebo, CI excluding zero            : {guard_placebo}")
    print(f"  GUARD   spike QLIKE not worse  (CI {guard_spike_ci}, "
          f"point {spike_pct:+.2f}% vs {SPIKE_POINT_BAR:.0f}% bar) : {guard_spike}")
    print(f"  GUARD   calm QLIKE within 1%  ({calm_pct:+.2f}% of calm base) : {guard_calm}")
    print(f"  GUARD   RAW coefficient signs stable across folds       : {guard_stable}"
          + (f"   unstable: {unstable}" if unstable else ""))
    n_clip_tot = sum(r["n_clipped_shrunk"] for r in rows_out)
    if n_clip_tot:
        print(f"  NOTE    the correction exponent hit its +/-{EXP_BOUND} bound at "
              f"{n_clip_tot} episodes -- those folds were extrapolating")
    print(f"  -> {out['verdict']}"
          + ("  (NOT an adoption: the barrier curves were not rebuilt)" if ok else ""))

    rep = P.Report()
    rep.add(P.check_ci_is_defined(out["shrunk"]["pooled"]["ci95"], "shrunk pooled"))
    rep.add(P.check_not_a_coin_flip(dlt["shrunk"]["pooled"], "shrunk pooled delta"))
    # These two counts are derived from the arms SEPARATELY -- `n_scored_shrunk`
    # is len(q_c) and `n_cov` is the coverage mask's own popcount. The first
    # version passed the same expression twice, which is a check that cannot
    # fail: the mirror image of pitfalls check 7's guard that cannot pass.
    rep.add(P.check_arms_matched(
        {"covered": sum(r["n_cov"] for r in rows_out),
         "scored": sum(r["n_scored_shrunk"] for r in rows_out)},
        what="scored episodes"))
    rep.add(P.check_rule_satisfiable(2, len(rows_out), "folds"))
    print("\n--- research/pitfalls on this experiment ---")
    print(rep.render())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
