"""
eval/blend_ceiling.py
=====================================================================
Is the ENSEMBLE WEIGHT the binding constraint, and how much is on the table?

WHERE THIS COMES FROM

19 found that `BASE_COLS` -- the input set of the Log-HAR anchor holding 75% of
the blended forecast -- has `har_1d` as its fastest input, so the dominant term
cannot respond to anything faster than a day. `eval/anchor_freshness.py` tested
the obvious fix (put `har_1h`/`har_6h` into the anchor) and 19 named, in
advance, what a negative result would mean:

    "The neural stage ALREADY sees har_1h/har_6h. If the blend, committee and
    calibration are already extracting what those features carry, adding them
    to the anchor buys nothing and the lag has a different cause. That would
    redirect the work away from feature placement and toward the blend weight
    itself -- which is the next thing to test, not a dead end."

This is that next thing. The premise is mechanical, not speculative:

  * Log-HAR is slow but robust. Pure NOCTUA lost +72.3% in the 2023 volatility
    collapse; the blend bounded that fold at +6.7% (the note on `infer.BLEND_W`).
  * The neural stage is the only term that sees below daily resolution.
  * `BLEND_W = 0.25` is a CONSTANT, chosen by a walk-forward sweep over
    constants (`model/artifacts/blend_sweep.json`).

A single constant that is simultaneously the best compromise for a calm night
and for a spike onset is exactly the thing worth doubting. Nobody has tested a
state-dependent weight.

WHY THIS COSTS ALMOST NOTHING TO MEASURE

The blend is affine in log space. `infer.predict` shifts the whole predictive
distribution so its median lands on the ensemble level:

    qa_med = qa_med_raw + (1 - w) * (har_logvol - qa_med_raw)
           = w * qa_med_raw + (1 - w) * har_logvol

Seed-averaging `qa` is linear, so the raw neural median inverts EXACTLY from
the recorded post-shift median:

    qa_med_raw = (qa_med - (1 - w0) * har_logvol) / w0

and any other weight is one exponential away. So the entire w-surface -- every
constant weight, every state-dependent rule -- comes out of ONE benchmark run
instead of one 6-fold retrain per value of w. `benchmark.run_fold` was extended
(additively) to record `qa_med`, `har_logvol`, `blend_w` and `H` for exactly
this.

THREE QUANTITIES, AND ONLY ONE OF THEM IS A RESULT

  1. CONSTANT-w SWEEP. Honest, but already known to land near 0.25.

  2. THE ORACLE CEILING -- and there are TWO of them, which is the whole point.

     2a. PER-EPISODE oracle: the w that minimises QLIKE for each episode
         separately, with the realized RV in hand. This is DEGENERATE and is
         reported only to be dismissed. It fits one free parameter per
         observation, so it drives sigma toward |rv| episode by episode and
         its "gain" is almost entirely the overfit, not headroom. Quoting it
         as headroom would repeat the withdrawn "92% is shape" figure exactly.

     2b. TWO-STATE in-sample oracle: ONE w for spike episodes and ONE for calm,
         fitted on the same fold being scored. Two free parameters against a
         fold's ~365 test episodes -- `benchmark.run_fold` scores the
         PRODUCTION slice, one 19-hour window opened at 17:00 UTC per day, so a
         year is ~365 episodes and ~20 of them carry the spike flag, not the
         ~500k of the full population. THIS is the meaningful upper bound,
         because it is the best any two-state rule could do even with perfect
         foresight about which two numbers to pick -- so the causal rule in
         (3), which must pick them from prior folds, cannot beat it.

         Two parameters against 365 observations is comfortable; the SPIKE
         weight, fitted on ~20, is not, and that is precisely why the
         shrinkage below is indexed to fold-to-fold instability rather than
         adopted raw.

     WHICH OBJECT THIS SCORES. `benchmark.run_fold` averages `sigma_med` over
  seeds ARITHMETICALLY and `qa` LOGARITHMICALLY, and exp(mean(log)) is not
  mean(exp). The sweep is computed on the log-space object, which makes the
  w-algebra exact; the arithmetic gap at w = 0.25 is measured per fold and
  reported (`seed_avg_gap`), so the sweep's w = 0.25 point is known to be a
  hair away from the benchmark's headline `noctua` QLIKE rather than assumed
  identical. `check_recovery` tests the AM-GM direction of that gap, which is
  a signed prediction the explanation could fail.

  AMENDED BEFORE ANY SCORE WAS READ. The first version of this rule made the
     2% floor apply to 2a. That was wrong on its face: 2a's number is an
     artifact of parameter count and would have passed a 2% floor no matter
     what the data said, which makes it a rule that cannot fail -- the same
     defect research/pitfalls.py check 9 exists to catch, arriving from the
     opposite direction. The floor now applies to 2b. No fold had been run when
     this was changed; the amendment is recorded here rather than made silently.

  3. THE CAUSAL, WALK-FORWARD RULE. Two weights, one for spike-risk episodes
     and one for calm, where the split uses `causal_spike_flag` -- information
     available at the anchor. Fitted on folds STRICTLY BEFORE the scored fold
     and applied out of sample. This is the only number that can be adopted,
     and it is estimated TWICE:

     3a. RAW ARGMIN -- the grid minimiser on the history folds, adopted as-is.
     3b. SHRUNK -- the same argmin pulled back toward the shipped constant
         w0 = 0.25 in proportion to its own fold-to-fold instability.

     3b is the pre-registered primary and 3a is reported beside it, because
     the literature says 3a should disappoint and the pair makes that visible
     instead of assumed.

WHY THE RAW ARGMIN IS THE WRONG ESTIMATOR, WITH CITATIONS

The forecast-combination puzzle (named by Stock & Watson 2004) is the finding
that estimated "optimal" combination weights routinely lose out of sample to a
simple fixed weight. Smith & Wallis (OBES 2009) and Claeskens, Magnus, Vasnev &
Wang (IJF 2016) give the mechanism: once the weight is ESTIMATED rather than
known, the combination is biased even when its inputs are not, and its variance
strictly exceeds the fixed-weight case, so nothing guarantees it beats the
constant. Genre, Kenny, Meyler & Timmermann (IJF 2013) show that even with
decades of panel data most schemes fail to beat a simple average once
multiple-comparison bias is corrected -- which is exactly what picking w by
grid-search argmin over a handful of folds is.

The sharpest warning is specific to this pipeline. Clements & Vasnev,
"Forecast combination puzzle in the HAR model" (J. Forecasting 43(1), 2024),
show the HAR model IS a three-way forecast combination -- daily, weekly,
monthly -- and that its OLS-optimal weights are a textbook instance of the
puzzle: simple averages of HAR components beat the fitted weights across
equities, commodities, FX and indices at every horizon. NOCTUA's anchor is
that model. That is a published reason to distrust any further weight fitted
on top of it, and it is also a second, independent reading of BENCHMARK.md 21:
adding two more regressors to the HAR anchor raised its estimation variance in
precisely the term the puzzle literature says to shrink.

There is one countervailing result, and it is why this experiment is worth
running rather than abandoning. Elliott & Timmermann, "Optimal forecast
combination under general loss functions and forecast error distributions"
(J. Econometrics 122(1), 2004), prove that the puzzle's conclusions "can be
overturned when asymmetries are introduced in the loss function and the
forecast error distribution is skewed", and characterise how the optimal
weights move. QLIKE is asymmetric and spike-period errors are right-skewed, so
the MSE-derived pessimism does not transfer unexamined. (What could NOT be
verified: whether that paper derives a closed form for QLIKE specifically --
its abstract says "the most commonly used alternatives to mean squared error
loss" without naming it, and QLIKE as used in the Patton volatility literature
is roughly contemporaneous. So this is a reason to test, not a prediction.)

The shrinkage design follows Blanc & Setzer, "Bias-Variance Trade-Off and
Shrinkage of Weights in Forecast Combination" (Management Science 66(12),
2020), which places equal/fixed weights at the zero-variance end and fitted
weights at the zero-bias end and shrinks between them. The expectation that
the SPIKE weight ends up shrunk harder than the calm one follows Liu, Hao &
Wang, "Solving the Forecast Combination Puzzle Using Double Shrinkages"
(OBES 86(3), 2024), who find shrinkage toward equal weights dominates
specifically in turbulent and recession regimes -- the regime with the fewest
and noisiest observations, which is exactly the spike state here.

No source was found stating a sample-size threshold at which an estimated
weight starts to beat a constant, and none is assumed. Nor was any paper found
testing a state-dependent HAR-versus-ML combination weight conditioned on a
jump or spike flag under QLIKE; "not found" is not "does not exist", but it
does mean this design is not a replication of a settled result.

PRE-REGISTERED RULE, fixed before the scores are read

  PRIMARY: pooled test QLIKE of the walk-forward two-state SHRUNK rule (3b)
  versus the shipped constant w = 0.25, over the folds that have at least one
  prior fold to fit on, with a bootstrap CI that must exclude zero on the
  favourable side. The raw argmin (3a) is reported beside it and is NOT the
  primary -- it is the estimator the puzzle literature predicts will lose, and
  it is carried so that prediction is tested rather than trusted.

  THE SHRINKAGE, fixed in full before any score is read. For each state,

      w_shrunk = w0 + lam * (mean_f w_argmin[f] - w0),        w0 = 0.25
      lam      = m / (m + (sd_f w_argmin[f] / SIGMA_W) ** 2)
      SIGMA_W  = 0.25,  m = number of history folds,  lam = 0.5 when m < 2

  Read it as: trust the fitted deviation in proportion to how stable it was
  across the folds that produced it. A weight that lands in the same place
  every year survives nearly intact; one that jumps around is pulled back to
  the constant. SIGMA_W = 0.25 is the scale of a "large" deviation from w0 and
  is set a priori, not tuned. No quantity in this formula is fitted on the
  scored fold, and none is fitted on test data at all.

  Nothing here shrinks the spike weight harder BY CONSTRUCTION -- the formula
  is identical for both states. It is expected to happen anyway, because spike
  episodes are ~6% of the data and their argmin is correspondingly noisier, so
  their sd is larger and their lam smaller. Whether it actually does is
  reported per fold, and if the spike weight turns out to be the STABLER of
  the two, that is a finding against the Liu et al. reading, not a bug.

  Pooled is primary HERE, unlike 19 -- and the reason is the 13 lesson read
  correctly rather than cargo-culted. 13's rule failed because a treatment
  aimed at 6% of episodes was judged on a pooled mean it could not move. This
  treatment changes the weight on EVERY episode, calm ones included: its
  target population IS the pooled one.

  GUARDS, all of which must hold:
    - spike-episode QLIKE must not worsen at all (own CI);
    - calm-episode QLIKE must not worsen by more than 1% (own CI);
    - the RAW argmin weights must be reported per fold, and if they oscillate
      in sign of deviation from 0.25 across folds the result is NULL
      regardless of the pooled CI -- an unstable parameter is not a lever,
      and the shrinkage must not be allowed to launder that instability into
      a small, apparently-stable number;
    - deep-tail barrier MCB is NOT scored here, because this script re-weights
      a recorded median and does not rebuild the barrier curves. Adoption
      therefore requires a follow-up full-pipeline run; this experiment can
      REJECT on its own but cannot ADOPT on its own.

  REJECTION: if the TWO-STATE in-sample ceiling (2b) is under 2% of pooled
  QLIKE, the lever is declared not worth building and (3) is not interpreted --
  there is nothing for a causal rule to recover.

    python -m model.eval.blend_ceiling
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import benchmark as B                                           # noqa: E402
from eval.direction import ci_excludes_zero, mean_ci                      # noqa: E402
from eval.levers import causal_spike_flag, qlike                          # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402

W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
W0 = 0.25          # the shipped constant, and the shrinkage target
SIGMA_W = 0.25     # scale of a "large" deviation from W0; fixed a priori
LAM_NO_HISTORY = 0.5   # lam when a single history fold gives no sd to read


def shrink(w_per_fold, w0: float = W0, sigma_w: float = SIGMA_W) -> tuple:
    """Pull a fitted weight back toward `w0` in proportion to its instability.

        lam = m / (m + (sd / sigma_w)**2)

    Blanc & Setzer (Management Science 2020) put fixed weights at the
    zero-variance end and fitted weights at the zero-bias end; this is a
    shrinkage between them whose only input is how much the fitted weight moved
    across the folds that produced it. Every constant is set a priori. Returns
    (w_shrunk, lam, mean, sd) so the adjustment is auditable rather than
    buried."""
    a = np.asarray(w_per_fold, np.float64)
    m = len(a)
    mean = float(a.mean())
    if m < 2:
        return (float(w0 + LAM_NO_HISTORY * (mean - w0)), LAM_NO_HISTORY,
                mean, float("nan"))
    sd = float(a.std(ddof=1))
    lam = m / (m + (sd / sigma_w) ** 2)
    return float(w0 + lam * (mean - w0)), float(lam), mean, sd


def snap(w: float) -> float:
    """Nearest grid weight -- the per-fold QLIKE sums are tabulated on W_GRID,
    so a shrunk weight is scored at the grid point it rounds to. The grid step
    is 0.05, which bounds the rounding error on w at 0.025."""
    return float(W_GRID[int(np.argmin(np.abs(W_GRID - w)))])


def sigma_at(w: float, raw: np.ndarray, har: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Window vol at ensemble weight `w`, from the recovered components."""
    return np.exp(w * raw + (1.0 - w) * har) * np.sqrt(H)


def recover_raw(pe: dict) -> np.ndarray:
    """Invert the affine blend to get the seed-averaged RAW neural log-vol rate."""
    w0 = float(pe["blend_w"])
    if w0 <= 0.0:
        raise ValueError("blend_w = 0 leaves no neural component to recover")
    return (pe["qa_med"] - (1.0 - w0) * pe["har_logvol"]) / w0


# The seed-averaging gap, measured rather than assumed.
#
# The first run of this file asserted that inverting the blend and re-applying
# it reproduces the recorded `sigma_med` to 1e-6, and it did not: max relative
# error 5.4e-4. The identity is not wrong -- the two quantities average over
# SEEDS differently.
#
#   `sigma_med`  is np.mean over seeds of exp(qa_med_s) * sqrt(H)  -- ARITHMETIC
#   `qa`         is np.mean over seeds of qa_s, so exp(qa_bar)     -- GEOMETRIC
#
# and exp(mean(log)) != mean(exp). That gives a SIGNED prediction, which is
# what makes this an explanation rather than a rationalisation: by AM-GM the
# geometric mean is never larger, so `sigma_geo <= sigma_med` must hold at
# every single episode, with equality only where the seeds agree exactly. The
# check below tests that direction and fails if it is ever violated.
#
# The w-sweep is therefore computed on the log-space (geometric) object, which
# is exactly self-consistent, and the arithmetic gap at w0 is reported so the
# reader knows the sweep's w = 0.25 point is not bit-identical to the
# benchmark's headline `noctua` QLIKE.
AM_GM_TOL = 5e-3


def check_recovery(pe: dict) -> dict:
    """Two separate checks, because they answer different questions.

    EXACT: does the affine inversion round-trip through the log-space object it
    was derived from? This must hold to floating-point precision -- if it does
    not, the algebra is wrong and every number downstream is fiction.

    SIGNED: how far is the log-space object from the recorded `sigma_med`, and
    is the gap in the direction AM-GM requires? A gap in the wrong direction
    would mean the seed-averaging story is not the explanation.
    """
    w0 = float(pe["blend_w"])
    raw = recover_raw(pe)
    geo = sigma_at(w0, raw, pe["har_logvol"], pe["H"])
    exact = np.exp(pe["qa_med"]) * np.sqrt(pe["H"])
    err = float(np.max(np.abs(geo - exact) / np.maximum(exact, 1e-12)))
    if not err < 1e-9:
        raise AssertionError(
            f"blend inversion does not round-trip: max relative error {err:.3e}. "
            f"exp(w*qa_raw + (1-w)*har)*sqrt(H) must reproduce exp(qa_med)*sqrt(H) "
            f"exactly; it does not, so the algebra is wrong and the whole "
            f"w-surface would be fiction.")

    rel = (pe["sigma_med"] - geo) / np.maximum(pe["sigma_med"], 1e-12)
    if rel.min() < -1e-9:
        raise AssertionError(
            f"seed-averaging gap runs the WRONG WAY at {int((rel < -1e-9).sum())} "
            f"episodes (min {rel.min():.3e}). AM-GM makes the geometric seed mean "
            f"no larger than the arithmetic one, so the 'sigma_med averages in "
            f"sigma space, qa averages in log space' explanation is refuted and "
            f"the real cause is something else -- do not proceed on it.")
    gap = float(rel.max())
    if gap > AM_GM_TOL:
        raise AssertionError(
            f"seed-averaging gap {gap:.3e} exceeds {AM_GM_TOL:.0e}; the seeds "
            f"disagree far more than a 3-seed committee should, which is a "
            f"finding in its own right and not something to average over.")
    return {"inversion_err": err, "seed_gap": gap}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is the ensemble weight the constraint?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/blend_ceiling.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    spike = causal_spike_flag(ep)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    raw_har = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(m):
        lo, hi = np.quantile(raw_har[m], [0.005, 0.995])
        return np.maximum(np.clip(raw_har, lo, hi), 1e-12)

    print(f"folds {len(folds)}  seeds {a.seeds}  "
          f"spike {spike.sum():,}/{len(ep):,} ({100*spike.mean():.2f}%)\n")

    per_fold = []
    for f in folds:
        t0 = time.time()
        r = B.run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds, sigma_ref_fn=sig_fn)
        if r is None:
            print(f"  {f['year']}  SKIPPED"); continue
        pe = r["per_episode"]
        for k in ("qa_med", "har_logvol", "H"):
            if k not in pe:
                raise SystemExit(
                    f"REFUSING: run_fold did not record '{k}'. This script needs "
                    f"the blend components; re-run against a benchmark.py that "
                    f"records them.")
        chk = check_recovery(pe)
        err = chk["inversion_err"]
        # `test_idx` and `anchor_ts` ride along so a LATER experiment can join
        # anything episode-aligned -- implied vol, funding, a regime label --
        # onto these same out-of-sample forecasts without retraining. Omitting
        # them would make every such experiment pay the 6-fold retrain again.
        rec = {"year": f["year"], "recovery_err": err,
               "seed_gap": chk["seed_gap"],
               "rv": pe["rv"], "raw": recover_raw(pe), "har": pe["har_logvol"],
               "H": pe["H"], "spike": spike[pe["test_idx"]],
               "test_idx": pe["test_idx"],
               "anchor_ts": ep["anchor_ts"].to_numpy(np.int64)[pe["test_idx"]],
               "n": int(len(pe["rv"]))}
        per_fold.append(rec)
        print(f"  {f['year']}  n={rec['n']:6d}  spike={int(rec['spike'].sum()):5d}  "
              f"inversion {err:.1e}  seed-avg gap {chk['seed_gap']:.1e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not per_fold:
        print("no fold produced results"); return 1

    # Cache the recovered components. Everything below is a reduction of these
    # four arrays per fold, so any FUTURE weighting rule -- a third state, a
    # continuous w(x), an IV-conditioned one -- can be scored against the same
    # forecasts without retraining a single model. The 6-fold retrain is the
    # only expensive step in this file and it should be paid once.
    npz = a.out.with_suffix(".npz")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **{
        f"{k}_{r['year']}": r[k] for r in per_fold
        for k in ("rv", "raw", "har", "H", "spike", "test_idx", "anchor_ts")})
    print(f"\ncached components -> {npz}")

    def fold_q(rec, w_calm, w_spike):
        w = np.where(rec["spike"], w_spike, w_calm)
        return qlike(rec["rv"], sigma_at(w, rec["raw"], rec["har"], rec["H"]))

    # QLIKE at every grid weight, once per fold. Every quantity below is a
    # different reduction of this one array, and the two-state grid search in
    # (3) is 441 combinations per history fold -- recomputing QLIKE inside that
    # loop would be ~1e9 pointless exponentials.
    for rec in per_fold:
        rec["Q"] = np.stack([qlike(rec["rv"], sigma_at(w, rec["raw"], rec["har"],
                                                       rec["H"])) for w in W_GRID])
        sp = rec["spike"]
        rec["Qsum_spike"] = rec["Q"][:, sp].sum(axis=1)      # (len(W),)
        rec["Qsum_calm"] = rec["Q"][:, ~sp].sum(axis=1)

    # ---- 1. constant-w sweep ------------------------------------------------
    print("\n--- constant w, mean over folds (w=0.25 is what ships) ---")
    print(f"{'w':>6} {'pooled':>9} {'spike':>9} {'calm':>9}")
    sweep = {}
    for i, w in enumerate(W_GRID):
        pl = [rec["Q"][i].mean() for rec in per_fold]
        sk = [rec["Q"][i][rec["spike"]].mean() for rec in per_fold if rec["spike"].any()]
        cm = [rec["Q"][i][~rec["spike"]].mean() for rec in per_fold]
        sweep[float(w)] = {"pooled": float(np.mean(pl)), "spike": float(np.mean(sk)),
                           "calm": float(np.mean(cm))}
        mark = "  <- shipped" if abs(w - 0.25) < 1e-9 else ""
        print(f"{w:6.2f} {sweep[float(w)]['pooled']:9.4f} "
              f"{sweep[float(w)]['spike']:9.4f} {sweep[float(w)]['calm']:9.4f}{mark}")

    best_const = min(sweep, key=lambda w: sweep[w]["pooled"])
    ship = sweep[0.25]
    print(f"\nbest constant w = {best_const:.2f}  pooled {sweep[best_const]['pooled']:.4f} "
          f"vs shipped {ship['pooled']:.4f}  "
          f"({100*(sweep[best_const]['pooled']-ship['pooled'])/ship['pooled']:+.2f}%)")

    # ---- 2. the oracle ceiling -- USES THE FUTURE, NOT ACHIEVABLE -----------
    frac_ep, frac_2s, two_state_w = [], [], []
    orac_w_by_slice = {"spike": [], "calm": []}
    j25 = int(np.argmin(np.abs(W_GRID - 0.25)))
    for rec in per_fold:
        Q = rec["Q"]
        base = float(Q[j25].mean())
        # 2a -- degenerate, one parameter per observation
        wstar = W_GRID[np.argmin(Q, axis=0)]
        orac_w_by_slice["spike"].append(float(np.mean(wstar[rec["spike"]]))
                                        if rec["spike"].any() else np.nan)
        orac_w_by_slice["calm"].append(float(np.mean(wstar[~rec["spike"]])))
        frac_ep.append(float((base - float(Q.min(axis=0).mean())) / base))
        # 2b -- two parameters, fitted on this fold: the meaningful bound
        surf = (rec["Qsum_calm"][:, None] + rec["Qsum_spike"][None, :]) / rec["n"]
        ic, isp = np.unravel_index(int(np.argmin(surf)), surf.shape)
        frac_2s.append(float((base - float(surf[ic, isp])) / base))
        two_state_w.append({"year": rec["year"], "w_calm": float(W_GRID[ic]),
                            "w_spike": float(W_GRID[isp]),
                            "gain_pct": 100 * frac_2s[-1]})
    ceiling_ep = float(np.mean(frac_ep))
    ceiling = float(np.mean(frac_2s))          # the pre-registered quantity
    print(f"\n--- ORACLE CEILINGS (both use realized RV; NEITHER is achievable) ---")
    print(f"  2a per-episode oracle: {100*ceiling_ep:.2f}%  "
          f"-- DEGENERATE, one parameter per observation, reported to be dismissed")
    print(f"  2b two-state oracle  : {100*ceiling:.2f}%  <- the pre-registered ceiling")
    for t in two_state_w:
        print(f"       {t['year']}  w_calm {t['w_calm']:.2f}  w_spike {t['w_spike']:.2f}"
              f"  gain {t['gain_pct']:+.2f}%")
    print(f"  mean per-episode oracle w: spike {np.nanmean(orac_w_by_slice['spike']):.3f}, "
          f"calm {np.nanmean(orac_w_by_slice['calm']):.3f}")

    # ---- 3. the causal walk-forward two-state rule --------------------------
    # Fit (w_calm, w_spike) on folds strictly EARLIER than the scored fold. The
    # first fold has no history and is excluded rather than given a peek.
    print(f"\n--- CAUSAL walk-forward two-state rule (fit on prior folds only) ---")
    print("  RAW = grid argmin on the history folds.  SHRUNK = the same argmin "
          "pulled toward 0.25\n  by lam = m/(m + (sd/0.25)^2).  SHRUNK is the "
          "pre-registered primary.\n")
    print(f"{'year':>6} {'raw_c':>6} {'raw_s':>6} {'lam_c':>6} {'lam_s':>6} "
          f"{'shr_c':>6} {'shr_s':>6} {'raw%':>7} {'shrunk%':>8}")
    dlt = {"raw": {"pooled": [], "spike": [], "calm": []},
           "shrunk": {"pooled": [], "spike": [], "calm": []}}
    fitted = []
    # each fold's own argmin, computed once and reused as history downstream
    for rec in per_fold:
        sf = (rec["Qsum_calm"][:, None] + rec["Qsum_spike"][None, :]) / rec["n"]
        a_, b_ = np.unravel_index(int(np.argmin(sf)), sf.shape)
        rec["w_argmin"] = (float(W_GRID[a_]), float(W_GRID[b_]))

    for i, rec in enumerate(per_fold):
        hist = per_fold[:i]
        if not hist:
            print(f"{rec['year']:>6}    (no prior fold -- excluded from the test)")
            continue
        # RAW: argmin of the pooled history surface.
        surf = np.zeros((len(W_GRID), len(W_GRID)))
        for h in hist:
            surf += (h["Qsum_calm"][:, None] + h["Qsum_spike"][None, :]) / h["n"]
        surf /= len(hist)
        ic, isp = np.unravel_index(int(np.argmin(surf)), surf.shape)
        raw = (float(W_GRID[ic]), float(W_GRID[isp]))
        # SHRUNK: read the per-fold argmins for a dispersion, then shrink.
        wc_hist = [h["w_argmin"][0] for h in hist]
        ws_hist = [h["w_argmin"][1] for h in hist]
        sc, lam_c, _, sdc = shrink(wc_hist)
        ss, lam_s, _, sds = shrink(ws_hist)
        shr = (snap(sc), snap(ss))

        q_base = fold_q(rec, W0, W0)
        row = {"year": rec["year"], "w_calm_raw": raw[0], "w_spike_raw": raw[1],
               "w_calm_shrunk": shr[0], "w_spike_shrunk": shr[1],
               "lam_calm": lam_c, "lam_spike": lam_s,
               "sd_calm": sdc, "sd_spike": sds,
               "hist_argmin_calm": wc_hist, "hist_argmin_spike": ws_hist,
               "pooled_base": float(q_base.mean())}
        for tag, w in (("raw", raw), ("shrunk", shr)):
            q = fold_q(rec, *w)
            dlt[tag]["pooled"].append(float(q.mean() - q_base.mean()))
            dlt[tag]["calm"].append(float(q[~rec["spike"]].mean()
                                          - q_base[~rec["spike"]].mean()))
            if rec["spike"].any():
                dlt[tag]["spike"].append(float(q[rec["spike"]].mean()
                                               - q_base[rec["spike"]].mean()))
            row[f"pooled_{tag}"] = float(q.mean())
        fitted.append(row)
        print(f"{rec['year']:>6} {raw[0]:6.2f} {raw[1]:6.2f} {lam_c:6.2f} "
              f"{lam_s:6.2f} {shr[0]:6.2f} {shr[1]:6.2f} "
              f"{100*dlt['raw']['pooled'][-1]/row['pooled_base']:+7.2f} "
              f"{100*dlt['shrunk']['pooled'][-1]/row['pooled_base']:+8.2f}")

    d_pool = dlt["shrunk"]["pooled"]
    d_spike = dlt["shrunk"]["spike"]
    d_calm = dlt["shrunk"]["calm"]

    out = {"sweep": sweep, "best_constant_w": float(best_const),
           "oracle_ceiling_frac": ceiling,
           "oracle_ceiling_frac_per_episode_DEGENERATE": ceiling_ep,
           "oracle_two_state_w": two_state_w,
           "oracle_w_spike": float(np.nanmean(orac_w_by_slice["spike"])),
           "oracle_w_calm": float(np.nanmean(orac_w_by_slice["calm"])),
           "fitted_weights": fitted,
           "recovery_err": [r["recovery_err"] for r in per_fold],
           "seed_avg_gap": [r["seed_gap"] for r in per_fold],
           "n_folds_scored": len(d_pool)}

    out["raw_deltas"] = {}
    if len(dlt["raw"]["pooled"]) >= 2:
        print(f"\n--- RAW argmin arm (reported, NOT the primary) ---")
        for nm in ("pooled", "spike", "calm"):
            arr = np.asarray(dlt["raw"][nm], np.float64)
            ci = mean_ci(arr, seed=31)
            out["raw_deltas"][nm] = {"delta": float(arr.mean()), "ci95": ci["ci95"],
                                     "n_negative": ci["n_negative"],
                                     "n_positive": ci["n_positive"]}
            print(f"{nm + ' QLIKE':>16} {arr.mean():+11.5f}   "
                  f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]   "
                  f"{ci['n_negative']}-/{ci['n_positive']}+")

    if len(d_pool) >= 2:
        print(f"\n--- SHRUNK arm (the pre-registered primary) ---")
        print(f"{'quantity':>16} {'delta':>11} {'95% CI':>26}")
        for nm, dd in (("pooled QLIKE", d_pool), ("spike QLIKE", d_spike),
                       ("calm QLIKE", d_calm)):
            arr = np.asarray(dd, np.float64)
            # mean_ci, not block_bootstrap_ci: the unit is a FOLD and that
            # helper returns (nan, nan) below n = 20, which a comparison would
            # silently read as "did not clear the bar" (pitfalls check 11).
            ci = mean_ci(arr, seed=29)
            lo, hi = ci["ci95"]
            out[nm.split()[0] + "_delta"] = {"delta": float(arr.mean()), "ci95": [lo, hi],
                                             "ci95_iid": ci["ci95_iid"],
                                             "n_negative": ci["n_negative"],
                                             "n_positive": ci["n_positive"]}
            print(f"{nm:>16} {arr.mean():+11.5f}   [{lo:+.5f}, {hi:+.5f}]"
                  f"   iid [{ci['ci95_iid'][0]:+.5f}, {ci['ci95_iid'][1]:+.5f}]"
                  f"   {ci['n_negative']}-/{ci['n_positive']}+")

        # ---- the pre-registered rule, applied verbatim ----------------------
        signs = {np.sign(f["w_calm_raw"] - W0) for f in fitted} | \
                {np.sign(f["w_spike_raw"] - W0) for f in fitted}
        stable = not ({-1.0, 1.0} <= signs)
        lo_p, hi_p = out["pooled_delta"]["ci95"]
        lo_s, hi_s = out["spike_delta"]["ci95"]
        calm_pct = 100 * np.mean(d_calm) / sweep[0.25]["calm"]
        if ceiling < 0.02:
            verdict = "REJECT"
            why = (f"two-state oracle ceiling {100*ceiling:.2f}% is under the "
                   f"pre-registered 2% floor: even a rule that knew the two best "
                   f"weights in advance could not recover enough to matter")
        elif not ci_excludes_zero(out["pooled_delta"]["ci95"], -1):
            verdict = "NULL"
            why = f"pooled CI [{lo_p:+.5f}, {hi_p:+.5f}] does not clear zero favourably"
        elif hi_s > 0:
            verdict = "NULL"
            why = f"guard: spike QLIKE worsens, CI [{lo_s:+.5f}, {hi_s:+.5f}]"
        elif calm_pct > 1.0:
            verdict = "NULL"
            why = f"guard: calm QLIKE worsens {calm_pct:+.2f}%, over the 1% bar"
        elif not stable:
            verdict = "NULL"
            why = ("guard: the RAW argmin weights straddle 0.25 in both "
                   "directions across folds -- an unstable parameter is not a "
                   "lever, and shrinkage must not launder that into a small, "
                   "apparently-stable number")
        else:
            verdict = "ADVANCE"
            why = ("clears the pooled CI and every guard; NOT an adoption -- the "
                   "barrier curves were not rebuilt, so a full-pipeline run is "
                   "required before this ships")
        out["verdict"], out["why"] = verdict, why
        print(f"\nPRE-REGISTERED VERDICT: {verdict}\n  {why}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
