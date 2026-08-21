"""
eval/shape.py
=====================================================================
Is path SHAPE predictable at all -- and if it is, does it move a number
anyone would act on?

DECISION RULE, FIXED BEFORE THIS RUN (ROADMAP.md, Priority 1)

Shape counts as predictable-and-useful only if, for some arm on the
PRODUCTION test slice:

    (a) SIGNIFICANCE.  Out-of-sample skill (R^2 against the climatology
        constant) clears the p95 of a shuffled-target permutation null.
        Clearing zero is not evidence; clearing the null is.
    (b) RELEVANCE.     That skill, pushed back through the Gaussian
        first-passage law fed REALIZED volatility, moves the implied 2%
        touch probability by an average of >= 1 percentage point versus
        using the unconditional (climatology) shape for every episode.

Both must hold. This rule was written into ROADMAP.md before this file
existed and is not re-derived or loosened here -- see the note at the end of
this docstring on why (b) exists and what it has already cost one finding in
this project.

WHY THIS EXPERIMENT, AND WHY NOW

`eval/firstpassage.py` fed the Gaussian first-passage law the REALIZED
volatility -- a perfect oracle -- and found that doing so removes only
6.0-8.5% of the barrier error across five barrier widths. The other
91.5-94% is SHAPE: given moves of the right size, how far the path actually
travels before settlement, which the Gaussian law gets wrong by assuming a
Brownian relationship between range and volatility that BTC does not obey
(range/RV = 1.3311 measured vs. 1.5831 Brownian-sampled, gap -0.2519, CI
[-0.2851, -0.2212] -- see firstpassage.py and ROADMAP.md section 1).

The obvious response -- give the existing network path-shape features and
see if it uses them -- was already tried in `eval/efficiency.py` and came
back a clean null: DSC/UNC moved from 0.04980 to 0.05000, 2 wins of 6,
t-like +0.22. Indistinguishable from noise.

Those two facts together are the reason this file exists. A large,
systematic error (fact 1) that more features to the SAME architecture do not
touch (fact 2) has exactly two honest explanations, and nothing in the
repository yet tells them apart:

    (i)  Shape's conditional variation is not predictable at this horizon --
         it is the crypto analogue of the efficient-market result
         `eval/direction.py` already established for sign. If so, 92% of the
         barrier error is a FLOOR, not a bug, and every remaining hour
         belongs to volatility and regime work (ROADMAP items 2-4).
    (ii) Shape IS predictable and NOCTUA's architecture cannot express it --
         a shared trunk dominated by volatility terms, a coupling penalty
         that ties the excursion heads to a return head with zero
         directional skill, or simply capacity spent in the wrong place. If
         so, this is the largest available gain in the project by an order
         of magnitude over anything measured so far.

An in-sample model score cannot tell these apart, because in-sample fitting
manufactures apparent skill from noise -- exactly the failure `direction.py`
retracted once already (BENCHMARK.md section 5, "Kronos carries conditional
information"). What is needed is an ATTAINABILITY test: give the SAME causal
feature set the field already trusts to the strongest tractable estimators
(a linear ceiling and a nonlinear one), on the SAME splits, against the SAME
climatology adversary, and ask whether ANY of them clears a shuffled null.
That is exactly the method `direction.py` used to settle the sign question,
and this file copies it column for column, changing only the loss (squared
error / R^2 in place of Brier / DSC, because the target here is continuous)
and the null construction it implies.

THE TARGETS, AND WHY THEY ARE EXACTLY NOCTUA'S OWN STAGE-B TARGETS

    y_range = (M_up + |M_dn|) / RV      the travel ratio
    y_max   = max(M_up, |M_dn|) / RV    the seller's version, "what breaks a
                                        strike"

`RV` in this codebase already IS a volatility (`episodes.py`: "RV = sqrt(sum
of hourly realized variances)"), so no further sqrt belongs in the
denominator -- dividing by RV once reproduces exactly the
"range / sqrt(realized variance)" statistic firstpassage.py and ROADMAP.md
report (1.3311 on the production slice). Both targets are scale-free by
construction: dividing by RV removes the volatility level, so a model that
scores here has learned something about PATH SHAPE and nothing a volatility
forecast could have supplied for free -- which is the entire point, since
volatility skill leaking into a "shape" result would misattribute the gain.

These are not new quantities invented for this file. `noctua.train.prepare`,
called with its default `sigma_ref=None` (so sigma = RV, the realized value,
not a forecast), already computes

    m_up = M_up / RV,   m_dn = -M_dn / RV,   m_mx = max(m_up, m_dn)

as NOCTUA's own Stage-B fitting targets. `y_range` here is exactly
`m_up + m_dn`, and `y_max` is exactly `m_mx`. Reusing `prepare()` rather than
recomputing the ratio by hand means this file's targets, feature
standardization, and train/test slicing are the SAME code path the shipped
model uses -- not a parallel implementation that could quietly drift from
it. The one deliberate difference from how NOCTUA itself is trained: this
file passes no `sigma_ref`, i.e. uses the REALIZED RV as the scale, not a
causal forecast. That is correct here and would be wrong for training the
network: `prepare()`'s docstring explains at length why NOCTUA must be
fitted against the sigma it will actually SERVE with (`exp(har_1d)*sqrt(H)`)
to avoid a train/serve skew. This file is not training anything that will
serve. It is asking the oracle question firstpassage.py asked -- "given the
realized volatility, is the leftover shape error structured or is it noise"
-- so the realized RV is not a leak here, it is the point.

THE ARMS

    climatology   the train-split weighted mean of the target -- a constant,
                  input-blind, and the required adversary. If nothing beats
                  it, shape is unpredictable and that is the answer.
    ridge         RidgeCV over the FULL causal feature set (every column in
                  features.parquet except the three EFFICIENCY_COLS, which
                  NON_MODEL_COLS keeps out of the shipped model and are kept
                  out here too, for the same reason: they were tested and
                  rejected in efficiency.py, and letting them back in through
                  a side door would not be honest). Standardized on TRAIN
                  only (`prepare()`'s `Standardizer`). Alpha is chosen by
                  5-fold CV WITHIN the training split (ordinary KFold, not
                  time-ordered -- a minor simplification that could very
                  slightly flatter ridge's cross-validated fit, but the TEST
                  period is never touched by it). This is the linear ceiling.
    gbdt          HistGradientBoostingRegressor, same feature set, same
                  hyperparameters `direction.py` uses for its own nonlinear
                  ceiling (`max_depth=3, max_iter=200, learning_rate=0.05,
                  l2_regularization=1.0, min_samples_leaf=200`) --
                  deliberately shallow and regularized, because an
                  unconstrained GBDT on hundreds of thousands of overlapping
                  episodes will memorize the sample and say nothing about the
                  population. This is the nonlinear ceiling.

The optional fourth arm -- NOCTUA's own implied shape -- is SKIPPED. A
trained checkpoint exists (`artifacts/noctua.pt`), so extracting its
Stage-B quantiles would be cheap, but the checkpoint was fitted on the
single fixed split in `noctua.splits` (train through 2023-01, calibrate
through 2024-07), not on this file's walk-forward folds. Scoring it against
years 2021-2024 would be scoring it on data it trained on; restricting to
2025-2026 only would silently change the test population relative to every
other arm in this file and reopen exactly the stale-fit problem ROADMAP.md
Priority 2 is about. Given the instruction to skip this arm if it is not
cheap to do RIGHT, and that doing it right means a fifth thing to get exactly
correct under a deadline, it is left out. It remains a natural follow-up if
ridge or gbdt clears the decision rule below and a captured-vs-not-captured
question becomes worth asking.

THE NULL, AND WHY IT IS BUILT THE WAY `direction.py` BUILDS ITS OWN

`direction.py`'s null does NOT retrain per replicate. It fits the model
ONCE, then permutes the FITTED forecast against the outcome and recomputes
the CORP decomposition's isotonic recalibration on each shuffle -- because
the thing that manufactures spurious skill there is the in-sample isotonic
fit, which finds a monotone staircase in any finite sample regardless of
whether the forecast carries real information.

The analogous manufacturing risk here is different in mechanism but the same
in spirit: R^2 against a constant is 1 - SSE(pred)/SSE(clim), and SSE(clim)
against a PERMUTED target is IDENTICAL to SSE(clim) against the real one --
sum of squared deviations from a constant does not care about order. So all
of the sampling variability in the null comes from how well THIS SPECIFIC,
ALREADY-FITTED prediction vector happens to pair with a randomly reordered
target, which is exactly the question "would this many apparently-skillful
pairings appear even with no true correspondence between prediction and
outcome." Refitting the model on each shuffle would answer a different and
strictly harder question (can this estimator ever fit pure noise at all,
which for a regularized ridge or a heavily-constrained GBDT it provably
cannot on average) and would cost 200x the training compute for no gain in
what the null is meant to rule out. So: fit once per fold, then randomly
re-pair the ALREADY-FITTED prediction against the target within the test
period >= 200 times, recompute R^2 and Spearman each time, and report the
p95. Concretely this is done by permuting the PREDICTION rather than the
target -- exactly what `direction.py`'s `shuffled_dsc_null` does
(`corp_decomposition(rng.permutation(p), y)`) -- which is the identical
random pairing either way but has one extra convenience here: the
climatology reference this file's R^2 is measured against is itself an
ARRAY that varies by fold once folds are pooled (each fold's climatology is
its own train-split mean), and permuting the prediction leaves both the
target AND that reference untouched at every index, so the reference term
of R^2 is exactly as fixed under the null as it is in direction.py's DSC
null. Permuting the target instead would work out to the same distribution
in expectation but would needlessly re-couple the target to a DIFFERENT
episode's climatology on every replicate. This is `direction.py`'s
`shuffled_dsc_null` translated to a continuous target, not a different idea.

Episodes at nearby anchors overlap (a 19-hour window shares up to 18 hours
with its neighbour), so confidence intervals use the SAME moving-block
bootstrap `direction.py` defines, block length n^(1/3), imported directly
rather than reimplemented. This file adds one thing block_bootstrap_ci does
not do -- a paired-statistic version for Spearman's rho, built with the
identical block-index construction, since rho is not a sample mean and the
mean-based bootstrap cannot be reused for it directly.

ONLY ONE TEST SLICE, DELIBERATELY

`direction.py` reports both `production` (H=19, anchor 17:00, ~365
episodes/fold) and `wide_H19` (every anchor hour, ~8,700/fold, overlapping
and hence lower power per episode but far more of them) because the sign
question needed the power and reporting only the underpowered slice would
have been misleading. ROADMAP.md's Priority 1 spec says to use "the
production episodes," and the rules of evidence for this run say not to go
looking for a positive by trying variants until one passes. A second,
higher-power slice is exactly the kind of second attempt that practice
warns against -- so only `production` is run and reported. Pooled across the
six walk-forward folds it is ~2,046 test episodes, the same population
`direction.py` scores its production slice on, and if that turns out to be
underpowered, that is itself part of the answer, not a reason to go add a
slice until one clears.

THE RELEVANCE TRANSLATION (CONDITION B)

`firstpassage.py` already diagnoses WHY the Gaussian law is wrong: it
implicitly assumes the Brownian range-to-volatility ratio, and BTC's is
smaller, so a Gaussian law fed correct sigma OVERSTATES touch risk. The
natural way to turn a shape PREDICTION into a probability is to correct the
same mechanism firstpassage.py identifies: rescale the sigma the Gaussian
law is fed, proportionally to how far the predicted (or climatological)
shape ratio sits from the Brownian reference ratio for the SAME statistic --

    sigma_eff = RV * (shape_estimate / brownian_reference)

so that an estimate exactly at the Brownian reference reproduces the plain
Gaussian law, and a below-Brownian estimate (as BTC's climatology is)
shrinks the effective sigma and therefore the touch probability, matching
the direction firstpassage.py already found. The Brownian reference for
range/RV (1.5831 here, matching firstpassage.py's 1.5831-ish sampled
control) is reproduced by an independent simulation using the identical
sampling convention `firstpassage.brownian_control` documents at length
(10-second-resolution extremes, 5-minute RV, because that is what the data
actually is at each end and firstpassage.py already showed sampling coarser
than the data biases the benchmark toward zero). The reference for
max(M_up,|M_dn|)/RV does not exist anywhere in the repository -- firstpassage.py
never needed it, because its oracle test only ever used the one-sided M_up
barrier -- so it is computed here by the same simulation, reported
alongside the range figure as a cross-check that the two agree with
firstpassage.py's own number.

For each test episode, this gives P(touch 2%) computed two ways -- once
using this arm's conditional shape prediction, once using the same
episode's climatology (unconditional mean shape) -- and the mean ABSOLUTE
difference across episodes is the number condition (b) is judged against.
This is one reasonable, fully transparent translation, not a new
first-passage derivation; its only job is to convert "this cleared the
null" into "this would or would not move a quote," which is precisely what
ROADMAP.md Priority 1 condition (b) requires and what this project has
already been burned by skipping once: `direction.py` found a statistically
real signal (best skill 0.180% DSC/UNC) that, translated to the same
units the barrier work uses, was 27x too small to matter (ROADMAP.md
section 2). Condition (b) exists specifically so that mistake cannot repeat
here silently.

WHAT A NEGATIVE RESULT LOOKS LIKE, AND WHY IT WOULD BE A COMPLETE SUCCESS

If ridge and gbdt both fail to clear their null on the production slice --
or clear it but move P(touch 2%) by a few tenths of a point instead of a
full point -- the honest reading is NOT "try harder," it is that the 92% of
barrier error firstpassage.py located is close to irreducible at this
feature set and horizon, the same conclusion `direction.py` reached for
sign. That redirects the entire project on evidence: every remaining hour
belongs to ROADMAP items 2-4 (whether the network is even converged, where
the remaining error lives conditionally, and refitting to the post-ETF
regime) rather than to more shape features or a bigger shape head. Both
outcomes of this file are informative for exactly that reason, and the
arms below are the pre-registered set -- not the first of several attempts.

    python -m model.eval.shape --out model/artifacts/shape.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# RidgeCV's small-alpha candidates are ill-conditioned on this feature set by
# design -- several HAR/momentum/calendar columns are near-collinear -- and
# the warning fires on every candidate alpha, on every fold. It is harmless
# (the solve still succeeds; CV simply penalises whichever alpha it hurts)
# and just floods the log, so it is silenced here rather than suppressed
# with a broader `-W ignore` that would hide something unrelated too.
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import block_bootstrap_ci                                # noqa: E402
from eval.firstpassage import gaussian_touch                                 # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.spec import NON_MODEL_COLS                                       # noqa: E402
from noctua.train import load_all, prepare                                   # noqa: E402

ARTIFACTS = Path("model/artifacts")
EPS = 1e-9
BARRIER_PCT_HEADLINE = 2.0        # the decision-rule barrier
BARRIER_PCT_CONTEXT = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
TARGETS = ("range", "max")
RIDGE_ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
               3000.0, 10000.0, 30000.0)


# --------------------------------------------------------------------------
# Brownian shape references -- same sampling convention as
# firstpassage.brownian_control (10s extremes, 5-min RV), extended to also
# return the reference for max(m_up, m_dn)/rv, which firstpassage.py never
# computed because its oracle test only used the one-sided M_up barrier.
# --------------------------------------------------------------------------
def brownian_shape_refs(H: int = 19, n: int = 60_000, steps_per_hour: int = 360,
                        rv_minutes: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N = H * steps_per_hour
    inc = (rng.standard_normal((n, N)) / np.sqrt(N)).astype(np.float32)
    path = np.cumsum(inc, axis=1, dtype=np.float32)
    m_up = np.maximum(path.max(axis=1), 0.0)
    m_dn = np.maximum(-path.min(axis=1), 0.0)
    k = int(rv_minutes * steps_per_hour / 60)
    p0 = np.concatenate([np.zeros((n, 1), dtype=np.float32), path], axis=1)
    r = np.diff(p0[:, ::k], axis=1)
    rv = np.sqrt((r.astype(np.float64) ** 2).sum(axis=1))
    return {
        "range": float(((m_up + m_dn) / rv).mean()),
        "max": float((np.maximum(m_up, m_dn) / rv).mean()),
        "n": n, "H": H, "steps_per_hour": steps_per_hour, "rv_minutes": rv_minutes,
    }


# --------------------------------------------------------------------------
# forecasters
# --------------------------------------------------------------------------
def climatology_predict(ytr: np.ndarray, wtr: np.ndarray, n_te: int):
    """The constant, weighted train-split mean. The required adversary."""
    c = float(np.average(ytr, weights=wtr))
    return np.full(n_te, c, dtype=np.float64), c


def fit_ridge(Xtr, ytr, wtr, Xte):
    """RidgeCV on the full causal feature set, already standardized on TRAIN
    by `prepare()`. The linear ceiling: if shape is linearly present in the
    feature set at all, this finds it.
    """
    from sklearn.linear_model import RidgeCV

    m = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xte), float(m.alpha_)


def fit_gbdt(Xtr, ytr, wtr, Xte, seed=0, max_rows=60_000):
    """Gradient boosting -- the nonlinear ceiling.

    Same hyperparameters direction.py uses for its own nonlinear ceiling:
    deliberately shallow and heavily regularized so that what survives is
    population structure, not memorization of hundreds of thousands of
    overlapping episodes.

    Later walk-forward folds' training sets run past 250k rows -- more than
    the "~50k rows is fine" compute budget this run was given, and a CPU-
    heavy job (`noctua.train_v2`) is competing for the same cores while this
    file runs. Rather than shrink `max_iter` (which would thin every fold
    equally, including the smaller early ones that do not need it) this
    subsamples the TRAINING rows down to `max_rows`, drawn WITHOUT
    replacement and weighted by the same recency weights `sample_weights`
    already computes -- so the cap changes how much data GBDT sees, not
    which era it is biased toward. Ridge is left on the full training set:
    its closed-form solve is cheap at any of these sizes and there is no
    reason to throw away precision on the linear ceiling to save time on the
    nonlinear one.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    n = len(ytr)
    if n > max_rows:
        rng = np.random.default_rng(seed)
        p = wtr / wtr.sum()
        idx = rng.choice(n, size=max_rows, replace=False, p=p)
        Xtr, ytr, wtr = Xtr[idx], ytr[idx], wtr[idx]

    m = HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=200,
        early_stopping=False, random_state=seed,
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict(Xte)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def r2_vs_clim(pred: np.ndarray, clim, y: np.ndarray) -> float:
    """Out-of-sample R^2 against a climatology reference (Campbell-Thompson
    style): 1 - SSE(model) / SSE(climatology). `clim` is either a single
    float (one fold) or a per-episode array (pooled across folds, since each
    fold's climatology is its own train-split mean). Zero by construction
    for the climatology arm itself, exactly like DSC == 0 for a constant
    forecaster in the classification template this file copies.
    """
    y = np.asarray(y, dtype=np.float64)
    sse_m = float(np.sum((y - pred) ** 2))
    sse_c = float(np.sum((y - clim) ** 2))
    return 1.0 - sse_m / max(sse_c, EPS)


def shuffled_null(pred: np.ndarray, clim, y: np.ndarray,
                  n_rep: int = 200, seed: int = 0):
    """Null distributions of R^2 and Spearman rho when the target carries no
    conditional information relative to THIS ALREADY-FITTED prediction.

    Permutes the PREDICTION (not the target), holding the target and the
    climatology reference fixed at every index -- see the module docstring
    for why this is the right null, why it matches `direction.py`'s
    `shuffled_dsc_null` mechanism, and why it does not refit per replicate.
    """
    rng = np.random.default_rng(seed)
    r2s = np.empty(n_rep)
    rhos = np.empty(n_rep)
    for i in range(n_rep):
        pp = rng.permutation(pred)
        r2s[i] = r2_vs_clim(pp, clim, y)
        rho, _ = spearmanr(pp, y)
        rhos[i] = rho if np.isfinite(rho) else 0.0
    return r2s, rhos


def block_bootstrap_pair_ci(pred: np.ndarray, y: np.ndarray, stat_fn,
                            n_rep: int = 500, seed: int = 0, alpha: float = 0.05):
    """Moving-block bootstrap CI for a PAIRED statistic (here, Spearman rho).

    `direction.py`'s `block_bootstrap_ci` resamples blocks of a 1-D series
    and returns a CI for its MEAN -- it cannot be reused as-is for rho, which
    is not a sample mean. This uses the identical block-index construction
    (same block length n^(1/3), same start-sampling scheme) and applies it to
    a pair of arrays instead of one, recomputing `stat_fn` on each resample.
    """
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 20:
        return (float("nan"), float("nan"))
    L = max(1, int(round(n ** (1 / 3))))
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - L + 1, size=(n_rep, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_rep, -1)[:, :n]
    vals = np.empty(n_rep)
    for i in range(n_rep):
        ii = idx[i]
        vals[i] = stat_fn(pred[ii], y[ii])
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def score_all(name: str, pred: np.ndarray, clim, y: np.ndarray,
             seed: int = 0, null_reps: int = 200) -> dict:
    """`clim` is a float (one fold) or a per-episode array (pooled)."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r2 = r2_vs_clim(pred, clim, y)
    rho, rho_p = spearmanr(pred, y)
    rho = float(rho) if np.isfinite(rho) else 0.0
    null_r2, null_rho = shuffled_null(pred, clim, y, n_rep=null_reps, seed=seed)

    d = (y - clim) ** 2 - (y - pred) ** 2       # per-episode SSE reduction
    UNC = float(np.mean((y - clim) ** 2))
    lo_d, hi_d = block_bootstrap_ci(d, seed=seed)
    r2_ci = [lo_d / UNC if np.isfinite(lo_d) else float("nan"),
             hi_d / UNC if np.isfinite(hi_d) else float("nan")]

    return {
        "model": name, "n": int(len(y)),
        "r2": r2, "r2_null_mean": float(null_r2.mean()),
        "r2_null_p95": float(np.quantile(null_r2, 0.95)),
        "clears_null": bool(r2 > np.quantile(null_r2, 0.95)),
        "r2_ci95_block_bootstrap": r2_ci,
        "spearman": rho, "spearman_pvalue": float(rho_p),
        "spearman_null_mean": float(null_rho.mean()),
        "spearman_null_p95": float(np.quantile(null_rho, 0.95)),
        "spearman_clears_null": bool(rho > np.quantile(null_rho, 0.95)),
        "UNC": UNC, "mean_pred": float(np.mean(pred)),
        "clim_mean": float(np.mean(clim)),
    }


def pp_impact(pred: np.ndarray, clim, RV: np.ndarray,
             brownian_ref: float, barrier_pct: np.ndarray) -> dict:
    """Translate a shape prediction into a change in P(touch barrier_pct).

    sigma_eff = RV * (shape_estimate / brownian_reference); see the module
    docstring's "THE RELEVANCE TRANSLATION" section for why this is the
    natural correction given what firstpassage.py already diagnosed. Fed the
    REALIZED RV (the oracle), per ROADMAP.md's instruction, so the number
    isolates what SHAPE skill alone would move, uncontaminated by volatility
    forecast error.
    """
    RV = np.asarray(RV, dtype=np.float64)
    out = {}
    for pct in barrier_pct:
        u = np.log1p(float(pct) / 100.0)
        sigma_pred = RV * (pred / brownian_ref)
        sigma_clim = RV * (clim / brownian_ref)
        p_pred = gaussian_touch(u, sigma_pred)
        p_clim = gaussian_touch(u, sigma_clim)
        delta_pp = 100.0 * (p_pred - p_clim)
        out[f"{pct:g}pct"] = {
            "mean_abs_pp": float(np.mean(np.abs(delta_pp))),
            "mean_pp": float(np.mean(delta_pp)),
            "p95_abs_pp": float(np.quantile(np.abs(delta_pp), 0.95)),
        }
    return out


# --------------------------------------------------------------------------
def run(ep, X, folds, feat_cols, brownian_ref, args):
    fin = np.isfinite(X[feat_cols].to_numpy()).all(1)
    prod = S.production_mask(ep)

    per_fold = []
    pooled = {tgt: {nm: {"pred": [], "y": []} for nm in ("climatology", "ridge", "gbdt")}
             for tgt in TARGETS}
    pooled_RV = []

    for f in folds:
        m_tr = f["train"] & fin
        m_te = f["test"] & fin & prod
        if m_tr.sum() < 5000 or m_te.sum() < 50:
            print(f"  {f['year']}: SKIPPED (train {m_tr.sum():,}, test {m_te.sum():,})")
            continue
        t0 = time.time()

        tr, stds = prepare(ep, X, m_tr)
        te, _ = prepare(ep, X, m_te, *stds)
        wtr = S.sample_weights(ep, m_tr)
        pooled_RV.append(te["RV"])

        y_tr = {"range": tr["m_up"] + tr["m_dn"], "max": tr["m_mx"]}
        y_te = {"range": te["m_up"] + te["m_dn"], "max": te["m_mx"]}

        fold_targets = {}
        for tgt in TARGETS:
            ytr = y_tr[tgt].astype(np.float64)
            yte = y_te[tgt].astype(np.float64)

            clim_pred, clim_val = climatology_predict(ytr, wtr, len(yte))
            ridge_pred, ridge_alpha = fit_ridge(tr["Xa"], ytr, wtr, te["Xa"])
            gbdt_pred = fit_gbdt(tr["Xa"], ytr, wtr, te["Xa"], seed=f["year"],
                                max_rows=args.gbdt_max_rows)

            preds = {"climatology": clim_pred, "ridge": ridge_pred, "gbdt": gbdt_pred}
            rows = []
            for nm, p in preds.items():
                r = score_all(nm, p, clim_val, yte, seed=f["year"], null_reps=args.null_reps)
                rows.append(r)
                pooled[tgt][nm]["pred"].append(p)
                pooled[tgt][nm]["y"].append(yte)
            fold_targets[tgt] = {"rows": rows, "ridge_alpha": ridge_alpha}

        per_fold.append({"year": f["year"], "n_train": int(m_tr.sum()),
                         "n_test": int(m_te.sum()), "targets": fold_targets})
        r2_line = "  ".join(
            f"{tgt}: ridge R2={fold_targets[tgt]['rows'][1]['r2']:+.4f} "
            f"(p95 {fold_targets[tgt]['rows'][1]['r2_null_p95']:+.4f})  "
            f"gbdt R2={fold_targets[tgt]['rows'][2]['r2']:+.4f} "
            f"(p95 {fold_targets[tgt]['rows'][2]['r2_null_p95']:+.4f})"
            for tgt in TARGETS
        )
        print(f"  {f['year']}: n_tr={m_tr.sum():,} n_te={m_te.sum():,}  {r2_line}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not per_fold:
        return None

    RV_pool = np.concatenate(pooled_RV)
    pooled_out = {}
    for tgt in TARGETS:
        y_pool = np.concatenate(pooled[tgt]["climatology"]["y"])
        # Pooled climatology reference: NOT a single blended scalar. Each
        # fold's climatology is its own train-split mean, so pooling it
        # keeps a per-episode array that is piecewise-constant within a fold
        # and varies across folds -- exactly how direction.py keeps
        # `base_rate`'s per-episode array alive through its pooled section
        # rather than collapsing it to one number. Passing this array as
        # `clim` to r2_vs_clim/score_all/pp_impact is what makes R^2 and the
        # touch-probability translation compare each episode to the
        # climatology ITS OWN fold actually used.
        clim_pool = np.concatenate(pooled[tgt]["climatology"]["pred"])
        preds_by_name = {nm: np.concatenate(pooled[tgt][nm]["pred"])
                        for nm in ("climatology", "ridge", "gbdt")}

        rows = []
        for nm in ("climatology", "ridge", "gbdt"):
            p = preds_by_name[nm]
            r = score_all(nm, p, clim_pool, y_pool, seed=1234, null_reps=args.null_reps)
            if nm in ("ridge", "gbdt"):
                lo, hi = block_bootstrap_pair_ci(
                    p, y_pool, lambda a, b: float(spearmanr(a, b)[0]),
                    n_rep=args.spearman_boot_reps, seed=1234)
                r["spearman_ci95_block_bootstrap"] = [lo, hi]
                r["pp_impact_2pct_barrier"] = pp_impact(
                    p, clim_pool, RV_pool, brownian_ref[tgt],
                    np.array([BARRIER_PCT_HEADLINE]))[f"{BARRIER_PCT_HEADLINE:g}pct"]
                r["pp_impact_by_barrier"] = pp_impact(
                    p, clim_pool, RV_pool, brownian_ref[tgt], BARRIER_PCT_CONTEXT)
            rows.append(r)
        pooled_out[tgt] = rows

    return {"per_fold": per_fold, "pooled": pooled_out, "n_pooled": int(len(RV_pool))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Is BTC path SHAPE predictable at all, and does it move a quote?")
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    p.add_argument("--out", type=Path, default=ARTIFACTS / "shape.json")
    p.add_argument("--null-reps", type=int, default=200)
    p.add_argument("--spearman-boot-reps", type=int, default=500)
    p.add_argument("--brownian-n", type=int, default=60_000)
    p.add_argument("--gbdt-max-rows", type=int, default=60_000,
                   help="cap GBDT's training rows via weighted subsampling "
                        "(recency weights preserved); ridge uses all rows")
    a = p.parse_args(argv)

    ep, X = load_all(a.artifacts)
    feat_cols = [c for c in X.columns if c not in set(NON_MODEL_COLS)]
    folds = S.walk_forward_folds(ep)
    print(f"episodes {len(ep):,}  features {X.shape}  "
          f"causal feature set {len(feat_cols)} cols  folds {[f['year'] for f in folds]}")
    prod = S.production_mask(ep)
    print(f"production slice: {prod.sum():,} episodes total\n")

    print("simulating Brownian shape references "
          f"(n={a.brownian_n:,}, 10s extremes, 5min RV -- same convention as "
          "firstpassage.brownian_control)...")
    t0 = time.time()
    bref = brownian_shape_refs(H=19, n=a.brownian_n)
    brownian_ref = {"range": bref["range"], "max": bref["max"]}
    print(f"  range/RV Brownian reference: {bref['range']:.4f}  "
          f"(firstpassage.py's own sampled control is ~1.5831-1.5860 -- "
          f"{'CONSISTENT' if abs(bref['range']-1.586) < 0.02 else 'CHECK THIS'})")
    print(f"  max/RV   Brownian reference: {bref['max']:.4f}  "
          f"(not computed anywhere else in the repository)")
    print(f"  ({time.time()-t0:.0f}s)\n")

    print("[production]")
    result = run(ep, X, folds, feat_cols, brownian_ref, a)
    if result is None:
        print("no usable folds -- nothing to report")
        return 1

    out = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "decision_rule": (
            "shape is predictable-and-useful only if an arm's pooled production "
            "R^2 clears its permutation null p95 AND the resulting mean |change| "
            "in P(touch 2%) is >= 1.0 pp, versus using climatology for every "
            "episode (ROADMAP.md Priority 1)"),
        "brownian_shape_refs": bref,
        "targets": TARGETS,
        "n_pooled_production": result["n_pooled"],
        "per_fold": result["per_fold"],
        "pooled": result["pooled"],
    }

    print(f"\n{'target':>7} {'model':>12} {'R2':>9} {'null p95':>9} {'clears':>7} "
          f"{'spearman':>9} {'null p95':>9} {'clears':>7} {'pp |impact| 2%':>15} {'>=1pp':>6}")
    verdicts = []
    for tgt in TARGETS:
        for r in result["pooled"][tgt]:
            pp = r.get("pp_impact_2pct_barrier", {}).get("mean_abs_pp")
            pp_str = f"{pp:14.3f}" if pp is not None else " " * 14
            relevant = bool(pp is not None and pp >= 1.0)
            useful = bool(r["clears_null"] and relevant)
            if r["model"] in ("ridge", "gbdt"):
                verdicts.append({"target": tgt, "model": r["model"],
                                 "r2": r["r2"], "clears_null": r["clears_null"],
                                 "pp_impact_mean_abs": pp, "relevant": relevant,
                                 "predictable_and_useful": useful})
            print(f"{tgt:>7} {r['model']:>12} {r['r2']:9.5f} {r['r2_null_p95']:9.5f} "
                  f"{str(r['clears_null']):>7} {r['spearman']:9.4f} "
                  f"{r['spearman_null_p95']:9.4f} {str(r['spearman_clears_null']):>7} "
                  f"{pp_str} {str(relevant) if pp is not None else '':>6}")

    out["verdicts"] = verdicts
    out["any_predictable_and_useful"] = bool(any(v["predictable_and_useful"] for v in verdicts))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")

    print(f"\nANY arm predictable-and-useful (clears null AND >= 1pp): "
          f"{out['any_predictable_and_useful']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
