"""
eval/lag.py
=====================================================================
BENCHMARK.md 7a measured the defect: NOCTUA's predicted sigma correlates with
REALIZED RV at 0.920 one day AFTER the fact and 0.507 one day BEFORE it. On
the first day of a stress cluster the forecast sits at its own unconditional
median; by the second day it has widened 2-3x. On 2025-04-06 -- a top-10
volatility day -- it was MORE confident than usual. The model reacts to
volatility. It does not anticipate it. That file's own diagnosis was
structural: "every input the model has is a trailing statistic, so a trailing
forecast is what the feature set can express."

This file tests whether that diagnosis is right, wrong, or half right, in four
independent arms. It does not re-measure the lag -- 7a's numbers stand and are
the baseline every arm here is judged against.

DECISION RULE, FIXED BEFORE RUNNING

  ARM 1 is the fork in the road. A binary classifier -- spike tomorrow, yes or
  no -- is fit on the causal feature matrix, walk-forward, and scored against
  a permutation null exactly as `eval/direction.py` scores its sign
  classifiers (shuffled-target DSC null, plus an AUC null built the same way).

    - If NO classifier clears its null (or clears it with negative skill vs
      the constant forecaster -- see the "nulls centred below zero" warning
      below): the lag is an INFORMATION CEILING. Nothing in the causal
      feature set forecasts tomorrow's spike, so no reweighting or
      architecture change inside NOCTUA's current inputs can fix it, and
      that is reported as the answer, not chased with more model variants.
      Arm 4 (response speed) becomes the only lever left, and is run
      regardless because it is a different question -- not "can we predict
      the spike" but "how fast can we stop being wrong about it once it has
      already started" -- and it does not depend on Arm 1's outcome.

    - If a classifier DOES clear its null with positive skill: the lag is
      partly a MODELLING CHOICE, not a hard ceiling, and Arm 2 asks which
      features carry the signal so the fix is targeted rather than a shot in
      the dark.

  ARM 2 exists because this project has already been misled once by the
  obvious method here. AUDIT.md's dow_sin episode: permutation importance
  ranked `cal_dow_sin` above the entire HAR cascade, not because day-of-week
  matters more than trailing volatility, but because the HAR terms are
  strongly intercorrelated (har_1d vs har_5d, r=0.851) and permuting any ONE
  of them leaves its correlated siblings to absorb the loss, while dow_sin,
  being nearly orthogonal to everything (|r| <= 0.094), collects full credit
  for information the whole block actually carries. Permutation importance
  measures redundancy, not importance. So this file does not use it. Feature
  GROUPS are dropped wholesale (leave-one-group-out) and fit ALONE
  (single-group), which is the one design that cannot be fooled by
  redundancy inside a group -- dropping the whole HAR cascade removes every
  copy of whatever it knows at once, rather than leaving correlated
  understudies to cover for a single permuted column.

  ARM 3 measures the asymmetric-loss lever whether or not Arm 1 clears its
  null, because it is a different mechanism: QLIKE punishes under-forecasts
  far more than over-forecasts (measured elsewhere: 1.60x at a factor-2 error,
  4.67x at factor-4), so upweighting high-volatility TRAINING episodes could
  move the fitted model toward the expensive-to-miss tail even without any
  new predictive information -- it is a statement about where the loss
  function points the model, not about what the model can see. There WILL be
  a trade-off (a model that is less wrong on spikes and unchanged everywhere
  else is not on offer), and it is quantified, not hidden: spike-episode
  RV/sigma, overall QLIKE, and calm-episode QLIKE are all reported for every
  weight tried, including the arm that makes calm nights worse.

  ARM 4 is the fallback framed in the brief: if the lag is an information
  ceiling, the remaining lever is not better prediction but faster UPDATING --
  swapping the trailing window the model's volatility anchor is built from
  for a shorter one. NOCTUA's served sigma is 75% `log_har_cal`
  (`har_1d, har_5d, har_22d, cal_H, cal_weekend_frac`, see infer.py's
  BLEND_W=0.25 and the shift it applies) and 25% its own network, and the
  network's Stage-A base is ALSO the Log-HAR cascade -- so in both the linear
  anchor and the seed the network starts from, the fastest information
  available is a full trailing DAY. `har_1h` (a trailing HOUR) and `har_6h`
  already exist in features.parquet and are computed with the same
  no-lookahead contract; `noctua/baselines.py` already defines a
  `har_short` baseline that includes them. This arm asks directly: does
  substituting or adding that faster window recover loss on spike days, and
  specifically on the FIRST day of a cluster -- the exact day 7a showed the
  model sits at its unconditional median.

WHY THIS MATTERS TO AN OPTION SELLER

7a already established the stakes: 7.7% of nights carry 25.8% of the QLIKE
loss, concentrated on exactly the nights a strike is likeliest to break. If
Arm 1 says the information to see that lag coming is not in the features at
all, the honest deliverable is a documented ceiling and a decision to stop
spending research time hunting for a smarter Stage-A architecture -- the fix,
if any, lives in Arm 3 (price the asymmetry into training) or Arm 4 (react
faster once it has started), not in a better predictor. If Arm 1 says the
information IS there, then NOCTUA is leaving a measured, recoverable amount of
loss on the table through a specification choice, and Arm 2 says where to
spend the next engineering hour.

A NEGATIVE RESULT ON ARM 1, AND WHAT IT WOULD LOOK LIKE

No classifier's DSC clears its shuffled-target p95 (or the pooled skill vs
the causal base-rate constant sits at or below zero -- a permutation null
only proves the forecast is not RANDOM noise; it says nothing about whether
the forecast beats the trivial "predict the training era's spike rate every
day" alternative, and a forecast can clear the first bar while failing the
second, which is exactly the "nulls centred below zero" trap this session was
warned about). That would mean tomorrow's spike status is not linearly or
non-linearly recoverable from today's trailing statistics at any resolution
this feature set represents -- consistent with volatility being closer to a
regime-switching or externally-shocked process than a smoothly forecastable
one at a one-day horizon, and a completely respectable thing for the data to
say. It would be reported as the headline finding, not buried under Arm 3/4's
more actionable-sounding results.

SCOPE AND CAVEATS, STATED ONCE

  * Arm 3 retrains NOCTUA itself (Stage A + Stage B, `noctua.train.train_model`,
    hidden=32/epochs<=40, the shipped hyperparameters) but with FEWER seeds
    per arm than the shipped multi-seed, multi-specialist committee
    (`serve.runtime.load_model()`) that BENCHMARK.md's headline numbers come
    from, and it skips the barrier-calibration layer entirely -- QLIKE and
    RV/sigma depend only on `sigma_med`, which `noctua.infer.predict`
    produces directly from Stage A blended with `log_har_cal`, exactly as
    `eval/benchmark.py:run_fold`'s own `predict_avg` computes it, so nothing
    is skipped that the reported quantity depends on. The unweighted (1.0x)
    arm is the reproducibility check: it is expected to land close to, not
    exactly on, 7a's shipped numbers (0.2458 pooled QLIKE, 1.453 spike
    RV/sigma) -- and it does (see the run log), which is what licenses
    reading the WEIGHTED arms' deltas as real rather than as noise from a
    different harness.
  * Every "causal" object in this file -- the spike flag, its 180-day
    trailing threshold, the next-day target, the onset/continuation split --
    is built by `eval.anatomy.trailing_p95` / `attach_spike_flag`, imported
    rather than re-derived, so the definition of "spike" here is identical to
    7a's, not a lookalike that happens to differ in the third decimal place.
  * Two test populations, as `eval/direction.py` and `eval/anatomy.py`
    established: PRODUCTION (H=19, anchor 17:00, one non-overlapping episode
    per day -- what ships, thin) and WIDE / SERVED (H=19, every anchor hour --
    ~18,500 test episodes, where the power is, overlapping by construction).
    Arm 1 reports both. Arm 2's group search (23 model variants -- 11 groups x
    {leave-one-group-out, single-group-only} plus the full model -- times a
    reduced fold count, see `--arm2-folds`) runs on PRODUCTION only -- Arm 1
    finds the same signal on both populations, so the search does not need
    WIDE's extra power, and WIDE's ~60k-row training folds do not fit this
    session's compute budget on a machine shared with
    another agent's concurrent job (measured: one WIDE leave-one-group-out
    fit over 2 folds took 124s under that contention; the full sweep would
    not finish). If a result reported here looks like it needs WIDE-level
    power to trust, that is said explicitly rather than presented as settled.

    python -m model.eval.lag
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.anatomy import attach_spike_flag, build_populations, trailing_p95  # noqa: E402
from eval.direction import (                                                 # noqa: E402
    block_bootstrap_ci, fit_gbdt, fit_logistic, score_all, shuffled_dsc_null,
)
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import load_all, prepare, train_model                      # noqa: E402

EPS = 1e-12
ARTIFACTS = Path("model/artifacts")

# Fixed ex ante -- see module docstring's rules-of-evidence note on not
# variant-hunting after a result is in.
WEIGHT_MULTIPLIERS = [1.0, 3.0, 8.0]
# 1, not the shipped committee's 3+ -- this machine's CPU is shared with
# another agent's concurrent job (confirmed via `ps`), and a single seed
# already reproduces 7a's shipped numbers closely (measured in this session:
# pooled QLIKE 0.2412 vs the shipped 0.2458, spike RV/sigma 1.444 vs 1.453),
# which is what licenses trusting the WEIGHTED arms' deltas from a 1-seed
# harness. See module docstring's scope note.
RETRAIN_SEEDS = 1
RETRAIN_HIDDEN = 32
RETRAIN_EPOCHS = 40
TRAILING_WINDOW_DAYS = 180
NULL_REPS = 200

# The 42 causal columns in features.parquet, partitioned once into groups of
# genuinely correlated siblings -- see module docstring on why permutation
# importance is not used. Every column appears in exactly one group; the
# partition is asserted at import time so a future feature added to
# features.py cannot silently fall outside the sweep.
FEATURE_GROUPS: dict[str, list[str]] = {
    "har": ["har_1h", "har_6h", "har_1d", "har_5d", "har_22d"],
    "semi": ["semi_neg_share_1d", "semi_signed_jump_1d", "semi_neg_1d",
             "semi_neg_share_5d", "semi_signed_jump_5d", "semi_neg_5d"],
    "jump": ["jump_share_1d", "jump_share_5d"],
    "rq": ["rq_noise_1d"],
    "rng": ["rng_park_1d", "rng_gk_1d", "rng_park_5d", "rng_gk_5d"],
    "eff": ["eff_1d", "eff_3d", "eff_7d"],
    "mom": ["mom_ret_1d", "mom_ret_5d", "mom_ret_22d", "mom_dist_ma100",
            "mom_drawdown_90d"],
    "vov": ["vov_5d", "vov_22d"],
    "reg": ["reg_rv_vs_year", "reg_vol_trend", "reg_post_etf"],
    "seas": ["seas_1d", "seas_5d", "seas_22d"],
    "cal": ["cal_hour_sin", "cal_hour_cos", "cal_dow_sin", "cal_dow_cos",
            "cal_H", "cal_weekend_frac", "cal_month_sin", "cal_month_cos"],
}


# ==========================================================================
# shared causal machinery: the spike flag and the next-day target
# ==========================================================================
def daily_spike_series(ep: pd.DataFrame, thresh: pd.Series) -> pd.Series:
    """One causal spike flag per calendar day, reindexed onto a continuous
    daily calendar (gaps -> NaN, not forward-filled -- a missing production
    episode is missing information, not a repeat of yesterday's flag).

    Built from the PRODUCTION population only (`S.production_mask`), which is
    exactly how `eval.anatomy.trailing_p95`'s own threshold is built and how
    `attach_spike_flag` looks a day up -- reusing both here means "spike" has
    one definition in this file and it is 7a's.
    """
    prod = ep.loc[S.production_mask(ep), ["dt", "RV"]].copy()
    flagged = attach_spike_flag(prod, thresh)
    day = pd.DatetimeIndex(flagged["dt"]).floor("D")
    s = pd.Series(flagged["spike"].to_numpy(), index=day)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    full_range = pd.date_range(s.index.min(), s.index.max(), freq="D", tz=s.index.tz)
    return s.reindex(full_range)


def next_day_target(ep: pd.DataFrame, daily: pd.Series) -> np.ndarray:
    """y[i] = tomorrow's spike flag for the calendar day episode i sits in.

    NaN where the following day has no settled production episode (data gap)
    or falls outside the trailing window's warm-up. No look-ahead: y is a
    REALIZED future outcome used only as a training/scoring LABEL, never as a
    predictor, and the causal feature matrix X at row i is built from data
    strictly before day i's anchor regardless of what y says.
    """
    tomorrow = daily.shift(-1)
    day = pd.DatetimeIndex(ep["dt"]).floor("D")
    return pd.Series(day, index=ep.index).map(tomorrow).to_numpy()


def onset_continuation_flags(daily: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Per calendar day: is today the FIRST day of a spike streak (onset) or
    a later day of one already under way (continuation)?

    This is the day-resolution version of the exact pattern 7a described in
    prose ("on the first day of every stress cluster the prediction sits near
    its unconditional median; by day two it has widened 2-3x") -- Arm 4 uses
    it to ask whether a faster feature specifically closes the onset gap.
    """
    is_spike = daily == 1.0
    # `fill_value=False` (not `.shift().fillna(False)`) is load-bearing: a
    # plain `.shift(1)` on a bool Series has no bool NaN to put in the
    # vacated slot, so pandas upcasts to object dtype holding a mix of
    # Python `bool` and float `NaN` -- `.fillna(False)` then leaves that
    # object dtype in place. `~` on an object-dtype Python `bool` invokes
    # `int.__invert__` (bitwise complement: `~True == -2`, `~False == -1`),
    # NOT logical negation, and both results are truthy -- so `~prev_spike`
    # silently evaluated to "true" almost everywhere and `onset` collapsed
    # to "every spike day" (measured: 324 of 324, i.e. equal to `is_spike`
    # itself) rather than "first day of a streak". Caught by a cross-check
    # that `onset` and `continuation` must be disjoint; they overlapped by
    # exactly `continuation`'s full count before this fix.
    prev_spike = is_spike.shift(1, fill_value=False)
    onset = is_spike & ~prev_spike
    continuation = is_spike & prev_spike
    assert not (onset & continuation).any(), "onset/continuation must be disjoint"
    return onset, continuation


def broadcast_daily(ep: pd.DataFrame, daily_bool: pd.Series) -> np.ndarray:
    """Map a per-calendar-day boolean series onto every row of `ep` sharing
    that day. Non-bool / missing days come back False."""
    day = pd.DatetimeIndex(ep["dt"]).floor("D")
    full_range = pd.date_range(daily_bool.index.min(), daily_bool.index.max(),
                               freq="D", tz=daily_bool.index.tz)
    d = daily_bool.reindex(full_range).fillna(False)
    # `.map` against a Series introduces float NaN for any day outside `d`'s
    # index (e.g. ep rows before the trailing window has 30 settled days of
    # history) BEFORE `.fillna` runs, which upcasts the whole column to
    # object dtype -- `.to_numpy()` on that is neither bool nor int and numpy
    # fancy-indexing rejects it outright rather than silently doing the wrong
    # thing. The explicit `.astype(bool)` after `.fillna` is what makes this
    # safe to use as a boolean mask.
    return pd.Series(day, index=ep.index).map(d).fillna(False).astype(bool).to_numpy()


# ==========================================================================
# ARM 1: is the information there at all?
# ==========================================================================
def auc_null(p: np.ndarray, y: np.ndarray, n_rep: int = NULL_REPS, seed: int = 0) -> np.ndarray:
    """Distribution of ROC-AUC under the same shuffle `shuffled_dsc_null`
    uses: permute the forecast against the outcome, destroying the pairing,
    leaving the marginals intact. AUC of a truly uninformative forecast is
    0.5 in expectation but has real sampling variance at these class
    imbalances (~6-8% positive rate), so "beats 0.5" is not the bar -- clearing
    this null's p95 is."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for i in range(n_rep):
        out[i] = roc_auc_score(y, rng.permutation(p))
    return out


def score_with_auc(name, p, y, seed=0):
    from sklearn.metrics import roc_auc_score

    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, np.float64)
    rec = score_all(name, p, y, seed=seed, null_reps=NULL_REPS)
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")
    null = auc_null(p, y, seed=seed)
    rec["auc"] = auc
    rec["auc_null_p95"] = float(np.quantile(null, 0.95))
    rec["auc_null_mean"] = float(null.mean())
    rec["auc_clears_null"] = bool(auc > np.quantile(null, 0.95)) if np.isfinite(auc) else False
    return rec


def attainability_slice(ep, X, folds, mask_name, extra_mask, y_next, spike_today,
                        feat_cols, seed_base=0):
    fin = np.isfinite(X[feat_cols].to_numpy()).all(1)
    have_y = ~np.isnan(y_next)
    base_mask = extra_mask & fin & have_y

    names = ["constant", "persistence", "logistic", "gbdt"]
    per_fold, pooled = [], {k: {"p": [], "y": []} for k in names}

    for f in folds:
        m_tr = f["train"] & base_mask
        m_te = f["test"] & base_mask
        if m_tr.sum() < 500 or m_te.sum() < 30:
            print(f"  {f['year']} {mask_name}: SKIPPED "
                  f"(train {m_tr.sum():,}, test {m_te.sum():,})")
            continue
        t0 = time.time()
        Xtr, Xte = X.loc[m_tr, feat_cols], X.loc[m_te, feat_cols]
        ytr, yte = y_next[m_tr], y_next[m_te]
        wtr = S.sample_weights(ep, m_tr)

        sp_tr = np.nan_to_num(spike_today[m_tr], nan=0.0)
        sp_te = np.nan_to_num(spike_today[m_te], nan=0.0)
        from sklearn.linear_model import LogisticRegression
        pers_model = LogisticRegression(C=1.0, max_iter=1000)
        # constant-column guard: a fold whose training slice never saw a
        # spike day (or never saw a non-spike day) cannot fit a slope --
        # fall back to the unconditional training rate for that fold only.
        if len(np.unique(sp_tr)) < 2:
            p_pers = np.full(len(yte), float(ytr.mean()))
        else:
            pers_model.fit(sp_tr[:, None], ytr)
            p_pers = pers_model.predict_proba(sp_te[:, None])[:, 1]

        preds = {
            "constant": np.full(len(yte), float(np.average(ytr, weights=wtr))),
            "persistence": p_pers,
            "logistic": fit_logistic(Xtr, ytr, wtr, Xte),
            "gbdt": fit_gbdt(Xtr, ytr, wtr, Xte, seed=seed_base + f["year"]),
        }
        rows = []
        for k, p in preds.items():
            rec = score_with_auc(k, p, yte, seed=seed_base + f["year"])
            rows.append(rec)
            pooled[k]["p"].append(p)
            pooled[k]["y"].append(yte)
        per_fold.append({"year": f["year"], "n_train": int(m_tr.sum()),
                         "n_test": int(m_te.sum()), "spike_rate_test": float(yte.mean()),
                         "rows": rows})
        best = max(rows, key=lambda r: r["DSC"])
        print(f"  {f['year']} {mask_name}: n_te={m_te.sum():6,}  "
              f"spike_rate={yte.mean():.4f}  best={best['model']:11} "
              f"AUC={best['auc']:.4f} (null p95 {best['auc_null_p95']:.4f})  "
              f"DSC={best['DSC']:.5f} (null p95 {best['DSC_null_p95']:.5f})  "
              f"({time.time()-t0:.1f}s)", flush=True)

    if not per_fold:
        return None

    pooled_rows = []
    y_pool = np.concatenate(pooled["constant"]["y"])
    p_const = np.concatenate(pooled["constant"]["p"])
    ll_const = -(y_pool * np.log(np.clip(p_const, 1e-6, 1 - 1e-6))
                + (1 - y_pool) * np.log(np.clip(1 - p_const, 1e-6, 1 - 1e-6)))
    for k in names:
        p = np.concatenate(pooled[k]["p"])
        r = score_with_auc(k, p, y_pool, seed=1234)
        ll = -(y_pool * np.log(np.clip(p, 1e-6, 1 - 1e-6))
              + (1 - y_pool) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6)))
        lo, hi = block_bootstrap_ci(ll_const - ll)
        r["vs_constant_logloss_gain"] = float((ll_const - ll).mean())
        r["vs_constant_ci95"] = [lo, hi]
        r["beats_constant"] = bool(lo > 0.0)
        pooled_rows.append(r)
    return {"slice": mask_name, "per_fold": per_fold, "pooled": pooled_rows}


# ==========================================================================
# ARM 2: which features, if any -- leave-one-group-out and single-group
# ==========================================================================
def group_variant_score(ep, X, folds, extra_mask, y_next, cols, seed_base):
    fin = np.isfinite(X[cols].to_numpy()).all(1) if cols else np.ones(len(X), bool)
    have_y = ~np.isnan(y_next)
    base_mask = extra_mask & fin & have_y
    p_all, y_all = [], []
    for f in folds:
        m_tr = f["train"] & base_mask
        m_te = f["test"] & base_mask
        if m_tr.sum() < 500 or m_te.sum() < 30 or not cols:
            continue
        Xtr, Xte = X.loc[m_tr, cols], X.loc[m_te, cols]
        ytr, yte = y_next[m_tr], y_next[m_te]
        wtr = S.sample_weights(ep, m_tr)
        p = fit_gbdt(Xtr, ytr, wtr, Xte, seed=seed_base + f["year"])
        p_all.append(p)
        y_all.append(yte)
    if not p_all:
        return None
    p_pool, y_pool = np.concatenate(p_all), np.concatenate(y_all)
    return score_with_auc("gbdt", p_pool, y_pool, seed=seed_base)


def feature_group_sweep(ep, X, folds, extra_mask, y_next, feat_cols, seed_base=5000):
    full = group_variant_score(ep, X, folds, extra_mask, y_next, feat_cols, seed_base)
    print(f"  full model (all {len(feat_cols)} cols): AUC={full['auc']:.4f}  "
          f"DSC={full['DSC']:.5f}", flush=True)

    logo, solo = {}, {}
    for g, cols in FEATURE_GROUPS.items():
        remaining = [c for c in feat_cols if c not in cols]
        t0 = time.time()
        r_logo = group_variant_score(ep, X, folds, extra_mask, y_next, remaining, seed_base)
        r_solo = group_variant_score(ep, X, folds, extra_mask, y_next,
                                     [c for c in cols if c in feat_cols], seed_base)
        logo[g] = r_logo
        solo[g] = r_solo
        d_auc = (full["auc"] - r_logo["auc"]) if r_logo else float("nan")
        print(f"  group {g:6} (n={len(cols)}): LOGO AUC={r_logo['auc'] if r_logo else float('nan'):.4f} "
              f"(drop {d_auc:+.4f})   solo AUC={r_solo['auc'] if r_solo else float('nan'):.4f} "
              f"(null p95 {r_solo['auc_null_p95'] if r_solo else float('nan'):.4f})  "
              f"({time.time()-t0:.1f}s)", flush=True)
    return {"full": full, "leave_one_group_out": logo, "single_group": solo}


# ==========================================================================
# ARM 3: does upweighting high-vol training episodes trade calm for spike?
# ==========================================================================
def qlike_ledger(RV: np.ndarray, sigma_med: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pv = np.maximum(sigma_med, EPS) ** 2
    rv2 = np.maximum(RV, EPS) ** 2
    r = rv2 / pv
    return r - np.log(r) - 1.0, RV / np.maximum(sigma_med, EPS)


def train_weighted_variant(ep, X, m_tr, m_va, served_mask, thresh, extra_w_full,
                           seeds=RETRAIN_SEEDS, hidden=RETRAIN_HIDDEN,
                           epochs=RETRAIN_EPOCHS, seed0=0):
    """One retrain of NOCTUA (Stage A + Stage B) at a given per-episode
    TRAINING weight multiplier, scored on the served test population.

    Mirrors `eval/benchmark.py:run_fold`'s own `predict_avg` exactly (same
    `log_har_cal` blend, same `I.predict` call) so the QLIKE this reports is
    the same quantity BENCHMARK.md and anatomy.py report -- just from a
    lighter (fewer-seed, no barrier-calibration) harness. See module
    docstring's scope note.
    """
    wtr = S.sample_weights(ep, m_tr)
    wtr = wtr * extra_w_full[m_tr]
    wtr = wtr / max(wtr.mean(), 1e-12)

    tr, stds = prepare(ep, X, m_tr)
    va, _ = prepare(ep, X, m_va, *stds)
    Xb_std = pd.DataFrame(tr["Xb"], columns=BASE_COLS)
    ols = B.OLS(BASE_COLS).fit(Xb_std, tr["y"].astype(np.float64), wtr)

    yall = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))
    bl = B.fit_vol_baselines(X[m_tr], yall[m_tr], wtr)

    models = []
    for s in range(seeds):
        m, best = train_model(tr, wtr, va, hidden=hidden, epochs=epochs,
                              seed=seed0 + s, verbose=False, ols_beta=ols.beta)
        models.append(m)
        print(f"      seed {s}: best_val {best:.5f}", flush=True)

    d, _ = prepare(ep, X, served_mask, *stds)
    lp = bl["log_har_cal"].predict(X[served_mask])
    preds = [I.predict(m, d, har_logvol=lp) for m in models]
    sigma_med = np.mean([p["sigma_med"] for p in preds], axis=0)

    e = ep.loc[served_mask, ["dt", "RV"]].reset_index(drop=True)
    qlike, ratio = qlike_ledger(e["RV"].to_numpy(np.float64), sigma_med)
    df = pd.DataFrame({"dt": e["dt"], "RV": e["RV"], "qlike": qlike, "ratio": ratio})
    df = attach_spike_flag(df, thresh)
    return df


def summarize_variant(df: pd.DataFrame) -> dict:
    have = df.dropna(subset=["spike"])
    spike = have[have["spike"] == 1]
    normal = have[have["spike"] == 0]
    return {
        "n_served": int(len(df)),
        "n_spike": int(len(spike)),
        "n_normal": int(len(normal)),
        "spike_share_pct": 100.0 * len(spike) / len(have),
        "pooled_qlike": float(df["qlike"].mean()),
        "spike_qlike": float(spike["qlike"].mean()),
        "normal_qlike": float(normal["qlike"].mean()),
        "spike_qlike_share_of_total_pct": 100.0 * float(spike["qlike"].sum() / df["qlike"].sum()),
        "spike_median_rv_sigma": float(spike["ratio"].median()),
        "normal_median_rv_sigma": float(normal["ratio"].median()),
    }


def asymmetry_arm(ep, X, out_dir_meta):
    sp = S.time_splits(ep)
    fin = np.isfinite(X.to_numpy()).all(1)
    m_tr, m_va = sp["train"] & fin, sp["calib"] & fin
    served_mask, _ = build_populations(ep, X)
    print(f"  train={m_tr.sum():,}  calib={m_va.sum():,}  served_test={served_mask.sum():,}")

    thresh = trailing_p95(ep, window_days=TRAILING_WINDOW_DAYS)
    train_flag = attach_spike_flag(ep[["dt", "RV"]].copy(), thresh)["spike"].to_numpy()
    spike_train_frac = np.nanmean(train_flag[m_tr])
    print(f"  causal spike share of TRAINING episodes: {100*spike_train_frac:.2f}%")

    results = {}
    for wmul in WEIGHT_MULTIPLIERS:
        extra_w = np.where(np.nan_to_num(train_flag, nan=0.0) == 1.0, wmul, 1.0)
        t0 = time.time()
        print(f"  -- weight={wmul}x --", flush=True)
        df = train_weighted_variant(ep, X, m_tr, m_va, served_mask, thresh, extra_w,
                                    seed0=int(1000 * wmul))
        summ = summarize_variant(df)
        summ["weight_multiplier"] = wmul
        summ["elapsed_s"] = round(time.time() - t0, 1)
        results[f"w{wmul}"] = summ
        print(f"    pooled QLIKE={summ['pooled_qlike']:.4f}  spike QLIKE={summ['spike_qlike']:.4f}  "
              f"normal QLIKE={summ['normal_qlike']:.4f}  spike RV/sigma={summ['spike_median_rv_sigma']:.4f}  "
              f"({summ['elapsed_s']}s)", flush=True)

    base = results["w1.0"]
    for k, r in results.items():
        r["delta_spike_qlike_vs_w1_pct"] = 100.0 * (r["spike_qlike"] / base["spike_qlike"] - 1.0)
        r["delta_normal_qlike_vs_w1_pct"] = 100.0 * (r["normal_qlike"] / base["normal_qlike"] - 1.0)
        r["delta_pooled_qlike_vs_w1_pct"] = 100.0 * (r["pooled_qlike"] / base["pooled_qlike"] - 1.0)
        r["delta_spike_rv_sigma_vs_w1_pct"] = 100.0 * (
            r["spike_median_rv_sigma"] / base["spike_median_rv_sigma"] - 1.0
        )
    return {"causal_spike_share_of_training_pct": 100 * float(spike_train_frac),
            "variants": results}


# ==========================================================================
# ARM 4: the honest alternative -- a faster-updating volatility anchor
# ==========================================================================
def freshness_arm(ep, X):
    sp = S.time_splits(ep)
    fin = np.isfinite(X.to_numpy()).all(1)
    m_tr = sp["train"] & fin
    served_mask, headline_mask = build_populations(ep, X)
    wtr = S.sample_weights(ep, m_tr)
    yall = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))

    baselines = {
        # NOCTUA's own served anchor -- 75% of the shipped sigma_med, see
        # infer.BLEND_W -- and the columns Stage A's linear base is seeded
        # from. The bar this arm is trying to beat.
        "log_har_cal": B.OLS(BASE_COLS).fit(X[m_tr], yall[m_tr], wtr),
        # same cascade and the same two calendar terms, PLUS the two windows
        # shorter than a day that already exist in features.parquet and are
        # already used nowhere in NOCTUA's linear anchor or Stage-A base.
        "har_fast_cal": B.OLS(
            ["har_1h", "har_6h", "har_1d", "har_5d", "har_22d", "cal_H", "cal_weekend_frac"]
        ).fit(X[m_tr], yall[m_tr], wtr),
        # the floor: the single fastest-updating feature alone, no cascade,
        # no calendar -- how far does freshness alone get without the rest of
        # the model's structure.
        "har_1h_only": B.OLS(["har_1h"]).fit(X[m_tr], yall[m_tr], wtr),
    }

    thresh = trailing_p95(ep, window_days=TRAILING_WINDOW_DAYS)
    daily = daily_spike_series(ep, thresh)
    onset, continuation = onset_continuation_flags(daily)
    is_onset = broadcast_daily(ep, onset)
    is_cont = broadcast_daily(ep, continuation)

    out = {}
    for pop_name, mask in (("served", served_mask), ("headline", headline_mask)):
        e = ep.loc[mask, ["dt", "RV"]].reset_index(drop=True)
        Xp = X.loc[mask].reset_index(drop=True)
        onset_p = is_onset[mask]
        cont_p = is_cont[mask]
        pop_out = {}
        Hp = ep.loc[mask, "H"].to_numpy(np.float64)
        for name, ols in baselines.items():
            logvol = ols.predict(Xp)
            sig = np.exp(logvol) * np.sqrt(Hp)
            qlike, ratio = qlike_ledger(e["RV"].to_numpy(np.float64), sig)
            df = pd.DataFrame({"dt": e["dt"], "RV": e["RV"], "qlike": qlike, "ratio": ratio})
            df = attach_spike_flag(df, thresh)
            have = df.dropna(subset=["spike"])
            spike = have[have["spike"] == 1]
            normal = have[have["spike"] == 0]
            onset_rows = qlike[onset_p]
            cont_rows = qlike[cont_p]
            pop_out[name] = {
                "pooled_qlike": float(df["qlike"].mean()),
                "spike_qlike": float(spike["qlike"].mean()) if len(spike) else None,
                "normal_qlike": float(normal["qlike"].mean()) if len(normal) else None,
                "spike_median_rv_sigma": float(spike["ratio"].median()) if len(spike) else None,
                "onset_day_qlike": float(np.mean(onset_rows)) if onset_rows.size else None,
                "onset_day_n": int(onset_rows.size),
                "continuation_day_qlike": float(np.mean(cont_rows)) if cont_rows.size else None,
                "continuation_day_n": int(cont_rows.size),
            }
        # recovery: what fraction of the log_har_cal -> har_fast_cal onset-day
        # gap is closed, relative to the log_har_cal -> spike-population gap
        # 7a already measured -- expressed as a % QLIKE reduction, not a claim
        # about the whole defect being fixed.
        base_onset = pop_out["log_har_cal"]["onset_day_qlike"]
        fast_onset = pop_out["har_fast_cal"]["onset_day_qlike"]
        if base_onset:
            pop_out["onset_day_qlike_change_pct"] = 100.0 * (fast_onset / base_onset - 1.0)
        base_spike = pop_out["log_har_cal"]["spike_qlike"]
        fast_spike = pop_out["har_fast_cal"]["spike_qlike"]
        if base_spike:
            pop_out["spike_qlike_change_pct"] = 100.0 * (fast_spike / base_spike - 1.0)
        base_pooled = pop_out["log_har_cal"]["pooled_qlike"]
        fast_pooled = pop_out["har_fast_cal"]["pooled_qlike"]
        pop_out["pooled_qlike_change_pct"] = 100.0 * (fast_pooled / base_pooled - 1.0)
        out[pop_name] = pop_out
        print(f"  [{pop_name}] log_har_cal pooled={pop_out['log_har_cal']['pooled_qlike']:.4f}  "
              f"har_fast_cal pooled={pop_out['har_fast_cal']['pooled_qlike']:.4f}  "
              f"({pop_out['pooled_qlike_change_pct']:+.2f}%)")
        print(f"    onset-day QLIKE: log_har_cal={base_onset}  har_fast_cal={fast_onset}  "
              f"n_onset={pop_out['log_har_cal']['onset_day_n']}")
        print(f"    spike QLIKE:     log_har_cal={base_spike}  har_fast_cal={fast_spike}")
    return out


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is NOCTUA's one-day lag fixable, or a ceiling?")
    ap.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    ap.add_argument("--out", type=Path, default=ARTIFACTS / "lag.json")
    ap.add_argument("--skip-arm3", action="store_true", help="skip the retrain arm (slow)")
    ap.add_argument("--skip-arm2", action="store_true", help="skip the feature-group sweep")
    ap.add_argument("--arm2-folds", type=int, default=2,
                    help="most-recent N walk-forward folds used for the "
                         "23-variant group sweep (default 2, not all 6 -- see "
                         "module docstring compute note)")
    a = ap.parse_args(argv)

    t_start = time.time()
    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    # ALL 42 columns of features.parquet, including `eff_*` (NON_MODEL_COLS).
    # NOCTUA itself is barred from consuming those three -- `eval/efficiency.py`
    # found no gain for the excursion heads they were built for -- but that is
    # a decision about what NOCTUA's architecture should see, not a claim that
    # they carry no causal information about tomorrow. Task 1 asks whether ANY
    # causal feature predicts a spike, so the search is over the full matrix.
    feat_cols = list(X.columns)
    assert sum(len(v) for v in FEATURE_GROUPS.values()) == len(feat_cols)
    assert set(sum(FEATURE_GROUPS.values(), [])) == set(feat_cols)
    print(f"episodes {len(ep):,}  causal feature cols {len(feat_cols)}  "
          f"folds {[f['year'] for f in folds]}\n")

    thresh = trailing_p95(ep, window_days=TRAILING_WINDOW_DAYS)
    daily = daily_spike_series(ep, thresh)
    y_next = next_day_target(ep, daily)
    spike_today = attach_spike_flag(ep[["dt", "RV"]].copy(), thresh)["spike"].to_numpy()
    print(f"causal daily spike series: {daily.notna().sum()} days, "
          f"{100*np.nanmean(daily):.2f}% flagged spike\n")

    prod = S.production_mask(ep)
    wide = (ep.H == 19).to_numpy()

    # ---- ARM 1: attainability ------------------------------------------------
    print("=" * 78)
    print("ARM 1: is tomorrow's spike status in the causal features at all?")
    print("=" * 78)
    arm1 = {"slices": []}
    for nm, msk in (("production", prod), ("wide_H19", wide)):
        print(f"\n[{nm}]")
        r = attainability_slice(ep, X, folds, nm, msk, y_next, spike_today, feat_cols)
        if r is not None:
            arm1["slices"].append(r)

    print(f"\n{'slice':>12} {'model':>12} {'AUC':>8} {'AUC p95':>8} {'clr':>4} "
          f"{'DSC':>10} {'DSC p95':>10} {'clr':>4} {'gain':>9} {'CI lo':>9} {'CI hi':>9}")
    any_cleared = False
    for sl in arm1["slices"]:
        for r in sl["pooled"]:
            cleared = r["auc_clears_null"] and r["clears_null"] and r.get("beats_constant", False)
            any_cleared = any_cleared or cleared
            print(f"{sl['slice']:>12} {r['model']:>12} {r['auc']:8.4f} {r['auc_null_p95']:8.4f} "
                  f"{str(r['auc_clears_null']):>4} {r['DSC']:10.6f} {r['DSC_null_p95']:10.6f} "
                  f"{str(r['clears_null']):>4} {r.get('vs_constant_logloss_gain', 0):+9.5f} "
                  f"{r['vs_constant_ci95'][0]:+9.5f} {r['vs_constant_ci95'][1]:+9.5f}")
    print(f"\nANY model clears BOTH nulls AND beats the constant (CI95 lo > 0): {any_cleared}")

    # ---- ARM 2: which features -----------------------------------------------
    arm2 = None
    if not a.skip_arm2:
        print("\n" + "=" * 78)
        print(f"ARM 2: leave-one-group-out / single-group (PRODUCTION population, "
              f"gbdt, most recent {a.arm2_folds} folds)")
        print("=" * 78)
        # PRODUCTION, not WIDE: Arm 1 already found the signal on BOTH
        # populations (production pooled DSC clears its null by >9x; wide by
        # >150x), so production has plenty of power for a group-level search.
        # Folds are further limited to the most recent `arm2_folds` (default
        # 2 of 6): a single WIDE-population LOGO fit (n_tr ~60k) measured
        # 124s for 2 folds on this machine, which this session shares with
        # another agent's concurrent job (confirmed via `ps`, not assumed) --
        # even on PRODUCTION-scale data (~2,900 train rows/fold) the identical
        # contention measured 57.8s for one group's 2-fold LOGO+solo pair, so
        # the full 11-group x 6-fold sweep does not fit this session's budget.
        # This is a scope cut on Arm 2 ONLY -- Arm 1's headline finding above
        # already ran the full 6-fold walk-forward on both populations.
        arm2_folds = folds[-a.arm2_folds:]
        arm2 = feature_group_sweep(ep, X, arm2_folds, prod, y_next, feat_cols)

    # ---- ARM 3: asymmetric upweighting ----------------------------------------
    arm3 = None
    if not a.skip_arm3:
        print("\n" + "=" * 78)
        print(f"ARM 3: upweighting spike-flagged training episodes {WEIGHT_MULTIPLIERS}x "
              f"({RETRAIN_SEEDS} seeds/arm, hidden={RETRAIN_HIDDEN})")
        print("=" * 78)
        arm3 = asymmetry_arm(ep, X, {})

    # ---- ARM 4: response speed -------------------------------------------------
    print("\n" + "=" * 78)
    print("ARM 4: a faster-updating volatility anchor (har_1h/har_6h)")
    print("=" * 78)
    arm4 = freshness_arm(ep, X)

    # ---- write -----------------------------------------------------------------
    out = {
        "meta": {
            "n_episodes": int(len(ep)),
            "n_feature_cols": len(feat_cols),
            "trailing_window_days": TRAILING_WINDOW_DAYS,
            "weight_multipliers": WEIGHT_MULTIPLIERS,
            "retrain_seeds": RETRAIN_SEEDS,
            "retrain_hidden": RETRAIN_HIDDEN,
            "null_reps": NULL_REPS,
            "any_arm1_model_clears_both_nulls_and_beats_constant": any_cleared,
            "elapsed_seconds": round(time.time() - t_start, 1),
        },
        "arm1_attainability": arm1,
        "arm2_feature_groups": arm2,
        "arm3_asymmetric_upweighting": arm3,
        "arm4_freshness": arm4,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}  (total {time.time()-t_start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
