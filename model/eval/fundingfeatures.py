"""
eval/fundingfeatures.py
=====================================================================
A candidate funding-rate feature family: perpetual-swap funding, joined
causally.

WHY FUNDING IS THE RIGHT LEVER

Funding is the periodic payment perpetual-swap longs make to shorts (or vice
versa) to keep the perpetual's price tethered to spot. A POSITIVE funding
rate means longs are paying shorts: the market is willing to pay a premium
to stay long, i.e. long positioning is crowded relative to short positioning.
Crowded positioning is the standard precondition for a liquidation cascade --
when price moves against the crowded side, forced liquidations sell (or buy)
into an already-thin book, and that forced flow is a canonical mechanism for
a realized-vol spike that has nothing to do with new information arriving.
Unlike implied vol (`eval/ivfeatures.py`), which is a forward-looking price
set by option sellers, funding is a mechanical, backward-looking readout of
*who currently holds the crowded position* -- a different information class
again from both DVOL and this repo's own trailing realized-vol cascade
(`har_1h`/`har_6h`/... in `noctua/features.py`, absent from `BASE_COLS`, see
`ivfeatures.py`'s docstring for that argument in full). Whether crowding
PRECEDES the cascade with enough lead time to be useful, or is itself
symptomatic of the same event that produces the cascade (concurrent, not
predictive), is exactly the question a future onset-classifier run would
answer -- not claimed here.

WHAT THIS FILE DOES AND DOES NOT DO

Does: (1) establish the funding row's timestamp semantics and the units of
`interest_1h` / `interest_8h` from the data itself, reusing
`eval/leakage.py`'s `audit_new_data` for the semantics half and an
independent empirical check (in `main()`) for the units half; (2) build
seven candidate features, strictly causal, joined to the existing
`episodes.parquet` population; (3) report per-feature coverage against the
episode population, split by `noctua.splits.time_splits`; (4) adversarially
attack the join for a leak, with a positive-control decoy that MUST be
caught, modelled directly on `ivfeatures.py`'s corrupt-and-diff design
(multiple cuts, dual corruption styles, forced boundary probes, a decoy that
deliberately reads funding at the anchor hour itself).

Does not: wire anything into `noctua/features.py`, `noctua/spec.py`, or the
served model; run the onset walk-forward comparison; train anything;
benchmark anything. This is a data-join artifact, produced once.

    python -m model.eval.fundingfeatures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import block_bootstrap_ci        # noqa: E402
from eval.leakage import audit_new_data               # noqa: E402
from noctua import splits as S                        # noqa: E402
from noctua.train import load_all                     # noqa: E402

HOUR = 3600
DAY_H = 24
Z_WINDOW_H = 20 * DAY_H            # 20 trailing days, in hours -- same
                                    # convention as ivfeatures.py's Z_WINDOW_H
CUM_24H = DAY_H                    # 1 day, in hours
CUM_7D = 7 * DAY_H                 # 7 days, in hours
HOURS_PER_YEAR = 365 * DAY_H       # 8,760 -- same convention as ivfeatures.py
                                    # and serve/predict.py's sigma_annualized_pct;
                                    # used ONLY as a diagnostic scale-check
                                    # below (a linear, non-compounding
                                    # annualization of the median/mean hourly
                                    # rate), never as a feature.

FUNDING_FEATURE_COLS = [
    "fund_rate", "fund_chg_6h", "fund_chg_24h",
    "fund_z_20d", "fund_abs_z_20d", "fund_cum_24h", "fund_cum_7d",
]


# ==========================================================================
# TIMESTAMP SEMANTICS AND THE CAUSAL LAG
# ==========================================================================
#
# `eval/leakage.py:audit_new_data` already establishes, from the funding
# file's own structure (reused below, not re-derived): 64,206 rows spanning
# 2019-04-30 10:00 UTC -> 2026-08-26 17:00 UTC, hourly spacing throughout
# except ONE gap of 10,800s (3 hours, i.e. TWO missing hourly stamps, at
# 2020-08-27 05:00 -> 08:00 UTC), zero duplicated timestamps, and -- unlike
# DVOL -- a positive, structural check for "known as of t": row i's
# `prev_index_price` equals row i-1's own `index_price` in 100.00% of rows
# (reconfirmed here as a precondition, see `main()`). A forward-settlement
# stamp would not need yesterday's own realized price to define itself; a
# trailing, already-known-by-t quantity would, and does. `interest_8h` also
# updates on nearly every hourly row rather than only at the 00/08/16 UTC
# settlement instants, ruling out "this column is stale between
# settlements." Given that, and `audit_new_data`'s own stated contract --
# "a funding/DVOL row stamped at hour h should not be joined to an anchor a
# until a >= h+1" -- this file applies EXACTLY that lag: for an episode
# anchored at `anchor_ts = a`, the most recent ADMISSIBLE funding row is the
# one stamped at `a - 1 hour`. One hour of lag, applied once, at the same
# boundary the rest of the pipeline uses -- not the extra, non-contractual
# shift `noctua/features.py` used to apply and has since removed (see that
# file's `extra_lag_hours` docstring, at a measured cost of 3.33% QLIKE).


class FundingSeries:
    """A dense, hourly-indexed funding array (`interest_1h`) with O(1)
    exact-timestamp lookup and windowed-sum machinery.

    Built the same way `ivfeatures.py`'s `DvolSeries` is built: each row is
    placed at its OWN position on a complete hourly grid spanning
    [min(ts), max(ts)], not assumed gap-free. A missing hour lands as NaN
    rather than silently reading a neighbour. `main()` separately confirms,
    on THIS file, that there is exactly ONE such gap (2 missing hours) --
    this matters more here than for DVOL, because DVOL's own harvest has
    zero gaps and this one does not: the dense-grid machinery below is
    actually exercised, not just defensive scaffolding.

    Unlike `DvolSeries`, which only ever needs one trailing window
    (`Z_WINDOW_H`), this file needs three different window lengths (24h,
    168h, 480h) for `fund_cum_24h` / `fund_cum_7d` / `fund_z_20d`. Rather
    than duplicate `DvolSeries`'s hardcoded `trail_sum`/`trail_cnt`/`trail_sq`
    arrays three times, the same cumulative-sum construction is generalized
    into a single `window()` method parameterized by `k`, called at
    whichever grid position each feature needs (see `build_funding_features`
    for exactly which position each call uses, and why).
    """

    def __init__(self, fund: pd.DataFrame):
        d = fund.sort_values("ts").reset_index(drop=True)
        ts = d["ts"].to_numpy(np.int64)
        assert (np.diff(ts) > 0).all(), "funding ts must be strictly increasing"
        self.ts0 = int(ts[0])
        self.ts1 = int(ts[-1])
        n_grid = (self.ts1 - self.ts0) // HOUR + 1
        grid_val = np.full(n_grid, np.nan)
        pos = (ts - self.ts0) // HOUR
        grid_val[pos] = d["interest_1h"].to_numpy(np.float64)
        self.n = n_grid
        self.val = grid_val
        finite = np.isfinite(grid_val)
        safe = np.where(finite, grid_val, 0.0)
        # cumulative arrays, length n_grid+1: c_sum[i] = sum(safe[0:i]),
        # c_cnt[i] = count of finite values in [0:i). Identical construction
        # to noctua/features.py's `_trailing_sum` and DvolSeries's trail
        # arrays -- just kept as the raw cumulative form so `window()` below
        # can slice it at any (position, k) pair instead of one fixed k.
        self.c_sum = np.concatenate([[0.0], np.cumsum(safe)])
        self.c_cnt = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
        self.c_sq = np.concatenate([[0.0], np.cumsum(safe * safe)])

    def pos_of(self, ts: np.ndarray) -> np.ndarray:
        """Grid position for each `ts`, or -1 where out of [ts0, ts1] or not
        hour-aligned to the grid."""
        aligned = (ts - self.ts0) % HOUR == 0
        inrange = (ts >= self.ts0) & (ts <= self.ts1)
        return np.where(aligned & inrange, (ts - self.ts0) // HOUR, -1)

    def value_at(self, pos: np.ndarray) -> np.ndarray:
        out = np.full(len(pos), np.nan)
        ok = pos >= 0
        out[ok] = self.val[pos[ok]]
        return out

    def window(self, pos: np.ndarray, k: int):
        """sum / count / sum-of-squares over the `k` grid positions
        `[pos-k+1, pos]` -- INCLUSIVE of `pos` itself. NaN (via a zero count)
        wherever the window runs off the front of the grid, `pos` itself is
        out of range, OR any one of the `k` hours in the window is missing
        (the real gap this file has -- and DVOL's zero-gap file does not --
        makes this last case reachable in practice, not just in principle).
        Caller decides what count-below-`k` means for a given feature (here:
        always discarded, never partially averaged -- no stale forward-fill,
        see `build_funding_features`).
        """
        p = np.asarray(pos, dtype=np.int64)
        start = p - k + 1
        ok = (p >= 0) & (p < self.n) & (start >= 0)
        hi = np.clip(p + 1, 0, self.n)
        lo = np.clip(start, 0, self.n)
        s = np.full(len(p), np.nan)
        c = np.full(len(p), np.nan)
        sq = np.full(len(p), np.nan)
        s[ok] = self.c_sum[hi[ok]] - self.c_sum[lo[ok]]
        c[ok] = self.c_cnt[hi[ok]] - self.c_cnt[lo[ok]]
        sq[ok] = self.c_sq[hi[ok]] - self.c_sq[lo[ok]]
        return s, c, sq


def build_funding_features(ep: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Seven candidate features, row-aligned with `ep` (positional, like
    `noctua.features.build_features` is row-aligned with `episodes`).

    Every feature below is a function of funding rows at-or-before `a - 1`
    (`ha`, the causal cutoff derived above) ONLY. No feature reads `fund` at
    or after `a`.

    `fund_rate` and the two `fund_chg_*` are RAW (non-log) differences on
    `interest_1h`, unlike `ivfeatures.py`'s log-differenced `iv_chg_*`: DVOL
    is a strictly-positive index, so log differences are well-defined and
    give a %-like scale; `interest_1h` is signed and frequently crosses (or
    sits at) exactly zero (25th percentile of the raw series is 0.0), so a
    log difference would be undefined or explode near zero. Plain arithmetic
    differences are the only construction that stays well-behaved across the
    whole sign range -- a judgment call made because the source series
    itself rules out the precedent's construction, not a preference.

    `fund_z_20d`'s baseline mean/sd are computed over the 480 hours
    ENDING JUST BEFORE `a-1` (i.e. `window(pos0 - 1, 480)`), deliberately
    EXCLUDING `a-1` itself from its own reference distribution -- identical
    to `ivfeatures.py`'s `iv_z_20d` construction (its `trail_sum` arrays are
    evaluated at `p0c = pos0`, and `DvolSeries`'s trailing arrays are
    exclusive-of-index by construction, so the effective window there is
    also `[pos0-k, pos0-1]`). Copied for consistency with the reference file
    rather than re-derived; the one point excluded out of 480 has no
    material effect either way.

    `fund_cum_24h` / `fund_cum_7d` are SUMS of `interest_1h` over the 24 /
    168 hours INCLUDING `a-1` (`window(pos0, k)`), matching
    `noctua/features.py`'s own convention for e.g. `har_1d` (trailing
    aggregate evaluated so that the causal cutoff hour is the last one
    counted, not excluded from it) -- deliberately a different inclusivity
    convention from the z-score baseline above, because a cumulative "funding
    paid" quantity should count the most recent complete hour, while a
    distribution used to z-score that same hour's value should not include
    the value being scored.
    """
    fs = FundingSeries(fund)
    anchor_ts = ep["anchor_ts"].to_numpy(np.int64)
    ha = anchor_ts - HOUR                        # causal cutoff, a - 1

    pos0 = fs.pos_of(ha)
    ok0 = pos0 >= 0
    fund_rate = fs.value_at(pos0)                # interest_1h at a-1

    def _chg(lag_h: int) -> np.ndarray:
        pos_lag = np.where(pos0 >= 0, pos0 - lag_h, -1)
        pos_lag = np.where(pos_lag >= 0, pos_lag, -1)
        out = fund_rate - fs.value_at(pos_lag)
        return out

    fund_chg_6h = _chg(6)
    fund_chg_24h = _chg(24)

    # ---- fund_z_20d / fund_abs_z_20d --------------------------------------
    pos_base = np.where(pos0 >= 0, pos0 - 1, -1)
    s, c, sq = fs.window(pos_base, Z_WINDOW_H)
    full = ok0 & (c == Z_WINDOW_H)
    mean = np.full(len(pos0), np.nan)
    sd = np.full(len(pos0), np.nan)
    denom = np.maximum(c, 1)
    m = s / denom
    var = sq / denom - m ** 2
    mean[full] = m[full]
    sd[full] = np.sqrt(np.maximum(var[full], 1e-18))
    fund_z_20d = np.where(full, (fund_rate - mean) / sd, np.nan)
    fund_abs_z_20d = np.abs(fund_z_20d)

    # ---- fund_cum_24h / fund_cum_7d ---------------------------------------
    s24, c24, _ = fs.window(pos0, CUM_24H)
    fund_cum_24h = np.where(ok0 & (c24 == CUM_24H), s24, np.nan)
    s7d, c7d, _ = fs.window(pos0, CUM_7D)
    fund_cum_7d = np.where(ok0 & (c7d == CUM_7D), s7d, np.nan)

    return pd.DataFrame({
        "fund_rate": fund_rate,
        "fund_chg_6h": fund_chg_6h,
        "fund_chg_24h": fund_chg_24h,
        "fund_z_20d": fund_z_20d,
        "fund_abs_z_20d": fund_abs_z_20d,
        "fund_cum_24h": fund_cum_24h,
        "fund_cum_7d": fund_cum_7d,
    })


# ==========================================================================
# additional-feature and BLOCKED review
# ==========================================================================
#
# All seven requested features are constructible from `funding_btc.parquet`
# as it exists (`interest_1h`, plus `ts` for the causal grid) -- nothing
# below is BLOCKED. One candidate addition was considered and DECLINED:
# a sign-persistence run-length feature ("hours since funding last flipped
# sign"), motivated by the same crowding story (a long streak of one-sided
# funding is a purer read on "how long has this position been crowded" than
# a single trailing window). It is not built here because (a) it was not
# asked for, (b) it would need its own causality and coverage treatment
# rather than reusing `FundingSeries.window()`, and (c) `ivfeatures.py`'s
# own discipline is to build exactly what is asked plus what is BLOCKED, not
# to free-associate additional candidates -- adding one un-requested feature
# invites the question of why not five more, and every one enlarges the
# multiple-comparisons surface the eventual onset test has to account for.
# Left as a note here rather than built, so the decision is visible instead
# of silent.
NO_BLOCKED_FEATURES_NOTE = (
    "All seven requested funding features (fund_rate, fund_chg_6h, "
    "fund_chg_24h, fund_z_20d, fund_abs_z_20d, fund_cum_24h, fund_cum_7d) "
    "are built above. None are BLOCKED. One additional candidate (a "
    "sign-persistence / run-length feature) was considered and deliberately "
    "NOT built -- see the comment above this constant for why."
)


# ==========================================================================
# LEAKAGE CHECK -- modelled on ivfeatures.py's leakage_check design, itself
# modelled on leakage.py's per_column_causality: multi-cut, dual corruption
# style, forced boundary probes, a decoy that MUST be caught.
# ==========================================================================
def _decoy_funding_feature(ep: pd.DataFrame, fund: pd.DataFrame) -> np.ndarray:
    """Deliberately wrong: reads funding at the ANCHOR HOUR ITSELF (`a`),
    not `a - 1`. Must be flagged VIOLATED below, or the harness has no
    power."""
    fs = FundingSeries(fund)
    pos = fs.pos_of(ep["anchor_ts"].to_numpy(np.int64))
    return fs.value_at(pos)


def _corrupt_funding(fund: pd.DataFrame, cut_ts: int, style: str,
                     rng: np.random.Generator) -> pd.DataFrame:
    """`scale_2_5x` here is `v -> v*factor + offset`, not a pure multiplier
    like `leakage.py`'s / `ivfeatures.py`'s otherwise-identical style. Pure
    multiplication is a no-op on an exact zero (`0 * anything == 0`), and
    `interest_1h` genuinely has exact zeros (its 25th percentile IS 0.0,
    confirmed in `main()`'s units check) -- unlike DVOL / the hourly OHLC
    columns those two files corrupt, which are never exactly zero. A run of
    this file's leakage check WITH the pure-multiplier style found exactly
    this: one boundary episode's decoy went uncaught because the underlying
    `interest_1h` value at that hour happened to be 0.0 (2022-12-28 03:00
    UTC). The additive `offset` (order 1e-5, comfortably above the series'
    own float precision and within its realistic range) closes that blind
    spot while keeping the corruption's spirit -- `leakage.py`'s own
    docstring names exactly this failure mode ("a sufficiently unlucky
    multiplier could... leave a... feature's value numerically close to
    unchanged") as the reason a second, orthogonal style exists at all; this
    is that same reasoning applied one level further, to the multiplier
    style itself, because THIS column can be exactly zero and the reference
    files' columns cannot.
    """
    d = fund.copy()
    v = d["interest_1h"].to_numpy(np.float64).copy()
    mask = d["ts"].to_numpy(np.int64) >= cut_ts
    n = int(mask.sum())
    if style == "scale_2_5x":
        factor = rng.uniform(2.0, 5.0, size=n)
        offset = rng.choice([-1.0, 1.0], size=n) * rng.uniform(5e-6, 5e-5, size=n)
        v[mask] = v[mask] * factor + offset
    elif style == "nan_inject":
        v[mask] = np.nan
    else:
        raise ValueError(style)
    d["interest_1h"] = v
    return d


def leakage_check(ep: pd.DataFrame, fund: pd.DataFrame,
                  cut_fracs=(0.1, 0.3, 0.5, 0.7, 0.9), n_probe_per_cut=300,
                  seed: int = 0) -> dict:
    """Corrupt funding at and after a cut timestamp; assert every real
    feature at anchors whose causal cutoff (`ha = a-1`) falls STRICTLY
    BEFORE the cut is unchanged, across several cuts spanning the
    funding-covered era. The decoy (reads `a`, not `a-1`) is force-probed at
    every boundary episode (`anchor_ts == cut`) and must move -- the
    positive control.
    """
    rng = np.random.default_rng(seed)
    f_sorted = fund.sort_values("ts").reset_index(drop=True)
    ts_arr = f_sorted["ts"].to_numpy(np.int64)

    base = build_funding_features(ep, fund)
    decoy_base = _decoy_funding_feature(ep, fund)
    anchor_ts = ep["anchor_ts"].to_numpy(np.int64)
    ha = anchor_ts - HOUR

    cols = list(base.columns)
    max_bad = {c: 0.0 for c in cols}
    n_tested = {c: 0 for c in cols}
    trials = []

    for frac in cut_fracs:
        cut_ts = int(ts_arr[int(len(ts_arr) * frac)])
        before = np.flatnonzero(ha < cut_ts)
        boundary = np.flatnonzero(anchor_ts == cut_ts)
        hist_probe = (rng.choice(before, size=min(n_probe_per_cut, len(before)),
                                 replace=False) if len(before) else
                     np.array([], dtype=np.int64))
        probe = np.unique(np.concatenate([boundary, hist_probe])).astype(np.int64)
        if len(probe) == 0:
            continue

        for style in ("scale_2_5x", "nan_inject"):
            corrupt = _corrupt_funding(fund, cut_ts, style, rng)
            after = build_funding_features(ep, corrupt)
            decoy_after = _decoy_funding_feature(ep, corrupt)

            a = base.loc[probe, cols].to_numpy(np.float64)
            b = after.loc[probe, cols].to_numpy(np.float64)
            a_fin, b_fin = np.isfinite(a), np.isfinite(b)
            mismatch = a_fin & ~b_fin
            numeric = np.where(a_fin & b_fin, np.abs(a - b), 0.0)
            status = np.where(mismatch, np.inf, numeric)

            trial_bad = 0
            for j, c in enumerate(cols):
                n_tested[c] += int(a_fin[:, j].sum())
                colmax = float(status[:, j].max()) if len(probe) else 0.0
                if colmax > max_bad[c]:
                    max_bad[c] = colmax
                if colmax > 0:
                    trial_bad += 1

            d0 = decoy_base[boundary]
            d1 = decoy_after[boundary]
            d0_fin, d1_fin = np.isfinite(d0), np.isfinite(d1)
            decoy_mismatch = bool((d0_fin & ~d1_fin).any())
            decoy_diff = float(np.nanmax(np.abs(
                np.where(d0_fin & d1_fin, d0 - d1, 0.0)))) if len(boundary) else 0.0
            decoy_caught = decoy_mismatch or decoy_diff > 0

            trials.append({
                "cut_frac": frac, "cut_ts": cut_ts, "style": style,
                "n_probed": int(len(probe)), "n_boundary_episodes": int(len(boundary)),
                "n_columns_moved": trial_bad, "decoy_caught": decoy_caught,
            })

    verdict = {}
    for c in cols:
        if n_tested[c] == 0:
            verdict[c] = "UNTESTED"
        elif max_bad[c] > 0:
            verdict[c] = "VIOLATED"
        else:
            verdict[c] = "CAUSAL"

    trials_with_boundary = [t for t in trials if t["n_boundary_episodes"] > 0]
    decoy_all_caught = (all(t["decoy_caught"] for t in trials_with_boundary)
                        if trials_with_boundary else False)
    n_boundary_caught = sum(1 for t in trials_with_boundary if t["decoy_caught"])

    return {
        "verdict": verdict,
        "max_abs_violation": max_bad,
        "n_tested_cells": n_tested,
        "trials": trials,
        "positive_control": {
            "description": "decoy reads funding at the anchor hour itself "
                            "(a), not a-1; must be flagged in every trial "
                            "with a boundary episode",
            "n_trials": len(trials),
            "n_trials_with_boundary_episode": len(trials_with_boundary),
            "n_trials_with_boundary_episode_caught": n_boundary_caught,
            "catch_rate": (n_boundary_caught / len(trials_with_boundary)
                          if trials_with_boundary else None),
            "caught_in_every_trial": decoy_all_caught,
        },
        "no_blocked_features_note": NO_BLOCKED_FEATURES_NOTE,
    }


# ==========================================================================
# COVERAGE REPORT
# ==========================================================================
def coverage_report(ep: pd.DataFrame, fd: pd.DataFrame) -> dict:
    """Per feature: how many of the episode population get a non-null
    value, split by `noctua.splits.time_splits` (reused, not
    reimplemented)."""
    sp = S.time_splits(ep)
    out = {"n_episodes_total": int(len(ep)),
           "n_train": int(sp["train"].sum()),
           "n_calib": int(sp["calib"].sum()),
           "n_test": int(sp["test"].sum())}
    per_feature = {}
    for c in FUNDING_FEATURE_COLS:
        ok = np.isfinite(fd[c].to_numpy(np.float64))
        per_feature[c] = {
            "total": int(ok.sum()),
            "train": int((ok & sp["train"]).sum()),
            "calib": int((ok & sp["calib"]).sum()),
            "test": int((ok & sp["test"]).sum()),
            "pct_of_total": round(100.0 * ok.sum() / len(ep), 2),
            "pct_of_train": round(100.0 * (ok & sp["train"]).sum() /
                                  max(sp["train"].sum(), 1), 2),
            "pct_of_calib": round(100.0 * (ok & sp["calib"]).sum() /
                                  max(sp["calib"].sum(), 1), 2),
            "pct_of_test": round(100.0 * (ok & sp["test"]).sum() /
                                 max(sp["test"].sum(), 1), 2),
        }
    out["per_feature"] = per_feature
    return out


# ==========================================================================
# UNITS -- established from the data, not asserted
# ==========================================================================
def units_check(fund: pd.DataFrame) -> dict:
    """What `interest_1h` and `interest_8h` ARE, established empirically.

    Deribit's own API documentation for these two field names was NOT
    fetched this session (no network call was made) -- so, exactly like
    `ivfeatures.py` flags for DVOL, the reading below is inferred from the
    data's internal structure, not confirmed against a spec. What IS
    established from the data alone: `interest_8h[i]` tracks the TRAILING
    8-HOUR SUM of `interest_1h` (correlation 0.999, mean absolute residual
    tiny relative to the series' own scale, and stable across every
    calendar year 2019-2026 -- computed below, not asserted) far more
    closely than it tracks the trailing 8-hour MEAN or `interest_1h * 8`.
    That is only possible if `interest_1h` is an ADDITIVE per-hour rate
    contribution and `interest_8h` is its rolling accumulation -- i.e. both
    are DIMENSIONLESS FRACTIONAL RATES (not percentages, not annualized),
    with `interest_1h` the finer-grained, purely additive one.

    This file uses `interest_1h`, not `interest_8h`, for every feature
    (`fund_rate`'s level, the `fund_chg_*` differences, `fund_z_20d`'s
    baseline, and -- critically -- the ADDITIVE terms in `fund_cum_24h` /
    `fund_cum_7d`). Summing `interest_8h` over a 24h or 7d window instead
    would not give cumulative funding paid: because `interest_8h` is
    ITSELF a trailing 8-hour sum, summing 24 consecutive `interest_8h`
    values would count most hours' contribution roughly 8 times over
    (each hour appears in 8 different 8-hour windows), overstating
    cumulative funding by roughly an order of magnitude. `interest_1h` is
    the correct additive unit for that quantity; `interest_8h` is not.
    """
    i1 = fund["interest_1h"].to_numpy(np.float64)
    i8 = fund["interest_8h"].to_numpy(np.float64)
    roll_sum = pd.Series(i1).rolling(8).sum().to_numpy()
    roll_mean = pd.Series(i1).rolling(8).mean().to_numpy()
    resid_sum = i8 - roll_sum
    resid_mean = i8 - roll_mean
    ok = np.isfinite(resid_sum) & np.isfinite(i8)

    yr = pd.to_datetime(fund["ts"], unit="s", utc=True).dt.year.to_numpy()
    per_year = {}
    for y in sorted(set(yr[ok])):
        m = ok & (yr == y)
        mean_abs_i8 = float(np.mean(np.abs(i8[m])))
        per_year[str(int(y))] = {
            "mean_abs_resid_vs_trailing8_sum": float(np.mean(np.abs(resid_sum[m]))),
            "mean_abs_interest_8h": mean_abs_i8,
            "relative_residual_pct": (100.0 * float(np.mean(np.abs(resid_sum[m]))) /
                                      mean_abs_i8 if mean_abs_i8 > 0 else None),
        }

    corr_sum = float(np.corrcoef(i8[ok], roll_sum[ok])[0, 1])
    corr_mean = float(np.corrcoef(i8[ok], roll_mean[ok])[0, 1])

    return {
        "interpretation": (
            "interest_1h and interest_8h are both dimensionless fractional "
            "rates. interest_8h tracks the trailing 8-hour SUM of "
            "interest_1h (not the mean, not interest_1h*8 exactly), so "
            "interest_1h is the additive per-hour building block; used as "
            "such for fund_rate / fund_chg_* / fund_z_20d / fund_cum_*. "
            "UNVERIFIED against Deribit's own API docs -- inferred from the "
            "data's internal structure only, no network call made this "
            "session."
        ),
        "corr_interest_8h_vs_trailing8_sum_interest_1h": corr_sum,
        "corr_interest_8h_vs_trailing8_mean_interest_1h": corr_mean,
        "per_year_residual_check": per_year,
        "interest_1h_median": float(np.nanmedian(i1)),
        "interest_1h_mean": float(np.nanmean(i1)),
        "interest_1h_median_abs": float(np.nanmedian(np.abs(i1))),
        "interest_8h_median": float(np.nanmedian(i8)),
        "interest_8h_mean": float(np.nanmean(i8)),
        "linear_annualized_pct_diagnostic": {
            "note": "hourly rate * HOURS_PER_YEAR * 100, LINEAR (non-"
                    "compounding) approximation, diagnostic scale-check "
                    "only -- not a feature.",
            "median_annualized_pct": float(np.nanmedian(i1)) * HOURS_PER_YEAR * 100.0,
            "mean_annualized_pct": float(np.nanmean(i1)) * HOURS_PER_YEAR * 100.0,
        },
    }


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the candidate funding-rate feature set",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--newdata", type=Path, default=Path("data/newdata"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/funding_features.parquet"))
    ap.add_argument("--report-out", type=Path,
                    default=Path("model/artifacts/funding_features.json"))
    a = ap.parse_args(argv)

    print("=" * 78)
    print("FUNDING FEATURES -- candidate feature set (NOT wired into the model)")
    print("=" * 78)

    ep, X = load_all(a.artifacts)
    fund = pd.read_parquet(a.newdata / "funding_btc.parquet")
    print(f"\n{len(ep):,} episodes, {len(fund):,} funding rows "
          f"({pd.to_datetime(int(fund['ts'].min()), unit='s', utc=True)} "
          f"-> {pd.to_datetime(int(fund['ts'].max()), unit='s', utc=True)})")

    # ---- 1. timestamp semantics (reused from leakage.py, not re-derived) --
    print("\n" + "-" * 78)
    print("1. TIMESTAMP SEMANTICS (reusing eval.leakage.audit_new_data)")
    print("-" * 78)
    nd = audit_new_data(a.newdata)
    f_ = nd["funding"]
    print(f"  funding_btc: {f_['n_rows']:,} rows, monotonic={f_['monotonic_increasing']}, "
          f"duplicated_ts={f_['duplicated_timestamps']}, "
          f"non_hourly_gaps={f_['non_hourly_gaps']}, "
          f"aligned_to_hour={f_['aligned_to_hour_boundary']}")
    print(f"  spacing value counts: {f_['spacing_seconds_value_counts']}")
    print(f"  gap locations: {f_['gap_locations']}")
    print(f"  prev_index_price chains to prior row's index_price: "
          f"{f_['prev_index_price_equals_prior_row_index_price']['fraction_matching']:.4f}")
    print(f"  interest_8h changes on fraction of hourly rows: "
          f"{f_['interest_8h_changes_every_hour_fraction']:.4f}")
    for u in nd["unverified"]:
        print(f"  UNVERIFIED: {u[:100]}...")

    # ---- 2. units check -----------------------------------------------------
    print("\n" + "-" * 78)
    print("2. UNITS (interest_1h vs interest_8h, established empirically)")
    print("-" * 78)
    units = units_check(fund)
    print(f"  corr(interest_8h, trailing8-SUM(interest_1h))  = "
          f"{units['corr_interest_8h_vs_trailing8_sum_interest_1h']:.6f}")
    print(f"  corr(interest_8h, trailing8-MEAN(interest_1h)) = "
          f"{units['corr_interest_8h_vs_trailing8_mean_interest_1h']:.6f}")
    print(f"  -> interest_8h is the trailing 8h SUM of interest_1h, not the mean:")
    print(f"     interest_1h is the additive per-hour unit, used throughout below.")
    for y, r in units["per_year_residual_check"].items():
        print(f"    {y}: relative residual vs trailing8-sum = "
              f"{r['relative_residual_pct']:.2f}%" if r['relative_residual_pct'] is not None
              else f"    {y}: n/a")
    diag = units["linear_annualized_pct_diagnostic"]
    print(f"  median interest_1h={units['interest_1h_median']:.3e}  "
          f"mean={units['interest_1h_mean']:.3e}  "
          f"(linear-annualized: median={diag['median_annualized_pct']:+.3f}%  "
          f"mean={diag['mean_annualized_pct']:+.3f}%)")

    # ---- 3. build features -------------------------------------------------
    print("\n" + "-" * 78)
    print("3. BUILDING FEATURES")
    print("-" * 78)
    fd = build_funding_features(ep, fund)
    for c in FUNDING_FEATURE_COLS:
        v = fd[c].to_numpy(np.float64)
        ok = np.isfinite(v)
        print(f"  {c:<14} non-null {ok.sum():>7,} / {len(v):,}  "
              f"mean={np.nanmean(v[ok]) if ok.any() else float('nan'):+.4e}  "
              f"median={np.nanmedian(v[ok]) if ok.any() else float('nan'):+.4e}")
    print(f"  {NO_BLOCKED_FEATURES_NOTE}")

    # ---- sign-of-fund_rate diagnostic (crowding direction) -----------------
    rate = fd["fund_rate"].to_numpy(np.float64)
    ok = np.isfinite(rate)
    lo, hi = block_bootstrap_ci(rate[ok], seed=0) if ok.sum() >= 20 else (float("nan"),) * 2
    frac_pos = float((rate[ok] > 0).mean()) if ok.any() else float("nan")
    print(f"\n  fund_rate: n={ok.sum():,}  mean={np.nanmean(rate[ok]):+.4e}  "
          f"95% block-bootstrap CI [{lo:+.4e}, {hi:+.4e}]  "
          f"fraction positive (longs paying shorts)={frac_pos:.4f}")

    # ---- 4. coverage report -------------------------------------------------
    print("\n" + "-" * 78)
    print("4. COVERAGE REPORT")
    print("-" * 78)
    cov = coverage_report(ep, fd)
    print(f"  episodes: total={cov['n_episodes_total']:,} "
          f"train={cov['n_train']:,} calib={cov['n_calib']:,} test={cov['n_test']:,}")
    for c in FUNDING_FEATURE_COLS:
        r = cov["per_feature"][c]
        print(f"  {c:<14} total {r['total']:>7,} ({r['pct_of_total']:5.1f}%)  "
              f"train {r['train']:>6,} ({r['pct_of_train']:5.1f}%)  "
              f"calib {r['calib']:>6,} ({r['pct_of_calib']:5.1f}%)  "
              f"test {r['test']:>6,} ({r['pct_of_test']:5.1f}%)")

    # ---- 5. leakage check -----------------------------------------------------
    print("\n" + "-" * 78)
    print("5. LEAKAGE CHECK (corrupt-and-diff, modelled on ivfeatures.py's design)")
    print("-" * 78)
    leak = leakage_check(ep, fund)
    ctrl = leak["positive_control"]
    print(f"  positive control (decoy reads anchor hour a, not a-1): "
          f"boundary episode available in "
          f"{ctrl['n_trials_with_boundary_episode']}/{ctrl['n_trials']} trials, "
          f"caught in {ctrl['n_trials_with_boundary_episode_caught']}/"
          f"{ctrl['n_trials_with_boundary_episode']} "
          f"(catch rate={ctrl['catch_rate']:.4f}), "
          f"caught in all of them = {ctrl['caught_in_every_trial']}")
    if not ctrl["caught_in_every_trial"]:
        print("  !! POSITIVE CONTROL FAILED -- verdicts below are UNRELIABLE !!")
    for c, v in leak["verdict"].items():
        tag = "  " if v == "CAUSAL" else ("**" if v == "VIOLATED" else "??")
        print(f"  {tag} {c:<14} {v}  (n_tested={leak['n_tested_cells'][c]:,}, "
              f"max_violation={leak['max_abs_violation'][c]:.3e})")

    overall_causal = (all(v == "CAUSAL" for v in leak["verdict"].values())
                      and ctrl["caught_in_every_trial"])
    print(f"\n  OVERALL: {'CAUSAL' if overall_causal else 'VIOLATION FOUND'}")

    # ---- write outputs ----------------------------------------------------
    a.artifacts.mkdir(parents=True, exist_ok=True)
    out_df = fd.copy()
    out_df.insert(0, "anchor_ts", ep["anchor_ts"].to_numpy(np.int64))
    out_df.to_parquet(a.out, index=False, compression="zstd")
    print(f"\nwrote {a.out} ({a.out.stat().st_size/1e6:.2f} MB), "
          f"{out_df.shape[0]:,} x {out_df.shape[1]}")
    print(f"  row-alignment check: out_df['anchor_ts'] equals ep['anchor_ts'] "
          f"positionally = "
          f"{bool((out_df['anchor_ts'].to_numpy() == ep['anchor_ts'].to_numpy()).all())}")

    report = {
        "candidate_artifact_only": True,
        "not_wired_into": ["noctua/features.py", "noctua/spec.py",
                          "serve/", "eval/benchmark.py"],
        "timestamp_semantics": nd,
        "units": units,
        "causal_lag_applied_hours": 1,
        "causal_lag_rationale": (
            "a row stamped at hour h is joined only to anchors a >= h+1, "
            "matching eval/leakage.py:audit_new_data's stated contract and "
            "the same convention noctua/features.py applies to the hourly "
            "OHLC bars"),
        "features": FUNDING_FEATURE_COLS,
        "no_blocked_features_note": NO_BLOCKED_FEATURES_NOTE,
        "coverage": cov,
        "fund_rate_diagnostic": {
            "n": int(ok.sum()),
            "mean": float(np.nanmean(rate[ok])) if ok.any() else None,
            "ci95": [lo, hi],
            "fraction_positive": frac_pos,
        },
        "leakage": leak,
        "overall_leakage_verdict": "CAUSAL" if overall_causal else "VIOLATION FOUND",
    }
    a.report_out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {a.report_out}")
    return 0 if overall_causal else 1


if __name__ == "__main__":
    sys.exit(main())
