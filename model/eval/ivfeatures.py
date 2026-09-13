"""
eval/ivfeatures.py
=====================================================================
A candidate DVOL feature family: implied volatility, joined causally.

WHY IMPLIED VOL IS THE RIGHT LEVER, NOT JUST THE NEXT ONE

BENCHMARK.md's own chain gets here in three steps and this file is the fourth.
BENCHMARK.md 12 measured a spike-ONSET classifier -- the first day of a volatility cluster,
as distinct from a day that is merely continuing one -- and found it caps at
AUC 0.7332, CI [0.6547, 0.8052]: real, but weak, against continuation's 0.9293.
BENCHMARK.md 19 then supplied the mechanism: `har_1h` and `har_6h` are computed and fed to
the neural stage, but are ABSENT from `BASE_COLS`, the input set of the
Log-HAR anchor that carries 75% of the blended forecast. So the dominant term
sees nothing faster than a day, and every feature in the set -- fast or slow
-- is a function of BTC's OWN PAST BARS. Onset, by construction, is the day
that trailing statistic has not moved yet: the thing being predicted (a
volatility cluster starting) has not happened yet in the very series being
used to predict it.

Implied volatility breaks that pattern structurally, not just empirically. It
is not a statistic of BTC's past bars at all -- it is the price at which
professional option sellers are willing to write insurance against BTC's
future bars, aggregated across every market participant with a position and
therefore a reason to have priced the risk correctly. If a cluster is about
to start for a reason the options market can see coming (a scheduled event,
a positioning squeeze building, dealer gamma flipping sign) that
forward-looking judgment shows up in the DVOL level or its recent trajectory
BEFORE the realized-vol trailing statistics move, because it is a forecast,
not a description of what already happened. That is the exact information
class an onset classifier needs and the exact information class this
feature set has never had access to.

None of that is claimed to be true here -- it is the mechanism that makes DVOL
worth testing, not a result. This file does not run the onset classifier at
all; `eval/newdata.py`'s pre-registered decision rule (>=6 folds x 3 seeds,
candidate beats baseline AND clears the shuffled-label null in >=5/6 folds,
>=20 onset positives inside the feature's own coverage, continuation AUC does
not drop by more than 0.02) is what will decide whether it earns a place, on
a future run. What this file does is the necessary step before that can even
be attempted: build the candidate features causally, prove they do not leak,
and measure exactly how much of the episode population they can cover -- so
that when the classifier is run, it is run on numbers whose provenance is
already checked rather than assumed.

WHAT A NEGATIVE RESULT WOULD LOOK LIKE, STATED IN ADVANCE

Three ways this could fail, distinguished here so a negative result is
diagnosable rather than just disappointing:

  (a) DVOL turns out to be persistence-in-a-different-costume. Implied vol is
      itself computed partly from recent realized moves (options market
      makers condition on what just happened too), so it may correlate
      strongly with `har_1d` and add nothing INCREMENTAL once the existing
      cascade is already in the model. This would show up as `ivrv_ratio`
      (the part of DVOL not explained by trailing realized vol) carrying the
      onset signal while `iv_level` alone does not -- i.e. the premium is
      informative, the level is not, because the level's information is
      already in `har_1d`.
  (b) The onset days that hurt this model most are crypto-idiosyncratic
      (exchange failures, regulatory shocks, forced-liquidation cascades) and
      genuinely unforeseeable by ANYONE, including options desks pricing
      DVOL. In that case DVOL is a real forward-looking signal in general
      (the options market is not naive) but does not move ahead of the
      SPECIFIC onset days this model's spike flag captures, because those
      onset days are, definitionally, surprises. This is the efficient-
      market-adjacent null: DVOL prices known risk, and the residual after
      pricing known risk is exactly what is left unpriced by construction.
  (c) The coverage gap (see COVERAGE REPORT below -- DVOL has zero overlap
      with roughly two-thirds of the episode timeline) means any onset
      classifier trained on it is evaluated on a shorter, more recent, and
      structurally different regime (post-2021, mostly post-ETF) than the
      one BENCHMARK.md 12's headline 0.7332 was measured on. A win on that restricted
      window would not be directly comparable to the existing ceiling without
      re-measuring the baseline on the SAME restricted window -- exactly the
      apples-to-apples discipline `eval/newdata.py` already prescribes for
      DVOL specifically.

WHAT THIS FILE DOES AND DOES NOT DO

Does: (1) establish the DVOL row's timestamp semantics from the data itself,
by reusing `eval/leakage.py`'s `audit_new_data` rather than re-deriving it;
(2) build six candidate features, strictly causal, joined to the existing
`episodes.parquet` population; (3) report per-feature coverage against the
episode population, split by the same `noctua.splits.time_splits` train /
calib / test partition the rest of the pipeline uses; (4) adversarially
attack the join for a leak, with a positive-control decoy that MUST be
caught, modelled directly on `leakage.py`'s corrupt-and-diff design
(force-included boundary episodes, dual corruption styles, a decoy that
deliberately reads the wrong hour).

Does not: wire anything into `noctua/features.py`, `noctua/spec.py`, or the
served model; run the onset walk-forward comparison itself (that is
`eval/newdata.py`'s job, once this file's coverage and leakage numbers are in
hand); train anything. This is a data-join artifact, produced once, read by
whatever runs the actual onset test next.

    python -m model.eval.ivfeatures
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

# The causal lag, in hours, between the DVOL stamp and the episode anchor.
#
# WHY THIS IS A PARAMETER AND NOT A CONSTANT. A data audit raised an
# unresolved question: Deribit's `public/get_volatility_index_data` returns
# CANDLES, `[ts_ms, open, high, low, close]`, and the harvester keeps the
# CLOSE tagged with the candle's own timestamp. If that timestamp marks the
# candle's START -- the overwhelming convention across exchanges, including
# Deribit's own TradingView-format chart endpoint -- then the close stamped at
# hour h is not determined until h+1, and reading it at a-1 = h means reading a
# value knowable only AT the anchor. That would cancel the whole safety margin.
#
# The endpoint is unreachable from this container (the proxy blocks it) and the
# fetched docs do not state the convention, so the question CANNOT be settled
# here. Rather than argue it, the lag is made a parameter: a 2-hour lag is
# correct under EITHER convention, so a result that survives LAG_HOURS = 2 does
# not depend on the answer. See BENCHMARK.md 29.
DEFAULT_LAG_HOURS = 1
DAY_H = 24
Z_WINDOW_H = 20 * DAY_H          # 20 trailing days, in hours
HOURS_PER_YEAR = 365 * DAY_H     # 8,760 -- same convention as serve/predict.py's
                                  # `sigma_annualized_pct` (sqrt(365*24/H)), read
                                  # but not modified, so the units line up with
                                  # what the rest of this repo already reports.

IV_FEATURE_COLS = [
    "iv_level", "iv_chg_1h", "iv_chg_6h", "iv_chg_24h", "iv_z_20d", "ivrv_ratio",
]


# ==========================================================================
# TIMESTAMP SEMANTICS AND THE CAUSAL LAG
# ==========================================================================
#
# `eval/leakage.py:audit_new_data` already establishes, from the DVOL file's
# own structure (reused below, not re-derived): 47,563 rows, perfectly
# regular hourly spacing (every consecutive gap is exactly 3600s, zero
# duplicated timestamps -- reconfirmed here as a precondition, see
# `main()`), and a documented UNVERIFIED-FROM-DOCS caveat: unlike the funding
# series (whose `prev_index_price` chains exactly to the PRIOR row's own
# `index_price`, positive evidence the value is "known as of t"), DVOL is a
# single `volatility` column with no OHLC-style internal cross-check, so
# whether the hourly value is a point sample AT the hour or an intra-hour
# aggregate cannot be distinguished from the data alone.
#
# Two things ARE established from the data itself, beyond what leakage.py's
# generic spacing check reports: the spacing is exactly regular (perfectly
# uniform 3600s steps across all 47,562 gaps, not "mostly regular with a few
# irregular stretches" -- see `main()`'s reprint of that count), which is the
# signature of either a genuine point-in-time index tick sampled once an
# hour, or a resampled series with no missing bars; either way there is no
# evidence of the batched, multi-hour-delay pattern a settlement-stamped
# forward commitment would show. And DVOL's OWN definition (a 30-day
# constant-maturity IMPLIED volatility index, Deribit's analogue of VIX) is
# by its nature a continuously-quoted market price, not a value that only
# becomes determined at a future settlement instant the way a fixed-rate
# coupon is -- so the "known as of t" reading is not merely the conservative
# default here, it is the reading consistent with what the field measures.
#
# Given that, and because `leakage.py` already states the applicable rule in
# `audit_new_data`'s own `required_lag_if_used_as_a_feature` field --
# "a funding/DVOL row stamped at hour h should not be joined to an anchor a
# until a >= h+1" -- this file applies EXACTLY that lag: for an episode
# anchored at `anchor_ts = a`, the most recent ADMISSIBLE DVOL row is the one
# stamped at `a - 1 hour`. Not `a` (that would read the anchor hour itself,
# the same off-by-one `noctua/features.py`'s own contract exists to forbid),
# and not `a - 2` or later (that would be the extra, non-contractual shift
# `features.py` used to apply and has since removed, at a measured cost of
# 3.33% QLIKE -- see that file's `extra_lag_hours` docstring). One hour of
# lag, applied once, exactly at the boundary the rest of the pipeline uses.


class DvolSeries:
    """A dense, hourly-indexed DVOL array with O(1) exact-timestamp lookup.

    Built by placing each row at its OWN position on a complete hourly grid
    spanning [min(ts), max(ts)] -- not by assuming the file has no gaps. Any
    missing hour lands as NaN in the grid rather than silently reading a
    neighbour, so a future harvest with a real gap fails closed (features go
    NaN at that hour) instead of quietly substituting stale data. `main()`
    separately confirms, on THIS file, that there are in fact zero gaps
    (`non_hourly_gaps == 0`), so the dense-grid machinery below is exercised
    but not currently doing any gap-filling work.
    """

    def __init__(self, dvol: pd.DataFrame):
        d = dvol.sort_values("ts").reset_index(drop=True)
        ts = d["ts"].to_numpy(np.int64)
        assert (np.diff(ts) > 0).all(), "dvol ts must be strictly increasing"
        self.ts0 = int(ts[0])
        self.ts1 = int(ts[-1])
        n_grid = (self.ts1 - self.ts0) // HOUR + 1
        grid_val = np.full(n_grid, np.nan)
        pos = (ts - self.ts0) // HOUR
        grid_val[pos] = d["volatility"].to_numpy(np.float64)
        self.n = n_grid
        self.val = grid_val
        self.logv = np.log(np.maximum(grid_val, 1e-12))
        self.logv[~np.isfinite(grid_val)] = np.nan
        # trailing sum / count of logv over Z_WINDOW_H hours, EXCLUSIVE of the
        # row itself -- identical construction to noctua/features.py's
        # `_trailing_sum`, applied here to the DVOL grid instead of the
        # hourly BTC bars.
        finite = np.isfinite(self.logv)
        safe = np.where(finite, self.logv, 0.0)
        c_sum = np.concatenate([[0.0], np.cumsum(safe)])
        c_cnt = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
        c_sq = np.concatenate([[0.0], np.cumsum(safe * safe)])
        k = Z_WINDOW_H
        idx = np.arange(n_grid)
        ok = idx >= k
        self.trail_sum = np.full(n_grid, np.nan)
        self.trail_cnt = np.full(n_grid, np.nan)
        self.trail_sq = np.full(n_grid, np.nan)
        self.trail_sum[ok] = c_sum[idx[ok]] - c_sum[idx[ok] - k]
        self.trail_cnt[ok] = c_cnt[idx[ok]] - c_cnt[idx[ok] - k]
        self.trail_sq[ok] = c_sq[idx[ok]] - c_sq[idx[ok] - k]

    def pos_of(self, ts: np.ndarray) -> np.ndarray:
        """Grid position for each `ts`, or -1 where out of [ts0, ts1] or
        not hour-aligned to the grid."""
        aligned = (ts - self.ts0) % HOUR == 0
        inrange = (ts >= self.ts0) & (ts <= self.ts1)
        p = np.where(aligned & inrange, (ts - self.ts0) // HOUR, -1)
        return p

    def value_at(self, pos: np.ndarray) -> np.ndarray:
        out = np.full(len(pos), np.nan)
        ok = pos >= 0
        out[ok] = self.val[pos[ok]]
        return out

    def logv_at(self, pos: np.ndarray) -> np.ndarray:
        out = np.full(len(pos), np.nan)
        ok = pos >= 0
        out[ok] = self.logv[pos[ok]]
        return out


def build_iv_features(ep: pd.DataFrame, X: pd.DataFrame, dvol: pd.DataFrame,
                      lag_hours: int = DEFAULT_LAG_HOURS) -> pd.DataFrame:
    """Six candidate features, row-aligned with `ep` (positional, like
    `noctua.features.build_features` is row-aligned with `episodes`).

    Every feature below is a function of DVOL rows at-or-before `a - 1`
    (`ha`, the causal cutoff derived above) ONLY, plus -- for `ivrv_ratio`
    alone -- `har_1d` from the already-built, already-causal `features.parquet`
    (itself a function of hourly bars at-or-before `a - 1`, see
    `noctua/features.py`'s contract, read here and not modified). No feature
    reads `dvol` at or after `a`.
    """
    dv = DvolSeries(dvol)
    anchor_ts = ep["anchor_ts"].to_numpy(np.int64)
    ha = anchor_ts - lag_hours * HOUR           # the causal cutoff, a - lag_hours

    pos0 = dv.pos_of(ha)
    iv_level = dv.logv_at(pos0)                 # log(DVOL points) at a-1

    def _chg(lag_h: int) -> np.ndarray:
        pos_lag = np.where(pos0 >= 0, pos0 - lag_h, -1)
        pos_lag = np.where(pos_lag >= 0, pos_lag, -1)
        return iv_level - dv.logv_at(pos_lag)

    iv_chg_1h = _chg(1)
    iv_chg_6h = _chg(6)
    iv_chg_24h = _chg(24)

    ok0 = pos0 >= 0
    mean = np.full(len(pos0), np.nan)
    sd = np.full(len(pos0), np.nan)
    p0c = np.clip(pos0, 0, dv.n - 1)
    full_window = ok0 & (dv.trail_cnt[p0c] == Z_WINDOW_H)
    m = dv.trail_sum[p0c] / np.maximum(dv.trail_cnt[p0c], 1)
    var = dv.trail_sq[p0c] / np.maximum(dv.trail_cnt[p0c], 1) - m ** 2
    mean[full_window] = m[full_window]
    sd[full_window] = np.sqrt(np.maximum(var[full_window], 1e-12))
    iv_z_20d = np.where(full_window & ok0, (iv_level - mean) / sd, np.nan)

    # ---- ivrv_ratio: the variance-risk-premium proxy ---------------------
    # log(DVOL / 100)  -- DVOL is quoted in ANNUALIZED VOL POINTS (55.57
    #                      median means 55.57%), so /100 converts points to
    #                      a fraction before taking logs.
    # minus
    # log(annualized realized vol)  =  har_1d + 0.5*log(HOURS_PER_YEAR)
    #   -- har_1d (noctua/features.py) is 0.5*log(sum(rv5, 24h)/24), i.e. the
    #      log of the trailing HOURLY vol RATE (per-hour, not annualized).
    #      exp(har_1d) is therefore an hourly standard-deviation-like rate;
    #      annualizing it the same way serve/predict.py's
    #      `sigma_annualized_pct` does (sqrt(365*24/H) applied to an H-hour
    #      sigma is the same operation as sqrt(365*24) applied to a 1-hour
    #      rate) gives log(annualized RV) = har_1d + 0.5*log(8760).
    log100 = float(np.log(100.0))
    har_1d = X["har_1d"].to_numpy(np.float64)
    log_annual_rv = har_1d + 0.5 * np.log(HOURS_PER_YEAR)
    ivrv_ratio = (iv_level - log100) - log_annual_rv

    return pd.DataFrame({
        "iv_level": iv_level,
        "iv_chg_1h": iv_chg_1h,
        "iv_chg_6h": iv_chg_6h,
        "iv_chg_24h": iv_chg_24h,
        "iv_z_20d": iv_z_20d,
        "ivrv_ratio": ivrv_ratio,
    })


# ==========================================================================
# iv_term_slope -- BLOCKED
# ==========================================================================
#
# A term-structure SLOPE needs at least two tenors observed at the same
# instant (e.g. 7-day IV vs 30-day IV) to take a difference of. The harvested
# file (`data/newdata/dvol_btc.parquet`) has exactly one `volatility` column
# and no tenor field anywhere in its schema -- confirmed by
# `pd.read_parquet(...).columns` below in `main()`, not assumed. Deribit's
# DVOL is DEFINED as a single 30-day constant-maturity index (this is
# ASSERTED -- general knowledge about the product, not verified against
# Deribit's docs this session, consistent with how `eval/newdata.py` already
# flags claims of this kind), so even a live re-fetch of the SAME endpoint
# would not produce a second tenor: `get_volatility_index_data` is the same
# single-maturity series regardless of the time range requested. A slope
# would need a DIFFERENT endpoint entirely (e.g. per-expiry option chains,
# to build an IV surface and read off two maturities), which nothing in this
# harvest touches.
#
# BLOCKED. iv_term_slope is not built here, and no placeholder or synthetic
# proxy is substituted for it -- a fabricated slope would be worse than an
# absent feature, because it would look like coverage this repo does not
# have.
IV_TERM_SLOPE_STATUS = {
    "feature": "iv_term_slope",
    "status": "BLOCKED",
    "reason": (
        "dvol_btc.parquet carries a single 'volatility' column (one 30-day "
        "constant-maturity index), no second tenor anywhere in its schema. "
        "A term slope needs at least two maturities sampled at the same "
        "instant; Deribit's DVOL is defined as a single-maturity index, so "
        "re-fetching the same endpoint over any time range cannot produce "
        "one -- it would require a different endpoint (per-expiry option "
        "chains) that this harvest does not fetch. Not built; no synthetic "
        "proxy substituted."
    ),
}


# ==========================================================================
# LEAKAGE CHECK -- modelled on leakage.py's per_column_causality design:
# multi-cut, dual corruption style, forced boundary probes, a decoy that
# MUST be caught. `eval/leakage.py:audit_new_data` is imported and reused
# directly for the timestamp-semantics half (see `main()`); this section is
# the corrupt-and-diff attack specific to the new DVOL-derived columns,
# which `leakage.py` does not itself contain (its `per_column_causality` is
# built around the hourly OHLC bars and `noctua.features.build_features`,
# a different data shape).
# ==========================================================================
def _decoy_iv_feature(ep: pd.DataFrame, dvol: pd.DataFrame) -> np.ndarray:
    """Deliberately wrong: reads DVOL at the ANCHOR HOUR ITSELF (`a`), not
    `a - 1`. Must be flagged VIOLATED below, or the harness has no power."""
    dv = DvolSeries(dvol)
    pos = dv.pos_of(ep["anchor_ts"].to_numpy(np.int64))
    return dv.logv_at(pos)


def _corrupt_dvol(dvol: pd.DataFrame, cut_ts: int, style: str,
                  rng: np.random.Generator) -> pd.DataFrame:
    d = dvol.copy()
    v = d["volatility"].to_numpy(np.float64).copy()
    mask = d["ts"].to_numpy(np.int64) >= cut_ts
    if style == "scale_2_5x":
        v[mask] = v[mask] * rng.uniform(2.0, 5.0, size=int(mask.sum()))
    elif style == "nan_inject":
        v[mask] = np.nan
    else:
        raise ValueError(style)
    d["volatility"] = v
    return d


def leakage_check(ep: pd.DataFrame, X: pd.DataFrame, dvol: pd.DataFrame,
                  lag_hours: int = DEFAULT_LAG_HOURS,
                  cut_fracs=(0.1, 0.3, 0.5, 0.7, 0.9), n_probe_per_cut=300,
                  seed: int = 0) -> dict:
    """Corrupt DVOL at and after a cut timestamp; assert every real feature
    at anchors whose causal cutoff (`ha = a-1`) falls STRICTLY BEFORE the cut
    is unchanged, across several cuts spanning the DVOL-covered era. The
    decoy (reads `a`, not `a-1`) is force-probed at every boundary episode
    (`anchor_ts == cut`) and must move -- the positive control.
    """
    rng = np.random.default_rng(seed)
    dv_sorted = dvol.sort_values("ts").reset_index(drop=True)
    ts_arr = dv_sorted["ts"].to_numpy(np.int64)

    base = build_iv_features(ep, X, dvol, lag_hours)
    decoy_base = _decoy_iv_feature(ep, dvol)
    anchor_ts = ep["anchor_ts"].to_numpy(np.int64)
    ha = anchor_ts - HOUR

    cols = list(base.columns)
    max_bad = {c: 0.0 for c in cols}
    n_tested = {c: 0 for c in cols}
    trials = []

    for frac in cut_fracs:
        cut_ts = int(ts_arr[int(len(ts_arr) * frac)])
        # episodes whose causal cutoff (ha) is strictly before the cut --
        # these are the ones the real features must be provably blind to
        # anything corrupted at/after cut_ts.
        before = np.flatnonzero(ha < cut_ts)
        # force-include every boundary episode (anchor_ts == cut_ts exactly,
        # i.e. ha == cut_ts - HOUR, one hour before the cut -- the tightest
        # case, and the one an off-by-one bug would fail on first) plus a
        # random historical sample, same discipline as leakage.py's
        # `per_column_causality`.
        boundary = np.flatnonzero(anchor_ts == cut_ts)
        hist_probe = (rng.choice(before, size=min(n_probe_per_cut, len(before)),
                                 replace=False) if len(before) else
                     np.array([], dtype=np.int64))
        probe = np.unique(np.concatenate([boundary, hist_probe])).astype(np.int64)
        if len(probe) == 0:
            continue

        for style in ("scale_2_5x", "nan_inject"):
            corrupt = _corrupt_dvol(dvol, cut_ts, style, rng)
            after = build_iv_features(ep, X, corrupt, lag_hours)
            decoy_after = _decoy_iv_feature(ep, corrupt)

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

    return {
        "verdict": verdict,
        "max_abs_violation": max_bad,
        "n_tested_cells": n_tested,
        "trials": trials,
        "positive_control": {
            "description": "decoy reads DVOL at the anchor hour itself (a), "
                            "not a-1; must be flagged in every trial with a "
                            "boundary episode",
            "n_trials": len(trials),
            "n_trials_with_boundary_episode": len(trials_with_boundary),
            "caught_in_every_trial": decoy_all_caught,
        },
        "iv_term_slope": IV_TERM_SLOPE_STATUS,
    }


# ==========================================================================
# COVERAGE REPORT
# ==========================================================================
def coverage_report(ep: pd.DataFrame, iv: pd.DataFrame) -> dict:
    """Per feature: how many of the episode population get a non-null
    value, split by `noctua.splits.time_splits` (the train/calib/test
    partition `noctua/train.py` ships the served model from -- reused, not
    reimplemented)."""
    sp = S.time_splits(ep)
    out = {"n_episodes_total": int(len(ep)),
           "n_train": int(sp["train"].sum()),
           "n_calib": int(sp["calib"].sum()),
           "n_test": int(sp["test"].sum())}
    per_feature = {}
    for c in IV_FEATURE_COLS:
        ok = np.isfinite(iv[c].to_numpy(np.float64))
        per_feature[c] = {
            "total": int(ok.sum()),
            "train": int((ok & sp["train"]).sum()),
            "calib": int((ok & sp["calib"]).sum()),
            "test": int((ok & sp["test"]).sum()),
            "pct_of_total": round(100.0 * ok.sum() / len(ep), 2),
            "pct_of_train": round(100.0 * (ok & sp["train"]).sum() /
                                  max(sp["train"].sum(), 1), 2),
            "pct_of_test": round(100.0 * (ok & sp["test"]).sum() /
                                 max(sp["test"].sum(), 1), 2),
        }
    out["per_feature"] = per_feature
    out["iv_term_slope"] = IV_TERM_SLOPE_STATUS
    return out


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the candidate DVOL implied-vol feature set",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--newdata", type=Path, default=Path("data/newdata"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/iv_features.parquet"))
    ap.add_argument("--report-out", type=Path,
                    default=Path("model/artifacts/iv_features.json"))
    ap.add_argument("--lag-hours", type=int, default=DEFAULT_LAG_HOURS,
                    help="hours between the DVOL stamp and the anchor. 1 is the "
                         "original; 2 is correct under EITHER candle-timestamp "
                         "convention and is the robustness setting (BENCHMARK 29).")
    a = ap.parse_args(argv)

    print("=" * 78)
    print("IV FEATURES -- candidate DVOL feature set (NOT wired into the model)")
    print("=" * 78)

    ep, X = load_all(a.artifacts)
    dvol = pd.read_parquet(a.newdata / "dvol_btc.parquet")
    print(f"\n{len(ep):,} episodes, {len(dvol):,} DVOL rows "
          f"({pd.to_datetime(int(dvol['ts'].min()), unit='s', utc=True).date()} "
          f"-> {pd.to_datetime(int(dvol['ts'].max()), unit='s', utc=True).date()})")

    # ---- 1. timestamp semantics (reused from leakage.py, not re-derived) --
    print("\n" + "-" * 78)
    print("1. TIMESTAMP SEMANTICS (reusing eval.leakage.audit_new_data)")
    print("-" * 78)
    nd = audit_new_data(a.newdata)
    d_ = nd["dvol"]
    print(f"  dvol_btc: {d_['n_rows']:,} rows, monotonic={d_['monotonic_increasing']}, "
          f"duplicated_ts={d_['duplicated_timestamps']}, "
          f"non_hourly_gaps={d_['non_hourly_gaps']}, "
          f"aligned_to_hour={d_['aligned_to_hour_boundary']}")
    print(f"  spacing value counts: {d_['spacing_seconds_value_counts']}")
    print(f"  -> regular hourly point series, zero gaps: applying the SAME "
          f"a >= h+1 lag as everything else in the pipeline")
    for u in nd["unverified"]:
        print(f"  UNVERIFIED: {u[:100]}...")

    # ---- 2. build features -------------------------------------------------
    print("\n" + "-" * 78)
    print("2. BUILDING FEATURES")
    print("-" * 78)
    iv = build_iv_features(ep, X, dvol, a.lag_hours)
    for c in IV_FEATURE_COLS:
        v = iv[c].to_numpy(np.float64)
        ok = np.isfinite(v)
        print(f"  {c:<12} non-null {ok.sum():>7,} / {len(v):,}  "
              f"mean={np.nanmean(v[ok]) if ok.any() else float('nan'):+.4f}  "
              f"median={np.nanmedian(v[ok]) if ok.any() else float('nan'):+.4f}")
    print(f"  iv_term_slope: {IV_TERM_SLOPE_STATUS['status']} -- "
          f"{IV_TERM_SLOPE_STATUS['reason'][:90]}...")

    # ---- units reconciliation for ivrv_ratio -------------------------------
    print("\n" + "-" * 78)
    print("UNITS RECONCILIATION -- ivrv_ratio (variance risk premium proxy)")
    print("-" * 78)
    ratio = iv["ivrv_ratio"].to_numpy(np.float64)
    ok = np.isfinite(ratio)
    lo, hi = block_bootstrap_ci(ratio[ok], seed=0) if ok.sum() >= 20 else (float("nan"),) * 2
    dvol_at = np.exp(iv["iv_level"].to_numpy(np.float64))          # raw DVOL points
    har_1d = X["har_1d"].to_numpy(np.float64)
    rv_annual_pts = 100.0 * np.exp(har_1d + 0.5 * np.log(HOURS_PER_YEAR))
    diff_pts = dvol_at - rv_annual_pts
    print(f"  log(DVOL/100) - [har_1d + 0.5*log(HOURS_PER_YEAR)], "
          f"HOURS_PER_YEAR={HOURS_PER_YEAR}")
    print(f"  ivrv_ratio: n={ok.sum():,}  mean={np.nanmean(ratio[ok]):+.4f}  "
          f"median={np.nanmedian(ratio[ok]):+.4f}  "
          f"95% block-bootstrap CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  equivalent vol-point difference (DVOL_pts - annualized trailing "
          f"RV_pts), diagnostic only, n={ok.sum():,}:")
    print(f"    mean={np.nanmean(diff_pts[ok]):+.2f} pts  "
          f"median={np.nanmedian(diff_pts[ok]):+.2f} pts")
    print(f"  cross-check against OPTION_BUYER_ALPHA.md's reported +8.7 vol-pt "
          f"post-ETF forward VRP (DVOL minus SUBSEQUENT 10-day realized vol -- "
          f"a DIFFERENT, forward-looking quantity from this file's TRAILING "
          f"proxy, so not expected to match exactly, only to be the same sign "
          f"and the same order of magnitude)")

    # ---- 3. coverage report -------------------------------------------------
    print("\n" + "-" * 78)
    print("3. COVERAGE REPORT")
    print("-" * 78)
    cov = coverage_report(ep, iv)
    print(f"  episodes: total={cov['n_episodes_total']:,} "
          f"train={cov['n_train']:,} calib={cov['n_calib']:,} test={cov['n_test']:,}")
    for c in IV_FEATURE_COLS:
        r = cov["per_feature"][c]
        print(f"  {c:<12} total {r['total']:>7,} ({r['pct_of_total']:5.1f}%)  "
              f"train {r['train']:>6,} ({r['pct_of_train']:5.1f}%)  "
              f"calib {r['calib']:>6,}  "
              f"test {r['test']:>6,} ({r['pct_of_test']:5.1f}%)")

    # ---- 4. leakage check -----------------------------------------------------
    print("\n" + "-" * 78)
    print("4. LEAKAGE CHECK (corrupt-and-diff, modelled on eval.leakage's design)")
    print("-" * 78)
    leak = leakage_check(ep, X, dvol, a.lag_hours)
    ctrl = leak["positive_control"]
    print(f"  positive control (decoy reads anchor hour a, not a-1): "
          f"boundary episode available in "
          f"{ctrl['n_trials_with_boundary_episode']}/{ctrl['n_trials']} trials, "
          f"caught in all of them = {ctrl['caught_in_every_trial']}")
    if not ctrl["caught_in_every_trial"]:
        print("  !! POSITIVE CONTROL FAILED -- verdicts below are UNRELIABLE !!")
    for c, v in leak["verdict"].items():
        tag = "  " if v == "CAUSAL" else ("**" if v == "VIOLATED" else "??")
        print(f"  {tag} {c:<12} {v}  (n_tested={leak['n_tested_cells'][c]:,}, "
              f"max_violation={leak['max_abs_violation'][c]:.3e})")

    overall_causal = (all(v == "CAUSAL" for v in leak["verdict"].values())
                      and ctrl["caught_in_every_trial"])
    print(f"\n  OVERALL: {'CAUSAL' if overall_causal else 'VIOLATION FOUND'}")

    # ---- write outputs ----------------------------------------------------
    a.artifacts.mkdir(parents=True, exist_ok=True)
    out_df = iv.copy()
    out_df.insert(0, "anchor_ts", ep["anchor_ts"].to_numpy(np.int64))
    out_df.to_parquet(a.out, index=False, compression="zstd")
    print(f"\nwrote {a.out} ({a.out.stat().st_size/1e6:.2f} MB), "
          f"{out_df.shape[0]:,} x {out_df.shape[1]}")

    report = {
        "candidate_artifact_only": True,
        "not_wired_into": ["noctua/features.py", "noctua/spec.py",
                          "serve/", "eval/benchmark.py"],
        "timestamp_semantics": nd,
        "causal_lag_applied_hours": 1,
        "causal_lag_rationale": (
            "a row stamped at hour h is joined only to anchors a >= h+1, "
            "matching eval/leakage.py:audit_new_data's stated contract and "
            "the same convention noctua/features.py applies to the hourly "
            "OHLC bars"),
        "features": IV_FEATURE_COLS,
        "iv_term_slope": IV_TERM_SLOPE_STATUS,
        "coverage": cov,
        "units_reconciliation": {
            "hours_per_year": HOURS_PER_YEAR,
            "formula": "ivrv_ratio = log(DVOL/100) - (har_1d + 0.5*log(HOURS_PER_YEAR))",
            "ivrv_ratio_mean": float(np.nanmean(ratio[ok])) if ok.any() else None,
            "ivrv_ratio_median": float(np.nanmedian(ratio[ok])) if ok.any() else None,
            "ivrv_ratio_ci95": [lo, hi],
            "equivalent_vol_point_diff_mean": float(np.nanmean(diff_pts[ok])) if ok.any() else None,
            "equivalent_vol_point_diff_median": float(np.nanmedian(diff_pts[ok])) if ok.any() else None,
            "cross_check_reference": "OPTION_BUYER_ALPHA.md: forward VRP "
                "(DVOL - subsequent 10d realized) averages +8.7 vol points "
                "post-ETF -- a different, forward-looking quantity from this "
                "file's trailing proxy; same-sign, same-order-of-magnitude "
                "is the bar, not an exact match",
        },
        "leakage": leak,
        "overall_leakage_verdict": "CAUSAL" if overall_causal else "VIOLATION FOUND",
    }
    a.report_out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"wrote {a.report_out}")
    return 0 if overall_causal else 1


if __name__ == "__main__":
    sys.exit(main())
