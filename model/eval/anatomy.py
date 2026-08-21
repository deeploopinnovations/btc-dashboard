"""
eval/anatomy.py
=====================================================================
The aggregate scores are known. Where the remaining error actually LIVES is
not, and this file is the first attempt to map it.

Every number published elsewhere in this project -- QLIKE 0.24, DSC/UNC
0.046-0.078, deep-tail MCB 1.09pp -- is a MEAN over the test episodes. A mean
tells you nothing about its own shape. A model that is uniformly mediocre on
every night needs different work from one that is excellent on 95% of nights
and catastrophic on the other 5%, and the two are indistinguishable from the
mean alone. This file cuts the mean apart.

THE FOUR QUESTIONS, AND WHY EACH ONE MATTERS TO A SELLER

1. CONDITIONAL ERROR MAP. QLIKE and pinball, broken down by realized-vol
   decile, year, day of week, hour of day, and by whether the episode sits in
   a volatility SPIKE (RV above its own CAUSAL trailing 95th percentile). A
   seller who is told "QLIKE is 0.24" cannot act on that number -- they need
   to know whether the 0.24 is spread evenly across every night (a pricing
   problem, fix the general level) or concentrated in Tuesdays, or in the
   last two hours of the US session, or in the 5% of nights that are already
   violent (a tail problem, fix the extremes, and a very different kind of
   fix). NEGATIVE RESULT, and what it would look like: every cell within
   ~20% of the pooled mean, no cell carrying more than its share of episodes'
   share of total loss. That would say the error is a property of the whole
   distribution, not of any identifiable regime -- which is itself useful,
   because it rules out "just handle Tuesdays specially" as a research
   direction.

2. THE WORST EPISODES. The 100 worst nights by QLIKE, out of ~18,500. Are
   they 100 different nights, or 4-5 real events multiplied by the 24
   overlapping anchor hours that each one touches? And -- the question that
   actually matters commercially -- were they over-forecast (safe, costs
   premium) or under-forecast (the strike breaks)? NEGATIVE RESULT: the worst
   episodes split roughly 50/50 over- and under-forecast, matching the
   population base rate, and are spread across many independent dates. That
   would mean the tail of the loss distribution is not disproportionately the
   dangerous direction, which is a genuinely reassuring finding for a seller
   and would be reported as one.

3. THE ASYMMETRY THAT MATTERS. Nobody sizes a strike off QLIKE. A seller sizes
   off "how far out do I need to write this so it doesn't break," which is
   the model's own 95th/99th-percentile excursion level -- exactly
   `safe_level(pred, 0.05)` and `safe_level(pred, 0.01)`, the functions
   `serve/predict.py` calls to build the numbers a user is actually shown.
   This section asks the only question that matters for a strike sized off
   those levels: how often does reality breach them, and by how much when it
   does. NEGATIVE RESULT: breach rate approximately equal to the nominal
   alpha (5% breaches a 95th-percentile level about 5% of the time) with
   small mean overshoot. That would say the deployed calibration is doing its
   job at the levels a seller actually writes, which is worth knowing because
   it is not automatic -- `AUDIT.md` 3.4 already found Stage B queried
   outside its trained range 2% of the time, and this is the first time the
   SERVED (committee, calibrated) levels are checked against reality on the
   full test period rather than on the walk-forward folds' pooled alphas.

4. SPIKE RESPONSE. `BENCHMARK.md` 6i established that the model's fitted
   weight is 100% pre-ETF and that every arm over-forecasts in 2025-26. That
   is a LEVEL story -- the model runs generally too hot for the current
   regime, and `serve/adaptive.py` patches the level nightly. This section
   asks a different, sharper question: around the worst individual spikes in
   the test period (the kind of event the ROADMAP names -- "a volatility
   spike like the 2025 US-Iran episode ... from features alone, without
   being shown the data"), does the model's predicted vol move BEFORE the
   spike, DURING it, or is it caught flat and only reacts after trailing
   features (har_1h, har_1d, ...) have already absorbed the shock into their
   own averages? Every input this model sees is a trailing statistic, so
   there is a real, structural reason to expect the honest answer is "caught
   flat, and mechanically so" -- and that is reported plainly if the numbers
   say it, per the rules of evidence this session was given.

WHAT THIS FILE DOES NOT DO

It does not retrain anything. It loads the SHIPPED committee artifact exactly
as `serve.app` and `serve.predict` do (`serve.runtime.load_model()`, which
resolves to `noctua_v2.npz`) and runs it forward over already-computed
features and episodes. The only "training" -- and the two heavy jobs this
session shares CPU with are actual re-training runs, so this matters -- is a
plain OLS refit of `log_har_cal` inside `noctua.baselines`, which is a
closed-form least-squares solve, not gradient descent. Everything else is
inference: four matmuls and a cumulative sum per episode (`noctua/infer.py`'s
own description of the cost), times the sizes of population involved.

It does not duplicate the training-dynamics audit. That is a different
question -- whether the network's gradients, seeds and permutation-importance
rankings behave sensibly -- being pursued elsewhere at the same time as this
file. This file only ever asks what the ALREADY-TRAINED artifact gets wrong
and where, treating it as a black box that maps features to a forecast.

POPULATIONS, AND WHY THERE ARE TWO OF THEM

`splits.production_mask` (H=19, anchor_hour=17, one episode per calendar day)
is what every BENCHMARK.md headline is scored on: 769 independent episodes on
the shipped train/calib/test boundary, no window overlap at all (H=19 < 24h
between anchors, so consecutive days' windows never touch). Clean, but too
thin for a decile x year x dow x hour breakdown, and by construction it
cannot answer "hour of day" at all -- there IS only one hour.

`eval/anchors.py` (BENCHMARK.md 6e) already established that ~95% of what is
actually served comes from anchor hours other than 17:00, and that the
model's edge over Log-HAR is flat across anchors (-6.14% at 17:00 vs -6.06%
across all served anchors). So the population used for items 1, 2 and 4 here
is the SERVED one: H=19, every anchor hour, still `ts >= CALIB_END`, ~18,500
episodes. This is the population `eval/anchors.py` calls `anchor_all_served`.

The cost of that choice, stated plainly because it bears on how hard the
per-cell numbers below should be trusted: these episodes overlap by up to
18 of their 19 hours, so neighbouring anchor-hour episodes on the same day
are close to the same observation wearing 24 different clocks.
`BENCHMARK.md` 6b measured this project's redundancy factor directly on the
full augmented grid at ~60.9x (8,380 effective observations from 510,496
episodes). The decile/year/dow/hour breakdown below is therefore
DESCRIPTIVE -- it shows where loss mass concentrates -- and is cross-checked
against the independent 769-episode production population wherever the two
can be compared (by year, by spike, and in the excursion-breach section,
which is cheap enough to run on both populations in full). Where the two
populations disagree, both numbers are reported rather than the more
convenient one.

CAUSALITY

The one place this file could leak the future is the "spike" flag in item 1
and the trailing-percentile threshold in item 4's baseline. Both are built by
`trailing_p95()`, which uses ONLY production-anchor (17:00, non-overlapping)
episodes strictly BEFORE the calendar day being flagged -- the same causal
pattern `serve/adaptive.py` uses for its nightly correction (settle strictly
before use), with the same 180-day trailing window BENCHMARK.md 6 found the
adaptive correction is insensitive to (30/60/90/180 days all land within
0.98-0.99 there). No walk-forward retraining is involved, so there is no
train/test boundary to violate here beyond that.

SCOPE

The shipped `noctua_v2.npz` committee predates the `q_mx` (max-excursion)
head -- measured here (`model.has_mx()` is False on this artifact) -- so
"excursion" throughout this file means the two MARGINALS, up and down,
scored through the exact calibrated `safe_level`/`touch_prob` methods
`serve/predict.py` calls. It does not mean the combined `max(M_up, M_dn)`
"either" quantity `eval/either.py` studies; that head is not in the artifact
that ships. This is stated once here rather than qualified in every table.

Also out of scope: `serve/adaptive.py`'s nightly volatility correction. Every
number below is the raw artifact, exactly as `noctua/evaluate.py` and
`eval/regimes.py` score it and exactly as BENCHMARK.md's headline QLIKE
(0.2386-0.2408 depending on population) is computed -- BEFORE the adaptive
shrink that is layered on in production. That scoping is deliberate and
matches house convention: BENCHMARK.md 6 reports "raw" and "after the
adaptive correction" as two different numbers for exactly this reason, and
conflating them here would misattribute the adaptive layer's work to the
model being audited.

    python -m model.eval.anatomy
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

from noctua import baselines as B                                            # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402
from serve.runtime import load_model                                         # noqa: E402

EPS = 1e-12
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ==========================================================================
# populations
# ==========================================================================
def build_populations(ep: pd.DataFrame, X: pd.DataFrame):
    """The two test populations described in the module docstring."""
    fin = np.isfinite(X.to_numpy()).all(1)
    sp = S.time_splits(ep)
    prod = S.production_mask(ep)
    served = sp["test"] & fin & (ep["H"] == 19).to_numpy()
    headline = sp["test"] & fin & prod
    return served, headline


# ==========================================================================
# causal trailing percentile (for the SPIKE flag; see module docstring)
# ==========================================================================
def trailing_p95(ep: pd.DataFrame, window_days: int = 180, min_obs: int = 30,
                 q: float = 0.95) -> pd.Series:
    """Trailing q-th percentile of RV, built ONLY from settled production
    anchors strictly before the day being scored.

    One value per calendar day. `min_periods=min_obs` so early dates without
    enough trailing history return NaN rather than a threshold estimated from
    a handful of days -- those episodes are simply excluded from the spike
    breakdown, and how many are excluded is reported.
    """
    prod = ep.loc[S.production_mask(ep)].sort_values("anchor_ts")
    day = prod["dt"].dt.floor("D")
    s = pd.Series(prod["RV"].to_numpy(np.float64), index=pd.DatetimeIndex(day))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # shift(1) before rolling: the value attributed to day d is a function of
    # days strictly before d, never of d itself.
    return s.shift(1).rolling(f"{window_days}D", min_periods=min_obs).quantile(q)


def attach_spike_flag(df: pd.DataFrame, thresh: pd.Series) -> pd.DataFrame:
    """Look up each episode's causal trailing threshold by calendar date.

    `day` (one entry per episode) carries heavy duplication -- the served
    population has up to 24 episodes per calendar date, one per anchor hour --
    so this is a many-to-one lookup, not a reindex. `Series.map` against a
    UNIQUE daily reference (`t_full`, built once via `reindex().ffill()` over
    every calendar day in range) does that correctly; `Series.reindex` does
    not, because it requires the axis being read FROM to be unique-safe
    against a duplicated target, and raises rather than silently doing the
    wrong thing.
    """
    df = df.copy()
    day = pd.DatetimeIndex(df["dt"]).floor("D")
    if len(thresh) == 0:
        df["trailing_p95"] = np.nan
        df["spike"] = np.nan
        return df
    full_range = pd.date_range(thresh.index.min(), max(thresh.index.max(), day.max()),
                               freq="D", tz=thresh.index.tz)
    t_full = thresh.reindex(full_range).ffill()
    # NOTE: map against `day` directly, not `day.values` -- `.values` on a
    # tz-aware DatetimeIndex silently drops the tz (returns tz-naive
    # datetime64[us]), which then matches nothing in t_full's tz-aware index
    # and every lookup comes back NaN. Cost real time to find once already.
    df["trailing_p95"] = pd.Series(day, index=df.index).map(t_full).to_numpy()
    # float, not bool: rows lacking enough trailing history get NaN rather
    # than a coerced True/False, and pandas' bool dtype cannot hold NaN.
    have = df["trailing_p95"].notna().to_numpy()
    spike = np.full(len(df), np.nan)
    spike[have] = (df["RV"].to_numpy()[have] > df["trailing_p95"].to_numpy()[have]).astype(float)
    df["spike"] = spike
    return df


# ==========================================================================
# scoring: run the shipped committee once, build the per-episode ledger
# ==========================================================================
def score_population(model, ep: pd.DataFrame, X: pd.DataFrame, mask: np.ndarray):
    """One forward pass of the shipped committee over `mask`, plus the
    per-episode QLIKE and pinball ledger everything else in this file reads.
    """
    e = ep.loc[mask].reset_index(drop=True)
    Xm = X.loc[mask].reset_index(drop=True)
    H = e["H"].to_numpy(np.float64)

    d = model.prepare(Xm, H)
    pred = model.predict(d)

    sigma_med = np.asarray(pred["sigma_med"], dtype=np.float64)
    RV = e["RV"].to_numpy(np.float64)
    pv = sigma_med ** 2
    rv2 = RV ** 2
    r = np.maximum(rv2, EPS) / np.maximum(pv, EPS)
    qlike = r - np.log(r) - 1.0                     # per-episode QLIKE

    y = B.har_target(RV, H)                          # log hourly vol rate, the Stage-A target
    qa = np.asarray(pred["qa"], dtype=np.float64)
    levels = np.asarray(model.levels, dtype=np.float64)[None, :]
    dq = y[:, None] - qa
    pinball = np.maximum(levels * dq, (levels - 1.0) * dq).mean(axis=1)

    df = pd.DataFrame({
        "anchor_ts": e["anchor_ts"].to_numpy(),
        "dt": e["dt"],
        "year": e["dt"].dt.year.to_numpy(),
        "dow": e["dow"].to_numpy(),
        "anchor_hour": e["anchor_hour"].to_numpy(),
        "H": H,
        "RV": RV,
        "sigma_med": sigma_med,
        "ratio_rv_sigma": RV / np.maximum(sigma_med, EPS),
        "qlike": qlike,
        "pinball": pinball,
        "M_up": np.abs(e["M_up"].to_numpy(np.float64)),
        "M_dn": np.abs(e["M_dn"].to_numpy(np.float64)),
    })
    return df, pred


# ==========================================================================
# 1. conditional error map
# ==========================================================================
def summarize_by(df: pd.DataFrame, col: str, label_fn=None) -> list[dict]:
    overall_qlike = float(df["qlike"].mean())
    overall_pinball = float(df["pinball"].mean())
    total_loss = float(df["qlike"].sum())
    rows = []
    for k, sub in df.groupby(col, dropna=True):
        lbl = label_fn(k) if label_fn else k
        n = len(sub)
        rows.append({
            "cell": lbl,
            "n": int(n),
            "share_of_episodes_pct": 100.0 * n / len(df),
            "mean_qlike": float(sub["qlike"].mean()),
            "qlike_vs_pooled_pct": 100.0 * (float(sub["qlike"].mean()) / overall_qlike - 1.0),
            "mean_pinball": float(sub["pinball"].mean()),
            "pinball_vs_pooled_pct": 100.0 * (float(sub["pinball"].mean()) / overall_pinball - 1.0),
            "share_of_total_qlike_loss_pct": 100.0 * float(sub["qlike"].sum()) / total_loss,
            "median_rv_sigma_ratio": float(sub["ratio_rv_sigma"].median()),
            "median_rv_pct": float(100 * sub["RV"].median()),
        })
    return rows


def conditional_error_map(df: pd.DataFrame, thresh: pd.Series) -> dict:
    df = df.copy()
    df["rv_decile"] = pd.qcut(df["RV"], 10, labels=False, duplicates="drop") + 1
    df = attach_spike_flag(df, thresh)

    out = {
        "n": int(len(df)),
        "pooled_mean_qlike": float(df["qlike"].mean()),
        "pooled_mean_pinball": float(df["pinball"].mean()),
        "by_rv_decile": summarize_by(df, "rv_decile", lambda k: f"D{int(k)}"),
        "by_year": summarize_by(df, "year", lambda k: int(k)),
        "by_dow": summarize_by(df, "dow", lambda k: DOW_NAMES[int(k)]),
        "by_hour": summarize_by(df, "anchor_hour", lambda k: int(k)),
    }
    have_spike = df.dropna(subset=["spike"])
    out["spike_flag_coverage_pct"] = 100.0 * len(have_spike) / len(df)
    out["by_spike"] = summarize_by(
        have_spike.assign(spike=have_spike["spike"].astype(bool)), "spike",
        lambda k: "spike" if bool(k) else "normal",
    )
    return out, df


# ==========================================================================
# 2. worst episodes
# ==========================================================================
def worst_episodes(df: pd.DataFrame, n: int = 100) -> dict:
    worst = df.nlargest(n, "qlike").copy()
    over = worst["ratio_rv_sigma"] < 1.0            # RV < forecast: safe, costs premium
    under = ~over                                    # RV >= forecast: the strike breaks

    dates = pd.DatetimeIndex(worst["dt"]).floor("D")
    date_counts = dates.value_counts().sort_values(ascending=False)
    weeks = pd.PeriodIndex(dates, freq="W").astype(str)

    base_under = float((df["ratio_rv_sigma"] >= 1.0).mean())

    return {
        "n": int(n),
        "qlike_threshold": float(worst["qlike"].min()),
        "mean_qlike_worst100": float(worst["qlike"].mean()),
        "mean_qlike_population": float(df["qlike"].mean()),
        "worst100_mean_vs_population_x": float(worst["qlike"].mean() / df["qlike"].mean()),
        "over_forecast_n": int(over.sum()),
        "under_forecast_n": int(under.sum()),
        "over_forecast_pct": 100.0 * float(over.mean()),
        "under_forecast_pct": 100.0 * float(under.mean()),
        "population_under_forecast_pct": 100.0 * base_under,
        "under_forecast_enrichment_x": (
            float(under.mean() / base_under) if base_under > 0 else None
        ),
        "median_rv_pct_worst_under": (
            float(100 * worst.loc[under, "RV"].median()) if under.any() else None
        ),
        "median_rv_pct_worst_over": (
            float(100 * worst.loc[over, "RV"].median()) if over.any() else None
        ),
        "median_rv_pct_population": float(100 * df["RV"].median()),
        "n_unique_calendar_dates": int(dates.nunique()),
        "n_unique_iso_weeks": int(pd.Series(weeks).nunique()),
        "top_dates": [
            {"date": str(d.date()), "n_of_worst100": int(c)}
            for d, c in date_counts.head(15).items()
        ],
    }


# ==========================================================================
# 3. excursion breach: does reality exceed the model's own 95th/99th
#    percentile of excursion, and by how much when it does
# ==========================================================================
def subsample_pred(pred: dict, idx: np.ndarray) -> dict:
    n = len(pred["sigma_med"])
    out = {}
    for k, v in pred.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def excursion_breach(model, pred: dict, df: pd.DataFrame,
                     alphas=(0.05, 0.01), tags=("p95", "p99")) -> dict:
    out = {"n": int(len(df))}
    M = {"up": df["M_up"].to_numpy(np.float64), "dn": df["M_dn"].to_numpy(np.float64)}
    for alpha, tag in zip(alphas, tags):
        for side, up in (("up", True), ("dn", False)):
            t0 = time.time()
            lev = np.asarray(model.safe_level(pred, alpha, up=up), dtype=np.float64)
            m = M[side]
            breach = m >= lev
            rate = float(breach.mean())
            rec = {
                "alpha_nominal": alpha,
                "breach_rate": rate,
                "breach_rate_vs_nominal_x": rate / alpha,
                "n_breaches": int(breach.sum()),
                "median_level_pct": float(100 * np.expm1(np.median(lev))),
                "elapsed_s": round(time.time() - t0, 1),
            }
            if breach.any():
                mb, lb = m[breach], lev[breach]
                rec["mean_excess_log"] = float((mb - lb).mean())
                rec["mean_ratio_realized_over_level"] = float((mb / np.maximum(lb, EPS)).mean())
                rec["median_realized_pct_when_breached"] = float(100 * np.expm1(np.median(mb)))
                rec["median_level_pct_when_breached"] = float(100 * np.expm1(np.median(lb)))
            out[f"{tag}_{side}"] = rec
    return out


# ==========================================================================
# 4. spike response
# ==========================================================================
def spike_response(df: pd.DataFrame, n_top_days: int = 20, cluster_gap_days: int = 3,
                   n_events: int = 6, window_before: int = 5, window_after: int = 3) -> dict:
    daily = (
        df.assign(date=pd.DatetimeIndex(df["dt"]).floor("D"))
          .groupby("date")
          .agg(realized_rv=("RV", "max"), pred_sigma=("sigma_med", "median"), n=("RV", "size"))
          .sort_index()
    )
    baseline = float(df["sigma_med"].median())        # unconditional median prediction

    top = daily.sort_values("realized_rv", ascending=False).head(n_top_days)
    top_dates = sorted(top.index)

    # cluster adjacent top-vol dates into events
    clusters, cur = [], [top_dates[0]]
    for prev, nxt in zip(top_dates, top_dates[1:]):
        if (nxt - prev).days <= cluster_gap_days:
            cur.append(nxt)
        else:
            clusters.append(cur)
            cur = [nxt]
    clusters.append(cur)
    clusters.sort(key=lambda c: -daily.loc[c, "realized_rv"].max())

    events = []
    for c in clusters[:n_events]:
        peak = daily.loc[c, "realized_rv"].idxmax()
        window = pd.date_range(peak - pd.Timedelta(days=window_before),
                               peak + pd.Timedelta(days=window_after))
        timeline = []
        for dt_ in window:
            if dt_ in daily.index:
                row = daily.loc[dt_]
                timeline.append({
                    "date": str(dt_.date()),
                    "days_from_peak": int((dt_ - peak).days),
                    "realized_rv_pct": float(100 * row["realized_rv"]),
                    "pred_sigma_pct": float(100 * row["pred_sigma"]),
                    "pred_vs_unconditional_median_x": float(row["pred_sigma"] / baseline),
                    "n_anchors": int(row["n"]),
                })
        events.append({
            "peak_date": str(peak.date()),
            "peak_realized_rv_pct": float(100 * daily.loc[peak, "realized_rv"]),
            "pred_sigma_at_peak_pct": float(100 * daily.loc[peak, "pred_sigma"]),
            "pred_vs_unconditional_median_x_at_peak": float(daily.loc[peak, "pred_sigma"] / baseline),
            "timeline": timeline,
        })

    # summary across ALL top-vol days: was T-1's prediction already elevated?
    lead1, lead2, lead3, atday = [], [], [], []
    for d in top.index:
        for lst, k in ((lead1, 1), (lead2, 2), (lead3, 3)):
            dm = d - pd.Timedelta(days=k)
            if dm in daily.index:
                lst.append(daily.loc[dm, "pred_sigma"] / baseline)
        atday.append(daily.loc[d, "pred_sigma"] / baseline)

    def _stat(lst):
        a = np.asarray(lst, dtype=np.float64)
        return {"n": int(len(a)),
                "mean_ratio_to_unconditional_median": float(a.mean()) if len(a) else None,
                "median_ratio_to_unconditional_median": float(np.median(a)) if len(a) else None}

    # lead/lag correlation: pred(t - lag) vs realized(t). lag>0 means the
    # prediction was made `lag` days BEFORE the realized value it is compared
    # to -- a real lead relationship there is "detects it coming."  lag<0
    # compares a prediction made AFTER the realized value it sits next to --
    # a strong correlation there instead of at lag>=1 is "reacts, does not
    # predict."
    corr_by_lag = {}
    for lag in range(-3, 4):
        shifted = daily["pred_sigma"].shift(lag)
        both = pd.concat([shifted, daily["realized_rv"]], axis=1).dropna()
        corr_by_lag[lag] = (
            float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) > 10 else None
        )

    return {
        "n_days": int(len(daily)),
        "unconditional_median_sigma_pct": float(100 * baseline),
        "n_top_vol_days_examined": n_top_days,
        "top_vol_days": [
            {"date": str(d.date()), "realized_rv_pct": float(100 * daily.loc[d, "realized_rv"]),
             "pred_sigma_pct": float(100 * daily.loc[d, "pred_sigma"]),
             "pred_vs_unconditional_median_x": float(daily.loc[d, "pred_sigma"] / baseline)}
            for d in top.index
        ],
        "events": events,
        "lead_ratio": {"t_minus_1": _stat(lead1), "t_minus_2": _stat(lead2),
                       "t_minus_3": _stat(lead3), "t_at_peak": _stat(atday)},
        "corr_pred_lag_vs_realized": corr_by_lag,
    }


# ==========================================================================
# printing
# ==========================================================================
def _print_table(rows: list[dict], cols: list[str], title: str):
    print(f"\n{title}")
    header = f"{'cell':>10}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    for r in rows:
        print(f"{str(r['cell']):>10}" + "".join(f"{r[c]:14.4f}" if isinstance(r[c], float)
                                                 else f"{r[c]:>14}" for c in cols))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Anatomy of NOCTUA's remaining error")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/anatomy.json"))
    ap.add_argument("--n-worst", type=int, default=100)
    ap.add_argument("--excursion-subsample", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--trailing-window-days", type=int, default=180)
    a = ap.parse_args(argv)

    t_start = time.time()
    ep, X = load_all(a.artifacts)
    served_mask, headline_mask = build_populations(ep, X)
    print(f"served population (H=19, all anchor hours, test split): {served_mask.sum():,}")
    print(f"headline population (H=19, anchor=17, test split, non-overlapping): "
          f"{headline_mask.sum():,}\n")

    model = load_model()
    print(f"shipped artifact: {model.meta.get('version')}  "
          f"has_mx={model.has_mx()}  blend_w={model.blend_w}")

    t0 = time.time()
    df_served, pred_served = score_population(model, ep, X, served_mask)
    print(f"[served] forward pass over {len(df_served):,} episodes: "
          f"{time.time()-t0:.1f}s")
    print(f"[served] pooled QLIKE = {df_served['qlike'].mean():.4f}   "
          f"pooled pinball = {df_served['pinball'].mean():.6f}   "
          f"median RV/sigma = {df_served['ratio_rv_sigma'].median():.4f}")

    # headline is a subset of served at anchor_hour == 17 -- no second forward
    # pass needed, just slice the already-computed ledger and predictive object.
    df_headline = df_served[df_served["anchor_hour"] == 17].reset_index(drop=True)
    idx_headline = np.flatnonzero((df_served["anchor_hour"] == 17).to_numpy())
    pred_headline = subsample_pred(pred_served, idx_headline)
    print(f"[headline] pooled QLIKE = {df_headline['qlike'].mean():.4f}   "
          f"median RV/sigma = {df_headline['ratio_rv_sigma'].median():.4f}   "
          f"n={len(df_headline):,}\n")

    # ---- 1. conditional error map ----------------------------------------
    thresh = trailing_p95(ep, window_days=a.trailing_window_days)
    cmap, df_served_flagged = conditional_error_map(df_served, thresh)
    cmap_headline, df_headline_flagged = conditional_error_map(df_headline, thresh)
    print("=== 1. CONDITIONAL ERROR MAP (served population) ===")
    cols = ["n", "mean_qlike", "qlike_vs_pooled_pct", "share_of_total_qlike_loss_pct",
            "median_rv_sigma_ratio"]
    _print_table(cmap["by_rv_decile"], cols, "-- by realized-vol decile --")
    _print_table(cmap["by_year"], cols, "-- by year --")
    _print_table(cmap["by_dow"], cols, "-- by day of week --")
    _print_table(cmap["by_hour"], cols, "-- by anchor hour --")
    _print_table(cmap["by_spike"], cols,
                f"-- by spike flag (coverage {cmap['spike_flag_coverage_pct']:.1f}% of episodes) --")
    print("\n=== 1b. cross-check on the independent (non-overlapping) headline population ===")
    _print_table(cmap_headline["by_year"], cols, "-- by year --")
    _print_table(cmap_headline["by_spike"], cols,
                f"-- by spike flag (coverage {cmap_headline['spike_flag_coverage_pct']:.1f}%) --")

    # ---- 2. worst episodes -------------------------------------------------
    worst = worst_episodes(df_served, n=a.n_worst)
    worst_headline = worst_episodes(df_headline, n=min(a.n_worst, max(20, len(df_headline) // 10)))
    print(f"\n=== 2. WORST {a.n_worst} EPISODES (served population) ===")
    print(json.dumps({k: v for k, v in worst.items() if k != "top_dates"}, indent=2))
    print(f"top dates: {worst['top_dates'][:8]}")
    print(f"\n-- cross-check, worst {worst_headline['n']} of the independent headline "
          f"population --")
    print(json.dumps({k: v for k, v in worst_headline.items() if k != "top_dates"}, indent=2))

    # ---- 3. excursion breach -----------------------------------------------
    rng = np.random.default_rng(a.seed)
    n = len(df_served)
    if n > a.excursion_subsample:
        sub_idx = np.sort(rng.choice(n, size=a.excursion_subsample, replace=False))
    else:
        sub_idx = np.arange(n)
    df_sub = df_served.iloc[sub_idx].reset_index(drop=True)
    pred_sub = subsample_pred(pred_served, sub_idx)

    print(f"\n=== 3. EXCURSION BREACH -- served population, subsample n={len(df_sub):,} ===")
    breach_served = excursion_breach(model, pred_sub, df_sub)
    for k, v in breach_served.items():
        if isinstance(v, dict):
            print(f"  {k:10} nominal {v['alpha_nominal']:5.2f}  breach {v['breach_rate']:7.4f} "
                  f"({v['breach_rate_vs_nominal_x']:5.2f}x nominal)  "
                  f"n_breach={v['n_breaches']:4d}  "
                  f"mean_ratio_when_breached={v.get('mean_ratio_realized_over_level', float('nan')):.3f}"
                  f"  ({v['elapsed_s']}s)")

    print(f"\n=== 3b. EXCURSION BREACH -- independent headline population, full n={len(df_headline):,} ===")
    breach_headline = excursion_breach(model, pred_headline, df_headline)
    for k, v in breach_headline.items():
        if isinstance(v, dict):
            print(f"  {k:10} nominal {v['alpha_nominal']:5.2f}  breach {v['breach_rate']:7.4f} "
                  f"({v['breach_rate_vs_nominal_x']:5.2f}x nominal)  "
                  f"n_breach={v['n_breaches']:4d}  "
                  f"mean_ratio_when_breached={v.get('mean_ratio_realized_over_level', float('nan')):.3f}"
                  f"  ({v['elapsed_s']}s)")

    # ---- 4. spike response ---------------------------------------------------
    spikes = spike_response(df_served)
    print(f"\n=== 4. SPIKE RESPONSE (served population, daily aggregates) ===")
    print(f"unconditional median predicted sigma: {spikes['unconditional_median_sigma_pct']:.3f}%")
    print(f"top {spikes['n_top_vol_days_examined']} volatility days:")
    for r in spikes["top_vol_days"][:10]:
        print(f"  {r['date']}  realized {r['realized_rv_pct']:6.2f}%  "
              f"predicted {r['pred_sigma_pct']:6.2f}%  "
              f"({r['pred_vs_unconditional_median_x']:.2f}x unconditional median)")
    print("\nlead ratio (predicted sigma N days before a top-vol day, vs the "
          "unconditional median -- 1.00 = no detectable widening):")
    for k, v in spikes["lead_ratio"].items():
        print(f"  {k:12} n={v['n']:3d}  mean={v['mean_ratio_to_unconditional_median']}  "
              f"median={v['median_ratio_to_unconditional_median']}")
    print("\ncorrelation of predicted sigma at lag L (days) vs realized RV "
          "(positive lag = prediction made BEFORE the realized value; "
          "lag 0 = same day; negative lag = prediction made AFTER):")
    for lag, c in sorted(spikes["corr_pred_lag_vs_realized"].items()):
        print(f"  lag {lag:+d}  corr={c}")
    print("\nevents (clustered top-vol windows):")
    for ev in spikes["events"]:
        print(f"  peak {ev['peak_date']}  realized {ev['peak_realized_rv_pct']:.2f}%  "
              f"predicted-at-peak {ev['pred_sigma_at_peak_pct']:.2f}% "
              f"({ev['pred_vs_unconditional_median_x_at_peak']:.2f}x)")
        for row in ev["timeline"]:
            print(f"      T{row['days_from_peak']:+d}  {row['date']}  "
                  f"realized {row['realized_rv_pct']:6.2f}%  "
                  f"predicted {row['pred_sigma_pct']:6.2f}%  "
                  f"({row['pred_vs_unconditional_median_x']:.2f}x)")

    # ---- write ---------------------------------------------------------------
    out = {
        "meta": {
            "artifact_version": model.meta.get("version"),
            "has_mx_head": bool(model.has_mx()),
            "blend_w": float(model.blend_w),
            "n_served": int(len(df_served)),
            "n_headline": int(len(df_headline)),
            "served_test_start": str(df_served["dt"].min()),
            "served_test_end": str(df_served["dt"].max()),
            "pooled_qlike_served": float(df_served["qlike"].mean()),
            "pooled_qlike_headline": float(df_headline["qlike"].mean()),
            "excursion_subsample_n": int(len(df_sub)),
            "excursion_subsample_seed": a.seed,
            "trailing_window_days": a.trailing_window_days,
            "elapsed_seconds": round(time.time() - t_start, 1),
        },
        "conditional_error_map": cmap,
        "conditional_error_map_headline_crosscheck": cmap_headline,
        "worst_episodes": worst,
        "worst_episodes_headline_crosscheck": worst_headline,
        "excursion_breach_served_subsample": breach_served,
        "excursion_breach_headline_full": breach_headline,
        "spike_response": spikes,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}  (total {time.time()-t_start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
