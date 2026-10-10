"""
context/features.py
=====================================================================
Turns each data/context source into decision-time features for a vector of
anchor times t (Unix s). Every value comes through pit.asof / pit.next_scheduled,
so a feature at t only sees rows with available_ts < t.

Changes are computed as asof(t) vs asof(t - k): both sides are what a trader
could have read at those moments, so a revision published later never enters.

    groups()  -> {group_name: builder}   builder(ctx, t) -> DataFrame
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from context import pit

H, D = 3600, 86400


def _a(ctx, series, t, lag=0, source=None, max_age=None):
    return pit.asof(ctx, series, np.asarray(t, np.int64) - lag, source=source,
                    max_age_s=max_age)


def _logchg(ctx, series, t, k, source=None, max_age=10 * D):
    now = _a(ctx, series, t, 0, source, max_age)
    then = _a(ctx, series, t, k, source, max_age)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.maximum(now, 1e-12) / np.maximum(then, 1e-12))


def _diff(ctx, series, t, k, source=None, max_age=10 * D):
    return _a(ctx, series, t, 0, source, max_age) - _a(ctx, series, t, k, source, max_age)


def _window_sum(ctx, series, t, k, source=None):
    """Sum of values with available_ts in [t-k, t)."""
    d = ctx[(ctx.series == series) & np.isfinite(ctx.value)]
    if source is not None:
        d = d[d.source == source]
    d = d.sort_values("available_ts")
    av = d.available_ts.to_numpy(np.int64)
    c = np.concatenate([[0.0], np.cumsum(d.value.to_numpy(np.float64))])
    t = np.asarray(t, np.int64)
    hi = np.searchsorted(av, t, side="left")
    lo = np.searchsorted(av, t - k, side="left")
    out = c[hi] - c[lo]
    out[hi == 0] = np.nan                       # nothing published yet
    return out


def _sched(ctx, series, t, hold_h=24, cap_h=240.0):
    """Hours to the next scheduled event (capped) and whether one falls
    inside the holding window [t, t + hold_h)."""
    nxt = pit.next_scheduled(ctx, series, t)
    within = (nxt < hold_h).astype(float)
    return np.minimum(np.nan_to_num(nxt, nan=cap_h), cap_h) / cap_h, within


def macro_calendar(ctx, t):
    F = {}
    for s, name in [("fomc_statement", "fomc"), ("cpi_release", "cpi"), ("nfp_release", "nfp")]:
        F[f"cal_{name}_hours_to"], F[f"cal_{name}_in_hold"] = _sched(ctx, s, t)
    F["cal_fomc_target_chg_bp"] = 100 * _diff(ctx, "fomc_upper_target", t, 60 * D, max_age=400 * D)
    cpi = _a(ctx, "cpi_mom_first", t, 0, max_age=45 * D)
    F["cal_cpi_mom_recent"] = cpi
    nfp = _a(ctx, "nfp_first", t, 0, max_age=45 * D)
    F["cal_nfp_recent_k"] = nfp / 100.0
    return pd.DataFrame(F)


def macro_daily(ctx, t):
    F = {
        "mac_vix_log": np.log(_a(ctx, "VIXCLS", t, max_age=6 * D)),
        "mac_vix_chg5": _logchg(ctx, "VIXCLS", t, 5 * D),
        "mac_spx_chg1": _logchg(ctx, "SP500", t, 1 * D),
        "mac_spx_chg5": _logchg(ctx, "SP500", t, 5 * D),
        "mac_ndx_chg1": _logchg(ctx, "NASDAQCOM", t, 1 * D),
        "mac_ndx_chg5": _logchg(ctx, "NASDAQCOM", t, 5 * D),
        "mac_us10y_chg5": _diff(ctx, "DGS10", t, 5 * D),
        "mac_us2y_chg5": _diff(ctx, "DGS2", t, 5 * D),
        "mac_dxy_chg7": _logchg(ctx, "DTWEXBGS", t, 7 * D, max_age=14 * D),
        "mac_oil_chg5": _logchg(ctx, "DCOILWTICO", t, 5 * D),
        "mac_stable_chg7": _logchg(ctx, "stablecoin_mcap_total", t, 7 * D),
    }
    return pd.DataFrame(F)


def derivatives(ctx, t):
    F = {
        "der_oi_chg24": _logchg(ctx, "oi_btc", t, 1 * D, max_age=3 * H),
        "der_oi_chg7d": _logchg(ctx, "oi_btc", t, 7 * D, max_age=3 * H),
        "der_top_ls": np.log(_a(ctx, "toptrader_ls_sum", t, max_age=3 * H)),
        "der_glob_ls": np.log(_a(ctx, "global_ls_count", t, max_age=3 * H)),
        "der_taker_ls": np.log(_a(ctx, "taker_ls_vol", t, max_age=3 * H)),
        "der_bn_funding_bp": 1e4 * _a(ctx, "funding_rate_8h", t, max_age=9 * H),
    }
    if (ctx.series == "liq_long_usd").any():
        oi_usd = _a(ctx, "oi_usd", t, max_age=3 * H)
        for side in ("long", "short"):
            F[f"der_liq_{side}_24h"] = _window_sum(ctx, f"liq_{side}_usd", t, D) / oi_usd
    F["der_exp_m_hours_to"], F["der_exp_m_in_hold"] = _sched(ctx, "expiry_monthly", t)
    F["der_exp_q_hours_to"], F["der_exp_q_in_hold"] = _sched(ctx, "expiry_quarterly", t, cap_h=24 * 90.0)
    return pd.DataFrame(F)


def onchain(ctx, t):
    # Coin Metrics exchange flows/supply are NOT used: they are rewritten when
    # exchange addresses are re-tagged (audits/onchain.md D1), so today's
    # history is not what a trader saw. They stay in the file for a future
    # append-only vintage capture.
    F = {
        "onc_hash_chg30": _logchg(ctx, "hash-rate", t, 30 * D, source="blockchain_com", max_age=4 * D),
        "onc_ntx_chg7": _logchg(ctx, "n-transactions", t, 7 * D, max_age=4 * D),
        "onc_active_chg7": _logchg(ctx, "AdrActCnt", t, 7 * D, max_age=4 * D),
        "onc_fees_chg7": _logchg(ctx, "transaction-fees-usd", t, 7 * D, max_age=4 * D),
        "onc_mempool_log": np.log1p(_a(ctx, "mempool-size", t, max_age=6 * H)),
        "onc_mempool_chg24": _logchg(ctx, "mempool-size", t, D, max_age=6 * H),
        "onc_mvrv_log": np.log(_a(ctx, "CapMVRVCur", t, max_age=4 * D)),
    }
    return pd.DataFrame(F)


def attention(ctx, t):
    F = {}
    if (ctx.series == "gdelt_articles").any():
        a24 = _window_sum(ctx, "gdelt_articles", t, D)
        a30 = _window_sum(ctx, "gdelt_articles", t, 30 * D) / 30.0
        F["att_gdelt_vol_rel"] = np.log1p(a24) - np.log1p(a30)
        F["att_gdelt_tone"] = _a(ctx, "gdelt_tone", t, max_age=6 * H)
    F["att_fng"] = _a(ctx, "fear_greed", t, max_age=3 * D) / 100.0
    F["att_fng_chg7"] = _diff(ctx, "fear_greed", t, 7 * D, max_age=3 * D) / 100.0
    for s, name in (("wiki_views_bitcoin", "btc"), ("wiki_views_cryptocurrency", "crypto")):
        if (ctx.series == s).any():
            F[f"att_wiki_{name}_chg7"] = _logchg(ctx, s, t, 7 * D, max_age=4 * D)
            F[f"att_wiki_{name}_log"] = np.log1p(_a(ctx, s, t, max_age=4 * D))
    return pd.DataFrame(F)


def etf_flows(ctx, t):
    tot = _window_sum(ctx, "etf_netflow_usd_total", t, 5 * D)
    return pd.DataFrame({"etf_net5_bn": tot / 1e9, "etf_last_bn": _a(ctx, "etf_netflow_usd_total", t, max_age=4 * D) / 1e9})


GROUPS = {
    "macro_calendar": macro_calendar,
    "macro_daily": macro_daily,
    "derivatives": derivatives,
    "onchain": onchain,
    "attention": attention,
    "etf_flows": etf_flows,
}


def build(group: str, t: np.ndarray, ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features of one source group at times t, with a <group>_present flag.
    Missing values become 0 and the flag says whether anything was known."""
    if ctx is None:
        ctx = pit.load(pit.CONTEXT / f"{group}.parquet")
    F = GROUPS[group](ctx, t).replace([np.inf, -np.inf], np.nan)
    present = F.notna().any(axis=1).astype(float)
    # one missing-flag per feature that is ever missing, so a zero-filled gap
    # is never confused with a real zero (audits/derivatives.md D1)
    na = {f"{c}_missing": F[c].isna().astype(float).to_numpy() for c in F.columns if F[c].isna().any()}
    F = F.fillna(0.0)
    for c, v in na.items():
        F[c] = v
    F[f"{group}_present"] = present.to_numpy()
    return F.reset_index(drop=True)
