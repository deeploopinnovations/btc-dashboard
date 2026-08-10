"""
serve/fetch.py
=====================================================================
Live market data for inference.

The model needs ~22 days of history to fill the Log-HAR cascade's longest
trailing window. It does NOT need 1-minute bars to do that: the hourly
aggregator is exact on any input granularity that divides the 5-minute
realized-variance grid, which was verified bit-for-bit against the 1-minute
build. So we fetch 5-MINUTE bars and make ~7 paginated calls instead of ~32.

Primary source is Bitstamp BTC/USD -- the same venue and the same pair the
model was trained on. Venue consistency matters more than it might seem: the
model predicts BARRIER TOUCHES, and different venues wick to different
extremes, so training on Bitstamp and serving on Binance perpetuals would
quietly shift the thing being predicted. Binance is wired as a fallback but
flags itself in the output so the mismatch is never silent.
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd

STEP = 300
LOOKBACK_HOURS = 24 * 23           # 22-day HAR window plus slack
BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
BINANCE = "https://api.binance.com/api/v3/klines"
UA = {"User-Agent": "noctua/1.0 (+https://github.com/deeploopinnovations/btc-dashboard)"}


def _get(url: str, timeout: int = 25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_bitstamp(end_ts: int | None = None, lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Paginated 5-minute OHLCV from Bitstamp, oldest-first, de-duplicated."""
    end = int(end_ts or time.time())
    start = end - lookback_hours * 3600
    rows, cursor = [], start
    while cursor < end:
        url = f"{BITSTAMP}?step={STEP}&limit=1000&start={cursor}&end={end}"
        data = _get(url)["data"]["ohlc"]
        if not data:
            break
        rows.extend(data)
        last = int(data[-1]["timestamp"])
        if last <= cursor:
            break
        cursor = last + STEP
        time.sleep(0.25)  # be polite to a free public endpoint

    if not rows:
        raise RuntimeError("bitstamp returned no data")
    df = pd.DataFrame(rows)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(np.float64)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.drop_duplicates("timestamp").sort_values("timestamp", ignore_index=True)
    df["source"] = "bitstamp:btcusd"
    return df[["timestamp", "open", "high", "low", "close", "volume", "source"]]


def fetch_binance(lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    """Fallback only. Different venue AND a USDT pair -- flagged in the output."""
    end = int(time.time() * 1000)
    span = lookback_hours * 3600 * 1000
    cursor, rows = end - span, []
    while cursor < end:
        url = f"{BINANCE}?symbol=BTCUSDT&interval=5m&limit=1000&startTime={cursor}"
        data = _get(url)
        if not data:
            break
        rows.extend(data)
        nxt = int(data[-1][0]) + STEP * 1000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.2)

    if not rows:
        raise RuntimeError("binance returned no data")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = (df["timestamp"].astype(np.int64) // 1000)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(np.float64)
    df = df.drop_duplicates("timestamp").sort_values("timestamp", ignore_index=True)
    df["source"] = "binance:btcusdt (FALLBACK - venue mismatch)"
    return df


def fetch_bars(lookback_hours: int = LOOKBACK_HOURS) -> pd.DataFrame:
    errors = []
    for fn in (fetch_bitstamp, fetch_binance):
        try:
            df = fetch_bars_validate(fn(lookback_hours=lookback_hours))
            return df
        except Exception as e:  # noqa: BLE001 - report every source that failed
            errors.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    raise RuntimeError("all data sources failed -> " + " | ".join(errors))


def fetch_bars_validate(df: pd.DataFrame, min_hours: int = 22 * 24) -> pd.DataFrame:
    """Refuse to serve a forecast from data that cannot support the features."""
    span_h = (df.timestamp.iloc[-1] - df.timestamp.iloc[0]) / 3600.0
    if span_h < min_hours:
        raise RuntimeError(f"only {span_h:.1f}h of history, need >= {min_hours}h")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("non-positive prices in feed")
    gaps = np.diff(df.timestamp.to_numpy())
    if (gaps > 6 * STEP).sum() > span_h * 0.02:
        raise RuntimeError("feed has too many gaps to build realized measures")
    df = df.copy()
    df["bad_print"] = False
    return df
