#!/usr/bin/env python3
"""Derivatives positioning context source, hourly and point-in-time.

Writes data/context/derivatives.parquet (SPEC.md record contract) from:

  binance_metrics  BTCUSDT USD-M 5-minute "metrics" daily zips. Per UTC hour the
                   LAST snapshot is kept. event_ts = that snapshot's create_time,
                   available_ts = event_ts + 300 s, pit_quality "lagged".
  binance_funding  BTCUSDT USD-M funding-rate monthly zips. event_ts =
                   available_ts = funding time, pit_quality "exact".
  deribit_expiry   Deterministic calendar: Deribit options expire 08:00 UTC on the
                   last Friday of each month (expiry_monthly) and of Mar/Jun/Sep/Dec
                   (expiry_quarterly). available_ts = max(event - 90 d, Jan 1 of the
                   event year), pit_quality "scheduled".
  liquidations     Binance liquidationSnapshot archive. Probed; skipped if empty.

Re-runnable: archive files are cached under /tmp/binance_cache and reused.
Rows whose available_ts is later than the retrieval time (plus the validator's
one-day slack) are withheld; re-running after that date adds them.

Usage: python3 -I model/context/sources/derivatives.py
"""
from __future__ import annotations

import datetime as dt
import io
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))
from context import pit  # noqa: E402

CACHE = Path("/tmp/binance_cache")
OUT = ROOT / "data" / "context" / "derivatives.parquet"
BASE = "https://data.binance.vision"
UA = "btc-dashboard-context/1.0"
NOW = int(time.time())
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
METRICS_START = dt.date(2020, 9, 1)     # 2020-08-31 is 404; first existing day
FUNDING_START = (2019, 9)               # 2019-09..2019-12 probed; 2020-01 first hit
LAST_DAY = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
EXPIRY_YEARS = range(2019, 2028)
SNAPSHOT_AVAIL_LAG = 300                # snapshot at T is public by T+5 min
RETRIEVED_SLACK = 86400                 # pit.validate allows available <= retrieved + 1 d


def fetch(url: str, dest: Path):
    """True if the file is on disk, None if the archive returns 404, else an error string."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = repr(e)
        time.sleep(2 ** attempt)
    return f"error: {err}"


def read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(io.BytesIO(f.read()))


def ts_seconds(series: pd.Series) -> np.ndarray:
    return ((series - EPOCH) // pd.Timedelta(seconds=1)).astype("int64").to_numpy()


def long_rows(source, series, ts, avail, values, url, retrieved, quality) -> pd.DataFrame:
    return pd.DataFrame({
        "source": source, "series": series,
        "event_ts": ts.astype("int64"), "available_ts": avail.astype("int64"),
        "value": np.asarray(values, dtype="float64"), "text": "",
        "source_url": url, "retrieved_at": int(retrieved), "pit_quality": quality,
    })


# ---------- A. binance_metrics -------------------------------------------------

METRIC_COLS = {
    "oi_btc": "sum_open_interest",
    "oi_usd": "sum_open_interest_value",
    "toptrader_ls_count": "count_toptrader_long_short_ratio",
    "toptrader_ls_sum": "sum_toptrader_long_short_ratio",
    "global_ls_count": "count_long_short_ratio",
    "taker_ls_vol": "sum_taker_long_short_vol_ratio",
}


def metrics_day(day: dt.date):
    name = f"BTCUSDT-metrics-{day.isoformat()}"
    url = f"{BASE}/data/futures/um/daily/metrics/BTCUSDT/{name}.zip"
    path = CACHE / "metrics" / f"{name}.zip"
    st = fetch(url, path)
    if st is not True:
        return day, st, None
    raw = read_zip_csv(path)
    t = pd.to_datetime(raw["create_time"], format="%Y-%m-%d %H:%M:%S", utc=True)
    df = pd.DataFrame({"ts": ts_seconds(t)})
    for c in METRIC_COLS.values():
        df[c] = raw[c].astype("float64").to_numpy()
    # The archive repeats each 5-minute row; duplicates are identical, keep one.
    df = df.drop_duplicates("ts", keep="last").sort_values("ts")
    # Last 5-minute snapshot inside each UTC hour [h, h+1h).
    df["hb"] = df["ts"] // 3600
    last = df.groupby("hb", sort=True).tail(1)
    retrieved = int(path.stat().st_mtime)
    parts = []
    for series, col in METRIC_COLS.items():
        ok = last[col].notna().to_numpy()
        s = last.loc[ok]
        parts.append(long_rows("binance_metrics", series, s["ts"].to_numpy(),
                               s["ts"].to_numpy() + SNAPSHOT_AVAIL_LAG,
                               s[col].to_numpy(), url, retrieved, "lagged"))
    return day, True, pd.concat(parts, ignore_index=True)


# ---------- B. binance_funding -------------------------------------------------

def funding_month(y: int, m: int):
    name = f"BTCUSDT-fundingRate-{y:04d}-{m:02d}"
    url = f"{BASE}/data/futures/um/monthly/fundingRate/BTCUSDT/{name}.zip"
    path = CACHE / "funding" / f"{name}.zip"
    st = fetch(url, path)
    if st is not True:
        return (y, m), st, None, None
    raw = read_zip_csv(path)
    ts = np.round(raw["calc_time"].to_numpy(np.float64) / 1000.0).astype("int64")
    df = long_rows("binance_funding", "funding_rate_8h", ts, ts,
                   raw["last_funding_rate"].to_numpy(), url, int(path.stat().st_mtime),
                   "exact")
    interval = raw["funding_interval_hours"].value_counts().to_dict()
    return (y, m), True, df, interval


# ---------- C. deribit_expiry --------------------------------------------------

def last_friday(y: int, m: int) -> dt.date:
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    d = nxt - dt.timedelta(days=1)
    while d.weekday() != 4:  # Monday=0 ... Friday=4
        d -= dt.timedelta(days=1)
    return d


def expiry_calendar():
    rows, withheld = [], 0
    for y in EXPIRY_YEARS:
        for m in range(1, 13):
            d = last_friday(y, m)
            event = int(dt.datetime(d.year, d.month, d.day, 8, tzinfo=dt.timezone.utc).timestamp())
            jan1 = int(dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp())
            avail = max(event - 90 * 86400, jan1)  # 90 d lead, never before Jan 1 of event year
            series = ["expiry_monthly"] + (["expiry_quarterly"] if m in (3, 6, 9, 12) else [])
            if avail > NOW + RETRIEVED_SLACK:
                withheld += len(series)
                continue
            for s in series:
                rows.append(long_rows("deribit_expiry", s, np.array([event]), np.array([avail]),
                                      np.array([1.0]), "scheduled-calendar:deterministic",
                                      NOW, "scheduled"))
    return pd.concat(rows, ignore_index=True), withheld


# ---------- D. liquidations ----------------------------------------------------

def probe_liquidations() -> int:
    """Number of objects under the liquidationSnapshot prefix (0 = not in the archive)."""
    prefix = "data/futures/um/daily/liquidationSnapshot/BTCUSDT/"
    url = ("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
           f"?prefix={prefix}&max-keys=1000")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}),
                                    timeout=60) as r:
            body = r.read().decode("utf-8", "replace")
        return len(re.findall(r"<Key>[^<]*</Key>", body))
    except (urllib.error.URLError, OSError) as e:
        print(f"liquidations: listing failed ({e!r}); skipped")
        return 0


# ---------- main ---------------------------------------------------------------

def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    days = []
    d = METRICS_START
    while d <= LAST_DAY:
        days.append(d)
        d += dt.timedelta(days=1)

    frames, missing_metrics, errors = [], [], []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for day, st, df in ex.map(metrics_day, days):
            if st is None:
                missing_metrics.append(day.isoformat())
            elif st is not True:
                errors.append(f"metrics {day}: {st}")
            else:
                frames.append(df)
    metrics = pd.concat(frames, ignore_index=True)
    print(f"binance_metrics: {len(days)} days requested, "
          f"{len(days) - len(missing_metrics) - len(errors)} parsed, "
          f"404 on {len(missing_metrics)}, errors {len(errors)}")

    fmonths = [(y, m) for y in range(FUNDING_START[0], LAST_DAY.year + 1)
               for m in range(1, 13)
               if (y, m) >= FUNDING_START and (y, m) <= (LAST_DAY.year, LAST_DAY.month)]
    fframes, missing_f, intervals = [], [], {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for ym, st, df, interval in ex.map(lambda p: funding_month(*p), fmonths):
            if st is None:
                missing_f.append(f"{ym[0]}-{ym[1]:02d}")
            elif st is not True:
                errors.append(f"funding {ym}: {st}")
            else:
                fframes.append(df)
                for k, v in (interval or {}).items():
                    intervals[k] = intervals.get(k, 0) + v
    funding = pd.concat(fframes, ignore_index=True)
    print(f"binance_funding: {len(fmonths)} months requested, 404 on {missing_f}, "
          f"funding_interval_hours counts {intervals}")

    expiry, withheld = expiry_calendar()
    print(f"deribit_expiry: {len(expiry)} rows; {withheld} withheld (available_ts after "
          f"retrieval date)")

    n_liq = probe_liquidations()
    print(f"liquidations: {n_liq} objects under liquidationSnapshot/BTCUSDT"
          + ("; skipped (no archive files)" if n_liq == 0 else
             "; files present but parser not implemented; skipped"))

    out = pd.concat([metrics, funding, expiry], ignore_index=True)
    out = pit.validate(out, "derivatives")
    out["value"] = out["value"].astype("float32").astype("float64")  # float32-valued
    out = out[np.isfinite(out.value)]
    pit.validate(out, "derivatives")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False, compression="zstd")
    print(f"wrote {OUT} ({len(out)} rows, {OUT.stat().st_size / 1e6:.1f} MB)")
    if errors:
        print("ERRORS (re-run to retry):")
        for e in errors:
            print("  " + e)
    if missing_metrics:
        print(f"metrics days missing (404): {missing_metrics}")


if __name__ == "__main__":
    main()
