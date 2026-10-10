#!/usr/bin/env python3
"""Fetch daily Bitcoin on-chain series -> data/context/onchain.parquet.

Sources (free, no key, no account):
  blockchain_com  api.blockchain.info/charts/<name>   (daily; mempool-size is 15-min)
  coinmetrics     community-api.coinmetrics.io v4 asset-metrics, btc, 1d
  halving         blockstream.info: block-height -> block (timestamps);
                  next expected halving is an estimate from the chain tip

Timing convention (see model/context/SPEC.md):
  daily point for day D (UTC)   event_ts = D+1 00:00, available_ts = D+1 06:00
  intraday point at time x      event_ts = x + its interval (gap to next point),
                                available_ts = event_ts + 1 h
  halving (past, block time T)  event_ts = T, available_ts = T + 2 h, lagged
  next halving                  not stored (a date estimated today would leak)

Revision risk: blockchain.com and Coin Metrics may recompute history, so all
rows from those two sources are tagged `revisable`.

Run:  python3 model/context/sources/onchain.py
"""
from __future__ import annotations

import calendar
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "model"))
from context import pit  # noqa: E402

OUT = ROOT / "data" / "context" / "onchain.parquet"
UA = "btc-dashboard-context/1.0 (research)"
DAY = 86400
HOUR = 3600
RETRIEVED = int(time.time())
SCHEDULE_PUBLIC = calendar.timegm((2009, 1, 3, 0, 0, 0))  # 2009-01-03 UTC

BLOCKCHAIN = ["hash-rate", "difficulty", "n-transactions", "mempool-size",
              "miners-revenue", "transaction-fees-usd",
              "estimated-transaction-volume-usd", "n-unique-addresses"]
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CM_METRICS = ["FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "AdrActCnt",
              "TxTfrValAdjUSD", "CapMVRVCur", "HashRate"]
BS = "https://blockstream.info/api"
HALVINGS = {  # height -> (fallback UTC date, used only if blockstream is down)
    210000: "2012-11-28", 420000: "2016-07-09",
    630000: "2020-05-11", 840000: "2024-04-20",
}
NEXT_HALVING_HEIGHT = 1050000
BLOCK_TARGET_S = 600


def _ssl_ctx():
    bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=bundle) if bundle else ssl.create_default_context()


def get(url: str, *, as_json: bool = True, tries: int = 4):
    last: Exception | None = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=_ssl_ctx()) as r:
                body = r.read().decode("utf-8")
            return json.loads(body) if as_json else body.strip()
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            if e.code in (429, 500, 502, 503, 504):
                last = e
                time.sleep(3 * 2 ** k)
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {body!r}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 * 2 ** k)
    raise RuntimeError(f"gave up on {url}: {last!r}")


def _frame(source, series, event, avail, value, url, quality, text=""):
    return pd.DataFrame({
        "source": source,
        "series": series,
        "event_ts": np.asarray(event, np.int64),
        "available_ts": np.asarray(avail, np.int64),
        "value": np.asarray(value, np.float64),
        "text": text,
        "source_url": url,
        "retrieved_at": RETRIEVED,
        "pit_quality": quality,
    })


# ---------------------------------------------------------------- A. blockchain.com
def fetch_blockchain(name: str, failures: list) -> pd.DataFrame | None:
    url = f"https://api.blockchain.info/charts/{name}?timespan=all&format=json&sampled=false"
    try:
        doc = get(url)
    except Exception as e:  # noqa: BLE001
        failures.append((f"blockchain_com/{name}", str(e)[:200]))
        return None
    pts = doc.get("values", [])
    if not pts:
        failures.append((f"blockchain_com/{name}", "no values"))
        return None
    x = np.array([p["x"] for p in pts], np.int64)
    y = np.array([p["y"] for p in pts], np.float64)
    if doc.get("period") == "day":
        x = x // DAY * DAY                       # day start, UTC
        event = x + DAY
        avail = event + 6 * HOUR
    else:                                       # intraday: own time + own interval
        gaps = np.diff(x)
        interval = np.append(gaps, int(np.median(gaps)))
        event = x + interval
        avail = event + HOUR
    return _frame("blockchain_com", name, event, avail, y, url, "revisable")


# ---------------------------------------------------------------- B. Coin Metrics
def _cm_time(s: str) -> int:
    return calendar.timegm(datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").timetuple())


def fetch_coinmetrics(metric: str, failures: list) -> pd.DataFrame | None:
    page = f"{CM_URL}?assets=btc&metrics={metric}&frequency=1d&page_size=10000"
    days, vals, status, urls = [], [], [], []
    try:
        while page:
            doc = get(page)
            for row in doc.get("data", []):
                raw = row.get(metric)
                if raw in (None, ""):
                    continue
                days.append(_cm_time(row["time"]) // DAY * DAY)
                vals.append(float(raw))
                status.append(str(row.get(f"{metric}-status", "")))
                urls.append(page)
            page = doc.get("next_page_url")
            time.sleep(0.5)                     # stay under community rate limits
    except Exception as e:  # noqa: BLE001
        failures.append((f"coinmetrics/{metric}", str(e)[:200]))
        if not days:
            return None
    if not days:
        failures.append((f"coinmetrics/{metric}", "no rows"))
        return None
    d = np.array(days, np.int64)
    event = d + DAY                             # end of day D
    avail = event + 6 * HOUR
    return _frame("coinmetrics", metric, event, avail, vals, urls, "revisable",
                  text=pd.Series(status).to_numpy())


# ---------------------------------------------------------------- C. halvings
def fetch_halving(failures: list) -> pd.DataFrame:
    rows = []
    for k, (h, fallback_date) in enumerate(HALVINGS.items(), start=1):
        try:
            block_hash = get(f"{BS}/block-height/{h}", as_json=False)
            url = f"{BS}/block/{block_hash}"
            t = int(get(url)["timestamp"])
            verified = "verified"
        except Exception as e:  # noqa: BLE001
            failures.append((f"halving/{h}", str(e)[:200]))
            t = calendar.timegm(datetime.strptime(fallback_date, "%Y-%m-%d").timetuple())
            url, verified = f"{BS}/block-height/{h}", "UNVERIFIED date"
            t += 2 * DAY                        # extra lag for unverified time
        rows.append(_frame(
            "halving", "halving", [t], [t + 2 * HOUR], [float(h)], url, "lagged",
            text=f"halving {k}: block height {h}, block time {verified}"))

    # The next halving's DATE is not recorded: an estimate made from today's
    # chain tip would be stored as if known in 2009 (audits/onchain.md D2).
    # Its height is protocol-fixed; derive blocks-to-go from past rows if needed.
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    failures: list = []
    frames: list[pd.DataFrame] = []

    for name in BLOCKCHAIN:
        f = fetch_blockchain(name, failures)
        if f is not None:
            frames.append(f)
    for metric in CM_METRICS:
        f = fetch_coinmetrics(metric, failures)
        if f is not None:
            frames.append(f)
    frames.append(fetch_halving(failures))

    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)
    # Drop NaN values and anything not yet published at retrieval time
    # (e.g. today's incomplete daily bar).
    keep = np.isfinite(df.value.to_numpy()) & (df.available_ts.to_numpy() <= RETRIEVED)
    df = df[keep].reset_index(drop=True)
    df = pit.validate(df, "onchain")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"wrote {OUT}: {len(df)} rows (dropped {n_raw - len(df)} unpublished/NaN)")
    print(df.groupby(["source", "series"]).agg(
        n=("value", "size"), first=("event_ts", "min"), last=("event_ts", "max")))
    if failures:
        print("FAILED:")
        for name, err in failures:
            print(f"  {name}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
