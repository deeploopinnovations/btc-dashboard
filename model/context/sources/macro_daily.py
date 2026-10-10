#!/usr/bin/env python3
"""Daily cross-market macro context: FRED closes + DefiLlama stablecoin supply.

Re-runnable:  python3 model/context/sources/macro_daily.py
Writes:       data/context/macro_daily.parquet   (SPEC.md record contract)

Publication-lag rules (see SPEC.md rules 3-5):
  FRED daily closes (DGS10, DGS2, VIXCLS, SP500, NASDAQCOM, DFF, T10YIE,
      DCOILWTICO): event = D 21:00 UTC, available = D+1 12:00 UTC.
      pit_quality "lagged".
  DTWEXBGS (H.10, weekly release): event = D 21:00 UTC, available = the first
      Monday strictly after D at 21:30 UTC. FRED's release page shows the H.10
      update at 3:16 pm CDT (20:16 UTC) on Monday; 21:30 UTC covers the EST
      release (4:15 pm EST = 21:15 UTC). pit_quality "revisable".
  BAMLH0A0HYM2 (ICE BofA HY OAS): FRED's release page shows the update the
      morning after the close (about 8 am CDT on D+1, i.e. 13:00 UTC).
      available = D+2 12:00 UTC for margin. pit_quality "lagged".
  DefiLlama daily stablecoin supply: event = D+1 00:00 UTC (end of day D),
      available = D+1 06:00 UTC. pit_quality "revisable" (backfills).

Rows whose event_ts lies after the fetch time are dropped (they are not closed
yet and must not be emitted). Missing FRED values ("." ) are dropped.
"""
from __future__ import annotations

import csv
import io
import json
import os
import ssl
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parents[2]          # .../model
sys.path.insert(0, str(MODEL_DIR))
from context import pit              # noqa: E402  (validator + paths)

START = "2012-01-01"
OUT = pit.CONTEXT / "macro_daily.parquet"
UA = "Mozilla/5.0 (btc-dashboard point-in-time context fetcher)"
DAY = 86400
UTC = timezone.utc

FRED_DAILY = ["DGS10", "DGS2", "VIXCLS", "SP500", "NASDAQCOM", "DFF",
              "T10YIE", "DCOILWTICO"]
FRED_WEEKLY_H10 = ["DTWEXBGS"]
FRED_HY = ["BAMLH0A0HYM2"]
LLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
LLAMA_USDT_URL = "https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1"


def _ssl_context() -> ssl.SSLContext:
    """Verify TLS normally; use the sandbox proxy CA bundle when one is provided."""
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if not cafile and Path("/root/.ccr/ca-bundle.crt").exists():
        cafile = "/root/.ccr/ca-bundle.crt"
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


def fetch(url: str, tries: int = 4) -> tuple[bytes, int]:
    """Return (body, retrieved_at_unix). Retries with backoff."""
    ctx = _ssl_context()
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
                body = r.read()
            return body, int(time.time())
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(2 ** k)
    raise RuntimeError(f"fetch failed after {tries} tries: {url}: {last}")


def utc_ts(d: date, hh: int = 0, mm: int = 0) -> int:
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=UTC).timestamp())


def next_monday_after(d: date) -> date:
    """First Monday strictly after d (H.10 for week ending Fri is out the Monday after)."""
    days = (7 - d.weekday()) % 7
    return d + timedelta(days=days or 7)


def fred_rows(series: str, retrieved_at: int, url: str, body: bytes) -> list[dict]:
    rows = []
    reader = csv.DictReader(io.StringIO(body.decode("utf-8")))
    for rec in reader:
        raw = rec.get("observation_date") or rec.get("DATE")
        val = rec.get(series)
        if not raw or val is None or val.strip() in (".", ""):
            continue                               # FRED missing marker
        d = date.fromisoformat(raw)
        if d < date.fromisoformat(START):
            continue
        event = utc_ts(d, 21, 0)
        if series in FRED_WEEKLY_H10:
            # Tuesday after: H.10 slips to Tuesday when Monday is a holiday
            # (audits/macro_daily.md D2)
            avail = utc_ts(next_monday_after(d) + timedelta(days=1), 21, 30)
            quality = "revisable"
        elif series == "DCOILWTICO":
            # EIA spot prices reach FRED 2-8 days late (ALFRED, audit D1)
            avail = utc_ts(d + timedelta(days=9), 12, 0)
            quality = "lagged"
        elif series == "DFF":
            # published the next business day; weekend values arrive Monday (D5)
            nb = d + timedelta(days=1)
            while nb.weekday() >= 5:
                nb += timedelta(days=1)
            avail = utc_ts(nb, 14, 0)
            quality = "lagged"
        elif series in FRED_HY:
            avail = utc_ts(d + timedelta(days=2), 12, 0)
            quality = "lagged"
        else:
            avail = utc_ts(d + timedelta(days=1), 12, 0)
            quality = "lagged"
        if avail > retrieved_at:
            continue                               # not public yet under our rule
        rows.append(dict(source="fred", series=series, event_ts=event,
                         available_ts=avail, value=float(val), text="",
                         source_url=url, retrieved_at=retrieved_at,
                         pit_quality=quality))
    return rows


def llama_rows(series: str, url: str, body: bytes, retrieved_at: int) -> list[dict]:
    rows = []
    for rec in json.loads(body.decode("utf-8")):
        ts = int(rec["date"])                      # unix seconds at 00:00 UTC of day D
        d = datetime.fromtimestamp(ts - ts % DAY, tz=UTC).date()
        if d < date.fromisoformat(START):
            continue
        usd = (rec.get("totalCirculatingUSD") or {}).get("peggedUSD")
        if usd is None:
            continue
        event = utc_ts(d + timedelta(days=1), 0, 0)  # end of day D
        rows.append(dict(source="defillama", series=series, event_ts=event,
                         available_ts=event + 6 * 3600, value=float(usd), text="",
                         source_url=url, retrieved_at=retrieved_at,
                         pit_quality="revisable"))
    return rows


def main() -> None:
    all_rows: list[dict] = []
    fred_ids = FRED_DAILY + FRED_WEEKLY_H10 + FRED_HY
    for sid in fred_ids:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
               f"&cosd={START}")
        body, ts = fetch(url)
        rows = fred_rows(sid, ts, url, body)
        print(f"fred {sid}: {len(rows)} rows", file=sys.stderr)
        all_rows += rows

    for series, url in [("stablecoin_mcap_total", LLAMA_URL),
                        ("stablecoin_mcap_usdt", LLAMA_USDT_URL)]:
        body, ts = fetch(url)
        rows = llama_rows(series, url, body, ts)
        print(f"defillama {series}: {len(rows)} rows", file=sys.stderr)
        all_rows += rows

    df = pd.DataFrame(all_rows)
    # Drop anything whose event had not yet occurred at fetch time.
    df = df[df.event_ts <= df.retrieved_at].copy()
    df = df.sort_values(["source", "series", "event_ts"]).reset_index(drop=True)
    df = pit.validate(df, OUT.name)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
