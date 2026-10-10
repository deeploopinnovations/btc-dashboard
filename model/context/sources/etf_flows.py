"""
context/sources/etf_flows.py
=====================================================================
US spot Bitcoin ETF daily net flows (US$m on Farside) -> data/context/etf_flows.parquet.

Source: https://farside.co.uk/bitcoin-etf-flow-all-data/ (the "All Data" table,
launch 2024-01-11 onward). Farside returns 403 to plain HTTP clients from some
networks, so the script runs in one of two modes:

  1. Live:      python3 etf_flows.py            (urllib with a browser UA)
  2. Snapshot:  python3 etf_flows.py --snapshot FILE
                FILE is either the raw HTML, or a JSON object with a
                "markdown" key (the Firecrawl scrape format). Use this when
                the live fetch is refused; pass --snapshot-url to record the
                URL the snapshot was read from.

Conventions (SPEC.md rules 1 and 4, with the task's ETF convention):
  event_ts      = D 21:00 UTC  (US trading day D)
  available_ts  = D+1 12:00 UTC (the morning after publication)
  pit_quality   = "revisable"  (no capture time; Farside revisions unverified)
  retrieved_at  = fetch time
Values are converted from US$m to US$. Parenthesised numbers are negative.
A "-" cell (fund not yet trading, or no data) produces no row for that fund.
Rows whose Total is "-" (market holiday rows with no flows) are skipped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as htmlmod
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))          # model/ -> import context.pit
from context import pit  # noqa: E402

URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
OUT = pit.CONTEXT / "etf_flows.parquet"
SOURCE = "farside"
DATE_RE = re.compile(r"^\s*(\d{1,2}) ([A-Za-z]{3}) (\d{4})\s*$")
NUM_RE = re.compile(r"^\(?-?[\d,]+\.?\d*\)?$")


def fetch_live(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def load_snapshot(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json" or raw.lstrip().startswith("{"):
        obj = json.loads(raw)
        if isinstance(obj, dict) and "markdown" in obj:
            return obj["markdown"]
    return raw


def table_rows(text: str) -> list[list[str]]:
    """Return table rows as cell lists from either markdown pipes or HTML <tr>."""
    if "<table" in text.lower():
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
            rows.append([htmlmod.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
        return rows
    rows = []
    for line in text.splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def parse_num(cell: str) -> float | None:
    s = cell.strip().replace(",", "")
    if s in ("", "-", "–", "—"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if not NUM_RE.match(s.replace("-", "", 1) if s.startswith("-") else s):
        return None
    v = float(s)
    return -v if neg else v


def parse(text: str) -> dict[str, dict[dt.date, float]]:
    rows = table_rows(text)
    header_idx = next((i for i, r in enumerate(rows)
                       if r and r[0].lower() == "date" and "Total" in r), None)
    if header_idx is None:
        raise ValueError("Farside table header (Date ... Total) not found")
    header = [h.strip() for h in rows[header_idx]]
    out: dict[str, dict[dt.date, float]] = {}
    n_days = 0
    for r in rows[header_idx + 1:]:
        if len(r) < len(header):
            continue
        m = DATE_RE.match(r[0])
        if not m:
            continue                       # footers, notes, blank rows
        day = dt.datetime.strptime(" ".join(m.groups()), "%d %b %Y").date()
        n_days += 1
        cells = {("total" if name.lower() == "total" else name.upper()): parse_num(cell)
                 for name, cell in zip(header[1:], r[1:len(header)])}
        # market holidays appear as a 0.0 total with no fund cells: no trading day
        if all(v is None for k, v in cells.items() if k != "total"):
            continue
        for key, v in cells.items():
            if v is not None:
                out.setdefault(key, {})[day] = v * 1e6   # US$m -> US$
    if n_days == 0:
        raise ValueError("no dated rows parsed from Farside table")
    return out


def to_frame(series_days: dict[str, dict[dt.date, float]], retrieved_at: int,
             url: str) -> pd.DataFrame:
    recs = []
    for key, days in series_days.items():
        name = "etf_netflow_usd_total" if key == "total" else f"etf_netflow_usd_{key}"
        for day, v in days.items():
            ev = int(dt.datetime(day.year, day.month, day.day, 21, tzinfo=dt.timezone.utc).timestamp())
            av = ev + 15 * 3600                 # D+1 12:00 UTC
            recs.append({"source": SOURCE, "series": name, "event_ts": ev,
                         "available_ts": av, "value": v, "text": "",
                         "source_url": url, "retrieved_at": retrieved_at,
                         "pit_quality": "revisable"})   # Farside revisions unverified
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, help="saved Farside HTML or Firecrawl JSON")
    ap.add_argument("--snapshot-url", default=URL, help="URL the snapshot was read from")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.snapshot:
        text, src_url = load_snapshot(args.snapshot), args.snapshot_url
    else:
        try:
            text, src_url = fetch_live(URL), URL
        except Exception as e:  # 403 from some networks
            sys.exit(f"live fetch failed ({e}); rerun with --snapshot FILE")

    retrieved_at = int(time.time())
    df = to_frame(parse(text), retrieved_at, src_url)
    df = pit.validate(df, "etf_flows")
    df = df.sort_values(["series", "event_ts"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
