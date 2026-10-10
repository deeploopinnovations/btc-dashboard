#!/usr/bin/env python3
"""
Public attention / news-tone context series -> data/context/attention.parquet
(record format: model/context/SPEC.md, validated by model/context/pit.py).

Sources (free, no account or key):
  gdelt          GDELT DOC 2.0 timelines for query "bitcoin", hourly:
                 gdelt_articles, gdelt_all_articles (GDELT monitored total),
                 gdelt_tone (article-count-weighted mean tone of the hour).
  alternative_me Crypto Fear & Greed Index, daily: fear_greed.
  wikipedia      Daily en.wikipedia pageviews (all-access, user agent) of
                 "Bitcoin" and "Cryptocurrency": wiki_views_bitcoin,
                 wiki_views_cryptocurrency.

Re-runnable and resumable:
  * Every raw response is cached under CACHE (/tmp/gdelt_cache).
  * GDELT is fetched in 7-day windows (hourly resolution), newest first. A
    window whose end is more than 2 days old and that is already cached is
    never refetched. The newest windows are refetched each run.
  * GDELT allows one request per 5 s from this IP: requests are spaced 6 s
    apart. HTTP 429 or a "Please limit requests" body backs off 30, 60, 120 s;
    if it persists, GDELT fetching stops for this run and what is cached is
    kept. Unfetched windows are picked up on the next run.
  * --budget-min caps GDELT wall time (default 60).
  * Fear & Greed and Wikipedia are small and are refetched every run.

No lookahead: a row is written only when available_ts <= the fetch time of the
response it came from. Unfinished GDELT hours are dropped.

Usage:  python3 model/context/sources/attention.py [--budget-min 60] [--no-gdelt]
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # sources/ -> context/ -> model/ -> repo
sys.path.insert(0, str(ROOT / "model"))
from context import pit                             # noqa: E402

OUT = ROOT / "data" / "context" / "attention.parquet"
CACHE = Path("/tmp/gdelt_cache")
UA = "btc-dashboard-context-research/1.0 (point-in-time context dataset; polite public-API client)"

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = "bitcoin"
GDELT_MODES = ("timelinevolraw", "timelinetone")
GDELT_START = datetime(2017, 1, 1, tzinfo=timezone.utc)   # DOC API coverage start
GDELT_MIN_GAP = 6.0                                          # s between requests
GDELT_BACKOFF = (0, 30, 60, 120)                             # first try, then retries
WINDOW_DAYS = 7
SETTLED_AFTER = timedelta(days=2)                            # window older than this is final
GDELT_LAG = 900                                              # s: GDELT 15-min update cycle
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
WIKI_ARTICLES = {"Bitcoin": "wiki_views_bitcoin", "Cryptocurrency": "wiki_views_cryptocurrency"}
WIKI_START = "20150701"
WIKI_LAG = 36 * 3600   # s after D+1 00:00 UTC; the API "takes a full 24 hours to populate" (audits/attention.md)

_last_gdelt = [0.0]


class GdeltStop(Exception):
    """GDELT kept rate-limiting after all back-offs; stop GDELT for this run."""


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def http_get(url: str, timeout: int = 90) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


_deadline = [float("inf")]


def gdelt_request(url: str):
    """One GDELT call, throttled. On 429 / 'Please limit requests' it backs off
    30, 60, 120 s and then keeps retrying every 120 s until the run's time budget
    is spent (GdeltStop)."""
    waits = iter(GDELT_BACKOFF + (120,) * 10_000)
    while True:
        backoff = next(waits)
        if backoff:
            if time.time() + backoff > _deadline[0]:
                raise GdeltStop("rate-limited until the time budget ran out")
            print(f"    GDELT rate-limited; backing off {backoff}s", flush=True)
            time.sleep(backoff)
        gap = GDELT_MIN_GAP - (time.time() - _last_gdelt[0])
        if gap > 0:
            time.sleep(gap)
        _last_gdelt[0] = time.time()
        status, body = http_get(url)
        text = body.decode("utf-8", "replace")
        if status == 429 or "Please limit requests" in text[:300]:
            continue
        if status != 200:
            raise RuntimeError(f"GDELT HTTP {status}: {text[:200]}")
        return json.loads(text) if text.strip() else {}


def gdelt_path(mode: str, start: datetime) -> Path:
    return CACHE / f"gdelt_{mode}_{start:%Y%m%d}.json"


def gdelt_fetch(mode: str, start: datetime, end: datetime) -> dict:
    params = {
        "query": GDELT_QUERY,
        "mode": mode,
        "format": "json",
        "startdatetime": f"{start:%Y%m%d%H%M%S}",
        "enddatetime": f"{end - timedelta(seconds=1):%Y%m%d%H%M%S}",
    }
    url = GDELT_BASE + "?" + urllib.parse.urlencode(params)
    rec = {"url": url, "retrieved_at": int(time.time()), "body": gdelt_request(url)}
    _write_json(gdelt_path(mode, start), rec)
    return rec


def fetch_gdelt(now: datetime, budget_s: float) -> None:
    t0 = time.time()
    grid = []
    s = GDELT_START
    while s < now:
        grid.append(s)
        s += timedelta(days=WINDOW_DAYS)
    print(f"GDELT: {len(grid)} weekly windows, newest first, budget {budget_s/60:.0f} min", flush=True)
    _deadline[0] = t0 + budget_s
    for s in reversed(grid):
        e = min(s + timedelta(days=WINDOW_DAYS), now)
        settled = e <= now - SETTLED_AFTER
        for mode in GDELT_MODES:
            if settled and gdelt_path(mode, s).exists():
                continue
            if time.time() - t0 > budget_s:
                print("GDELT: time budget reached; stopping, cache kept", flush=True)
                return
            try:
                rec = gdelt_fetch(mode, s, e)
                n = len((rec["body"] or {}).get("timeline", []))
                print(f"  {mode:<15} {s:%Y-%m-%d}..{e:%Y-%m-%d} ok ({n} series)", flush=True)
            except GdeltStop as ex:
                print(f"GDELT: {ex}; stopping for this run, cache kept", flush=True)
                return
            except Exception as ex:  # keep going; window is retried next run
                print(f"  {mode} {s:%Y-%m-%d}: {ex}", flush=True)
    print("GDELT: all windows cached", flush=True)


def _gdelt_ts(s: str) -> int:
    return calendar.timegm(time.strptime(s, "%Y%m%dT%H%M%SZ"))


def _series(body: dict, name: str):
    for ser in body.get("timeline", []):
        if ser.get("series") == name:
            return ser.get("data", [])
    return None


def gdelt_rows() -> list[dict]:
    rows: list[dict] = []
    grid = []
    s = GDELT_START
    now_ts = int(time.time())
    while s < datetime.now(timezone.utc):
        grid.append(s)
        s += timedelta(days=WINDOW_DAYS)
    settle_cut = datetime.now(timezone.utc) - SETTLED_AFTER
    for s in grid:
        # Recent GDELT timelines are incomplete (the newest week came back with
        # 3 of 168 hourly bins), so only settled windows are written.
        if s + timedelta(days=WINDOW_DAYS) > settle_cut:
            continue
        vf, tf = gdelt_path("timelinevolraw", s), gdelt_path("timelinetone", s)
        if not vf.exists():
            continue
        vrec = json.loads(vf.read_text())
        vdata = _series(vrec["body"], "Article Count")
        if vdata is None:
            continue
        counts: dict[int, tuple[float, float]] = {}
        for pt in vdata:
            counts[_gdelt_ts(pt["date"])] = (float(pt["value"]), float(pt.get("norm") or 0.0))
        tones: dict[int, float] = {}
        retrieved = vrec["retrieved_at"]
        url = vrec["url"]
        if tf.exists():
            trec = json.loads(tf.read_text())
            retrieved = min(retrieved, trec["retrieved_at"])   # both responses must be known
            for pt in _series(trec["body"], "Average Tone") or []:
                tones[_gdelt_ts(pt["date"])] = float(pt["value"])
        if len(counts) >= 2:
            step = min(b - a for a, b in zip(sorted(counts), sorted(counts)[1:]))
            if step not in (900, 3600):
                print(f"  warning: window {s:%Y-%m-%d} has {step}s bins; skipped", flush=True)
                continue
        hours: dict[int, list[float]] = {}
        for ts, (c, n) in counts.items():
            h = ts - ts % 3600
            acc = hours.setdefault(h, [0.0, 0.0, 0.0, 0.0])   # count, norm, tone*count, count w/ tone
            acc[0] += c
            acc[1] += n
            if ts in tones and c > 0:
                acc[2] += tones[ts] * c
                acc[3] += c
        for h, (c, n, tw, w) in sorted(hours.items()):
            event = h + 3600
            avail = event + GDELT_LAG
            if avail > retrieved:                      # hour not yet published when we fetched
                continue
            base = dict(source="gdelt", event_ts=event, available_ts=avail, text="",
                        source_url=url, retrieved_at=retrieved, pit_quality="lagged")
            rows.append(dict(base, series="gdelt_articles", value=c))
            rows.append(dict(base, series="gdelt_all_articles", value=n))
            if w > 0:
                rows.append(dict(base, series="gdelt_tone", value=tw / w))
    return rows


def fetch_json(url: str, cache: Path, now_ts: int) -> dict:
    """Refetch a small JSON endpoint and cache the raw response with its fetch time.
    429 / 5xx are retried with back-off (the egress IP is shared)."""
    for backoff in (0, 15, 30, 60, 120):
        if backoff:
            print(f"    HTTP throttled; retrying in {backoff}s", flush=True)
            time.sleep(backoff)
        status, body = http_get(url)
        if status == 200:
            break
        if status != 429 and status < 500:
            raise RuntimeError(f"HTTP {status} from {url}")
    else:
        raise RuntimeError(f"HTTP {status} from {url} after back-off")
    rec = {"url": url, "retrieved_at": int(time.time()), "body": json.loads(body)}
    _write_json(cache, rec)
    return rec


def fng_rows() -> list[dict]:
    rec = json.loads((CACHE / "fng.json").read_text())
    rows = []
    for d in rec["body"]["data"]:
        ts = int(d["timestamp"])               # 00:00 UTC; describes previous 24 h
        avail = ts + 12 * 3600          # published ~00:00 UTC; margin per audits/attention.md
        if avail > rec["retrieved_at"]:
            continue
        rows.append(dict(source="alternative_me", series="fear_greed", event_ts=ts,
                         available_ts=avail, value=float(d["value"]), text="",
                         source_url=rec["url"], retrieved_at=rec["retrieved_at"],
                         pit_quality="lagged"))
    return rows


def wiki_rows() -> list[dict]:
    rows = []
    for article, series in WIKI_ARTICLES.items():
        rec = json.loads((CACHE / f"wiki_{article}.json").read_text())
        for it in rec["body"]["items"]:
            day = datetime.strptime(it["timestamp"][:8], "%Y%m%d").replace(tzinfo=timezone.utc)
            event = int(day.timestamp()) + 86400   # D+1 00:00 UTC (end of day D)
            avail = event + WIKI_LAG
            if avail > rec["retrieved_at"]:
                continue
            rows.append(dict(source="wikipedia", series=series, event_ts=event,
                             available_ts=avail, value=float(it["views"]), text="",
                             source_url=rec["url"], retrieved_at=rec["retrieved_at"],
                             pit_quality="lagged"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-min", type=float, default=60.0, help="GDELT wall-time budget (minutes)")
    ap.add_argument("--no-gdelt", action="store_true", help="skip GDELT (fetch the other sources only)")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    now_ts = int(now.timestamp())
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    print("fear & greed ...", flush=True)
    fetch_json(FNG_URL, CACHE / "fng.json", now_ts)
    for article in WIKI_ARTICLES:
        print(f"wikipedia {article} ...", flush=True)
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
               f"all-access/user/{urllib.parse.quote(article)}/daily/{WIKI_START}/{yesterday}")
        fetch_json(url, CACHE / f"wiki_{article}.json", now_ts)

    if not args.no_gdelt:
        fetch_gdelt(now, args.budget_min * 60)

    rows = gdelt_rows() if not args.no_gdelt else []
    rows += fng_rows() + wiki_rows()
    df = pd.DataFrame(rows, columns=pit.COLUMNS)
    df = pit.validate(df, "attention.parquet")
    df = df.sort_values(["source", "series", "event_ts"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
