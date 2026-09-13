"""
eval/harvest_events.py
=====================================================================
Build an hourly ATTENTION series for BTC from GDELT, so the model can know that
something happened rather than only that the price moved.

WHY THIS EXISTS, AND WHY IT IS A CI JOB

`P2-dataset-audit` established that all 42 features derive from OHLCV and the
clock, and that the single binary `reg_post_etf` is the entire extent of the
model's world knowledge. `P2-adaptive-wrong-functional`, the DST work and the
event-window run then narrowed WHAT is missing, which is not what this file
first looked like it should fetch:

  * `P2-dst-shift-result` -- the H=1 error footprint moves by exactly one hour
    with US daylight saving (argmax lag +1, permutation p = 0.0005). The
    concentrated errors ARE event-driven.
  * `P2-event-window-result` -- an event-hour INDICATOR is worth nothing once
    it competes for capacity: at H=1 the UTC-defined one is statistically
    indistinguishable from a randomly shuffled label, and even the correctly
    aligned one does not beat base.
  * `P2-intraday-basis-result` -- the clock information is real at every horizon
    and worth almost exactly the capacity it consumes.

So TIMING IS ALREADY FREE AND NEARLY WORTHLESS. The model can learn "14:00 ET
is dangerous" from the clock it already has. What it cannot know is whether
THIS 14:00 is the one where something actually happened -- whether the
statement was a surprise, or whether someone said something unscheduled at
03:00 on a Sunday. **The missing quantity is SURPRISE MAGNITUDE, not schedule.**

That is why this harvests VOLUME AND TONE rather than headlines. Fifteen
headlines cannot label 45,000 hourly episodes; an hourly attention intensity
joins to every one of them, and "unusual attention right now" is the
machine-readable form of "something happened".

WHY CI AND NOT HERE

Identical to `eval/harvest.py`: this container's egress proxy returns 403 for
federalreserve.gov, api.stlouisfed.org, bls.gov and huggingface.co, and 000 for
GDELT. The repository's own `fetch-data` cron has been calling
`api.gdeltproject.org/api/v2/doc/doc` every 30 minutes for months from a GitHub
runner -- see `scripts/fetch-enrichment.js` -- and throwing everything away but
the latest 15 headlines. The runner is the data channel; no proxy is bypassed
and no restriction is worked around. The fetch simply happens where the network
already permits it.

THE CAUSALITY RULE, AND THE TRAP IN IT

GDELT's `seendate` is when its crawler FIRST SAW an article, which is at or
after publication. For an anchor at time t the feature reads the bucket
[t - 1h, t) only, so `feature_available_time <= prediction_time` holds (R11).

THE TRAP: GDELT revises. An article crawled late can be attributed to an
earlier bucket, so a BACKFILL query run today can return a bucket that nobody
could have observed at the time -- a look-ahead that a live accumulation cannot
have. Two defences, both required:

  1. every row carries `as_of`, the harvest timestamp, so backfilled rows are
     distinguishable from live ones forever and the leak is auditable;
  2. `--mode live` appends only the most recent complete bucket, which is
     point-in-time correct BY CONSTRUCTION. Rows written that way are the only
     ones a strict experiment should trust.

The backfill is still worth having -- it is the only way to reach 2021 -- but
it is labelled, and any result that depends on it must be reproduced on the
live-accumulated slice before it is believed.

    python -m model.eval.harvest_events --self-test
    python -m model.eval.harvest_events --backfill 2021-01-01 2026-09-01
    python -m model.eval.harvest_events --mode live
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
# Kept deliberately broad. A narrow query measures how often one phrase is
# used, not how much attention the asset is getting.
QUERY = "(bitcoin OR btc OR crypto)"
OUT = Path("data/event_history.parquet")
MAX_WINDOW_DAYS = 60          # GDELT rejects very long timeline spans
SLEEP_S = 2.0                 # courtesy rate limit for a public research API
SCHEMA = ("hour_ts", "art_volume", "tone_mean", "as_of", "provenance")


def _url(mode: str, start: datetime, end: datetime) -> str:
    q = {"query": QUERY, "mode": mode, "format": "json",
         "startdatetime": start.strftime("%Y%m%d%H%M%S"),
         "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    return f"{GDELT}?{urllib.parse.urlencode(q)}"


def _get(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def parse_timeline(payload: dict, value_key: str) -> pd.DataFrame:
    """GDELT timeline JSON -> (hour_ts, value). Tolerant of empty responses.

    Kept separate from the fetch so the self-test can exercise the parsing on a
    recorded fixture without a network call -- the parsing is where the bugs
    live, and a harvester whose only test needs the internet is untested here.
    """
    rows = []
    for series in payload.get("timeline", []):
        for pt in series.get("data", []):
            d = pt.get("date")
            if not d:
                continue
            try:
                ts = datetime.strptime(d[:15], "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc)
            except ValueError:
                continue
            rows.append((int(ts.timestamp()) // 3600 * 3600,
                         float(pt.get("value", 0.0))))
    if not rows:
        return pd.DataFrame({"hour_ts": [], value_key: []})
    df = pd.DataFrame(rows, columns=["hour_ts", value_key])
    return df.groupby("hour_ts", as_index=False)[value_key].mean()


def fetch_window(start: datetime, end: datetime, as_of: int,
                 provenance: str) -> pd.DataFrame:
    vol = parse_timeline(_get(_url("timelinevolraw", start, end)), "art_volume")
    time.sleep(SLEEP_S)
    tone = parse_timeline(_get(_url("timelinetone", start, end)), "tone_mean")
    if vol.empty and tone.empty:
        return pd.DataFrame(columns=list(SCHEMA))
    df = (vol.merge(tone, on="hour_ts", how="outer") if not vol.empty
          else tone.assign(art_volume=np.nan))
    df["as_of"] = as_of
    df["provenance"] = provenance
    return df.reindex(columns=list(SCHEMA))


# Preference order for a duplicated hour. LIVE always beats BACKFILL, whatever
# the timestamps say. Lower sorts first and is kept.
_PROV_RANK = {"live": 0, "backfill": 1}


def merge_archive(new: pd.DataFrame, path: Path = OUT) -> pd.DataFrame:
    """Append-only, and a LIVE row is never displaced by a backfilled one.

    THE CAUSALITY GUARANTEE, AND THE BUG IT SHIPPED WITH FOR ONE AFTERNOON

    The first version sorted by ["hour_ts", "as_of"] and kept the first row,
    i.e. it kept the EARLIEST HARVEST. That is wrong, and wrong in the ordinary
    workflow rather than in some exotic corner: the backfill runs ONCE at setup
    with as_of = T1, live accumulates afterwards with as_of > T1, and for every
    hour the backfill also covered the backfilled row therefore sorted first
    and WON. The append-only rule was silently preferring the revision-prone
    row over the point-in-time one -- the exact inversion of what the docstring
    promised. Found by an adversarial audit, reproduced, and fixed here.

    Correct order: prefer LIVE over BACKFILL on provenance alone; only within
    the same provenance does the earliest observation win, because two live
    harvests of one hour are two observations of the same thing and the first
    is the one nobody could have revised.
    """
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=list(SCHEMA))
    both = pd.concat([old, new], ignore_index=True)
    if both.empty:
        return both.reindex(columns=list(SCHEMA))
    both = both.assign(
        _rank=both["provenance"].map(_PROV_RANK).fillna(len(_PROV_RANK)))
    both = (both.sort_values(["hour_ts", "_rank", "as_of"])
                .drop_duplicates(subset=["hour_ts"], keep="first")
                .drop(columns="_rank"))
    return both.sort_values("hour_ts").reset_index(drop=True)


def self_test() -> int:
    ok = []
    fixture = {"timeline": [{"series": "Volume Intensity", "data": [
        {"date": "20240115T140000Z", "value": 12.5},
        {"date": "20240115T150000Z", "value": 3.0},
        {"date": "20240115T150000Z", "value": 5.0},
        {"date": "bogus", "value": 1.0},
    ]}]}
    df = parse_timeline(fixture, "art_volume")
    ok.append(("parses-timeline", len(df) == 2,
               f"{len(df)} hourly rows from 4 points (one malformed, two same hour)"))
    ok.append(("aggregates-duplicate-hours",
               abs(float(df.loc[df.hour_ts == int(datetime(2024, 1, 15, 15,
                   tzinfo=timezone.utc).timestamp()), "art_volume"].iloc[0]) - 4.0) < 1e-9,
               "two points in one hour average rather than one silently winning"))
    ok.append(("drops-malformed-date", not df.art_volume.isna().any(),
               "a malformed date is dropped, not turned into NaT or 0"))
    ok.append(("empty-is-empty", len(parse_timeline({}, "art_volume")) == 0,
               "an empty payload yields no rows rather than raising"))

    # THE APPEND-ONLY GUARANTEE. These checks MUST go through merge_archive
    # itself. The first version of this block called it and then overwrote the
    # result with an inline re-implementation, so it asserted against its own
    # copy of the logic and could not have caught the bug that was actually in
    # the function (R2, R18). An adversarial audit found that; it is the reason
    # every assertion below reads from `m`, which comes only from the function.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        arch = Path(td) / "a.parquet"

        # (i) the ORDINARY workflow, which is where the bug lived: backfill
        # first with a SMALL as_of, live afterwards with a LARGER one.
        pd.DataFrame({"hour_ts": [100, 300], "art_volume": [99.0, 3.0],
                      "tone_mean": [9.9, 0.3], "as_of": [1000, 1000],
                      "provenance": ["backfill", "backfill"]}
                     ).to_parquet(arch, index=False)
        m = merge_archive(
            pd.DataFrame({"hour_ts": [100, 200], "art_volume": [1.0, 2.0],
                          "tone_mean": [0.1, 0.2], "as_of": [2000, 2000],
                          "provenance": ["live", "live"]}), arch)
        kept = float(m.loc[m.hour_ts == 100, "art_volume"].iloc[0])
        prov = m.loc[m.hour_ts == 100, "provenance"].iloc[0]
        ok.append(("live-wins-over-earlier-backfill",
                   kept == 1.0 and prov == "live",
                   f"hour 100 keeps LIVE {kept} (as_of 2000) over the backfill's "
                   f"99.0 (as_of 1000) -- the case the old test never ran"))
        ok.append(("backfill-fills-gaps", 300 in set(m.hour_ts),
                   "a backfill still contributes hours the live feed never saw"))
        ok.append(("provenance-survives", set(m.provenance) == {"live", "backfill"},
                   "every row stays attributable to how it was obtained"))

        # (ii) clock skew: a backfill whose as_of is SMALLER than a live row's
        pd.DataFrame({"hour_ts": [400], "art_volume": [5.0], "tone_mean": [0.5],
                      "as_of": [9000], "provenance": ["live"]}
                     ).to_parquet(arch, index=False)
        m2 = merge_archive(
            pd.DataFrame({"hour_ts": [400], "art_volume": [77.0],
                          "tone_mean": [7.7], "as_of": [10],
                          "provenance": ["backfill"]}), arch)
        ok.append(("live-survives-clock-skew",
                   float(m2.iloc[0]["art_volume"]) == 5.0,
                   "a backfill with a SMALLER as_of than the live row still loses"))

        # (iii) two live observations of one hour: earliest wins
        pd.DataFrame({"hour_ts": [500], "art_volume": [11.0], "tone_mean": [1.1],
                      "as_of": [100], "provenance": ["live"]}
                     ).to_parquet(arch, index=False)
        m3 = merge_archive(
            pd.DataFrame({"hour_ts": [500], "art_volume": [22.0],
                          "tone_mean": [2.2], "as_of": [200],
                          "provenance": ["live"]}), arch)
        ok.append(("earliest-live-observation-wins",
                   float(m3.iloc[0]["art_volume"]) == 11.0,
                   "within one provenance the first observation is kept"))

    # the URL must carry the window, or a backfill silently returns 'now'
    u = _url("timelinevolraw", datetime(2021, 3, 1, tzinfo=timezone.utc),
             datetime(2021, 3, 2, tzinfo=timezone.utc))
    ok.append(("url-carries-window",
               "startdatetime=20210301000000" in u and "enddatetime=20210302000000" in u,
               "start and end are in the query string"))

    print("harvest_events self-test")
    for n, good, m_ in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m_}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    if not bad:
        print("all checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GDELT hourly attention harvester")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"))
    ap.add_argument("--mode", choices=["live"], default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    now = int(time.time())
    if a.mode == "live":
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=6)
        new = fetch_window(start, end, now, "live")
    elif a.backfill:
        s = datetime.fromisoformat(a.backfill[0]).replace(tzinfo=timezone.utc)
        e = datetime.fromisoformat(a.backfill[1]).replace(tzinfo=timezone.utc)
        parts = []
        cur = s
        while cur < e:
            nxt = min(cur + timedelta(days=MAX_WINDOW_DAYS), e)
            try:
                parts.append(fetch_window(cur, nxt, now, "backfill"))
                print(f"  {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: "
                      f"{len(parts[-1]):,} hours", flush=True)
            except Exception as exc:                              # noqa: BLE001
                print(f"  {cur:%Y-%m-%d}: FAILED {type(exc).__name__}: {exc}",
                      flush=True)
            time.sleep(SLEEP_S)
            cur = nxt
        new = (pd.concat(parts, ignore_index=True) if parts
               else pd.DataFrame(columns=list(SCHEMA)))
    else:
        ap.error("one of --self-test, --backfill or --mode live is required")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_archive(new, a.out)
    merged.to_parquet(a.out, index=False)
    live_n = int((merged.provenance == "live").sum())
    print(f"\n{len(merged):,} hourly rows -> {a.out}")
    print(f"  {live_n:,} point-in-time (live), "
          f"{len(merged)-live_n:,} backfilled and labelled as such")
    if len(merged):
        print(f"  span {pd.to_datetime(merged.hour_ts.min(), unit='s')} .. "
              f"{pd.to_datetime(merged.hour_ts.max(), unit='s')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
