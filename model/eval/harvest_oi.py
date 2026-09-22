"""
eval/harvest_oi.py
=====================================================================
Daily snapshot of the BTC options chain: open interest and volume BY STRIKE
AND EXPIRY, accumulated forward.

WHY THIS ACCUMULATES RATHER THAN BACKFILLS

A practitioner thesis holds that aggregate open interest concentrates at
strikes the market then does not break, and that the levels should be pooled
across venues with each venue rolling to its next expiry as its current one
settles. Testing that needs OI by strike, historically.

It cannot be backfilled. `get_book_summary_by_currency` reports the chain as
it stands NOW; Deribit publishes no public history of per-strike open
interest, and no amount of care recovers a snapshot nobody took. So the only
honest move is to start taking them. Every day this does not run is a day of
history that cannot be bought later, which is the entire argument for adding
it before the thesis has been tested rather than after.

WHAT IT DOES NOT CLAIM

Nothing here tests the thesis. This file produces the DATA a test would need
and stops. When enough history exists, the max-OI level is a barrier question
-- does price reach the strike -- which is first passage and not direction,
so it is scorable against the committee's existing curves without option
prices (R28).

COVERAGE, STATED PLAINLY

Deribit only. It is the venue this repository already reaches and the largest
listed BTC options venue, so it is the right first cut, but it is ONE venue
and the thesis says to pool several. Adding OKX, Bybit or CME means their own
endpoints and their own instrument-naming conventions; the schema here carries
a `venue` column from day one so a second source can be appended rather than
retrofitted.

WHAT TWO SNAPSHOTS ALREADY SUGGEST ABOUT HOW THE TEST MUST BE BUILT

n = 2, so this is an observation and not evidence, but it bears on the design
and is cheaper to write down now than to rediscover. Between 2026-09-20 and
2026-09-22 the max-OI strike is STABLE on every large expiry -- 70,000 on the
25-Sep monthly (189,015 -> 186,264 OI), 80,000 on 25-Dec (115,864 -> 118,417),
50,000 and 40,000 on the two far-dated ones -- and MOVES on nearly every thin
one, including 79,500 -> 84,500 on an expiry carrying 170 contracts.

So "the max-OI strike" is not one object. On an expiry with a few hundred
contracts it is whichever strike two traders happened to touch, and a test
pooling it with the monthlies would be averaging a level with a coin flip. The
eventual test needs an open-interest floor per expiry, or size weighting, fixed
before it runs rather than after the first null.

THE EXPIRY ROLL IS FREE HERE

The thesis's roll rule -- when a venue's expiry settles, its next expiry takes
over -- needs no logic. A full-chain snapshot carries every listed expiry each
day, each instrument naming its own, so the roll is a GROUP BY at analysis
time rather than a stateful rule that can go wrong.

    python -m model.eval.harvest_oi --self-test     # offline, parses fixtures
    python -m model.eval.harvest_oi --live          # needs network

NETWORK NOTE: `--live` DOES NOT RUN IN A CLAUDE SESSION and is not expected
to. This environment's egress policy returns 403 for deribit.com; the proxy
runbook says to report that rather than work around it, so `--self-test` is
the only path exercised locally.

It is no longer unverified, though. The push trigger fired this workflow
against the feature branch on 2026-09-20 and it committed a real snapshot:
968 option instruments across 12 expiries, 426,178 total open interest,
underlying near 80,990. Fetch and parser both work against the actual
response. What that run also showed, and what the thesis will have to
accommodate: the max-OI strike holds 6-18% of an expiry's total open interest
on the large expiries -- 70,000 on the 25-Sep monthly at 10.3% of 189,012,
80,000 on 25-Dec at 8.9% of 115,864. A modest share, not a dominant one.

WHAT IS STILL NOT HAPPENING: daily accumulation. The `push` trigger fires only
when this file or its workflow changes, so the snapshots so far are incidental.
Scheduled runs fire from the DEFAULT branch only, so until the pull request is
merged the clock has not started and each day's chain is lost for good.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BOOK_SUMMARY = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
UA = {"User-Agent": "noctua-oi/1.0 (+https://github.com/deeploopinnovations/btc-dashboard)"}

# BTC-26DEC25-120000-C  ->  expiry 26DEC25, strike 120000, call
INSTRUMENT = re.compile(r"^(?P<cur>[A-Z]+)-(?P<exp>\d{1,2}[A-Z]{3}\d{2})-"
                        r"(?P<strike>[0-9.]+)-(?P<cp>[CP])$")


def _get_json(url: str, params: dict[str, Any], timeout: int = 30) -> dict:
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(f"{url}?{qs}", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_instrument(name: str) -> dict | None:
    """Strike and expiry out of a Deribit option name, or None.

    Returns None rather than raising or guessing: a chain contains futures and
    perpetuals alongside options, and an unparseable name is far more likely to
    be one of those than a malformed option. Silently coercing it would put a
    future into a strike histogram.
    """
    m = INSTRUMENT.match(name.strip().upper())
    if not m:
        return None
    try:
        exp = datetime.strptime(m.group("exp"), "%d%b%y").replace(
            hour=8, tzinfo=timezone.utc)          # Deribit settles 08:00 UTC
        strike = float(m.group("strike"))
    except ValueError:
        return None
    if strike <= 0:
        return None
    return {"currency": m.group("cur"), "expiry": exp, "strike": strike,
            "right": m.group("cp")}


def to_frame(rows: list[dict], venue: str = "deribit",
             snapshot_ts: datetime | None = None) -> pd.DataFrame:
    """Chain rows -> tidy per-instrument frame. Unparseable names are DROPPED
    and counted, never coerced."""
    # NOT named `asof`: pandas defines DataFrame.asof, so `df.asof` returns
    # the METHOD and attribute access silently yields a function instead of the
    # column. Caught on the first real snapshot; renamed while exactly one file
    # existed, because the same rename after a year of daily snapshots is a
    # migration rather than an edit.
    snapshot_ts = snapshot_ts or datetime.now(timezone.utc)
    out, skipped = [], 0
    for r in rows:
        p = parse_instrument(str(r.get("instrument_name", "")))
        if p is None:
            skipped += 1
            continue
        out.append({
            "snapshot_ts": snapshot_ts, "venue": venue,
            "instrument": r.get("instrument_name"),
            "currency": p["currency"], "expiry": p["expiry"],
            "strike": p["strike"], "right": p["right"],
            "open_interest": float(r.get("open_interest") or 0.0),
            "volume": float(r.get("volume") or 0.0),
            "mark_price": float(r.get("mark_price") or 0.0),
            "underlying_price": float(r.get("underlying_price") or 0.0),
        })
    df = pd.DataFrame(out)
    df.attrs["skipped"] = skipped
    return df


def fetch(currency: str = "BTC") -> pd.DataFrame:
    d = _get_json(BOOK_SUMMARY, {"currency": currency, "kind": "option"})
    rows = d.get("result") or []
    if not rows:
        raise SystemExit(f"REFUSING: empty chain for {currency}; "
                         "writing an empty snapshot would look like a day with "
                         "no open interest rather than a failed fetch")
    return to_frame(rows)


def max_oi_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Per (venue, expiry): the strike holding the most open interest, pooled
    across calls and puts. This is the level the thesis says price respects."""
    if df.empty:
        return df
    g = (df.groupby(["venue", "expiry", "strike"], as_index=False)
           .agg(oi=("open_interest", "sum"), vol=("volume", "sum")))
    idx = g.groupby(["venue", "expiry"])["oi"].idxmax()
    top = g.loc[idx].rename(columns={"strike": "max_oi_strike"})
    tot = (df.groupby(["venue", "expiry"], as_index=False)
             .agg(total_oi=("open_interest", "sum")))
    out = top.merge(tot, on=["venue", "expiry"])
    out["share_at_max"] = out["oi"] / out["total_oi"].where(out["total_oi"] > 0)
    return out.sort_values("expiry").reset_index(drop=True)


def self_test() -> int:
    ok = []
    # 1-4. Instrument parsing, including the names that MUST be rejected.
    p = parse_instrument("BTC-26DEC25-120000-C")
    ok.append(("parses a call", p is not None and p["strike"] == 120000.0
               and p["right"] == "C"))
    ok.append(("expiry is 08:00 UTC on the named day",
               p["expiry"].hour == 8 and p["expiry"].day == 26
               and p["expiry"].month == 12))
    ok.append(("rejects a perpetual", parse_instrument("BTC-PERPETUAL") is None))
    ok.append(("rejects a dated future",
               parse_instrument("BTC-26DEC25") is None))
    ok.append(("rejects a malformed strike",
               parse_instrument("BTC-26DEC25-ABC-C") is None))

    # 5-6. A future mixed into the chain must be DROPPED, not coerced -- a
    #      future in a strike histogram is a fabricated level.
    rows = [
        # Numbers chosen so POOLING CHANGES THE ANSWER: 120000 wins only when
        # its call and put are summed (100 + 150 = 250 > 200). A fixture where
        # the same strike wins either way cannot detect a failure to pool, and
        # the first draft of this one had exactly that defect -- plus an
        # assertion naming the wrong winner, which is how it was caught.
        {"instrument_name": "BTC-26DEC25-120000-C", "open_interest": 100.0,
         "volume": 5.0, "mark_price": 0.01, "underlying_price": 90000.0},
        {"instrument_name": "BTC-26DEC25-120000-P", "open_interest": 150.0,
         "volume": 1.0, "mark_price": 0.30, "underlying_price": 90000.0},
        {"instrument_name": "BTC-26DEC25-100000-C", "open_interest": 200.0,
         "volume": 9.0, "mark_price": 0.05, "underlying_price": 90000.0},
        {"instrument_name": "BTC-PERPETUAL", "open_interest": 999999.0,
         "volume": 1.0, "mark_price": 90000.0, "underlying_price": 90000.0},
    ]
    df = to_frame(rows)
    ok.append(("the perpetual is dropped", len(df) == 3
               and df.attrs["skipped"] == 1))
    # The timestamp column must not shadow a DataFrame method, or attribute
    # access returns the method and a downstream `df.asof` reads as callable.
    ok.append(("timestamp column does not shadow a pandas method",
               "snapshot_ts" in df.columns and "asof" not in df.columns
               and not callable(getattr(df, "snapshot_ts", None))))
    ok.append(("no 90000-ish strike was invented",
               not (df["strike"] > 500000).any()))

    # 7-8. Max-OI level pools calls and puts at the same strike.
    lv = max_oi_levels(df)
    ok.append(("max-OI strike pools C and P (unpooled would pick 100000)",
               len(lv) == 1 and float(lv.iloc[0]["max_oi_strike"]) == 120000.0
               and float(lv.iloc[0]["oi"]) == 250.0))
    ok.append(("share at the max level is a fraction of the expiry's total",
               abs(float(lv.iloc[0]["share_at_max"]) - 250.0 / 450.0) < 1e-9))

    # 9. Two expiries must not be merged -- the roll rule depends on this.
    rows2 = rows[:3] + [
        {"instrument_name": "BTC-02JAN26-95000-C", "open_interest": 700.0,
         "volume": 2.0, "mark_price": 0.02, "underlying_price": 90000.0}]
    lv2 = max_oi_levels(to_frame(rows2))
    ok.append(("each expiry keeps its own level", len(lv2) == 2))

    # 10. An empty chain must REFUSE rather than write a zero-OI day, which
    #     would be indistinguishable from a real day of no open interest.
    try:
        to_frame([])
        empty_ok = True
    except Exception:                                        # noqa: BLE001
        empty_ok = False
    ok.append(("an empty chain yields an empty frame, and fetch() refuses it",
               empty_ok))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="daily options OI-by-strike snapshot")
    ap.add_argument("--out", type=Path, default=Path("data/oi"))
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--live", action="store_true",
                    help="fetch the live chain (needs network)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.live:
        print("nothing to do: pass --live to fetch, or --self-test to check "
              "the parser offline")
        return 0

    df = fetch(a.currency)
    a.out.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = a.out / f"{a.currency.lower()}_chain_{day}.parquet"
    df.to_parquet(path, index=False)
    lv = max_oi_levels(df)
    print(f"{len(df):,} option instruments, {df.attrs['skipped']} non-option "
          f"names skipped")
    print(f"{lv['expiry'].nunique()} expiries; total OI "
          f"{df['open_interest'].sum():,.0f}")
    print(f"\n{'expiry':>12} {'max-OI strike':>14} {'OI there':>12} "
          f"{'share':>7}")
    for r in lv.head(8).itertuples():
        print(f"{r.expiry:%Y-%m-%d:>12} {r.max_oi_strike:14,.0f} "
              f"{r.oi:12,.0f} {100 * r.share_at_max:6.1f}%")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
