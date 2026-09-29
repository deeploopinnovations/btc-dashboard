"""
eval/harvest_iv1d.py
=====================================================================
Harvest the market's own forecast for the product window: Deribit BTC option
TRADES in the hour BEFORE each 17:00 UTC production anchor, from the public
history API (history.deribit.com), and summarise the at-the-money implied vol
of the option expiring at 08:00 UTC the next morning (15 h out -- the product
window is 19 h).

WHY: the only implied vol NOCTUA ever saw is the 30-day DVOL index
(P3-exogenous-dvol, E2c; rejected on the served base, P4-iv-served-pe-result).
A next-day expiry prices the next night specifically -- scheduled events
included. This harvester only COLLECTS; it reads no outcome. Any test of the
feature is pre-registered separately (research/ledger.json).

WHAT IS STORED
  model/artifacts/iv1d_trades.parquet   every option trade in [16:00, 17:00)
                                        UTC with |log(K/S)| <= 0.10 on the
                                        three nearest expiries (reproducibility)
  model/artifacts/iv1d.parquet          one row per day: the next-day expiry's
                                        ATM IV (|log K/S| <= 0.03), amount-
                                        weighted median, trade count, and the
                                        nearest-expiry fallback, all strictly
                                        before the anchor

NOT AN OPTIONS P&L: trade IVs are used as a forecast input only. Historical
tradable quotes, spreads and execution are not available, and no P&L is built.

    python -m model.eval.harvest_iv1d --start 2019-12-01 --end 2026-09-28
    python -m model.eval.harvest_iv1d --summarise-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua.iv1d import fetch_hour, summarise, to_frame  # noqa: E402  (one implementation)

OUT_TRADES = Path("model/artifacts/iv1d_trades.parquet")
OUT_DAYS = Path("model/artifacts/iv1d.parquet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="harvest pre-anchor option-trade IV")
    ap.add_argument("--start", default="2019-12-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(OUT_TRADES) if OUT_TRADES.exists() else pd.DataFrame()
    if not a.summarise_only:
        done = set(have["day"]) if len(have) else set()
        d0 = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
        d1 = (datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end
              else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
              - timedelta(days=1))
        from concurrent.futures import ThreadPoolExecutor
        todo, day = [], d0
        while day <= d1:
            if day.strftime("%Y-%m-%d") not in done:
                todo.append(day)
            day += timedelta(days=1)
        frames = [have]
        # four days in flight (each paginates sequentially): ~10 requests/s at
        # most, well inside Deribit's public limits
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i in range(0, len(todo), 100):
                chunk = todo[i:i + 100]
                frames.extend(pool.map(lambda dd: to_frame(dd, fetch_hour(dd)), chunk))
                pd.concat(frames, ignore_index=True).to_parquet(OUT_TRADES, index=False)
                print(f"  {chunk[-1]:%Y-%m-%d}: {i + len(chunk)}/{len(todo)} days fetched", flush=True)
        have = pd.concat(frames, ignore_index=True)
        have.to_parquet(OUT_TRADES, index=False)
    s = summarise(have)
    s.to_parquet(OUT_DAYS, index=False)
    cov = s["iv_nextday"].notna()
    print(f"{len(s)} days; next-day ATM IV available on {int(cov.sum())} "
          f"({100 * cov.mean():.1f}%); nearest-expiry fallback on "
          f"{int(s['iv_nearest'].notna().sum())}")
    print(s.assign(year=s["day"].str[:4]).groupby("year")["iv_nextday"]
          .apply(lambda x: f"{x.notna().mean():.0%} covered").to_string())
    print(f"wrote {OUT_TRADES} and {OUT_DAYS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
