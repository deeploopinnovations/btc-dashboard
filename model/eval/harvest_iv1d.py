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

API = ("https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
       "?currency=BTC&kind=option&start_timestamp={t0}&end_timestamp={t1}"
       "&count=1000&sorting=asc")
OUT_TRADES = Path("model/artifacts/iv1d_trades.parquet")
OUT_DAYS = Path("model/artifacts/iv1d.parquet")
MON = {m: i for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL",
                                    "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}


def parse_expiry(code: str) -> datetime:
    """'11MAR21' -> 2021-03-11 08:00 UTC (Deribit options expire at 08:00 UTC)."""
    i = len(code) - 5
    return datetime(2000 + int(code[-2:]), MON[code[i:i + 3]], int(code[:i]), 8,
                    tzinfo=timezone.utc)


def get(url: str, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError("unreachable")


def fetch_hour(day: datetime) -> list[dict]:
    t0 = int((day + timedelta(hours=16)).timestamp() * 1000)
    t1 = int((day + timedelta(hours=17)).timestamp() * 1000) - 1
    out, start = [], t0
    while True:
        res = get(API.format(t0=start, t1=t1))["result"]
        tr = res["trades"]
        out.extend(tr)
        if not res.get("has_more") or not tr:
            break
        start = tr[-1]["timestamp"] + 1
        time.sleep(0.12)
    seen, uniq = set(), []
    for t in out:
        if t["trade_id"] not in seen:
            seen.add(t["trade_id"]); uniq.append(t)
    return uniq


def to_frame(day: datetime, trades: list[dict]) -> pd.DataFrame:
    rows = []
    for t in trades:
        _, exp, strike, cp = t["instrument_name"].split("-")
        rows.append({"day": day.strftime("%Y-%m-%d"), "timestamp": t["timestamp"],
                     "expiry": exp, "strike": float(strike), "cp": cp, "iv": t.get("iv"),
                     "index_price": t["index_price"], "amount": t.get("amount", np.nan)})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d[np.isfinite(d["iv"].astype(float))]
    d["logm"] = np.log(d["strike"] / d["index_price"])
    exp_t = d["expiry"].map(lambda c: parse_expiry(c))
    anchor = day + timedelta(hours=17)
    d["hours_to_expiry"] = [(e - anchor).total_seconds() / 3600 for e in exp_t]
    near = sorted(d.loc[d["hours_to_expiry"] > 0, "hours_to_expiry"].unique())[:3]
    return d[d["hours_to_expiry"].isin(near) & (d["logm"].abs() <= 0.10)]


def wmedian(x, w) -> float:
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w); return float(x[np.searchsorted(c, 0.5 * c[-1])])


def summarise(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, g in trades.groupby("day"):
        atm = g[g["logm"].abs() <= 0.03]
        nd = atm[(atm["hours_to_expiry"] > 0) & (atm["hours_to_expiry"] <= 24)]
        near_h = atm["hours_to_expiry"].min() if len(atm) else np.nan
        nr = atm[atm["hours_to_expiry"] == near_h] if len(atm) else atm
        rows.append({"day": day,
                     "iv_nextday": wmedian(nd["iv"], nd["amount"].fillna(1)) if len(nd) else np.nan,
                     "n_nextday": int(len(nd)),
                     "iv_nearest": wmedian(nr["iv"], nr["amount"].fillna(1)) if len(nr) else np.nan,
                     "hours_nearest": float(near_h),
                     "n_nearest": int(len(nr))})
    return pd.DataFrame(rows)


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
