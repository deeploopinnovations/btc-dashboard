"""
noctua/iv1d.py
=====================================================================
The market's forecast for the product night, computed ONE way for research,
serving and the forward holdout (P4-iv1d-anchor, P4-iv1d-posthoc,
P4-iv1d-proxy-control): the at-the-money implied vol of the Deribit BTC option
expiring 08:00 UTC the next morning, from option TRADES in the hour BEFORE the
17:00 UTC anchor (history.deribit.com public API). Moved verbatim from
eval/harvest_iv1d.py, which now imports it; no research imports here.

Forecast input only -- never an options P&L.
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

API = ("https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
       "?currency=BTC&kind=option&start_timestamp={t0}&end_timestamp={t1}"
       "&count=1000&sorting=asc")
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




def iv_nextday_for(day: datetime) -> float | None:
    """Live or historical: the next-day ATM IV for the 17:00 anchor of `day`
    (UTC midnight), or None when no qualifying trade exists."""
    fr = to_frame(day, fetch_hour(day))
    if fr.empty:
        return None
    s = summarise(fr)
    v = float(s["iv_nextday"].iloc[0]) if len(s) else float("nan")
    return v if np.isfinite(v) and v > 0 else None


def log_hourly(iv_pct: float) -> float:
    """Annualised implied vol in percent -> log hourly vol (the anchor's unit)."""
    return float(np.log(iv_pct / 100.0 / np.sqrt(8760.0)))
