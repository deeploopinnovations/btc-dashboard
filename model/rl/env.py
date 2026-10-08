"""
rl/env.py
=====================================================================
The trading problem the agent learns, as registered in P5-rl-paper.

PAPER TRADING ONLY. Nothing here places an order or holds an exchange key.

A STEP. Slots E at 00/06/12/18 UTC, horizon H = 6. At E the agent may use only
bars BEFORE E: the hourly frame is cut at E, and a NaN placeholder bar is put
at E so serving's own `forecast` can anchor there (shown byte-identical to a
forecast with the real bar at E -- the features read rows before the anchor
only). It enters at the close of the hour before E (the price at E) and is
marked at the close of the hour starting E+5h (the price at E+6h).

THE REWARD is a mean-variance utility with stated, not measured, costs:

    u = w R - c |w - w_prev| - (gamma/2) (w R)^2

c = 10 bps per unit of exposure traded (5 bps fee + 5 bps slippage, the same
ASSUMPTION econ-voltarget used); gamma = 2 (moderate risk aversion).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

HOUR = 3600
SLOT_HOURS = (0, 6, 12, 18)
H = 6
ACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)
COST = 0.0010
GAMMA = 2.0
VT_TARGET = 0.0131            # 50% annualised over a 6-hour window
S_REF = 0.015                 # a typical 6-hour sigma, for the fixed transforms


class NotReady(RuntimeError):
    """The hour before the slot is not in the feed yet."""


class ClockAheadOfFeed(RuntimeError):
    """`now` is far beyond the newest bar: a wrong clock or a dead feed. Ticking
    anyway would mint a hold for every slot in between (audit G: a +365-day
    clock jump inserted 1,459 holds and froze decisions)."""


MAX_FEED_LAG = 24 * HOUR      # refuse to tick when the newest bar ended longer ago
VOID_AFTER = 7 * 86400        # a window whose bars never arrive is logged void after this


def utility(a: float, R: float, w_prev: float, cost: float = COST, gamma: float = GAMMA) -> float:
    return a * R - cost * abs(a - w_prev) - 0.5 * gamma * (a * R) ** 2


def slot_floor(t: int) -> int:
    """The latest slot boundary at or before t."""
    day = t - t % 86400
    return max(day + h * HOUR for h in SLOT_HOURS if day + h * HOUR <= t)


def features(sigma: float, trailing_rv: float, p_up: float, p_vol_amplify: float,
             r24: float) -> np.ndarray:
    """Fixed transforms, nothing fitted (P5-rl-paper)."""
    return np.array([
        (np.log(sigma) - np.log(S_REF)) / 0.5,
        np.log(sigma / max(trailing_rv, 1e-4)),
        p_vol_amplify - 0.5,
        p_up - 0.5,
        r24 / 0.03,
    ], dtype=np.float64)


def baseline_weights(sigma: float) -> dict:
    return {"FLAT": 0.0, "HALF": 0.5, "HOLD": 1.0,
            "VT": float(np.clip(VT_TARGET / sigma, 0.0, 1.0))}


def _row(hours: pd.DataFrame, t: int):
    ts = hours["hour_ts"].to_numpy(np.int64)
    i = int(np.searchsorted(ts, t))
    return i if i < len(ts) and ts[i] == t else None


def noctua_forecaster(model=None):
    """serve/predict.forecast at the slot, H = 6, on bars before E only."""
    from serve import predict as P
    from serve.runtime import load_model
    m = model or load_model()

    def f(h_upto: pd.DataFrame, E_slot: int) -> dict:
        ph = pd.concat([h_upto, pd.DataFrame({"hour_ts": [E_slot]})], ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()):
            pay = P.forecast(m, ph, H=H, anchor_ts=E_slot, fetch_iv=False)
        if pay["anchor_utc"] != str(pd.Timestamp(E_slot, unit="s", tz="UTC")):
            raise RuntimeError(f"forecast anchored at {pay['anchor_utc']}, not the slot")
        return {"sigma": pay["sigma_window_pct"] / 100.0,
                "trailing_rv": pay["trailing_rv_pct"] / 100.0,
                "p_up": pay["p_up"], "p_vol_amplify": pay["p_vol_amplify"]}
    return f


def inputs_at(model, hours: pd.DataFrame, E_slot: int, forecaster=None) -> dict:
    """Everything the agent sees at slot E, from bars strictly before E."""
    h = hours[hours["hour_ts"] < E_slot].reset_index(drop=True)
    last = _row(h, E_slot - HOUR)
    back = _row(h, E_slot - 25 * HOUR)
    if last is None or back is None:
        raise NotReady(f"the hour before {pd.Timestamp(E_slot, unit='s', tz='UTC')} is not in the feed")
    fc = (forecaster or noctua_forecaster(model))(h, E_slot)
    close = h["close"].to_numpy(np.float64)
    r24 = close[last] / close[back] - 1.0
    return {"E": int(E_slot), "spot": float(close[last]), "r24": float(r24), **fc,
            "x": features(fc["sigma"], fc["trailing_rv"], fc["p_up"], fc["p_vol_amplify"], r24)}


def realized(hours: pd.DataFrame, E_slot: int):
    """R over [E, E+6h): close of the hour starting E+5h over close of the hour
    before E; None until the hour starting E+5h is in the feed."""
    a, b = _row(hours, E_slot - HOUR), _row(hours, E_slot + (H - 1) * HOUR)
    if a is None or b is None:
        return None
    c = hours["close"].to_numpy(np.float64)
    return float(c[b] / c[a] - 1.0)
