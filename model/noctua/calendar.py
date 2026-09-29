"""
noctua/calendar.py
=====================================================================
The TRUE weekend fraction of a forecast window (P4-weekend-bug,
P4-weekend-fix-result).

features.py's `cal_weekend_frac` counts FRIDAY and SATURDAY (its weekday
offset is +4 where the epoch, a Thursday, needs +3). The shipped network was
trained on that column, so it stays as it is (tests/test_weekend_column.py).
This module is the one place the correct fraction is computed -- by the
artifact increment (noctua/add_weekend_anchor.py) and by serving
(serve/runtime.py) -- and it is checked against pandas' calendar, never
against a re-implementation of its own formula.

Two equivalent forms:
  weekend_frac(anchor_ts, H)                from timestamps (fitting)
  weekend_frac_from_clock(dow, hour, H)     from the anchor's weekday and hour,
                                            which serving recovers from the
                                            cal_dow_* / cal_hour_* features
"""
from __future__ import annotations

import numpy as np

HOUR = 3600
WEEKEND = (5, 6)              # Saturday, Sunday; 0 = Monday (pandas dayofweek)
EPOCH_DOW = 3                 # 1970-01-01 was a Thursday


def weekend_frac(anchor_ts, H) -> np.ndarray:
    """Fraction of the window's hours [anchor, anchor + H) on Sat/Sun, UTC."""
    anchor_ts = np.asarray(anchor_ts, np.int64)
    H = np.asarray(H, np.int64)
    offs = np.arange(int(H.max()))
    dow = (((anchor_ts[:, None] + offs[None, :] * HOUR) // 86400) + EPOCH_DOW) % 7
    valid = offs[None, :] < H[:, None]
    return (np.isin(dow, WEEKEND) & valid).sum(1) / np.maximum(H, 1)


def weekend_frac_from_clock(dow, hour, H) -> np.ndarray:
    """The same fraction from the anchor's weekday (0 = Mon) and UTC hour."""
    dow = np.asarray(dow, np.int64)
    hour = np.asarray(hour, np.int64)
    H = np.asarray(H, np.float64).astype(np.int64)
    offs = np.arange(int(H.max()))
    d = (dow[:, None] + (hour[:, None] + offs[None, :]) // 24) % 7
    valid = offs[None, :] < H[:, None]
    return (np.isin(d, WEEKEND) & valid).sum(1) / np.maximum(H, 1)


def dow_from_cal(dow_sin, dow_cos) -> np.ndarray:
    """Invert features.py's cal_dow_sin/cos (2*pi*dow/7) back to 0..6."""
    ang = np.arctan2(np.asarray(dow_sin, np.float64), np.asarray(dow_cos, np.float64))
    return np.round(ang * 7 / (2 * np.pi)).astype(np.int64) % 7


# ---- the full week (P4-dow-anchor-result) -----------------------------------
# The anchor increment uses the window's fraction on Mon..Sat (Sunday is the
# reference): the residual regression carries no Fri+Sat column, so all six are
# needed to span the week (with five, Saturday and Sunday would be forced to
# share a level -- measured in eval/dow_staleness.py).
DOW_COLS = (0, 1, 2, 3, 4, 5)


def day_fracs(anchor_ts, H) -> np.ndarray:
    """(n, 7) fraction of the window's hours on each weekday, 0 = Monday."""
    anchor_ts = np.asarray(anchor_ts, np.int64)
    H = np.asarray(H, np.int64)
    offs = np.arange(int(H.max()))
    dow = (((anchor_ts[:, None] + offs[None, :] * HOUR) // 86400) + EPOCH_DOW) % 7
    valid = offs[None, :] < H[:, None]
    return np.stack([((dow == d) & valid).sum(1) for d in range(7)], 1) / np.maximum(H, 1)[:, None]


def day_fracs_from_clock(dow, hour, H) -> np.ndarray:
    """The same (n, 7) fractions from the anchor's weekday and UTC hour."""
    dow = np.asarray(dow, np.int64)
    hour = np.asarray(hour, np.int64)
    H = np.asarray(H, np.float64).astype(np.int64)
    offs = np.arange(int(H.max()))
    d = (dow[:, None] + (hour[:, None] + offs[None, :]) // 24) % 7
    valid = offs[None, :] < H[:, None]
    return np.stack([((d == k) & valid).sum(1) for k in range(7)], 1) / np.maximum(H, 1)[:, None]
