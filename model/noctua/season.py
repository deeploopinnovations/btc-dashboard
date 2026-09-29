"""
noctua/season.py
=====================================================================
The intraday clock for the served anchor (P4-hour-anchor-result).

One source for the arithmetic, imported by the evaluation that justified it
(eval/hour_anchor.py) and by the runtime that serves it (serve/runtime.py), so
the two cannot compute different numbers under the same name.

    profile  s(h), h = 0..23 CLOCK hours: the mean log vol of that hour on the
             training slice, centred. har_1h at an anchor of hour a is the vol
             of the hour ENDING at a, i.e. clock hour a - 1.
    season_fwd(a, H) = 0.5 * log( mean_{k=0..H-1} v((a + k) mod 24) / mean v )
             with v = exp(2 s): the log of the average seasonal VARIANCE factor
             over the forecast window's own clock hours, in sigma units,
             zero for any whole number of days.

Pure NumPy; the runtime has no pandas dependency on this path.
"""
from __future__ import annotations

import numpy as np

# log(1e-6) encodes a MISSING hour (P2-floor-defect), not a quiet one;
# anything at or below log(1e-5) is excluded from the profile.
FLOOR_LOG = float(np.log(1e-5))


def hour_profile(har_1h, anchor_hour) -> np.ndarray:
    """Centred log-vol offset for each clock hour, from anchors' har_1h."""
    v = np.asarray(har_1h, np.float64)
    ok = np.isfinite(v) & (v > FLOOR_LOG)
    clock = (np.asarray(anchor_hour, np.int64)[ok] - 1) % 24
    v = v[ok]
    s = np.array([v[clock == h].mean() for h in range(24)])
    return s - s.mean()


def season_fwd(profile, anchor_hour, H) -> np.ndarray:
    """0.5 log of the mean seasonal variance factor over [a, a + H), centred."""
    var = np.exp(2.0 * np.asarray(profile, np.float64))
    var = var / var.mean()
    a = np.asarray(anchor_hour, np.int64) % 24
    Hh = np.asarray(np.rint(H), np.int64)
    reps = int(np.ceil((int(Hh.max()) + 24) / 24)) + 1
    c = np.concatenate([[0.0], np.cumsum(np.tile(var, reps))])
    tot = c[a + Hh] - c[a]
    return 0.5 * np.log(tot / Hh)


def anchor_hour_from_cal(hour_sin, hour_cos) -> np.ndarray:
    """Invert features.py's cal_hour_sin/cos = sin/cos(2 pi a / 24)."""
    ang = np.arctan2(np.asarray(hour_sin, np.float64),
                     np.asarray(hour_cos, np.float64))
    return np.rint(ang * 24.0 / (2.0 * np.pi)).astype(np.int64) % 24


# ------------------------------------------------------------------ week ----
# The same construction on the HOUR OF THE WEEK (168 slots, 0 = Monday 00:00
# UTC). P4-hour-anchor-result's clock-aware anchor left its residual bias at
# the 17:00 product anchor almost entirely in windows that touch the weekend:
# a 24-slot profile is weekday-dominated and cannot say that the US session is
# missing on a Saturday.
HOURS_PER_WEEK = 168
_EPOCH_MONDAY_OFFSET_H = 96          # 1970-01-01 00:00 UTC was a Thursday


def hour_of_week(ts) -> np.ndarray:
    """Slot 0..167 of the clock hour STARTING at unix time `ts` (Monday 0)."""
    h = np.asarray(ts, np.int64) // 3600
    return (h - _EPOCH_MONDAY_OFFSET_H) % HOURS_PER_WEEK


def week_profile(har_1h, anchor_ts) -> np.ndarray:
    """Centred log-vol offset for each hour-of-week slot. har_1h at anchor t
    is the vol of the hour starting at t - 1h."""
    v = np.asarray(har_1h, np.float64)
    ok = np.isfinite(v) & (v > FLOOR_LOG)
    slot = hour_of_week(np.asarray(anchor_ts, np.int64)[ok] - 3600)
    v = v[ok]
    s = np.array([v[slot == k].mean() for k in range(HOURS_PER_WEEK)])
    return s - s.mean()


def season_fwd_week(profile, anchor_slot, H) -> np.ndarray:
    """0.5 log mean seasonal variance factor over [slot, slot + H), centred
    over the week (a whole WEEK scores 0; a weekend day does not)."""
    var = np.exp(2.0 * np.asarray(profile, np.float64))
    var = var / var.mean()
    a = np.asarray(anchor_slot, np.int64) % HOURS_PER_WEEK
    Hh = np.asarray(np.rint(H), np.int64)
    reps = int(np.ceil((int(Hh.max()) + HOURS_PER_WEEK) / HOURS_PER_WEEK)) + 1
    c = np.concatenate([[0.0], np.cumsum(np.tile(var, reps))])
    tot = c[a + Hh] - c[a]
    return 0.5 * np.log(tot / Hh)


def slot_from_cal(hour_sin, hour_cos, dow_sin, dow_cos) -> np.ndarray:
    """Hour-of-week slot from features.py's cal_hour_* and cal_dow_*."""
    hr = anchor_hour_from_cal(hour_sin, hour_cos)
    ang = np.arctan2(np.asarray(dow_sin, np.float64), np.asarray(dow_cos, np.float64))
    dow = np.rint(ang * 7.0 / (2.0 * np.pi)).astype(np.int64) % 7
    return dow * 24 + hr
