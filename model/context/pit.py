"""
context/pit.py
=====================================================================
Point-in-time loader and as-of join for data/context/*.parquet.

Every context row carries `available_ts`, the earliest moment a trader
could have read it. `asof(...)` returns, for each decision time t, the
latest row with available_ts < t -- never anything published at or after
t. That is the only join features should use. See SPEC.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "data" / "context"
COLUMNS = ["source", "series", "event_ts", "available_ts", "value", "text",
           "source_url", "retrieved_at", "pit_quality"]
QUALITIES = {"exact", "lagged", "scheduled", "revisable"}


def validate(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    df = df[COLUMNS].copy()
    for c in ("event_ts", "available_ts", "retrieved_at"):
        df[c] = df[c].astype("int64")
    df["value"] = df["value"].astype("float64")
    df["text"] = df["text"].fillna("").astype(str)
    bad_q = set(df.pit_quality.unique()) - QUALITIES
    if bad_q:
        raise ValueError(f"{name}: unknown pit_quality {bad_q}")
    early = (df.available_ts < df.event_ts) & (df.pit_quality != "scheduled")
    if early.any():
        raise ValueError(f"{name}: {int(early.sum())} rows available before their event")
    future = df.available_ts > df.retrieved_at + 86400
    if future.any():
        raise ValueError(f"{name}: {int(future.sum())} rows available after retrieval")
    return df


def load(path: Path) -> pd.DataFrame:
    if "explain_only" in Path(path).parts:
        raise PermissionError("explain_only/ holds hindsight labels; never features")
    return validate(pd.read_parquet(path), Path(path).name)


def load_all(context_dir: Path = CONTEXT) -> pd.DataFrame:
    files = sorted(p for p in context_dir.glob("*.parquet"))
    return pd.concat([load(p) for p in files], ignore_index=True) if files else \
        pd.DataFrame(columns=COLUMNS)


def asof(df: pd.DataFrame, series: str, t: np.ndarray, source: str | None = None,
         max_age_s: int | None = None) -> np.ndarray:
    """Value of `series` known strictly before each time in `t` (NaN if none,
    or if the freshest row is older than max_age_s)."""
    d = df[df.series == series]
    if source is not None:
        d = d[d.source == source]
    d = d[np.isfinite(d.value)].sort_values(["available_ts", "event_ts"])
    t = np.asarray(t, np.int64)
    out = np.full(len(t), np.nan)
    if d.empty:
        return out
    av = d.available_ts.to_numpy(np.int64)
    val = d.value.to_numpy(np.float64)
    i = np.searchsorted(av, t, side="left") - 1      # last row with av < t
    ok = i >= 0
    out[ok] = val[i[ok]]
    if max_age_s is not None:
        age = np.full(len(t), np.inf)
        age[ok] = t[ok] - av[i[ok]]
        out[age > max_age_s] = np.nan
    return out


def next_scheduled(df: pd.DataFrame, series: str, t: np.ndarray) -> np.ndarray:
    """Hours from t until the next scheduled event of `series` whose schedule
    was public before t (NaN if none known)."""
    d = df[df.series == series].sort_values("event_ts")
    ev, av = d.event_ts.to_numpy(np.int64), d.available_ts.to_numpy(np.int64)
    # `<series>_cancel` rows withdraw a scheduled event from their available_ts on
    c = df[df.series == series + "_cancel"]
    cev, cav = c.event_ts.to_numpy(np.int64), c.available_ts.to_numpy(np.int64)
    t = np.asarray(t, np.int64)
    out = np.full(len(t), np.nan)
    for k, tk in enumerate(t):
        m = (ev > tk) & (av < tk)
        if len(cev):
            m &= ~np.isin(ev, cev[cav < tk])
        if m.any():
            out[k] = (ev[m].min() - tk) / 3600.0
    return out
