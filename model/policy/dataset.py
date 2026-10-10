"""
policy/dataset.py
=====================================================================
Builds the decision dataset for NOCTUA-Trader, the open-weight BTCUSDT
position policy that sits on top of the NOCTUA volatility forecaster.

One row per (anchor hour a). At anchor a the policy sees ONLY rows <= a-1
(the same causal contract as noctua/features.py) and chooses a position in
[-1, +1] that it holds for the next 24 hours:

    entry  = close[a-1]            (the last closed hour before the anchor)
    exit   = close[a+23]
    ret    = log(exit / entry)
    fund   = sum(interest_1h[a .. a+23])   perp funding a long pays

Decisions are made once per day at one anchor hour (17:00 UTC in production,
NOCTUA's own anchor), so every anchor hour h in 0..23 is a separate,
non-overlapping daily sequence ("phase"). Training may use all 24 phases as
augmentation; validation and test are scored on the production phase only.

Inputs, all known at the anchor:
  * NOCTUA's 42 engineered features (noctua/features.build_features)
  * NOCTUA's own forecast for the 19 h window: log sigma (median, mean),
    and the location/skew of its return quantiles in sigma units
  * trailing log returns over 1 h .. 30 d, distance from 20 d / 50 d means
  * DVOL (Deribit implied vol) and perp funding, each with a "present" flag
    because neither exists for the whole history
  * clock: hour-of-day and day-of-week on the unit circle

The result is cached at model/policy/cache/dataset.parquet (git-ignored,
rebuildable in about a minute).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
sys.path.insert(0, str(MODEL))

from noctua.features import build_features  # noqa: E402

NOCTUA_ARTIFACT = MODEL / "serve" / "noctua_v2.npz"
CACHE = Path(__file__).with_name("cache") / "dataset.parquet"
HOLD_H = 24
NOCTUA_H = 19
PROD_HOUR = 17

# splits.py's production boundaries: train < 2023-01-01, calib < 2024-07-01,
# test after. NOCTUA itself was fit on the same train/calib split, so its
# forecasts are in-sample before 2024-07 and genuinely out-of-sample after.
TRAIN_END = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
VAL_END = int(pd.Timestamp("2024-07-01", tz="UTC").timestamp())
EMBARGO_S = HOLD_H * 3600


def load_hours() -> pd.DataFrame:
    """Committed hourly bars: the long asset history plus the live bundle."""
    a = pd.read_parquet(ROOT / "data/assets/btc_history.parquet")
    b = pd.read_parquet(ROOT / "data/noctua_history.parquet")
    h = pd.concat([a, b[b.hour_ts > a.hour_ts.max()]])
    h = h.sort_values("hour_ts").drop_duplicates("hour_ts").reset_index(drop=True)
    gaps = np.diff(h.hour_ts.to_numpy())
    if (gaps != 3600).any():
        raise RuntimeError(f"hourly history has {(gaps != 3600).sum()} gaps")
    return h


def _hourly_series(path: Path, col: str, ts_col: str, hour_ts: np.ndarray):
    """Align an exogenous hourly series to hour_ts. Value at hour t is the
    observation stamped t, so using index a-1 at anchor a is causal."""
    d = pd.read_parquet(path)[[ts_col, col]].dropna()
    d = d.groupby(ts_col)[col].last()
    s = pd.Series(np.nan, index=hour_ts)
    common = s.index.intersection(d.index)
    s.loc[common] = d.loc[common].to_numpy()
    return s.to_numpy(np.float64)


def _trail_sum(x: np.ndarray, k: int) -> np.ndarray:
    """out[i] = sum(x[i-k:i]) (exclusive of i); NaN where not enough data."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan)
    out[k:] = c[k:len(x)] - c[: len(x) - k]
    return out


def noctua_outputs(hours: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """NOCTUA's forecast for the 19 h window at each anchor, in batch."""
    from serve.runtime import load_model
    # PINNED, not "newest": serve/ also holds noctua_v2_refreshed_2026-08-09.npz,
    # fit through 2026-08 -- it would make the test-period forecasts in-sample.
    m = load_model(NOCTUA_ARTIFACT)
    out = []
    for s in range(0, len(X), 8192):
        xb = X.iloc[s:s + 8192]
        p = m.predict(m.prepare(xb, np.full(len(xb), float(NOCTUA_H))))
        sig = np.maximum(p["sigma_med"], 1e-6)
        qr = p["q_r"].mean(axis=1)               # (n, levels) return quantiles
        lv = m.levels
        i05, i50, i95 = (int(np.argmin(np.abs(lv - q))) for q in (0.05, 0.5, 0.95))
        out.append(pd.DataFrame({
            "nx_logsig_med": np.log(sig),
            "nx_logsig_mean": np.log(np.maximum(p["sigma_mean"], 1e-6)),
            "nx_qr_med_z": qr[:, i50] / sig,
            "nx_qr_skew_z": (qr[:, i95] + qr[:, i05] - 2 * qr[:, i50]) / sig,
            "nx_qr_width_z": (qr[:, i95] - qr[:, i05]) / sig,
        }, index=xb.index))
    return pd.concat(out)


def features_at(h: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Policy inputs at anchor rows `rows` of hourly frame `h`.

    Only rows <= a-1 are read for anchor a. Shared by training (`build`) and
    the paper trader (`policy/paper.py`), so the two cannot drift apart."""
    ts = h.hour_ts.to_numpy(np.int64)
    close = h.close.to_numpy(np.float64)
    rows = np.asarray(rows, np.int64)
    dt = pd.to_datetime(ts[rows], unit="s", utc=True)
    ep = pd.DataFrame({
        "anchor_ts": ts[rows], "H": NOCTUA_H, "row": rows, "dt": dt,
        "anchor_hour": dt.hour, "dow": dt.dayofweek,
    })
    X = build_features(h, ep)
    X.index = ep.index
    nx = noctua_outputs(h, X)

    lc = np.log(close)
    F = {}
    prev = rows - 1                               # last closed hour
    for k in (1, 4, 24, 72, 168, 720):
        F[f"ret_{k}h"] = lc[prev] - lc[prev - k]
    c = np.concatenate([[0.0], np.cumsum(lc)])
    for k in (480, 1200):
        ma = (c[rows] - c[rows - k]) / k          # mean of lc[a-k .. a-1]
        F[f"dist_ma_{k // 24}d"] = lc[prev] - ma

    dvol = _hourly_series(ROOT / "data/newdata/dvol_btc.parquet", "volatility", "ts", ts)
    fund = _hourly_series(ROOT / "data/newdata/funding_btc.parquet", "interest_1h", "ts", ts)
    dv = dvol[prev]
    F["dvol_log"] = np.where(np.isfinite(dv), np.log(np.maximum(dv, 1.0) / 100.0), 0.0)
    F["dvol_present"] = np.isfinite(dv).astype(float)
    f24 = _trail_sum(fund, 24)[rows]              # funding over the last 24 h
    F["fund_24h_bp"] = np.where(np.isfinite(f24), f24 * 1e4, 0.0)
    F["fund_present"] = np.isfinite(fund[prev]).astype(float)
    # DVOL vs NOCTUA: the market's 30 d implied vs NOCTUA's 19 h forecast,
    # both annualised
    ann_nx = np.exp(nx["nx_logsig_med"].to_numpy()) * np.sqrt(24 * 365 / NOCTUA_H)
    F["dvol_minus_nx"] = np.where(np.isfinite(dv), dv / 100.0 - ann_nx, 0.0)
    hr = dt.hour.to_numpy()
    dw = dt.dayofweek.to_numpy()
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)

    return pd.concat([
        pd.DataFrame({"anchor_ts": ts[rows], "hour": hr, "row": rows}),
        X.reset_index(drop=True).add_prefix("nf_"),
        nx.reset_index(drop=True),
        pd.DataFrame(F),
    ], axis=1)


def targets(h: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """FUTURE quantities for anchor rows -- never features."""
    ts = h.hour_ts.to_numpy(np.int64)
    lc = np.log(h.close.to_numpy(np.float64))
    fund = _hourly_series(ROOT / "data/newdata/funding_btc.parquet", "interest_1h", "ts", ts)
    rows = np.asarray(rows, np.int64)
    tgt_ret = lc[rows + HOLD_H - 1] - lc[rows - 1]
    fwd_fund = _trail_sum(fund, HOLD_H)[rows + HOLD_H]   # sum fund[a .. a+23]
    return pd.DataFrame({"tgt_ret": tgt_ret,
                         "tgt_fund": np.where(np.isfinite(fwd_fund), fwd_fund, 0.0)})


def build(min_history_h: int = 24 * 400) -> pd.DataFrame:
    h = load_hours()
    rows = np.arange(min_history_h, len(h) - HOLD_H)
    D = pd.concat([features_at(h, rows), targets(h, rows)], axis=1)
    D["split"] = np.where(D.anchor_ts + EMBARGO_S <= TRAIN_END, "train",
                  np.where((D.anchor_ts >= TRAIN_END) & (D.anchor_ts + EMBARGO_S <= VAL_END), "val",
                  np.where(D.anchor_ts >= VAL_END, "test", "embargo")))
    return D


def feature_cols(D: pd.DataFrame) -> list[str]:
    skip = {"anchor_ts", "hour", "row", "tgt_ret", "tgt_fund", "split"}
    return [c for c in D.columns if c not in skip]


def load(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    D = build()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(CACHE)
    return D


if __name__ == "__main__":
    D = load(rebuild=True)
    print(D.shape)
    print(D.groupby("split").anchor_ts.agg(["count", "min", "max"]))
    print(D[feature_cols(D)].describe().T[["mean", "std", "min", "max"]].to_string())
