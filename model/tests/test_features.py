#!/usr/bin/env python3
"""
model/tests/test_features.py
=====================================================================
Tests for the feature construction path -- previously untested entirely.

WHY THIS FILE DID NOT EXIST AND SHOULD HAVE

`noctua/features.py` is the widest blast radius in the repository. It runs in
BOTH pipelines: training reads it to build `features.parquet`, and
`serve/predict.py:86` calls the identical function on the live tail. A silent
change to a lag, a window length, or a column name would move every number the
model produces, in training and in production simultaneously, and no test
would have said a word. Until this file, none did -- the coverage audit found
`build_features` imported by tests but never called.

Two of the defects this repository has actually shipped were in exactly this
blast radius:

  * the artifact declaring 42 feature columns while its weights expected 39,
    which killed serving with a matmul error the moment it was rebuilt; and
  * every trailing aggregate silently stopping an hour earlier than the
    module's own stated contract.

Both are checkable in milliseconds. Test 3 covers the first, tests 1-2 the
second.

  1. the no-lookahead contract holds NUMERICALLY, at both lag settings:
     corrupting the future must not move a single feature
  2. the freshest permitted hour is actually USED at extra_lag_hours=0 and
     actually DISCARDED at the default -- the two settings must differ, or
     the parameter is a lie
  3. every column the SHIPPED ARTIFACT declares it consumes is present in
     build_features' output, with no extras silently widening the matrix
  4. the output is finite outside the warm-up region, so `nan_to_num` in
     `prepare()` is never quietly substituting the training mean
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noctua.features import audit_lookahead, build_features  # noqa: E402
from noctua.spec import NON_MODEL_COLS                       # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  -> ' + detail}")
    if not cond:
        FAILS.append(name)


def synth_hours(n: int = 12000, seed: int = 0) -> pd.DataFrame:
    """A synthetic hourly series with realistic volatility clustering.

    Self-contained: CI has no training parquet, and the properties under test
    are structural, so they must not depend on the real data being present.

    Length is not arbitrary. `reg_rv_vs_year` compares a trailing day against a
    trailing YEAR, so it is NaN until 8,760 hours have accumulated; a shorter
    series makes the finiteness check vacuous for that column. The semivariance
    and bipower series must also be genuinely random rather than fixed
    fractions of `rv5` -- with `rv5_pos == rv5_neg` exactly, `semi_neg_share`
    is the constant 0.5 and would register as "unaffected by the lag" for a
    reason that has nothing to do with the lag.
    """
    rng = np.random.default_rng(seed)
    logv = np.zeros(n)
    for i in range(1, n):                       # persistent log-vol, as in the data
        logv[i] = 0.98 * logv[i - 1] + 0.15 * rng.standard_normal()
    sig = 0.004 * np.exp(logv)
    ret = sig * rng.standard_normal(n)
    close = 30000.0 * np.exp(np.cumsum(ret))
    openp = np.concatenate([[30000.0], close[:-1]])
    span = np.abs(ret) * close * rng.uniform(1.0, 2.0, n)
    rv = sig**2
    share = rng.uniform(0.25, 0.75, n)          # genuinely varying up/down split
    return pd.DataFrame({
        "hour_ts": np.arange(n, dtype=np.int64) * 3600 + 1_500_000_000,
        "open": openp, "close": close,
        "high": np.maximum(openp, close) + span, "low": np.minimum(openp, close) - span,
        "volume": rng.uniform(50, 500, n),
        "rv5": rv, "rv5_pos": share * rv, "rv5_neg": (1.0 - share) * rv,
        "bpv5": rv * rng.uniform(0.70, 0.99, n), "rq5": 3.0 * rv**2,
    })


def synth_episodes(hours: pd.DataFrame, first: int = 9000) -> pd.DataFrame:
    rows = np.arange(first, len(hours) - 30)
    ts = hours["hour_ts"].to_numpy()[rows]
    dt = pd.to_datetime(ts, unit="s", utc=True)
    return pd.DataFrame({
        "anchor_ts": ts, "H": 19, "row": rows, "dt": dt,
        "anchor_hour": dt.hour, "dow": dt.dayofweek,
    })


def main() -> int:
    hours = synth_hours()
    ep = synth_episodes(hours)
    print("features: no-lookahead contract")

    # ---- 1. causality, both settings ------------------------------------
    for lag in (1, 0):
        a = audit_lookahead(hours, ep, n_probe=60, extra_lag_hours=lag)
        check(f"extra_lag_hours={lag}: corrupting the future moves nothing",
              a["leak_free"],
              f"max change {a['max_abs_feature_change']:.3e} in {a['offending_features']}")

    # ---- 2. the parameter does what it claims ---------------------------
    X1 = build_features(hours, ep, extra_lag_hours=1)
    X0 = build_features(hours, ep, extra_lag_hours=0)
    trailing = [c for c in X1.columns
                if c.split("_")[0] in ("har", "semi", "jump", "rq", "rng",
                                       "mom", "vov", "eff")]
    moved = [c for c in trailing
             if not np.allclose(X1[c].to_numpy(), X0[c].to_numpy(), equal_nan=True)]
    check("lag 0 and lag 1 differ on trailing features",
          len(moved) > 0.8 * len(trailing),
          f"only {len(moved)} of {len(trailing)} moved")

    # har_1h IS a single hour, so it is the sharpest probe: at lag 0 it must
    # equal the realized variance of hour a-1, and at lag 1 it must not.
    rv5 = hours["rv5"].to_numpy(np.float64)
    rows = ep["row"].to_numpy()
    want = 0.5 * np.log(rv5[rows - 1])
    check("lag 0: har_1h is the realized vol of the last complete hour",
          np.allclose(X0["har_1h"].to_numpy(), want, rtol=1e-9),
          f"max diff {np.nanmax(np.abs(X0['har_1h'].to_numpy() - want)):.3e}")
    check("lag 1 (shipped): har_1h is NOT that hour -- one hour is discarded",
          not np.allclose(X1["har_1h"].to_numpy(), want, rtol=1e-9))

    # calendar features describe the anchor itself and must not shift with lag
    for c in ("cal_hour_sin", "cal_dow_cos", "cal_H", "cal_weekend_frac"):
        check(f"{c} is independent of extra_lag_hours",
              np.allclose(X1[c].to_numpy(), X0[c].to_numpy(), equal_nan=True))

    # ---- 3. the artifact's declared columns must exist ------------------
    # This is the check that would have caught the 42-declared/39-trained
    # packaging bug before it reached serving.
    try:
        from serve.runtime import load_model
        m = load_model()
        produced = set(X1.columns)
        for name, cols in (("feat_cols", m.feat_cols),
                           ("base_cols", m.base_cols),
                           ("shape_cols", m.shape_cols)):
            missing = [c for c in cols if c not in produced]
            check(f"every {name} the artifact declares is produced",
                  not missing, f"missing {missing}")
        leaked = [c for c in NON_MODEL_COLS if c in set(m.feat_cols)]
        check("research-only columns are not in the artifact's inputs",
              not leaked, f"{leaked} leaked into feat_cols")
    except FileNotFoundError:
        print("  SKIP  artifact checks (no served weights on disk)")

    # ---- 4. finite outside the warm-up ----------------------------------
    warm = ep["row"].to_numpy() >= 9000
    body = X1.loc[warm].to_numpy(np.float64)
    n_bad = int((~np.isfinite(body)).sum())
    check("no NaN/inf past the warm-up region", n_bad == 0,
          f"{n_bad} non-finite entries")

    print(f"\n{'all checks passed' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
