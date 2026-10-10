"""
tests/test_context_pit.py
=====================================================================
Gates the point-in-time context dataset (data/context/, model/context/).

1. Every committed context file passes pit.validate.
2. pit.asof is strict: a row published exactly at t is NOT visible at t.
3. Hindsight labels in explain_only/ cannot be loaded as features.
4. Truncation: features at t are identical whether or not every row with
   available_ts >= t is deleted. A lookahead in the feature code or the
   join would make them differ.

    python model/tests/test_context_pit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context import features as CF  # noqa: E402
from context import pit  # noqa: E402


def test_files_validate():
    files = sorted(pit.CONTEXT.glob("*.parquet"))
    assert files, "no context files"
    for p in files:
        pit.load(p)


def test_asof_is_strict():
    df = pd.DataFrame({"source": "x", "series": "s", "event_ts": [0, 100],
                       "available_ts": [10, 200], "value": [1.0, 2.0], "text": "",
                       "source_url": "", "retrieved_at": 10_000, "pit_quality": "exact"})
    df = pit.validate(df)
    got = pit.asof(df, "s", np.array([5, 10, 11, 200, 201]))
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert got[2] == 1.0 and got[3] == 1.0 and got[4] == 2.0


def test_explain_only_refused():
    for p in (pit.CONTEXT / "explain_only").glob("*"):
        try:
            pit.load(p)
        except PermissionError:
            continue
        raise AssertionError(f"{p} loaded as features")


def test_truncation_invariance(n=12, seed=0):
    rng = np.random.default_rng(seed)
    t_all = np.arange(pd.Timestamp("2019-01-01", tz="UTC").timestamp(),
                      pd.Timestamp("2026-09-01", tz="UTC").timestamp(), 86400, dtype=np.int64)
    t_all = t_all + 17 * 3600
    ts = np.sort(rng.choice(t_all, n, replace=False))
    for group in CF.GROUPS:
        path = pit.CONTEXT / f"{group}.parquet"
        if not path.exists():
            continue
        ctx = pit.load(path)
        full = CF.build(group, ts, ctx)
        for i, t in enumerate(ts):
            cut = CF.build(group, np.array([t]), ctx[ctx.available_ts < t])
            a = full.iloc[i][cut.columns].to_numpy(np.float64)
            b = cut.iloc[0].to_numpy(np.float64)
            assert np.allclose(a, b, equal_nan=True), f"{group}: features at {t} see the future"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
