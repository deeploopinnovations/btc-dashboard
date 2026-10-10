#!/usr/bin/env python3
"""
model/tests/test_policy_runtime.py
=====================================================================
Self-test for NOCTUA-Trader's open-weight artifact. No PyTorch, no training
cache: committed hourly history and the NumPy runtime only.

  1. the weights load and carry the metadata the runtime needs
  2. the NumPy runtime reproduces the positions the trained torch ensemble
     recorded for the test split (noctua_trader_v1_test_daily.npz), so the
     open weights are the model that was scored
  3. positions respect the artifact's bounds (long-only, <= max_leverage)
  4. features are causal: corrupting every hour at or after the anchor
     leaves the policy's inputs unchanged
  5. the paper trader produces a decision offline
  6. torch is never imported on this path
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy import dataset as DS               # noqa: E402
from policy import paper                       # noqa: E402
from policy.runtime import DEFAULT_WEIGHTS, NumpyTrader  # noqa: E402

W = DEFAULT_WEIGHTS.parent


def main() -> int:
    m = NumpyTrader()
    for k in ("features", "n_seeds", "n_layers", "max_leverage", "long_only", "clip"):
        assert k in m.meta, k
    print(f"[1] loaded {DEFAULT_WEIGHTS.name}: {len(m.features)} inputs, {m.meta['n_seeds']} seeds")

    h = DS.load_hours()
    ts = h.hour_ts.to_numpy(np.int64)
    rec = np.load(W / "noctua_trader_v1_test_daily.npz")
    pick = np.linspace(0, len(rec["anchor_ts"]) - 1, 8).astype(int)
    rows = np.searchsorted(ts, rec["anchor_ts"][pick])
    pos = m.position(DS.features_at(h, rows))
    err = float(np.abs(pos - rec["pos"][pick]).max())
    assert err < 1e-6, f"runtime/torch parity broken: {err:.3e}"
    print(f"[2] parity with recorded torch positions: max |diff| {err:.2e}")

    lo = 0.0 if m.meta["long_only"] else -m.meta["max_leverage"]
    assert (pos >= lo - 1e-12).all() and (pos <= m.meta["max_leverage"] + 1e-12).all()
    print(f"[3] positions in [{lo}, {m.meta['max_leverage']}]")

    worst = 0.0
    for r in rows[:4]:
        base = DS.features_at(h, np.array([r]))[m.features].to_numpy()
        hc = h.copy()
        for c in ("open", "high", "low", "close", "volume", "rv5", "rv5_pos",
                  "rv5_neg", "bpv5", "rq5"):
            v = hc[c].to_numpy(np.float64).copy()
            v[r:] = v[r:] * 7.3 + 1.0
            hc[c] = v
        bad = DS.features_at(hc, np.array([r]))[m.features].to_numpy()
        worst = max(worst, float(np.nanmax(np.abs(bad - base))))
    assert worst == 0.0, f"lookahead: features moved by {worst:.3e} when the future changed"
    print("[4] causal: corrupting hours >= anchor changes no input (max 0.0)")

    d = paper.decide(h, m)
    assert lo <= d["position"] <= m.meta["max_leverage"]
    print(f"[5] paper decision at {d['anchor_utc']}: position {d['position']}")

    assert "torch" not in sys.modules, "torch imported on the NumPy path"
    print("[6] torch not imported")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
