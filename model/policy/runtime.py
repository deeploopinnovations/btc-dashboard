"""
policy/runtime.py
=====================================================================
NOCTUA-Trader inference in plain NumPy: no PyTorch at serve time, same as
serve/runtime.py does for NOCTUA itself. The forward pass is a few matmuls,
so it is reimplemented here and pinned to the torch model by
tests/test_policy_runtime.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import erf

DEFAULT_WEIGHTS = Path(__file__).with_name("weights") / "noctua_trader_v1.npz"


def _gelu(x):
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


class NumpyTrader:
    def __init__(self, path: Path | str = DEFAULT_WEIGHTS):
        z = np.load(path, allow_pickle=False)
        self.w = {k: z[k] for k in z.files if k != "meta_json"}
        self.meta = json.loads(bytes(z["meta_json"]).decode())
        self.features = self.meta["features"]

    def standardize(self, X: np.ndarray) -> np.ndarray:
        c = self.meta["clip"]
        return np.clip((X - self.w["scaler_med"]) / self.w["scaler_scale"], -c, c)

    def _one(self, s: int, x: np.ndarray) -> np.ndarray:
        L = self.meta["n_layers"]
        for i in range(L):
            x = x @ self.w[f"s{s}.l{i}.W"].T.astype(np.float64) + self.w[f"s{s}.l{i}.b"]
            if i < L - 1:
                x = _gelu(x)
        z = x[:, 0]
        lev = self.meta["max_leverage"]
        return lev * (1.0 / (1.0 + np.exp(-z)) if self.meta["long_only"] else np.tanh(z))

    def position(self, frame) -> np.ndarray:
        """Target position for each row of `frame` (a DataFrame carrying
        every column in self.features). Seed-ensemble mean."""
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(f"missing policy inputs: {missing}")
        X = self.standardize(frame[self.features].to_numpy(np.float64))
        return np.mean([self._one(s, X) for s in range(self.meta["n_seeds"])], axis=0)
