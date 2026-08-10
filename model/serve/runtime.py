"""
serve/runtime.py
=====================================================================
Dependency-free NOCTUA inference: NumPy + SciPy only, no PyTorch.

Why this exists: the trained network is 49,866 parameters. Shipping a ~2 GB
PyTorch install to evaluate it on a free 2 vCPU Space would be absurd, and the
cold start would dominate the latency. The forward pass is four matmuls and a
cumulative sum, so it is reimplemented here directly.

The genuinely delicate parts -- the monotone quantile construction, the
survival function with exponential tail extrapolation, and the mixing integral
over Stage A's volatility uncertainty -- are NOT reimplemented. They are
imported from `noctua.infer`, which was already pure NumPy, so the served
numbers are computed by exactly the same code the evaluation used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import erf

DEFAULT_WEIGHTS = Path(__file__).with_name("noctua_weights.npz")


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU, matching torch.nn.GELU(approximate='none')."""
    return x * 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def softplus(x: np.ndarray) -> np.ndarray:
    # log1p(exp(x)) computed stably
    return np.logaddexp(0.0, x)


class NumpyNoctua:
    """NumPy re-implementation of the trained network plus its wrappers."""

    def __init__(self, path: Path | str = DEFAULT_WEIGHTS):
        z = np.load(path, allow_pickle=False)
        self.w = {k: z[k] for k in z.files if k != "meta_json"}
        self.meta = json.loads(bytes(z["meta_json"]).decode())
        self.levels = z["levels"]
        self.median_idx = int(np.argmin(np.abs(self.levels - 0.5)))

        self.feat_cols = self.meta["feat_cols"]
        self.base_cols = self.meta["base_cols"]
        self.shape_cols = self.meta["shape_cols"]
        self.blend_w = self.meta["blend_w"]
        self.cal_shrink = self.meta["cal_shrink"]

    # ---- primitives -------------------------------------------------------
    def _lin(self, prefix: str, x: np.ndarray) -> np.ndarray:
        return x @ self.w[f"{prefix}.weight"].T.astype(np.float64) + self.w[f"{prefix}.bias"].astype(np.float64)

    def _body(self, prefix: str, x: np.ndarray) -> np.ndarray:
        h = gelu(self._lin(f"{prefix}.0", x))
        return gelu(self._lin(f"{prefix}.2", h))

    def _qhead(self, prefix: str, h: np.ndarray, nonneg: bool) -> np.ndarray:
        """Mirror of MonotoneQuantileHead: median plus outward softplus steps."""
        med = self._lin(f"{prefix}.median", h)
        up = np.cumsum(softplus(self._lin(f"{prefix}.up", h)) + 1e-4, axis=1)
        dn = np.cumsum(softplus(self._lin(f"{prefix}.dn", h)) + 1e-4, axis=1)[:, ::-1]
        q = np.concatenate([med - dn, med, med + up], axis=1)
        return softplus(q) if nonneg else q

    # ---- stages -----------------------------------------------------------
    def stage_a(self, Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
        return self._lin("a.base", Xb) + self._qhead("a.head", self._body("a.body", Xa), False)

    def stage_b(self, Xs: np.ndarray, log_sigma: np.ndarray):
        h = self._body("b.body", np.concatenate([Xs, log_sigma], axis=1))
        return (self._qhead("b.q_r", h, False),
                self._qhead("b.q_up", h, True),
                self._qhead("b.q_dn", h, True))

    # ---- standardization / feature assembly -------------------------------
    def _std(self, name: str, A: np.ndarray) -> np.ndarray:
        mu, sd = self.w[f"{name}_mu"], self.w[f"{name}_sd"]
        return np.nan_to_num((A - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    def prepare(self, features: "object", H: np.ndarray) -> dict:
        """`features` is a pandas DataFrame with at least `self.feat_cols`."""
        Xa = features[self.feat_cols].to_numpy(np.float64)
        Xb = features[self.base_cols].to_numpy(np.float64)
        Xs = features[self.shape_cols].to_numpy(np.float64)
        return {
            "Xa": self._std("std_all", Xa),
            "Xb": self._std("std_base", Xb),
            "Xs": self._std("std_shape", Xs),
            "H": np.asarray(H, dtype=np.float64),
        }

    def har_logvol(self, d: dict) -> np.ndarray:
        """The Log-HAR ensemble partner, from the stored OLS coefficients."""
        beta = self.w["har_beta"]
        return beta[0] + d["Xb"] @ beta[1:]

    # ---- full predictive object -------------------------------------------
    def predict(self, d: dict, n_atoms: int = 32) -> dict:
        from noctua import infer as I  # pure NumPy; same code the eval used

        qa = self.stage_a(d["Xa"], d["Xb"])
        har = self.har_logvol(d)
        if self.blend_w < 1.0:
            qa = qa + ((1.0 - self.blend_w) * (har - qa[:, self.median_idx]))[:, None]

        atom_levels = (np.arange(n_atoms) + 0.5) / n_atoms
        atoms_y = np.stack([np.interp(atom_levels, self.levels, row) for row in qa])
        H = d["H"]
        sigma_atoms = np.exp(atoms_y) * np.sqrt(H)[:, None]

        qr, qu, qd = [], [], []
        for i in range(n_atoms):
            ls = np.log(np.maximum(sigma_atoms[:, i], 1e-12))[:, None]
            r_, u_, d_ = self.stage_b(d["Xs"], ls)
            qr.append(r_); qu.append(u_); qd.append(d_)

        return {
            "qa": qa,
            "sigma_atoms": sigma_atoms,
            "sigma_med": np.exp(qa[:, self.median_idx]) * np.sqrt(H),
            "sigma_mean": np.sqrt(np.mean(np.exp(2.0 * atoms_y), axis=1) * H),
            "q_r": np.stack(qr, 1),
            "q_up": np.stack(qu, 1),
            "q_dn": np.stack(qd, 1),
            "H": H,
        }

    # ---- calibrated served quantities -------------------------------------
    def _cal_map(self, side: str) -> tuple[np.ndarray, np.ndarray] | None:
        gk, mk = f"cal_{side}_grid", f"cal_{side}_map"
        if gk not in self.w:
            return None
        g, m = self.w[gk], self.w[mk]
        eff = (1.0 - self.cal_shrink) * g + self.cal_shrink * m
        return g, eff

    def touch_prob(self, pred: dict, u: np.ndarray, up: bool = True) -> np.ndarray:
        from noctua import infer as I

        raw = I.touch_prob(pred, u, up=up)
        cm = self._cal_map("up" if up else "dn")
        if cm is None:
            return raw
        g, eff = cm
        return np.clip(1.0 - np.interp(np.clip(1.0 - raw, 0, 1), g, eff), 0.0, 1.0)

    def safe_level(self, pred: dict, alpha: float, up: bool = True, iters: int = 36) -> np.ndarray:
        n = pred["sigma_atoms"].shape[0]
        lo, hi = np.full(n, 1e-5), np.full(n, 2.0)
        for _ in range(iters):
            m = 0.5 * (lo + hi)
            risky = self.touch_prob(pred, m, up) > alpha
            lo = np.where(risky, m, lo)
            hi = np.where(risky, hi, m)
        return 0.5 * (lo + hi)

    def prob_up(self, pred: dict) -> np.ndarray:
        from noctua import infer as I

        return I.prob_up(pred)
