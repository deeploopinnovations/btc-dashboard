"""
policy/backtest.py
=====================================================================
One backtester for training, model selection, baselines and the paper
trader, so the number a run reports and the number the PR quotes come out
of the same function.

Daily positions p_t in [-L, L], held 24 h from the anchor:

    R_t   = exp(ret_t) - 1                      simple return of BTC
    pnl_t = p_t * R_t
            - fee * |p_t - p_{t-1}|             taker fee + slippage on turnover
            - p_t * fund_t                      perp funding (longs pay when > 0)

Yearly figures are COMPOUNDED: CAGR = prod(1 + pnl)^(365/n) - 1.
"""
from __future__ import annotations

import numpy as np

DAYS = 365.0
DEFAULT_FEE_BPS = 6.0


def pnl(pos: np.ndarray, ret: np.ndarray, fund: np.ndarray,
        fee_bps: float = DEFAULT_FEE_BPS, prev0: float = 0.0) -> np.ndarray:
    pos = np.asarray(pos, np.float64)
    prev = np.concatenate([[prev0], pos[:-1]])
    R = np.expm1(ret)
    return pos * R - fee_bps * 1e-4 * np.abs(pos - prev) - pos * fund


def metrics(p: np.ndarray, pos: np.ndarray | None = None) -> dict:
    p = np.asarray(p, np.float64)
    n = len(p)
    wealth = np.cumprod(1.0 + p)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    sd = p.std(ddof=1) if n > 1 else np.nan
    out = {
        "n_days": int(n),
        "cagr": float(wealth[-1] ** (DAYS / n) - 1.0) if n and wealth[-1] > 0 else -1.0,
        "total_return": float(wealth[-1] - 1.0) if n else 0.0,
        "sharpe": float(p.mean() / sd * np.sqrt(DAYS)) if sd and sd > 0 else 0.0,
        "ann_vol": float(sd * np.sqrt(DAYS)) if n > 1 else 0.0,
        "max_drawdown": float((wealth / peak - 1.0).min()) if n else 0.0,
        "hit_rate": float((p > 0).mean()) if n else 0.0,
    }
    if pos is not None:
        pos = np.asarray(pos, np.float64)
        out["avg_abs_position"] = float(np.abs(pos).mean())
        out["avg_position"] = float(pos.mean())
        out["turnover_per_day"] = float(np.abs(np.diff(pos, prepend=0.0)).mean())
        out["frac_long"] = float((pos > 0.05).mean())
        out["frac_short"] = float((pos < -0.05).mean())
    return out


def block_bootstrap_sharpe_diff(a: np.ndarray, b: np.ndarray, block: int = 20,
                                reps: int = 2000, seed: int = 0) -> dict:
    """Circular block bootstrap of Sharpe(a) - Sharpe(b) on PAIRED days."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    diffs = np.empty(reps)

    def sh(x):
        s = x.std(ddof=1)
        return x.mean() / s * np.sqrt(DAYS) if s > 0 else 0.0
    for r in range(reps):
        starts = rng.integers(0, n, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        diffs[r] = sh(a[idx]) - sh(b[idx])
    return {
        "point": float(sh(a) - sh(b)),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "p_le_0": float((diffs <= 0).mean()),
    }


# ---------------------------------------------------------------- baselines
def baselines(D, fee_bps: float = DEFAULT_FEE_BPS, target_vol: float = 0.5) -> dict:
    """Positions for the reference strategies on frame D (one phase, in time
    order). Every one of them uses only information at the anchor."""
    ret, fund = D.tgt_ret.to_numpy(), D.tgt_fund.to_numpy()
    sig_ann = np.exp(D.nx_logsig_med.to_numpy()) * np.sqrt(24 * DAYS / 19.0)
    pos = {
        "flat": np.zeros(len(D)),
        "buy_and_hold": np.ones(len(D)),
        "noctua_vol_target_long": np.clip(target_vol / sig_ann, 0, 1),
        "momentum_20d": np.sign(D.dist_ma_20d.to_numpy()),
        "momentum_20d_voltarget": np.sign(D.dist_ma_20d.to_numpy()) * np.clip(target_vol / sig_ann, 0, 1),
    }
    out = {}
    for k, p in pos.items():
        pl = pnl(p, ret, fund, fee_bps)
        out[k] = {"pos": p, "pnl": pl, "metrics": metrics(pl, p)}
    return out
