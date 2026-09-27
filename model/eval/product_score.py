"""
eval/product_score.py
=====================================================================
Rescore the barrier product from saved curves, per EPISODE, and score any
volatility forecast through the Gaussian first-passage law without retraining.

run_fold returns every competitor's quantile curves and the realised
excursions (per_episode["curves"], per_episode["M_abs"]). Every metric in the
six-metric battery is a function of those two objects, so this module can

  * reproduce the battery EXACTLY (`battery`), which is how a caller proves it
    is scoring the same thing run_fold scored;
  * return Brier, log score, pinball and CRPS PER EPISODE (`per_episode`) --
    each battery value is the plain mean of its per-episode array, so a
    moving-block bootstrap over episodes is a proper interval (R88: the fold
    bootstrap cannot fail when all six folds share a sign);
  * build a Gaussian first-passage curve from any sigma (`gaussian_curves`),
    the classical no-network barrier model, through identical arithmetic.

DSC and MCB are isotonic-recalibration quantities of the whole sample and have
no per-episode form; they stay fold-level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import (BARRIER_PCT, BARRIER_U, Q_LEVELS_DESC,     # noqa: E402
                            GaussianFirstPassage, Forecaster, corp_decomposition,
                            crps_from_curve, log_score, pinball_curve)
from noctua.committee import ALPHA_GRID                                # noqa: E402

SIDES = ("up", "dn")


class _Fixed(Forecaster):
    def __init__(self, Q):
        self.Q = Q

    def curve(self, ctx, up):
        return self.Q


def touch(Q: np.ndarray) -> np.ndarray:
    """Touch probabilities on the fixed barrier grid, exactly as run_fold."""
    return _Fixed(Q).touch({}, BARRIER_U, True)


def gaussian_curves(sigma: np.ndarray) -> dict:
    g = GaussianFirstPassage("g", "s")
    ctx = {"s": np.asarray(sigma, np.float64)}
    return {side: g.curve(ctx, side == "up") for side in SIDES}


def battery(curves: dict, M: dict) -> dict:
    """The six numbers scale_adopt.barrier_cols reads off a run_fold row."""
    acc = {k: [] for k in ("pinball", "crps", "brier", "DSC", "MCB", "logs")}
    for side in SIDES:
        Q, y = curves[side], M[side]
        acc["pinball"].append(pinball_curve(Q, y))
        acc["crps"].append(crps_from_curve(Q, y))
        P = touch(Q)
        for k, _ in enumerate(BARRIER_PCT):
            out = (y >= BARRIER_U[k]).astype(float)
            d = corp_decomposition(P[:, k], out)
            acc["brier"].append(d["brier"]); acc["DSC"].append(d["DSC"])
            acc["MCB"].append(d["MCB"]); acc["logs"].append(log_score(P[:, k], out))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def per_episode(curves: dict, M: dict) -> dict:
    """Per-episode Brier, log score, pinball and CRPS; each mean equals the
    corresponding `battery` value to floating-point precision."""
    order = np.argsort(Q_LEVELS_DESC)
    taus = Q_LEVELS_DESC[order]
    out = {k: [] for k in ("brier", "logs", "pinball", "crps")}
    for side in SIDES:
        Q, y = curves[side], M[side]
        L = np.stack([np.maximum(t * (y - Q[:, j]), (t - 1.0) * (y - Q[:, j]))
                      for j, t in enumerate(Q_LEVELS_DESC)], axis=1)   # (n, J)
        out["pinball"].append(L.mean(axis=1))
        out["crps"].append(2.0 * np.trapezoid(L[:, order], taus, axis=1))
        P = np.clip(touch(Q), 1e-6, 1 - 1e-6)
        for k, _ in enumerate(BARRIER_PCT):
            o = (y >= BARRIER_U[k]).astype(float)
            p = P[:, k]
            out["brier"].append((p - o) ** 2)
            out["logs"].append(-(o * np.log(p) + (1 - o) * np.log(1 - p)))
    return {k: np.mean(np.stack(v, 0), axis=0) for k, v in out.items()}


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(2)
    n = 400
    sig = np.exp(rng.normal(np.log(0.02), 0.3, n))
    M = {s: np.abs(rng.normal(0, 1, n)) * sig * rng.uniform(0.5, 1.5, n)
         for s in SIDES}
    cur = gaussian_curves(sig)
    b = battery(cur, M)
    pe = per_episode(cur, M)
    for k in ("brier", "logs", "pinball", "crps"):
        ok.append((f"per-episode {k} averages back to the battery",
                   abs(pe[k].mean() - b[k]) < 1e-12))
    ok.append(("a constant forecast has DSC 0",
               abs(battery(gaussian_curves(np.full(n, 0.02)), M)["DSC"]) < 1e-12))
    ok.append(("touch probabilities fall with barrier distance",
               bool(np.all(np.diff(touch(cur["up"]), axis=1) <= 1e-12))))
    ok.append(("alpha grid matches the committee's", np.allclose(ALPHA_GRID,
                                                                1 - Q_LEVELS_DESC)))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
