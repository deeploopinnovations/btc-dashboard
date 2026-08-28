"""
eval/econ_voltarget.py
=====================================================================
The economic test that CAN be run here, pre-registered as `econ-voltarget`.

WHAT IS NOT HERE, AND WHY

There is no options P&L in this file and there will not be one. `econ-scope`
records the evidence: `model/artifacts/datasources.json` logs 18 probes and 2
reachable endpoints, every exchange API and aggregator returning 403 through
the container's egress proxy. The only option-adjacent series on disk is the
Deribit DVOL volatility INDEX -- an index level, with no strikes, no expiries,
no bid/ask, no size and no prints. An options P&L could therefore only be
SIMULATED, and every assumption the simulation needed (a pricing model, a
spread, a fill rule) would do more work than the forecast being tested.

A directional backtest is also absent, for a different reason: `D1` measured
the signal to carry no information at n ~ 49,000 per horizon, so its equity
curve would be a random walk with a fee drag.

WHAT IS HERE

A volatility-targeting overlay on spot BTC, which is the actual commercial use
of a volatility forecast for anyone without an options book. At each anchor,

    w_t = clip(target_window_sigma / sigma_hat_t, 0, W_MAX)

and the portfolio return over the window is w_t * R_t. Every arm sees the same
anchors, the same target and the same cap; the only thing that varies is which
forecast supplies sigma_hat.

THE PRIMARY IS RISK CONTROL, NOT RETURN

|realised annualised volatility - target|. Return-based quantities (Sharpe,
cost-adjusted return, drawdown) are reported and are explicitly NOT pass
conditions, because BTC's spot drift dominates them and no volatility forecast
controls it: an arm can win on Sharpe purely by having been more levered during
a bull fold. Judging a volatility forecast on P&L is how a risk model gets
credit for a trend.

NON-OVERLAPPING WINDOWS

The episode table is anchored hourly, so H=24 episodes overlap 23 hours out of
24 and a portfolio built from all of them would be rebalancing 24 times a day
into the same exposure. Only anchors at a single hour of the day are used, so
consecutive windows tile without overlap and the turnover figure means what it
says.

COSTS ARE ASSUMPTIONS, AND ARE TREATED AS SUCH

This repository has no order book and no fee schedule. 10 bps round-trip is a
stated premise, not a measurement, and 5 and 25 bps are reported beside it. If
the ranking of arms changes across the three, the result is COST-DEPENDENT and
no arm is declared better.

    python -m model.eval.econ_voltarget
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                       # noqa: E402
from eval.vol_matrix import build_h4_table                               # noqa: E402
from eval.vol_matrix import run_fold as vm_run_fold                      # noqa: E402
from noctua import splits as S                                           # noqa: E402

HOURS_PER_YEAR = 24 * 365.0
TARGET_ANNUAL = 0.60
W_MAX = 3.0
COST_BPS = (5.0, 10.0, 25.0)
BASE_COST_BPS = 10.0
REBALANCE_HOUR = 0                 # UTC; H=24 windows from here tile the day


def realised_annual_vol(r: np.ndarray, H: float) -> float:
    """Annualised volatility of a series of non-overlapping H-hour returns."""
    return float(np.std(r, ddof=1) * np.sqrt(HOURS_PER_YEAR / H))


def simulate(sig: np.ndarray, R: np.ndarray, H: float, w_const: float,
             cost_bps: float) -> dict:
    """One arm's overlay. `w_const` is only used for the control arm, which
    passes sig=None."""
    tgt = TARGET_ANNUAL * np.sqrt(H / HOURS_PER_YEAR)
    if sig is None:
        w = np.full(len(R), w_const)
    else:
        w = np.clip(tgt / np.maximum(sig, 1e-9), 0.0, W_MAX)
    # Turnover is charged on the CHANGE in exposure, with the first period
    # charged from flat -- an overlay that starts fully invested for free
    # would be given a position it never paid for.
    dw = np.abs(np.diff(np.concatenate([[0.0], w])))
    cost = dw * (cost_bps / 1e4)
    net = w * R - cost
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    rv = realised_annual_vol(w * R, H)
    return {
        "realised_vol": rv,
        "vol_error": abs(rv - TARGET_ANNUAL),
        "turnover": float(np.mean(dw)),
        "gross_return": float(np.sum(w * R)),
        "net_return": float(np.sum(net)),
        "cost_drag": float(np.sum(cost)),
        "sharpe": float(np.mean(net) / (np.std(net, ddof=1) + 1e-12)
                        * np.sqrt(HOURS_PER_YEAR / H)),
        "max_drawdown": float(np.max(peak - eq)),
        "mean_w": float(np.mean(w)),
        "net_series": net,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="volatility-targeting overlay")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/econ_voltarget.json"))
    a = ap.parse_args(argv)

    ep, X = build_h4_table(a.artifacts)
    keep = (ep.H == a.horizon).to_numpy()
    ep, X = ep[keep].reset_index(drop=True), X[keep].reset_index(drop=True)

    try:
        from eval.garch import hourly_returns
        ret = hourly_returns(a.artifacts)
    except Exception as e:                                        # noqa: BLE001
        print(f"GARCH unavailable ({e}); those arms are omitted, not faked")
        ret = None

    folds = S.walk_forward_folds(ep)
    print(f"H={a.horizon}, {len(folds)} folds, rebalancing at {REBALANCE_HOUR:02d}:00 UTC "
          f"so consecutive windows do not overlap")
    print(f"target {TARGET_ANNUAL:.0%} annualised, w capped at {W_MAX}, "
          f"costs {COST_BPS} bps round-trip\n")

    per_fold = {}
    for f in folds:
        t0 = time.time()
        f = dict(f)
        f["train_end_ts"] = int(ep.anchor_ts.to_numpy()[f["train"]].max())
        r = vm_run_fold(ep, X, f, ret, hidden=a.hidden, seeds=a.seeds)
        if r is None or a.horizon not in r:
            print(f"  fold {f['year']}: skipped"); continue
        d = r[a.horizon]
        hour = (d["anchor_ts"] // 3600) % 24
        sel = hour == REBALANCE_HOUR
        if sel.sum() < 60:
            print(f"  fold {f['year']}: only {sel.sum()} non-overlapping anchors"); continue
        # the constant-weight control's weight is set on TRAIN, never on test
        tgt = TARGET_ANNUAL * np.sqrt(a.horizon / HOURS_PER_YEAR)
        tr_sig = np.exp(X.loc[f["train"], "har_1d"].to_numpy()) * np.sqrt(a.horizon)
        w_const = float(np.clip(np.mean(tgt / np.maximum(tr_sig, 1e-9)), 0.0, W_MAX))

        arms = {k: v[sel] for k, v in d["sigma"].items()}
        R = d["R"][sel]
        row = {"constant_w": {c: simulate(None, R, a.horizon, w_const, c) for c in COST_BPS}}
        for k, sg in arms.items():
            row[k] = {c: simulate(sg, R, a.horizon, w_const, c) for c in COST_BPS}
        per_fold[f["year"]] = {"n": int(sel.sum()), "w_const": w_const, "arms": row}
        print(f"  fold {f['year']}  n={int(sel.sum()):4d}  w_const={w_const:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    if not per_fold:
        print("no usable folds"); return 1

    years = sorted(per_fold)
    names = sorted(set().intersection(*[set(per_fold[y]["arms"]) for y in years]))
    order = [k for k in ("noctua", "noctua40", "constant_w") if k in names]
    order += [k for k in names if k not in order]

    out = {"target": TARGET_ANNUAL, "w_max": W_MAX, "horizon": a.horizon,
           "rebalance_hour": REBALANCE_HOUR, "cost_bps": list(COST_BPS),
           "years": years, "arms": {}}

    # PRIMARY: |realised vol - target|, per fold, at the base cost. Weights do
    # not depend on the cost, so this number is the same at all three; the
    # cost sensitivity lives in the return columns.
    print("\n" + "=" * 104)
    print(f"PRIMARY -- |realised annualised vol - {TARGET_ANNUAL:.0%}| at "
          f"{BASE_COST_BPS:.0f} bps.  Lower is better.  Paired CI is against the best arm by CALIB.")
    print("=" * 104)
    print(f"{'arm':>13} {'|err|':>8} {'realised':>9} {'worst':>8} {'turnover':>9} "
          f"{'mean w':>7}   " + "  ".join(f"net@{c:.0f}bp" for c in COST_BPS)
          + "   sharpe@10bp  maxDD")
    err = {}
    for k in order:
        e = np.array([per_fold[y]["arms"][k][BASE_COST_BPS]["vol_error"] for y in years])
        err[k] = e
    best = min(err, key=lambda k: float(np.mean(err[k])))
    for k in order:
        b = [per_fold[y]["arms"][k][BASE_COST_BPS] for y in years]
        nets = ["%+.3f" % np.mean([per_fold[y]["arms"][k][c]["net_return"] for y in years])
                for c in COST_BPS]
        d = err[best] - err[k]                  # positive => k is better
        ci = mean_ci(d, alpha=0.05) if k != best else None
        cis = "—" if ci is None else f"[{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"
        print(f"{k:>13} {np.mean(err[k]):8.4f} "
              f"{np.mean([x['realised_vol'] for x in b]):9.4f} "
              f"{np.max(err[k]):8.4f} "
              f"{np.mean([x['turnover'] for x in b]):9.4f} "
              f"{np.mean([x['mean_w'] for x in b]):7.3f}   "
              + "  ".join(f"{v:>9}" for v in nets)
              + f"   {np.mean([x['sharpe'] for x in b]):10.3f} "
              f"{np.mean([x['max_drawdown'] for x in b]):6.3f}   {cis}")
        out["arms"][k] = {
            "vol_error_mean": float(np.mean(err[k])),
            "vol_error_worst": float(np.max(err[k])),
            "vol_error_per_fold": err[k].tolist(),
            "realised_vol": float(np.mean([x["realised_vol"] for x in b])),
            "turnover": float(np.mean([x["turnover"] for x in b])),
            "mean_w": float(np.mean([x["mean_w"] for x in b])),
            "net_return_by_cost": {str(c): float(np.mean(
                [per_fold[y]["arms"][k][c]["net_return"] for y in years]))
                for c in COST_BPS},
            "sharpe_by_cost": {str(c): float(np.mean(
                [per_fold[y]["arms"][k][c]["sharpe"] for y in years]))
                for c in COST_BPS},
            "max_drawdown": float(np.mean([x["max_drawdown"] for x in b])),
            "paired_ci_vs_best": None if ci is None else list(ci["ci95"]),
        }
    out["best_arm_by_primary"] = best

    # COST DEPENDENCE: does the ranking survive?
    print("\n--- is the ranking cost-dependent? (rank by mean net return at each cost) ---")
    ranks = {}
    for c in COST_BPS:
        v = {k: float(np.mean([per_fold[y]["arms"][k][c]["net_return"] for y in years]))
             for k in order}
        ranks[c] = [k for k in sorted(v, key=v.get, reverse=True)]
        print(f"  {c:>5.0f} bps: " + " > ".join(ranks[c]))
    stable = len({tuple(v) for v in ranks.values()}) == 1
    print(f"  ranking identical at all three cost levels: {stable}")
    out["net_return_ranking_by_cost"] = {str(c): ranks[c] for c in COST_BPS}
    out["ranking_cost_stable"] = bool(stable)

    print(f"\n--- pre-registered rule ---")
    print(f"   best arm on the primary (|realised vol - target|): {best}")
    for k in ("noctua", "noctua40"):
        if k not in out["arms"]:
            continue
        ci = out["arms"][k]["paired_ci_vs_best"]
        v = "n/a (is the best arm)" if ci is None else (
            "CLEARS" if ci[0] > 0 else "DOES NOT CLEAR")
        print(f"   {k} vs {best}: {v}")
    print("   return columns are SECONDARY and are not pass conditions: BTC's spot "
          "drift dominates them and no volatility forecast controls it")

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
