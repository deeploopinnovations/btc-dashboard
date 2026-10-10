"""
context/evaluate.py
=====================================================================
Does a context source carry information the trader does not already have?

For each source group, three arms on NOCTUA-Trader's own dataset
(model/policy/dataset.py) and its v1 configuration:

    base     PR #15's features
    real     base + the source's point-in-time features
    placebo  base + the same features with days shuffled within each split
             (same values, same distribution, timing destroyed)

Two tests per arm, both fit on train(+val) and scored ONCE on test
(anchors >= 2024-07-01, where NOCTUA's own forecasts are out of sample):

  forecast  ridge regressions on the production phase (17:00 UTC) for
            log|24 h return| (volatility) and the 24 h return itself
            (direction); paired block-bootstrap CI of the test loss change
  trader    policy/train.run with the v1 config: test Sharpe, and a paired
            block-bootstrap CI of Sharpe(real) - Sharpe(placebo)

A source COUNTS only if `real` beats both `base` and `placebo` on test with
the 95% CI excluding zero. This is PR #14's forward-holdout rule applied to
the historical test split.

Run from model/ (model/policy is on main since PR #15):
    cd model && python3 -m context.evaluate --groups all --n-placebo-trader 99
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from context import features as CF  # noqa: E402

OUT = HERE.parents[2] / "data" / "context" / "evaluation"


def shuffle_days(F: pd.DataFrame, D: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Placebo: within each split and anchor hour, give each day the feature
    row of a randomly chosen other day."""
    rng = np.random.default_rng(seed)
    P = F.copy()
    for _, idx in D.groupby(["split", "hour"]).groups.items():
        idx = np.asarray(idx)
        P.iloc[idx] = F.iloc[rng.permutation(idx)].to_numpy()
    return P


def shift_days(F: pd.DataFrame, D: pd.DataFrame, seed: int, min_gap_days: int = 90) -> pd.DataFrame:
    """Placebo that keeps each feature's own autocorrelation: within each split,
    rotate the daily feature sequence by a random offset of at least
    min_gap_days, the same offset for every anchor hour."""
    rng = np.random.default_rng(seed)
    P = F.copy()
    for split, g in D.groupby("split"):
        days = (g.anchor_ts // 86400).to_numpy()
        n_days = days.max() - days.min() + 1
        if n_days <= 2 * min_gap_days:
            continue
        k = int(rng.integers(min_gap_days, n_days - min_gap_days))
        for _, gh in g.groupby("hour"):
            idx = gh.sort_values("anchor_ts").index.to_numpy()
            P.iloc[idx] = F.iloc[np.roll(idx, k)].to_numpy()
    return P


def _block_ci(delta: np.ndarray, block: int = 20, reps: int = 2000, seed: int = 0):
    """Mean of a paired per-day loss difference with circular block bootstrap."""
    n = len(delta)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    m = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, n, nb)
        m[r] = delta[((st[:, None] + np.arange(block)) .ravel()[:n]) % n].mean()
    return {"point": float(delta.mean()), "ci95": [float(np.quantile(m, .025)), float(np.quantile(m, .975))]}


def ridge_test(D, cols, lam=10.0):
    """Fit on train+val production-phase rows, predict test. Returns per-day
    squared errors for the vol and direction targets."""
    prod = D[D.hour == 17].sort_values("anchor_ts")
    fit, te = prod[prod.split.isin(["train", "val"])], prod[prod.split == "test"]
    cols = [c for c in cols if fit[c].std() > 1e-9]     # constant in fit: no information
    X = fit[cols].to_numpy(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = lambda A: np.clip((A[cols].to_numpy(np.float64) - mu) / sd, -5, 5)  # noqa: E731
    out = {}
    for name, y_of in {"vol": lambda F: np.log(np.abs(F.tgt_ret.to_numpy()) + 1e-3),
                       "dir": lambda F: F.tgt_ret.to_numpy() * 100}.items():
        Xf = np.c_[np.ones(len(fit)), Z(fit)]
        y = y_of(fit)
        beta = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ y)
        pred = np.c_[np.ones(len(te)), Z(te)] @ beta
        out[name] = (pred - y_of(te)) ** 2
    return out


def _rank_p(real: float, placebos: list[float], higher_is_better: bool) -> float:
    """Share of (placebos + real) at least as good as real: the permutation p."""
    pl = np.asarray(placebos)
    better = (pl >= real) if higher_is_better else (pl <= real)
    return float((1 + better.sum()) / (1 + len(pl)))


def evaluate_group(group, D, DS, TR, BT, cfg, trader=True, seed=0,
                   n_placebo_forecast=39, n_placebo_trader=19, placebo_kind="shift"):
    t = D.anchor_ts.to_numpy(np.int64)
    F = CF.build(group, t)
    real_cols = list(F.columns)
    base_cols = DS.feature_cols(D)
    D0 = D.reset_index(drop=True)
    Dr = pd.concat([D0, F], axis=1)
    scramble = shift_days if placebo_kind == "shift" else shuffle_days
    placebo = lambda k: pd.concat([D0, scramble(F, D0, seed + 1000 + k)], axis=1)  # noqa: E731

    cov = {s: float(Dr.loc[Dr.split == s, f"{group}_present"].mean()) for s in ("train", "val", "test")}
    res = {"group": group, "features": real_cols, "coverage": cov, "placebo": placebo_kind}
    if cov["train"] < 0.2:
        res["verdict"] = "UNTESTABLE: under 20% of training days have this source"
        return res

    eb, er = ridge_test(Dr, base_cols), ridge_test(Dr, base_cols + real_cols)
    eps = [ridge_test(placebo(k), base_cols + real_cols) for k in range(n_placebo_forecast)]
    fc = {}
    for k in ("vol", "dir"):
        pm = [float(e[k].mean()) for e in eps]
        fc[k] = {"base_mse": float(eb[k].mean()), "real_mse": float(er[k].mean()),
                 "placebo_mse_median": float(np.median(pm)),
                 "real_vs_base": _block_ci(eb[k] - er[k]),
                 "real_vs_base_pct": 100 * float((eb[k] - er[k]).mean() / eb[k].mean()),
                 "placebo_rank_p": _rank_p(float(er[k].mean()), pm, higher_is_better=False)}
        fc[k]["counts"] = bool(fc[k]["real_vs_base"]["ci95"][0] > 0 and fc[k]["placebo_rank_p"] <= 0.05)
    res["forecast"] = fc

    if trader:
        keys = ("sharpe", "cagr", "max_drawdown", "avg_abs_position")
        with tempfile.TemporaryDirectory() as tmp:
            def arm(name, frame):
                (Path(tmp) / name).mkdir()
                r = TR.run(cfg, Path(tmp) / name, final=True, D=frame)
                return r, np.load(Path(tmp) / name / "test_daily.npz")["pnl"]
            rr, pr = arm("real", Dr)
            pls = [arm(f"p{k}", placebo(k)) for k in range(n_placebo_trader)]
        ps = [r["test"]["sharpe"] for r, _ in pls]
        tr = {"real": {k: rr["test"][k] for k in keys},
              "placebo_sharpes": ps, "placebo_sharpe_median": float(np.median(ps)),
              "placebo_rank_p": _rank_p(rr["test"]["sharpe"], ps, higher_is_better=True),
              "real_vs_placebo0": BT.block_bootstrap_sharpe_diff(pr, pls[0][1]),
              "features_used": [c for c in rr["features"] if c in real_cols]}
        tr["counts"] = bool(tr["placebo_rank_p"] <= 0.05)
        res["trader"] = tr
    counts = [k for k, v in fc.items() if v["counts"]] + (["trader"] if res.get("trader", {}).get("counts") else [])
    res["verdict"] = f"COUNTS ({', '.join(counts)})" if counts else "DOES NOT COUNT"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="all")
    ap.add_argument("--no-trader", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-placebo-trader", type=int, default=19)
    ap.add_argument("--placebo", choices=["shift", "shuffle"], default="shift",
                    help="shift keeps autocorrelation (default); shuffle permutes days")
    ap.add_argument("--tag", default="", help="suffix for the result file")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from policy import backtest as BT, dataset as DS, train as TR
    D = DS.load().reset_index(drop=True)
    meta = json.loads((Path(DS.__file__).parent / "weights/noctua_trader_v1_metrics.json").read_text())
    cfg = meta["config"]
    groups = list(CF.GROUPS) if a.groups == "all" else a.groups.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    base = None
    if not a.no_trader:
        with tempfile.TemporaryDirectory() as tmp:
            r = TR.run(cfg, Path(tmp), final=True, D=D)
            base = {k: r["test"][k] for k in ("sharpe", "cagr", "max_drawdown", "avg_abs_position")}
        print("base trader test", base, flush=True)
    for g in groups:
        if not (CF.pit.CONTEXT / f"{g}.parquet").exists():
            print(g, "missing, skipped"); continue
        res = evaluate_group(g, D, DS, TR, BT, cfg, trader=not a.no_trader, seed=a.seed,
                             n_placebo_trader=a.n_placebo_trader, placebo_kind=a.placebo)
        if base:
            res.setdefault("trader", {})["base"] = base
            if "real" in res["trader"]:
                tr = res["trader"]
                tr["counts"] = bool(tr["counts"] and tr["real"]["sharpe"] > base["sharpe"])
                counts = [k for k, v in res["forecast"].items() if v["counts"]] + (["trader"] if tr["counts"] else [])
                res["verdict"] = f"COUNTS ({', '.join(counts)})" if counts else "DOES NOT COUNT"
        (OUT / f"{g}{a.tag}.json").write_text(json.dumps(res, indent=2, default=float))
        print(g, res["verdict"], json.dumps({k: res.get(k) for k in ("coverage",)}), flush=True)
        if "forecast" in res:
            for k, v in res["forecast"].items():
                print(f"  {k}: vs base {v['real_vs_base_pct']:+.2f}% {v['real_vs_base']['ci95']}  placebo p {v['placebo_rank_p']:.3f}")
        if "trader" in res and "real" in res["trader"]:
            t_ = res["trader"]
            print(f"  trader: sharpe {t_['real']['sharpe']:.3f} vs base {t_['base']['sharpe']:.3f}, placebo median {t_['placebo_sharpe_median']:.3f}, p {t_['placebo_rank_p']:.3f}")


if __name__ == "__main__":
    main()
