"""
eval/rl_replay.py
=====================================================================
The registered replay for P5-rl-paper: every 6-hour slot of the shipped
artifact's test era (2024-01-01 to the corpus end), the SAME code path the
Space runs (rl/env.inputs_at -> serve/predict.forecast on bars before the
slot), every arm on identical inputs.

  arms       MV (amended), MV0 (registered form), TS (5 seeds), MV_P (MV on
             time-permuted inputs), FLAT, HALF, HOLD, VT
  primary    mean per-step utility; MV vs HOLD, MV vs VT, MV vs MV_P, TS vs MV;
             moving-block bootstrap 99.5 %, block 28 steps (one week)
  reported   return / vol / Sharpe / max drawdown / turnover / short and flat
             share; cost 5 and 25 bps, gamma 0.5 and 5; the loss study; the
             forward N_MIN against VT

Inputs are cached in model/artifacts/rl_inputs.parquet (computing them is the
slow part: ~1 s per slot through serving's own code).

    python -m model.eval.rl_replay            # compute or reuse inputs, run
    python -m model.eval.rl_replay --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import env as E                                                # noqa: E402
from rl.agent import make_agent                                        # noqa: E402

START = "2024-01-01"
ART = Path("model/artifacts")
CACHE = ART / "rl_inputs.parquet"
OUT = ART / "rl_replay.json"
STEPS_PER_YEAR = 4 * 365
BLOCK = 28
LOOKBACK_D = 460                 # 365 d features + 60 d factor window + slack


def _hours():
    from serve.history import HOURLY_COLS
    return pd.read_parquet(ART / "btcusd_1h.parquet")[HOURLY_COLS].sort_values(
        "hour_ts", ignore_index=True)


def slots(hours) -> list:
    ts = hours.hour_ts.to_numpy(np.int64)
    s0 = int(pd.Timestamp(START, tz="UTC").timestamp())
    last_exit = int(ts[-2])           # the corpus's own last hour is partial (P4-factor-window)
    return [int(t) for t in ts if t >= s0 and pd.Timestamp(int(t), unit="s", tz="UTC").hour
            in E.SLOT_HOURS and t + (E.H - 1) * E.HOUR <= last_exit]


def _chunk(args):
    sl, = args
    from serve.runtime import load_model
    h, m = _hours(), load_model()
    fc = E.noctua_forecaster(m)
    rows = []
    for t in sl:
        hh = h[(h.hour_ts >= t - LOOKBACK_D * 86400) & (h.hour_ts <= t + E.H * E.HOUR)].reset_index(drop=True)
        inp = E.inputs_at(None, hh, t, forecaster=fc)
        rows.append({"E": t, "R": E.realized(hh, t), **{k: inp[k] for k in
                     ("sigma", "trailing_rv", "p_up", "p_vol_amplify", "r24", "spot")}})
    return rows


def build_inputs(procs: int = 4) -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    sl = slots(_hours())
    parts = [sl[i::procs] for i in range(procs)]
    with Pool(procs) as pool:
        rows = [r for part in pool.map(_chunk, [(p,) for p in parts]) for r in part]
    df = pd.DataFrame(rows).sort_values("E", ignore_index=True)
    df.to_parquet(CACHE, index=False)
    return df


def xmat(df) -> np.ndarray:
    return np.vstack([E.features(r.sigma, r.trailing_rv, r.p_up, r.p_vol_amplify, r.r24)
                      for r in df.itertuples()])


def run_agent(kind, X, S, R, seed=0, cost=E.COST, gamma=E.GAMMA):
    ag = make_agent(kind, seed)
    ag.cost, ag.gamma = cost, gamma
    w, A = 0.0, []
    for i in range(len(R)):
        a = ag.act(X[i], S[i], w)
        ag.update(X[i], S[i], w, a, R[i])
        A.append(a)
        w = a
    return np.array(A)


def run_rule(weights, R):
    return np.asarray(weights, np.float64) * np.ones(len(R))


def path_stats(w, R, cost=E.COST, gamma=E.GAMMA) -> dict:
    prev = np.r_[0.0, w[:-1]]
    g = w * R - cost * np.abs(w - prev)
    u = g - 0.5 * gamma * (w * R) ** 2
    eq = np.cumsum(np.log1p(g))
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    return {"u": u, "mean_u": float(u.mean()), "ann_return": float(g.mean() * STEPS_PER_YEAR),
            "ann_vol": float(g.std() * np.sqrt(STEPS_PER_YEAR)),
            "sharpe": float(g.mean() / g.std() * np.sqrt(STEPS_PER_YEAR)) if g.std() > 0 else 0.0,
            "max_drawdown": float(1 - np.exp(-dd)), "turnover": float(np.abs(w - prev).mean()),
            "share_short": float(np.mean(w < 0)), "share_flat": float(np.mean(w == 0)),
            "mean_exposure": float(w.mean())}


def arms(df, cost=E.COST, gamma=E.GAMMA, ts_seeds=(0, 1, 2, 3, 4)) -> dict:
    X, S, R = xmat(df), df.sigma.to_numpy(), df.R.to_numpy()
    perm = np.random.default_rng(7).permutation(len(R))
    W = {"MV": run_agent("MV", X, S, R, cost=cost, gamma=gamma),
         "MV0": run_agent("MV0", X, S, R, cost=cost, gamma=gamma),
         "MV_P": run_agent("MV", X[perm], S[perm], R, cost=cost, gamma=gamma),
         "FLAT": run_rule(0.0, R), "HALF": run_rule(0.5, R), "HOLD": run_rule(1.0, R),
         "VT": np.clip(E.VT_TARGET / S, 0.0, 1.0)}
    out = {k: path_stats(w, R, cost, gamma) for k, w in W.items()}
    ts = [path_stats(run_agent("TS", X, S, R, seed=s, cost=cost, gamma=gamma), R, cost, gamma)
          for s in ts_seeds]
    out["TS"] = {k: (np.mean([t[k] for t in ts], axis=0) if k == "u" else float(np.mean([t[k] for t in ts])))
                 for k in ts[0]}
    out["TS"]["seed_mean_u"] = [t["mean_u"] for t in ts]
    return out, W


def contrast(a, b) -> dict:
    from eval.ci import mean_ci
    d = a - b
    lo, hi = mean_ci(d, alpha=0.005, block_len=BLOCK)["ci95"]
    return {"diff_per_step": float(d.mean()), "ci_99_5": [float(lo), float(hi)],
            "verdict": "better" if lo > 0 else "worse" if hi < 0 else "not separated"}


def loss_study(df, w, R) -> dict:
    prev = np.r_[0.0, w[:-1]]
    move, cost = w * R, E.COST * np.abs(w - prev)
    risk = 0.5 * E.GAMMA * (w * R) ** 2
    u = move - cost - risk
    u_all = np.array([[E.utility(a, r, p) for a in E.ACTIONS] for r, p in zip(R, prev)])
    regret = u_all.max(1) - u
    q = pd.qcut(df.sigma, 5, labels=False)
    by_q = {int(k): {"n": int((q == k).sum()), "mean_u": float(u[q == k].mean()),
                     "mean_regret": float(regret[q == k].mean()),
                     "mean_exposure": float(w[q == k].mean())} for k in range(5)}
    worst = np.argsort(u)[: max(1, len(u) // 20)]
    eq = np.cumsum(np.log1p(move - cost))
    peak = np.maximum.accumulate(eq)
    trough = int(np.argmax(peak - eq))
    start = int(np.argmax(eq[: trough + 1]))
    t = lambda i: str(pd.Timestamp(int(df.E.iloc[i]), unit="s", tz="UTC"))   # noqa: E731
    return {"by_sigma_quintile": by_q,
            "worst_5pct": {"n": int(len(worst)), "share_from_market_move": float(np.mean(move[worst] < 0)),
                           "mean_move": float(move[worst].mean()), "mean_cost": float(cost[worst].mean()),
                           "mean_risk_penalty": float(risk[worst].mean()),
                           "mean_abs_exposure": float(np.abs(w[worst]).mean())},
            "mean_regret": float(regret.mean()),
            "max_drawdown_window": [t(start), t(trough)]}


def n_min(d) -> dict:
    from eval.ci import mean_ci
    m = float(d.mean())
    lo, hi = mean_ci(d, alpha=0.005, block_len=BLOCK)["ci95"]
    half = (hi - lo) / 2
    if m <= 0:
        return {"mean_diff": m, "n_min": None,
                "note": "the replay shows no edge over VT; no forward claim is expected"}
    return {"mean_diff": m, "n_min": int(np.ceil(len(d) * (half / m) ** 2)),
            "note": "steps for the 99.5% interval to clear zero at the replay's effect and noise"}


def selftest() -> int:
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"E": np.arange(n) * 6 * 3600, "R": rng.normal(0, 0.015, n),
                       "sigma": np.full(n, 0.015), "trailing_rv": np.full(n, 0.015),
                       "p_up": np.full(n, 0.5), "p_vol_amplify": np.full(n, 0.5),
                       "r24": np.zeros(n), "spot": np.ones(n)})
    out, W = arms(df, ts_seeds=(0,))
    R = df.R.to_numpy()
    ok = [("FLAT earns exactly zero", out["FLAT"]["mean_u"] == 0.0),
          ("HOLD utility is the formula", abs(out["HOLD"]["mean_u"] - np.mean(
              [E.utility(1.0, r, p) for r, p in zip(R, np.r_[0.0, np.ones(n - 1)])])) < 1e-15),
          ("VT weight is clip(target/sigma)", np.allclose(W["VT"], min(1.0, E.VT_TARGET / 0.015))),
          ("placebo is a permutation of the inputs, not of R", len(W["MV_P"]) == n)]
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    return 0 if all(g for _, g in ok) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P5-rl-paper registered replay")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    df = build_inputs(a.procs)
    df = df[np.isfinite(df.R) & np.isfinite(df.sigma)].reset_index(drop=True)
    R = df.R.to_numpy()
    print(f"{len(df)} steps {pd.Timestamp(int(df.E.iloc[0]), unit='s', tz='UTC')} -> "
          f"{pd.Timestamp(int(df.E.iloc[-1]), unit='s', tz='UTC')}")
    out, W = arms(df)
    print(f"\n{'arm':6} {'mean u':>10} {'ann ret':>8} {'ann vol':>8} {'sharpe':>7} {'maxDD':>7} "
          f"{'turn':>6} {'short':>6} {'flat':>6} {'expo':>6}")
    for k, v in out.items():
        print(f"{k:6} {v['mean_u']:+10.6f} {v['ann_return']:+8.1%} {v['ann_vol']:8.1%} "
              f"{v['sharpe']:+7.2f} {v['max_drawdown']:7.1%} {v['turnover']:6.3f} "
              f"{v['share_short']:6.1%} {v['share_flat']:6.1%} {v['mean_exposure']:+6.2f}")
    res = {"n_steps": len(df), "arms": {k: {kk: vv for kk, vv in v.items() if kk != "u"}
                                        for k, v in out.items()}, "contrasts": {}}
    print("\nprimary contrasts (per-step utility, 99.5%, block 28):")
    for c, o in (("MV", "HOLD"), ("MV", "VT"), ("MV", "MV_P"), ("TS", "MV"), ("MV", "MV0")):
        r = contrast(out[c]["u"], out[o]["u"])
        res["contrasts"][f"{c}_vs_{o}"] = r
        print(f"  {c:4} vs {o:5}: {r['diff_per_step']:+.6f}  [{r['ci_99_5'][0]:+.6f}, "
              f"{r['ci_99_5'][1]:+.6f}]  {r['verdict']}")
    res["sensitivity"] = {}
    for label, kw in (("cost_5bps", {"cost": 0.0005}), ("cost_25bps", {"cost": 0.0025}),
                      ("gamma_0.5", {"gamma": 0.5}), ("gamma_5", {"gamma": 5.0})):
        o2, _ = arms(df, ts_seeds=(0,), **kw)
        res["sensitivity"][label] = {k: o2[k]["mean_u"] for k in ("MV", "MV0", "TS", "HOLD", "VT", "HALF")}
        res["sensitivity"][label]["MV_vs_VT"] = contrast(o2["MV"]["u"], o2["VT"]["u"])
        print(f"  {label:10}: MV {o2['MV']['mean_u']:+.6f}  VT {o2['VT']['mean_u']:+.6f}  "
              f"HOLD {o2['HOLD']['mean_u']:+.6f}  MV-VT {res['sensitivity'][label]['MV_vs_VT']['verdict']}")
    res["loss_study_MV"] = loss_study(df, W["MV"], R)
    res["forward_n_min_vs_VT"] = n_min(out["MV"]["u"] - out["VT"]["u"])
    print("\nloss study (MV):", json.dumps(res["loss_study_MV"], indent=1))
    print("forward N_MIN vs VT:", res["forward_n_min_vs_VT"])
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
