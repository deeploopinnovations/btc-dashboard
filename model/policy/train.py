"""
policy/train.py
=====================================================================
Trains NOCTUA-Trader: a small MLP that maps what is known at the anchor to
a position in [-L, L] for the next 24 h, optimised END TO END on net P&L
(fees and funding inside the loss), not on a price forecast.

    python policy/train.py --config policy/configs/base.json --out runs/base
    python policy/train.py ... --final        # also scores the TEST split

Without --final the test split is never touched: model selection (epoch and
configuration) uses the validation split only, scored on the production
phase (17:00 UTC), the way it would be traded.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import backtest as BT              # noqa: E402
from policy import dataset as DS               # noqa: E402

DEFAULT = {
    "features": "all",          # all | noctua_only | no_noctua_forecast | price_only
    "drop_features": [],
    "hidden": [32, 16],
    "dropout": 0.1,
    "weight_decay": 1e-3,
    "lr": 1e-3,
    "epochs": 60,
    "patience": 15,
    "batch": 64,
    "seq_len": 64,
    "phases": "all",            # all | prod  (training augmentation)
    "objective": "sharpe",      # sharpe | utility | logwealth
    "gamma": 5.0,               # risk aversion for 'utility'
    "max_leverage": 1.0,
    "long_only": False,
    "fee_bps": BT.DEFAULT_FEE_BPS,
    "turnover_penalty_bps": 0.0,  # extra training-only cost on turnover
    "seeds": 5,
    "input_noise": 0.0,
}


# ---------------------------------------------------------------- features
def select_features(D: pd.DataFrame, cfg: dict) -> list[str]:
    cols = DS.feature_cols(D)
    f = cfg["features"]
    if f == "noctua_only":
        cols = [c for c in cols if c.startswith(("nf_", "nx_"))]
    elif f == "no_noctua_forecast":
        cols = [c for c in cols if not c.startswith("nx_")]
    elif f == "price_only":
        cols = [c for c in cols if c.startswith(("ret_", "dist_ma", "hour_", "dow_"))]
    elif f != "all":
        raise ValueError(f"unknown feature set {f}")
    cols = [c for c in cols if c not in set(cfg["drop_features"])]
    # constant columns carry nothing and break scaling
    return [c for c in cols if D.loc[D.split == "train", c].std() > 1e-9]


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(X, axis=0)
    iqr = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
    scale = np.where(iqr > 1e-9, iqr / 1.349, X.std(axis=0) + 1e-9)
    return med, scale


def apply_scaler(X, med, scale, clip=5.0):
    return np.clip((X - med) / scale, -clip, clip)


# ---------------------------------------------------------------- model
class Policy(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float,
                 max_lev: float, long_only: bool):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.max_lev, self.long_only = max_lev, long_only

    def forward(self, x):
        z = self.net(x).squeeze(-1)
        return self.max_lev * (torch.sigmoid(z) if self.long_only else torch.tanh(z))


def seq_pnl(pos, ret, fund, fee):
    """pos/ret/fund: (B, T+1); the first column is only the 'previous' day."""
    R = torch.expm1(ret[:, 1:])
    p = pos[:, 1:]
    return p * R - fee * (p - pos[:, :-1]).abs() - p * fund[:, 1:]


def objective(pl, cfg):
    if cfg["objective"] == "sharpe":
        flat = pl.reshape(-1)
        return -(flat.mean() / (flat.std() + 1e-6)) * np.sqrt(BT.DAYS)
    if cfg["objective"] == "utility":
        return -(pl.mean() - 0.5 * cfg["gamma"] * pl.var())
    if cfg["objective"] == "logwealth":
        return -torch.log1p(pl.clamp(min=-0.99)).mean()
    raise ValueError(cfg["objective"])


# ---------------------------------------------------------------- data prep
def phase_arrays(D, cols, med, scale, hours=None):
    """Per-phase, time-ordered (X, ret, fund) arrays."""
    out = []
    for h, g in D.groupby("hour"):
        if hours is not None and h not in hours:
            continue
        g = g.sort_values("anchor_ts")
        out.append((apply_scaler(g[cols].to_numpy(np.float64), med, scale),
                    g.tgt_ret.to_numpy(), g.tgt_fund.to_numpy()))
    return out


def sample_batch(phases, B, T, rng):
    xs, rs, fs = [], [], []
    for _ in range(B):
        X, r, f = phases[rng.integers(len(phases))]
        s = rng.integers(0, len(X) - T - 1)
        xs.append(X[s:s + T + 1]); rs.append(r[s:s + T + 1]); fs.append(f[s:s + T + 1])
    t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)  # noqa: E731
    return t(xs), t(rs), t(fs)


def eval_positions(models, X):
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        return np.mean([m.eval()(xt).numpy() for m in models], axis=0)


def prod_frame(D, split):
    return D[(D.split == split) & (D.hour == DS.PROD_HOUR)].sort_values("anchor_ts")


# ---------------------------------------------------------------- train
def train_one(seed, cfg, tr_phases, Xv, rv, fv, n_in):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Policy(n_in, cfg["hidden"], cfg["dropout"], cfg["max_leverage"], cfg["long_only"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    fee = (cfg["fee_bps"] + cfg["turnover_penalty_bps"]) * 1e-4
    steps = max(1, sum(len(p[0]) for p in tr_phases) // (cfg["batch"] * cfg["seq_len"]))
    best, best_state, bad, hist = -np.inf, None, 0, []
    for ep in range(cfg["epochs"]):
        m.train()
        for _ in range(steps):
            x, r, f = sample_batch(tr_phases, cfg["batch"], cfg["seq_len"], rng)
            if cfg["input_noise"] > 0:
                x = x + cfg["input_noise"] * torch.randn_like(x)
            loss = objective(seq_pnl(m(x), r, f, fee), cfg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        pv = eval_positions([m], Xv)
        sv = BT.metrics(BT.pnl(pv, rv, fv, cfg["fee_bps"]))["sharpe"]
        hist.append(round(sv, 4))
        if sv > best:
            best, best_state, bad = sv, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    m.load_state_dict(best_state)
    return m.eval(), {"seed": seed, "best_val_sharpe": best, "val_curve": hist}


def run(cfg: dict, out: Path, final: bool = False, D: pd.DataFrame | None = None) -> dict:
    t0 = time.time()
    cfg = {**DEFAULT, **cfg}
    D = DS.load() if D is None else D
    cols = select_features(D, cfg)
    tr = D[D.split == "train"]
    med, scale = fit_scaler(tr[cols].to_numpy(np.float64))
    hours = None if cfg["phases"] == "all" else {DS.PROD_HOUR}
    tr_phases = phase_arrays(tr, cols, med, scale, hours)

    V = prod_frame(D, "val")
    Xv = apply_scaler(V[cols].to_numpy(np.float64), med, scale)
    rv, fv = V.tgt_ret.to_numpy(), V.tgt_fund.to_numpy()

    models, seed_logs = [], []
    for s in range(cfg["seeds"]):
        m, log = train_one(s, cfg, tr_phases, Xv, rv, fv, len(cols))
        models.append(m); seed_logs.append(log)

    res = {"config": cfg, "n_features": len(cols), "features": cols,
           "n_params_per_seed": sum(p.numel() for p in models[0].parameters()),
           "seeds": seed_logs}
    for split in ["train", "val"] + (["test"] if final else []):
        F = prod_frame(D, split)
        X = apply_scaler(F[cols].to_numpy(np.float64), med, scale)
        pos = eval_positions(models, X)
        pl = BT.pnl(pos, F.tgt_ret.to_numpy(), F.tgt_fund.to_numpy(), cfg["fee_bps"])
        res[split] = BT.metrics(pl, pos)
        base = BT.baselines(F, cfg["fee_bps"])
        res[split]["baselines"] = {k: v["metrics"] for k, v in base.items()}
        res[split]["vs_buy_and_hold"] = BT.block_bootstrap_sharpe_diff(pl, base["buy_and_hold"]["pnl"])
        res[split]["vs_vol_target"] = BT.block_bootstrap_sharpe_diff(pl, base["noctua_vol_target_long"]["pnl"])
        if split != "train":
            np.savez_compressed(out / f"{split}_daily.npz", anchor_ts=F.anchor_ts.to_numpy(),
                                pos=pos, pnl=pl, ret=F.tgt_ret.to_numpy())
    res["elapsed_s"] = round(time.time() - t0, 1)

    out.mkdir(parents=True, exist_ok=True)
    export(models, cols, med, scale, cfg, out / "policy_weights.npz")
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def export(models, cols, med, scale, cfg, path: Path):
    """Plain-NumPy artifact: per-seed Linear weights + scaler + metadata."""
    arrs = {"scaler_med": med, "scaler_scale": scale}
    for s, m in enumerate(models):
        lin = [mod for mod in m.net if isinstance(mod, nn.Linear)]
        for i, l in enumerate(lin):
            arrs[f"s{s}.l{i}.W"] = l.weight.detach().numpy().astype(np.float32)
            arrs[f"s{s}.l{i}.b"] = l.bias.detach().numpy().astype(np.float32)
    meta = {"name": "NOCTUA-Trader", "version": 1, "features": cols,
            "n_seeds": len(models), "n_layers": len(lin),
            "max_leverage": cfg["max_leverage"], "long_only": cfg["long_only"],
            "activation": "gelu", "clip": 5.0, "hold_hours": DS.HOLD_H,
            "anchor_hour_utc": DS.PROD_HOUR, "config": cfg}
    arrs["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--set", nargs="*", default=[], help="key=json overrides")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", action="store_true", help="also score TEST")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text()) if a.config else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    res = run(cfg, a.out, a.final)
    keys = ["cagr", "sharpe", "max_drawdown", "avg_abs_position", "turnover_per_day"]
    for split in ("train", "val", "test"):
        if split in res:
            r = res[split]
            print(split, {k: round(r[k], 4) for k in keys},
                  "| B&H sharpe", round(r["baselines"]["buy_and_hold"]["sharpe"], 3),
                  "| voltgt sharpe", round(r["baselines"]["noctua_vol_target_long"]["sharpe"], 3),
                  "| dSharpe vs B&H", round(r["vs_buy_and_hold"]["point"], 3), r["vs_buy_and_hold"]["ci95"])
    print("elapsed", res["elapsed_s"], "s")


if __name__ == "__main__":
    main()
