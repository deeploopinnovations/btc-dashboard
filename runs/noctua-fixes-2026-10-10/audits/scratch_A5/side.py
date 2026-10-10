"""Run served-path computations from one source tree (HEAD or working copy).
usage: python -I side.py <model_dir> <out.npz> [--no-adaptive]"""
import sys
tree, out = sys.argv[1], sys.argv[2]
noad = "--no-adaptive" in sys.argv
sys.path.insert(0, tree)
import numpy as np, pandas as pd
import serve.predict as P
from serve.runtime import load_model
from noctua.features import build_features
import noctua.features as NF
assert NF.__file__.startswith(tree), NF.__file__
hours = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
model = load_model()
if noad:
    P.volatility_correction = lambda *a, **k: {"applied": False, "factor": 1.0, "n_episodes": 0,
                                               "window_days": 60, "reason": "disabled"}
hour_ts = hours["hour_ts"].to_numpy(np.int64)
rng = np.random.default_rng(5)
rows = np.sort(rng.choice(np.arange(9000, len(hours) - 1), 40, replace=False))
rows = np.concatenate([rows, [len(hours) - 1]])
res = {}
for H in (19,):
    for r in rows:
        ts = hour_ts[r]
        raw = {}
        f = P.forecast(model, hours, H=H, anchor_ts=int(ts), source="audit", raw=raw)
        pred = raw["pred"]
        k = f"r{r}_H{H}"
        for name in ("qa", "sigma_med", "sigma_mean", "sigma_atoms", "q_r", "q_up", "q_dn"):
            res[f"{k}__{name}"] = np.asarray(pred[name])
        res[f"{k}__factor"] = np.array([raw["cal"]["factor"]])
        # features too (what the network consumes)
        dt = pd.to_datetime(ts, unit="s", utc=True)
        ep = pd.DataFrame({"anchor_ts": [ts], "H": [H], "row": [r], "dt": [dt],
                           "anchor_hour": [dt.hour], "dow": [dt.dayofweek]})
        X = build_features(hours, ep)
        res[f"{k}__feat_cal_legacy"] = X["cal_weekend_frac"].to_numpy()
        res[f"{k}__Xa"] = model.prepare(X, np.array([float(H)]))["Xa"]
np.savez(out, **res)
print("wrote", out, len(res), "arrays; model feat_cols has cal_weekend_frac:", "cal_weekend_frac" in model.feat_cols)
