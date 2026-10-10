import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.runtime import load_model
from noctua.features import build_features
model = load_model()
bundle = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
full = pd.read_parquet("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/evals/hours_through_now.parquet").sort_values("hour_ts", ignore_index=True)
for name, h in (("bundle400d", bundle), ("full_history", full)):
    r = len(h) - 1
    rows = A._settled_anchors(None, r, 19, 60, 24)
    ts = h["hour_ts"].to_numpy(np.int64)
    dt = pd.to_datetime(ts[rows], unit="s", utc=True)
    ep = pd.DataFrame({"anchor_ts": ts[rows], "H": 19, "row": rows, "dt": dt, "anchor_hour": dt.hour, "dow": dt.dayofweek})
    X = build_features(h, ep)
    ok = np.isfinite(X.to_numpy()).all(1)
    info = A.volatility_correction(model, h, r, 19)
    print(f"{name}: hours={len(h)} last={pd.to_datetime(ts[r],unit='s')} candidates={len(rows)} complete={int(ok.sum())} factor={info['factor']:.4f} applied={info['applied']} n={info['n_episodes']}")
