import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.runtime import load_model
hours = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
model = load_model()
ts = hours["hour_ts"].to_numpy(np.int64)
for r in (9000, 9300, len(hours)-1):
    rows = A._settled_anchors(None, r, 19, 60, A.STRIDE_HOURS)
    info = A.volatility_correction(model, hours, r, 19)
    print(r, "n_rows", len(rows), "hod-match", bool(np.all((ts[rows] // 3600) % 24 == (ts[r] // 3600) % 24)) if len(rows) else None,
          "factor", round(info["factor"], 4), "applied", info["applied"], "reason:", info["reason"][:160])
