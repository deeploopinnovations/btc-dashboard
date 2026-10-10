import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.runtime import load_model
model = load_model()
bundle = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
full = pd.read_parquet("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/evals/hours_through_now.parquet").sort_values("hour_ts", ignore_index=True)
anchor = int(pd.Timestamp("2026-10-05 17:00", tz="UTC").timestamp())
for name, h in (("bundle400d", bundle), ("full_history", full)):
    r = int(np.searchsorted(h["hour_ts"].to_numpy(np.int64), anchor))
    info = A.volatility_correction(model, h, r, 19)
    print(f"{name}: anchor row {r} factor={info['factor']:.4f} applied={info['applied']} n={info['n_episodes']} reason={info['reason'][:70]}")
# shipped HEAD-style selector on full history for same anchor (stride 6, pooled hours), for reference
