import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.runtime import load_model
model = load_model()
h = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
for r in (9239, 9263):
    info = A.volatility_correction(model, h, r, 19)
    print("row", r, "applied", info["applied"], "factor", round(info["factor"], 4), "reason:", info["reason"][:75])
# served sigma effect of the jump
