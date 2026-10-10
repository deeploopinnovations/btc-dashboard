import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.history import check_continuity
h = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
gap = h.drop(index=9400).reset_index(drop=True)          # one missing hour, as a dropped bar hour would leave
print("continuity guard on gapped history:", check_continuity(gap))
ts = gap["hour_ts"].to_numpy(np.int64)
r = len(gap) - 1
rows = A._settled_anchors(None, r, 19, 60, 24)
off = (ts[r] - ts[rows]) // 3600 % 24
print("anchor hour", (ts[r] // 3600) % 24, "| rows whose hour-of-day differs:", int(np.sum(off != 0)), "of", len(rows))
print("example mismatch offsets (hours):", sorted(set(((ts[r]-ts[rows])//3600 - 24*np.arange(1,len(rows)+1)).tolist()))[:5])
