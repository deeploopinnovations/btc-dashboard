"""Factor at every 17:00 UTC anchor over the last 120 days, for one tree. usage: <tree> <out.npz>"""
import sys
tree, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import numpy as np, pandas as pd
import serve.adaptive as A
from serve.runtime import load_model
hours = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
model = load_model()
ts = hours["hour_ts"].to_numpy(np.int64)
hod = (ts // 3600) % 24
cand = np.flatnonzero(hod == 17)
cand = cand[cand >= len(hours) - 24 * 120]
f, n, app = [], [], []
for r in cand:
    info = A.volatility_correction(model, hours, int(r), 19)
    f.append(info["factor"]); n.append(info["n_episodes"]); app.append(info["applied"])
np.savez(out, factor=np.array(f), n=np.array(n), applied=np.array(app), rows=cand)
print("tree", tree, "anchors", len(cand), "applied", int(sum(app)))
