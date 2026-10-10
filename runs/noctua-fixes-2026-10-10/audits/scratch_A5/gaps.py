import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
from serve.history import load_bundle, check_continuity
h = load_bundle()
ts = h["hour_ts"].to_numpy(np.int64)
gaps = np.diff(ts) // 3600
print("bundle rows", len(h), "span days", (ts[-1]-ts[0])/86400)
print("gap count >1h:", int((gaps > 1).sum()), "largest:", int(gaps.max()))
print("check_continuity:", check_continuity(h))
idx = np.flatnonzero(gaps > 1)
for i in idx[:10]:
    print("  gap after row", i, "from", pd.to_datetime(ts[i], unit="s"), "to", pd.to_datetime(ts[i+1], unit="s"))
# hour-of-day mismatch of row-24 relative to anchor
hod = (ts // 3600) % 24
bad = (hod[np.arange(24, len(ts))] != hod[24:]).sum()
print("rows where row-24 is not same hour:", int(bad))
