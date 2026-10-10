import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
import serve.adaptive as A
from noctua.features import build_features
from serve.runtime import load_model
hours = pd.read_parquet("/home/user/btc-dashboard/data/noctua_history.parquet").sort_values("hour_ts", ignore_index=True)
model = load_model()
ts = hours["hour_ts"].to_numpy(np.int64)
r = len(hours) - 1
rows = A._settled_anchors(None, r, 19, 60, 24)
dt = pd.to_datetime(ts[rows], unit="s", utc=True)
ep = pd.DataFrame({"anchor_ts": ts[rows], "H": 19, "row": rows, "dt": dt, "anchor_hour": dt.hour, "dow": dt.dayofweek})
X = build_features(hours, ep)
ok = np.isfinite(X.to_numpy()).all(1)
print("candidate rows", len(rows), "complete-feature rows", ok.sum())
print("first complete row offset (hours before anchor):", int(r - rows[ok].min()), " last:", int(r - rows[ok].max()))
Xo, ro = X[ok], rows[ok]
pred = model.predict(model.prepare(Xo, np.full(len(ro), 19.0)))
sig = np.asarray(pred["sigma_med"], float)
rv5 = hours["rv5"].to_numpy(float)
realized = np.array([np.sqrt(rv5[q:q+19].sum()) for q in ro])
ratio = realized / sig
print("ratio sorted:", np.round(np.sort(ratio), 3))
print("median ratio", round(float(np.median(ratio)), 4), "frac realized<sigma", round(float(np.mean(realized < sig)), 3))
rng = np.random.default_rng(0)
bs = [np.median(rng.choice(ratio, len(ratio))) for _ in range(4000)]
print("bootstrap 90% CI of median ratio:", np.round(np.percentile(bs, [5, 95]), 3))
# same-hour ratio over ALL 17:00 anchors of the full bundle, for context (in-sample for model, descriptive)
print("sigma_med sample:", np.round(sig[:5], 5), " realized sample:", np.round(realized[:5], 5))
