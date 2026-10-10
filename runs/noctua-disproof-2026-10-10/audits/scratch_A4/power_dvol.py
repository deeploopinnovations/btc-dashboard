import sys
ROOT = "/home/user/btc-dashboard"
sys.path.insert(0, f"{ROOT}/model")
import numpy as np, pandas as pd
from noctua.features import build_features
from serve.runtime import load_model
RUN = "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10"
h = pd.read_parquet(f"{RUN}/evals/hours_through_now.parquet").reset_index(drop=True)
ts = h.hour_ts.to_numpy(np.int64)
model = load_model(f"{ROOT}/model/serve/noctua_v2.npz")
a = int(np.flatnonzero(ts == int(pd.Timestamp("2025-05-13 17:00", tz="UTC").timestamp()))[0])
def pred(hh):
    dt = pd.to_datetime(ts[a], unit="s", utc=True)
    ep = pd.DataFrame({"anchor_ts":[ts[a]],"H":[19],"row":[a],"dt":[dt],"anchor_hour":[dt.hour],"dow":[dt.dayofweek]})
    X = build_features(hh, ep); p = model.predict(model.prepare(X, np.array([19.0])))
    return float(np.asarray(p["sigma_med"]).ravel()[0])
s0 = pred(h)
hp = h.copy(); hp.loc[a-1, "rv5"] *= 50.0   # POWER CHECK: change a PAST bar, should move the forecast
print("power check: past-bar perturbation changes sigma_med:", s0, "->", pred(hp))
# DVOL alignment: parquet ldvol at anchor a should equal log(DVOL stamped a-1 /100 * sqrt(19/8760))
d = pd.read_parquet(f"{ROOT}/data/newdata/dvol_btc.parquet")[["ts","volatility"]].dropna().groupby("ts").volatility.last()
E = pd.read_parquet(f"{RUN}/evals/fresh_test_episodes.parquet")
ok = E[E.ldvol.notna()].head(400)
rec = []
for _, r in ok.iterrows():
    row = int(np.flatnonzero(ts == int(pd.Timestamp(r["dt"]).timestamp()))[0])
    v = d.get(int(ts[row-1]), np.nan)
    rec.append(np.log(v/100*np.sqrt(19/8760)) - r.ldvol if np.isfinite(v) else np.nan)
rec = np.array(rec, float)
print("ldvol alignment: max |parquet - recomputed from stamp a-1| =", np.nanmax(np.abs(rec)), " n=", np.isfinite(rec).sum())
# is DVOL stamp a (same hour as anchor) available? compare correlation of stamp a-1 vs a for sanity, not used
