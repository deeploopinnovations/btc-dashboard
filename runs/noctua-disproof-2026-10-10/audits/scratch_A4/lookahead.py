import sys, json
ROOT = "/home/user/btc-dashboard"
sys.path.insert(0, f"{ROOT}/model")
import numpy as np, pandas as pd
from noctua.features import build_features
from serve.runtime import load_model
RUN = "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10"
h = pd.read_parquet(f"{RUN}/evals/hours_through_now.parquet").reset_index(drop=True)
ts = h.hour_ts.to_numpy(np.int64)
E = pd.read_parquet(f"{RUN}/evals/fresh_test_episodes.parquet")
model = load_model(f"{ROOT}/model/serve/noctua_v2.npz")
print("artifact meta keys:", list(model.meta.keys())[:20])
for k in ("train_end","trained_through","fit_end","fit_through","train_window","trained_on"):
    if k in model.meta: print("  meta", k, model.meta[k])
COLS = ["rv5","rv5_pos","rv5_neg","bpv5","rq5","open","high","low","close","volume"]
rng = np.random.default_rng(0)
def episode(row):
    dt = pd.to_datetime(ts[row], unit="s", utc=True)
    return pd.DataFrame({"anchor_ts":[ts[row]],"H":[19],"row":[row],"dt":[dt],"anchor_hour":[dt.hour],"dow":[dt.dayofweek]})
def pred(hh, row):
    X = build_features(hh, episode(row))
    p = model.predict(model.prepare(X, np.array([19.0])))
    return X, float(np.asarray(p["sigma_med"]).ravel()[0])
# anchors: a few train-era, test-era, and fresh-era 17:00 rows
rows = []
for dtp in ["2019-03-04 17:00","2022-06-01 17:00","2024-07-01 17:00","2025-05-13 17:00","2026-03-02 17:00","2026-09-30 17:00"]:
    t = pd.Timestamp(dtp, tz="UTC")
    rows.append(int(np.flatnonzero(ts == int(t.timestamp()))[0]))
out = []
for a in rows:
    X0, s0 = pred(h, a)
    for mode, start in (("strict_future(a+1..)", a+1), ("anchor_and_future(a..)", a)):
        hp = h.copy()
        sl = slice(start, None)
        for c in COLS:
            v = hp[c].to_numpy(float).copy()
            v[sl] = v[sl] * np.exp(rng.normal(0, 1.0, len(v[sl])))  # heavy random perturbation of all future bars
            hp[c] = v
        X1, s1 = pred(hp, a)
        d = np.nanmax(np.abs(X0.to_numpy(float) - X1.to_numpy(float)))
        bad = [c for c in X0.columns if not np.allclose(X0[c].to_numpy(float), X1[c].to_numpy(float), equal_nan=True, rtol=0, atol=0)]
        rec = dict(anchor=str(pd.to_datetime(ts[a], unit="s", utc=True)), mode=mode, sigma_orig=s0, sigma_pert=s1,
                   max_abs_feature_change=float(d), n_changed_features=len(bad), changed=bad[:8])
        out.append(rec); print(rec)
json.dump(out, open("lookahead_out.json","w"), indent=2, default=float)
