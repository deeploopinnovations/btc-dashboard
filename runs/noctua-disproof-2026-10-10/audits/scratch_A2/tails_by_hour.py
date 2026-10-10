"""A2 attack (F5, hour dimension): served-level breach rates by UTC anchor hour, 30 test-era anchors per hour,
batch replica of serve/adaptive (factor from hour_sweep_test.parquet, verified vs forecast() in verify_replica.py)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/user/btc-dashboard"); sys.path.insert(0, str(ROOT / "model"))
from noctua.features import build_features
from serve.runtime import load_model
from serve.adaptive import apply_correction
RUN = ROOT / "runs/noctua-disproof-2026-10-10"
hours = pd.read_parquet(RUN / "evals/hours_through_now.parquet").reset_index(drop=True)
ts = hours.hour_ts.to_numpy(np.int64); dt = pd.to_datetime(ts, unit="s", utc=True)
hi, lo, cl = hours.high.values, hours.low.values, hours.close.values
H = 19
model = load_model(ROOT / "model/serve/noctua_v2.npz")
D = pd.read_parquet(RUN / "audits/scratch_A2/hour_sweep_test.parquet")
rng = np.random.default_rng(1)
recs = []
for h, g in D.groupby("hour"):
    g = g[g.row + H <= len(ts)]
    pick = g.iloc[np.linspace(0, len(g) - 1, 30).round().astype(int)]
    rows = pick.row.values.astype(int); fac = pick.factor.values
    d = pd.to_datetime(ts[rows], unit="s", utc=True)
    ep = pd.DataFrame({"anchor_ts": ts[rows], "H": H, "row": rows, "dt": d, "anchor_hour": d.hour, "dow": d.dayofweek})
    X = build_features(hours, ep)
    p = model.predict(model.prepare(X, np.full(len(rows), float(H))))
    m = len(rows)
    for j in range(m):
        pj = {k: (v[j:j + 1] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == m else v)
              for k, v in p.items() if not k.startswith("_pooled_")}
        pj = apply_correction(pj, float(fac[j]))
        a = rows[j]; spot = cl[a - 1]
        mu = max(0.0, np.log(hi[a:a + H].max() / spot)); md = -min(0.0, np.log(lo[a:a + H].min() / spot))
        r = dict(hour=h, dt=str(d[j]), mup=mu, mdn=md)
        for al in (0.05, 0.10):
            r[f"lu_{al}"] = float(model.safe_level(pj, al, True)[0])
            r[f"ld_{al}"] = float(model.safe_level(pj, al, False)[0])
        recs.append(r)
R = pd.DataFrame(recs)
S = []
for h, g in R.groupby("hour"):
    S.append(dict(hour=h, n=len(g),
                  up05=round(100 * (g.mup >= g["lu_0.05"]).mean(), 1), dn05=round(100 * (g.mdn >= g["ld_0.05"]).mean(), 1),
                  up10=round(100 * (g.mup >= g["lu_0.1"]).mean(), 1), dn10=round(100 * (g.mdn >= g["ld_0.1"]).mean(), 1)))
S = pd.DataFrame(S)
pd.set_option("display.width", 200)
print(S.to_string())
print("pooled (30/hour):", {c: round(float(S[c].mean()), 2) for c in ["up05", "dn05", "up10", "dn10"]})
R.to_parquet(RUN / "audits/scratch_A2/tails_by_hour_records.parquet")
