"""A2 attack (b): does the served-vs-raw bias differ by UTC anchor hour? Batch replica of serve/adaptive
(verified against forecast() in verify_replica.py to ~1e-12 on the factor)."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/user/btc-dashboard")
sys.path.insert(0, str(ROOT / "model"))
from noctua.features import build_features
from serve.runtime import load_model
RUN = ROOT / "runs/noctua-disproof-2026-10-10"
H = 19
hours = pd.read_parquet(RUN / "evals/hours_through_now.parquet").reset_index(drop=True)
ts = hours.hour_ts.to_numpy(np.int64); n = len(ts)
dt = pd.to_datetime(ts, unit="s", utc=True)
rv5 = hours.rv5.to_numpy(np.float64)
c = np.concatenate([[0.0], np.cumsum(rv5)])
RV = np.full(n, np.nan); RV[: n - H + 1] = np.sqrt(c[H:] - c[: n - H + 1])
model = load_model(ROOT / "model/serve/noctua_v2.npz")

def batch(rws):
    om, oa = [], []
    for s in range(0, len(rws), 4096):
        r = rws[s:s + 4096]
        d = pd.to_datetime(ts[r], unit="s", utc=True)
        ep = pd.DataFrame({"anchor_ts": ts[r], "H": H, "row": r, "dt": d, "anchor_hour": d.hour, "dow": d.dayofweek})
        X = build_features(hours, ep)
        p = model.predict(model.prepare(X, np.full(len(r), float(H))))
        om.append(np.asarray(p["sigma_med"], float)); oa.append(np.asarray(p["sigma_mean"], float))
    return np.concatenate(om), np.concatenate(oa)

t0 = time.time()
test0 = pd.Timestamp("2024-07-01", tz="UTC")
first = int(np.searchsorted(ts, int((test0 - pd.Timedelta(days=63)).timestamp())))
rows_all = np.arange(first, n)
sm, sa = batch(rows_all)
sig_med = np.full(n, np.nan); sig_mean = np.full(n, np.nan)
sig_med[rows_all] = sm; sig_mean[rows_all] = sa
print("batch done", round(time.time() - t0), "s", len(rows_all), flush=True)

anch = np.flatnonzero((dt >= test0) & np.isfinite(RV) & (np.arange(n) > 24 * 400))
factor = np.ones(n)
for a in anch:
    last = a - H
    first_r = max(24 * 30, last - 60 * 24)
    rr = np.arange(first_r, last, 6)
    s, rvv = sig_med[rr], RV[rr]
    g = np.isfinite(s) & np.isfinite(rvv) & (s > 0) & (rvv > 0)
    if g.sum() >= 20:
        factor[a] = float(np.clip(np.median(rvv[g] / s[g]), 0.70, 1.40))
print("factors done", round(time.time() - t0), "s", flush=True)

D = pd.DataFrame({"row": anch, "hour": dt[anch].hour, "dt": dt[anch], "RV": RV[anch],
                  "raw_med": sig_med[anch], "raw_mean": sig_mean[anch], "factor": factor[anch]})
D["served"] = D.raw_mean * D.factor
D.to_parquet(RUN / "audits/scratch_A2/hour_sweep_test.parquet")

def ql(rv, s):
    r = np.maximum(rv, 1e-12) ** 2 / np.maximum(s, 1e-12) ** 2
    return r - np.log(r) - 1.0

rows_out = []
for h, g in D.groupby("hour"):
    rv = g.RV.to_numpy()
    rec = {"hour": int(h), "n": len(g), "factor_mean": g.factor.mean(), "factor_min": g.factor.min(), "factor_max": g.factor.max()}
    for k in ("raw_med", "raw_mean", "served"):
        rec[f"med_RV_over_{k}"] = float(np.median(rv / g[k].to_numpy()))
        rec[f"QLIKE_{k}"] = float(ql(rv, g[k].to_numpy()).mean())
    rows_out.append(rec)
S = pd.DataFrame(rows_out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S.round(4).to_string())
S.to_csv(RUN / "audits/scratch_A2/hour_sweep_summary.csv", index=False)
print("ALL-HOURS QLIKE raw_med %.4f raw_mean %.4f served %.4f" % tuple(ql(D.RV.values, D[k].values).mean() for k in ("raw_med", "raw_mean", "served")))
print("elapsed", round(time.time() - t0), "s")
