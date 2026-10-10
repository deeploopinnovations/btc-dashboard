"""A2 attack (F5): call the REAL forecast() at every 17:00 test-era anchor and record the published safe_levels."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/user/btc-dashboard")
sys.path.insert(0, str(ROOT / "model"))
from serve.runtime import load_model
from serve.predict import forecast
RUN = ROOT / "runs/noctua-disproof-2026-10-10"
hours = pd.read_parquet(RUN / "evals/hours_through_now.parquet").reset_index(drop=True)
ts = hours.hour_ts.to_numpy(np.int64)
dt = pd.to_datetime(ts, unit="s", utc=True)
model = load_model(ROOT / "model/serve/noctua_v2.npz")
H = 19
rows = np.flatnonzero((dt.hour == 17) & (dt >= pd.Timestamp("2024-07-01", tz="UTC")) & (np.arange(len(dt)) + H <= len(dt)))
rows = rows[rows > 24 * 400]
out = []
t0 = time.time()
for k, a in enumerate(rows):
    raw = {}
    f = forecast(model, hours, anchor_ts=int(ts[a]), raw=raw, source="tails")
    rec = dict(row=int(a), dt=str(dt[a]), spot=f["spot"], sigma=f["sigma_window_pct"] / 100,
               factor=raw["cal"]["factor"], p_up=f["p_up"])
    for s in f["safe_levels"]:
        al = s["alpha"]
        rec[f"call_pct_{al:.2f}"] = s["call_pct"]
        rec[f"put_pct_{al:.2f}"] = s["put_pct"]
    out.append(rec)
    if k % 50 == 0:
        print(k, len(rows), round(time.time() - t0), flush=True)
pd.DataFrame(out).to_csv(RUN / "audits/scratch_A2/tails_payload_17utc.csv", index=False)
print("done", len(out), round(time.time() - t0), flush=True)
