"""Spot-check the hour_sweep replica against real forecast() at non-17:00 anchors."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/user/btc-dashboard"); sys.path.insert(0, str(ROOT / "model"))
from serve.runtime import load_model
from serve.predict import forecast
RUN = ROOT / "runs/noctua-disproof-2026-10-10"
hours = pd.read_parquet(RUN / "evals/hours_through_now.parquet").reset_index(drop=True)
ts = hours.hour_ts.to_numpy(np.int64)
model = load_model(ROOT / "model/serve/noctua_v2.npz")
D = pd.read_parquet(RUN / "audits/scratch_A2/hour_sweep_test.parquet")
out = []
for h in (3, 11, 22):
    g = D[D.hour == h]
    pick = g.iloc[np.linspace(0, len(g) - 1, 4).round().astype(int)]
    for _, r in pick.iterrows():
        raw = {}
        f = forecast(model, hours, anchor_ts=int(ts[int(r["row"])]), raw=raw, source="x")
        out.append(dict(hour=h, dt=str(r["dt"]), cal=raw["cal"]["factor"], rep=float(r["factor"]),
                        pub=f["sigma_window_pct"] / 100, rep_served=float(r["served"]),
                        med_pub=f["sigma_functional"]["sigma_med_pct"] / 100, rep_med=float(r["raw_med"]) * float(r["factor"])))
O = pd.DataFrame(out)
O["fac_err"] = O.cal - O.rep
O["served_rel_err"] = O.pub / O.rep_served - 1
print(O.to_string())
print("max |factor err|", O.fac_err.abs().max(), " max |served rel err|", O.served_rel_err.abs().max())
