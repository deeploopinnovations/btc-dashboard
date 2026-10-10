import sys
sys.path.insert(0, "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4")
import numpy as np, pandas as pd
from common import *
E = load_E(); h = load_hours()
ts = h.hour_ts.to_numpy(np.int64)
print("hours range", pd.to_datetime(ts[0], unit="s", utc=True), pd.to_datetime(ts[-1], unit="s", utc=True), "n", len(h), "contiguous", bool((np.diff(ts)==3600).all()))
# alignment: row index -> hour_ts equals dt
hd = pd.to_datetime(ts[E.row.to_numpy()], unit="s", utc=True)
print("row->dt alignment ok:", bool((hd == E.dt).all()), " all 17:00:", bool((E.dt.dt.hour==17).all()))
# lag recomputation (exclusive trailing sums of rv5)
rv5 = h.rv5.to_numpy(float)
def tsum(x,k):
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan); out[k:] = c[k:len(x)] - c[:len(x)-k]; return out
mx = 0
for k in (1,6,24,120,528):
    lv = 0.5*np.log(np.maximum(tsum(rv5,k)*H/k,1e-14))
    diff = np.abs(lv[E.row.to_numpy()] - E[f"lv{k}"].to_numpy())
    mx = max(mx, np.nanmax(diff))
    print(f"lv{k} max abs diff vs recompute (exclusive of anchor bar): {np.nanmax(diff):.3e}")
# an inclusive version should NOT match (sanity that test has power)
lv1_incl = 0.5*np.log(np.maximum(rv5*H,1e-14))
print("sanity: inclusive lv1 (uses anchor bar) max diff:", np.nanmax(np.abs(lv1_incl[E.row.to_numpy()] - E.lv1.to_numpy())))
# forward RV check
c = np.concatenate([[0.0], np.cumsum(rv5)]); r = E.row.to_numpy()
fwd = c[r+H] - c[r]
print("RV recompute max abs diff:", np.nanmax(np.abs(np.sqrt(fwd)-E.RV.to_numpy())))
print("noctua_raw_med summary:", E.noctua_raw_med.describe().to_dict())
print("ldvol finite by year:", E.assign(f=np.isfinite(E.ldvol)).groupby(E.dt.dt.year).f.mean().round(2).to_dict())
print("test-period ldvol finite:", bool(np.isfinite(E.ldvol[E.dt>=TEST_START]).all()))
print("count test days", int((E.dt>=TEST_START).sum()), "train<2024-06-30", int((E.dt<TRAIN_END).sum()))
