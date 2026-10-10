import sys, json
sys.path.insert(0, "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4")
import numpy as np, pandas as pd
from common import *

E = load_E(); h = load_hours()
R = E.row.to_numpy()
rv5 = h.rv5.to_numpy(float); rp = h.rv5_pos.to_numpy(float); rn = h.rv5_neg.to_numpy(float)
bpv = h.bpv5.to_numpy(float)
park = (np.log(h.high.to_numpy(float)) - np.log(h.low.to_numpy(float)))**2 / (4*np.log(2))
def tsum(x,k):   # exclusive of anchor bar: sum x[i-k:i]
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out = np.full(len(x), np.nan); out[k:] = c[k:len(x)] - c[:len(x)-k]; return out
def lg(s, k): return 0.5*np.log(np.maximum(s/k, 1e-14))

# richer causal baseline features at each anchor (all use bars <= a-1)
F = pd.DataFrame(index=E.index)
for nm, k in (("1d",24),("5d",120)):
    sp, sn = tsum(rp,k), tsum(rn,k); s = tsum(rv5,k); b = tsum(bpv,k); pk = tsum(park,k)
    F[f"semi_pos_{nm}"] = lg(sp,k)[R]
    F[f"semi_neg_{nm}"] = lg(sn,k)[R]
    F[f"jump_{nm}"] = (np.maximum(s-b,0)/(s+1e-14))[R]
    F[f"park_{nm}"] = lg(pk,k)[R]
for c in ["semi_pos_1d","semi_neg_1d","semi_pos_5d","semi_neg_5d","jump_1d","jump_5d","park_1d","park_5d"]:
    E[c] = F[c].to_numpy()
for d in range(1,7):
    E[f"dowlv24_{d}"] = E[f"dow{d}"]*E.lv24
RICH_NODVOL = ["semi_pos_1d","semi_neg_1d","semi_pos_5d","semi_neg_5d","jump_1d","jump_5d","park_1d","park_5d"] + [f"dowlv24_{d}" for d in range(1,7)]
RICH = RICH_NODVOL + ["ldvol"]
BASE = ["lv1","lv6","lv24","lv120","lv528"]
print("nan check rich:", {c: int(E[c].isna().sum()) for c in RICH_NODVOL})
ldv = E.ldvol.notna()
print("first finite ldvol date:", E.dt[ldv].min())

def gain_row(cols, fs, label, extra=None):
    qa, ba, ka, ntr, nte, te = fit_eval(E, cols, fs)
    qb, bb, kb, _, _, _ = fit_eval(E, cols + ["lnoct"], fs)
    d = qa - qb
    m, ci, ple = block_boot_mean(d)
    return dict(label=label, fit_from=fs, ntrain=ntr, ntest=nte, qlike_base=float(qa.mean()), qlike_aug=float(qb.mean()),
                gain=m, ci=ci, p_le0=ple, coef_lnoct=float(bb[-1]))

specs = {"lv_har": BASE, "rich": RICH_NODVOL, "rich_dvol": RICH}
windows = ["2018-01-01","2021-01-01","2022-01-01","2022-07-01","2023-01-01","2023-07-01"]
res = []
for sname, cols in specs.items():
    for fs in windows:
        if sname=="rich_dvol" and fs=="2018-01-01":
            pass  # rows with NaN ldvol drop out of the fit automatically (ok-mask)
        r = gain_row(cols, fs, sname)
        res.append(r)
        print(f"{sname:10s} fit_from {fs}: ntr={r['ntrain']:5d} base={r['qlike_base']:.4f} aug={r['qlike_aug']:.4f} gain={r['gain']:+.4f} CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p<=0={r['p_le0']:.4f} coef={r['coef_lnoct']:+.3f}")
json.dump(res, open("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4/attack12_out.json","w"), indent=2, default=float)
