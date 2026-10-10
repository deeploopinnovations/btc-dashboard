import sys, json
sys.path.insert(0, "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4")
import numpy as np, pandas as pd
from common import *
exec(open("attack12.py").read().split("specs = {")[0])
sel = json.load(open("attack1b_out.json"))
dd = E.dt.diff().dt.total_seconds().dropna()
print("daily-contiguous anchors:", bool((dd==86400).all()), "n", len(E))
rng = np.random.default_rng(7)
E["noise"] = rng.normal(size=len(E))
E["lnoct_lag7"] = E.lnoct.shift(7)
E["lnoct_lag1"] = E.lnoct.shift(1)
# placebo: shifted NOCTUA (causal, 7 days stale)
res = {}
def gain(cols, fs, extra):
    E2 = E
    qa, *_ = fit_eval(E2, cols, fs)
    qb, b, k, *_ = fit_eval(E2, cols + [extra], fs)
    d = qa - qb
    m, ci, p = block_boot_mean(d)
    return m, ci, float(b[-1])
for label, cols in [("claim_lvhar", BASE), ("selected", None)]:
    for fs in ["2018-01-01","2023-01-01"]:
        c = BASE if cols is not None else sel[fs]["cols"]
        for ex in ["lnoct","noise","lnoct_lag7"]:
            m, ci, coef = gain(c, fs, ex)
            print(f"{label:10s} fit {fs} add {ex:11s}: gain={m:+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] coef={coef:+.3f}")
            res[f"{label}|{fs}|{ex}"] = dict(gain=m, ci=ci, coef=coef)
# in-sample vs out-of-sample NOCTUA raw QLIKE (scale fitted on train)
tr = (E.dt < TRAIN_END).to_numpy(); te = (E.dt >= TEST_START).to_numpy()
k = np.sqrt(np.mean(E.RV.to_numpy()[tr]**2/E.noctua_raw_med.to_numpy()[tr]**2))
q_tr = qlike(E.RV.to_numpy()[tr], (E.noctua_raw_med.to_numpy()*k)[tr]).mean()
q_te = qlike(E.RV.to_numpy()[te], (E.noctua_raw_med.to_numpy()*k)[te]).mean()
print(f"NOCTUA raw QLIKE (scale fit pre-2024-07): in-sample 2018-2024H1 {q_tr:.4f}, out-of-sample 2024-07+ {q_te:.4f}")
res["noctua_insample_vs_oos"] = dict(train=q_tr, test=q_te, scale=k)
json.dump(res, open("placebo_out.json","w"), indent=2, default=float)
