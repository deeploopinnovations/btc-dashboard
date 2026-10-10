import sys, json
sys.path.insert(0, "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4")
import numpy as np
from common import *

E = load_E()
base = ["lv1", "lv6", "lv24", "lv120", "lv528"]
out = {}
print("rows", len(E), "dt range", E.dt.min(), E.dt.max())
print("nan counts", E[["RV", "noctua_raw_med", "lv1", "lv6", "lv24", "lv120", "lv528"]].isna().sum().to_dict())
for fs in ["2018-01-01", "2023-01-01"]:
    qa, ba, ka, ntr, nte, te = fit_eval(E, base, fs)
    qb, bb, kb, _, _, _ = fit_eval(E, base + ["lnoct"], fs)
    d = qa - qb  # >0 means NOCTUA helps
    mean, ci, ple = block_boot_mean(d)
    print(f"fit from {fs}: ntrain={ntr} ntest={nte} QLIKE base={qa.mean():.4f} +noct={qb.mean():.4f} gain={mean:.4f} CI={ci} p<=0={ple:.4f}")
    print("   coef lnoct:", bb[-7], " scale base/aug:", ka, kb)
    out[fs] = dict(ntrain=ntr, ntest=nte, qlike_base=qa.mean(), qlike_aug=qb.mean(), gain=mean, ci=ci, p_le0=ple, coef_lnoct=float(bb[-7]), scale_base=float(ka), scale_aug=float(kb))
json.dump(out, open("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4/replicate_out.json", "w"), indent=2, default=float)
