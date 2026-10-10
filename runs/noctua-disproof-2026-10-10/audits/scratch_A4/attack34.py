import sys, json
sys.path.insert(0, "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4")
import numpy as np, pandas as pd
exec(open("attack12.py").read().split("specs = {")[0])
sel = json.load(open("attack1b_out.json"))
CFG = {
 "claim_lvhar_fit2018": (BASE, "2018-01-01"),
 "claim_lvhar_fit2023": (BASE, "2023-01-01"),
 "sel_fit2018": (sel["2018-01-01"]["cols"], "2018-01-01"),
 "sel_fit2021": (sel["2021-01-01"]["cols"], "2021-01-01"),
 "sel_fit2023": (sel["2023-01-01"]["cols"], "2023-01-01"),
}
def dseries(cols, fs):
    qa, ba, ka, ntr, nte, te = fit_eval(E, cols, fs)
    qb, *_ = fit_eval(E, cols+["lnoct"], fs)
    return E.dt[te].reset_index(drop=True), qa-qb
HALVES = [("2024H2","2024-07-01","2025-01-01"),("2025H1","2025-01-01","2025-07-01"),("2025H2","2025-07-01","2026-01-01"),("2026H1","2026-01-01","2026-07-01"),("2026H2-to-Oct9","2026-07-01","2026-10-11")]
print("== Attack 3: half-year gains (QLIKE base - QLIKE with NOCTUA; >0 = NOCTUA helps) ==")
hy = {}
for name,(cols,fs) in CFG.items():
    dts, d = dseries(cols, fs)
    row = {}
    line=[]
    for lab,a,b in HALVES:
        m_ = ((dts>=pd.Timestamp(a,tz="UTC"))&(dts<pd.Timestamp(b,tz="UTC"))).to_numpy()
        x = np.asarray(d)[m_]
        mean, ci, ple = block_boot_mean(x, block=20, reps=2000)
        row[lab] = dict(n=int(m_.sum()), gain=mean, ci=ci, frac_pos=float((x>0).mean()))
        line.append(f"{lab}: n={m_.sum()} g={mean:+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] pos={(x>0).mean():.2f}")
    hy[name]=row
    print(name); [print("   ",l) for l in line]
    # daily gain distribution, to see whether the mean is driven by a few days
    q = np.quantile(np.asarray(d), [0.1,0.5,0.9])
    print(f"    daily gain p10/p50/p90 = {q[0]:+.4f}/{q[1]:+.4f}/{q[2]:+.4f}; mean trimmed(5%) = {np.mean(np.sort(np.asarray(d))[int(0.05*len(d)):int(0.95*len(d))]):+.4f}; frac days>0 = {(np.asarray(d)>0).mean():.3f}")
    hy[name]["daily"]=dict(p10=q[0],p50=q[1],p90=q[2])
print("\n== Attack 4: multiplicity ==")
# block-length sensitivity and 20k-rep CIs for claim spec
for name in ["claim_lvhar_fit2018","claim_lvhar_fit2023"]:
    cols,fs = CFG[name]
    dts,d = dseries(cols,fs)
    for blk in (10,20,40):
        m_, ci, ple = block_boot_mean(np.asarray(d), block=blk, reps=20000, seed=1)
        print(f"{name} block={blk}: gain={m_:+.4f} CI95=[{ci[0]:+.4f},{ci[1]:+.4f}] P(gain<=0)={ple:.4f}")
K = 24   # gain tests examined in this audit: 6 windows x 3 specs (18) + 6 selected-baseline windows (6)
mult = {}
for name in CFG:
    cols,fs = CFG[name]
    dts,d = dseries(cols,fs)
    alpha_adj = 0.05/K
    m_, ci95, _ = block_boot_mean(np.asarray(d), block=20, reps=20000, seed=2)
    rng = np.random.default_rng(3)
    x = np.asarray(d); n=len(x); nb=int(np.ceil(n/20)); reps=20000
    starts = rng.integers(0,n,(reps,nb)); idx=(starts[:,:,None]+np.arange(20)[None,None,:]).reshape(reps,-1)[:,:n]%n
    mm = x[idx].mean(1)
    lo, hi = np.quantile(mm,[alpha_adj/2,1-alpha_adj/2])
    p_one = float((mm<=0).mean())
    mult[name]=dict(gain=float(x.mean()), ci95=ci95, p_one_sided=p_one, bonf_K=K, bonferroni_p=min(1,2*p_one*K) if False else min(1,p_one*K),
                    ci_bonf=[float(lo),float(hi)], alpha_adj=alpha_adj)
    print(f"{name}: gain={x.mean():+.4f}; 95% CI=[{ci95[0]:+.4f},{ci95[1]:+.4f}] P(<=0)={p_one:.4f}; Bonferroni K={K}: p={min(1,p_one*K):.3f}; CI at 1-{alpha_adj:.4f}=[{lo:+.4f},{hi:+.4f}]")
json.dump(dict(halves=hy, multiplicity=mult), open("attack34_out.json","w"), indent=2, default=float)
