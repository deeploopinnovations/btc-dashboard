import sys, json
sys.path.insert(0, "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4")
import numpy as np, pandas as pd
exec(open("attack12.py").read().split("specs = {")[0])   # reuse feature construction only

DOW = [f"dowlv24_{d}" for d in range(1,7)] + [f"dow{d}" for d in range(1,7)]
GROUPS = {
  "semi": ["semi_pos_1d","semi_neg_1d","semi_pos_5d","semi_neg_5d"],
  "jump": ["jump_1d","jump_5d"],
  "park": ["park_1d","park_5d"],
  "dow_x_lv24": [f"dowlv24_{d}" for d in range(1,7)],
  "dow": [f"dow{d}" for d in range(1,7)],
  "dvol": ["ldvol"],
}
base_cols = BASE

def inner_cv(cols, fs):
    """Expanding yearly folds 2020..2024H1, using only dt < 2024-06-30. Pooled QLIKE."""
    qs = []
    fs_ts = pd.Timestamp(fs, tz="UTC")
    folds = [("2020-01-01","2021-01-01"),("2021-01-01","2022-01-01"),("2022-01-01","2023-01-01"),("2023-01-01","2024-01-01"),("2024-01-01","2024-06-30")]
    allq = []
    for a,b in folds:
        a=pd.Timestamp(a,tz="UTC"); b=pd.Timestamp(b,tz="UTC")
        trm = ((E.dt>=fs_ts)&(E.dt<a)).to_numpy()
        tem = ((E.dt>=a)&(E.dt<b)).to_numpy()
        X = np.column_stack([np.ones(len(E))]+[E[c].to_numpy(float) for c in cols])
        ok = np.isfinite(X).all(1)&np.isfinite(E.y.to_numpy())
        trm = trm & ok; tem = tem & ok
        if trm.sum()<200 or tem.sum()==0: continue
        bb,*_ = np.linalg.lstsq(X[trm], E.y.to_numpy()[trm], rcond=None)
        s = np.exp(X@bb); k = np.sqrt(np.mean(E.RV.to_numpy()[trm]**2/s[trm]**2))
        allq.append(qlike(E.RV.to_numpy()[tem], (s*k)[tem]))
    return float(np.concatenate(allq).mean())

def select(fs):
    fs_ts = pd.Timestamp(fs, tz="UTC")
    avail = {g:c for g,c in GROUPS.items() if all(E.loc[E.dt>=fs_ts, x].notna().all() for x in c)}
    chosen = []; cur_cols = list(base_cols); cur = inner_cv(cur_cols, fs)
    hist=[("base", cur)]
    while True:
        best=None
        for g,c in avail.items():
            if g in chosen: continue
            if g=="dow" and "dow_x_lv24" in chosen: pass
            v = inner_cv(cur_cols+c, fs)
            if best is None or v<best[1]: best=(g,v,c)
        if best is None or best[1] > cur - 0.0005: break
        chosen.append(best[0]); cur_cols += best[2]; cur = best[1]; hist.append((best[0], cur))
    return chosen, cur_cols, hist, avail

out = {}
for fs in ["2018-01-01", "2021-01-01", "2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01"]:
    if fs == "2023-07-01":   # too few pre-fold rows for inner CV; reuse the 2023-01 selection (pre-2024-07 data only)
        chosen = ["dvol","dow_x_lv24","dow","park"]
        cols = BASE + sum([GROUPS[g] for g in chosen], [])
        hist = [("reused_from_2023-01", None)]; avail = {}
    else:
        chosen, cols, hist, avail = select(fs)
    qa, ba, ka, ntr, nte, te = fit_eval(E, cols, fs)
    qb, bb, kb, _, _, _ = fit_eval(E, cols + ["lnoct"], fs)
    d = qa-qb; m, ci, ple = block_boot_mean(d)
    # same-window reference: lv_har only (no selection)
    qh,_,_,_,_,_ = fit_eval(E, BASE, fs)
    print(f"fit {fs}: groups available={list(avail)} chosen={chosen} inner-CV path={[(g,(round(v,4) if v is not None else None)) for g,v in hist]}")
    print(f"   test QLIKE selected-base={qa.mean():.4f} (lv_har base {qh.mean():.4f}); +noct={qb.mean():.4f}; gain={m:+.4f} CI=[{ci[0]:+.4f},{ci[1]:+.4f}] p<=0={ple:.4f} coef_lnoct={bb[-1]:+.3f}")
    out[fs] = dict(chosen=chosen, cols=cols, inner_path=hist, test_qlike_base=float(qa.mean()), test_qlike_lv_har=float(qh.mean()),
                   test_qlike_aug=float(qb.mean()), gain=m, ci=ci, p_le0=ple, coef_lnoct=float(bb[-1]), ntrain=ntr)
json.dump(out, open("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A4/attack1b_out.json","w"), indent=2, default=float)
