import numpy as np, pandas as pd, json
exec(open('attacks.py').read().split('# ---------------- Attack 1')[0])
b0 = fit_ols(HAR + [f'D{d}' for d in range(1, 7)], PRE['2023H1-2024H1'])
b = fit_qlike(HAR + [f'D{d}' for d in range(1, 7)], PRE['2023H1-2024H1'], b0)
s_dow = predict(HAR + [f'D{d}' for d in range(1, 7)], b)
b1 = fit_qlike(HAR, PRE['2023H1-2024H1'], fit_ols(HAR, PRE['2023H1-2024H1']))
s_har = predict(HAR, b1)
print('coef HAR5+dow', np.round(b, 3))
per = {'2024H2': ('2024-07-01', '2025-01-01'), '2025H1': ('2025-01-01', '2025-07-01'), '2025H2': ('2025-07-01', '2026-01-01'), '2026H1': ('2026-01-01', '2026-07-01'), '2026Q3': ('2026-07-01', '2026-10-10')}
out = []
for k, (a, c) in per.items():
    m = ((E.dt >= pd.Timestamp(a, tz='UTC')) & (E.dt < pd.Timestamp(c, tz='UTC'))).to_numpy()
    qn = qlike(RV[m], noct[m]); qd = qlike(RV[m], s_dow[m]); qh = qlike(RV[m], s_har[m])
    d = qn - qd; ci, p = block_boot(d)
    out.append((k, int(m.sum()), round(float(qn.mean()), 4), round(float(qd.mean()), 4), round(float(100*d.mean()/qn.mean()), 1), [round(x,4) for x in ci], round(float(qh.mean()),4)))
for o in out: print('per-period', o)
ok = test
qn = qlike(RV[ok], noct[ok]); qd = qlike(RV[ok], s_dow[ok]); d = qn - qd
print('TEST whole: NOCTUA raw', round(qn.mean(),4), 'HAR5+dow(2023H1-24H1)', round(qd.mean(),4), 'NOCTUA-minus-base (pct of NOCTUA)', round(100*d.mean()/qn.mean(),2))
ci, p = block_boot(d); print('CI of (noct - base) per day', ci, 'p(noct<=base)', p)
print('share of test days where HAR5+dow better', round(float((d > 0).mean()), 3))
dd = np.sort(d)[::-1]; n = len(d)
for f in (0.01, 0.05):
    k = int(round(f*n)); keep = np.ones(n, bool); keep[np.argsort(-np.abs(d))[:k]] = False
    print(f'drop top {f:.0%} |diff| days -> pct', round(100*d[keep].mean()/qn.to_numpy()[keep].mean(),2) if False else round(100*d[keep].mean()/qn[keep].mean(),2))
print('median per-day diff (noct - base)', round(float(np.median(d)),4))
# MZ / losses for this baseline
print('MSE var  noct', np.mean((RV[ok]**2-noct[ok]**2)**2), 'base', np.mean((RV[ok]**2-s_dow[ok]**2)**2))
print('MAE logvol noct', np.mean(np.abs(np.log(RV[ok])-np.log(noct[ok]))), 'base', np.mean(np.abs(np.log(RV[ok])-np.log(s_dow[ok]))))
json.dump({'per': out}, open('stab_out.json','w'))
