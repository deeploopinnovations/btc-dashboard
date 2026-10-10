import numpy as np, pandas as pd, json
exec(open('attacks.py').read().split('# ---------------- Attack 1')[0])
def bb(d):
    return block_boot(d)
res = []
def rep(label, s, m=test):
    ok = m & np.isfinite(s)
    qb = qlike(RV[ok], s[ok]); qn = qlike(RV[ok], noct[ok]); d = qb - qn
    ci, p = bb(d)
    return {'label': label, 'qlike': round(float(qb.mean()), 4), 'noct_gain_pct': round(float(100 * d.mean() / qb.mean()), 2),
            'ci': [round(ci[0], 4), round(ci[1], 4)], 'p_noct_le0': round(p, 4)}
specs = {'HAR5': HAR, 'HAR5+wf': HAR + ['wf'], 'HAR5+dow': HAR + [f'D{d}' for d in range(1, 7)], 'HAR5+wf+dow': CAL}
for win in PRE:
    for sn, cols in specs.items():
        b0 = fit_ols(cols, PRE[win]); b = fit_qlike(cols, PRE[win], b0)
        s = predict(cols, b)
        res.append(rep(f'{sn} QLIKE-direct fit {win}', s))
# NOCTUA + per-weekday / per-wf correction (scale fit on pre window only)
for win in PRE:
    m = PRE[win]
    adj_dow = np.ones(len(E))
    for d in range(7):
        mm = m & (E.dow.to_numpy() == d)
        c = float(np.sqrt(np.mean(RV[mm] ** 2 / noct[mm] ** 2)))
        adj_dow[E.dow.to_numpy() == d] = c
    res.append(rep(f'NOCTUA raw_med x per-weekday QLIKE scale fit {win}', noct * adj_dow))
    # per-wf-bin scale
    bins = pd.cut(E.wf.to_numpy(), [-0.01, 0.01, 0.5, 0.99, 1.01], labels=False)
    adj_wf = np.ones(len(E))
    for k in np.unique(bins):
        mm = m & (bins == k)
        c = float(np.sqrt(np.mean(RV[mm] ** 2 / noct[mm] ** 2)))
        adj_wf[bins == k] = c
    res.append(rep(f'NOCTUA raw_med x per-wf-bin QLIKE scale fit {win}', noct * adj_wf))
    # plain constant
    c = float(np.sqrt(np.mean(RV[m] ** 2 / noct[m] ** 2)))
    res.append(rep(f'NOCTUA raw_med x constant QLIKE scale fit {win} (c={c:.3f})', noct * c))
# recent-window variants for the original-style frozen HAR (no calendar) for reference
for r in res:
    print(json.dumps(r))
json.dump(res, open('ablate_out.json', 'w'), indent=1)
