import numpy as np, pandas as pd, json
exec(open('attacks.py').read().split('# ---------------- Attack 1')[0])   # reuse data + helpers
variants = {}
def add(name, cols, win, method):
    m = PRE[win]
    b = fit_ols(cols, m)
    if method == 'qlike_direct':
        b = fit_qlike(cols, m, b); s = predict(cols, b)
    else:
        s = predict(cols, b) * qscale(predict(cols, b), m)
    variants[name] = s
    return s
for win in PRE:
    add(f'HAR5+cal QLIKE-direct [{win}]', CAL, win, 'qlike_direct')

S = {}
S['noct'] = noct
S['har5'] = E.har5_frozen.to_numpy()
cal_s = variants['HAR5+cal QLIKE-direct [2023H1-2024H1]']
S['har5cal_recal'] = cal_s
S['har5cal_full'] = variants['HAR5+cal QLIKE-direct [2018-2024H1]']
# NOTE: calendar weights for the recal model
b = fit_qlike(CAL, PRE['2023H1-2024H1'], fit_ols(CAL, PRE['2023H1-2024H1']))
names = ['const'] + CAL
print('recal (2023H1-2024H1) QLIKE-direct coefficients:'); print({n: round(float(c), 3) for n, c in zip(names, b)})
# log residual by anchor dow, pre vs test
def dowtab(mask, label):
    rows = []
    for d in range(7):
        m = mask & (E.dow.to_numpy() == d)
        rows.append({'dow': d, 'n': int(m.sum()),
                     'noct_med_logRV-logS': round(float(np.median(np.log(RV[m]) - np.log(noct[m]))), 3),
                     'har5cal_med': round(float(np.median(np.log(RV[m]) - np.log(cal_s[m]))), 3),
                     'har5_med': round(float(np.median(np.log(RV[m]) - np.log(S['har5'][m]))), 3)})
    print(label); print(pd.DataFrame(rows).to_string(index=False))
dowtab(test, 'TEST: anchor dow, median log(RV/sigma)')
dowtab(PRE['2023H1-2024H1'], 'PRE 2023H1-2024H1 (in-sample for recal)')
dowtab(pre_full, 'PRE full 2018-2024H1')
# weekend-share bins in test
E['wfb'] = pd.cut(E.wf, [-0.01, 0.01, 0.5, 0.99, 1.01], labels=['0', '(0,.5]', '(.5,1)', '1'])
t = E[test].copy(); t['e_noct'] = np.log(RV[test]) - np.log(noct[test]); t['e_cal'] = np.log(RV[test]) - np.log(cal_s[test])
print('TEST by forward weekend share bin'); print(t.groupby('wfb', observed=True)[['e_noct', 'e_cal']].agg(['median', 'count']).round(3).to_string())
p = E[PRE['2023H1-2024H1']].copy(); p['e_noct'] = np.log(RV[PRE['2023H1-2024H1']]) - np.log(noct[PRE['2023H1-2024H1']])
print('PRE 2023H1-2024H1 by wf bin, NOCTUA log residual'); print(p.groupby('wfb', observed=True)['e_noct'].agg(['median', 'count']).round(3).to_string())
# half-year stability of the gain vs NOCTUA for recal model and original frozen har5
per = {'2024H2': ('2024-07-01', '2025-01-01'), '2025H1': ('2025-01-01', '2025-07-01'), '2025H2': ('2025-07-01', '2026-01-01'), '2026H1': ('2026-01-01', '2026-07-01'), '2026Q3': ('2026-07-01', '2026-10-10')}
print('per-period QLIKE (noct, har5 frozen, har5+cal recal) and gain of recal vs noct %')
for k, (a, c) in per.items():
    m = (E.dt >= pd.Timestamp(a, tz='UTC')) & (E.dt < pd.Timestamp(c, tz='UTC'))
    m = m.to_numpy()
    qn = qlike(RV[m], noct[m]).mean(); qh = qlike(RV[m], S['har5'][m]).mean(); qc = qlike(RV[m], cal_s[m]).mean()
    print(k, int(m.sum()), round(qn, 4), round(qh, 4), round(qc, 4), 'recal gain vs noct %', round(100 * (qn - qc) / qn, 1))
# Placebo check: does calendar help a model with NO noctua knowledge, in the pre-2024 in-sample window only?
