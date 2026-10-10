import json, numpy as np, pandas as pd
from scipy.optimize import minimize
from pathlib import Path
R = Path('/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10')
E = pd.read_parquet(R / 'evals/fresh_test_episodes.parquet')
E['dt'] = pd.to_datetime(E.dt, utc=True)
H = 19
TEST = pd.Timestamp('2024-07-01', tz='UTC')
PRE_END = TEST - pd.Timedelta(hours=H)           # embargoed: window must end before test
E['dow'] = E.dt.dt.dayofweek
E['wf'] = [np.mean(pd.date_range(t, periods=H, freq='h').dayofweek >= 5) for t in E.dt]  # correct forward weekend share
for d in range(1, 7):
    E[f'D{d}'] = (E.dow == d).astype(float)
RV = E.RV.to_numpy(); y = np.log(RV)
test = (E.dt >= TEST).to_numpy()
pre_full = (E.dt < PRE_END).to_numpy()
PRE = {'2018-2024H1': pre_full,
       '2022H2-2024H1': pre_full & (E.dt >= pd.Timestamp('2022-07-01', tz='UTC')).to_numpy(),
       '2023H1-2024H1': pre_full & (E.dt >= pd.Timestamp('2023-01-01', tz='UTC')).to_numpy()}
HAR = ['lv1', 'lv6', 'lv24', 'lv120', 'lv528']
CAL = HAR + ['wf'] + [f'D{d}' for d in range(1, 7)]
LOG3 = ['lv24', 'lv120', 'lv528']

def qlike(rv, s):
    r = np.maximum(rv, 1e-12) ** 2 / np.maximum(s, 1e-12) ** 2
    return r - np.log(r) - 1.0

def design(cols, m):
    return np.column_stack([np.ones(m.sum())] + [E.loc[m, c].to_numpy() for c in cols])

def fit_ols(cols, m):
    A = design(cols, m); b, *_ = np.linalg.lstsq(A, y[m], rcond=None); return b

def fit_qlike(cols, m, b0):
    A = design(cols, m); u0 = y[m]
    def f(b):
        u = u0 - A @ b
        e = np.exp(2 * u)
        return np.mean(e - 2 * u - 1), -A.T @ (2 * e - 2) / len(u)
    r = minimize(f, b0, jac=True, method='BFGS', options={'gtol': 1e-9, 'maxiter': 5000})
    return r.x

def predict(cols, b):
    A = np.column_stack([np.ones(len(E))] + [E[c].to_numpy() for c in cols])
    return np.exp(A @ b)

def qscale(s, m):
    return float(np.sqrt(np.mean(RV[m] ** 2 / s[m] ** 2)))

def block_boot(x, block=20, reps=4000, seed=0):
    n = len(x); rng = np.random.default_rng(seed); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, (reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n] % n
    m = x[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))], float((m <= 0).mean())

noct = E.noctua_raw_med.to_numpy()
out = {}

# ---------------- Attack 1: recalibrated baselines ----------------
variants = {}
def add(name, cols, win, method):
    m = PRE[win]
    b = fit_ols(cols, m)
    if method == 'qlike_direct':
        b = fit_qlike(cols, m, b)
        s = predict(cols, b)            # QLIKE fit already sets level (no extra scale)
    else:
        s = predict(cols, b) * qscale(predict(cols, b), m)   # the original frozen convention
    variants[name] = s
    return s

base_frozen = {'log_har3_frozen': E.log_har3_frozen.to_numpy(), 'har5_frozen': E.har5_frozen.to_numpy(),
               'har5_refit': E.har5_refit.to_numpy()}
for win in PRE:
    add(f'HAR5 OLS+Qscale [{win}]', HAR, win, 'ols')
    add(f'HAR5 +cal OLS+Qscale [{win}]', CAL, win, 'ols')
    add(f'HAR5 QLIKE-direct [{win}]', HAR, win, 'qlike_direct')
    add(f'HAR5+cal QLIKE-direct [{win}]', CAL, win, 'qlike_direct')

def summarize(name, s, m=test):
    ok = m & np.isfinite(s)
    q_b = qlike(RV[ok], s[ok]); q_n = qlike(RV[ok], noct[ok])
    d = q_b - q_n
    ci, p = block_boot(d)
    return {'name': name, 'n': int(ok.sum()), 'qlike_base': float(q_b.mean()), 'qlike_noctua': float(q_n.mean()),
            'gain_pct': float(100 * d.mean() / q_b.mean()), 'ci95_abs': ci, 'p_gain_le0': p,
            'median_RV_over_sigma_test': float(np.median(RV[ok] / s[ok]))}

rows = [summarize('frozen log_har3 (original)', base_frozen['log_har3_frozen']),
        summarize('frozen har5 (original)', base_frozen['har5_frozen']),
        summarize('expanding refit har5 (original)', base_frozen['har5_refit'])]
for k, s in variants.items():
    rows.append(summarize(k, s))
out['attack1'] = rows
print('ATTACK 1')
for r in rows:
    print(f"{r['name']:45s} n={r['n']} base={r['qlike_base']:.4f} gain={r['gain_pct']:+.2f}% CI=[{r['ci95_abs'][0]:.4f},{r['ci95_abs'][1]:.4f}] p={r['p_gain_le0']:.4f} medRV/s={r['median_RV_over_sigma_test']:.3f}")
print('noctua raw_med test qlike', float(qlike(RV[test], noct[test]).mean()))

# pre-window level diagnostics
print('\nLEVEL DIAGNOSTICS (median RV/sigma)')
for nm, s in [('noctua raw_med', noct), ('har5 frozen', base_frozen['har5_frozen'])]:
    print(nm, 'pre-2024H1', round(float(np.median(RV[pre_full] / s[pre_full])), 3),
          'test', round(float(np.median(RV[test] / s[test])), 3))
out['level'] = {}

# ---------------- Attack 2: concentration ----------------
print('\nATTACK 2 concentration (test, per-day QLIKE diff; >0 = NOCTUA better)')
conc = {}
bases = {'log_har3_frozen': base_frozen['log_har3_frozen'], 'har5_frozen': base_frozen['har5_frozen'],
         'har5_refit': base_frozen['har5_refit'],
         'best_recal HAR5+cal QLIKE-direct [2023H1-2024H1]': variants['HAR5+cal QLIKE-direct [2023H1-2024H1]'],
         'best_recal HAR5+cal QLIKE-direct [2022H2-2024H1]': variants['HAR5+cal QLIKE-direct [2022H2-2024H1]']}
for nm, s in bases.items():
    ok = test & np.isfinite(s)
    qb = qlike(RV[ok], s[ok]); qn = qlike(RV[ok], noct[ok]); d = qb - qn
    dates = E.dt[ok].to_numpy()
    order = np.argsort(-np.abs(d))
    n = len(d)
    res = {'mean_d': float(d.mean()), 'rel_pct': float(100 * d.mean() / qb.mean()), 'median_d': float(np.median(d)),
           'frac_days_noctua_better': float((d > 0).mean())}
    for frac in (0.01, 0.05):
        k = int(round(frac * n))
        keep = np.ones(n, bool); keep[order[:k]] = False
        res[f'drop_top{int(frac*100)}pct_absdiff_rel_pct'] = float(100 * d[keep].mean() / qb[keep].mean())
        res[f'drop_top{int(frac*100)}pct_absdiff_mean_d'] = float(d[keep].mean())
    # drop the largest NOCTUA-favouring days only (the gain side)
    pos = np.argsort(-d)
    for frac in (0.01, 0.05):
        k = int(round(frac * n)); keep = np.ones(n, bool); keep[pos[:k]] = False
        res[f'drop_top{int(frac*100)}pct_favouring_rel_pct'] = float(100 * d[keep].mean() / qb[keep].mean())
    # sign-flip / trimmed: remove top-5% NOCTUA-favour AND bottom-5% (pure concentration)
    res['top10_days'] = [(str(pd.Timestamp(dates[i]).date()), round(float(d[i]), 3)) for i in order[:10]]
    conc[nm] = res
    print(nm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'top10_days'}))
out['attack2'] = conc

# ---------------- Attack 3: other losses + MZ + constant scale ----------------
print('\nATTACK 3 alternative losses on test')
def losses(s, m=test):
    rv = RV[m]; sg = s[m]
    return {'QLIKE': float(qlike(rv, sg).mean()),
            'MSE_var(RV2-sig2)^2': float(np.mean((rv ** 2 - sg ** 2) ** 2)),
            'MSE_vol(RV-sig)^2': float(np.mean((rv - sg) ** 2)),
            'MAE_logvol': float(np.mean(np.abs(np.log(rv) - np.log(sg))))}
cands = {'NOCTUA raw_med': noct, 'NOCTUA raw_mean': E.noctua_raw_mean.to_numpy(),
         'har5_frozen': base_frozen['har5_frozen'], 'log_har3_frozen': base_frozen['log_har3_frozen'],
         'har5_refit': base_frozen['har5_refit'],
         'HAR5+cal QLIKE-direct [2023H1-2024H1]': variants['HAR5+cal QLIKE-direct [2023H1-2024H1]']}
L = {}
for nm, s in cands.items():
    ok = test & np.isfinite(s)
    L[nm] = losses(s, ok)
    print(nm, {k: round(v, 5) for k, v in L[nm].items()})
out['attack3_losses'] = L

def mz(s, m=test):
    ok = m & np.isfinite(s)
    x = s[ok] ** 2; yy = RV[ok] ** 2
    A = np.column_stack([np.ones(ok.sum()), x]); b, *_ = np.linalg.lstsq(A, yy, rcond=None)
    # bootstrap (block 20) of (a, b)
    n = len(yy); rng = np.random.default_rng(1); nb = int(np.ceil(n / 20)); bs = []
    for _ in range(2000):
        st = rng.integers(0, n, nb); idx = (st[:, None] + np.arange(20)[None, :]).ravel()[:n] % n
        bb, *_ = np.linalg.lstsq(A[idx], yy[idx], rcond=None); bs.append(bb)
    bs = np.array(bs)
    # log-vol MZ
    A2 = np.column_stack([np.ones(ok.sum()), np.log(s[ok])]); b2, *_ = np.linalg.lstsq(A2, np.log(RV[ok]), rcond=None)
    return {'MZ_var_a': float(b[0]), 'MZ_var_b': float(b[1]), 'MZ_var_a_ci': np.quantile(bs[:, 0], [.025, .975]).tolist(),
            'MZ_var_b_ci': np.quantile(bs[:, 1], [.025, .975]).tolist(),
            'MZ_var_R2': float(1 - np.sum((yy - A @ b) ** 2) / np.sum((yy - yy.mean()) ** 2)),
            'logvol_slope': float(b2[1]), 'logvol_intercept': float(b2[0])}
MZ = {nm: mz(s) for nm, s in cands.items()}
for nm, v in MZ.items():
    print('MZ', nm, {k: (np.round(v_, 3).tolist() if isinstance(v_, list) else round(v_, 4)) for k, v_ in v.items()})
out['attack3_mz'] = MZ

# constant multipliers fitted on PRE window only, applied to test
print('\nConstant multipliers fitted on pre-2024-07 only (applied to baseline, test QLIKE)')
const = {}
for nm in ['har5_frozen', 'log_har3_frozen', 'har5_refit']:
    s = cands[nm]; m = pre_full
    c_q = float(np.sqrt(np.mean(RV[m] ** 2 / s[m] ** 2)))             # QLIKE-optimal
    c_v = float(np.sum(RV[m] ** 2 * s[m] ** 2) / np.sum(s[m] ** 4))  # MSE-var-optimal
    c_l = float(np.exp(np.median(np.log(RV[m]) - np.log(s[m]))))      # MAE-log-optimal
    res = {}
    for lab, c in [('qlike_c', c_q), ('mse_var_c', c_v), ('mae_log_c', c_l)]:
        res[lab] = {'c': round(c, 4), **{k: round(v, 5) for k, v in losses(s * c, test).items()}}
    const[nm] = res
    print(nm, json.dumps(res))
# NOCTUA's own pre-window multiplier fitted on pre (in-sample for NOCTUA; diagnostic only)
m = pre_full; s = noct
print('NOCTUA raw_med pre-window QLIKE-opt c (in-sample diagnostic):', round(float(np.sqrt(np.mean(RV[m]**2/s[m]**2))),4))
out['attack3_constant'] = const

# ---------------- Attack 4 helpers: dump for lookahead script ----------------
json.dump(out, open(R / 'audits/scratch_A1/attacks_out.json', 'w'), indent=1, default=float)
print('\nwrote attacks_out.json')
