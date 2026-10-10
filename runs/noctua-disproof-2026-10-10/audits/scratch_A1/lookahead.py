import sys, numpy as np, pandas as pd, json
from pathlib import Path
R = Path('/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10')
ROOT = Path('/home/user/btc-dashboard')
E = pd.read_parquet(R / 'evals/fresh_test_episodes.parquet')
hours = pd.read_parquet(R / 'evals/hours_through_now.parquet').reset_index(drop=True)
ts = hours.hour_ts.to_numpy(np.int64); rv5 = hours.rv5.to_numpy(float); n = len(hours)
H = 19
rows = E.row.to_numpy()
# 1) target: forward 19h realized vol from bars a..a+18
c = np.concatenate([[0.0], np.cumsum(rv5)])
RVc = np.sqrt(c[rows + H] - c[rows])
print('RV max abs diff vs stored:', np.max(np.abs(RVc - E.RV.to_numpy())))
# 2) trailing features exclusive of anchor
def lv(k):
    s = np.array([rv5[r - k:r].sum() for r in rows]); return 0.5 * np.log(s * H / k)
for k, col in [(1, 'lv1'), (6, 'lv6'), (24, 'lv24'), (120, 'lv120'), (528, 'lv528')]:
    print(col, 'max abs diff', float(np.max(np.abs(lv(k) - E[col].to_numpy()))))
# 3) DVOL: stamped a-1
d = pd.read_parquet(ROOT / 'data/newdata/dvol_btc.parquet')[['ts', 'volatility']].dropna().groupby('ts')['volatility'].last()
dv = pd.Series(np.nan, index=ts); com = dv.index.intersection(d.index); dv.loc[com] = d.loc[com].to_numpy()
dv = dv.to_numpy()
ld = np.log(dv[rows - 1] / 100.0 * np.sqrt(H / (24 * 365)))
m = np.isfinite(ld) & np.isfinite(E.ldvol.to_numpy())
print('ldvol max abs diff', float(np.max(np.abs(ld[m] - E.ldvol.to_numpy()[m]))), 'n', int(m.sum()), 'nan-pattern equal', bool((np.isfinite(ld) == np.isfinite(E.ldvol.to_numpy())).all()))
print('dvol last available ts', pd.to_datetime(d.index.max(), unit='s', utc=True), 'anchor check: ldvol uses value at a-1, ts a-1 stamp')
# 4) NOCTUA raw median replay with future corrupted
sys.path.insert(0, str(ROOT / 'model'))
from noctua.features import build_features
from serve.runtime import load_model
model = load_model(ROOT / 'model/serve/noctua_v2.npz')
def sig_med(h, a):
    ep = pd.DataFrame({'anchor_ts': [ts[a]], 'H': [H], 'row': [a],
                       'dt': [pd.to_datetime(ts[a], unit='s', utc=True)], 'anchor_hour': [17], 'dow': [pd.to_datetime(ts[a], unit='s', utc=True).dayofweek]})
    X = build_features(h, ep)
    p = model.predict(model.prepare(X, np.full(1, float(H))))
    return float(np.asarray(p['sigma_med'], float)[0])
rng = np.random.default_rng(0)
idx = rng.choice(np.flatnonzero(E.dt >= pd.Timestamp('2024-07-01', tz='UTC')), 6, replace=False)
res = []
for i in idx:
    a = int(rows[i])
    base = sig_med(hours, a)
    hc = hours.copy()
    cols = ['open', 'high', 'low', 'close', 'volume', 'rv5', 'rv5_pos', 'rv5_neg', 'bpv5', 'rq5']
    k = len(hc) - a
    for col in cols:
        hc.loc[a:, col] = hc.loc[a:, col].to_numpy() * rng.uniform(0.2, 5.0, k)
    corrupt = sig_med(hc, a)
    res.append({'dt': str(E.dt.iloc[i]), 'stored_raw_med': float(E.noctua_raw_med.iloc[i]), 'replay_clean': base, 'replay_future_corrupted': corrupt})
    print(res[-1])
print('max |replay_clean - stored|', max(abs(r['replay_clean'] - r['stored_raw_med']) for r in res))
print('max |corrupted - clean|', max(abs(r['replay_future_corrupted'] - r['replay_clean']) for r in res))
json.dump(res, open('lookahead_out.json', 'w'), indent=1)
