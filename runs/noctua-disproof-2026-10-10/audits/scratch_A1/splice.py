import numpy as np, pandas as pd
H = pd.read_parquet('/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/evals/hours_through_now.parquet')
ts = pd.to_datetime(H.hour_ts.astype('int64'), unit='s', utc=True)
H['dt'] = ts
gap = np.diff(H.hour_ts.values)
print('rows', len(H), 'start', ts.iloc[0], 'end', ts.iloc[-1], 'max gap s', gap.max(), 'dups', H.hour_ts.duplicated().sum())
m = H.set_index('dt')
w = m.resample('MS').agg({'rv5':'mean','volume':'median','close':'last'})
print(w.tail(10).to_string())
sp = m.loc['2026-08-10':'2026-08-20', ['rv5','volume','close']]
print(sp.resample('D').agg({'rv5':'mean','volume':'median','close':'last'}).to_string())
# ratio of rv5 level before/after splice
pre = m.loc['2026-06-01':'2026-08-14','rv5']; post = m.loc['2026-08-15':,'rv5']
print('mean rv5 pre', pre.mean(), 'post', post.mean(), 'median vol pre', m.loc['2026-06-01':'2026-08-14','volume'].median(), 'post', m.loc['2026-08-15':,'volume'].median())
