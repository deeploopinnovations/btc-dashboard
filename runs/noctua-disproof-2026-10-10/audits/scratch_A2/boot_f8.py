import numpy as np, pandas as pd
RUN='/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10'
def ql(rv,s):
    r=np.maximum(rv,1e-12)**2/np.maximum(s,1e-12)**2; return r-np.log(r)-1
def bb(x,block=20,reps=4000,seed=0):
    n=len(x); rng=np.random.default_rng(seed); nb=int(np.ceil(n/block))
    st=rng.integers(0,n,(reps,nb)); idx=(st[:,:,None]+np.arange(block)[None,None,:]).reshape(reps,-1)[:,:n]%n
    m=x[idx].mean(1); return np.quantile(m,[.025,.975]).round(5), round(float((m<=0).mean()),4)
E=pd.read_parquet(RUN+'/evals/fresh_test_episodes.parquet')
T=E[E.dt>=pd.Timestamp('2024-07-01',tz='UTC')]
rv=T.RV.values
for base in ['noctua_raw_med','noctua_raw_mean','har5_frozen','har5_refit']:
    d=ql(rv,T[base].values)-ql(rv,T.noctua_served_mean.values)   # >0 served better
    print('17utc served vs',base,'mean gain',round(d.mean(),5),'rel% ',round(100*d.mean()/ql(rv,T[base].values).mean(),2),'CI',bb(d)[0],'p(gain<=0)',bb(d)[1])
D=pd.read_parquet(RUN+'/audits/scratch_A2/hour_sweep_test.parquet')
rv=D.RV.values
for base in ['raw_med','raw_mean']:
    d=ql(rv,D[base].values)-ql(rv,D.served.values)
    print('ALL-hours served vs',base,'mean gain',round(d.mean(),5),'rel%',round(100*d.mean()/ql(rv,D[base].values).mean(),2),'CI',bb(d,block=24)[0])
# 17:00 sub-period: served vs raw_mean and vs har5
for p0,p1 in [('2024-07-01','2025-01-01'),('2025-01-01','2025-07-01'),('2025-07-01','2026-01-01'),('2026-01-01','2026-07-01'),('2026-07-01','2099-01-01')]:
    m=(T.dt>=pd.Timestamp(p0,tz='UTC'))&(T.dt<pd.Timestamp(p1,tz='UTC'))
    r=T.RV.values[m.values]; s=T.noctua_served_mean.values[m.values]; rm=T.noctua_raw_med.values[m.values]; h=T.har5_frozen.values[m.values]
    print(p0,'n',m.sum(),'served',round(ql(r,s).mean(),4),'raw_med',round(ql(r,rm).mean(),4),'har5',round(ql(r,h).mean(),4),'medRV/served',round(np.median(r/s),3))
