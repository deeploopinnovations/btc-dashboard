import json, sys, urllib.request, datetime as dt
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
start = int(dt.datetime(2026,8,1,tzinfo=dt.timezone.utc).timestamp()*1000)
end = int(dt.datetime(2026,10,10,16,tzinfo=dt.timezone.utc).timestamp()*1000)
rows=[]; cur=start
while cur<end:
    url=f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=5m&startTime={cur}&endTime={end}&limit=1000"
    with urllib.request.urlopen(url,timeout=60) as r: b=json.load(r)
    if not b: break
    rows+=b; cur=b[-1][0]+300_000
df=pd.DataFrame(rows,columns=["t","o","h","l","c","v","ct","qv","n","tb","tq","i"])
df["ts"]=df.t//1000; df["c"]=df.c.astype(float)
df["hour_ts"]=(df.ts//3600)*3600
df["lr"]=np.log(df.c).diff()
# hourly rv5 = sum of squared 5m log returns within the hour (first bar's return vs previous bar close)
g=df.groupby("hour_ts")
bn=pd.DataFrame({"rv5_bn":g.lr.apply(lambda x: np.nansum(x.values**2)), "n":g.size(), "close_bn": g.c.last()})
h=pd.read_parquet("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/evals/hours_through_now.parquet").set_index("hour_ts")
a=pd.read_parquet("/home/user/btc-dashboard/data/assets/btc_history.parquet").set_index("hour_ts")
m=bn.join(h[["rv5","close"]],how="inner")
m=m[m.n==12]
m["ratio"]=m.rv5/m.rv5_bn
m["day"]=pd.to_datetime(m.index,unit="s",utc=True).date
print("hours compared", len(m), "first", pd.to_datetime(m.index.min(),unit='s',utc=True), "last", pd.to_datetime(m.index.max(),unit='s',utc=True))
print("corr(rv5 parquet, rv5 binance 5m):", round(np.corrcoef(m.rv5,m.rv5_bn)[0,1],4))
print("median ratio parquet/binance:", round(m.ratio.median(),4), " IQR", m.ratio.quantile([.25,.75]).round(4).tolist())
print("daily sum rv ratio (median over days):")
dd=m.groupby("day").agg(rv_p=("rv5","sum"),rv_b=("rv5_bn","sum"),n=("rv5","size"))
dd["r"]=dd.rv_p/dd.rv_b
print(dd.r.describe().round(3).to_string())
print("days with ratio outside [0.5,2]:", dd[(dd.r<0.5)|(dd.r>2)].index.tolist())
# high/low check vs binance hourly
cut=pd.Timestamp("2026-08-15",tz="UTC").timestamp()
for name,mm in [("pre 08-15 (committed)", m[m.index<cut]), ("08-15 onward (live tail)", m[m.index>=cut])]:
    print(name, "n", len(mm), "corr", round(np.corrcoef(mm.rv5,mm.rv5_bn)[0,1],4), "median ratio", round(mm.ratio.median(),4), "mean daily ratio", round((mm.groupby('day').rv5.sum()/mm.groupby('day').rv5_bn.sum()).mean(),4))
