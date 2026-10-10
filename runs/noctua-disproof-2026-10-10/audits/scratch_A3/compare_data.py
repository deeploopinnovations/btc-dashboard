import json, sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
ROOT = "/home/user/btc-dashboard"
bn = pd.DataFrame(json.load(open("binance_1h.json")), columns=["open_time","open","high","low","close","volume","close_time","qv","n","tbv","tqv","ig"])
bn["hour_ts"] = bn.open_time.astype(np.int64)//1000
bn["close"] = bn.close.astype(float)
bn = bn.set_index("hour_ts")
h = pd.read_parquet(f"{ROOT}/runs/noctua-disproof-2026-10-10/evals/hours_through_now.parquet").set_index("hour_ts")
a = pd.read_parquet(f"{ROOT}/data/assets/btc_history.parquet").set_index("hour_ts")
print("assets end:", pd.to_datetime(a.index.max(), unit="s", utc=True), " hours_through_now end:", pd.to_datetime(h.index.max(), unit="s", utc=True))
# hourly comparison over overlap
for name, df in [("assets", a), ("hours_through_now", h)]:
    common = df.index.intersection(bn.index)
    d = (df.loc[common, "close"] / bn.loc[common, "close"] - 1)
    d = d[df.index.intersection(bn.index).isin(common)]
    dd = pd.to_datetime(common, unit="s", utc=True)
    print(f"\n[{name}] hourly overlap n={len(common)} first={dd.min()} last={dd.max()}")
    print(f"  close diff: mean={d.mean()*100:.4f}% median={d.median()*100:.4f}% p05={d.quantile(.05)*100:.4f}% p95={d.quantile(.95)*100:.4f}% max|={d.abs().max()*100:.3f}%")
    for per in [("2026-07-25","2026-08-15"),("2026-08-15","2026-10-11")]:
        s,e = (pd.Timestamp(x,tz="UTC").timestamp() for x in per)
        m = (common>=s)&(common<e)
        if m.sum():
            print(f"  period {per}: n={m.sum()} mean|diff|={d[m].abs().mean()*100:.4f}% max|diff|={d[m].abs().max()*100:.3f}%")
# daily 16:00 UTC entry bars
rows=[]
for day in pd.date_range("2026-08-01","2026-10-10",freq="D",tz="UTC"):
    t = int(day.timestamp()) + 16*3600
    rows.append({"day":day.date(), "bn_16":bn.close.get(t,np.nan), "assets_16":a.close.get(t,np.nan), "now_16":h.close.get(t,np.nan)})
R = pd.DataFrame(rows)
R["assets_vs_bn_%"] = (R.assets_16/R.bn_16-1)*100
R["now_vs_bn_%"] = (R.now_16/R.bn_16-1)*100
pd.set_option("display.width",200)
print(R.to_string(float_format=lambda x: f"{x:,.4f}"))
R.to_csv("daily16_compare.csv", index=False)
# flags
flag = R[(R["assets_vs_bn_%"].abs()>0.5)|(R["now_vs_bn_%"].abs()>0.5)]
print("\nDays with |diff|>0.5%:", len(flag), flag.day.tolist())
print("max |now diff| over days with now data:", R["now_vs_bn_%"].abs().max())
