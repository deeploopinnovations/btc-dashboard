"""A2 attack (F5): independent breach recount from hourly high/low using the PUBLISHED forecast() safe_levels."""
import numpy as np, pandas as pd, json
from scipy.stats import norm
RUN = "/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10"
hours = pd.read_parquet(RUN + "/evals/hours_through_now.parquet").reset_index(drop=True)
hi = hours.high.values; lo = hours.low.values; cl = hours.close.values
H = 19
P = pd.read_csv(RUN + "/audits/scratch_A2/tails_payload_17utc.csv")
ts = hours.hour_ts.values
rows = P.row.values
# independent window extremes: bars a..a+18 vs close of bar a-1
maxhi = np.array([hi[a:a + H].max() for a in rows])
minlo = np.array([lo[a:a + H].min() for a in rows])
spot = cl[rows - 1]
print("spot matches payload max abs diff:", np.abs(spot - P.spot.values).max())
mup = np.maximum(0, np.log(maxhi / spot))
mdn = -np.minimum(0, np.log(minlo / spot))  # positive excursion down
dt = pd.to_datetime(P.dt)
res = {}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 3), round(100 * (c + h), 3)
# Gaussian reflection baseline with HAR5 frozen sigma from fresh_test episodes
E = pd.read_parquet(RUN + "/evals/fresh_test_episodes.parquet")
E = E.set_index("row")
har5 = E.loc[rows, "har5_frozen"].values if set(rows) <= set(E.index) else None
print("har5 available for all rows:", har5 is not None and np.isfinite(har5).all())
for al in [0.01, 0.02, 0.05, 0.10, 0.20]:
    cu = P[f"call_pct_{al:.2f}"].values; pdn = P[f"put_pct_{al:.2f}"].values
    uu = np.log1p(cu / 100.0); ud = -np.log1p(pdn / 100.0)
    bu = mup >= uu; bd = mdn >= ud
    n = len(rows)
    r = dict(n=n, up_breaches=int(bu.sum()), up_rate=round(100 * bu.mean(), 3), up_wilson=wilson(bu.sum(), n),
             dn_breaches=int(bd.sum()), dn_rate=round(100 * bd.mean(), 3), dn_wilson=wilson(bd.sum(), n),
             med_up_level_pct=round(100 * np.median(np.expm1(uu)), 3), med_dn_level_pct=round(100 * np.median(np.expm1(ud)), 3))
    if har5 is not None:
        gu = -har5 * norm.ppf(al / 2)
        r["gauss_har5_up_rate"] = round(100 * (mup >= gu).mean(), 3)
        r["gauss_har5_dn_rate"] = round(100 * (mdn >= gu).mean(), 3)
    # sub-periods
    per = {}
    for p0, p1 in [("2024-07-01", "2025-01-01"), ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"),
                   ("2026-01-01", "2026-07-01"), ("2026-07-01", "2026-08-15"), ("2026-08-15", "2099-01-01")]:
        m = ((dt >= pd.Timestamp(p0, tz="UTC")) & (dt < pd.Timestamp(p1, tz="UTC"))).values
        if m.sum() == 0: continue
        per[p0 + ".." + p1[:7]] = dict(n=int(m.sum()), up=round(100 * bu[m].mean(), 2), dn=round(100 * bd[m].mean(), 2))
    r["periods"] = per
    res[str(al)] = r
print(json.dumps(res, indent=1))
# sanity: payload levels monotone in alpha?
print("monotone up levels in alpha (median):", [round(float(np.median(P[f'call_pct_{a:.2f}'])), 3) for a in [0.01, 0.02, 0.05, 0.10, 0.20]])
# compare to fresh_test.json tails
F = json.load(open(RUN + "/evals/fresh_test.json"))["tails"]
print("fresh_test tails up_a0.01_noctua_raw:", F.get("up_a0.01_noctua_raw", {}).get("rate"), "served:", F.get("up_a0.01_noctua_served", {}).get("rate"))
print("fresh_test tails dn_a0.01_noctua_served:", F.get("dn_a0.01_noctua_served", {}).get("rate"))
# level scale vs realized excursion: how far beyond the realized maximum is the level?
u = np.log1p(P["call_pct_0.10"].values / 100)
print("alpha .10 up: share of days level > realized max excursion:", round(100 * (u > mup).mean(), 2),
      " median(level/realized max) for days where realized max>0:", round(float(np.median(np.expm1(u[mup > 0]) / np.expm1(mup[mup > 0]))), 3))
