import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
from policy import backtest as BT
P = pd.read_parquet("trader_positions.parquet")
p = P.pos.to_numpy(); R = P.tgt_ret.to_numpy(); fund = P.tgt_fund.to_numpy(); n=len(p)
vol = P.tr_vol30_ann.to_numpy()
def sh(x): return x.mean()/x.std(ddof=1)*np.sqrt(365)
def block_boot_diff(a, b, block=20, reps=4000, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(a)/block)); d = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, len(a), nb); ix = (st[:,None]+np.arange(block)[None,:]).ravel()[:len(a)] % len(a)
        d[r] = sh(a[ix]) - sh(b[ix])
    return float(sh(a)-sh(b)), [float(np.quantile(d,.025)), float(np.quantile(d,.975))], float((d<=0).mean())
print("== fee sensitivity (funding kept) ==")
for fee in [0, 3, 6, 12]:
    tr = BT.pnl(p, R, fund, fee); bh = BT.pnl(np.ones(n), R, fund, fee); c = BT.pnl(np.full(n,p.mean()), R, fund, fee)
    print(f"fee {fee:>2} bps: trader Sharpe {sh(tr):.4f} total {np.prod(1+tr)-1:+.4%} | B&H Sharpe {sh(bh):.4f} total {np.prod(1+bh)-1:+.4%} | const(0.139) Sharpe {sh(c):.4f} | trader-B&H {sh(tr)-sh(bh):+.4f}")
tr0 = BT.pnl(p,R,fund,0.0); c0 = BT.pnl(np.full(n,p.mean()),R,fund,0.0)
d,ci,pl = block_boot_diff(tr0, c0); print(f"fee=0: Sharpe(trader)-Sharpe(const) {d:+.4f} CI {ci} P<=0 {pl:.3f}")
tr6 = BT.pnl(p,R,fund,6.0); c6 = BT.pnl(np.full(n,p.mean()),R,fund,6.0)
d,ci,pl = block_boot_diff(tr6, c6); print(f"fee=6: Sharpe(trader)-Sharpe(const) {d:+.4f} CI {ci} P<=0 {pl:.3f}")
turn = np.abs(np.diff(p, prepend=0.0))
print(f"fee drag trader (6bps) per year: {6e-4*turn.sum()/n*365:.4%}; total {6e-4*turn.sum():.4%} of log wealth; mean fee/day {6e-4*turn.mean():.3e}")
print(f"daily sd of trader pnl {tr6.std(ddof=1):.5f}")
print("\n== matched-exposure vol-scaling benchmark (mean exposure = trader mean) ==")
for name, sigma in [("trailing30d rv", vol)]:
    base = np.clip(1.0/sigma, 0, None)
    # choose k so that mean(clip(k/sigma,0,0.5)) == mean(p)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        k = (lo+hi)/2
        m = np.clip(k/sigma, 0, 0.5).mean()
        lo, hi = (k, hi) if m < p.mean() else (lo, k)
    pv = np.clip(k/sigma, 0, 0.5)
    x = BT.pnl(pv, R, fund)
    print(f"{name}: mean pos {pv.mean():.4f}, Sharpe {sh(x):.4f}, corr(pos, trader pos) {np.corrcoef(pv,p)[0,1]:+.3f}")
# also a plain constant position matched exactly
x = BT.pnl(np.full(n,p.mean()),R,fund); print(f"const matched: Sharpe {sh(x):.4f}")
# Oracle-free sanity: best constant in-sample is irrelevant (Sharpe invariant to scale).
# trader's position regressed on trailing vol: R^2
b = np.polyfit(np.log(vol), p, 1); fit = np.polyval(b, np.log(vol))
print(f"position ~ log(trailing vol): slope {b[0]:+.4f}, R^2 {1-((p-fit)**2).sum()/((p-p.mean())**2).sum():.3f}")
xf = BT.pnl(fit,R,fund); print(f"fitted-vol-only position: Sharpe {sh(xf):.4f} mean {fit.mean():.4f}")
print("\n== Sharpe-vs-exposure edge at fee=6 bps, by sub-period (trader-minus-const, daily pnl mean*1e4) ==")
dt = pd.to_datetime(P.anchor_ts, unit="s", utc=True)
for h0,h1 in [("2024-07-01","2025-01-01"),("2025-01-01","2025-07-01"),("2025-07-01","2026-01-01"),("2026-01-01","2026-07-01"),("2026-07-01","2026-10-10")]:
    m = ((dt>=pd.Timestamp(h0,tz="UTC"))&(dt<pd.Timestamp(h1,tz="UTC"))).to_numpy()
    print(h0, h1, m.sum(), "mean diff bp/day", round((tr6[m]-c6[m]).mean()*1e4,3))
# Drop the single best 2026H1 stretch? Check leave-one-quarter-out Sharpe edge
print("\n== leave-one-quarter-out Sharpe(trader)-Sharpe(const) ==")
q = dt.dt.tz_localize(None).dt.to_period("Q").astype(str).to_numpy()
for qq in np.unique(q):
    m = q!=qq
    print(qq, round(sh(tr6[m])-sh(c6[m]),4))
