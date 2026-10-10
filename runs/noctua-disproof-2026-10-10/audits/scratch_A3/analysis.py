import sys, json
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
from policy import backtest as BT
from scipy import stats
P = pd.read_parquet("trader_positions.parquet")
p = P.pos.to_numpy(); R = P.tgt_ret.to_numpy(); fund = P.tgt_fund.to_numpy()
Rs = np.expm1(R)
n = len(p)
rng = np.random.default_rng(0)
def sh(x):
    s = x.std(ddof=1); return x.mean()/s*np.sqrt(365) if s>0 else 0.0
def mdd(pnl):
    w = np.cumprod(1+pnl); pk = np.maximum.accumulate(np.concatenate([[1.0],w]))[1:]
    return (w/pk-1).min()
def cagr(pnl):
    w = np.cumprod(1+pnl)[-1]; return w**(365/len(pnl))-1
def block_boot_diff(a, b, block=20, reps=4000, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(a)/block)); d = np.empty(reps)
    for r in range(reps):
        st = rng.integers(0, len(a), nb); ix = (st[:,None]+np.arange(block)[None,:]).ravel()[:len(a)] % len(a)
        d[r] = sh(a[ix]) - sh(b[ix])
    return float(sh(a)-sh(b)), [float(np.quantile(d,.025)), float(np.quantile(d,.975))], float((d<=0).mean())
out = {}
# 1. reproduce headline
tr = BT.pnl(p, R, fund)
bh = BT.pnl(np.ones(n), R, fund)
print("== reproduction ==")
print(f"trader Sharpe {sh(tr):.4f} CAGR {cagr(tr):.4%} MDD {mdd(tr):.4%} total {np.prod(1+tr)-1:.4%}")
print(f"B&H    Sharpe {sh(bh):.4f} CAGR {cagr(bh):.4%} MDD {mdd(bh):.4%} total {np.prod(1+bh)-1:.4%}")
print("fresh window (anchors >= 2026-08-15): trader", np.prod(1+tr[P.anchor_ts>=pd.Timestamp('2026-08-15',tz='UTC').timestamp()])-1,
      "B&H", np.prod(1+bh[P.anchor_ts>=pd.Timestamp('2026-08-15',tz='UTC').timestamp()])-1)
fresh = P.anchor_ts.to_numpy() >= pd.Timestamp('2026-08-15',tz='UTC').timestamp()
print("fresh n anchors", fresh.sum(), "first", pd.to_datetime(P.anchor_ts[fresh].min(),unit='s',utc=True), "last", pd.to_datetime(P.anchor_ts[fresh].max(),unit='s',utc=True))
# 2a. constant fractions (fee charged on entry only, as BT.pnl does)
print("\n== constant-fraction long positions (same costs) ==")
const = {}
for c in [0.05, 0.10, 0.1393, 0.20, 0.30, 0.50, 1.0]:
    x = BT.pnl(np.full(n, c), R, fund)
    const[c] = x
    print(f"c={c:.4f} Sharpe {sh(x):.4f} CAGR {cagr(x):.4%} MDD {mdd(x):.4%} total {np.prod(1+x)-1:.4%}")
d, ci, pl = block_boot_diff(tr, const[0.1393])
print(f"Sharpe(trader)-Sharpe(const 0.1393): {d:+.4f} CI95 {ci} P(<=0) {pl:.3f}")
d, ci, pl = block_boot_diff(tr, bh)
print(f"Sharpe(trader)-Sharpe(B&H): {d:+.4f} CI95 {ci} P(<=0) {pl:.3f}")
# 2b. trailing 60-day trailing mean of trader's own position, lagged (causal)
cs = np.concatenate([[0.0], np.cumsum(p)])
K = 60
t_idx = np.arange(K, n)
ptr = (cs[t_idx] - cs[t_idx-K]) / K              # mean of p[t-60..t-1]
sub = t_idx
tr_s = tr[sub]; prox = BT.pnl(ptr, R[sub], fund[sub])  # pnl of proxy on the same days
print("\n== causal 60d trailing-average position (on days 60..829) ==")
print(f"trader on same days: Sharpe {sh(tr_s):.4f} total {np.prod(1+tr_s)-1:.4%}")
print(f"proxy (lagged 60d mean of trader pos): Sharpe {sh(prox):.4f} total {np.prod(1+prox)-1:.4%} mean pos {ptr.mean():.4f} MDD {mdd(prox):.4%}")
print(f"corr(trader pos, proxy pos) on same days: {np.corrcoef(p[sub], ptr)[0,1]:.4f}")
d, ci, pl = block_boot_diff(tr_s, prox)
print(f"Sharpe(trader)-Sharpe(proxy): {d:+.4f} CI95 {ci} P(<=0) {pl:.3f}")
print(f"const 0.1393 on same days: Sharpe {sh(const[0.1393][sub]):.4f}")
# 3. correlation of daily position with next-day (held-period) return
print("\n== position vs return ==")
rho, pv = stats.pearsonr(p, R); print(f"pearson(pos, log ret held 24h) = {rho:+.4f} p={pv:.3f}")
rho, pv = stats.spearmanr(p, R); print(f"spearman(pos, ret) = {rho:+.4f} p={pv:.3f}")
rho, pv = stats.pearsonr(p, Rs); print(f"pearson(pos, simple ret) = {rho:+.4f} p={pv:.3f}")
# block bootstrap of the correlation
def bboot_corr(x, y, block=20, reps=4000, seed=1):
    rng = np.random.default_rng(seed); nb=int(np.ceil(len(x)/block)); c=np.empty(reps)
    for r in range(reps):
        st = rng.integers(0,len(x),nb); ix=(st[:,None]+np.arange(block)[None,:]).ravel()[:len(x)]%len(x)
        c[r]=np.corrcoef(x[ix],y[ix])[0,1]
    return [float(np.quantile(c,.025)), float(np.quantile(c,.975))], float((c<=0).mean())
print("block-bootstrap CI corr(pos, ret):", bboot_corr(p, R))
# within-position-level: does position predict return *relative to mean*? 
rho_pc = stats.pearsonr(p[:-1], p[1:])[0]; print(f"autocorr pos lag1 = {rho_pc:+.4f}")
print(f"corr(pos, trailing 30d ann vol) = {stats.pearsonr(p, P.tr_vol30_ann)[0]:+.4f}")
print(f"corr(pos, trailing 24h ret) = {stats.pearsonr(p, P.ret_24h_prev)[0]:+.4f}  (ret_24h feature, sign as in features)")
print(f"corr(pos, ret_1h feature) = {stats.pearsonr(p, P.ret_1h)[0]:+.4f}")
# regression of trader pnl on B&H pnl (beta, alpha)
b, a_ = np.polyfit(R, tr, 1)
print(f"regress trader pnl on BH log ret: beta {b:.4f} (mean pos {p.mean():.4f}), intercept/day {a_*1e4:.3f} bp, t-stat approx")
X = np.column_stack([np.ones(n), R]); resid = tr - X@np.linalg.lstsq(X,tr,rcond=None)[0]
cov = np.linalg.inv(X.T@X) * resid.var(ddof=2)
coef = np.linalg.lstsq(X,tr,rcond=None)[0]
print(f"  alpha/day {coef[0]*1e4:.3f} bp t={coef[0]/np.sqrt(cov[0,0]):.2f}; beta {coef[1]:.4f} t={coef[1]/np.sqrt(cov[1,1]):.2f}")
# covariance decomposition: timing term = cov(p, R)
print(f"cov(p,R)*n/ (mean(p) mean(R)) ratio: {np.cov(p,R)[0,1]/(p.mean()*R.mean()):.4f}")
# 2c. placebo: permute positions across days (keeps average exposure and dist)
rng = np.random.default_rng(42)
pl_sh = np.array([sh(BT.pnl(rng.permutation(p), R, fund)) for _ in range(2000)])
print(f"shuffle placebo (2000 perms): Sharpe mean {pl_sh.mean():.4f} sd {pl_sh.std():.4f}; trader {sh(tr):.4f}; one-sided p(placebo>=trader) = {(pl_sh>=sh(tr)).mean():.3f}")
# 4. cash yield / excess-return Sharpe
print("\n== excess-return (cash yield) sensitivity ==")
for rf_ann in [0.0, 0.03, 0.045]:
    rfd = (1+rf_ann)**(1/365)-1
    trx = p*(Rs - rfd) - 6e-4*np.abs(np.diff(p,prepend=0)) - p*fund
    bhx = (Rs - rfd) - fund - 6e-4*np.abs(np.r_[1.0, np.zeros(n-1)])*0  # fee on entry negligible
    bhx = (Rs - rfd) - fund
    print(f"rf={rf_ann:.3f}: trader excess Sharpe {sh(trx):.4f}  B&H excess Sharpe {sh(bhx):.4f}  diff {sh(trx)-sh(bhx):+.4f}")
# costs: funding and fees totals
print(f"\nfunding paid by trader (sum of p*fund): {-np.sum(p*fund):.5f} (log units), by B&H {-np.sum(fund):.5f}")
print(f"fee cost trader total: {np.sum(6e-4*np.abs(np.diff(p,prepend=0))):.6f}; mean turnover/day {np.abs(np.diff(p,prepend=0)).mean():.5f}")
print(f"mean daily funding (interest, bp): {fund.mean()*1e4:.2f}; fraction days funding>0: {(fund>0).mean():.3f}")
# per-half-year
dt = pd.to_datetime(P.anchor_ts, unit='s', utc=True)
per = dt.dt.to_period('Q')
print("\n== per quarter: trader vs B&H (compounded) ==")
g = pd.DataFrame({'q':per.astype(str),'tr':tr,'bh':bh,'pos':p}).groupby('q').agg(tr=('tr', lambda x: np.prod(1+x)-1), bh=('bh', lambda x: np.prod(1+x)-1), pos=('pos','mean'), n=('pos','size'))
print(g.to_string(float_format=lambda x: f"{x:.4f}"))
# sign flip check: per half
for h0,h1 in [("2024-07-01","2025-01-01"),("2025-01-01","2025-07-01"),("2025-07-01","2026-01-01"),("2026-01-01","2026-07-01"),("2026-07-01","2026-10-10")]:
    m = (dt>=pd.Timestamp(h0,tz='UTC'))&(dt<pd.Timestamp(h1,tz='UTC'))
    print(h0,h1,"n",m.sum(),"trader-minus-constant(0.1393) pnl sum",round(float(np.sum(tr[m]-const[0.1393][m])),5),
          "trader total",round(float(np.prod(1+tr[m])-1),4),"B&H total",round(float(np.prod(1+bh[m])-1),4))
json.dump({"sharpe_tr":sh(tr),"sharpe_bh":sh(bh)}, open("analysis_summary.json","w"))
