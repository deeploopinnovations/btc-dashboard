import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
from policy import backtest as BT
P = pd.read_parquet("trader_positions.parquet")
p = P.pos.to_numpy(); R = P.tgt_ret.to_numpy(); fund = P.tgt_fund.to_numpy(); n = len(p)
def sh(x): return x.mean()/x.std(ddof=1)*np.sqrt(365)
tr = BT.pnl(p, R, fund); c = BT.pnl(np.full(n, p.mean()), R, fund)
obs = sh(tr) - sh(c)
# circular-shift placebo: keeps the position's autocorrelation and turnover, breaks its alignment to returns
diffs = []
for k in range(1, n):
    ps = np.roll(p, k)
    diffs.append(sh(BT.pnl(ps, R, fund)) - sh(c))
diffs = np.array(diffs)
print(f"circular-shift placebo ({n-1} shifts): mean Sharpe edge {diffs.mean():+.4f}, observed {obs:+.4f}, one-sided p={(diffs>=obs).mean():.3f}")
# Sharpe of shifted trader (no alignment) vs B&H
# cash-yield CAGR view: investor holds the rest in cash at rf
for rf in [0.0, 0.045]:
    rfd = (1+rf)**(1/365)-1
    tot = p*np.expm1(R) + (1-p)*rfd - BT.DEFAULT_FEE_BPS*1e-4*np.abs(np.diff(p,prepend=0)) - p*fund
    w = np.cumprod(1+tot)[-1]
    bh_tot = BT.pnl(np.ones(n), R, fund)
    print(f"cash at {rf:.1%}: trader CAGR {w**(365/n)-1:.4%} total {w-1:.4%} | B&H CAGR {np.prod(1+bh_tot)**(365/n)-1:.4%}")
