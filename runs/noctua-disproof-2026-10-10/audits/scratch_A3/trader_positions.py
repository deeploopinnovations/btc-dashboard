# Recompute NOCTUA-Trader v1 positions exactly as fresh_test.py's trader block does.
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/user/btc-dashboard")
sys.path.insert(0, str(ROOT / "model"))
from policy import dataset as DS
from policy.runtime import NumpyTrader
t0 = time.time()
hours = pd.read_parquet(ROOT / "runs/noctua-disproof-2026-10-10/evals/hours_through_now.parquet").reset_index(drop=True)
ts = hours.hour_ts.to_numpy(np.int64)
dt_all = pd.to_datetime(ts, unit="s", utc=True)
n = len(hours)
TEST_START = pd.Timestamp("2024-07-01", tz="UTC")
rows_tr = np.flatnonzero((dt_all.hour == 17) & (dt_all >= TEST_START))
rows_tr = rows_tr[rows_tr + DS.HOLD_H <= n]
F = DS.features_at(hours, rows_tr)
T = DS.targets(hours, rows_tr)
trader = NumpyTrader()
pos = trader.position(F)
out = pd.DataFrame({"anchor_ts": F.anchor_ts.to_numpy(np.int64), "pos": pos,
                    "tgt_ret": T.tgt_ret.to_numpy(), "tgt_fund": T.tgt_fund.to_numpy(),
                    "dvol_present": F.dvol_present.to_numpy(), "fund_present": F.fund_present.to_numpy(),
                    "ret_24h_prev": F.ret_24h.to_numpy() if "ret_24h" in F else np.nan,
                    "ret_1h": F.ret_1h.to_numpy() if "ret_1h" in F else np.nan})
# trailing 30d realised vol (causal, rv5 up to anchor-1), like fresh_test
rv5 = hours.rv5.to_numpy(np.float64)
c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(rv5))])
idx = rows_tr
out["tr_vol30_ann"] = np.sqrt((c[idx] - c[idx - 720]) / 720 * 24 * 365)
out.to_parquet("/home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A3/trader_positions.parquet")
print("n", len(out), "avg pos", out.pos.mean(), "min", out.pos.min(), "max", out.pos.max(), "time", round(time.time()-t0))
print("quantiles", out.pos.quantile([0,.05,.25,.5,.75,.95,1]).round(4).tolist())
print("dvol_present mean", out.dvol_present.mean(), "fund_present mean", out.fund_present.mean())
print("feature cols with ret_ :", [c for c in F.columns if c.startswith("ret_")])
