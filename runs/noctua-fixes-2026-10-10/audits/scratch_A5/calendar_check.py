"""Claim 1a/1b: cal_weekend_frac_ss vs independent pandas calendar; lookahead audit."""
import sys
sys.path.insert(0, "/home/user/btc-dashboard/model")
import numpy as np, pandas as pd
from serve.history import load_bundle
from noctua.features import build_features, audit_lookahead
HOUR = 3600
hours = load_bundle()
hour_ts = hours["hour_ts"].to_numpy(np.int64)
rng = np.random.default_rng(11)

# anchors: random rows + every hour-of-week slot represented, plus explicit Sunday 23:00 rows
rows = rng.choice(np.arange(9000, len(hours) - 200), 400, replace=False)
dtr = pd.to_datetime(hour_ts, unit="s", utc=True)
sun23 = np.flatnonzero((dtr.dayofweek == 6) & (dtr.hour == 23))[-30:]
sat00 = np.flatnonzero((dtr.dayofweek == 5) & (dtr.hour == 0))[-30:]
fri23 = np.flatnonzero((dtr.dayofweek == 4) & (dtr.hour == 23))[-30:]
rows = np.concatenate([rows, sun23, sat00, fri23])
Hs_choice = np.array([1, 2, 3, 6, 12, 19, 23, 24, 25, 47, 48, 97, 167, 168])
H = rng.choice(Hs_choice, len(rows))
H[: len(sun23)] = 1          # H=1 at Sunday 23:00 explicitly
H[len(sun23): len(sun23)+len(sat00)] = 168  # full week windows from Sat 00:00
ep = pd.DataFrame({"anchor_ts": hour_ts[rows], "H": H, "row": rows,
                   "dt": dtr[rows], "anchor_hour": dtr[rows].hour,
                   "dow": dtr[rows].dayofweek})
F = build_features(hours, ep)
ss = F["cal_weekend_frac_ss"].to_numpy(np.float64)
legacy = F["cal_weekend_frac"].to_numpy(np.float64)

def indep(t, h, days):
    out = []
    for a, n in zip(t, h):
        hrs = pd.to_datetime(int(a) + HOUR * np.arange(int(n)), unit="s", utc=True)
        out.append(np.isin(hrs.dayofweek, days).mean())
    return np.array(out)

ref_ss = indep(hour_ts[rows], H, (5, 6))
ref_fs = indep(hour_ts[rows], H, (4, 5))
err_ss = np.abs(ss - ref_ss).max()
err_fs = np.abs(legacy - ref_fs).max()
print(f"n episodes {len(rows)}  H values {sorted(set(H.tolist()))}")
print(f"max |cal_weekend_frac_ss - pandas Sat+Sun| = {err_ss:.3e}")
print(f"max |cal_weekend_frac    - pandas Fri+Sat| = {err_fs:.3e}")
print("exact equality (==) ss vs ref:", bool(np.array_equal(ss, ref_ss)))
print("sun23 H=1 values:", ss[: len(sun23)][:8], "ref", ref_ss[: len(sun23)][:8])
print("sat00 H=168 values (expect 2/7=%.6f):" % (2/7), np.unique(np.round(ss[len(sun23):len(sun23)+len(sat00)], 12)))
print("H=168 all within 1e-12 of 2/7:", bool(np.allclose(ss[H == 168], 2/7, atol=1e-12, rtol=0)))
print("values lie in [0,1]:", bool(((ss >= 0) & (ss <= 1)).all()))
print("count-based: ss*H is integer:", bool(np.allclose(ss * H, np.round(ss * H), atol=1e-9)))

# DST-irrelevance: UTC-only arithmetic; check a random set of arbitrary (non-hour-aligned) anchors too
t_arb = hour_ts[rows] + rng.integers(0, 3600, len(rows))
ep2 = ep.copy(); ep2["anchor_ts"] = t_arb
F2 = build_features(hours, ep2)
print("non-hour-aligned anchor ss vs pandas (max err):",
      f"{np.abs(F2['cal_weekend_frac_ss'].to_numpy() - indep(t_arb, H, (5,6))).max():.3e}")

# lookahead audit with the real audit function (corrupts rows >= cut, checks probed rows bit-identical)
ep3 = pd.DataFrame({"anchor_ts": hour_ts[rows], "H": H, "row": rows,
                    "dt": dtr[rows], "anchor_hour": dtr[rows].hour, "dow": dtr[rows].dayofweek})
res = audit_lookahead(hours, ep3, n_probe=200)
print("audit_lookahead:", {k: res[k] for k in res if k in ("leak_free", "max_abs_feature_change", "episodes_checked", "offenders")})
# direct: cal_weekend_frac_ss must not change when hours after the anchor are corrupted
corrupt = hours.copy()
mask = np.arange(len(corrupt)) > rows.max()
for c in ("rv5", "close", "volume", "rv5_pos", "rv5_neg", "bpv5", "rq5", "high", "low", "open"):
    v = corrupt[c].to_numpy(np.float64).copy(); v[mask] *= 3.0; corrupt[c] = v
F3 = build_features(corrupt, ep)
print("ss unchanged under future corruption (bitwise):",
      bool(np.array_equal(F3["cal_weekend_frac_ss"].to_numpy(), F["cal_weekend_frac_ss"].to_numpy())))
