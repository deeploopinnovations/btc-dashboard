"""A2 attack (a): compare the real serve.predict.forecast() output to the fresh_test replica columns."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/user/btc-dashboard")
sys.path.insert(0, str(ROOT / "model"))
from serve.runtime import load_model
from serve.predict import forecast
from serve.adaptive import apply_correction

RUN = ROOT / "runs/noctua-disproof-2026-10-10"
hours = pd.read_parquet(RUN / "evals/hours_through_now.parquet").reset_index(drop=True)
E = pd.read_parquet(RUN / "evals/fresh_test_episodes.parquet")
ts = hours.hour_ts.to_numpy(np.int64)
model = load_model(ROOT / "model/serve/noctua_v2.npz")
print("meta version", model.meta.get("version"), "seeds", model.n_seeds, "hour_anchor", model.hour_anchor)

test = E[(E.dt >= pd.Timestamp("2024-07-01", tz="UTC"))].reset_index(drop=True)
idx = np.linspace(0, len(test) - 1, 20).round().astype(int)
sample = test.iloc[idx]
rows = []
t0 = time.time()
for _, r in sample.iterrows():
    raw = {}
    f = forecast(model, hours, anchor_ts=int(ts[int(r["row"])]), raw=raw, source="verify")
    pred, cal = raw["pred"], raw["cal"]
    sig_mean = float(pred["sigma_mean"][0]); sig_med = float(pred["sigma_med"][0])
    rows.append(dict(
        dt=str(r["dt"]), row=int(r["row"]),
        pub_sigma=f["sigma_window_pct"] / 100, par_served=float(r["noctua_served_mean"]),
        pub_sigma_pre_corr=sig_mean, par_raw_mean=float(r["noctua_raw_mean"]),
        raw_med=sig_med, par_raw_med=float(r["noctua_raw_med"]),
        cal_factor=cal["factor"], cal_applied=cal["applied"], cal_n=cal["n_episodes"],
        par_factor=float(r["factor"]),
        pub_sigma_med_pct=f["sigma_functional"]["sigma_med_pct"] / 100,
        pub_p_up=f["p_up"], safe_05=f["safe_levels"][2]["call_pct"],
    ))
df = pd.DataFrame(rows)
df["rel_served"] = df.pub_sigma / df.par_served - 1
df["rel_raw_mean"] = df.pub_sigma_pre_corr / df.par_raw_mean - 1
df["rel_raw_med"] = df.raw_med / df.par_raw_med - 1
df["fac_diff"] = df.cal_factor - df.par_factor
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df[["dt", "cal_factor", "par_factor", "cal_n", "rel_raw_med", "rel_raw_mean", "rel_served"]].to_string())
print("max |rel| served:", df.rel_served.abs().max(), " raw_mean:", df.rel_raw_mean.abs().max(), " raw_med:", df.rel_raw_med.abs().max())
print("max |fac diff|:", df.fac_diff.abs().max(), " all applied:", df.cal_applied.all())
print("elapsed", round(time.time() - t0), "s")
df.to_csv(RUN / "audits/scratch_A2/replica_check.csv", index=False)
