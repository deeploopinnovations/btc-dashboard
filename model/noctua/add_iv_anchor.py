"""
noctua/add_iv_anchor.py
=====================================================================
Attach the next-day implied-vol increment to the served anchor
(P4-iv1d-posthoc / P4-iv1d-proxy-control; forward test frozen in
research/DATA_USE.md) without retraining -- the pattern of add_hour_anchor.

At the 17:00 UTC / H = 19 production anchor only:

    anchor += a + b * (log hourly implied vol - shipped anchor)

fitted by OLS of the shipped anchor's residual, y - har_beta . [1, Xb], on
[1, x] over production nights WITH a next-day IV (noctua/iv1d.py) from
2019-12-01 to 2026-06-30 (the whole IV era up to three months before the
freeze), sample weights as in training. The shipped har_beta stays
byte-identical. Only valid with the other anchor increments OFF (it is fitted
on the shipped anchor's residual).

    har_beta_iv   (2,)  [a, b]

    python -m model.noctua.add_iv_anchor             # dry run
    python -m model.noctua.add_iv_anchor --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua import baselines as B                                      # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.iv1d import log_hourly                                     # noqa: E402
from noctua.train import load_all                                      # noqa: E402

FIT_END = "2026-07-01"
IV_PATH = Path("model/artifacts/iv1d.parquet")


def fit(artifact: Path, artifacts_dir: Path) -> dict:
    z = np.load(artifact, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    base = list(meta["base_cols"])
    ep, X = load_all(artifacts_dir)
    fin = np.isfinite(X.to_numpy()).all(1)
    prod = np.asarray(S.production_mask(ep), bool)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    day = pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d")
    iv = pd.read_parquet(IV_PATH)
    lut = dict(zip(iv["day"], iv["iv_nextday"]))
    ivv = np.array([lut.get(d, np.nan) for d in day], np.float64)
    end = int(pd.Timestamp(FIT_END, tz="UTC").timestamp())
    m = fin & prod & np.isfinite(ivv) & (ivv > 0) & (ts + 19 * 3600 < end)
    mu, sd = z["std_base_mu"], z["std_base_sd"]
    Xb = (X.loc[m, base].to_numpy(np.float64) - mu) / sd
    hb = np.asarray(z["har_beta"], np.float64)
    anc = hb[0] + Xb @ hb[1:]
    y = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))[m]
    x = np.array([log_hourly(v) for v in ivv[m]]) - anc
    w = S.sample_weights(ep, m)
    beta = np.asarray(B.OLS(["x"]).fit(pd.DataFrame({"x": x}), y - anc, w).beta, np.float64)
    return {"z": z, "meta": meta, "beta": beta, "n": int(m.sum()),
            "span": (day[m].min(), day[m].max()), "x_median": float(np.median(x))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="add the next-day implied-vol increment")
    ap.add_argument("--artifact", type=Path, default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    r = fit(a.artifact, a.artifacts)
    z, meta, beta = r["z"], r["meta"], r["beta"]
    print(f"production nights with IV: {r['n']} ({r['span'][0]} .. {r['span'][1]}), "
          f"median x {r['x_median']:+.3f}")
    print(f"har_beta_iv  a {beta[0]:+.4f}  b {beta[1]:+.4f}")
    wb = 1 - meta["blend_w"]
    for xv in (-0.4, 0.0, 0.1, 0.4):
        print(f"  x = {xv:+.1f}: served log level moves {wb * (beta[0] + beta[1] * xv):+.4f}")
    if not a.write:
        print("dry run; pass --write to add the array to the artifact")
        return 0
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    arrays["har_beta_iv"] = beta
    meta = dict(meta)
    meta["iv_anchor"] = {
        "source": "P4-iv1d-posthoc / P4-iv1d-proxy-control; forward holdout in DATA_USE.md",
        "feature": "ATM IV of the Deribit option expiring 08:00 UTC next day, trades in "
                   "[16:00, 17:00) UTC (noctua/iv1d.py); x = log hourly IV - shipped anchor",
        "applies": "17:00 UTC anchor, H = 19 only; other anchor increments must be off",
        "fit": f"residual of the shipped anchor on [1, x], production nights with IV "
               f"{r['span'][0]}..{r['span'][1]} (< {FIT_END}), n = {r['n']}",
        "a": float(beta[0]), "b": float(beta[1])}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(a.artifact, **arrays)
    print(f"wrote {a.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
