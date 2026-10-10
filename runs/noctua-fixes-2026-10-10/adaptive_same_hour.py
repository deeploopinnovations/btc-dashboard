"""
runs/noctua-fixes-2026-10-10/adaptive_same_hour.py
=====================================================================
Does estimating serve/adaptive.py's trailing correction from settled episodes
at the SAME hour of day as the anchor (stride 24) beat the shipped estimate
pooled over every hour (stride 6)? Scored on the reported sigma_mean at all 24
anchor hours, 2024-07-01 onward. NOT out of sample for design: this era was
scored before (runs/noctua-disproof-2026-10-10). Pinned noctua_v2.npz.

    python runs/noctua-fixes-2026-10-10/adaptive_same_hour.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "runs/noctua-disproof-2026-10-10"))

from noctua.features import build_features      # noqa: E402
from serve.runtime import load_model             # noqa: E402
from fresh_test import load_hours_through_now    # noqa: E402

H, WIN, MINN, CLIP = 19, 60, 20, (0.70, 1.40)
TEST0 = pd.Timestamp("2024-07-01", tz="UTC")


def main() -> int:
    hours = load_hours_through_now()
    ts = hours.hour_ts.to_numpy(np.int64)
    n = len(hours)
    rv5 = hours.rv5.to_numpy(np.float64)
    c = np.concatenate([[0.0], np.cumsum(rv5)])
    RV = np.full(n, np.nan)
    RV[: n - H + 1] = np.sqrt(c[H:] - c[: n - H + 1])

    t0 = int(np.searchsorted(ts, int(TEST0.timestamp())))
    rows = np.arange(t0 - 24 * (2 * WIN + 3), n - H + 1)
    m = load_model(ROOT / "model/serve/noctua_v2.npz")
    med, mean, ok = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    for s in range(0, len(rows), 4096):
        r = rows[s:s + 4096]
        dt = pd.to_datetime(ts[r], unit="s", utc=True)
        X = build_features(hours, pd.DataFrame({"anchor_ts": ts[r], "H": H, "row": r, "dt": dt,
                                                "anchor_hour": dt.hour, "dow": dt.dayofweek}))
        p = m.predict(m.prepare(X, np.full(len(r), float(H))))
        med[r], mean[r] = p["sigma_med"], p["sigma_mean"]
        ok[r] = np.isfinite(X.to_numpy()).all(1)

    def factor(a, stride, same_hour, win=WIN):
        last = a - H
        first = max(24 * 30, last - win * 24)
        rr = np.arange(a - 24, first - 1, -24)[::-1] if same_hour else np.arange(first, last, stride)
        rr = rr[(rr + H <= a) & ok[rr]]
        s, v = med[rr], RV[rr]
        g = np.isfinite(s) & np.isfinite(v) & (s > 0) & (v > 0)
        if g.sum() < MINN:
            return 1.0
        return float(np.clip(np.median(v[g] / s[g]), *CLIP))

    test = np.arange(t0, n - H + 1)
    test = test[np.isfinite(RV[test]) & np.isfinite(mean[test])]
    f_old = np.array([factor(a, 6, False) for a in test])
    f_new = np.array([factor(a, 24, True) for a in test])
    # the two variants registered BEFORE seeing their numbers (see README):
    f_120 = np.array([factor(a, 24, True, 2 * WIN) for a in test])
    f_geo = np.sqrt(f_old * f_new)
    hr = pd.to_datetime(ts[test], unit="s", utc=True).hour.to_numpy()

    def ql(sig):
        r = RV[test] ** 2 / sig ** 2
        return r - np.log(r) - 1

    q_raw, q_old, q_new = ql(mean[test]), ql(mean[test] * f_old), ql(mean[test] * f_new)
    rng = np.random.default_rng(0)
    out = {"n_anchors": int(len(test)), "by_hour": {}}

    def boot(d, block=24 * 20, reps=2000):
        k = len(d)
        st = rng.integers(0, k, (reps, int(np.ceil(k / block))))
        idx = (st[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :k] % k
        mm = d[idx].mean(1)
        return [float(np.quantile(mm, .025)), float(np.quantile(mm, .975))]

    d = q_old - q_new                                    # > 0: same-hour better
    out["pooled"] = {"qlike_raw_mean": float(q_raw.mean()), "qlike_served_shipped": float(q_old.mean()),
                     "qlike_served_same_hour": float(q_new.mean()), "gain": float(d.mean()),
                     "gain_ci95_block20d": boot(d)}
    for h in range(24):
        k = hr == h
        dh = d[k]
        out["by_hour"][h] = {"shipped": float(q_old[k].mean()), "same_hour": float(q_new[k].mean()),
                             "gain": float(dh.mean()), "gain_ci95": boot(dh, block=20),
                             "factor_shipped_mean": float(f_old[k].mean()),
                             "factor_same_hour_mean": float(f_new[k].mean()),
                             "median_RV_over_served_med_shipped": float(np.median(RV[test][k] / (med[test][k] * f_old[k]))),
                             "median_RV_over_served_med_same_hour": float(np.median(RV[test][k] / (med[test][k] * f_new[k])))}
    out["hours_same_hour_better"] = int(sum(v["gain"] > 0 for v in out["by_hour"].values()))
    for name, f in (("same_hour_120d", f_120), ("geo_blend", f_geo)):
        dv = q_old - ql(mean[test] * f)
        out[name] = {"qlike": float(ql(mean[test] * f).mean()), "gain": float(dv.mean()),
                     "gain_ci95_block20d": boot(dv),
                     "hours_better": int(sum(dv[hr == h].mean() > 0 for h in range(24))),
                     "gain_by_hour": {h: float(dv[hr == h].mean()) for h in range(24)}}
    (HERE / "evals/adaptive_same_hour.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["pooled"], indent=1), out["hours_same_hour_better"])
    print({h: round(v["gain"], 4) for h, v in out["by_hour"].items()})
    for name in ("same_hour_120d", "geo_blend"):
        o = out[name]
        print(name, round(o["qlike"], 4), round(o["gain"], 4), o["gain_ci95_block20d"], o["hours_better"],
              {h: round(g, 4) for h, g in o["gain_by_hour"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
