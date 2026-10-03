"""
noctua/add_weekend_anchor.py
=====================================================================
Attach the TRUE weekend fraction to the served anchor (P4-weekend-fix-result)
without retraining anything -- the pattern of noctua/add_hour_anchor.py.

The shipped anchor's weekend column counts Fri+Sat (P4-weekend-bug) and stays
as it is: the network was trained on it. What is added is an INCREMENT, fitted
by regressing the shipped anchor's own residual on [1, true weekend fraction]
over the artifact's training split (same split, finite mask, the ARTIFACT's
standardisation, sample weights and target). The shipped har_beta is kept
byte-identical; the joint-refit coefficient is printed beside the incremental
one so the difference is measured, not assumed.

Two increments, because the clock-aware anchor (HOUR_ANCHOR) may be on too:

    har_beta_weekend         (2,)  on the residual of har_beta . [1, Xb]
    har_beta_weekend_season  (2,)  on the residual of the clock-aware anchor,
                                   har_beta_season . [1, Xb, season_fwd]

Serving reads them only when serve/predict.WEEKEND_ANCHOR is on, and picks the
second when HOUR_ANCHOR is on as well. Every existing array is written back
unchanged.

    python -m model.noctua.add_weekend_anchor             # dry run
    python -m model.noctua.add_weekend_anchor --write
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
from noctua.calendar import weekend_frac                               # noqa: E402
from noctua.season import season_fwd                                   # noqa: E402
from noctua.train import load_all                                      # noqa: E402

PROD_A, PROD_H = 17, 19


def fit(artifact: Path, artifacts_dir: Path) -> dict:
    z = np.load(artifact, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    if "season_profile" not in z.files or "har_beta_season" not in z.files:
        raise SystemExit("REFUSING: the artifact has no clock-aware anchor arrays; "
                         "run add_hour_anchor first so both increments are fitted")
    base = list(meta["base_cols"])
    ep, X = load_all(artifacts_dir)
    fin = np.isfinite(X.to_numpy()).all(1)
    sp = S.time_splits(ep, train_end=meta.get("train_end", S.TRAIN_END),
                       calib_end=meta.get("calib_end", S.CALIB_END))
    m_tr = sp["train"] & fin
    mu, sd = z["std_base_mu"], z["std_base_sd"]
    Xb = (X.loc[m_tr, base].to_numpy(np.float64) - mu) / sd
    H = ep.H.to_numpy(np.float64)
    y = B.har_target(ep.RV.to_numpy(), H)[m_tr]
    w = S.sample_weights(ep, m_tr)
    ts = ep["anchor_ts"].to_numpy(np.int64)[m_tr]
    wf = weekend_frac(ts, H[m_tr].astype(np.int64))

    hb = np.asarray(z["har_beta"], np.float64)
    bs = np.asarray(z["har_beta_season"], np.float64)
    sf = season_fwd(z["season_profile"], ep["anchor_hour"].to_numpy()[m_tr], H[m_tr])
    r0 = y - (hb[0] + Xb @ hb[1:])
    rs = y - (bs[0] + Xb @ bs[1:-1] + bs[-1] * sf)
    col = pd.DataFrame({"wf": wf})
    inc0 = np.asarray(B.OLS(["wf"]).fit(col, r0, w).beta, np.float64)
    incs = np.asarray(B.OLS(["wf"]).fit(col, rs, w).beta, np.float64)
    Xj = pd.DataFrame(Xb, columns=base)
    Xj["wf"] = wf
    joint = np.asarray(B.OLS(base + ["wf"]).fit(Xj, y, w).beta)
    return {"z": z, "meta": meta, "inc0": inc0, "incs": incs,
            "joint_coef": float(joint[-1]), "n_train": int(m_tr.sum()),
            "corr_sf_wf": float(np.corrcoef(sf, wf)[0, 1])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="add the true-weekend anchor increment")
    ap.add_argument("--artifact", type=Path, default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    r = fit(a.artifact, a.artifacts)
    z, meta, inc0, incs = r["z"], r["meta"], r["inc0"], r["incs"]
    wb = 1 - meta["blend_w"]
    print(f"training rows {r['n_train']:,}")
    print(f"har_beta_weekend        {np.round(inc0, 5)}  (on the shipped anchor)")
    print(f"har_beta_weekend_season {np.round(incs, 5)}  (on the clock-aware anchor)")
    print(f"weekend coef: incremental {inc0[1]:+.4f}   joint refit {r['joint_coef']:+.4f}")
    print(f"corr(season_fwd, true weekend) on training rows {r['corr_sf_wf']:+.4f}")
    for name, wf in (("Mon-Thu night", 0.0), ("Fri night", 12 / 19), ("Sat night", 1.0),
                     ("Sun night", 7 / 19)):
        print(f"  {name:>13}: served log level moves {wb * (inc0[0] + inc0[1] * wf):+.4f}")
    if not a.write:
        print("dry run; pass --write to add the arrays to the artifact")
        return 0
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    arrays["har_beta_weekend"] = inc0
    arrays["har_beta_weekend_season"] = incs
    meta = dict(meta)
    meta["weekend_anchor"] = {
        "source": "P4-weekend-fix-result",
        "column": "true Sat+Sun UTC fraction of the forward window "
                  "(noctua/calendar.py); the shipped cal_weekend_frac counts Fri+Sat",
        "fit": "increment on the residual of the shipped anchor (and, separately, "
               "of the clock-aware anchor) regressed on [1, weekend fraction] over "
               "the artifact's training split; har_beta unchanged",
        "weekend_coef": float(inc0[1]),
        "weekend_coef_on_season": float(incs[1]),
        "joint_refit_coef": r["joint_coef"]}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(a.artifact, **arrays)
    print(f"wrote {a.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
