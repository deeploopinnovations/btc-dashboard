"""
noctua/add_hour_anchor.py
=====================================================================
Attach the hour-aware anchor (P4-hour-anchor-result) to an EXISTING artifact
without retraining anything.

The anchor is an OLS with no hidden state, so a clock term can be added after
the fact. The shipped har_beta is KEPT EXACTLY: the script fits only an
INCREMENT -- an intercept adjustment and the season coefficient -- by
regressing the shipped anchor's own residual, y - har_beta . [1, Xb], on
[1, season_fwd] over the artifact's training split (same split, same finite
mask, the ARTIFACT's standardisation, same sample weights, same target).

Why not refit jointly: the first version refused to write because har_beta,
reproduced from today's features.parquet on the same 189,831 training rows,
differed from the artifact's by 1.1e-2 -- the feature values have moved since
the artifact was fitted (base-column means by ~2.6e-4). A joint refit would
silently replace the shipped anchor with a different one. The increment
leaves it byte-identical and adds only what the clock contributes; season_fwd
is built from whole-window averages of a daily cycle and is close to
orthogonal to the HAR regressors, and the joint coefficient is printed beside
the incremental one so the difference is measured rather than assumed.

    season_profile     (24,)   centred log-vol per clock hour, from train
    har_beta_season    (7,)    har_beta with the intercept adjusted, then the
                               season coefficient

Every existing array is written back unchanged. Serving reads the new ones
only when serve/predict.HOUR_ANCHOR is on.

    python -m model.noctua.add_hour_anchor                 # dry run: fit + report
    python -m model.noctua.add_hour_anchor --write         # add to the artifact
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
from noctua.season import hour_profile, season_fwd                     # noqa: E402
from noctua.train import load_all                                      # noqa: E402

PROFILE_H = 19          # one row per anchor; the table repeats features per H


def fit(artifact: Path, artifacts_dir: Path) -> dict:
    z = np.load(artifact, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
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

    ref = B.OLS(base).fit(pd.DataFrame(Xb, columns=base), y, w).beta
    # staleness of today's features against the artifact, REPORTED; the
    # increment below does not depend on reproducing har_beta
    err = float(np.max(np.abs(np.asarray(ref) - z["har_beta"])))
    hb = np.asarray(z["har_beta"], np.float64)
    resid = y - (hb[0] + Xb @ hb[1:])

    ah = ep["anchor_hour"].to_numpy()
    one = m_tr & (ep.H == PROFILE_H).to_numpy()
    prof = hour_profile(X["har_1h"].to_numpy(np.float64)[one], ah[one])
    sf = season_fwd(prof, ah[m_tr], H[m_tr])
    inc = np.asarray(B.OLS(["season_fwd"]).fit(
        pd.DataFrame({"season_fwd": sf}), resid, w).beta)
    beta = np.concatenate([[hb[0] + inc[0]], hb[1:], [inc[1]]])
    Xs = pd.DataFrame(Xb, columns=base)
    Xs["season_fwd"] = sf
    joint = np.asarray(B.OLS(base + ["season_fwd"]).fit(Xs, y, w).beta)
    return {"z": z, "meta": meta, "profile": prof, "beta": beta,
            "joint_coef": float(joint[-1]), "har_beta_err": err,
            "n_train": int(m_tr.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="add the hour-aware anchor")
    ap.add_argument("--artifact", type=Path,
                    default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    r = fit(a.artifact, a.artifacts)
    z, meta, prof, beta = r["z"], r["meta"], r["profile"], r["beta"]
    print(f"training rows {r['n_train']:,}; har_beta reproduced from today's "
          f"features to {r['har_beta_err']:.2e} (staleness, reported -- the "
          f"shipped har_beta is kept exactly)")
    print("season profile (centred log vol by clock hour):")
    print("  " + " ".join(f"{h:02d}:{v:+.2f}" for h, v in enumerate(prof)))
    print(f"har_beta          {np.round(z['har_beta'], 5)}")
    print(f"har_beta_season   {np.round(beta, 5)}")
    print(f"season coef: incremental {beta[-1]:+.4f}   joint refit {r['joint_coef']:+.4f}")
    at17 = season_fwd(prof, np.array([17]), np.array([19]))[0]
    print(f"season_fwd at the production anchor (17:00, H=19): {at17:+.4f} -> "
          f"anchor moves by {beta[-1] * at17:+.4f} in log vol, "
          f"{(1 - meta['blend_w']) * beta[-1] * at17:+.4f} after the blend")
    if not a.write:
        print("dry run; pass --write to add the arrays to the artifact")
        return 0
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    arrays["season_profile"] = prof.astype(np.float64)
    arrays["har_beta_season"] = beta.astype(np.float64)
    meta = dict(meta)
    meta["hour_anchor"] = {
        "source": "P4-hour-anchor-result",
        "profile": f"har_1h on training anchors at H={PROFILE_H}, clock hour "
                   f"= anchor_hour - 1, missing-hour floor excluded",
        "fit": "increment on the shipped anchor: residual y - har_beta.[1,Xb] "
               "regressed on [1, season_fwd] over the artifact's training "
               "split; har_beta itself unchanged",
        "season_coef": float(beta[-1]),
        "joint_refit_coef": r["joint_coef"],
        "features_staleness_har_beta": r["har_beta_err"]}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(a.artifact, **arrays)
    print(f"wrote {a.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
