"""
noctua/add_dow_anchor.py
=====================================================================
Attach the day-of-week increment to the served anchor (P4-dow-anchor-result)
without retraining -- the pattern of add_hour_anchor / add_weekend_anchor.

The shipped anchor's weekend column counts Fri+Sat (P4-weekend-bug) and stays.
The increment regresses the shipped anchor's own residual on
[1, window fraction on Mon, Tue, Wed, Thu, Fri, Sat] (Sunday the reference;
all six are needed because the residual regression has no Fri+Sat column --
eval/dow_staleness.py measured the five-column version losing most of the
2026 gain) over the artifact's training split, same weights and target. The
shipped har_beta stays byte-identical.

Fitted on the artifact's split (through 2023-01-01), the coefficients are
older than the walk-forward's latest folds; eval/dow_staleness.py measured
such frozen coefficients keeping 0.82 of a yearly refit's Brier gain over
2023-2026 (0.62 in 2026), positive in every year.

    har_beta_dow         (7,)  on the residual of har_beta . [1, Xb]
    har_beta_dow_season  (7,)  on the residual of the clock-aware anchor

    python -m model.noctua.add_dow_anchor             # dry run
    python -m model.noctua.add_dow_anchor --write
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
from noctua.calendar import DOW_COLS, day_fracs                        # noqa: E402
from noctua.season import season_fwd                                   # noqa: E402
from noctua.train import load_all                                      # noqa: E402

NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def fit(artifact: Path, artifacts_dir: Path) -> dict:
    z = np.load(artifact, allow_pickle=False)
    meta = json.loads(bytes(z["meta_json"]).decode())
    if "season_profile" not in z.files or "har_beta_season" not in z.files:
        raise SystemExit("REFUSING: run add_hour_anchor first so both increments are fitted")
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
    F = day_fracs(ep["anchor_ts"].to_numpy(np.int64)[m_tr], H[m_tr].astype(np.int64))
    cols = [f"d{d}" for d in DOW_COLS]
    D = pd.DataFrame(F[:, list(DOW_COLS)], columns=cols)

    hb = np.asarray(z["har_beta"], np.float64)
    bs = np.asarray(z["har_beta_season"], np.float64)
    sf = season_fwd(z["season_profile"], ep["anchor_hour"].to_numpy()[m_tr], H[m_tr])
    r0 = y - (hb[0] + Xb @ hb[1:])
    rs = y - (bs[0] + Xb @ bs[1:-1] + bs[-1] * sf)
    inc0 = np.asarray(B.OLS(cols).fit(D, r0, w).beta, np.float64)
    incs = np.asarray(B.OLS(cols).fit(D, rs, w).beta, np.float64)
    return {"z": z, "meta": meta, "inc0": inc0, "incs": incs, "n_train": int(m_tr.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="add the day-of-week anchor increment")
    ap.add_argument("--artifact", type=Path, default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    r = fit(a.artifact, a.artifacts)
    z, meta, inc0, incs = r["z"], r["meta"], r["inc0"], r["incs"]
    wb = 1 - meta["blend_w"]
    print(f"training rows {r['n_train']:,}")
    print("har_beta_dow         intercept {:+.4f}  ".format(inc0[0])
          + "  ".join(f"{NAMES[d]} {c:+.4f}" for d, c in zip(DOW_COLS, inc0[1:])))
    print("har_beta_dow_season  intercept {:+.4f}  ".format(incs[0])
          + "  ".join(f"{NAMES[d]} {c:+.4f}" for d, c in zip(DOW_COLS, incs[1:])))
    from noctua.calendar import day_fracs_from_clock
    for d in range(7):
        f = day_fracs_from_clock(np.array([d]), np.array([17]), np.array([19]))[0]
        move = wb * (inc0[0] + f[list(DOW_COLS)] @ inc0[1:])
        print(f"  {NAMES[d]} 17:00 night: served log level moves {move:+.4f}")
    if not a.write:
        print("dry run; pass --write to add the arrays to the artifact")
        return 0
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    arrays["har_beta_dow"] = inc0
    arrays["har_beta_dow_season"] = incs
    meta = dict(meta)
    meta["dow_anchor"] = {
        "source": "P4-dow-anchor-result",
        "columns": "window fraction on Mon..Sat (Sunday reference), true UTC calendar "
                   "(noctua/calendar.py)",
        "fit": "increment on the residual of the shipped anchor (and, separately, of the "
               "clock-aware anchor) over the artifact's training split; har_beta unchanged",
        "coefs": [float(c) for c in inc0[1:]],
        "coefs_on_season": [float(c) for c in incs[1:]],
        "staleness": "frozen coefficients kept 0.82 of a yearly refit's Brier gain over "
                     "2023-2026 (eval/dow_staleness.py)"}
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(a.artifact, **arrays)
    print(f"wrote {a.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
