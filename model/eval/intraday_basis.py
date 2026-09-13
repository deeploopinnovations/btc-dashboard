"""
eval/intraday_basis.py
=====================================================================
P2-intraday-basis: can a richer hour-of-day basis capture the intraday error
structure the single Fourier harmonic cannot represent?

WHY THIS EXISTS, MEASURED RATHER THAN ASSUMED

`P2-event-footprint` found that the worst-5% episodes at H=1 cluster by UTC
anchor hour with chi-square 324.7, p = 4.5e-55: lift 1.788 at 14:00 (which
contains the 14:30 US equity open), 1.533 and 1.465 at 18:00-19:00 (bracketing
the 19:00 FOMC statement), 1.201 at 13:00 (the 13:30 CPI print), against a
MINIMUM of 0.440 at 11:00.

The model's only hour-of-day inputs are `cal_hour_sin` and `cal_hour_cos` --
ONE harmonic, one maximum per 24 hours. It cannot produce a sharp peak at 14:00,
a trough three hours earlier, AND a second shoulder at 18-19:00. Spectrally,
harmonic 1 carries 44.1% of the observed power; harmonics 2-6 carry 43.7%.

THE ARMS

    base   unchanged: cal_hour_sin, cal_hour_cos
    B1     harmonics 1-3: sin/cos at periods 24, 12 and 8 hours (6 columns)
    B2     23 hour-of-day dummies (hour 0 is the reference level)
    shuf   THE CONTROL. B2's dummies built from a PERMUTED hour label.

WHY THE CONTROL IS NOT OPTIONAL

This treatment cannot leak -- a calendar column is known arbitrarily far in
advance, so there is no look-ahead question to answer. But it DOES add capacity:
B2 adds 23 columns to a model with 39 inputs. A capacity gain would look exactly
like a clock gain in the primary. `shuf` permutes the hour label across episodes
within each fold, destroying its alignment while preserving the column count and
the marginal distribution exactly. If `shuf` improves on `base`, the gain is
capacity and both real arms fall.

THE FOLD-LEVEL COMPARISON IS A NON-MEASUREMENT AND IS LABELLED SO BEFORE IT RUNS

Measured from the OOF artifact: the fold-level MDE at 80% power is 18.89% of
base at H=1 and 19.17% at H=6. No plausible calendar-basis effect is that large.
Only the paired per-episode estimator can resolve this, so it is the primary and
the fold spread is printed as a diagnostic with the label attached.

    python -m model.eval.intraday_basis --self-test
    python -m model.eval.intraday_basis
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.vol_matrix import (UNDEFINED_AT_1W, block_len_for, build_h4_table,  # noqa: E402
                             qlike_vec)
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.spec import SHAPE_COLS                                           # noqa: E402
from noctua.train import prepare, train_model                                # noqa: E402

HOUR_COLS = ("cal_hour_sin", "cal_hour_cos")
HORIZONS = (1, 6, 24, 168)
N_FAMILY = 8                       # 2 real arms x 4 horizons, fixed a priori
ARMS = ("base", "B1", "B2", "shuf")
# stated before the run, from the OOF artifact
FOLD_MDE_PCT = {1: 18.89, 6: 19.17}


def basis_cols(arm: str) -> list[str]:
    """The columns `build_basis` ADDS for `arm`, in the order it adds them."""
    if arm == "base":
        return []
    if arm == "B1":
        return [f"cal_h{k}_{t}" for k in (1, 2, 3) for t in ("sin", "cos")]
    return [f"cal_hr_{k:02d}" for k in range(1, 24)]


def shape_cols_for(arm: str) -> list[str]:
    """The stage-B column list `prepare` must be given for `arm`.

    NOT optional bookkeeping. `prepare` slices `X.loc[mask, shape_cols]` with
    the SHIPPED SHAPE_COLS unless told otherwise, so a frame whose hour basis
    has been swapped raises KeyError on the two columns that are gone -- which
    is exactly how the first run of this experiment died. It also decides what
    the WIDE block sees: anything in SHAPE_COLS but not in `shape_cols` is
    dropped from Xa as well, so passing the list keeps the new basis in both
    blocks and the old basis out of both.
    """
    if arm == "base":
        return list(SHAPE_COLS)
    return [c for c in SHAPE_COLS if c not in HOUR_COLS] + basis_cols(arm)


def build_basis(X: pd.DataFrame, hour: np.ndarray, arm: str,
                rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Return X with the hour-of-day basis replaced according to `arm`.

    `base` is returned untouched so the control arm is the shipped feature set
    exactly, not a re-derivation of it.
    """
    if arm == "base":
        return X
    Z = X.drop(columns=["cal_hour_sin", "cal_hour_cos"])
    h = hour if arm != "shuf" else rng.permutation(hour)
    if arm == "B1":
        for k in (1, 2, 3):
            Z[f"cal_h{k}_sin"] = np.sin(2 * np.pi * k * h / 24.0)
            Z[f"cal_h{k}_cos"] = np.cos(2 * np.pi * k * h / 24.0)
        return Z
    # B2 and shuf: 23 dummies, hour 0 is the reference level
    for k in range(1, 24):
        Z[f"cal_hr_{k:02d}"] = (h == k).astype(np.float64)
    return Z


def self_test() -> int:
    """The basis constructions must be what they claim."""
    n = 480
    hour = np.arange(n) % 24
    X = pd.DataFrame({"har_1d": np.zeros(n), "cal_hour_sin": np.sin(2*np.pi*hour/24),
                      "cal_hour_cos": np.cos(2*np.pi*hour/24)})
    rng = np.random.default_rng(0)
    ok = []
    b = build_basis(X, hour, "base")
    ok.append(("base-untouched", b is X, "base returns the frame unchanged"))
    b1 = build_basis(X, hour, "B1")
    ok.append(("B1-width", b1.shape[1] == X.shape[1] - 2 + 6,
               f"{b1.shape[1]} cols = {X.shape[1]}-2+6"))
    # harmonic 2 must actually be period 12, i.e. equal at h and h+12
    v = b1["cal_h2_sin"].to_numpy()
    ok.append(("B1-period-12", np.allclose(v[:12], v[12:24], atol=1e-9),
               "harmonic 2 repeats every 12 hours"))
    b2 = build_basis(X, hour, "B2")
    ok.append(("B2-width", b2.shape[1] == X.shape[1] - 2 + 23,
               f"{b2.shape[1]} cols = {X.shape[1]}-2+23"))
    d = b2[[c for c in b2.columns if c.startswith("cal_hr_")]].to_numpy()
    ok.append(("B2-onehot", bool(np.all(d.sum(1) <= 1)) and int(d.sum()) == n - n // 24,
               "at most one dummy hot; hour 0 is the reference"))
    s = build_basis(X, hour, "shuf", rng)
    hot = s[[c for c in s.columns if c.startswith("cal_hr_")]].to_numpy()
    same = np.array_equal(hot, d)
    ok.append(("shuf-differs", not same, "the shuffled control is not the real basis"))
    ok.append(("shuf-same-width", s.shape[1] == b2.shape[1],
               "the control has identical column count, so capacity is matched"))

    # THE PATH THAT ACTUALLY BROKE. The seven checks above all passed on the
    # first run and the experiment still died in `prepare`, because the frame
    # they were built on was three synthetic columns that never reached it.
    # A construction check is not an integration check, so run the real thing.
    cols = sorted(set(SHAPE_COLS) | set(BASE_COLS) | {"har_1d"})
    XF = pd.DataFrame({c: rng.normal(size=n) for c in cols})
    XF["cal_hour_sin"], XF["cal_hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    epF = pd.DataFrame({"H": np.full(n, 24.0), "RV": np.full(n, 0.02),
                        "R": rng.normal(size=n) * 0.02,
                        "M_up": np.abs(rng.normal(size=n)) * 0.02,
                        "M_dn": -np.abs(rng.normal(size=n)) * 0.02})
    mask = np.ones(n, bool)
    for arm in ARMS:
        want = len(SHAPE_COLS) + (0 if arm == "base" else len(basis_cols(arm)) - 2)
        try:
            Z = build_basis(XF, hour, arm, rng)
            tr, _ = prepare(epF, Z, mask, shape_cols=shape_cols_for(arm))
            got, wide = tr["Xs"].shape[1], tr["cols"]["all"]
            good = got == want and not (set(HOUR_COLS) & set(wide) and arm != "base")
            msg = f"stage-B width {got} (want {want}); old basis out of the wide block"
        except Exception as exc:                                    # noqa: BLE001
            good, msg = False, f"{type(exc).__name__}: {exc}"
        ok.append((f"prepare-{arm}", good, msg))

    print("intraday_basis self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def run_arm(ep, X, fold, H, arm, hour, hidden, seeds, rng):
    at_h = (ep.H == H).to_numpy()
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xb = X[cols40]
    Xu = build_basis(Xb, hour, arm, rng)
    fin = np.isfinite(Xu.to_numpy(np.float64)).all(1)
    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    m_tr = fold["train"] & fin & at_h
    m_va = fold["calib"] & fin & at_h
    m_te = fold["test"] & fin & at_h
    if m_tr.sum() < 2000 or m_va.sum() < 300 or m_te.sum() < 100:
        return None
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)
    lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
    sref = np.maximum(np.clip(raw, lo, hi), 1e-12)
    sc = shape_cols_for(arm)
    tr, stds = prepare(ep, Xu, m_tr, shape_cols=sc, sigma_ref=sref[m_tr])
    w = S.sample_weights(ep, m_tr)
    va, _ = prepare(ep, Xu, m_va, *stds, shape_cols=sc, sigma_ref=sref[m_va])
    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), w)
    bl = B.fit_vol_baselines(Xu[m_tr], yall[m_tr], w)
    models = [train_model(tr, w, va, hidden=hidden, epochs=40, seed=k,
                          verbose=False, ols_beta=ols.beta)[0] for k in range(seeds)]
    d, _ = prepare(ep, Xu, m_te, *stds, shape_cols=sc)
    lp = bl["log_har_cal"].predict(Xu[m_te])
    preds = [I.predict(m, d, har_logvol=lp) for m in models]
    sg = np.mean([p["sigma_med"] for p in preds], axis=0)
    return {"rv": ep.RV.to_numpy()[m_te], "sigma": np.asarray(sg, np.float64)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-intraday-basis")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/intraday_basis.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    hour = ((ep.anchor_ts.to_numpy(np.int64) // 3600) % 24).astype(int)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    print(f"P2-intraday-basis. family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("fold-level MDE at 80% power: "
          + ", ".join(f"H={k} {v:.2f}% of base" for k, v in FOLD_MDE_PCT.items())
          + "  -> the fold comparison is NOT POWERED and is a diagnostic only\n")

    out = {"family_size": N_FAMILY, "alpha": alpha, "fold_mde_pct": FOLD_MDE_PCT,
           "horizons": {}}
    for H in a.horizons:
        acc = {k: [] for k in ARMS}
        for f in folds:
            t0 = time.time()
            rng = np.random.default_rng(1000 + f["year"])
            for arm in ARMS:
                r = run_arm(ep, X, f, H, arm, hour, a.hidden, a.seeds, rng)
                if r is not None:
                    acc[arm].append(r)
            print(f"  H={H:>4} fold {f['year']}  ({time.time()-t0:.0f}s)", flush=True)
        if not acc["base"]:
            continue
        rv = np.concatenate([x["rv"] for x in acc["base"]])
        qb = qlike_vec(rv, np.concatenate([x["sigma"] for x in acc["base"]]))
        L = block_len_for(H, len(qb))
        print("\n" + "=" * 88)
        print(f"H = {H}h   {len(qb):,} test episodes   blocks of {L}")
        print("=" * 88)
        print(f"{'arm':>6} {'QLIKE':>9} {'vs base':>10} {'rel %':>8}   paired CI vs base")
        row = {}
        for arm in ARMS:
            if not acc[arm]:
                continue
            q = qlike_vec(rv, np.concatenate([x["sigma"] for x in acc[arm]]))
            if arm == "base":
                print(f"{arm:>6} {np.nanmean(q):9.5f} {'—':>10} {'—':>8}   (reference)")
                row[arm] = {"qlike": float(np.nanmean(q))}
                continue
            d_ = qb - q
            g = np.isfinite(d_)
            ci = mean_ci(d_[g], alpha=alpha, block_len=L)
            rel = 100 * np.nanmean(d_) / np.nanmean(qb)
            print(f"{arm:>6} {np.nanmean(q):9.5f} {np.nanmean(d_):+10.5f} {rel:+8.2f}   "
                  f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
                  + ("  <- CONTROL, must not clear" if arm == "shuf" else ""))
            row[arm] = {"qlike": float(np.nanmean(q)),
                        "vs_base": float(np.nanmean(d_)), "rel_pct": float(rel),
                        "ci": list(ci["ci95"]),
                        "clears": bool(ci["ci95"][0] > 0)}
        ctrl = row.get("shuf", {}).get("clears", False)
        msg = ("CLEARS -- the gain is CAPACITY, both arms fall" if ctrl
               else "does not clear, as required")
        print(f"\n   control: shuffled-hour {msg}")
        for arm in ("B1", "B2"):
            if arm in row:
                v = "CLEARS" if row[arm]["clears"] and not ctrl else "does not clear"
                print(f"   {arm}: {v}")
        row["control_invalidates"] = bool(ctrl)
        out["horizons"][str(H)] = row
        print()

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
