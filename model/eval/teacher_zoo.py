"""
eval/teacher_zoo.py
=====================================================================
Cross-fitted out-of-fold predictions for every teacher in the zoo.

Governed by `research/TEACHER_ZOO.md`, frozen before this file was written.

THE ONE RULE THIS FILE EXISTS TO ENFORCE

    An episode may never receive a teacher prediction from a teacher that was
    trained on that episode, on anything whose forward window overlaps it, or
    on anything after it.

A residual learner or a stacker trained on IN-SAMPLE teacher forecasts is
learning the teacher's TRAINING error. That error is smaller than its
forecasting error and differently shaped -- it is small exactly where the
teacher overfit, which is exactly where the student will be asked to correct
it. The resulting model looks excellent in development and fails in
production, and it fails in a way that no amount of held-out scoring of the
STUDENT will reveal, because the corruption is in the student's inputs.

So this file emits predictions for `calib` and `test` only. **Train-slice
predictions are not produced at all.** There is no legitimate Phase 2 use for
them, and a value that does not exist cannot be reached for by a later caller
in a hurry.

WHAT `calib` AND `test` ARE FOR, AND THE ASYMMETRY BETWEEN THEM

    calib   the ONLY teacher values any Phase 2 learner may train or select on
    test    scoring ONLY -- never a fit, never a weight, never a gate, never a
            hyperparameter, and never a decision about whether to include an arm

The asymmetry is not decoration. Phase 1 measured that at H = 1 and H = 6 the
best teacher ON TEST is `garch_t`, while the calibration slice chooses
`persistence` and `har_short`. Selecting teachers on test would import that
disagreement straight into every downstream arm, and the arm would then be
scored on the same test data that chose it.

THE GUARDS, AND THAT THEY CAN FAIL

`--self-test` corrupts the fold masks on purpose and asserts that each refusal
fires. A guard that has never been shown to fail is not a guard (R2); five
guards in this project printed reassuring output while being incapable of
returning the other answer.

    python -m model.eval.teacher_zoo --self-test
    python -m model.eval.teacher_zoo                 # write the OOF artifact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.vol_matrix import UNDEFINED_AT_1W, build_h4_table, qlike_vec        # noqa: E402
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import prepare, train_model                                # noqa: E402

HORIZONS = (1, 6, 24, 168)
EMITTED_SLICES = ("calib", "test")          # deliberately NOT "train"
OLS_TEACHERS = ("har_short", "log_har", "log_har_cal")


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------
class LeakageRefusal(RuntimeError):
    """Raised instead of returning a prediction that would violate §2."""


def assert_slices_disjoint(fold: dict) -> None:
    """train / calib / test must not intersect.

    `walk_forward_folds` builds them disjoint by construction, which is exactly
    why this is worth asserting: the check costs nothing and the day someone
    passes a hand-built fold dict is the day it earns its place.
    """
    for a, b in (("train", "calib"), ("train", "test"), ("calib", "test")):
        n = int((fold[a] & fold[b]).sum())
        if n:
            raise LeakageRefusal(
                f"REFUSING: fold {fold.get('year')} has {n:,} episodes in BOTH "
                f"`{a}` and `{b}`. A teacher fitted on `train` would be emitting "
                f"in-sample predictions into `{b}`, and every downstream learner "
                f"would inherit them.")


def assert_no_train_emission(slice_name: str) -> None:
    if slice_name not in EMITTED_SLICES:
        raise LeakageRefusal(
            f"REFUSING to emit teacher predictions for slice {slice_name!r}. "
            f"Only {EMITTED_SLICES} are produced. Train-slice teacher values are "
            f"in-sample by construction; TEACHER_ZOO section 2 forbids their use, "
            f"so they are not generated at all rather than generated and trusted "
            f"not to be used.")


def assert_causal_boundary(ep: pd.DataFrame, fold: dict) -> None:
    """Every emitted episode must START after every training episode ENDS.

    This is the temporal form of the rule, and it is stronger than mask
    disjointness: two masks can be disjoint while a training episode's 168-hour
    forward window still runs past the anchor of a test episode.
    """
    ts = ep["anchor_ts"].to_numpy(np.int64)
    end = ts + ep["H"].to_numpy(np.int64) * 3600
    tr_end = end[fold["train"]]
    if tr_end.size == 0:
        return
    latest_train_end = int(tr_end.max())
    for name in EMITTED_SLICES:
        m = fold[name]
        if not m.any():
            continue
        earliest = int(ts[m].min())
        if earliest < latest_train_end:
            raise LeakageRefusal(
                f"REFUSING: in fold {fold.get('year')} the earliest `{name}` anchor "
                f"({earliest}) starts {(latest_train_end - earliest)/3600:.1f}h BEFORE "
                f"the last training episode's forward window ends ({latest_train_end}). "
                f"The teacher would have been fitted on an outcome that overlaps the "
                f"episode it is predicting.")


# --------------------------------------------------------------------------
def _fit_predict_fold(ep, X, fold, ret, hidden, seeds, verbose=False):
    """Fit every teacher on this fold's TRAIN slice, predict calib and test."""
    assert_slices_disjoint(fold)
    assert_causal_boundary(ep, fold)

    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    fin40 = np.isfinite(X[cols40].to_numpy(np.float64)).all(1)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    out = {}
    for H in HORIZONS:
        at_h = (ep.H == H).to_numpy()
        m_tr = fold["train"] & fin40 & at_h
        if m_tr.sum() < 2000:
            continue
        w_tr = S.sample_weights(ep, m_tr)
        # PER-HORIZON fits. Phase 1 established that a baseline fitted on a
        # pooled multi-horizon sample is a straw man at the extremes (R39).
        bl = B.fit_vol_baselines(X[m_tr], yall[m_tr], w_tr)

        for sl in EMITTED_SLICES:
            assert_no_train_emission(sl)
            m = fold[sl] & fin40 & at_h
            if m.sum() < 30:
                continue
            sq = np.sqrt(Hall[m])
            preds = {
                "persistence": np.maximum(
                    np.exp(X.loc[m, "har_1d"].to_numpy()) * sq, 1e-12),
            }
            for k in OLS_TEACHERS:
                preds[k] = np.exp(bl[k].predict(X[m])) * sq
            if ret is not None:
                from eval.garch import fit_and_forecast
                for dist, nm in (("normal", "garch_normal"), ("t", "garch_t")):
                    preds[nm] = fit_and_forecast(
                        ret, int(fold["train_end_ts"]),
                        ep.anchor_ts.to_numpy(np.int64)[m], Hall[m],
                        dist=dist, verbose=False)
            out.setdefault(H, {})[sl] = {
                "anchor_ts": ep.anchor_ts.to_numpy(np.int64)[m],
                "rv": ep.RV.to_numpy()[m],
                "H": Hall[m],
                "sigma": preds,
                "idx": np.flatnonzero(m),
            }
    return out


def _noctua_fold(ep, X, fold, hidden, seeds, verbose=False):
    """NOCTUA V1 as a teacher: one multi-horizon network per fold, trained on
    TRAIN only, emitting calib and test. Same architecture Phase 1 scored."""
    assert_slices_disjoint(fold)
    Hall = ep.H.to_numpy(np.float64)
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xv = X[cols40]
    fin = np.isfinite(Xv.to_numpy(np.float64)).all(1)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    m_tr, m_va = fold["train"] & fin, fold["calib"] & fin
    if m_tr.sum() < 5000 or m_va.sum() < 500:
        return {}
    lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
    sref = np.maximum(np.clip(raw, lo, hi), 1e-12)
    tr, stds = prepare(ep, Xv, m_tr, sigma_ref=sref[m_tr])
    w_tr = S.sample_weights(ep, m_tr)
    va, _ = prepare(ep, Xv, m_va, *stds, sigma_ref=sref[m_va])
    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), w_tr)
    bl = B.fit_vol_baselines(Xv[m_tr], yall[m_tr], w_tr)
    models = [train_model(tr, w_tr, va, hidden=hidden, epochs=40, seed=s,
                          verbose=verbose, ols_beta=ols.beta)[0]
              for s in range(seeds)]

    out = {}
    for H in HORIZONS:
        at_h = (ep.H == H).to_numpy()
        for sl in EMITTED_SLICES:
            assert_no_train_emission(sl)
            m = fold[sl] & fin & at_h
            if m.sum() < 30:
                continue
            d, _ = prepare(ep, Xv, m, *stds)
            lp = bl["log_har_cal"].predict(Xv[m])
            ps = [I.predict(mm, d, har_logvol=lp) for mm in models]
            out.setdefault(H, {})[sl] = np.mean([p["sigma_med"] for p in ps], axis=0)
    return out


# --------------------------------------------------------------------------
def self_test() -> int:
    """Each refusal must fire on a deliberately corrupted input."""
    n = 400
    ep = pd.DataFrame({
        "anchor_ts": np.arange(n, dtype=np.int64) * 3600 + 1_600_000_000,
        "H": np.full(n, 24, np.int64),
        "RV": np.full(n, 0.02),
    })
    base = {"year": 2099,
            "train": np.zeros(n, bool), "calib": np.zeros(n, bool),
            "test": np.zeros(n, bool)}
    ok = []

    f = {**base}
    f["train"] = np.zeros(n, bool); f["train"][:200] = True
    f["calib"] = np.zeros(n, bool); f["calib"][150:250] = True   # overlaps train
    f["test"] = np.zeros(n, bool);  f["test"][300:] = True
    try:
        assert_slices_disjoint(f); ok.append(("slices-disjoint", False, "did NOT fire"))
    except LeakageRefusal as e:
        ok.append(("slices-disjoint", True, str(e).split(".")[0]))

    try:
        assert_no_train_emission("train"); ok.append(("no-train-emission", False, "did NOT fire"))
    except LeakageRefusal as e:
        ok.append(("no-train-emission", True, str(e).split(".")[0]))

    f2 = {**base}
    f2["train"] = np.zeros(n, bool); f2["train"][:200] = True
    f2["calib"] = np.zeros(n, bool); f2["calib"][205:260] = True  # inside the 24h window
    f2["test"] = np.zeros(n, bool);  f2["test"][300:] = True
    try:
        assert_causal_boundary(ep, f2); ok.append(("causal-boundary", False, "did NOT fire"))
    except LeakageRefusal as e:
        ok.append(("causal-boundary", True, str(e).split(".")[0]))

    # and the NEGATIVE control: a clean fold must NOT raise, or the guards are
    # simply always-on and prove nothing.
    f3 = {**base}
    f3["train"] = np.zeros(n, bool); f3["train"][:200] = True
    f3["calib"] = np.zeros(n, bool); f3["calib"][230:280] = True
    f3["test"] = np.zeros(n, bool);  f3["test"][300:] = True
    try:
        assert_slices_disjoint(f3); assert_causal_boundary(ep, f3)
        ok.append(("clean-fold-passes", True, "no refusal, as required"))
    except LeakageRefusal as e:
        ok.append(("clean-fold-passes", False, f"fired on a CLEAN fold: {e}"))

    print("teacher_zoo self-test")
    for name, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: {msg[:96]}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok) - len(bad)}/{len(ok)} guards behaved correctly")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="cross-fitted teacher predictions")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--no-noctua", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    ret = None
    if not a.no_garch:
        from eval.garch import hourly_returns
        ret = hourly_returns(a.artifacts)
        print(f"hourly returns for GARCH: {len(ret):,} rows")

    folds = S.walk_forward_folds(ep)
    print(f"{len(folds)} folds; emitting slices {EMITTED_SLICES} only "
          f"(train-slice predictions are not produced)\n")

    store, meta = {}, []
    for f in folds:
        t0 = time.time()
        f = dict(f)
        f["train_end_ts"] = int(ep.anchor_ts.to_numpy()[f["train"]].max())
        r = _fit_predict_fold(ep, X, f, ret, a.hidden, a.seeds)
        nn = {} if a.no_noctua else _noctua_fold(ep, X, f, a.hidden, a.seeds)
        for H, per_slice in r.items():
            for sl, d in per_slice.items():
                pre = f"{f['year']}/{H}/{sl}"
                store[f"{pre}/anchor_ts"] = d["anchor_ts"]
                store[f"{pre}/rv"] = d["rv"]
                store[f"{pre}/H"] = d["H"]
                store[f"{pre}/idx"] = d["idx"]
                for k, v in d["sigma"].items():
                    store[f"{pre}/sigma/{k}"] = np.asarray(v, np.float64)
                if H in nn and sl in nn[H]:
                    store[f"{pre}/sigma/noctua_v1"] = np.asarray(nn[H][sl], np.float64)
                meta.append({"year": f["year"], "H": int(H), "slice": sl,
                             "n": int(len(d["anchor_ts"]))})
        print(f"  fold {f['year']}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(a.out, **store)
    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    side = a.out.with_suffix(".json")
    side.write_text(json.dumps({
        "sha256": h, "n_arrays": len(store), "emitted_slices": list(EMITTED_SLICES),
        "horizons": list(HORIZONS), "seeds": a.seeds, "hidden": a.hidden,
        "teachers": sorted({k.rsplit("/", 1)[1] for k in store if "/sigma/" in k}),
        "slices": meta,
    }, indent=1) + "\n")
    print(f"\nwrote {a.out}  sha256 {h[:16]}…")
    print(f"wrote {side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
