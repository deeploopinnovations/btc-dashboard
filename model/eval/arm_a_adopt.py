"""
eval/arm_a_adopt.py
=====================================================================
P2-armA-adopt: the barrier battery for Arm A's one surviving cell.

WHY THIS EXISTS

`P2-armA-result` closed with a gate, not a conclusion: "BARRIERS ARE NOT YET
SCORED for Arm A. P2-scale-v2 established that a QLIKE gain on this model can
coexist with across-the-board barrier degradation, so no Arm A adoption claim
is available until the same barrier battery runs." This is that run, and the
battery is P2-scale-v2's unchanged -- a battery redesigned after seeing which
arm it will judge is not a battery.

SCOPE IS ONE HORIZON, ON PURPOSE

H=6 is the only cell of Arm A's twelve that beat its calib-best teacher BOTH
raw and rescaled. H=24 beat raw har_short (0.25835) and lost to har_short with
its own calib-fitted scalar (0.25696), so under R39 it is not a win over a
properly equipped teacher. H=1 and H=168 did not clear at all. Running the
battery on cells that already failed the QLIKE primary would be shopping.

THE ARCHITECTURE RUNS THROUGH run_fold, NOT BESIDE IT

`benchmark.run_fold(residual_anchor=...)` fits the network AND the blend anchor
on `y - anchor`, then adds the anchor back through the existing `post_shift_fn`
hook -- which places it in front of Stage B, the committee and every barrier.
That routing is the whole point. `eval/arm_a_residual.py` scored QLIKE in its
own loop and structurally could not say what the decomposition does to a touch
probability. run_fold's assertion that the achieved log-level shift equals the
requested one to 1e-6 is retained and is a pass condition.

THE ANCHOR IS REFITTED HERE, AND WHY THAT IS NOT A SHORTCUT

teacher_oof.npz is built on `episodes_h4` (horizons 1/6/24/168). run_fold
trains on the MAIN episodes table (6/12/19/24) and calibrates its committee on
H=19, so the artifact cannot supply values for the episodes this pipeline
needs. har_short is therefore refitted under the SAME cross-fitting rule:

    train slice   `teacher_zoo.inner_oof` expanding-forward, Amendment 1.
                  Block 1 gets no prediction and its episodes are DROPPED, not
                  filled -- a fallback would be a number the model reads as
                  information (R43).
    calib, test   one fit on that fold's TRAIN slice only, predicted forward.

`assert_slices_disjoint` and `assert_causal_boundary` run per fold. The
refitted anchor is then VALIDATED against teacher_oof.npz on the H=6 test
slice: within 2% relative QLIKE and Pearson r >= 0.98 in the log-vol domain
(P2-armA-adopt-a1 -- the two tables give har_short different coefficients, so
an exact identity check was unsatisfiable by construction).

THE CONTROL IS SAMPLE-MATCHED

A0 is the shipped architecture on the identical slice, run with `train_filter`
set to the SAME finite-anchor mask. Inner block 1 costs A1 roughly a fifth of
its training episodes; handing those to A0 alone would make the residual arm
look worse for a reason that has nothing to do with the residual.

    python -m model.eval.arm_a_adopt --self-test
    python -m model.eval.arm_a_adopt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                          # noqa: E402
from eval.direction import mean_ci                                           # noqa: E402
from eval.scale_adopt import barrier_cols, optimal_c                         # noqa: E402
from eval.teacher_zoo import (assert_causal_boundary,                        # noqa: E402
                              assert_slices_disjoint, inner_oof)
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402
from noctua import baselines as B                                            # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402

PROD_H = 6                        # the one cell that survived the QLIKE primary
TEACHER = "har_short"
N_FAMILY = 1                      # one arm, one horizon, fixed before results
RATIO_LO, RATIO_HI = 0.95, 1.05
SLICE_TOL = 0.02
VAL_QLIKE_TOL = 0.02              # anchor validation, P2-armA-adopt-a1
VAL_CORR_MIN = 0.98


def to_logvol(sigma, H):
    return np.log(np.maximum(sigma, 1e-12)) - 0.5 * np.log(np.maximum(H, 1))


def build_anchor(ep, X, fold, teacher=TEACHER):
    """The cross-fitted teacher anchor for one fold, in the log-vol rate domain.

    NaN wherever no out-of-fold value exists. Callers must drop those episodes;
    `run_fold` does.
    """
    assert_slices_disjoint(fold)
    assert_causal_boundary(ep, fold)
    n = len(ep)
    Hall = ep["H"].to_numpy(np.float64)
    yall = B.har_target(ep["RV"].to_numpy(), Hall)
    fin = np.isfinite(X.to_numpy(np.float64)).all(1)
    anch = np.full(n, np.nan)

    # TRAIN: expanding-forward inner cross-fit. `H=None` pools every horizon,
    # which is what this pipeline trains on; slicing per horizon and stitching
    # would give each horizon different inner block edges and would therefore
    # not be one estimator.
    inner = inner_oof(ep, X, fold, None, None)
    t_in = inner.get(teacher)
    if t_in is None:
        return None
    m_tr = fold["train"] & fin
    anch[m_tr] = to_logvol(t_in[m_tr], Hall[m_tr])

    # CALIB and TEST: one fit on this fold's train slice, predicted forward.
    m_fit = fold["train"] & fin
    bl = B.fit_vol_baselines(X[m_fit], yall[m_fit], S.sample_weights(ep, m_fit))
    m_out = (fold["calib"] | fold["test"]) & fin
    anch[m_out] = np.asarray(bl[teacher].predict(X[m_out]), np.float64)
    return anch


def validate_anchor(ep, X, folds, anchors, artifacts):
    """The refitted anchor must be the same teacher in substance (a1)."""
    path = Path(artifacts) / "teacher_oof.npz"
    if not path.exists():
        return {"ran": False, "reason": f"{path} absent"}
    z = np.load(path, allow_pickle=False)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    at_h = (ep["H"].to_numpy() == PROD_H)
    mine, theirs, rv = [], [], []
    for f, anch in zip(folds, anchors):
        if anch is None:
            continue
        y = f["year"]
        ka, ks = f"{y}/{PROD_H}/test/anchor_ts", f"{y}/{PROD_H}/test/sigma/{TEACHER}"
        if ka not in z or ks not in z:
            continue
        want = dict(zip(z[ka].tolist(), z[ks].tolist()))
        m = fold_test_mask(f, at_h) & np.isfinite(anch)
        idx = np.flatnonzero(m)
        hit = np.array([ts[i] in want for i in idx])
        if not hit.any():
            continue
        idx = idx[hit]
        mine.append(anch[idx])
        theirs.append(to_logvol(np.array([want[ts[i]] for i in idx]),
                                ep["H"].to_numpy(np.float64)[idx]))
        rv.append(ep["RV"].to_numpy()[idx])
    if not mine:
        return {"ran": False, "reason": "no overlapping test episodes"}
    a, b = np.concatenate(mine), np.concatenate(theirs)
    r = np.concatenate(rv)
    sq = np.sqrt(float(PROD_H))
    qa = float(np.nanmean(qlike_vec(r, np.exp(a) * sq)))
    qb = float(np.nanmean(qlike_vec(r, np.exp(b) * sq)))
    corr = float(np.corrcoef(a, b)[0, 1])
    rel = (qa - qb) / abs(qb)
    return {"ran": True, "n": int(len(a)), "qlike_refit": qa,
            "qlike_artifact": qb, "rel": float(rel), "corr": corr,
            "pass": bool(abs(rel) <= VAL_QLIKE_TOL and corr >= VAL_CORR_MIN)}


def fold_test_mask(fold, at_h):
    return fold["test"] & at_h


def self_test() -> int:
    """The pieces that can be checked without a six-fold training run."""
    import pandas as pd
    ok = []
    n = 600
    H = np.full(n, 6.0)
    sig = np.exp(np.random.default_rng(0).normal(-4, 0.5, n))
    back = np.exp(to_logvol(sig, H)) * np.sqrt(H)
    ok.append(("logvol-roundtrip", float(np.max(np.abs(back - sig))) < 1e-12,
               "sigma -> log-vol rate -> sigma is the identity"))
    # a residual anchor of exactly the teacher must leave a zero target
    yall = np.log(np.full(n, 0.02)) - 0.5 * np.log(H)
    anch = yall.copy()
    ok.append(("zero-residual", float(np.max(np.abs(yall - anch))) == 0.0,
               "y - anchor is exactly zero when the anchor is the target"))
    # run_fold must REFUSE both corrections at once
    from eval.benchmark import run_fold as rf
    ep = pd.DataFrame({"H": H, "RV": np.full(n, 0.02),
                       "anchor_ts": np.arange(n, dtype=np.int64) * 3600})
    fold = {"year": 2099, "train": np.zeros(n, bool), "calib": np.zeros(n, bool),
            "test": np.zeros(n, bool)}
    try:
        rf(ep, pd.DataFrame({"har_1d": np.zeros(n)}), fold,
           residual_anchor=np.zeros(n), post_shift_fn=lambda m, t: 0.0)
        ok.append(("refuse-double-shift", False, "did NOT refuse"))
    except ValueError as e:
        ok.append(("refuse-double-shift", "both write" in str(e), str(e)[:60]))
    # and must refuse a wrongly shaped anchor
    try:
        rf(ep, pd.DataFrame({"har_1d": np.zeros(n)}), fold,
           residual_anchor=np.zeros(n - 1))
        ok.append(("refuse-bad-shape", False, "did NOT refuse"))
    except ValueError as e:
        ok.append(("refuse-bad-shape", "expected one value" in str(e), str(e)[:60]))
    print("arm_a_adopt self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-armA-adopt")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/arm_a_adopt.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    at_h = Hall == PROD_H
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(train_mask):
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    alpha = 0.05 / N_FAMILY
    print(f"P2-armA-adopt  teacher={TEACHER}  slice H={PROD_H} "
          f"({int(at_h.sum()):,} episodes)  seeds={a.seeds}")
    print(f"family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals\n")

    anchors = [build_anchor(ep, X, f) for f in folds]
    val = validate_anchor(ep, X, folds, anchors, a.artifacts)
    if val.get("ran"):
        print(f"  anchor validation vs teacher_oof.npz on {val['n']:,} shared "
              f"H={PROD_H} test episodes:")
        print(f"    QLIKE refit {val['qlike_refit']:.5f} vs artifact "
              f"{val['qlike_artifact']:.5f}  ({100*val['rel']:+.2f}%, "
              f"tol +-{100*VAL_QLIKE_TOL:.0f}%)")
        print(f"    log-vol correlation r = {val['corr']:.4f} "
              f"(min {VAL_CORR_MIN})")
        print(f"    [{'PASS' if val['pass'] else 'FAIL'}]\n")
        if not val["pass"]:
            print("  REFUSING to score: the refitted anchor is not the teacher "
                  "P2-armA-result measured, so this run would be a different "
                  "experiment wearing its name.")
            return 1
    else:
        print(f"  anchor validation did NOT run: {val.get('reason')}\n")

    acc = []
    for f, anch in zip(folds, anchors):
        if anch is None:
            print(f"  fold {f['year']}: no anchor"); continue
        t0 = time.time()
        have = np.isfinite(anch)
        # A0 is sample-matched to A1 by construction, not by hope.
        r0 = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn,
                      prod_override=at_h, train_filter=have)
        r1 = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn,
                      prod_override=at_h, residual_anchor=anch)
        if r0 is None or r1 is None:
            print(f"  fold {f['year']}: skipped"); continue
        p0, p1 = r0["per_episode"], r1["per_episode"]
        if not np.array_equal(p0["test_idx"], p1["test_idx"]):
            raise SystemExit(
                f"REFUSING: fold {f['year']} scored A0 and A1 on different "
                f"episodes ({len(p0['test_idx']):,} vs {len(p1['test_idx']):,}). "
                f"The comparison would not be paired.")
        idx = p1["test_idx"]
        acc.append({
            "year": f["year"], "rv": p1["rv"],
            "s0": p0["sigma_med"], "s1": p1["sigma_med"],
            "s_teacher": np.exp(anch[idx]) * np.sqrt(Hall[idx]),
            "rv_cal": p1["rv_cal"], "sig_cal": p1["sigma_cal"],
            "anch_cal_rv": None,
            "vol0": r0["vol"], "vol1": r1["vol"],
            "bar0": barrier_cols(r0["rows"]), "bar1": barrier_cols(r1["rows"]),
            "chr0": r0["christoffersen"], "chr1": r1["christoffersen"],
        })
        print(f"  fold {f['year']}  n={len(idx):,}  ({time.time()-t0:.0f}s)",
              flush=True)

    if not acc:
        print("no usable folds"); return 1

    rv = np.concatenate([r["rv"] for r in acc])
    s0 = np.concatenate([r["s0"] for r in acc])
    s1 = np.concatenate([r["s1"] for r in acc])
    st = np.concatenate([r["s_teacher"] for r in acc])
    q0, q1, qt = qlike_vec(rv, s0), qlike_vec(rv, s1), qlike_vec(rv, st)
    hi = rv >= np.quantile(rv, 0.95)
    L = block_len_for(PROD_H, len(q0))
    d = q0 - q1
    ci = mean_ci(d[np.isfinite(d)], alpha=alpha, block_len=L)
    ratio0 = float(np.nanmean(rv ** 2 / np.maximum(s0, 1e-12) ** 2))
    ratio1 = float(np.nanmean(rv ** 2 / np.maximum(s1, 1e-12) ** 2))

    print("\n" + "=" * 92)
    print(f"H={PROD_H} PRODUCTION-PIPELINE SLICE   {len(q0):,} test episodes   "
          f"blocks of {L}")
    print("=" * 92)
    print(f"  QLIKE  A0 (shipped)      {np.nanmean(q0):.5f}")
    print(f"  QLIKE  A1 (residual)     {np.nanmean(q1):.5f}   "
          f"gain {np.nanmean(d):+.5f}  "
          f"CI [{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]")
    print(f"  QLIKE  teacher raw       {np.nanmean(qt):.5f}")

    # R39: the teacher gets its own calib-fitted scalar before being beaten.
    cs, qtc_parts, rvp = [], [], []
    for r in acc:
        c = optimal_c(r["rv"], r["s_teacher"])          # per fold, per R44
        cs.append(c)
        qtc_parts.append(qlike_vec(r["rv"], c * r["s_teacher"]))
        rvp.append(r["rv"])
    qtc = np.concatenate(qtc_parts)
    dt = qtc - q1
    cit = mean_ci(dt[np.isfinite(dt)], alpha=alpha, block_len=L)
    print(f"  QLIKE  teacher rescaled  {np.nanmean(qtc):.5f}   "
          f"(c per fold {', '.join(f'{c:.3f}' for c in cs)})")
    print(f"         A1 vs rescaled teacher {np.nanmean(dt):+.5f}  "
          f"CI [{cit['ci95'][0]:+.5f}, {cit['ci95'][1]:+.5f}]")
    print(f"  QLIKE spike  {np.nanmean(q0[hi]):.5f} -> {np.nanmean(q1[hi]):.5f}")
    print(f"  QLIKE calm   {np.nanmean(q0[~hi]):.5f} -> {np.nanmean(q1[~hi]):.5f}")
    print(f"  calib ratio  {ratio0:.4f} -> {ratio1:.4f}   "
          f"guard [{RATIO_LO}, {RATIO_HI}]")

    print("\n  --- THE BARRIER BATTERY, verbatim from P2-scale-v2 ---")
    bar = {}
    keys = sorted(set().union(*[set(r["bar0"]) for r in acc]))
    for k in keys:
        b = float(np.mean([r["bar0"][k] for r in acc if k in r["bar0"]]))
        aft = float(np.mean([r["bar1"][k] for r in acc if k in r["bar1"]]))
        better = (aft > b) if k == "DSC" else (aft < b)
        bar[k] = {"a0": b, "a1": aft, "better": bool(better)}
        print(f"  {k:>8}  {b:.6f} -> {aft:.6f}   "
              f"{'BETTER' if better else 'WORSE '}  ({100*(aft-b)/abs(b):+.2f}%)")

    guards = {
        "ratio_in_band": bool(RATIO_LO <= ratio1 <= RATIO_HI),
        "spike_not_worse_2pct":
            bool(np.nanmean(q1[hi]) <= np.nanmean(q0[hi]) * (1 + SLICE_TOL)),
        "calm_not_worse_2pct":
            bool(np.nanmean(q1[~hi]) <= np.nanmean(q0[~hi]) * (1 + SLICE_TOL)),
        "dsc_not_worse": bool(bar.get("DSC", {}).get("better", False)),
        "brier_not_worse": bool(bar.get("brier", {}).get("better", True)),
    }
    print("\n  --- pre-registered guards ---")
    for k, v in guards.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    primary = bool(ci["ci95"][0] > 0)
    secondary = bool(cit["ci95"][0] > 0)
    print(f"  [{'PASS' if primary else 'FAIL'}] primary: A1 beats A0, CI excludes zero")
    print(f"  [{'PASS' if secondary else 'FAIL'}] secondary: A1 beats the RESCALED teacher")
    verdict = ("ADOPTABLE" if primary and secondary and all(guards.values())
               else "NOT ADOPTABLE")
    print(f"\n  VERDICT: {verdict}")

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha, "block_len": L,
        "horizon": PROD_H, "teacher": TEACHER, "n_test": int(len(q0)),
        "anchor_validation": val,
        "qlike": {"a0": float(np.nanmean(q0)), "a1": float(np.nanmean(q1)),
                  "teacher_raw": float(np.nanmean(qt)),
                  "teacher_rescaled": float(np.nanmean(qtc)),
                  "c_per_fold": [float(c) for c in cs],
                  "gain_vs_a0": float(np.nanmean(d)), "ci_vs_a0": list(ci["ci95"]),
                  "gain_vs_teacher_rescaled": float(np.nanmean(dt)),
                  "ci_vs_teacher_rescaled": list(cit["ci95"]),
                  "spike_a0": float(np.nanmean(q0[hi])),
                  "spike_a1": float(np.nanmean(q1[hi])),
                  "calm_a0": float(np.nanmean(q0[~hi])),
                  "calm_a1": float(np.nanmean(q1[~hi]))},
        "calib_ratio": {"a0": ratio0, "a1": ratio1},
        "barriers": bar, "guards": guards,
        "primary_clears": primary, "secondary_clears": secondary,
        "verdict": verdict,
        "folds": [r["year"] for r in acc],
    }, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
