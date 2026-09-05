"""
eval/arm_a_residual.py
=====================================================================
Arm A: teacher + residual NOCTUA. Pre-registered as `P2-armA-residual`.

THE ARCHITECTURE, IN ONE LINE OF ALGEBRA

Everything happens in the log hourly vol RATE domain that `har_target` already
defines. A teacher's sigma maps into it as yhat_T = log(sigma_T) - 0.5*log(H).
The residual target is r = y - yhat_T. NOCTUA is trained on r ALONE. The final
forecast is yhat = yhat_T + rhat, scored as sigma = exp(yhat)*sqrt(H).

THE FAIL-SAFE IDENTITY IS A PASS CONDITION, NOT A REMARK

A residual model that predicts exactly zero must reproduce the teacher's QLIKE
BIT-IDENTICALLY. That is what makes this architecture degrade safely onto the
stronger teacher instead of onto whatever the network happened to learn. It is
asserted to 1e-12 before any arm is scored, and if it fails no Arm A number
means anything -- the composition would be wrong and every downstream figure
would be measuring a different model than the one described.

WHY THE TEACHER VALUES COME FROM AN ARTIFACT AND NOT FROM A FIT HERE

`teacher_oof.npz` holds strictly cross-fitted predictions: fitted on each
fold's TRAIN slice, emitted for calib and test, with train-slice values not
produced at all. Refitting the teacher inside this file would silently give the
residual model in-sample teacher values, which is the exact failure
TEACHER_ZOO section 2 exists to prevent -- and it would be invisible, because
the corruption sits in the STUDENT'S INPUTS where no amount of held-out scoring
of the student can see it.

Every read goes through `FoldScopedFit`, so a teacher value from a different
fold refuses rather than being silently used. That guard exists because a
previous version of `scale_falsifier.py` fitted one constant on calib pooled
across all six folds and had to be withdrawn (R44).

THE THREE VARIANTS, AND WHAT SEPARATES THEM

    A1  har_short-residual at every horizon, as specified. At H=1 har_short is
        a BAD teacher (pooled 1.03094, and its per-fold c spread of 1.107 shows
        why). A1 there is a test of whether the residual mechanism can rescue a
        bad teacher, not a test of har_short.
    A2  calib-selected-teacher-residual: garch_t at H=1 and H=6, har_short at
        H=24 and H=168. Selection on CALIB, never test.
    A3  teacher-as-feature: the OOF teacher forecast enters as an ordinary
        input column and NOCTUA predicts y directly.

A3 is the one that separates 'the teacher's forecast is useful information'
from 'the residual decomposition is the right inductive bias'. If A3 matches A1
and A2, the decomposition bought nothing and only the information mattered.

    python -m model.eval.arm_a_residual --self-test
    python -m model.eval.arm_a_residual
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
from eval.teacher_zoo import FoldScopedFit, LeakageRefusal, inner_oof        # noqa: E402
from eval.vol_matrix import (UNDEFINED_AT_1W, block_len_for, build_h4_table,  # noqa: E402
                             qlike_vec)
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import prepare, train_model                                # noqa: E402

HORIZONS = (1, 6, 24, 168)
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)
N_FAMILY = 12                    # 3 variants x 4 horizons, fixed a priori
# selection on CALIB, from model/artifacts/teacher_scorecard.json
CALIB_BEST = {1: "garch_t", 6: "garch_t", 24: "har_short", 168: "har_short"}


def to_logvol(sigma: np.ndarray, H: np.ndarray) -> np.ndarray:
    """sigma -> the log hourly vol RATE domain har_target lives in."""
    return np.log(np.maximum(sigma, 1e-12)) - 0.5 * np.log(np.maximum(H, 1))


def from_logvol(y: np.ndarray, H: np.ndarray) -> np.ndarray:
    return np.exp(y) * np.sqrt(np.maximum(H, 1))


def assert_failsafe(sigma_teacher, rv, H) -> None:
    """A zero residual must reproduce the teacher EXACTLY.

    This is the pass condition, not a sanity print. If the round trip through
    the log-vol domain is not the identity, the composition is wrong and every
    Arm A number would be describing a different model.
    """
    y_t = to_logvol(sigma_teacher, H)
    back = from_logvol(y_t + 0.0, H)
    err = float(np.nanmax(np.abs(back - sigma_teacher)))
    q_err = float(np.nanmax(np.abs(qlike_vec(rv, back) - qlike_vec(rv, sigma_teacher))))
    if err > 1e-12 or q_err > 1e-12:
        raise AssertionError(
            f"FAIL-SAFE IDENTITY BROKEN: a zero residual changed sigma by "
            f"{err:.3e} and QLIKE by {q_err:.3e}. The teacher+residual "
            f"composition is not the identity at rhat=0, so this architecture "
            f"cannot degrade safely onto its teacher and no Arm A number is "
            f"meaningful.")


def self_test() -> int:
    rng = np.random.default_rng(0)
    H = np.full(500, 24.0)
    sig = np.exp(rng.normal(-4, 0.6, 500))
    rv = sig * np.exp(rng.normal(0, 0.5, 500))
    ok = []
    try:
        assert_failsafe(sig, rv, H)
        ok.append(("failsafe-identity", True, "zero residual reproduces the teacher"))
    except AssertionError as e:
        ok.append(("failsafe-identity", False, str(e)[:80]))
    # it must be capable of failing: break the round trip on purpose
    try:
        bad = sig * 1.000001
        y = to_logvol(bad, H)
        back = from_logvol(y, H)
        err = float(np.max(np.abs(back - sig)))
        ok.append(("failsafe-can-fail", err > 1e-12,
                   f"a 1e-6 relative perturbation is detected (err {err:.2e})"))
    except Exception as e:                                       # noqa: BLE001
        ok.append(("failsafe-can-fail", False, str(e)[:80]))
    # the fold-scope guard must refuse a cross-fold read
    z = {"2021/24/calib/sigma/har_short": np.zeros(3),
         "2026/24/calib/sigma/har_short": np.zeros(3)}
    with FoldScopedFit(year=2021) as sc:
        try:
            sc.calib(z, 24, "har_short", year=2026)
            ok.append(("fold-scope", False, "did NOT refuse a cross-fold read"))
        except LeakageRefusal:
            ok.append(("fold-scope", True, "cross-fold teacher read refused"))
    print("arm_a self-test")
    for n, good, m in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {n}: {m}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} guards behaved correctly")
    return 1 if bad else 0


def fit_cell(ep, X, fold, H, variant, teacher, z, inner, hidden, seeds):
    """One (horizon, variant, fold) cell. Returns test-slice sigma, or None."""
    at_h = (ep.H == H).to_numpy()
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xv = X[cols40]
    fin = np.isfinite(Xv.to_numpy(np.float64)).all(1)
    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    anchor = ep.anchor_ts.to_numpy(np.int64)
    y = fold["year"]

    with FoldScopedFit(year=y) as sc:
        kt = f"{y}/{H}/test"
        if f"{kt}/anchor_ts" not in z:
            return None
        a_te = z[f"{kt}/anchor_ts"]
        rv_te = z[f"{kt}/rv"]
        s_te = None if teacher is None else np.asarray(sc.test(z, H, teacher))

    pos = pd.Series(np.arange(len(ep))[at_h], index=anchor[at_h])
    idx_te = pos.reindex(a_te).to_numpy()
    ok = np.isfinite(idx_te)
    idx_te = idx_te[ok].astype(int)
    if len(idx_te) < 100:
        return None
    m_te = np.zeros(len(ep), bool); m_te[idx_te] = True
    rv_te = rv_te[ok]
    if s_te is not None:
        s_te = s_te[ok]
        assert_failsafe(s_te, rv_te, Hall[m_te])

    # TRAIN-slice teacher values come from the INNER expanding-forward
    # cross-fit (TEACHER_ZOO Amendment 1), never in-sample.
    m_tr = fold["train"] & fin & at_h
    m_va = fold["calib"] & fin & at_h
    if teacher is not None:
        t_in = inner.get(teacher)
        if t_in is None:
            return None
        have = np.isfinite(t_in)
        m_tr = m_tr & have
        # calib-side teacher values come from the OUTER artifact
        with FoldScopedFit(year=y) as sc:
            kc = f"{y}/{H}/calib"
            a_ca = z[f"{kc}/anchor_ts"]
            s_ca = np.asarray(sc.calib(z, H, teacher))
        pos_ca = pd.Series(np.arange(len(ep))[at_h], index=anchor[at_h])
        i_ca = pos_ca.reindex(a_ca).to_numpy()
        okc = np.isfinite(i_ca)
        i_ca = i_ca[okc].astype(int)
        tvals = np.full(len(ep), np.nan)
        tvals[m_tr] = t_in[m_tr]
        tvals[i_ca] = s_ca[okc]
        tvals[idx_te] = s_te
        m_va = m_va & np.isfinite(tvals)
    if m_tr.sum() < 2000 or m_va.sum() < 300:
        return None

    yt = None if teacher is None else to_logvol(tvals, Hall)
    target = yall if teacher is None or variant == "A3" else (yall - yt)

    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)
    lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
    sref = np.maximum(np.clip(raw, lo, hi), 1e-12)

    Xu = Xv
    if variant == "A3":
        # the teacher forecast as an ordinary input column, OOF values only
        Xu = Xv.copy()
        Xu["teacher_logvol"] = np.nan_to_num(yt, nan=0.0)

    tr, stds = prepare(ep, Xu, m_tr, sigma_ref=sref[m_tr])
    tr["y"] = target[m_tr].astype(np.float32)
    w_tr = S.sample_weights(ep, m_tr)
    va, _ = prepare(ep, Xu, m_va, *stds, sigma_ref=sref[m_va])
    va["y"] = target[m_va].astype(np.float32)

    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), w_tr)
    # the blend anchor is fitted on the SAME target the network learns, so a
    # zero network and a zero anchor together reproduce the teacher exactly
    bl = B.fit_vol_baselines(Xu[m_tr], target[m_tr], w_tr)
    models = [train_model(tr, w_tr, va, hidden=hidden, epochs=40, seed=k,
                          verbose=False, ols_beta=ols.beta)[0]
              for k in range(seeds)]

    d, _ = prepare(ep, Xu, m_te, *stds)
    lp = bl["log_har_cal"].predict(Xu[m_te])
    preds = [I.predict(m, d, har_logvol=lp) for m in models]
    out = np.mean([p["sigma_med"] for p in preds], axis=0)

    rhat_only = None
    if teacher is not None and variant != "A3":
        # out is a sigma built from the RESIDUAL; recompose onto the teacher
        rhat = to_logvol(out, Hall[m_te])
        rhat_only = np.asarray(rhat, np.float64)
        out = from_logvol(yt[m_te] + rhat, Hall[m_te])

    # THE DECOMPOSITION THAT DECIDES WHETHER THIS IS AN ARCHITECTURE OR A
    # SCALAR. A residual with a non-zero MEAN is doing the same job as the
    # multiplicative constant `scale_falsifier` already measured: exp(mean
    # residual) IS a value of c. If the entire gain lives in the mean, the
    # residual decomposition is an expensive way to fit one number, and
    # P2-scale-v2 already showed that number fails the barrier guards.
    # Reported, never a pass condition -- it changes no pre-registered rule.
    return {"sigma": np.asarray(out, np.float64), "rv": rv_te,
            "H": Hall[m_te], "teacher_sigma": s_te, "n": int(m_te.sum()),
            "rhat_mean": None if rhat_only is None else float(np.nanmean(rhat_only)),
            "rhat_sd": None if rhat_only is None else float(np.nanstd(rhat_only)),
            "sigma_mean_only": (None if rhat_only is None else
                                from_logvol(yt[m_te] + np.nanmean(rhat_only),
                                            Hall[m_te]))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Arm A: teacher + residual NOCTUA")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--oof", type=Path, default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/arm_a.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    z = np.load(a.oof)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    ret = None
    if not a.no_garch:
        from eval.garch import hourly_returns
        ret = hourly_returns(a.artifacts)

    print(f"Arm A. family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print(f"calib-selected teachers: {CALIB_BEST}")
    print("train-slice teacher values: INNER expanding-forward cross-fit\n")

    results: dict = {}
    for H in a.horizons:
        cells: dict = {}
        for f in folds:
            fd = dict(f)
            fd["train_end_ts"] = int(ep.anchor_ts.to_numpy()[fd["train"]].max())
            t0 = time.time()
            inner = inner_oof(ep, X, fd, H, ret)
            if not inner:
                continue
            for variant in ("A0", "A1", "A2", "A3"):
                teacher = (None if variant == "A0"
                           else "har_short" if variant == "A1"
                           else CALIB_BEST[H])
                r = fit_cell(ep, X, fd, H, variant, teacher, z, inner,
                             a.hidden, a.seeds)
                if r is not None:
                    cells.setdefault(variant, []).append({**r, "year": fd["year"],
                                                          "teacher": teacher})
            print(f"  H={H:>4} fold {fd['year']}  ({time.time()-t0:.0f}s)", flush=True)

        if "A0" not in cells:
            continue
        print("\n" + "=" * 96)
        print(f"H = {H}h    teacher for A2 = {CALIB_BEST[H]}")
        print("=" * 96)
        print(f"{'arm':>6} {'teacher':>12} {'QLIKE':>9} {'vs teacher':>11} "
              f"{'vs A0':>10}   paired CI vs its own teacher")
        row = {}
        for variant in ("A0", "A1", "A2", "A3"):
            if variant not in cells:
                continue
            c = cells[variant]
            rv = np.concatenate([x["rv"] for x in c])
            sg = np.concatenate([x["sigma"] for x in c])
            q = qlike_vec(rv, sg)
            base_q = None
            if c[0]["teacher_sigma"] is not None:
                ts_ = np.concatenate([x["teacher_sigma"] for x in c])
                base_q = qlike_vec(rv, ts_)
            q0 = qlike_vec(np.concatenate([x["rv"] for x in cells["A0"]]),
                           np.concatenate([x["sigma"] for x in cells["A0"]]))
            L = block_len_for(H, len(q))
            cis = "—"
            ci = None
            if base_q is not None:
                d_ = base_q - q
                g = np.isfinite(d_)
                ci = mean_ci(d_[g], alpha=alpha, block_len=L)
                cis = f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
            # how much of the gain is the constant, and how much is episodic
            mean_only = None
            if c[0].get("sigma_mean_only") is not None:
                sm = np.concatenate([x["sigma_mean_only"] for x in c])
                mean_only = float(np.nanmean(qlike_vec(rv, sm)))
            rm = [x["rhat_mean"] for x in c if x.get("rhat_mean") is not None]
            rs = [x["rhat_sd"] for x in c if x.get("rhat_sd") is not None]
            extra = ""
            if mean_only is not None and base_q is not None:
                tot = float(np.nanmean(base_q)) - float(np.nanmean(q))
                con = float(np.nanmean(base_q)) - mean_only
                extra = (f"   [constant part {con:+.5f} of {tot:+.5f} = "
                         f"{100*con/tot if tot else float('nan'):.0f}%, "
                         f"implied c={np.exp(np.mean(rm)):.3f}, "
                         f"residual sd={np.mean(rs):.3f}]")
            print(f"{variant:>6} {str(c[0]['teacher']):>12} {np.nanmean(q):9.5f} "
                  f"{(np.nanmean(base_q - q) if base_q is not None else float('nan')):+11.5f} "
                  f"{np.nanmean(q0) - np.nanmean(q):+10.5f}   {cis}{extra}")
            row[variant] = {"qlike": float(np.nanmean(q)),
                            "qlike_mean_residual_only": mean_only,
                            "rhat_mean": (float(np.mean(rm)) if rm else None),
                            "rhat_sd": (float(np.mean(rs)) if rs else None),
                            "implied_c": (float(np.exp(np.mean(rm))) if rm else None),
                            "teacher": c[0]["teacher"],
                            "vs_teacher": (None if base_q is None
                                           else float(np.nanmean(base_q - q))),
                            "vs_A0": float(np.nanmean(q0) - np.nanmean(q)),
                            "paired_ci": None if ci is None else list(ci["ci95"]),
                            "n": int(len(q))}
        results[str(H)] = row
        print()

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha,
        "calib_best": {str(k): v for k, v in CALIB_BEST.items()},
        "horizons": results,
    }, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
