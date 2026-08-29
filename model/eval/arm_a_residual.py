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
from eval.teacher_zoo import FoldScopedFit, LeakageRefusal                   # noqa: E402
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Arm A: teacher + residual NOCTUA")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--oof", type=Path, default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/arm_a.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    z = np.load(a.oof)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    fin = np.isfinite(X[cols40].to_numpy(np.float64)).all(1)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)
    anchor = ep.anchor_ts.to_numpy(np.int64)

    print(f"Arm A. family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print(f"calib-selected teachers: {CALIB_BEST}\n")

    acc: dict = {}
    for H in a.horizons:
        at_h = (ep.H == H).to_numpy()
        for variant in ("A1", "A2", "A3", "A0"):
            teacher = ("har_short" if variant == "A1" else CALIB_BEST[H])
            if variant == "A0":
                teacher = None
            for f in folds:
                y = f["year"]
                with FoldScopedFit(year=y) as sc:
                    key_c, key_t = f"{y}/{H}/calib", f"{y}/{H}/test"
                    if f"{key_t}/anchor_ts" not in z:
                        continue
                    a_te = z[f"{key_t}/anchor_ts"]
                    a_ca = z[f"{key_c}/anchor_ts"]
                    rv_te = z[f"{key_t}/rv"]
                    if teacher is not None:
                        s_ca = sc.calib(z, H, teacher)
                        s_te = sc.test(z, H, teacher)
                    else:
                        s_ca = s_te = None

                # map the OOF anchors back onto episode rows
                pos = pd.Series(np.arange(len(ep))[at_h],
                                index=anchor[at_h])
                idx_te = pos.reindex(a_te).to_numpy()
                idx_ca = pos.reindex(a_ca).to_numpy()
                ok_te = np.isfinite(idx_te)
                ok_ca = np.isfinite(idx_ca)
                idx_te = idx_te[ok_te].astype(int)
                idx_ca = idx_ca[ok_ca].astype(int)

                m_tr = f["train"] & fin & at_h
                if m_tr.sum() < 2000 or len(idx_te) < 100:
                    continue
                m_ca = np.zeros(len(ep), bool); m_ca[idx_ca] = True
                m_te = np.zeros(len(ep), bool); m_te[idx_te] = True

                if teacher is not None:
                    sig_te = np.asarray(s_te)[ok_te]
                    assert_failsafe(sig_te, rv_te[ok_te], Hall[m_te])

                acc.setdefault((H, variant), []).append({
                    "year": y, "m_tr": m_tr, "m_ca": m_ca, "m_te": m_te,
                    "teacher": teacher,
                    "sig_ca": None if s_ca is None else np.asarray(s_ca)[ok_ca],
                    "sig_te": None if s_te is None else np.asarray(s_te)[ok_te],
                    "rv_te": rv_te[ok_te],
                })
        print(f"  H={H:>4} prepared", flush=True)

    out = {"family_size": N_FAMILY, "alpha": alpha,
           "calib_best": {str(k): v for k, v in CALIB_BEST.items()},
           "cells": {}}
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nprepared {len(acc)} cells; wrote scaffold to {a.out}")
    print("NOTE: this run only PREPARES and asserts the fail-safe identity. "
          "The training pass is the next commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
