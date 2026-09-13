"""
eval/regularisation.py
=====================================================================
P3-regularisation: the model's only regulariser was one hard-coded number.

WHERE THIS HYPOTHESIS CAME FROM, WHICH MATTERS

It was not designed. `P3-exogenous-dvol` ran a control arm whose entire purpose
was to SINK a result: one column of real values randomly reassigned to episodes,
carrying no information whatsoever. It improved pooled QLIKE by **1.46% at H=1**
and **1.24% at H=6**, both clearing Bonferroni-corrected intervals. That killed
the exogenous feature -- and it is also a measurement of something else entirely.

A model that is helped by a noise input is under-regularised. The regularisation
this model had was `AdamW(weight_decay=1e-4)`, hard-coded, with no dropout and no
input noise. So the obvious question is whether doing the thing deliberately beats
doing it by accident.

THE SHARP TEST

If the noise-column gain was regularisation, an explicit regulariser should reach
at least the same +1.46% at H=1. **If no setting reproduces it, "under-regularised"
is the wrong explanation** and something else produced that control-arm gain -- a
change in initialisation scale from the extra input, or an interaction with the
OneCycle schedule. Either way the entry reports what happened rather than keeping
the tidy story.

WHY THIS DESIGN IS CLEANER THAN THE ONE THAT PRODUCED IT

Every arm here uses the SAME 40 columns and the SAME full sample. Nothing differs
but two training hyperparameters, so there is no capacity term to control for and
no restricted sub-sample to caveat. The pairing is identical by construction.

    python -m model.eval.regularisation --self-test
    python -m model.eval.regularisation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                           # noqa: E402
from eval.vol_matrix import (HORIZONS, UNDEFINED_AT_1W, block_len_for,        # noqa: E402
                             build_h4_table, qlike_vec)
from noctua import baselines as B                                            # noqa: E402
from noctua import infer as I                                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.model import BASE_COLS                                           # noqa: E402
from noctua.train import prepare, train_model                                # noqa: E402

# (weight_decay, input_noise). `base` is the shipped setting, exactly.
ARMS = {
    "base":     (1e-4, 0.00),
    "wd-1e-3":  (1e-3, 0.00),
    "wd-1e-2":  (1e-2, 0.00),
    "noise-05": (1e-4, 0.05),
    "noise-15": (1e-4, 0.15),
}
N_FAMILY = len(HORIZONS) * (len(ARMS) - 1)        # 16, pre-registered
# what the accident achieved, and therefore the bar
ACCIDENT = {1: 1.46, 6: 1.24, 24: 0.12, 168: 0.30}


def run_arm(ep, X, fold, H, arm, hidden, seeds):
    wd, noise = ARMS[arm]
    at_h = (ep.H == H).to_numpy()
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xb = X[cols40]
    fin = np.isfinite(Xb.to_numpy(np.float64)).all(1)
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
    tr, stds = prepare(ep, Xb, m_tr, shape_cols=cols40, sigma_ref=sref[m_tr])
    w = S.sample_weights(ep, m_tr)
    va, _ = prepare(ep, Xb, m_va, *stds, shape_cols=cols40, sigma_ref=sref[m_va])
    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), w)
    bl = B.fit_vol_baselines(Xb[m_tr], yall[m_tr], w)
    models = [train_model(tr, w, va, hidden=hidden, epochs=40, seed=k,
                          verbose=False, ols_beta=ols.beta,
                          weight_decay=wd, input_noise=noise)[0]
              for k in range(seeds)]
    d, _ = prepare(ep, Xb, m_te, *stds, shape_cols=cols40)
    lp = bl["log_har_cal"].predict(Xb[m_te])
    preds = [I.predict(m, d, har_logvol=lp) for m in models]
    sg = np.asarray(np.mean([p["sigma_med"] for p in preds], axis=0), np.float64)
    return {"rv": ep.RV.to_numpy()[m_te], "sigma": sg,
            "anchor": ep.anchor_ts.to_numpy()[m_te]}


def self_test() -> int:
    import torch
    ok = []
    ok.append(("base-is-the-shipped-setting", ARMS["base"] == (1e-4, 0.0),
               "weight_decay 1e-4 and no input noise, as train.py hard-coded"))
    ok.append(("arms-are-distinct", len(set(ARMS.values())) == len(ARMS),
               f"{len(ARMS)} arms, {len(set(ARMS.values()))} distinct settings"))
    ok.append(("family-size", N_FAMILY == 16,
               f"{len(HORIZONS)} horizons x {len(ARMS)-1} non-base arms = "
               f"{N_FAMILY}"))

    # the two knobs must actually reach the optimiser and the loop
    rng = np.random.default_rng(0)
    n, k = 4096, len(BASE_COLS)
    def fake():
        return {"Xa": rng.normal(size=(n, k)).astype(np.float32),
                "Xb": rng.normal(size=(n, k)).astype(np.float32),
                "Xs": rng.normal(size=(n, k)).astype(np.float32),
                "y": rng.normal(size=n).astype(np.float32),
                "log_sigma": rng.normal(size=n).astype(np.float32),
                "r": rng.normal(size=n).astype(np.float32),
                "m_up": np.abs(rng.normal(size=n)).astype(np.float32),
                "m_dn": np.abs(rng.normal(size=n)).astype(np.float32),
                "m_mx": np.abs(rng.normal(size=n)).astype(np.float32)}
    tr, va = fake(), fake()
    w = np.ones(n, np.float64)
    def fit(**kw):
        m, _ = train_model(tr, w, va, hidden=8, epochs=2, seed=0,
                           verbose=False, **kw)
        return torch.cat([p.detach().flatten() for p in m.parameters()])
    p0 = fit()
    ok.append(("weight-decay-changes-the-fit",
               not torch.allclose(p0, fit(weight_decay=1e-1)),
               "wd 1e-4 vs 1e-1 gives different weights"))
    ok.append(("input-noise-changes-the-fit",
               not torch.allclose(p0, fit(input_noise=0.5)),
               "noise 0 vs 0.5 gives different weights"))
    ok.append(("defaults-are-a-no-op",
               torch.allclose(p0, fit(weight_decay=1e-4, input_noise=0.0)),
               "passing the defaults explicitly reproduces the shipped fit "
               "bit-for-bit -- so the production artifact cannot move"))

    # and the noise must be TRAINING-only: prediction stays deterministic
    m, _ = train_model(tr, w, va, hidden=8, epochs=2, seed=0, verbose=False,
                       input_noise=0.5)
    import torch as T
    with T.no_grad():
        x = T.tensor(tr["Xa"][:16]); xb = T.tensor(tr["Xb"][:16])
        a1, _ = m.a(x, xb, return_parts=True)
        a2, _ = m.a(x, xb, return_parts=True)
    ok.append(("noise-is-training-only", T.allclose(a1, a2),
               "a noise-trained model predicts deterministically"))

    print("regularisation self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-regularisation")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/regularisation_result.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    print(f"P3-regularisation   family {N_FAMILY} -> {100*(1-alpha):.4f}% "
          f"intervals")
    print("the bar is what a SHUFFLED COLUMN achieved by accident: "
          + ", ".join(f"H={k} {v:+.2f}%" for k, v in ACCIDENT.items()) + "\n")
    out = {"n_family": N_FAMILY, "alpha": alpha, "arms": {k: list(v) for k, v
                                                         in ARMS.items()},
           "accident_pct": ACCIDENT, "horizons": {}}

    for H in a.horizons:
        got = {}
        for arm in a.arms:
            parts = [r for f in folds
                     if (r := run_arm(ep, X, f, H, arm, a.hidden, a.seeds))]
            if parts:
                got[arm] = {k: np.concatenate([p[k] for p in parts])
                            for k in ("rv", "sigma", "anchor")}
        if "base" not in got:
            continue
        q0 = qlike_vec(got["base"]["rv"], got["base"]["sigma"])
        print("=" * 88)
        print(f"H = {H}h   {len(q0):,} test episodes   "
              f"accident bar {ACCIDENT.get(H, float('nan')):+.2f}%")
        print("=" * 88)
        print(f"{'arm':>10} {'pooled':>9} {'vs base':>10} {'rel %':>7} "
              f"{'beats accident?':>16} {'paired CI vs base':>28}")
        row = {}
        for arm, dd_ in got.items():
            if not np.array_equal(dd_["anchor"], got["base"]["anchor"]):
                raise SystemExit(f"REFUSING: {arm} scored different episodes")
            q = qlike_vec(dd_["rv"], dd_["sigma"])
            dd = q0 - q
            g = np.isfinite(dd)
            L = block_len_for(H, int(g.sum()))
            ci = mean_ci(dd[g], alpha=alpha, block_len=L)
            rel = 100 * np.nanmean(dd) / np.nanmean(q0)
            clears = bool(ci["ci95"][0] > 0)
            beats = rel >= ACCIDENT.get(H, np.inf)
            row[arm] = {"pooled": float(np.nanmean(q)),
                        "delta": float(np.nanmean(dd)), "rel_pct": float(rel),
                        "ci": [float(ci["ci95"][0]), float(ci["ci95"][1])],
                        "clears": clears, "beats_accident": bool(beats),
                        "block_len": int(L)}
            mark = "" if arm == "base" else (
                f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
                + ("  CLEARS" if clears else ""))
            print(f"{arm:>10} {np.nanmean(q):9.5f} {np.nanmean(dd):+10.5f} "
                  f"{rel:+7.2f} {('YES' if beats and arm != 'base' else '-'):>16} "
                  f"{mark:>28}")
        live = {k: v for k, v in row.items() if k != "base"}
        any_clear = [k for k, v in live.items() if v["clears"]]
        any_beat = [k for k, v in live.items() if v["beats_accident"]]
        print(f"\n  clears: {any_clear or 'none'}")
        verdict = (", ".join(any_beat) if any_beat else
                   "NONE -- under-regularisation is then the WRONG explanation "
                   "for that control arm")
        bar = ACCIDENT.get(H, float("nan"))
        print(f"  reaches the accident's {bar:+.2f}%: {verdict}")
        out["horizons"][str(H)] = row
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"  (partial result written)\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
