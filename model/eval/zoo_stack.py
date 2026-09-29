"""
eval/zoo_stack.py
=====================================================================
P4-zoo-stack: turn the teacher zoo from a scoreboard into an ensemble.

THE IDEA IN ONE PARAGRAPH

The eight teachers make OPPOSITE errors. Persistence over-reacts (MZ slope
0.92 / 0.85 / 0.71 / 0.53 at H = 1 / 6 / 24 / 168); the smooth models and
NOCTUA under-react (NOCTUA-mean 1.30 / 1.24 / 1.12). Averaging forecasts whose
errors point in opposite directions cancels part of both -- the most robust
empirical result in forecasting. The zoo has existed for weeks and has only
ever been used to rank its members against each other.

WHAT IS BUILT

    log sigma_ens = sum_k w_k log sigma_k + b,     w >= 0,  sum_k w_k = 1

CONVEX weights on purpose. Any change in how much the ensemble reacts must come
from MIXING teachers that react differently, never from stretching one
forecast's own variation -- that stretching is the affine slope correction
twenty-three experiments showed does not survive out of sample.

Weights are fitted per fold on CALIB by minimising QLIKE directly. With the
level b profiled out in closed form (the QLIKE-optimal constant), the objective
in w is

    J(w) = log E[x] - E[log x],    x = RV^2 / sigma_w^2

a log-mean-exp of a linear function minus a linear function, which is CONVEX
in w: one optimum, no restarts, no tuning.

THIS IS A GENERALISATION OF SOMETHING ALREADY SHIPPED. The served forecast is a
two-model log-space convex combination -- NOCTUA and Log-HAR at 0.25 / 0.75
(`infer.BLEND_W`), with the weight set by hand from a sweep. The stacker is the
same object with N models and the weight set by the loss the forecast is scored
on.

    python -m model.eval.zoo_stack --selftest
    python -m model.eval.zoo_stack
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                       # noqa: E402
from eval.mz_recalibration import YEARS, fit_mz, qlike_vec               # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                    # noqa: E402
from eval.teacher_zoo import FoldScopedFit                               # noqa: E402

ZOO = ("garch_normal", "garch_t", "har_short", "log_har", "log_har_cal",
       "noctua_v1", "noctua_v1_mean", "persistence")
NOCTUA = ("noctua_v1", "noctua_v1_mean")
REF = "noctua_v1_mean"
ARMS = ("ref", "eq", "stack", "stack_np",
        "stack_exp", "stack_half", "stack_exp_half")
# P4-zoo-stack-v2. 0.5 is R82's constant, fixed before this family existed.
SHRINK = 0.5


def _clean(rv, S):
    """Rows where the outcome and EVERY teacher are finite and positive."""
    ok = np.isfinite(rv) & (rv > 0)
    ok &= np.all(np.isfinite(S) & (S > 0), axis=1)
    return ok


def profiled_qlike(w, L, lrv2) -> float:
    """QLIKE at weights w with the level b set to its closed-form optimum.

    L is (n, K) log sigma; lrv2 is log RV^2. With u = lrv2 - 2 L @ w the
    optimal level makes E[exp(u - 2b)] = 1, and the loss reduces to the Jensen
    gap log E[e^u] - E[u]. Computed with a max-shift so it cannot overflow on
    a violent night.
    """
    u = lrv2 - 2.0 * (L @ w)
    m = float(np.max(u))
    return float(np.log(np.mean(np.exp(u - m))) + m - np.mean(u))


def level_for(w, L, lrv2) -> float:
    """The QLIKE-optimal log level b for fixed weights: b = 0.5 log E[e^u]."""
    u = lrv2 - 2.0 * (L @ w)
    m = float(np.max(u))
    return 0.5 * (float(np.log(np.mean(np.exp(u - m)))) + m)


def fit_convex(L, lrv2) -> np.ndarray:
    """QLIKE-optimal weights on the simplex, via a softmax parameterisation.

    The objective is convex in w, so a single start suffices; the softmax only
    enforces the constraint. Starts at equal weights, which is also the R82
    control, so an optimiser that cannot improve on the control returns it.
    """
    K = L.shape[1]

    def f(theta):
        w = np.exp(theta - theta.max()); w /= w.sum()
        return profiled_qlike(w, L, lrv2)

    r = minimize(f, np.zeros(K), method="L-BFGS-B")
    w = np.exp(r.x - r.x.max()); w /= w.sum()
    return w


def apply(w, b, L) -> np.ndarray:
    return np.exp(L @ w + b)


def run_horizon(z, H: int) -> dict:
    """Walk the folds: fit on calib, score on test, pool the test losses.

    EXPANDING POOL. `hist_L` / `hist_r` accumulate every PRIOR fold's calib and
    test out-of-fold forecasts, in fold order, so fold k's expanding fit sees
    exactly folds 1..k-1 plus its own calib and nothing later. A prior fold's
    TEST slice is a strictly earlier year, and its forecasts were produced by
    models that never saw it -- genuine out-of-fold predictions, which is what
    a stacker should be fitted on.
    """
    pooled = {a: [] for a in ARMS}
    rv_all, weights = [], []
    hist_L, hist_r = [], []
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kc, kt = f"{y}/{H}/calib", f"{y}/{H}/test"
            if any(f"{kt}/sigma/{t}" not in z or f"{kc}/sigma/{t}" not in z
                   for t in ZOO):
                continue
            Sc = np.column_stack([np.asarray(sc.calib(z, H, t), np.float64)
                                  for t in ZOO])
            St = np.column_stack([np.asarray(sc.test(z, H, t), np.float64)
                                  for t in ZOO])
            rc = np.asarray(z[f"{kc}/rv"], np.float64)
            rt = np.asarray(z[f"{kt}/rv"], np.float64)
        okc, okt = _clean(rc, Sc), _clean(rt, St)
        if okc.sum() < 500 or okt.sum() < 500:
            continue
        Lc, lc2 = np.log(Sc[okc]), np.log(rc[okc] ** 2)
        Lt, rt_ = np.log(St[okt]), rt[okt]

        K = len(ZOO)
        i_ref = ZOO.index(REF)
        e_ref = np.eye(K)[i_ref]
        w_eq = np.full(K, 1.0 / K)
        w_st = fit_convex(Lc, lc2)
        keep = [i for i, t in enumerate(ZOO) if t not in NOCTUA]
        w_np_sub = fit_convex(Lc[:, keep], lc2)
        w_np = np.zeros(K); w_np[keep] = w_np_sub

        # expanding: every prior fold's calib + test, plus this fold's calib
        Le = np.vstack(hist_L + [Lc]); le2 = np.concatenate(hist_r + [lc2])
        w_ex = fit_convex(Le, le2)
        w_hf = SHRINK * w_st + (1.0 - SHRINK) * e_ref
        w_eh = SHRINK * w_ex + (1.0 - SHRINK) * e_ref

        for arm, w, (Lf, lf2) in (
                ("ref", e_ref, (Lc, lc2)), ("eq", w_eq, (Lc, lc2)),
                ("stack", w_st, (Lc, lc2)), ("stack_np", w_np, (Lc, lc2)),
                # the level is fitted on the SAME data as the weights, so an
                # expanding arm gets an expanding level -- mixing the two would
                # score a weight choice and a level choice as one thing
                ("stack_exp", w_ex, (Le, le2)),
                ("stack_half", w_hf, (Lc, lc2)),
                ("stack_exp_half", w_eh, (Le, le2))):
            b = level_for(w, Lf, lf2)
            pooled[arm].append(qlike_vec(rt_, apply(w, b, Lt)))
        rv_all.append(rt_)
        weights.append({"year": y, "stack": dict(zip(ZOO, map(float, w_st))),
                        "stack_np": dict(zip(ZOO, map(float, w_np))),
                        "stack_exp": dict(zip(ZOO, map(float, w_ex))),
                        "stack_exp_half": dict(zip(ZOO, map(float, w_eh)))})
        # only AFTER this fold is scored does it join the history
        hist_L += [Lc, Lt]
        hist_r += [lc2, np.log(rt_ ** 2)]
    if not rv_all:
        return {}
    q = {a: np.concatenate(v) for a, v in pooled.items()}
    return {"q": q, "weights": weights, "n": int(sum(len(r) for r in rv_all))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QLIKE-stacked zoo ensemble")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/zoo_stack.json"))
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    contrasts = (("stack", "ref"), ("stack_exp", "ref"), ("stack_half", "ref"),
                 ("stack_exp_half", "ref"), ("eq", "ref"),
                 ("stack", "stack_np"))
    # P4-zoo-stack-v2: 4 candidates x 4 horizons, each against ref
    n_family = 4 * len(HORIZONS)
    alpha = 0.05 / n_family
    print("P4-zoo-stack   convex log-space ensemble, weights fitted on CALIB "
          "by QLIKE")
    print(f"family {n_family} -> {100*(1-alpha):.3f}% intervals, "
          f"block 2H, {a.boot} reps\n")
    out = {"zoo": ZOO, "ref": REF, "alpha": alpha, "horizons": {}}
    for H in HORIZONS:
        r = run_horizon(z, H)
        if not r:
            continue
        q = r["q"]
        print(f"=== H = {H}   ({r['n']:,} test episodes)")
        base = float(np.nanmean(q["ref"]))
        for arm in ARMS:
            m = float(np.nanmean(q[arm]))
            print(f"   {arm:>9}  QLIKE {m:.5f}   vs ref {100*(base-m)/base:+6.2f}%")
        L = max(int(round(len(q['ref']) ** (1 / 3))), 2 * H)
        ci = {}
        for c, o in contrasts:
            d = q[o] - q[c]
            g = np.isfinite(d)
            lo, hi = mean_ci(d[g], n_rep=a.boot, alpha=alpha,
                             block_len=L)["ci95"]
            tag = (f"{c} BETTER" if lo > 0 else f"{c} WORSE" if hi < 0
                   else "not separated")
            ci[f"{c}_vs_{o}"] = [float(lo), float(hi)]
            print(f"   {c:>9} vs {o:<9} {np.nanmean(d):+.5f} "
                  f"[{lo:+.5f}, {hi:+.5f}]  {tag}")
        wm = {t: float(np.mean([f["stack_exp"][t] for f in r["weights"]]))
              for t in ZOO}
        print("   mean stack_exp weights: " + "  ".join(
            f"{t}={v:.2f}" for t, v in sorted(wm.items(), key=lambda kv: -kv[1])
            if v >= 0.01))
        # does the ensemble REACT correctly? MZ slope of each arm on test
        print()
        out["horizons"][str(H)] = {
            "qlike": {arm: float(np.nanmean(q[arm])) for arm in ARMS},
            "ci": ci, "weights_by_fold": r["weights"], "mean_weights": wm,
            "n": r["n"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(5)
    n = 20000
    t = rng.normal(-4.0, 0.5, n)                       # true log sigma
    rv = np.exp(t + rng.normal(0, 0.3, n))
    lrv2 = np.log(rv ** 2)

    # 1. The profiled objective equals QLIKE at the closed-form level.
    A = t + rng.normal(0, 0.4, n); B_ = t + rng.normal(0, 0.4, n)
    L = np.column_stack([A, B_])
    w = np.array([0.3, 0.7])
    b = level_for(w, L, lrv2)
    direct = float(np.mean(qlike_vec(rv, apply(w, b, L))))
    ok.append(("profiled QLIKE equals QLIKE at the optimal level",
               abs(profiled_qlike(w, L, lrv2) - direct) < 1e-9))

    # 2. Two independent noisy copies of the truth: the optimum is ~50/50 and
    #    beats either alone. This is the whole premise.
    wf = fit_convex(L, lrv2)
    ok.append(("symmetric noise -> near-equal weights", abs(wf[0] - 0.5) < 0.05))
    q_ens = profiled_qlike(wf, L, lrv2)
    q_a = profiled_qlike(np.array([1.0, 0.0]), L, lrv2)
    ok.append(("the combination beats a single component", q_ens < q_a - 1e-4))

    # 3. A USELESS teacher gets (near) zero weight -- WHEN THE INFORMATIVE ONE
    #    IS ALREADY CORRECTLY SCALED. The first version paired junk with A,
    #    which carries noise in the regressor and therefore OVER-reacts (its MZ
    #    slope is 0.25/0.41 = 0.61), and the junk teacher took real weight. That
    #    was not a bug in the optimiser: an uninformative but correctly-centred
    #    forecast acts as SHRINKAGE, and shrinking an over-reacting forecast
    #    improves QLIKE. Check 3b pins that down, because it is how the real
    #    weights have to be read -- weight is not evidence of information.
    good = t + rng.normal(0, 0.05, n)
    junk = rng.normal(-4.0, 0.5, n)
    w3 = fit_convex(np.column_stack([good, junk]), lrv2)
    ok.append(("an uninformative teacher is weighted out", w3[1] < 0.05))
    w3b = fit_convex(np.column_stack([A, junk]), lrv2)
    ok.append(("...but it earns weight as SHRINKAGE beside an over-reactor",
               w3b[1] > 0.10))

    # 4. OPPOSITE REACTIVITY. One forecast over-reacts (too much variation),
    #    one under-reacts (too little). Neither alone has slope 1; the convex
    #    mix recovers it -- the mechanism this module is betting on.
    over = -4.0 + 1.6 * (t + 4.0) + rng.normal(0, 0.05, n)
    under = -4.0 + 0.6 * (t + 4.0) + rng.normal(0, 0.05, n)
    L4 = np.column_stack([over, under])
    w4 = fit_convex(L4, lrv2)
    mix = L4 @ w4
    beta = float(np.polyfit(mix, np.log(rv), 1)[0])
    ok.append(("mixing opposite reactivity recovers a slope near 1",
               abs(beta - 1.0) < 0.08))

    # 5. Weights stay on the simplex.
    ok.append(("weights are convex",
               np.all(w4 >= 0) and abs(w4.sum() - 1.0) < 1e-9))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
