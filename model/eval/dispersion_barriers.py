"""
eval/dispersion_barriers.py
=====================================================================
P3-dispersion-barriers-v2: the model is over-dispersed by 11-28%. Correcting
that costs QLIKE. Does it help the thing the dispersion actually builds?

THE SPLIT THAT MAKES THIS WORTH RUNNING AFTER A NEGATIVE SCREEN

QLIKE asks that sigma^2 equal E[RV^2]. `P3-dispersion` showed that at H=6 and
H=24 it is satisfied by a dispersion error and a level error CANCELLING --
0.8719 x 1.1371 = 0.9915 at H=24 -- and `P3-dispersion-screen` showed that
removing either one alone therefore makes QLIKE worse, by up to 5.5%.

The barrier battery asks something different: that the DISTRIBUTION be right.
A cancellation in a point forecast does nothing for a probability curve. The
committee builds every curve from `sigma_atoms`, so an 11-28% width error is an
error in the object the product is made of. That is the open question, and a
negative QLIKE screen cannot answer it.

THE ARMS, AND WHY M3 IS THE EXPERIMENT

    M0  shipped, disp_lambda = 1.0
    M1  disp_lambda fitted per fold on CALIB as s_realised / s_implied
    M2  disp_lambda fixed at the sample-median fitted value -- the control for
        "does fitting it per fold do anything a constant does not" (R82)
    M3  disp_lambda = 2 - lambda_fitted: the MIRROR, same magnitude of change
        in the opposite direction

M3 is what licenses the inference. `P2-mean-level` moved the LEVEL and found
all six barrier metrics degraded by about 20% -- with its SHUFFLED arm
degrading them identically, which proved the damage belonged to moving the
level at all rather than to the correction's content. If M1 and M3 improve or
degrade together here, the same conclusion follows for the width and the atoms
are not safe to move in any direction.

An earlier registration specified M3 as "disp_lambda permuted across episodes".
That is incoherent for a per-fold SCALAR -- permuting a constant leaves it
unchanged, so M3 would have been bit-identical to M1 while looking like a
placebo. The control's form was copied from a prior experiment instead of being
derived from what this arm varies. Amended before any arm ran.

    python -m model.eval.dispersion_barriers --selftest
    python -m model.eval.dispersion_barriers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                    # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                   # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

PROD_H = 19
ARMS = ("M1", "M2", "M3")
# P3-dispersion-deployable. M2's constant is the median over ALL SIX folds, so
# scoring 2021 uses a value computed from 2024-2026: fine for a control whose
# job is to show the per-fold fit adds nothing, disqualifying for a rule that
# could ship. M4 is the causal version -- the median of PAST folds' lambda
# only -- and M5 is its mirror, so the deployable arm carries its own placebo
# rather than borrowing M3's. The first fold has no past and runs at lambda =
# 1.0, which is not a handicap here the way `drift`'s w = 0 was: 1.0 is the
# SHIPPED behaviour, so a cold fold scores the incumbent rather than nothing.
ARMS_DEPLOY = ("M1", "M4", "M5")
# P3-dispersion-conditional. ONE arm, because the registration fixed the family
# at 6 metrics x 2 subsets = 12 and scoring three arms conditionally would be
# 36 -- a family widened after the rule was written.
# EACH ARM SET HAS THE SAME THREE ROLES AND THE VERDICT BLOCK READS THEM BY
# ROLE. It used to name M1, M2 and M3 literally, which was correct while there
# was one arm set and became a KeyError the moment there were two -- after a
# multi-hour run had already printed every number it computed. The roles are:
#   candidate  the arm the registration is about
#   mirror     its placebo, the same magnitude of change in the other direction
#   rival      the arm it must not be significantly worse than
ROLES = {
    ("M1", "M2", "M3"): {"candidate": "M1", "mirror": "M3", "rival": "M2"},
    ("M1", "M4", "M5"): {"candidate": "M4", "mirror": "M5", "rival": "M1"},
}
ARMS_COND = ("M1",)
SPIKE_Q = 0.95
# The floor `run_fold` already applies to a conditional subset, restated here
# so the refusal above and the silent skip down there cannot drift apart.
MIN_COND_EPISODES = 30
METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
# DSC is a discrimination score: higher is better. Every other metric here is
# a loss. Getting this backwards would report every result inverted for one
# column, so the direction lives in one place.
HIGHER_BETTER = {"DSC"}
# NO MODULE-LEVEL FAMILY SIZE. It used to be len(ARMS) * len(METRICS), which
# was right while there was one arm set; with two it would silently be the
# default one's size in both modes, i.e. an under-correction for whichever
# set the run actually used. The family is computed from the SELECTED arms.


def fit_lambda(rv_cal, sig_cal, sig_mean_cal) -> float:
    """s_realised / s_implied on the CALIB slice.

    s_implied comes from the model's own two scalars (sigma_mean/sigma_med =
    exp(s^2)); s_realised is the spread of outcomes about the predicted median.
    Both are calib-only, so the fold's test slice is untouched.
    """
    rv = np.asarray(rv_cal, np.float64)
    med = np.asarray(sig_cal, np.float64)
    mn = np.asarray(sig_mean_cal, np.float64)
    ok = (np.isfinite(rv) & np.isfinite(med) & np.isfinite(mn)
          & (rv > 0) & (med > 0) & (mn > 0))
    if ok.sum() < 50:
        return float("nan")
    s_imp = float(np.nanmean(np.sqrt(np.maximum(
        np.log(mn[ok] / med[ok]), 0.0))))
    s_real = float(np.nanstd(np.log(rv[ok] / med[ok]), ddof=1))
    if not (s_imp > 0):
        return float("nan")
    return s_real / s_imp


def spike_calm_masks(ep, fold, q: float = SPIKE_Q) -> dict:
    """SPIKE = realised vol in the top 5% of THIS fold's test slice, CALM its
    complement, both as masks over every episode in the table.

    DEFINED ON THE OUTCOME, deliberately and with the same wording
    eval/vol_matrix.spike_mask already carries: that makes it a conditioning
    variable for REPORTING and explicitly not something any arm may use. No
    forecast in this file sees it, and both arms are scored on identical
    episodes within each subset, so the contrast stays paired.

    The threshold is taken on the fold's own PRODUCTION test slice rather than
    on the whole table, because a fixed global threshold would put most of 2021
    in one bucket and most of 2025 in the other, and the question is about
    violent nights relative to their own era.
    """
    rv = ep["RV"].to_numpy(np.float64)
    te = np.asarray(fold["test"], bool) & S.production_mask(ep)
    sel = rv[te & np.isfinite(rv)]
    if len(sel) < 100:
        return {}
    thr = float(np.quantile(sel, q))
    spike = np.isfinite(rv) & (rv >= thr)
    return {"spike": spike, "calm": np.isfinite(rv) & ~spike}


def cond_barrier_cols(rows, cname: str, model: str = "noctua_v2") -> dict:
    """`barrier_cols` restricted to one conditional subset.

    Reads the nested dict `run_fold` writes under `rec["cond"]`, which is where
    it lives precisely so the aggregator that builds the pooled metric cannot
    see it (asserted in this module's selftest).
    """
    r = next((x for x in rows if x["model"] == model), None)
    if r is None or cname not in r.get("cond", {}):
        return {}
    slot = r["cond"][cname]
    out = {}
    for pat in ("pinball_", "crps_", "brier_", "DSC_", "MCB_", "logs_"):
        vals = [v for k, v in slot.items() if k.startswith(pat)
                and isinstance(v, (int, float))]
        if vals:
            out[pat.rstrip("_")] = float(np.mean(vals))
    return out


def expanding_median(lams: list, cold: float = 1.0) -> list:
    """One lambda per fold, each the median of the folds BEFORE it.

    The causal counterpart of M2's sample median. `lams` must be in
    walk-forward order; entry k sees exactly lams[:k], so no fold is corrected
    with a number computed from its own future. The first fold has no past and
    gets `cold`, which is 1.0 -- the SHIPPED behaviour, so a cold fold scores
    the incumbent rather than scoring nothing at all.

    Factored out of `main` so it can be asserted. Inline, its only test would
    have been a four-hour run of the thing it decides.
    """
    out = []
    for k in range(len(lams)):
        past = [float(x) for x in lams[:k] if np.isfinite(x)]
        out.append(float(np.median(past)) if past else float(cold))
    return out


def better(metric: str, arm_val: float, ref_val: float) -> bool:
    return arm_val > ref_val if metric in HIGHER_BETTER else arm_val < ref_val


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(5)
    n = 20000

    # 1-3. fit_lambda on planted data: a correctly-dispersed model returns 1,
    #      an over-dispersed one returns < 1 (narrow it), under-dispersed > 1.
    mu, s = -4.0, 0.40
    med = np.full(n, np.exp(mu))
    rv = np.exp(rng.normal(mu, s, n))
    ok.append(("correctly dispersed -> lambda ~ 1",
               abs(fit_lambda(rv, med, med * np.exp(s * s)) - 1.0) < 0.03))
    ok.append(("over-dispersed -> lambda < 1",
               fit_lambda(rv, med, med * np.exp((1.5 * s) ** 2)) < 0.75))
    ok.append(("under-dispersed -> lambda > 1",
               fit_lambda(rv, med, med * np.exp((0.6 * s) ** 2)) > 1.4))

    # 4. Refuses rather than inventing a lambda from nothing.
    ok.append(("refuses below 50 rows",
               not np.isfinite(fit_lambda(rv[:10], med[:10], med[:10]))))

    # 5-6. The mirror must be symmetric about 1 and preserve the magnitude.
    for lam in (0.8, 1.2):
        mirror = 2.0 - lam
        ok.append((f"mirror of {lam} is equidistant from 1",
                   abs((mirror - 1.0) + (lam - 1.0)) < 1e-12))

    # 7-8. Metric direction, which is the one thing that silently inverts a
    #      whole column if wrong.
    # 6-10. The causal constant. Order matters and the first fold has no past,
    #     which is exactly where an off-by-one would hide.
    seq = [0.66, 0.88, 0.82, 0.98, 0.87, 0.89]
    em = expanding_median(seq)
    ok.append(("the first fold has no past and ships unchanged", em[0] == 1.0))
    ok.append(("the second fold sees exactly one past fold",
               abs(em[1] - 0.66) < 1e-12))
    ok.append(("the third sees two", abs(em[2] - 0.77) < 1e-12))
    ok.append(("the last sees all but itself",
               abs(em[5] - float(np.median(seq[:5]))) < 1e-12))
    # A fold must never see its own value, which a median makes easy to miss:
    # replacing the LAST entry cannot change any earlier output.
    alt = seq[:5] + [0.10]
    ok.append(("changing a fold cannot change an earlier fold's lambda",
               expanding_median(alt)[:5] == em[:5]))

    # 11. THE FOOTGUN THE CONDITIONAL READOUT COULD HAVE BEEN. `barrier_cols`
    #     averages every NUMERIC key matching a metric prefix, so a conditional
    #     key such as `brier_up_2.0__spike` would be folded silently into the
    #     unconditional Brier -- a readout corrupting the number it exists to
    #     explain. run_fold puts the split in a NESTED dict instead, and this
    #     asserts that such a dict is invisible to the aggregator rather than
    #     trusting the isinstance check to stay in place.
    row = {"model": "noctua_v2", "brier_up_1.0": 0.2, "brier_dn_1.0": 0.4,
           "cond": {"spike": {"brier_up_1.0": 99.0, "brier_dn_1.0": 99.0}}}
    ok.append(("a conditional split cannot pollute the pooled metric",
               abs(barrier_cols([row])["brier"] - 0.3) < 1e-12))

    ok.append(("DSC higher is better", better("DSC", 0.9, 0.8)))
    ok.append(("brier lower is better", better("brier", 0.1, 0.2)))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="dispersion vs the barrier battery")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to dispersion_barriers.json, or "
                         "dispersion_deployable.json under --deployable, so "
                         "the two modes cannot overwrite each other")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--conditional", action="store_true",
                    help="split the battery into SPIKE and CALM subsets and "
                         "score M1 alone against M0 on each "
                         "(P3-dispersion-conditional)")
    ap.add_argument("--deployable", action="store_true",
                    help="score M1 against the CAUSAL expanding-median lambda "
                         "(M4) and its mirror (M5) instead of M2/M3")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    if a.conditional and a.deployable:
        print("REFUSING: --conditional fixes the arm set at M1 and --deployable "
              "at M1/M4/M5. Running both would score a family neither "
              "registration specified.")
        return 1
    arms = (ARMS_COND if a.conditional else
            ARMS_DEPLOY if a.deployable else ARMS)
    # FAMILY. len(arms) x 6 metrics against M0, PLUS the 6 candidate-vs-rival
    # contrasts the registration's third condition needs and the first version
    # of this runner did not count. Counting them widens every interval, which
    # can only make a clearing arm harder to clear -- the conservative
    # direction, and the only one available once the omission is known.
    n_family = (len(arms) * len(METRICS) * (2 if a.conditional else 1)
                + (0 if a.conditional else len(METRICS)))
    if a.out is None:
        a.out = Path("model/artifacts/dispersion_"
                     + ("conditional" if a.conditional else
                        "deployable" if a.deployable else "barriers") + ".json")
    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / n_family
    print(f"{'P3-dispersion-deployable' if a.deployable else 'P3-dispersion-barriers-v2'}"
          f"   production slice H={PROD_H}  seeds={a.seeds}")
    print(f"family {n_family} ({len(arms)} arms x {len(METRICS)} metrics) -> "
          f"{100*(1-alpha):.3f}% intervals\n")

    # PASS ONE: the reference, which also supplies each fold's calib slice and
    # therefore its lambda. lambda cannot be known before the model runs, so
    # two passes is exact rather than approximate.
    # R5 BEFORE THE COMPUTE, NOT AFTER IT. The production slice is ONE anchor
    # per day, so a test year holds ~365 episodes and its top 5% is ~18. A
    # six-fold interval whose per-fold metric is estimated from 18 points -- and
    # whose Brier decomposition needs an isotonic fit on those 18 -- is a
    # NON-MEASUREMENT, and discovering that from a wide interval after a
    # multi-hour run is exactly the failure R84 names. The check is here so the
    # run refuses rather than produces a table nobody should read.
    if a.conditional:
        thin = []
        for f in folds:
            m = spike_calm_masks(ep, f)
            if not m:
                continue
            n = int((m["spike"] & np.asarray(f["test"], bool)
                     & S.production_mask(ep)).sum())
            if n < MIN_COND_EPISODES:
                thin.append((f["year"], n))
        if thin:
            print("REFUSING: the SPIKE subset is too small to decompose on this "
                  "slice.\n  " + ", ".join(f"{y}: {n} episodes" for y, n in thin)
                  + f"\n  (floor is {MIN_COND_EPISODES}). Widening to every H=19 "
                  "anchor does NOT fix it: 24 hourly\n  anchors cover one night, "
                  "so ~438 wide spike episodes per fold are the same\n  ~18 "
                  "nights seen 24 times. The unit that carries independent "
                  "information is\n  the spike DAY, and there are ~107 of them "
                  "across all six folds pooled -- which\n  is a per-episode "
                  "design over the pooled set, not a per-fold one. See "
                  "P3-dispersion-conditional-result.")
            return 1

    acc, lams = [], []
    for f in folds:
        cmask = spike_calm_masks(ep, f) if a.conditional else None
        r0 = run_fold(ep, X, f, a.hidden, a.seeds, cond_masks=cmask)
        if r0 is None:
            continue
        pe0 = r0["per_episode"]
        lam = fit_lambda(pe0["rv_cal"], pe0["sigma_cal"], pe0["sigma_mean_cal"])
        if not np.isfinite(lam):
            print(f"  fold {f['year']}: no lambda, skipped"); continue
        acc.append({"year": f["year"], "lam": float(lam), "r0": r0,
                    "cmask": cmask})
        lams.append(float(lam))
        print(f"  fold {f['year']}  lambda {lam:.4f}  "
              f"(implied spread scaled to {100*lam:.1f}% of itself)")
    if not acc:
        print("no usable folds"); return 1
    k_const = float(np.median(lams))
    print(f"\nconstant arm uses the sample-median lambda = {k_const:.4f}")
    # EXPANDING MEDIAN, one per fold, from PAST folds only. `acc` is in
    # walk-forward order, so the k-th entry's past is exactly acc[:k].
    k_expand = dict(zip([r["year"] for r in acc],
                        expanding_median([r["lam"] for r in acc])))
    if a.deployable:
        print("expanding-median lambda, past folds only: "
              + ", ".join(f"{y}:{v:.4f}" for y, v in k_expand.items()))
    print()

    # PASS TWO
    rows = []
    for rec in acc:
        f = next(x for x in folds if x["year"] == rec["year"])
        r0, lam = rec["r0"], rec["lam"]
        pe0 = r0["per_episode"]
        row = {"year": rec["year"], "lam": lam,
               "q_M0": qlike_vec(pe0["rv"], pe0["sigma_mean"]),
               "bar_M0": barrier_cols(r0["rows"])}
        if a.conditional:
            for cn in ("spike", "calm"):
                row[f"bar_M0__{cn}"] = cond_barrier_cols(r0["rows"], cn)
        ke = k_expand[rec["year"]]
        lam_of = {"M1": lam, "M2": k_const, "M3": 2.0 - lam,
                  "M4": ke, "M5": 2.0 - ke}
        row["lam_expand"] = ke
        okf = True
        for arm in arms:
            r1 = run_fold(ep, X, f, a.hidden, a.seeds,
                          disp_lambda=lam_of[arm], cond_masks=rec["cmask"])
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], pe0["test_idx"]):
                raise SystemExit(
                    f"REFUSING: fold {rec['year']} arm {arm} scored different "
                    f"episodes than the reference. The contrast is not paired.")
            # The hook must leave the median EXACTLY alone, or a level change
            # is being scored as a dispersion effect.
            d = float(np.max(np.abs(pe1["sigma_med"] - pe0["sigma_med"])))
            if d > 0.0:
                raise SystemExit(
                    f"REFUSING: fold {rec['year']} arm {arm} moved sigma_med "
                    f"by {d:.3e}. disp_lambda must scale width only.")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
            if a.conditional:
                for cn in ("spike", "calm"):
                    row[f"bar_{arm}__{cn}"] = cond_barrier_cols(r1["rows"], cn)
        if okf:
            rows.append(row)
            print(f"  fold {rec['year']}: all arms scored")

    if not rows:
        print("no complete folds"); return 1

    print(f"\n{'metric':>9} {'M0':>10} " + " ".join(f"{m:>10}" for m in arms))
    bar, per_fold, bar_ci = {}, {}, {}
    for met in METRICS:
        vals, series = {}, {}
        for arm in ("M0",) + arms:
            v = [r[f"bar_{arm}"].get(met) for r in rows
                 if met in r.get(f"bar_{arm}", {})]
            series[arm] = [float(x) for x in v]
            vals[arm] = float(np.mean(v)) if v else float("nan")
        bar[met] = vals
        per_fold[met] = series
        print(f"{met:>9} " + " ".join(f"{vals[k]:10.6f}" for k in ("M0",) + arms))

    # PAIRED INTERVALS, which the registration demands and a first version of
    # this runner did not produce -- it printed bare inequalities, the exact
    # thing R69 exists to prevent, and the verdict rule below is not evaluable
    # without them. The unit is a FOLD, so n = 6: these will be wide, and that
    # is the honest width rather than a reason to quote the point estimates
    # instead. `mean_ci` is used because it is defined at every usable n and
    # reports the sign count beside the interval (anchor_freshness learned that
    # the hard way, deciding a rule on a NaN).
    print(f"\n{'metric':>9} {'arm':>4} {'delta vs M0':>13} "
          f"{'95% CI (corrected)':>28} {'folds better':>13}")
    for met in METRICS:
        sgn = 1.0 if met in HIGHER_BETTER else -1.0
        for arm in arms:
            a_s = np.asarray(per_fold[met][arm], np.float64)
            b_s = np.asarray(per_fold[met]["M0"], np.float64)
            if len(a_s) != len(b_s) or len(a_s) < 2:
                continue
            d = sgn * (a_s - b_s)          # >0 always means the arm is better
            ci = mean_ci(d, alpha=alpha)
            lo, hi = ci["ci95"]
            bar_ci[f"{met}_{arm}"] = {
                "delta": float(np.mean(d)), "ci95": [float(lo), float(hi)],
                "n_folds_better": int(np.sum(d > 0)), "n_folds": int(len(d)),
                "clears": bool(lo > 0)}
            print(f"{met:>9} {arm:>4} {np.mean(d):+13.6f} "
                  f"[{lo:+12.6f}, {hi:+12.6f}] {int(np.sum(d>0)):>6}/{len(d)}")

    # ---- P3-dispersion-conditional --------------------------------------
    cond_out = {}
    if a.conditional:
        print("\nCONDITIONAL READOUT. SPIKE is the top 5% of each fold's own "
              "test-slice\nrealised vol, defined on the OUTCOME and used for "
              "reporting only -- no arm\nsees it. PRIMARY is a SAFETY claim: "
              "M1 must not be significantly WORSE\nthan M0 on the SPIKE "
              "subset. Calm gains cannot adopt anything.")
        for cn in ("spike", "calm"):
            print(f"\n--- {cn.upper()}")
            print(f"{'metric':>9} {'M0':>11} {'M1':>11} {'delta':>12} "
                  f"{'CI (corrected)':>28} {'folds':>8}")
            for met in METRICS:
                sgn = 1.0 if met in HIGHER_BETTER else -1.0
                b = [r[f"bar_M0__{cn}"].get(met) for r in rows
                     if met in r.get(f"bar_M0__{cn}", {})]
                for arm in arms:
                    aa = [r[f"bar_{arm}__{cn}"].get(met) for r in rows
                          if met in r.get(f"bar_{arm}__{cn}", {})]
                    if len(aa) != len(b) or len(aa) < 2:
                        print(f"{met:>9} {'':>11} {'':>11} "
                              f"{'NON-MEASUREMENT: fewer than 2 paired folds':>50}")
                        continue
                    a_s = np.asarray(aa, np.float64)
                    b_s = np.asarray(b, np.float64)
                    d = sgn * (a_s - b_s)
                    ci = mean_ci(d, alpha=alpha)
                    lo, hi = ci["ci95"]
                    cond_out[f"{met}_{arm}_{cn}"] = {
                        "M0": float(np.mean(b_s)), arm: float(np.mean(a_s)),
                        "delta": float(np.mean(d)),
                        "ci95": [float(lo), float(hi)],
                        "n_folds_better": int(np.sum(d > 0)),
                        "n_folds": int(len(d)),
                        "worse": bool(hi < 0), "better": bool(lo > 0)}
                    tag = ("WORSE" if hi < 0 else "better" if lo > 0 else "-")
                    print(f"{met:>9} {np.mean(b_s):11.6f} {np.mean(a_s):11.6f} "
                          f"{np.mean(d):+12.6f} [{lo:+12.6f}, {hi:+12.6f}] "
                          f"{int(np.sum(d>0)):>3}/{len(d)} {tag}")
        hurt = [k for k, v in cond_out.items()
                if k.endswith("_spike") and v["worse"]]
        # The pre-registered verdict, evaluated here rather than left to a
        # reader. A null on the spike subset is weak evidence of safety and the
        # registration said so in advance (R84), so the count of metrics whose
        # interval spans zero is printed beside the verdict rather than folded
        # into it.
        undec = [k for k, v in cond_out.items()
                 if k.endswith("_spike") and not v["worse"] and not v["better"]]
        print(f"\nSPIKE-SUBSET VERDICT: "
              f"{'REALLOCATION -- ' + ', '.join(hurt) if hurt else 'no metric significantly worse'}"
              f"   ({len(undec)} of {len(METRICS)} intervals span zero)")

    print("\npaired per-episode QLIKE (reported, NOT a pass condition):")
    L = block_len_for(PROD_H, sum(len(r["q_M0"]) for r in rows))
    q0 = np.concatenate([r["q_M0"] for r in rows])
    qci = {}
    for arm in arms:
        qa = np.concatenate([r[f"q_{arm}"] for r in rows])
        d = q0 - qa
        g = np.isfinite(d)
        ci = mean_ci(d[g], alpha=alpha, block_len=L)["ci95"]
        qci[arm] = [float(v) for v in ci]
        print(f"   {arm}: {100*np.nanmean(d)/np.nanmean(q0):+6.2f}%  "
              f"CI [{ci[0]:+.5f}, {ci[1]:+.5f}]")

    print(f"\n--- pre-registered rule ---")
    verdicts, rival_ci = {}, {}
    for arm in arms:
        wins = [m for m in METRICS
                if np.isfinite(bar[m][arm]) and better(m, bar[m][arm], bar[m]["M0"])]
        clears = [m for m in METRICS
                  if bar_ci.get(f"{m}_{arm}", {}).get("clears")]
        verdicts[arm] = {"metrics_better": wins, "n_better": len(wins),
                         "metrics_clearing": clears, "n_clearing": len(clears)}
        print(f"   {arm}: {len(wins)}/6 better by point estimate {wins}")
        print(f"        {len(clears)}/6 with an interval EXCLUDING ZERO {clears}"
              f"   <- this is what the rule asks for")
    role = ROLES[tuple(arms)]
    cand, mirr, rival = role["candidate"], role["mirror"], role["rival"]
    nc, nm = verdicts[cand]["n_better"], verdicts[mirr]["n_better"]
    print(f"\n   {cand} majority-better: {nc >= 4}   "
          f"MIRROR {mirr} majority-better: {nm >= 4}")
    n_clear = verdicts[cand]["n_clearing"]
    print(f"\n   ADOPTION per the registration needs a MAJORITY of six with "
          f"intervals\n   excluding zero: {cand} has {n_clear}/6 -> "
          f"{'MET' if n_clear >= 4 else 'NOT MET'}")
    if nc >= 4 and nm >= 4:
        print("   -> BOTH DIRECTIONS HELP: the gain is perturbing the atoms at "
              "all,\n      not correcting the dispersion. Same shape as "
              "P2-mean-level's level result.")
    elif nc >= 4 and nm < 4:
        print("   -> the correction's DIRECTION matters; width is a different "
              f"intervention\n      class from level. Candidate, pending the "
              f"{cand}-vs-{rival} separation below.")
    else:
        print("   -> the dispersion correction does not improve the battery.")

    # CANDIDATE AGAINST ITS RIVAL, PAIRED PER METRIC. The registration's third
    # condition -- the candidate must not be significantly WORSE than its rival
    # on any metric it clears -- names a comparison the first version of this
    # runner never computed, so the rule could not be evaluated with the
    # intervals the design produced. That is the second half of R84, and the
    # fix is to compute the contrast and to COUNT it in the family rather than
    # to reinterpret the rule.
    print(f"\n   {cand} vs {rival}, paired per metric "
          f"(the registration's third condition):")
    for met in METRICS:
        sgn = 1.0 if met in HIGHER_BETTER else -1.0
        a_s = np.asarray(per_fold[met][cand], np.float64)
        r_s = np.asarray(per_fold[met][rival], np.float64)
        if len(a_s) != len(r_s) or len(a_s) < 2:
            continue
        d = sgn * (a_s - r_s)
        lo, hi = mean_ci(d, alpha=alpha)["ci95"]
        rival_ci[met] = {"delta": float(np.mean(d)),
                         "ci95": [float(lo), float(hi)],
                         "worse": bool(hi < 0), "better": bool(lo > 0),
                         "n_folds_better": int(np.sum(d > 0))}
        tag = "WORSE" if hi < 0 else "better" if lo > 0 else "not separated"
        print(f"      {met:>9} {np.mean(d):+11.6f} "
              f"[{lo:+11.6f}, {hi:+11.6f}] {tag}")
    hurt = [m for m in verdicts[cand]["metrics_clearing"]
            if rival_ci.get(m, {}).get("worse")]
    print(f"   -> third condition: "
          f"{'FAILS on ' + ', '.join(hurt) if hurt else 'MET -- not significantly worse than ' + rival + ' on anything it clears'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"n_family": n_family, "alpha": alpha, "prod_H": PROD_H,
         "arms": list(arms), "deployable": bool(a.deployable),
         "conditional": bool(a.conditional), "cond": cond_out,
         "lambda_expand": k_expand,
         "k_const": k_const, "lambdas": {str(r["year"]): r["lam"] for r in rows},
         "barriers": bar, "barriers_per_fold": per_fold, "barrier_ci": bar_ci,
         "qlike_ci": qci, "verdicts": verdicts,
         "vs_rival": rival_ci, "roles": ROLES[tuple(arms)]}, indent=2,
        default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
