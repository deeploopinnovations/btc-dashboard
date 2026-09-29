"""
eval/prod_fairbaseline.py
=====================================================================
E-prod-fairbaseline: does the project's own headline survive a horizon-aware
baseline?

WHY THIS HAD TO BE RUN

`vol-matrix-fair-result` established that a Log-HAR fitted ONCE on a pooled
multi-horizon training sample is a straw man at horizons far from the pooled
centre. Refitting the same family per horizon turned a +0.14462 NOCTUA win at
H = 168 into a -0.02362 loss, and the pooled fit was costing the baseline a
factor of 2.06 there.

`eval/benchmark.py` fits its baseline once per fold on the pooled training
sample spanning H in {6,12,19,24}, and that baseline is what the project's
headline advantage over Log-HAR is measured against. A correction applied only
to the result one dislikes is not a correction, so the mechanism gets pointed
at the headline.

A CORRECTION TO MY OWN FIRST FRAMING OF THIS FILE, kept rather than quietly
edited. I initially wrote that the headline was measured against a
horizon-BLIND baseline, by analogy with the matrix. It is not.
`benchmark.run_fold` builds its comparison from `bl["log_har_cal"]` --
har_1d/5d/22d plus cal_H and cal_weekend_frac -- and merely STORES it under the
key `"log_har"`. The incumbent baseline already has a horizon term. What is
still open is whether fitting it POOLED across {6,12,19,24} rather than at
H = 19 costs it anything, which is a much narrower question than the one the
matrix answered, and it is the question this file actually measures.

WHAT IS HELD FIXED

Everything except the baseline fit. Same episodes (`episodes.parquet`, the
production slice: H = 19, anchor 17:00 UTC), same six walk-forward folds, same
seeds, same causal Stage-B sigma reference, same QLIKE, same paired
per-episode primary with the 2H block length. NOCTUA's forecasts are produced
by `benchmark.run_fold`, unmodified and reused rather than reimplemented, so
the NOCTUA side of the comparison is identical to the shipped benchmark's by
construction.

THE ARMS

    log_har_cal_pooled THE ACTUAL INCUMBENT BASELINE. `benchmark.run_fold`
                       computes `vol["log_har"]` from `bl["log_har_cal"]`, not
                       from `bl["log_har"]` -- the artifact key is misleading
                       and cost me a wrong claim once already (see below). So
                       the incumbent is har_1d/5d/22d PLUS cal_H and
                       cal_weekend_frac, fitted on the pooled sample. It is
                       horizon-AWARE, and the headline is therefore NOT the
                       straw-man comparison I first assumed.
    log_har_pooled     har_1d/5d/22d only, pooled. Reported for contrast; it
                       is NOT what the headline is measured against.
    log_har            the same columns, refitted on H = 19 training episodes.
    har_short          + har_1h and har_6h, refitted on H = 19.
    log_har_cal        + cal_H and cal_weekend_frac, refitted on H = 19.
                       At a single horizon cal_H is constant, so this reduces
                       to log_har plus the weekend fraction.
    persistence        exp(har_1d) * sqrt(H). Never episodes.RV1, which looks
                       forward from the anchor.

The baseline to beat is the best of these by CALIB QLIKE, chosen on the
calibration slice and never on test. `log_har_pooled` is reported but is NOT
eligible to be chosen -- picking the horizon-blind fit as the bar is the
confound this file exists to remove.

    python -m model.eval.prod_fairbaseline
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

from eval.benchmark import run_fold                                      # noqa: E402
from eval.direction import mean_ci                                       # noqa: E402
from eval.levers import causal_spike_flag                                # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                     # noqa: E402
from noctua import baselines as B                                        # noqa: E402
from noctua import splits as S                                           # noqa: E402
from noctua.train import load_all                                        # noqa: E402

# The volatility-matrix family had 4 rows; this contrast joins it as the 5th.
N_FAMILY = 5
FAIR_ARMS = ("log_har", "har_short", "log_har_cal")
# scored, reported, and never eligible to be chosen as the bar
POOLED_ARMS = ("log_har", "log_har_cal")
# the one the published headline is actually measured against
INCUMBENT = "log_har_cal_pooled"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E-prod-fairbaseline")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/prod_fairbaseline.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    spike_all = causal_spike_flag(ep)
    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    Hall = ep["H"].to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)

    # the same causal Stage-B reference benchmark.main uses, refit per fold
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(train_mask):
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    print(f"production slice: H=19, anchor 17:00 UTC, {len(folds)} folds, "
          f"seeds={a.seeds}")
    print(f"Bonferroni: volatility family grows from 4 rows to {N_FAMILY} -> "
          f"{100*(1-alpha):.2f}% intervals\n")

    acc = []
    for f in folds:
        t0 = time.time()
        r = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn)
        if r is None:
            print(f"  fold {f['year']}: skipped"); continue
        pe = r["per_episode"]
        idx = pe["test_idx"]
        Ht = pe["H"]
        sq = np.sqrt(Ht)
        rv = pe["rv"]

        fin = np.isfinite(X.to_numpy(np.float64)).all(1)
        m_tr_pool = f["train"] & fin
        # PER-HORIZON: the production slice is H=19, so "this horizon" is the
        # H=19 training episodes and nothing else.
        m_tr_h = m_tr_pool & (ep.H == 19).to_numpy()
        bl_pool = B.fit_vol_baselines(X[m_tr_pool], yall[m_tr_pool],
                                      S.sample_weights(ep, m_tr_pool))
        bl_h = B.fit_vol_baselines(X[m_tr_h], yall[m_tr_h],
                                   S.sample_weights(ep, m_tr_h))

        Xte = X.iloc[idx]
        arms = {
            "noctua": pe["sigma_med"],
            # REPORTED, NOT A FAMILY MEMBER. N_FAMILY stays at 5 and no verdict
            # is issued for this arm: the family was fixed before the functional
            # question existed, and admitting an arm to it afterwards is
            # enlarging a family until something clears. It is here because the
            # production headline -- the number that matters most, since this is
            # the slice that is actually served -- had never been scored on the
            # functional that is now actually served. Omitting it would leave
            # that gap invisible.
            "noctua_mean": pe["sigma_mean"],
            "persistence": np.maximum(np.exp(Xte["har_1d"].to_numpy()) * sq, 1e-12),
        }
        for k in POOLED_ARMS:
            arms[k + "_pooled"] = np.exp(bl_pool[k].predict(Xte)) * sq
        for k in FAIR_ARMS:
            arms[k] = np.exp(bl_h[k].predict(Xte)) * sq

        # baseline selection on CALIB, never on test
        m_va = f["calib"] & fin & (ep.H == 19).to_numpy()
        sel = {}
        if m_va.sum() >= 50:
            Xv = X[m_va]
            sqv = np.sqrt(Hall[m_va])
            rvv = ep.RV.to_numpy()[m_va]
            sel["persistence"] = float(np.nanmean(qlike_vec(
                rvv, np.exp(Xv["har_1d"].to_numpy()) * sqv)))
            for k in FAIR_ARMS:
                sel[k] = float(np.nanmean(qlike_vec(
                    rvv, np.exp(bl_h[k].predict(Xv)) * sqv)))
        # P3-spike-ratio-refresh: MODEL_CARD 5.3's spike/calm RV/sigma ratios
        # were measured against sigma_med, and serving now reports sigma_mean.
        # Both functionals are recorded here, on the SAME episodes and the same
        # causal spike flag, so the refresh costs no extra model run. The ratio
        # reported is mean(RV^2/sigma^2), which is the quantity QLIKE is
        # calibrated by -- 1.0 is a correctly-levelled conditional mean.
        sp = spike_all[idx]
        ratios = {}
        for k in ("noctua", "noctua_mean"):
            if k not in arms:
                continue
            sg = np.asarray(arms[k], np.float64)
            r2 = rv ** 2 / np.maximum(sg, 1e-12) ** 2       # QLIKE's calibration
            r1 = rv / np.maximum(sg, 1e-12)                 # the card's ratio
            # BOTH, because they are not the same number and the
            # pre-registration compares against the card. MODEL_CARD 5.3 quotes
            # a MEDIAN of RV/sigma; QLIKE is calibrated by the MEAN of its
            # SQUARE. A first run reported only the latter and got 3.14 against
            # the card's 1.45, which looked like a discrepancy in the data and
            # was a discrepancy in the statistic -- the pre-registered
            # reproduce-the-median check caught it, which is what it was for.
            ratios[k] = {
                "spike": float(np.nanmean(r2[sp])) if sp.any() else float("nan"),
                "calm": float(np.nanmean(r2[~sp])) if (~sp).any() else float("nan"),
                "all": float(np.nanmean(r2)),
                "med_spike": float(np.nanmedian(r1[sp])) if sp.any() else float("nan"),
                "med_calm": float(np.nanmedian(r1[~sp])) if (~sp).any() else float("nan"),
                "med_all": float(np.nanmedian(r1)),
                "n_spike": int(sp.sum()), "n_calm": int((~sp).sum())}
        acc.append({"year": f["year"], "n": len(idx),
                    "qlike": {k: qlike_vec(rv, np.asarray(v, np.float64))
                              for k, v in arms.items()},
                    "ratios": ratios, "calib_qlike": sel})
        print(f"  fold {f['year']}  n={len(idx)}  ({time.time()-t0:.0f}s)", flush=True)

    if not acc:
        print("no usable folds"); return 1

    cal = {}
    for k in ("persistence",) + FAIR_ARMS:
        vs = [r["calib_qlike"][k] for r in acc if k in r["calib_qlike"]]
        if vs:
            cal[k] = float(np.mean(vs))
    best = min(cal, key=cal.get)

    arms = sorted(set.intersection(*[set(r["qlike"]) for r in acc]))
    pooled = {k: np.concatenate([r["qlike"][k] for r in acc]) for k in arms}
    per_fold = {k: [float(np.nanmean(r["qlike"][k])) for r in acc] for k in arms}
    n = len(pooled[best])
    L = block_len_for(19, n)

    print("\n" + "=" * 92)
    print(f"PRODUCTION SLICE  {n:,} test episodes over {len(acc)} folds")
    print(f"best baseline by CALIB QLIKE (never test): {best}    "
          + "  ".join(f"{k} {v:.4f}" for k, v in sorted(cal.items(), key=lambda kv: kv[1])))
    print("=" * 92)
    print(f"{'arm':>16} {'QLIKE':>9} {'vs best':>10} {'rel %':>8} {'worst fold':>11}   "
          f"paired per-episode CI vs {best} (blocks of {L})")
    out_arms = {}
    order = ["noctua"] + [k for k in arms if k != "noctua"]
    for k in order:
        v = pooled[k]
        d = pooled[best] - v
        good = np.isfinite(d)
        ci = mean_ci(d[good], alpha=alpha, block_len=L) if k != best else None
        cis = "— (is the baseline)" if ci is None else \
            f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
        rel = 100.0 * np.nanmean(d) / np.nanmean(pooled[best])
        print(f"{k:>16} {np.nanmean(v):9.5f} {np.nanmean(d):+10.5f} {rel:+8.2f} "
              f"{max(per_fold[k]):11.5f}   {cis}")
        out_arms[k] = {"qlike": float(np.nanmean(v)),
                       "delta_vs_best": float(np.nanmean(d)),
                       "rel_pct_vs_best": float(rel),
                       "worst_fold": float(max(per_fold[k])),
                       "per_fold": per_fold[k],
                       "paired_ci": None if ci is None else list(ci["ci95"])}

    # the incumbent comparison, stated in the incumbent's own terms
    inc = pooled[INCUMBENT]
    d_inc = inc - pooled["noctua"]
    ci_inc = mean_ci(d_inc[np.isfinite(d_inc)], alpha=alpha, block_len=L)
    rel_inc = 100.0 * np.nanmean(d_inc) / np.nanmean(inc)
    print(f"\n   THE INCUMBENT CLAIM, restated: NOCTUA vs {INCUMBENT} "
          f"{np.nanmean(d_inc):+.5f} ({rel_inc:+.2f}%)  "
          f"CI [{ci_inc['ci95'][0]:+.5f}, {ci_inc['ci95'][1]:+.5f}]")

    d_best = pooled[best] - pooled["noctua"]
    ci95 = mean_ci(d_best[np.isfinite(d_best)], alpha=0.05, block_len=L)
    print(f"   (unadjusted 95% interval on the same contrast, for context only "
          f"and NOT the rule: [{ci95['ci95'][0]:+.5f}, {ci95['ci95'][1]:+.5f}])")

    ci_n = out_arms["noctua"]["paired_ci"]
    verdict = "CLEARS" if (ci_n and ci_n[0] > 0.0) else "DOES NOT CLEAR"
    print(f"\n   --- pre-registered rule ---")
    print(f"   NOCTUA vs {best} (horizon-aware, chosen on calib): {verdict} zero favourably")
    print(f"   headline against the POOLED incumbent {INCUMBENT}: "
          f"{rel_inc:+.2f}%   against the horizon-AWARE best baseline: "
          f"{out_arms['noctua']['rel_pct_vs_best']:+.2f}%")

    # --- THE FUNCTIONAL CONTRAST ON THE SERVED PIPELINE.
    # P3-functional-adopt switched serving to sigma_mean on evidence from the
    # teacher zoo, where noctua_v1 is the RAW network. This slice is BLENDED
    # (blend_w = 0.25, applied inside the serving runtime too), and a blend
    # that pulls the median toward Log-HAR already does most of the level
    # correction the mean functional would do -- so applying both can
    # double-correct. A uniform log-shift cannot change the mean/median RATIO,
    # but it changes the level that ratio is applied to, which is exactly why
    # the zoo result does not transfer here by itself. This is the paired
    # contrast that decides it, on identical episodes.
    if "noctua" in pooled and "noctua_mean" in pooled:
        d_fn = pooled["noctua"] - pooled["noctua_mean"]      # >0 favours mean
        g_fn = np.isfinite(d_fn)
        ci_fn = mean_ci(d_fn[g_fn], alpha=alpha, block_len=L)
        rel_fn = 100.0 * np.nanmean(d_fn) / np.nanmean(pooled["noctua"])
        lo, hi = ci_fn["ci95"]
        if hi < 0:
            call = "MEDIAN is better on the served pipeline"
        elif lo > 0:
            call = "MEAN is better on the served pipeline"
        else:
            call = "NOT SEPARATED on this slice"
        print(f"\n   --- functional contrast, served (blended) pipeline ---")
        print(f"   noctua(median) {np.nanmean(pooled['noctua']):.5f} vs "
              f"noctua_mean {np.nanmean(pooled['noctua_mean']):.5f}")
        print(f"   delta {np.nanmean(d_fn):+.5f} ({rel_fn:+.2f}%)  "
              f"CI [{lo:+.5f}, {hi:+.5f}]  ->  {call}")
        print(f"   folds favouring mean: "
              f"{sum(1 for a_, b_ in zip(per_fold['noctua'], per_fold['noctua_mean']) if b_ < a_)}"
              f" of {len(per_fold['noctua'])}")

    # --- P3-spike-ratio-refresh, reported beside the headline
    rr = {}
    for k in ("noctua", "noctua_mean"):
        vs = [r["ratios"][k] for r in acc if k in r.get("ratios", {})]
        if not vs:
            continue
        rr[k] = {q: float(np.nanmean([v[q] for v in vs]))
                 for q in ("spike", "calm", "all",
                           "med_spike", "med_calm", "med_all")}
        rr[k]["n_spike"] = int(sum(v["n_spike"] for v in vs))
        rr[k]["n_calm"] = int(sum(v["n_calm"] for v in vs))
    if rr:
        print("\n   --- P3-spike-ratio-refresh: calibration ratio "
              "mean(RV^2/sigma^2), 1.0 is correctly levelled ---")
        print(f"   {'functional':>12} | {'mean(RV^2/s^2) -- QLIKE':>26} | "
              f"{'median(RV/s) -- the card':>26}")
        print(f"   {'':>12} | {'spike':>8} {'calm':>8} {'all':>8} | "
              f"{'spike':>8} {'calm':>8} {'all':>8}")
        for k, v in rr.items():
            print(f"   {k:>12} | {v['spike']:8.4f} {v['calm']:8.4f} {v['all']:8.4f}"
                  f" | {v['med_spike']:8.4f} {v['med_calm']:8.4f} "
                  f"{v['med_all']:8.4f}")
        print("   MODEL_CARD 5.3 quoted 1.453 spike / 0.964 calm against "
              "sigma_med.")
        print("   The median column above is the check that this harness "
              "measures the")
        print("   same thing 7a did; a mismatch there voids the refresh rather "
              "than the card.")

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha, "block_len": L,
        "spike_ratio_refresh": rr,
        "functional_contrast": (
            {"delta": float(np.nanmean(pooled["noctua"] - pooled["noctua_mean"])),
             "ci95": [float(v) for v in mean_ci(
                 (pooled["noctua"] - pooled["noctua_mean"])[
                     np.isfinite(pooled["noctua"] - pooled["noctua_mean"])],
                 alpha=alpha, block_len=L)["ci95"]]}
            if ("noctua" in pooled and "noctua_mean" in pooled) else None),
        "seeds": a.seeds, "n_test": n, "years": [r["year"] for r in acc],
        "best_baseline": best, "calib_qlike": cal, "arms": out_arms,
        "incumbent_claim": {"delta": float(np.nanmean(d_inc)),
                            "rel_pct": float(rel_inc),
                            "ci": list(ci_inc["ci95"])},
        "verdict": verdict,
        "unadjusted_ci95": list(ci95["ci95"]),
    }, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
