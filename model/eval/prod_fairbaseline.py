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
        acc.append({"year": f["year"], "n": len(idx),
                    "qlike": {k: qlike_vec(rv, np.asarray(v, np.float64))
                              for k, v in arms.items()},
                    "calib_qlike": sel})
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

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha, "block_len": L,
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
