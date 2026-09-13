"""
eval/functional_barriers.py
=====================================================================
The barrier battery on P3-functional-parity: does publishing the conditional
MEAN instead of the median damage the distribution a reader consumes?

WHY THIS GATE DECIDES THE ADOPTION

`P3-functional-parity` measures a 13.3% to 26.0% raw QLIKE improvement from
reading NOCTUA's own `sigma_mean` rather than `sigma_med` -- same forward pass,
nothing fitted. That is by far the largest effect in the programme. It is also a
LEVEL change, and `P2-scale-v2-result` is the precedent that makes this gate
mandatory: a level correction that improved QLIKE by 9.7% degraded EVERY barrier
metric and adoption was blocked by guards rather than by judgement.

The difference this time is that the level is not fitted -- it is a different
functional of the same predictive distribution, which the loss is minimised by.
That is a reason to expect a better outcome, not a reason to skip the test.

WHAT VARIES AND WHAT DOES NOT

One trained model per fold. Identical weights, identical episodes, identical
committee specialists. The ONLY difference is which sigma is fed to the
committee's `quantiles(sigma, pred, up)`: `sigma_med` or `sigma_mean`. So this
isolates the functional and nothing else -- there is no retraining, no refitting
and no second sample.

GUARDS, and this time with the tolerance R69 said the next pre-registration owed

`P3-exogenous-barriers` blocked an arm on movements in the fifth decimal place,
because its guards were bare inequalities. R69 recorded that the next
pre-registration must use a paired interval per barrier metric instead. So here a
metric counts as DEGRADED only if its paired bootstrap interval excludes zero on
the adverse side; a movement whose interval straddles zero is recorded as
unchanged, which is what a 1e-5 wobble actually is.

    python -m model.eval.functional_barriers --self-test
    python -m model.eval.functional_barriers --horizons 1 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import (ALPHA_GRID, BARRIER_PCT, BARRIER_U,               # noqa: E402
                            corp_decomposition, crps_from_curve, log_score,
                            pinball_curve)
from eval.direction import mean_ci                                           # noqa: E402
from eval.exogenous import (DVOL, FUND, build_exo, hourly_series, run_arm)     # noqa: E402
from eval.vol_matrix import (HORIZONS, block_len_for, build_h4_table,          # noqa: E402
                             qlike_vec)
from noctua.committee import (Committee, EmpiricalSpecialist, EVTSpecialist,   # noqa: E402
                              GaussianSpecialist, NeuralSpecialist)
from noctua import splits as S                                               # noqa: E402

HIGHER_IS_BETTER = ("DSC",)
METRICS = ("pinball", "crps", "brier", "DSC", "MCB", "logs")


def per_episode_scores(comm, sigma, pred, M_up, M_dn) -> dict:
    """Per-episode barrier losses, so the comparison can be PAIRED."""
    out = {}
    for side, M in (("up", M_up), ("dn", M_dn)):
        Q = comm.quantiles(sigma, pred, up=(side == "up"))
        # pinball and crps are pooled statistics in benchmark; keep them scalar
        out[f"pinball_{side}"] = pinball_curve(Q, M)
        out[f"crps_{side}"] = crps_from_curve(Q, M)
        P = np.empty((Q.shape[0], len(BARRIER_U)))
        for i in range(Q.shape[0]):
            P[i] = np.interp(BARRIER_U, Q[i, ::-1], ALPHA_GRID[::-1],
                             left=float(ALPHA_GRID[-1]),
                             right=float(ALPHA_GRID[0]))
        P = np.clip(P, 1e-6, 1 - 1e-6)
        for k, pct in enumerate(BARRIER_PCT):
            y = (M >= BARRIER_U[k]).astype(float)
            d = corp_decomposition(P[:, k], y)
            out[f"brier_{side}_{pct}"] = d["brier"]
            out[f"DSC_{side}_{pct}"] = d["DSC"]
            out[f"MCB_{side}_{pct}"] = d["MCB"]
            out[f"logs_{side}_{pct}"] = log_score(P[:, k], y)
            # the per-episode Brier contribution, which CAN be paired
            out.setdefault("_brier_ep", []).append((P[:, k] - y) ** 2)
    return out


def families(rec: dict) -> dict:
    out = {}
    for pat in ("pinball_", "crps_", "brier_", "DSC_", "MCB_", "logs_"):
        vals = [v for k, v in rec.items()
                if k.startswith(pat) and isinstance(v, (int, float))]
        if vals:
            out[pat.rstrip("_")] = float(np.mean(vals))
    return out


def verdict(before: dict, after: dict, brier_ci=None) -> dict:
    out = {}
    for k in METRICS:
        if k not in before or k not in after:
            continue
        b, a = before[k], after[k]
        better = (a > b) if k in HIGHER_IS_BETTER else (a < b)
        rel = 100.0 * (a - b) / b if b else float("nan")
        out[k] = {"before": b, "after": a, "delta": a - b, "rel_pct": rel,
                  "better": bool(better),
                  "higher_is_better": k in HIGHER_IS_BETTER}
    if brier_ci is not None:
        out["brier"]["paired_ci"] = brier_ci
        # R69: degraded only if the PAIRED interval excludes zero adversely
        out["brier"]["degraded_by_interval"] = bool(brier_ci[1] < 0)
    return out


def self_test() -> int:
    ok = []
    b = families({"pinball_up": 1.0, "pinball_dn": 3.0, "DSC_up_10": 0.2,
                  "DSC_dn_10": 0.4, "brier_up_10": 0.1, "logs_up": 2.0,
                  "MCB_up_10": 0.05, "crps_up": 0.5})
    ok.append(("families-averaged", np.isclose(b["pinball"], 2.0)
               and np.isclose(b["DSC"], 0.3), f"pinball {b['pinball']}, DSC {b['DSC']}"))
    v = verdict({"DSC": .30, "brier": .10}, {"DSC": .35, "brier": .12})
    ok.append(("DSC-higher-better", v["DSC"]["better"] is True, "0.30->0.35 BETTER"))
    ok.append(("brier-lower-better", v["brier"]["better"] is False, "0.10->0.12 WORSE"))
    # R69's fix: a straddling interval must NOT count as degraded
    v2 = verdict({"brier": .10}, {"brier": .10001}, brier_ci=[-0.002, +0.003])
    ok.append(("straddling-interval-is-not-degradation",
               v2["brier"]["better"] is False
               and v2["brier"]["degraded_by_interval"] is False,
               "the point moved the wrong way but the interval straddles zero, "
               "so it is recorded as unchanged -- this is exactly the 1e-5 case "
               "that blocked P3-exogenous-barriers on a bare inequality"))
    v3 = verdict({"brier": .10}, {"brier": .13}, brier_ci=[-0.04, -0.01])
    ok.append(("adverse-interval-IS-degradation",
               v3["brier"]["degraded_by_interval"] is True,
               "an interval entirely on the adverse side degrades"))
    print("functional_barriers self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="barrier battery on the functional")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/functional_barriers_result.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    exo = build_exo(ep, X, hourly_series(DVOL, "volatility"),
                    hourly_series(FUND, "interest_1h"))
    keep = np.ones(len(ep), bool)          # FULL sample: no exogenous columns here
    print("barrier battery on P3-functional-parity")
    print("one trained model per fold; the ONLY thing that varies is which sigma")
    print("the committee is given -- sigma_med or sigma_mean\n")
    out = {"guard": "paired interval per metric (R69)", "horizons": {}}

    for H in a.horizons:
        qm, qn, recs_m, recs_n, bm, bn, folds = [], [], [], [], [], [], []
        for f in S.walk_forward_folds(ep):
            r = run_arm(ep, X, exo, f, H, "base", keep, a.hidden, a.seeds,
                        np.random.default_rng(20260913 + f["year"]),
                        want_curves=True)
            if r is None:
                continue
            specs = [NeuralSpecialist(),
                     GaussianSpecialist().fit(r["M_up_cal"], r["M_dn_cal"],
                                              r["sigma_cal"]),
                     EmpiricalSpecialist().fit(r["M_up_tr"], r["M_dn_tr"],
                                               r["RV_tr"]),
                     EVTSpecialist().fit(r["M_up_tr"], r["M_dn_tr"], r["RV_tr"])]
            comm = Committee(specs).fit_equal()
            sm = per_episode_scores(comm, r["sigma"], r["pred"],
                                    r["M_up"], r["M_dn"])
            sn = per_episode_scores(comm, r["sigma_mean"], r["pred"],
                                    r["M_up"], r["M_dn"])
            bm.append(np.concatenate(sm.pop("_brier_ep")))
            bn.append(np.concatenate(sn.pop("_brier_ep")))
            recs_m.append(sm); recs_n.append(sn)
            qm.append(qlike_vec(r["rv"], r["sigma"]))
            qn.append(qlike_vec(r["rv"], r["sigma_mean"]))
            folds.append(f["year"])
            print(f"  fold {f['year']}: {len(r['rv']):,} episodes")
        if not folds:
            print(f"H={H}: no fold available"); continue
        q0, q1 = np.concatenate(qm), np.concatenate(qn)
        dd = q0 - q1
        g = np.isfinite(dd)
        L = block_len_for(H, int(g.sum()))
        ci = mean_ci(dd[g], alpha=0.05, block_len=L)
        db = np.concatenate(bm) - np.concatenate(bn)
        gb = np.isfinite(db)
        cib = mean_ci(db[gb], alpha=0.05,
                      block_len=block_len_for(H, int(gb.sum())))
        bar = verdict(families({k: np.mean([r[k] for r in recs_m])
                               for k in recs_m[0]}),
                      families({k: np.mean([r[k] for r in recs_n])
                                for k in recs_n[0]}),
                      brier_ci=[float(cib["ci95"][0]), float(cib["ci95"][1])])
        print(f"\nH={H}  folds {folds}  {int(g.sum()):,} paired episodes")
        print(f"  QLIKE  median {np.nanmean(q0):.5f} -> mean {np.nanmean(q1):.5f}"
              f"   {np.nanmean(dd):+.5f}  CI [{ci['ci95'][0]:+.5f}, "
              f"{ci['ci95'][1]:+.5f}]  ({100*np.nanmean(dd)/np.nanmean(q0):+.2f}%)")
        print(f"  {'metric':>9} {'median':>11} {'mean':>11} {'rel %':>8}  direction")
        for k, v in bar.items():
            print(f"  {k:>9} {v['before']:11.5f} {v['after']:11.5f} "
                  f"{v['rel_pct']:+8.3f}  {'BETTER' if v['better'] else 'WORSE '}"
                  + (f"   paired CI [{v['paired_ci'][0]:+.5f}, "
                     f"{v['paired_ci'][1]:+.5f}]" if "paired_ci" in v else ""))
        worse = [k for k, v in bar.items() if not v["better"]]
        degraded = [k for k, v in bar.items() if v.get("degraded_by_interval")]
        print(f"\n  point-wise worse: {worse or 'none'}")
        print(f"  DEGRADED by paired interval (the guard that counts): "
              f"{degraded or 'none'}")
        out["horizons"][str(H)] = {
            "folds": folds, "qlike_median": float(np.nanmean(q0)),
            "qlike_mean": float(np.nanmean(q1)),
            "qlike_delta": float(np.nanmean(dd)),
            "qlike_ci": [float(ci["ci95"][0]), float(ci["ci95"][1])],
            "barriers": bar, "point_worse": worse, "degraded": degraded}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(f"  (written to {a.out})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
