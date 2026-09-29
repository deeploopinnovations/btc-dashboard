"""
eval/exogenous_barriers.py
=====================================================================
The barrier battery on the one arm of `P3-exogenous-dvol` that cleared.

WHY THIS IS NOT OPTIONAL

`P2-scale-v2-result` is the precedent and it is unambiguous: a correction that
improved QLIKE by 9.7% degraded EVERY barrier metric, and adoption was blocked by
guards rather than by judgement. QLIKE scores a variance forecast; what a reader
of this model actually consumes is the barrier curve and the safe level. A gain in
one bought by damage in the other is not an improvement, so the pre-registration
of `P3-exogenous-dvol` requires this battery for any arm that clears.

D2 at H=168 cleared: +0.00325 against base, CI [+0.00134, +0.00529]; its
width-matched control D2-shuf did not clear; and the deciding direct contrast, D2
against D2-shuf, was +0.00214 CI [+0.00069, +0.00360].

WHAT THIS MEASURES, AND WHY IT DOES NOT USE run_fold

The first version of this file called `benchmark.run_fold`, which is the driver
that produces the shipped barrier numbers. It crashed, and the crash was
informative: line 483 pins the committee's calibration slice to `ep.H == 19`, the
production horizon, which **does not exist in the h4 table at all** (horizons
1/6/24/168). `vol_matrix` and `intraday_basis` both build their own training loops
for exactly this reason, and that was available to be noticed before writing the
call.

So the barrier curves here come from the SAME fitted models that produced the
QLIKE number -- `exogenous.run_arm(..., want_curves=True)` returns them -- and the
committee is fitted on the CALIB slice AT H=168 rather than on a production slice
that the table does not contain. The specialist set and the equal-weight
combination mirror `benchmark.run_fold`'s, and `pinball_curve`, `crps_from_curve`,
`corp_decomposition` and `log_score` are imported from `benchmark` rather than
reimplemented, so the metric definitions cannot drift.

This is therefore the model's own committee calibrated at the horizon under test,
not the production committee. That is the correct object for the question --
"does adding these columns damage the distribution this model emits at H=168?" --
and it is labelled rather than passed off as the shipped number.

GUARDS, FIXED HERE BEFORE THE RUN

Adoption requires ALL of:
  * `qlike_better`        -- the gain survives this path too, not just the
                             lighter path `exogenous.py` used;
  * `dsc_not_worse`       -- DSC is a resolution term, so HIGHER is better;
  * `brier_not_worse`, `pinball_not_worse`, `crps_not_worse`, `logs_not_worse`
                          -- all losses, so LOWER is better.
A single FAIL blocks adoption. That is the same shape of rule that blocked
P2-scale-v2, and it is stated before the numbers exist so it cannot be relaxed
after them.

    python -m model.eval.exogenous_barriers --self-test
    python -m model.eval.exogenous_barriers
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
from eval.exogenous import (DVOL, FUND, build_exo, common_mask,               # noqa: E402
                            hourly_series, run_arm)
from eval.vol_matrix import block_len_for, build_h4_table, qlike_vec          # noqa: E402
from noctua.committee import (Committee, EmpiricalSpecialist, EVTSpecialist,  # noqa: E402
                              GaussianSpecialist, NeuralSpecialist)
from noctua import splits as S                                               # noqa: E402

ARM = "D2"
H_TARGET = 168
# DSC is a resolution term: higher is better. Everything else is a loss.
HIGHER_IS_BETTER = ("DSC",)
METRICS = ("pinball", "crps", "brier", "DSC", "MCB", "logs")


def barrier_cols(recs) -> dict:
    """Average each metric family over the per-fold records."""
    if isinstance(recs, dict):
        recs = [recs]
    r = {}
    for rec in recs:
        for k, v in rec.items():
            r.setdefault(k, []).append(v)
    r = {k: float(np.mean(v)) for k, v in r.items()}
    out = {}
    for pat in ("pinball_", "crps_", "brier_", "DSC_", "MCB_", "logs_"):
        vals = [v for k, v in r.items() if k.startswith(pat)]
        if vals:
            out[pat.rstrip("_")] = float(np.mean(vals))
    return out


def verdict(before: dict, after: dict) -> dict:
    """Per-metric direction, with DSC's sign handled explicitly."""
    out = {}
    for k in METRICS:
        if k not in before or k not in after:
            continue
        b, a = before[k], after[k]
        better = (a > b) if k in HIGHER_IS_BETTER else (a < b)
        out[k] = {"before": b, "after": a, "delta": a - b,
                  "better": bool(better),
                  "higher_is_better": k in HIGHER_IS_BETTER}
    return out


def guards(qlike_delta: float, bar: dict) -> dict:
    """All must pass. One FAIL blocks adoption."""
    g = {"qlike_better": bool(qlike_delta > 0)}
    for k in METRICS:
        if k in bar:
            g[f"{k.lower()}_not_worse"] = bool(bar[k]["better"])
    return g


def self_test() -> int:
    ok = []
    rows = [{"pinball_up": 1.0, "pinball_dn": 3.0,
             "crps_up": 0.5, "DSC_up_10": 0.2, "DSC_dn_10": 0.4,
             "brier_up_10": 0.1, "logs_up": 2.0, "MCB_up_10": 0.05}]
    b = barrier_cols(rows)
    ok.append(("metrics-are-averaged-per-family",
               np.isclose(b["pinball"], 2.0) and np.isclose(b["DSC"], 0.3),
               f"pinball mean(1,3)=2.0 -> {b['pinball']}, "
               f"DSC mean(.2,.4)=0.3 -> {b['DSC']}"))
    ok.append(("families-are-kept-separate",
               set(b) == {"pinball", "crps", "DSC", "brier", "logs", "MCB"},
               f"{sorted(b)}"))

    # DSC must be read in the opposite direction from the losses, and getting
    # that backwards is how a degradation reads as an improvement
    v = verdict({"DSC": 0.30, "brier": 0.10},
                {"DSC": 0.35, "brier": 0.12})
    ok.append(("DSC-higher-is-better", v["DSC"]["better"] is True,
               "DSC 0.30 -> 0.35 counts as BETTER"))
    ok.append(("brier-lower-is-better", v["brier"]["better"] is False,
               "brier 0.10 -> 0.12 counts as WORSE"))
    v2 = verdict({"DSC": 0.30}, {"DSC": 0.25})
    ok.append(("DSC-decrease-is-worse", v2["DSC"]["better"] is False,
               "DSC 0.30 -> 0.25 counts as WORSE"))

    # the guard set must be blockable, which is the whole point
    g_bad = guards(+0.01, verdict({"DSC": .3, "brier": .1},
                                  {"DSC": .3, "brier": .12}))
    ok.append(("one-degraded-metric-blocks",
               g_bad["qlike_better"] and not g_bad["brier_not_worse"],
               "QLIKE improves and brier degrades -> blocked"))
    g_ok = guards(+0.01, verdict({"DSC": .3, "brier": .12},
                                 {"DSC": .35, "brier": .10}))
    ok.append(("all-improved-passes", all(g_ok.values()),
               "every metric improves -> all guards pass"))
    g_q = guards(-0.01, verdict({"brier": .12}, {"brier": .10}))
    ok.append(("a-qlike-loss-blocks-on-its-own",
               not g_q["qlike_better"],
               "barriers can improve and a QLIKE loss still blocks"))

    print("exogenous_barriers self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def score_side(comm, sigma, pred, M, side: str) -> dict:
    """pinball, CRPS and the CORP terms for one side, from one committee."""
    up = side == "up"
    Q = comm.quantiles(sigma, pred, up=up)
    rec = {f"pinball_{side}": pinball_curve(Q, M),
           f"crps_{side}": crps_from_curve(Q, M)}
    P = np.empty((Q.shape[0], len(BARRIER_U)))
    for i in range(Q.shape[0]):
        P[i] = np.interp(BARRIER_U, Q[i, ::-1], ALPHA_GRID[::-1],
                         left=float(ALPHA_GRID[-1]), right=float(ALPHA_GRID[0]))
    P = np.clip(P, 1e-6, 1 - 1e-6)
    for k, pct in enumerate(BARRIER_PCT):
        out = (M >= BARRIER_U[k]).astype(float)
        d = corp_decomposition(P[:, k], out)
        rec[f"brier_{side}_{pct}"] = d["brier"]
        rec[f"DSC_{side}_{pct}"] = d["DSC"]
        rec[f"MCB_{side}_{pct}"] = d["MCB"]
        rec[f"logs_{side}_{pct}"] = log_score(P[:, k], out)
    return rec


def fold_barriers(r: dict) -> dict:
    """Fit the committee on this fold's CALIB at H=168, score its TEST."""
    specs = [
        NeuralSpecialist(),
        GaussianSpecialist().fit(r["M_up_cal"], r["M_dn_cal"], r["sigma_cal"]),
        EmpiricalSpecialist().fit(r["M_up_tr"], r["M_dn_tr"], r["RV_tr"]),
        EVTSpecialist().fit(r["M_up_tr"], r["M_dn_tr"], r["RV_tr"]),
    ]
    comm = Committee(specs).fit_equal()
    rec = {}
    for side, M in (("up", r["M_up"]), ("dn", r["M_dn"])):
        rec.update(score_side(comm, r["sigma"], r["pred"], M, side))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="barrier battery on D2 at H=168")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/exogenous_barriers_result.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = build_h4_table(a.artifacts)
    exo = build_exo(ep, X, hourly_series(DVOL, "volatility"),
                    hourly_series(FUND, "interest_1h"))
    keep = common_mask(exo)
    print(f"barrier battery: {ARM} against base at H={H_TARGET}")
    print(f"committee fitted on each fold's CALIB slice AT H={H_TARGET} "
          f"(run_fold pins its own to H=19, which the h4 table lacks)")
    print(f"common sample {keep.sum():,} episodes; identical for both sides\n")

    qb, qa_, folds_used, recs_b, recs_a = [], [], [], [], []
    for f in S.walk_forward_folds(ep):
        rng = np.random.default_rng(20260913 + f["year"])
        rb = run_arm(ep, X, exo, f, H_TARGET, "base", keep, a.hidden, a.seeds,
                     rng, want_curves=True)
        ra = run_arm(ep, X, exo, f, H_TARGET, ARM, keep, a.hidden, a.seeds,
                     rng, want_curves=True)
        if rb is None or ra is None:
            print(f"  fold {f['year']}: skipped")
            continue
        if not np.array_equal(rb["anchor"], ra["anchor"]):
            raise SystemExit(f"REFUSING: fold {f['year']} scored different "
                             f"episodes on the two sides -- pairing void")
        qb.append(qlike_vec(rb["rv"], rb["sigma"]))
        qa_.append(qlike_vec(ra["rv"], ra["sigma"]))
        recs_b.append(fold_barriers(rb))
        recs_a.append(fold_barriers(ra))
        folds_used.append(f["year"])
        print(f"  fold {f['year']}: {len(rb['rv']):,} episodes")

    if not folds_used:
        print("no fold produced both sides", file=sys.stderr)
        return 2
    q0, q1 = np.concatenate(qb), np.concatenate(qa_)
    dd = q0 - q1
    g = np.isfinite(dd)
    L = block_len_for(H_TARGET, int(g.sum()))
    ci = mean_ci(dd[g], alpha=0.05, block_len=L)
    bar = verdict(barrier_cols(recs_b), barrier_cols(recs_a))
    gd = guards(float(np.nanmean(dd)), bar)

    print(f"\nfolds {folds_used}   {int(g.sum()):,} paired episodes   "
          f"blocks of {L}")
    print(f"QLIKE  base {np.nanmean(q0):.5f} -> {ARM} {np.nanmean(q1):.5f}   "
          f"delta {np.nanmean(dd):+.5f}  CI [{ci['ci95'][0]:+.5f}, "
          f"{ci['ci95'][1]:+.5f}]")
    print(f"\n{'metric':>9} {'base':>11} {ARM:>11} {'delta':>11}  direction")
    for k, v in bar.items():
        arrow = "higher better" if v["higher_is_better"] else "lower better"
        print(f"{k:>9} {v['before']:11.5f} {v['after']:11.5f} "
              f"{v['delta']:+11.5f}  {'BETTER' if v['better'] else 'WORSE '} "
              f"({arrow})")
    print("\nguards:")
    for k, v in gd.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    blocked = [k for k, v in gd.items() if not v]
    print(f"\n{'ADOPTION BLOCKED by ' + ', '.join(blocked) if blocked else 'ALL GUARDS PASS'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"arm": ARM, "H": H_TARGET, "folds": folds_used,
         "committee": "fitted on calib at H=168; NOT the production committee",
         "n_paired": int(g.sum()), "block_len": int(L),
         "qlike_base": float(np.nanmean(q0)),
         "qlike_arm": float(np.nanmean(q1)),
         "qlike_delta": float(np.nanmean(dd)),
         "qlike_ci": [float(ci["ci95"][0]), float(ci["ci95"][1])],
         "barriers": bar, "guards": gd, "blocked_by": blocked},
        indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
