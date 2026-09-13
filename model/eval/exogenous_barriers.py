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

WHAT THIS MEASURES, AND THROUGH WHAT

`benchmark.run_fold` is called rather than reimplemented, so the quantiles, the
committee and the barrier curves are the ones that ship. The only difference
between the two sides is `shape_cols`: 40 columns, or 40 plus D2's three. Both
sides are trained on the same episodes (the DVOL-era common sample) and scored on
the same slice, so the comparison is paired by construction.

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

from eval.benchmark import run_fold                                          # noqa: E402
from eval.direction import mean_ci                                           # noqa: E402
from eval.exogenous import (DVOL, EXO_COLS, FUND, build_exo, common_mask,     # noqa: E402
                            hourly_series)
from eval.vol_matrix import (UNDEFINED_AT_1W, block_len_for, build_h4_table,  # noqa: E402
                             qlike_vec)
from noctua import splits as S                                               # noqa: E402

ARM = "D2"
H_TARGET = 168
# DSC is a resolution term: higher is better. Everything else is a loss.
HIGHER_IS_BETTER = ("DSC",)
METRICS = ("pinball", "crps", "brier", "DSC", "MCB", "logs")


def barrier_cols(rows, model="noctua_v2") -> dict:
    """Pull the barrier metrics for one model out of run_fold's row list."""
    r = next((x for x in rows if x["model"] == model), None)
    if r is None:
        return {}
    out = {}
    for pat in ("pinball_", "crps_", "brier_", "DSC_", "MCB_", "logs_"):
        vals = [v for k, v in r.items()
                if k.startswith(pat) and isinstance(v, (int, float))]
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
    rows = [{"model": "noctua_v2", "pinball_up": 1.0, "pinball_dn": 3.0,
             "crps_up": 0.5, "DSC_up_10": 0.2, "DSC_dn_10": 0.4,
             "brier_up_10": 0.1, "logs_up": 2.0, "MCB_up_10": 0.05},
            {"model": "other", "pinball_up": 99.0}]
    b = barrier_cols(rows)
    ok.append(("metrics-are-averaged-per-family",
               np.isclose(b["pinball"], 2.0) and np.isclose(b["DSC"], 0.3),
               f"pinball mean(1,3)=2.0 -> {b['pinball']}, "
               f"DSC mean(.2,.4)=0.3 -> {b['DSC']}"))
    ok.append(("other-models-are-not-mixed-in", b["pinball"] != 99.0,
               "the row for `other` is ignored"))

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
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    Xu = X.copy()
    for c in EXO_COLS[ARM]:
        Xu[c] = exo[c].to_numpy()
    at_h = (ep.H == H_TARGET).to_numpy()
    print(f"barrier battery: {ARM} at H={H_TARGET}, through benchmark.run_fold")
    print(f"common sample {keep.sum():,} episodes; both sides trained on exactly "
          f"these and scored on the same slice\n")

    qb, qa, folds_used = [], [], []
    rows_b, rows_a = [], []
    for f in S.walk_forward_folds(ep):
        kw = dict(hidden=a.hidden, seeds=a.seeds,
                  train_filter=keep, prod_override=at_h & keep)
        rb = run_fold(ep, Xu[cols40], f, shape_cols=cols40, **kw)
        ra = run_fold(ep, Xu, f, shape_cols=cols40 + list(EXO_COLS[ARM]), **kw)
        if rb is None or ra is None:
            print(f"  fold {f['year']}: skipped")
            continue
        pb, pa = rb["per_episode"], ra["per_episode"]
        if len(pb["rv"]) != len(pa["rv"]):
            raise SystemExit(f"REFUSING: fold {f['year']} scored "
                             f"{len(pb['rv'])} vs {len(pa['rv'])} episodes -- "
                             f"the pairing is void")
        qb.append(qlike_vec(pb["rv"], pb["sigma_med"]))
        qa.append(qlike_vec(pa["rv"], pa["sigma_med"]))
        rows_b += rb["rows"]; rows_a += ra["rows"]
        folds_used.append(f["year"])
        print(f"  fold {f['year']}: {len(pb['rv']):,} episodes")

    if not folds_used:
        print("no fold produced both sides", file=sys.stderr)
        return 2
    q0, q1 = np.concatenate(qb), np.concatenate(qa)
    dd = q0 - q1
    g = np.isfinite(dd)
    L = block_len_for(H_TARGET, int(g.sum()))
    ci = mean_ci(dd[g], alpha=0.05, block_len=L)
    bar = verdict(barrier_cols(rows_b), barrier_cols(rows_a))
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
              f"{v['delta']:+11.5f}  {'BETTER' if v['better'] else 'WORSE ':<6} "
              f"({arrow})")
    print("\nguards:")
    for k, v in gd.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    blocked = [k for k, v in gd.items() if not v]
    print(f"\n{'ADOPTION BLOCKED by ' + ', '.join(blocked) if blocked else 'ALL GUARDS PASS'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"arm": ARM, "H": H_TARGET, "folds": folds_used,
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
