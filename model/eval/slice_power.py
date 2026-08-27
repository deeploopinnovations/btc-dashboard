"""
eval/slice_power.py
=====================================================================
E-power: how much resolution does the production evaluation slice give up?

THE QUESTION, AND WHY IT BLOCKS THE QUEUE

`benchmark.run_fold` scores `fold["test"] & finite & production_mask`, and
`splits.production_mask` is `(H == 19) & (anchor_hour == 17)`. Every walk-forward
fold is therefore decided on ~365 episodes, ~20 of them spike-flagged: six folds
is ~2,190 test and ~119 spike episodes against a 510,496-episode population.

That is the right slice for a DEPLOYMENT decision -- 17:00/19h is the trade --
and a narrow one for deciding whether an internal effect exists. BENCHMARK.md 23
records two pre-registered tests of the ensemble weight that both returned
intervals too wide to decide anything, and on the second
`pitfalls.check_not_a_coin_flip` said so outright: the estimate was one
eighteenth of its own standard error. Running more experiments at an unmeasured
resolution produces more non-measurements, so the resolution gets measured
first.

WHAT THIS READS, AND WHY IT COMPUTES NOTHING ITSELF

Two runs of `anchor_freshness`, identical but for the scoring slice:

    python -m model.eval.anchor_freshness                  # production
    python -m model.eval.anchor_freshness --all-hours      # 24 anchor hours

Same treatment, same arms, same seeds, same folds, same rule, same code path.
This file only reads their stored per-fold records and reduces them. It fits
nothing and trains nothing, so it cannot introduce an effect that the two runs
did not already contain.

THE PRE-REGISTERED PRIMARY, AND THE AMENDMENT TO IT

BENCHMARK.md 22 made the primary the ratio of block-bootstrap spike-QLIKE CI
widths, production over 24-hour, against the sqrt(24) = 4.90x that independent
episodes would give.

That quantity is NOT scale-free, and the amendment recorded in 22 -- before any
width was computed -- says so. The two slices do not share a baseline: control
spike QLIKE runs around 3.24 / 1.35 / 3.65 on the production slice against
1.14 / 0.60 / 1.12 on the wide slice for the same fold years. An interval around
a quantity three times larger is wider for reasons that have nothing to do with
precision.

So BOTH are reported: the absolute ratio because it was pre-registered and
changing a rule quietly is worse than stating a flawed one plainly, and the
scale-normalised ratio -- each width over its own slice's control baseline -- as
the interpretable one.

THREE THINGS THAT ARE DIFFERENT QUESTIONS, KEPT APART

  RESOLUTION  how wide the interval is. What this experiment is about.
  EFFECT      how big the treatment delta is. A change here is NOT a change in
              precision, and conflating them is the trap this file exists to
              avoid.
  VERDICT     whether the pre-registered rule still rejects. If it FLIPS, every
              prior verdict resting on a marginal interval needs re-running.

WHAT A SMALL TIGHTENING WOULD MEAN

The prediction recorded before the run (ledger `E-power-prediction`): the
tightening will be materially less than 4.90x, because fold-level delta variance
has two components and only one of them shrinks. Within-fold sampling error
falls as more episodes are scored; between-fold treatment HETEROGENEITY -- 2021
and 2026 are different regimes and the treatment genuinely does different things
in them -- does not fall at all.

If the observed tightening is near 1x, the binding constraint is the number of
YEARS, not episodes, and six folds is six effectively independent observations
however many episodes each contains. The remedy would then not be a wider slice
but a different ESTIMATOR -- the paired per-episode bootstrap that
`anchor_freshness` now also reports.

    python -m model.eval.slice_power
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = ("qlike_spike", "qlike_calm", "qlike_pooled", "tail_mcb")
N_HOURS = 24


def width(ci) -> float:
    return float(ci[1] - ci[0])


def load(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(
            f"REFUSING: {path} not found. E-power needs BOTH runs:\n"
            f"  python -m model.eval.anchor_freshness\n"
            f"  python -m model.eval.anchor_freshness --all-hours")
    d = json.loads(path.read_text())
    d["_label"] = label
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E-power: resolution of the eval slice")
    ap.add_argument("--production", type=Path,
                    default=Path("model/artifacts/anchor_freshness.json"))
    ap.add_argument("--all-hours", type=Path,
                    default=Path("model/artifacts/anchor_freshness_allhours.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/slice_power.json"))
    a = ap.parse_args(argv)

    P = load(a.production, "production (17:00 only)")
    W = load(a.all_hours, "all 24 anchor hours")

    n_p = int(np.mean([f["control"]["n_test"] for f in P["folds"]]))
    n_w = int(np.mean([f["control"]["n_test"] for f in W["folds"]]))
    print(f"episodes per fold:  production {n_p:,}   all-hours {n_w:,}   "
          f"ratio {n_w / max(n_p, 1):.1f}x")
    print(f"folds:              production {len(P['folds'])}   "
          f"all-hours {len(W['folds'])}\n")
    if len(P["folds"]) != len(W["folds"]):
        print("  WARNING: fold counts differ; the two runs are not matched arms")

    out = {"n_per_fold": {"production": n_p, "all_hours": n_w},
           "sqrt_n_hours": float(np.sqrt(N_HOURS)), "keys": {}}

    print(f"{'quantity':>14} {'slice':>22} {'delta':>10} {'CI width':>10} "
          f"{'rel delta':>10} {'rel width':>10}")
    for k in KEYS:
        row = {}
        for D in (P, W):
            s = D["summary"][k]
            base = abs(s["control"]) if s["control"] else float("nan")
            row[D["_label"]] = {
                "delta": s["delta"], "control": s["control"],
                "ci95": s["ci95"], "width": width(s["ci95"]),
                "rel_delta_pct": 100.0 * s["delta"] / base,
                "rel_width": width(s["ci95"]) / base}
            print(f"{k:>14} {D['_label']:>22} {s['delta']:>+10.5f} "
                  f"{width(s['ci95']):>10.5f} "
                  f"{100.0 * s['delta'] / base:>+9.2f}% "
                  f"{width(s['ci95']) / base:>10.5f}")
        p, w = row[P["_label"]], row[W["_label"]]
        row["ratio_absolute"] = p["width"] / max(w["width"], 1e-12)
        row["ratio_scale_normalised"] = p["rel_width"] / max(w["rel_width"], 1e-12)
        out["keys"][k] = row
        print(f"{'':>14} {'-> width ratio':>22}  absolute "
              f"{row['ratio_absolute']:.2f}x   scale-normalised "
              f"{row['ratio_scale_normalised']:.2f}x\n")

    sp = out["keys"]["qlike_spike"]
    print("=" * 78)
    print("PRIMARY (BENCHMARK.md 22, as pre-registered): spike-QLIKE CI width ratio")
    print(f"  absolute          {sp['ratio_absolute']:.2f}x")
    print(f"  scale-normalised  {sp['ratio_scale_normalised']:.2f}x   "
          f"<- the interpretable one (22's amendment)")
    print(f"  independence would give sqrt({N_HOURS}) = {np.sqrt(N_HOURS):.2f}x")
    eff = (sp["ratio_scale_normalised"] ** 2)
    out["effective_n_multiplier"] = float(eff)
    print(f"\n  EFFECTIVE SAMPLE-SIZE MULTIPLIER = (scale-normalised ratio)^2 = "
          f"{eff:.2f}x")
    print(f"  against a NOMINAL {n_w / max(n_p, 1):.1f}x more episodes. "
          f"Of that nominal gain,\n  {100 * eff / max(n_w / max(n_p, 1), 1e-9):.1f}% "
          f"survives as usable precision.")

    # verdict comparison -- the thing that would force a re-run of prior work
    print("\n" + "=" * 78)
    print("VERDICT (the pre-registered rule of BENCHMARK.md 19, applied to each slice)")
    for D in (P, W):
        s = D["summary"]["qlike_spike"]
        favourable = s["ci95"][1] < 0
        unfavourable = s["ci95"][0] > 0
        v = ("ADOPT" if favourable else
             "REJECT (worse, CI excludes zero)" if unfavourable else
             "REJECT (CI contains zero)")
        print(f"  {D['_label']:>22}: spike CI [{s['ci95'][0]:+.5f}, "
              f"{s['ci95'][1]:+.5f}] -> {v}")
        out.setdefault("verdicts", {})[D["_label"]] = v
    vs = set(out["verdicts"].values())
    out["verdict_flipped"] = len(vs) > 1
    print(f"\n  verdict agrees across slices: {not out['verdict_flipped']}")
    if out["verdict_flipped"]:
        print("  *** THE VERDICT FLIPPED. Every prior verdict in BENCHMARK.md that\n"
              "      rested on a marginal interval must be re-run before any new\n"
              "      experiment starts (BENCHMARK.md 22's stated gate).")

    # The paired per-episode estimator, where a run recorded it.
    #
    # ASYMMETRY, stated rather than papered over: the paired arrays were added
    # to `anchor_freshness` AFTER the production run had already completed, so
    # only the all-hours JSON carries them. Re-deriving the production run's
    # from `--from-json` is not possible either -- the stored per-fold records
    # summarise, they do not keep per-episode arrays.
    #
    # That is a smaller loss than it looks. The question the paired estimator
    # answers -- how much of the FOLD-level interval is sampling error versus
    # between-year heterogeneity -- is answered WITHIN one slice by comparing
    # the two intervals on the same data. A cross-slice paired comparison would
    # be a nice-to-have; a within-slice one is the actual measurement, and one
    # run supplies it.
    missing = [D["_label"] for D in (P, W)
               if "spike" not in (D.get("paired_per_episode") or {})]
    if missing:
        print(f"\n  note: no paired per-episode arrays in {', '.join(missing)} "
              f"(recorded only from the run that post-dates the estimator);\n"
              f"        the within-slice comparison below is the measurement, "
              f"not a cross-slice one")
    for D in (P, W):
        pe = D.get("paired_per_episode") or {}
        if "spike" in pe:
            fl = width(D["summary"]["qlike_spike"]["ci95"])
            pw = width(pe["spike"]["ci95"])
            print(f"\n  {D['_label']}: paired per-episode spike CI is "
                  f"{fl / max(pw, 1e-12):.1f}x narrower than the fold-level one "
                  f"({pe['spike']['n_episodes']:,} episodes)")
            out.setdefault("paired", {})[D["_label"]] = {
                "fold_width": fl, "paired_width": pw,
                "n_episodes": pe["spike"]["n_episodes"]}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
