"""
eval/anchor_freshness.py
=====================================================================
Does the Log-HAR anchor's blindness below daily resolution cause the lag?

THE FINDING THIS TESTS (BENCHMARK.md 19)

    BASE_COLS = [har_1d, har_5d, har_22d, cal_H, cal_weekend_frac]
    BLEND_W   = 0.25   ->  Stage A median = 0.25 * neural + 0.75 * Log-HAR

`har_1h` and `har_6h` are computed, stored, and fed to the NEURAL stage -- and
absent from `BASE_COLS`, the anchor carrying 75% of the blended forecast. So
the dominant term's fastest input is `har_1d`, and a component whose finest
resolution is one day cannot respond to anything faster than a day.

That is a candidate mechanism for the lag measured in 7a: spike nights
under-forecast by 45% while carrying 25.8% of total loss, with predicted sigma
correlating 0.920 with realized RV dated one day AFTER and only 0.507 one day
BEFORE. 12 established the information is present (onset AUC 0.733, CI
[0.6547, 0.8052]); 19 says the dominant term cannot see it.

12's Arm 4 already measured the predicted consequence on an OLS-only harness
(-7.33% pooled, -10.44% spike, -5.48% onset). This runs it through the full
pipeline, where the neural stage, the committee, the calibration and the blend
all get a say -- any of which could absorb or reverse the effect.

THE PRE-REGISTERED RULE, fixed in BENCHMARK.md 19 before this ran

  PRIMARY, on the slice the treatment targets: spike-episode QLIKE, with a
  moving-block bootstrap CI that must exclude zero on the favourable side.

  This is primary BECAUSE 13 made pooled QLIKE primary for a 6%-of-episodes
  treatment and produced a verdict nobody could interpret: the lever improved
  its target decisively (spike QLIKE -0.165, CI excluding zero) and still
  failed, because a minority slice cannot move a pooled mean. The metric must
  match the population the change is aimed at.

  GUARDS, all of which must hold:
    - calm-episode QLIKE must not worsen by more than 1%, on its own CI
    - deep-tail barrier MCB must not worsen with a CI excluding zero unfavourably
    - pooled QLIKE is REPORTED but is NOT a condition
    - `log_har_gauss` is re-scored, not treated as fixed: it gains the same two
      inputs, so holding it constant would compare against a baseline that no
      longer exists

  REJECTION: if spike QLIKE's CI contains zero, 12's Arm 4 gain was an artifact
  of the OLS-only harness and does not survive the full pipeline.

WHAT A NEGATIVE RESULT LOOKS LIKE, AND WHY IT WOULD STILL BE WORTH HAVING

The neural stage ALREADY sees har_1h/har_6h. If the blend, committee and
calibration are already extracting what those features carry, adding them to
the anchor buys nothing and the lag has a different cause. That would redirect
the work away from feature placement and toward the blend weight itself --
which is the next thing to test, not a dead end.

    python -m model.eval.anchor_freshness
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlib import contextmanager                                     # noqa: E402

from eval import benchmark as B                                           # noqa: E402
from noctua import model as _model                                        # noqa: E402
from noctua import spec as _spec                                          # noqa: E402
from noctua import train as _train                                        # noqa: E402
from eval.direction import ci_excludes_zero, mean_ci                      # noqa: E402
from eval.levers import causal_spike_flag, qlike                          # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402
from research import pitfalls as P                                        # noqa: E402

FRESH = ["har_1h", "har_6h"]
TAIL_BARRIERS = (0.5, 1.0, 2.0)


# Every module that does `from .model import BASE_COLS` holds an INDEPENDENT
# binding -- six of them do. The first version of this script patched only
# `benchmark.BASE_COLS`, so `benchmark`'s OLS expected 7 columns while
# `train.prepare()` still built a 5-column Xb from its own binding, and the run
# died on `Shape of passed values is (102087, 5), indices imply (102087, 7)`.
#
# Loud is the right failure here. Had the shapes happened to agree -- say by
# swapping one column for another rather than adding two -- the run would have
# completed and silently compared the wrong thing, which is the failure mode
# this repository has been bitten by four times (see BENCHMARK.md 16).
_BINDINGS = (_spec, _model, _train, B)


@contextmanager
def base_cols(cols):
    """Swap BASE_COLS across every binding, and put them all back."""
    saved = [(m, getattr(m, "BASE_COLS", None)) for m in _BINDINGS]
    try:
        for m, _ in saved:
            if hasattr(m, "BASE_COLS"):
                setattr(m, "BASE_COLS", list(cols))
        seen = {tuple(getattr(m, "BASE_COLS")) for m, _ in saved
                if hasattr(m, "BASE_COLS")}
        assert seen == {tuple(cols)}, f"bindings disagree after patch: {seen}"
        yield
    finally:
        for m, v in saved:
            if v is not None:
                setattr(m, "BASE_COLS", v)


def tail_mcb(rows) -> float:
    r = next(x for x in rows if x["model"] == "noctua_v2")
    return float(sum(r[f"MCB_{s}_{p}"] for s in ("up", "dn") for p in TAIL_BARRIERS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="har_1h/har_6h in the Log-HAR anchor")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/anchor_freshness.json"))
    ap.add_argument("--all-hours", action="store_true",
                    help="score all 24 anchor hours at H = 19 instead of the "
                         "production slice (17:00 only). BENCHMARK.md 22: this "
                         "is the same experiment through the same code path on "
                         "a 24x wider scoring slice, so the CI width ratio "
                         "measures how much power the production slice gives "
                         "up. Training is unchanged; only scoring widens.")
    ap.add_argument("--from-json", type=Path, default=None,
                    help="re-derive the verdict from an earlier run's stored "
                         "per-fold records instead of retraining. Every fold "
                         "number is read, none recomputed, so this can fix a "
                         "broken STATISTIC but can never change a SCORE.")
    a = ap.parse_args(argv)

    if a.all_hours and a.out == Path("model/artifacts/anchor_freshness.json"):
        a.out = Path("model/artifacts/anchor_freshness_allhours.json")

    if a.from_json is not None:
        prev = json.loads(a.from_json.read_text())
        recs = prev["folds"]
        base_orig = prev["base_cols_control"]
        base_fresh = prev["base_cols_treated"]
        print(f"re-deriving from {a.from_json} -- {len(recs)} stored folds, "
              f"no model is retrained")
        return _verdict(recs, base_orig, base_fresh, a.out)

    ep, X = load_all(a.artifacts)
    spike = causal_spike_flag(ep)
    folds = S.walk_forward_folds(ep)
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def sig_fn(m):
        lo, hi = np.quantile(raw[m], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    base_orig = list(_spec.BASE_COLS)
    base_fresh = [base_orig[0]] + FRESH + base_orig[1:]
    missing = [c for c in FRESH if c not in X.columns]
    if missing:
        raise SystemExit(f"REFUSING: {missing} absent from the feature matrix")
    # The scoring slice. `prod_override=None` means run_fold uses
    # production_mask -- H == 19 AND anchor_hour == 17, ~365 episodes a year.
    # --all-hours keeps H == 19 and takes every anchor hour, 24x the episodes
    # with adjacent hours sharing 18 of their 19 hours; that dependence is
    # serial and is what the moving-block bootstrap is built for. Horizons are
    # deliberately NOT mixed: H = 6/12/19/24 at one anchor are nested inside
    # one another, which is a different kind of overlap the estimator does not
    # model.
    prod = None
    if a.all_hours:
        prod = (ep["H"] == S.PROD_H).to_numpy()
        print(f"scoring slice: ALL 24 anchor hours at H = {S.PROD_H} "
              f"({int(prod.sum()):,} episodes in the population, vs "
              f"{int(S.production_mask(ep).sum()):,} in the production slice)")
    print(f"control BASE_COLS ({len(base_orig)}): {base_orig}")
    print(f"treated BASE_COLS ({len(base_fresh)}): {base_fresh}")
    print(f"spike episodes: {spike.sum():,} of {len(ep):,} "
          f"({100*spike.mean():.2f}%)   folds {len(folds)}  seeds {a.seeds}\n")

    recs = []
    for f in folds:
        line = {"year": f["year"]}
        for nm, cols in (("control", base_orig), ("fresh_anchor", base_fresh)):
            t0 = time.time()
            # BASE_COLS is module-level state read by prepare() and the OLS fit,
            # so it is swapped around the call and restored in a finally -- a
            # leaked mutation would silently contaminate every later fold.
            with base_cols(cols):
                r = B.run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                               sigma_ref_fn=sig_fn, prod_override=prod)
            if r is None:
                print(f"  {f['year']}  {nm:13} SKIPPED"); continue
            pe = r["per_episode"]
            rv, sg, idx = pe["rv"], pe["sigma_med"], pe["test_idx"]
            q = qlike(rv, sg); sp = spike[idx]
            # Keep the per-episode loss vector. Both arms are trained
            # separately but scored on the SAME episode set (same fold, same
            # mask), so the deltas are PAIRED -- which makes a bootstrap over
            # episodes available as an estimator, not just a bootstrap over six
            # fold means. The two answer different questions: the fold-level
            # interval carries between-year treatment heterogeneity, the
            # paired-episode interval carries only sampling error. Storing this
            # costs a few MB and is the difference between being able to
            # separate those and not.
            line.setdefault("_q", {})[nm] = q
            line.setdefault("_idx", {})[nm] = idx
            line[nm] = {
                "qlike_pooled": float(q.mean()),
                "qlike_spike": float(q[sp].mean()) if sp.any() else float("nan"),
                "qlike_calm": float(q[~sp].mean()),
                "rv_sigma_spike": float(np.median((rv/np.maximum(sg,1e-12))[sp]))
                                   if sp.any() else float("nan"),
                "tail_mcb": tail_mcb(r["rows"]),
                "qlike_loghar": float(r["vol"]["log_har"]),
                "n_spike": int(sp.sum()), "n_test": int(len(idx))}
            s = line[nm]
            print(f"  {f['year']}  {nm:13} pooled {s['qlike_pooled']:.4f}  "
                  f"spike {s['qlike_spike']:.4f}  calm {s['qlike_calm']:.4f}  "
                  f"logHAR {s['qlike_loghar']:.4f}  tailMCB {s['tail_mcb']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if "control" in line and "fresh_anchor" in line:
            recs.append(line)

    if not recs:
        print("no fold produced both arms"); return 1
    return _verdict(recs, base_orig, base_fresh, a.out)


def _verdict(recs, base_orig, base_fresh, out_path):
    """Everything downstream of the fold scores. Separated so a statistic can be
    corrected without retraining -- and so the corrected verdict is produced by
    the SAME code path as the original, not a one-off recomputation."""
    def delta(key):
        return np.array([r["fresh_anchor"][key] - r["control"][key] for r in recs])

    print(f"\n{'quantity':>22} {'control':>10} {'fresh':>10} {'delta':>11} {'95% CI':>24}")
    summary = {}
    for key in ("qlike_spike", "qlike_calm", "qlike_pooled", "tail_mcb",
                "rv_sigma_spike", "qlike_loghar"):
        c = np.array([r["control"][key] for r in recs])
        t = np.array([r["fresh_anchor"][key] for r in recs])
        d = delta(key)
        # mean_ci, not block_bootstrap_ci: the unit here is a FOLD, and at
        # n = 6 that helper returns (nan, nan) by design. The first run of this
        # script compared those NaNs and printed a verdict that had never
        # looked at the data. See eval/direction.mean_ci.
        ci = mean_ci(d, seed=23)
        lo, hi = ci["ci95"]
        summary[key] = {"control": float(np.nanmean(c)), "fresh": float(np.nanmean(t)),
                        "delta": float(np.nanmean(d)), "ci95": [lo, hi],
                        "ci95_iid": ci["ci95_iid"], "n": ci["n"],
                        "block_len": ci["block_len"], "sd": ci["sd"],
                        "n_negative": ci["n_negative"], "n_positive": ci["n_positive"]}
        print(f"{key:>22} {np.nanmean(c):10.5f} {np.nanmean(t):10.5f} "
              f"{np.nanmean(d):+11.5f} [{lo:+.5f}, {hi:+.5f}]"
              f"   iid [{ci['ci95_iid'][0]:+.5f}, {ci['ci95_iid'][1]:+.5f}]"
              f"   {ci['n_negative']}-/{ci['n_positive']}+")

    sp_lo, sp_hi = summary["qlike_spike"]["ci95"]
    calm_pct = 100.0 * (summary["qlike_calm"]["fresh"] /
                        max(summary["qlike_calm"]["control"], 1e-12) - 1.0)
    calm_lo, calm_hi = summary["qlike_calm"]["ci95"]
    tail_lo, tail_hi = summary["tail_mcb"]["ci95"]

    # A negative delta is the win for a loss, so favourable_sign = -1.
    # ci_excludes_zero RAISES on a NaN interval rather than answering False.
    primary = ci_excludes_zero(summary["qlike_spike"]["ci95"], -1)
    guard_calm = calm_pct <= 1.0
    guard_tail = not ci_excludes_zero(summary["tail_mcb"]["ci95"], +1)
    adopt = primary and guard_calm and guard_tail
    print(f"\n--- pre-registered rule (BENCHMARK.md 19) ---")
    print(f"  PRIMARY spike QLIKE CI excludes zero favourably : {primary}")
    print(f"  GUARD   calm QLIKE within 1%  ({calm_pct:+.2f}%)         : {guard_calm}")
    print(f"  GUARD   deep-tail MCB not significantly worse    : {guard_tail}")
    print(f"  -> {'ADOPT' if adopt else 'DO NOT ADOPT'}")
    print(f"  (pooled QLIKE {summary['qlike_pooled']['delta']:+.5f} -- reported, NOT a condition)")

    # ---- the paired per-episode estimator, reported beside the fold-level one
    # This is a DIAGNOSTIC, not the pre-registered endpoint. 19's rule names a
    # moving-block bootstrap over folds and that is what decides the verdict
    # above; this quantifies how much of the fold-level interval is sampling
    # error and how much is year-to-year heterogeneity, which is the question
    # 22 exists to answer.
    paired = {}
    if all("_q" in r for r in recs):
        for key, sel in (("spike", True), ("calm", False)):
            d_ep = []
            for r in recs:
                ic, it = r["_idx"]["control"], r["_idx"]["fresh_anchor"]
                if not np.array_equal(ic, it):
                    d_ep = None; break
                m = spike[ic] if sel else ~spike[ic]
                if m.any():
                    d_ep.append((r["_q"]["fresh_anchor"] - r["_q"]["control"])[m])
            if d_ep:
                arr = np.concatenate(d_ep)
                ci = mean_ci(arr, seed=23)
                paired[key] = {"delta": float(arr.mean()), "ci95": ci["ci95"],
                               "n_episodes": int(len(arr))}
        d_all = [r["_q"]["fresh_anchor"] - r["_q"]["control"] for r in recs]
        arr = np.concatenate(d_all)
        ci = mean_ci(arr, seed=23)
        paired["pooled"] = {"delta": float(arr.mean()), "ci95": ci["ci95"],
                            "n_episodes": int(len(arr))}
    if paired:
        print(f"\n--- paired per-episode bootstrap (DIAGNOSTIC, not the rule) ---")
        print(f"{'quantity':>10} {'n episodes':>11} {'delta':>11} {'95% CI':>26} "
              f"{'vs fold-level':>14}")
        for key in ("spike", "calm", "pooled"):
            if key not in paired:
                continue
            pk, fl = paired[key], summary[f"qlike_{key}"]
            w_p = pk["ci95"][1] - pk["ci95"][0]
            w_f = fl["ci95"][1] - fl["ci95"][0]
            print(f"{key:>10} {pk['n_episodes']:>11,} {pk['delta']:+11.5f}   "
                  f"[{pk['ci95'][0]:+.5f}, {pk['ci95'][1]:+.5f}]   "
                  f"{w_f / max(w_p, 1e-12):>13.2f}x")
        print("  'vs fold-level' is how many times WIDER the fold-level interval "
              "is.\n  A large ratio means the fold-level interval is dominated by "
              "between-year\n  heterogeneity rather than sampling error -- which no "
              "number of extra\n  episodes can reduce.")
    for r in recs:
        r.pop("_q", None); r.pop("_idx", None)

    print("\n--- research/pitfalls on this experiment ---")
    rep = P.Report()
    rep.add(P.check_ci_is_defined(summary["qlike_spike"]["ci95"], "spike QLIKE"))
    rep.add(P.check_not_a_coin_flip(delta("qlike_spike"), "spike QLIKE delta"))
    rep.add(P.check_rule_satisfiable(1, len(recs), "folds"))
    rep.add(P.check_arms_matched({"control": len(recs), "fresh_anchor": len(recs)},
                                 what="folds scored"))
    print(rep.render())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"folds": recs, "summary": summary,
                                 "paired_per_episode": paired,
                                 "adopt": bool(adopt),
                                 "base_cols_control": base_orig,
                                 "base_cols_treated": base_fresh},
                                indent=2, default=float) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
