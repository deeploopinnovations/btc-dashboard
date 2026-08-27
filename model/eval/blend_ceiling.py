"""
eval/blend_ceiling.py
=====================================================================
Is the ENSEMBLE WEIGHT the binding constraint, and how much is on the table?

WHERE THIS COMES FROM

19 found that `BASE_COLS` -- the input set of the Log-HAR anchor holding 75% of
the blended forecast -- has `har_1d` as its fastest input, so the dominant term
cannot respond to anything faster than a day. `eval/anchor_freshness.py` tested
the obvious fix (put `har_1h`/`har_6h` into the anchor) and 19 named, in
advance, what a negative result would mean:

    "The neural stage ALREADY sees har_1h/har_6h. If the blend, committee and
    calibration are already extracting what those features carry, adding them
    to the anchor buys nothing and the lag has a different cause. That would
    redirect the work away from feature placement and toward the blend weight
    itself -- which is the next thing to test, not a dead end."

This is that next thing. The premise is mechanical, not speculative:

  * Log-HAR is slow but robust. Pure NOCTUA lost +72.3% in the 2023 volatility
    collapse; the blend bounded that fold at +6.7% (the note on `infer.BLEND_W`).
  * The neural stage is the only term that sees below daily resolution.
  * `BLEND_W = 0.25` is a CONSTANT, chosen by a walk-forward sweep over
    constants (`model/artifacts/blend_sweep.json`).

A single constant that is simultaneously the best compromise for a calm night
and for a spike onset is exactly the thing worth doubting. Nobody has tested a
state-dependent weight.

WHY THIS COSTS ALMOST NOTHING TO MEASURE

The blend is affine in log space. `infer.predict` shifts the whole predictive
distribution so its median lands on the ensemble level:

    qa_med = qa_med_raw + (1 - w) * (har_logvol - qa_med_raw)
           = w * qa_med_raw + (1 - w) * har_logvol

Seed-averaging `qa` is linear, so the raw neural median inverts EXACTLY from
the recorded post-shift median:

    qa_med_raw = (qa_med - (1 - w0) * har_logvol) / w0

and any other weight is one exponential away. So the entire w-surface -- every
constant weight, every state-dependent rule -- comes out of ONE benchmark run
instead of one 6-fold retrain per value of w. `benchmark.run_fold` was extended
(additively) to record `qa_med`, `har_logvol`, `blend_w` and `H` for exactly
this.

THREE QUANTITIES, AND ONLY ONE OF THEM IS A RESULT

  1. CONSTANT-w SWEEP. Honest, but already known to land near 0.25.

  2. THE ORACLE CEILING -- and there are TWO of them, which is the whole point.

     2a. PER-EPISODE oracle: the w that minimises QLIKE for each episode
         separately, with the realized RV in hand. This is DEGENERATE and is
         reported only to be dismissed. It fits one free parameter per
         observation, so it drives sigma toward |rv| episode by episode and
         its "gain" is almost entirely the overfit, not headroom. Quoting it
         as headroom would repeat the withdrawn "92% is shape" figure exactly.

     2b. TWO-STATE in-sample oracle: ONE w for spike episodes and ONE for calm,
         fitted on the same fold being scored. Two free parameters against
         ~90,000 observations. THIS is the meaningful upper bound, because it
         is the best any two-state rule could do even with perfect foresight
         about which two numbers to pick -- so the causal rule in (3), which
         must pick them from prior folds, cannot beat it.

     AMENDED BEFORE ANY SCORE WAS READ. The first version of this rule made the
     2% floor apply to 2a. That was wrong on its face: 2a's number is an
     artifact of parameter count and would have passed a 2% floor no matter
     what the data said, which makes it a rule that cannot fail -- the same
     defect research/pitfalls.py check 9 exists to catch, arriving from the
     opposite direction. The floor now applies to 2b. No fold had been run when
     this was changed; the amendment is recorded here rather than made silently.

  3. THE CAUSAL, WALK-FORWARD RULE. Two weights, one for spike-risk episodes
     and one for calm, where the split uses `causal_spike_flag` -- information
     available at the anchor. Fitted on folds STRICTLY BEFORE the scored fold
     and applied out of sample. This is the only number that can be adopted.

PRE-REGISTERED RULE, fixed before the scores are read

  PRIMARY: pooled test QLIKE of the walk-forward two-state rule versus the
  shipped constant w = 0.25, over the folds that have at least one prior fold
  to fit on, with a moving-block bootstrap CI that must exclude zero on the
  favourable side.

  Pooled is primary HERE, unlike 19 -- and the reason is the 13 lesson read
  correctly rather than cargo-culted. 13's rule failed because a treatment
  aimed at 6% of episodes was judged on a pooled mean it could not move. This
  treatment changes the weight on EVERY episode, calm ones included: its
  target population IS the pooled one.

  GUARDS, all of which must hold:
    - spike-episode QLIKE must not worsen at all (own CI);
    - calm-episode QLIKE must not worsen by more than 1% (own CI);
    - the fitted weights must be reported per fold, and if they oscillate in
      sign of deviation from 0.25 across folds the result is NULL regardless
      of the pooled CI -- an unstable parameter is not a lever;
    - deep-tail barrier MCB is NOT scored here, because this script re-weights
      a recorded median and does not rebuild the barrier curves. Adoption
      therefore requires a follow-up full-pipeline run; this experiment can
      REJECT on its own but cannot ADOPT on its own.

  REJECTION: if the TWO-STATE in-sample ceiling (2b) is under 2% of pooled
  QLIKE, the lever is declared not worth building and (3) is not interpreted --
  there is nothing for a causal rule to recover.

    python -m model.eval.blend_ceiling
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import benchmark as B                                           # noqa: E402
from eval.direction import ci_excludes_zero, mean_ci                      # noqa: E402
from eval.levers import causal_spike_flag, qlike                          # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402

W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)


def sigma_at(w: float, raw: np.ndarray, har: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Window vol at ensemble weight `w`, from the recovered components."""
    return np.exp(w * raw + (1.0 - w) * har) * np.sqrt(H)


def recover_raw(pe: dict) -> np.ndarray:
    """Invert the affine blend to get the seed-averaged RAW neural log-vol rate."""
    w0 = float(pe["blend_w"])
    if w0 <= 0.0:
        raise ValueError("blend_w = 0 leaves no neural component to recover")
    return (pe["qa_med"] - (1.0 - w0) * pe["har_logvol"]) / w0


def check_recovery(pe: dict, tol: float = 1e-9) -> float:
    """Round-trip the inversion. A silent mismatch here would corrupt everything
    downstream, and this repository's recurring failure is a wrong comparison
    that ran to completion, so the check is loud and unconditional."""
    raw = recover_raw(pe)
    got = sigma_at(float(pe["blend_w"]), raw, pe["har_logvol"], pe["H"])
    err = float(np.max(np.abs(got - pe["sigma_med"]) / np.maximum(pe["sigma_med"], 1e-12)))
    if not err < 1e-6:
        raise AssertionError(
            f"blend inversion does not round-trip: max relative error {err:.3e}. "
            f"The recorded sigma_med is not exp(w*qa_raw + (1-w)*har)*sqrt(H), so "
            f"the whole w-surface computed from it would be fiction.")
    return err


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is the ensemble weight the constraint?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/blend_ceiling.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    spike = causal_spike_flag(ep)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    raw_har = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(m):
        lo, hi = np.quantile(raw_har[m], [0.005, 0.995])
        return np.maximum(np.clip(raw_har, lo, hi), 1e-12)

    print(f"folds {len(folds)}  seeds {a.seeds}  "
          f"spike {spike.sum():,}/{len(ep):,} ({100*spike.mean():.2f}%)\n")

    per_fold = []
    for f in folds:
        t0 = time.time()
        r = B.run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds, sigma_ref_fn=sig_fn)
        if r is None:
            print(f"  {f['year']}  SKIPPED"); continue
        pe = r["per_episode"]
        for k in ("qa_med", "har_logvol", "H"):
            if k not in pe:
                raise SystemExit(
                    f"REFUSING: run_fold did not record '{k}'. This script needs "
                    f"the blend components; re-run against a benchmark.py that "
                    f"records them.")
        err = check_recovery(pe)
        rec = {"year": f["year"], "recovery_err": err,
               "rv": pe["rv"], "raw": recover_raw(pe), "har": pe["har_logvol"],
               "H": pe["H"], "spike": spike[pe["test_idx"]],
               "n": int(len(pe["rv"]))}
        per_fold.append(rec)
        print(f"  {f['year']}  n={rec['n']:6d}  spike={int(rec['spike'].sum()):5d}  "
              f"inversion max rel err {err:.2e}  ({time.time()-t0:.0f}s)", flush=True)

    if not per_fold:
        print("no fold produced results"); return 1

    def fold_q(rec, w_calm, w_spike):
        w = np.where(rec["spike"], w_spike, w_calm)
        return qlike(rec["rv"], sigma_at(w, rec["raw"], rec["har"], rec["H"]))

    # QLIKE at every grid weight, once per fold. Every quantity below is a
    # different reduction of this one array, and the two-state grid search in
    # (3) is 441 combinations per history fold -- recomputing QLIKE inside that
    # loop would be ~1e9 pointless exponentials.
    for rec in per_fold:
        rec["Q"] = np.stack([qlike(rec["rv"], sigma_at(w, rec["raw"], rec["har"],
                                                       rec["H"])) for w in W_GRID])
        sp = rec["spike"]
        rec["Qsum_spike"] = rec["Q"][:, sp].sum(axis=1)      # (len(W),)
        rec["Qsum_calm"] = rec["Q"][:, ~sp].sum(axis=1)

    # ---- 1. constant-w sweep ------------------------------------------------
    print("\n--- constant w, mean over folds (w=0.25 is what ships) ---")
    print(f"{'w':>6} {'pooled':>9} {'spike':>9} {'calm':>9}")
    sweep = {}
    for i, w in enumerate(W_GRID):
        pl = [rec["Q"][i].mean() for rec in per_fold]
        sk = [rec["Q"][i][rec["spike"]].mean() for rec in per_fold if rec["spike"].any()]
        cm = [rec["Q"][i][~rec["spike"]].mean() for rec in per_fold]
        sweep[float(w)] = {"pooled": float(np.mean(pl)), "spike": float(np.mean(sk)),
                           "calm": float(np.mean(cm))}
        mark = "  <- shipped" if abs(w - 0.25) < 1e-9 else ""
        print(f"{w:6.2f} {sweep[float(w)]['pooled']:9.4f} "
              f"{sweep[float(w)]['spike']:9.4f} {sweep[float(w)]['calm']:9.4f}{mark}")

    best_const = min(sweep, key=lambda w: sweep[w]["pooled"])
    ship = sweep[0.25]
    print(f"\nbest constant w = {best_const:.2f}  pooled {sweep[best_const]['pooled']:.4f} "
          f"vs shipped {ship['pooled']:.4f}  "
          f"({100*(sweep[best_const]['pooled']-ship['pooled'])/ship['pooled']:+.2f}%)")

    # ---- 2. the oracle ceiling -- USES THE FUTURE, NOT ACHIEVABLE -----------
    frac_ep, frac_2s, two_state_w = [], [], []
    orac_w_by_slice = {"spike": [], "calm": []}
    j25 = int(np.argmin(np.abs(W_GRID - 0.25)))
    for rec in per_fold:
        Q = rec["Q"]
        base = float(Q[j25].mean())
        # 2a -- degenerate, one parameter per observation
        wstar = W_GRID[np.argmin(Q, axis=0)]
        orac_w_by_slice["spike"].append(float(np.mean(wstar[rec["spike"]]))
                                        if rec["spike"].any() else np.nan)
        orac_w_by_slice["calm"].append(float(np.mean(wstar[~rec["spike"]])))
        frac_ep.append(float((base - float(Q.min(axis=0).mean())) / base))
        # 2b -- two parameters, fitted on this fold: the meaningful bound
        surf = (rec["Qsum_calm"][:, None] + rec["Qsum_spike"][None, :]) / rec["n"]
        ic, isp = np.unravel_index(int(np.argmin(surf)), surf.shape)
        frac_2s.append(float((base - float(surf[ic, isp])) / base))
        two_state_w.append({"year": rec["year"], "w_calm": float(W_GRID[ic]),
                            "w_spike": float(W_GRID[isp]),
                            "gain_pct": 100 * frac_2s[-1]})
    ceiling_ep = float(np.mean(frac_ep))
    ceiling = float(np.mean(frac_2s))          # the pre-registered quantity
    print(f"\n--- ORACLE CEILINGS (both use realized RV; NEITHER is achievable) ---")
    print(f"  2a per-episode oracle: {100*ceiling_ep:.2f}%  "
          f"-- DEGENERATE, one parameter per observation, reported to be dismissed")
    print(f"  2b two-state oracle  : {100*ceiling:.2f}%  <- the pre-registered ceiling")
    for t in two_state_w:
        print(f"       {t['year']}  w_calm {t['w_calm']:.2f}  w_spike {t['w_spike']:.2f}"
              f"  gain {t['gain_pct']:+.2f}%")
    print(f"  mean per-episode oracle w: spike {np.nanmean(orac_w_by_slice['spike']):.3f}, "
          f"calm {np.nanmean(orac_w_by_slice['calm']):.3f}")

    # ---- 3. the causal walk-forward two-state rule --------------------------
    # Fit (w_calm, w_spike) on folds strictly EARLIER than the scored fold. The
    # first fold has no history and is excluded rather than given a peek.
    print(f"\n--- CAUSAL walk-forward two-state rule (fit on prior folds only) ---")
    print(f"{'year':>6} {'w_calm':>8} {'w_spike':>8} {'pooled@rule':>12} "
          f"{'pooled@0.25':>12} {'delta%':>8}")
    d_pool, d_spike, d_calm, fitted = [], [], [], []
    for i, rec in enumerate(per_fold):
        hist = per_fold[:i]
        if not hist:
            print(f"{rec['year']:>6}    (no prior fold -- excluded from the test)")
            continue
        # mean over history folds of (calm-sum at w_c + spike-sum at w_s) / n,
        # as a (len(W), len(W)) surface: rows index w_calm, columns w_spike.
        surf = np.zeros((len(W_GRID), len(W_GRID)))
        for h in hist:
            surf += (h["Qsum_calm"][:, None] + h["Qsum_spike"][None, :]) / h["n"]
        surf /= len(hist)
        ic, isp = np.unravel_index(int(np.argmin(surf)), surf.shape)
        bw = (float(W_GRID[ic]), float(W_GRID[isp]))
        q_rule = fold_q(rec, *bw)
        q_base = fold_q(rec, 0.25, 0.25)
        d_pool.append(float(q_rule.mean() - q_base.mean()))
        d_calm.append(float(q_rule[~rec["spike"]].mean() - q_base[~rec["spike"]].mean()))
        if rec["spike"].any():
            d_spike.append(float(q_rule[rec["spike"]].mean() - q_base[rec["spike"]].mean()))
        fitted.append({"year": rec["year"], "w_calm": bw[0], "w_spike": bw[1]})
        print(f"{rec['year']:>6} {bw[0]:8.2f} {bw[1]:8.2f} {q_rule.mean():12.4f} "
              f"{q_base.mean():12.4f} {100*d_pool[-1]/q_base.mean():+8.2f}")

    out = {"sweep": sweep, "best_constant_w": float(best_const),
           "oracle_ceiling_frac": ceiling,
           "oracle_ceiling_frac_per_episode_DEGENERATE": ceiling_ep,
           "oracle_two_state_w": two_state_w,
           "oracle_w_spike": float(np.nanmean(orac_w_by_slice["spike"])),
           "oracle_w_calm": float(np.nanmean(orac_w_by_slice["calm"])),
           "fitted_weights": fitted,
           "recovery_err": [r["recovery_err"] for r in per_fold],
           "n_folds_scored": len(d_pool)}

    if len(d_pool) >= 2:
        print(f"\n{'quantity':>16} {'delta':>11} {'95% CI':>26}")
        for nm, d in (("pooled QLIKE", d_pool), ("spike QLIKE", d_spike),
                      ("calm QLIKE", d_calm)):
            arr = np.asarray(d, np.float64)
            # mean_ci, not block_bootstrap_ci: the unit is a FOLD and that
            # helper returns (nan, nan) below n = 20, which a comparison would
            # silently read as "did not clear the bar" (pitfalls check 11).
            ci = mean_ci(arr, seed=29)
            lo, hi = ci["ci95"]
            out[nm.split()[0] + "_delta"] = {"delta": float(arr.mean()), "ci95": [lo, hi],
                                             "ci95_iid": ci["ci95_iid"],
                                             "n_negative": ci["n_negative"],
                                             "n_positive": ci["n_positive"]}
            print(f"{nm:>16} {arr.mean():+11.5f}   [{lo:+.5f}, {hi:+.5f}]"
                  f"   iid [{ci['ci95_iid'][0]:+.5f}, {ci['ci95_iid'][1]:+.5f}]"
                  f"   {ci['n_negative']}-/{ci['n_positive']}+")

        # ---- the pre-registered rule, applied verbatim ----------------------
        signs = {np.sign(f["w_calm"] - 0.25) for f in fitted} | \
                {np.sign(f["w_spike"] - 0.25) for f in fitted}
        stable = not ({-1.0, 1.0} <= signs)
        lo_p, hi_p = out["pooled_delta"]["ci95"]
        lo_s, hi_s = out["spike_delta"]["ci95"]
        calm_pct = 100 * np.mean(d_calm) / sweep[0.25]["calm"]
        if ceiling < 0.02:
            verdict = "REJECT"
            why = (f"two-state oracle ceiling {100*ceiling:.2f}% is under the "
                   f"pre-registered 2% floor: even a rule that knew the two best "
                   f"weights in advance could not recover enough to matter")
        elif not ci_excludes_zero(out["pooled_delta"]["ci95"], -1):
            verdict = "NULL"
            why = f"pooled CI [{lo_p:+.5f}, {hi_p:+.5f}] does not clear zero favourably"
        elif hi_s > 0:
            verdict = "NULL"
            why = f"guard: spike QLIKE worsens, CI [{lo_s:+.5f}, {hi_s:+.5f}]"
        elif calm_pct > 1.0:
            verdict = "NULL"
            why = f"guard: calm QLIKE worsens {calm_pct:+.2f}%, over the 1% bar"
        elif not stable:
            verdict = "NULL"
            why = "guard: fitted weights straddle 0.25 in both directions across folds"
        else:
            verdict = "ADVANCE"
            why = ("clears the pooled CI and every guard; NOT an adoption -- the "
                   "barrier curves were not rebuilt, so a full-pipeline run is "
                   "required before this ships")
        out["verdict"], out["why"] = verdict, why
        print(f"\nPRE-REGISTERED VERDICT: {verdict}\n  {why}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
