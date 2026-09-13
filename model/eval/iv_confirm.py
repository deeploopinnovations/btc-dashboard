"""
eval/iv_confirm.py
=====================================================================
E2-confirm: does E2c's IV correction survive the FULL pipeline?

WHY THIS IS A SEPARATE EXPERIMENT AND NOT A RE-READING

E2c (BENCHMARK.md 25) cleared every condition of its rule -- pooled QLIKE
-0.03264, CI [-0.04356, -0.01403], 5 of 5 folds, beating a coverage-preserving
placebo -- and was recorded as ADVANCE, not ADOPT. The distinction is the whole
reason this file exists.

E2c re-weighted a RECORDED MEDIAN. It never rebuilt the barrier curves, the
committee, or the calibration. NOCTUA's product is not a sigma; it is a
touch-probability curve, and scaling sigma moves every touch probability on it.
Nothing in E2c checked what that does to CORP calibration or to Christoffersen
conditional coverage, so nothing in E2c licenses shipping it.

`benchmark.run_fold` gained `post_shift_fn` for exactly this. The correction is
applied to the ensemble level BEFORE Stage B sees it, so the quantile heads, the
four specialists, the equal-weight committee and the recalibration are all built
around the corrected forecast, through the same code path an uncorrected run
uses.

WHERE THE COEFFICIENTS COME FROM, AND WHY THAT IS THE DEPLOYABLE CHOICE

Beta for fold Y is fitted on the cached OUT-OF-SAMPLE forecasts of folds before
Y -- `blend_ceiling.npz`, written by an uncorrected run. It would be circular to
fit the correction on forecasts that already carry it, and it is also not what a
deployment would do: a desk fits a correction on the past out-of-sample
behaviour of the model it currently runs, then applies it going forward. That is
this design, and the second-order effect it ignores -- that the treated arm's
own calibration slice now sees corrected forecasts, so its residuals differ
slightly from the ones beta was fitted on -- is named here rather than hidden.

THREE ARMS, MATCHED

  control   post_shift_fn = None. Byte-identical to a normal run.
  treated   the E2c correction, five dynamics features, no intercept, shrunk.
  placebo   the same machinery on features circularly rotated inside the
            covered era. It ran in E2c and it runs again here, because a
            correction that survives the full pipeline still has to beat one
            that cannot carry information.

PRE-REGISTERED RULE, fixed before any score is read

  POPULATION: test episodes with a causal IV observation. All three arms are
  scored on the SAME episodes.

  PRIMARY: pooled QLIKE, treated versus control, bootstrap CI excluding zero on
  the favourable side. This is the FOURTH test in the IV family (E2, E2b, E2c,
  this), so intervals are at the Bonferroni-adjusted level, alpha = 0.05/4.

  GUARDS, all of which must hold:
    - deep-tail barrier MCB must not worsen with a CI excluding zero
      unfavourably. This is the guard E2c could not run and the reason this
      experiment exists;
    - Christoffersen conditional coverage at alpha = 5% must not degrade:
      the mean |hit_rate - 5%| across both sides, with a CI. Guarded on the
      calibration ERROR rather than on p_cc, because a p-value is evidence
      against a null and not a quality score -- it falls when n rises at fixed
      calibration error, so a p_cc guard would punish a wider slice for being
      wider;
    - spike QLIKE must not worsen, and calm QLIKE must not worsen by more than
      1% of the CALM base;
    - the treated arm must beat the placebo, CI excluding zero;
    - the blend-algebra assertion inside `run_fold` must pass on every fold --
      it verifies the achieved log-level shift equals the requested one to
      1e-6, so an attenuated correction cannot be scored under this name.

  ONLY A PASS HERE EARNS A PLACE IN THE ARTIFACT. A failure does not retract
  E2c, which measured what it said it measured; it means the gain does not
  survive contact with the object the model actually sells.

    python -m model.eval.iv_confirm
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

from eval import benchmark as B                                           # noqa: E402
from eval.blend_ceiling import W0, shrink                                 # noqa: E402
from eval.direction import ci_excludes_zero, mean_ci                      # noqa: E402
from eval.iv_correction import (EXP_BOUND, SIGMA_B, fit_beta,             # noqa: E402
                                standardise)
from eval.levers import causal_spike_flag, qlike                          # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402
from research import pitfalls as P                                        # noqa: E402

# E2c's feature set: iv_level dropped, dynamics kept.
COLS = ["iv_chg_1h", "iv_chg_6h", "iv_chg_24h", "iv_z_20d", "ivrv_ratio"]
N_TESTS = 4                     # E2, E2b, E2c, and this one
TAIL_BARRIERS = (0.5, 1.0, 2.0)
PLACEBO_SHIFT_DAYS = 365        # the INTENDED rotation, in calendar days

# The rotation distance is converted from days to ROWS at run time, because the
# covered set carries several episodes per anchor timestamp (one per horizon)
# and a row offset is therefore NOT an hour offset.
#
# The original constant was `PLACEBO_SHIFT_ROWS = 365 * 24` -- named for hours,
# used as a row index. A data audit measured the consequence: the covered era
# spans 1,944.8 days over 186,663 rows, i.e. 95.98 rows per day, so an 8,760-row
# roll moved 91.3 calendar days, not 365. Both this file and BENCHMARK.md 24
# claimed "one year".
#
# The direction of that error worked AGAINST the result -- a shorter rotation
# retains more residual autocorrelation, making the placebo harder to beat -- so
# it does not explain the headline. It was still a wrong claim, and the fix is
# to compute the offset from the data rather than assert it.

def _rotation_rows(covered_rows, anchor_ts, days: int = PLACEBO_SHIFT_DAYS) -> int:
    """Convert a calendar-day rotation into a ROW offset, measured from the data.

    `covered_rows` indexes the episode table, which carries several rows per
    anchor timestamp, so rows-per-day must be counted rather than assumed.
    """
    import numpy as _np
    ts = _np.asarray(anchor_ts, dtype=_np.int64)[covered_rows]
    span_days = max((ts.max() - ts.min()) / 86400.0, 1.0)
    rows_per_day = len(covered_rows) / span_days
    roll = int(round(days * rows_per_day))
    if len(covered_rows) < 3 * roll:
        raise SystemExit(
            f"REFUSING: {len(covered_rows):,} covered episodes cannot support a "
            f"{days}-day ({roll:,}-row) rotation. The modulo would wrap to a "
            f"small near-neighbour shift, which decorrelates nothing while "
            f"still passing the coverage check -- a placebo that is not one.")
    return roll % len(covered_rows)



def tail_mcb(rows) -> float:
    r = next(x for x in rows if x["model"] == "noctua_v2")
    return float(sum(r[f"MCB_{s}_{p}"] for s in ("up", "dn") for p in TAIL_BARRIERS))


def fit_history_beta(npz, years, upto: int, Z_all: np.ndarray, rotate: int = 0,
                     covered_rows: np.ndarray | None = None):
    """Beta and standardisation moments from folds strictly before `upto`."""
    zs, rs = [], []
    for y in years:
        if y >= upto:
            break
        idx = npz[f"test_idx_{y}"]
        if rotate and covered_rows is not None:
            pos = {int(r): i for i, r in enumerate(covered_rows)}
            src = np.array([covered_rows[(pos[int(r)] + rotate) % len(covered_rows)]
                            if int(r) in pos else -1 for r in idx])
            z = np.where((src >= 0)[:, None], Z_all[np.maximum(src, 0)], np.nan)
        else:
            z = Z_all[idx]
        cov = np.isfinite(z).all(axis=1)
        rv, raw, har, H = (npz[f"{k}_{y}"] for k in ("rv", "raw", "har", "H"))
        sig = np.exp(W0 * raw + (1.0 - W0) * har) * np.sqrt(H)
        zs.append(z[cov])
        rs.append(np.maximum(rv[cov] ** 2, 1e-18) / np.maximum(sig[cov] ** 2, 1e-18))
    if not zs:
        return None
    zh = np.vstack(zs)
    mu, sd = zh.mean(axis=0), zh.std(axis=0)
    per_fold = [fit_beta(standardise(z, mu, sd, False), r) for z, r in zip(zs, rs)]
    Bm = np.array(per_fold)
    beta = np.array([shrink(Bm[:, j], w0=0.0, sigma_w=SIGMA_B)[0]
                     for j in range(Bm.shape[1])])
    return {"beta": beta, "mu": mu, "sd": sd, "n_hist": len(zs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2-confirm: IV correction, full pipeline")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--components", type=Path,
                    default=Path("model/artifacts/blend_ceiling.npz"))
    ap.add_argument("--iv", type=Path,
                    default=Path("model/artifacts/iv_features.parquet"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--all-hours", action="store_true",
                    help="score all 24 anchor hours at H = 19 instead of the "
                         "production slice; see BENCHMARK.md 22")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/iv_confirm.json"))
    a = ap.parse_args(argv)

    if not a.components.exists():
        raise SystemExit(f"REFUSING: {a.components} not found; run blend_ceiling first")
    npz = np.load(a.components)
    years = sorted({int(k.split("_")[-1]) for k in npz.files})

    ep, X = load_all(a.artifacts)
    iv = pd.read_parquet(a.iv)
    if not np.array_equal(iv["anchor_ts"].to_numpy(np.int64),
                          ep["anchor_ts"].to_numpy(np.int64)):
        raise SystemExit(
            "REFUSING: iv_features.parquet is not positionally aligned with "
            "episodes.parquet; the join this file relies on is broken")
    Z_all = iv[COLS].to_numpy(np.float64)
    covered_rows = np.flatnonzero(np.isfinite(Z_all).all(axis=1))
    roll = _rotation_rows(covered_rows, iv["anchor_ts"].to_numpy(np.int64))
    pos_of = {int(r): i for i, r in enumerate(covered_rows)}

    spike = causal_spike_flag(ep)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    raw_har = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(m):
        lo, hi = np.quantile(raw_har[m], [0.005, 0.995])
        return np.maximum(np.clip(raw_har, lo, hi), 1e-12)

    prod = (ep["H"] == S.PROD_H).to_numpy() if a.all_hours else None
    print(f"slice: {'ALL 24 anchor hours at H=19' if a.all_hours else 'production'}"
          f"   features {COLS}   Bonferroni alpha = 0.05/{N_TESTS}\n")

    def rotated(rows):
        out = np.full((len(rows), len(COLS)), np.nan)
        for j, r in enumerate(rows):
            i = pos_of.get(int(r))
            if i is not None:
                out[j] = Z_all[covered_rows[(i + roll) % len(covered_rows)]]
        return out

    def make_shift(fit, rot: bool):
        """A post_shift_fn: the E2c correction on the episodes in `mask`,
        and exactly zero where IV is absent -- a missing feature must leave the
        forecast untouched, not push it somewhere."""
        def fn(mask, _train_mask):
            rows = np.flatnonzero(np.asarray(mask, bool))
            z = rotated(rows) if rot else Z_all[rows]
            cov = np.isfinite(z).all(axis=1)
            out = np.zeros(len(rows))
            if cov.any():
                zz = standardise(z[cov], fit["mu"], fit["sd"], False)
                out[cov] = np.clip(zz @ fit["beta"], -EXP_BOUND, EXP_BOUND)
            return out
        return fn

    recs = []
    for f in folds:
        y = f["year"]
        fit = fit_history_beta(npz, years, y, Z_all)
        fit_p = fit_history_beta(npz, years, y, Z_all, rotate=roll,
                                 covered_rows=covered_rows)
        if fit is None or fit_p is None:
            print(f"  {y}  (no prior fold -- excluded)")
            continue
        line = {"year": y, "n_hist": fit["n_hist"], "beta": fit["beta"].tolist()}
        for nm, shift in (("control", None),
                          ("treated", make_shift(fit, False)),
                          ("placebo", make_shift(fit_p, True))):
            t0 = time.time()
            r = B.run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                           sigma_ref_fn=sig_fn, prod_override=prod,
                           post_shift_fn=shift)
            if r is None:
                print(f"  {y}  {nm:8} SKIPPED"); line = None; break
            pe = r["per_episode"]
            idx = pe["test_idx"]
            cov = np.isfinite(Z_all[idx]).all(axis=1)
            q = qlike(pe["rv"], pe["sigma_med"])
            sp = spike[idx] & cov
            cm = (~spike[idx]) & cov
            line[nm] = {
                "qlike_pooled": float(q[cov].mean()),
                "qlike_spike": float(q[sp].mean()) if sp.any() else float("nan"),
                "qlike_calm": float(q[cm].mean()),
                "tail_mcb": tail_mcb(r["rows"]),
                # Coverage is guarded on |hit_rate - alpha|, NOT on p_cc. A
                # p-value is evidence against a null, not a quality score: it
                # falls when n rises at fixed calibration error, so "p_cc must
                # not drop" would punish a wider slice for being wider. The
                # calibration ERROR is the quantity that means what the guard
                # says it means.
                "cov_err": float(np.mean([
                    abs(r["christoffersen"][sd]["hit_rate"] - 0.05)
                    for sd in ("up", "dn")])),
                "p_cc_up": float(r["christoffersen"]["up"]["p_cc"]),
                "p_cc_dn": float(r["christoffersen"]["dn"]["p_cc"]),
                "n_cov": int(cov.sum()), "n_test": int(len(idx)),
                "n_spike": int(sp.sum())}
            s_ = line[nm]
            print(f"  {y}  {nm:8} pooled {s_['qlike_pooled']:.4f}  "
                  f"spike {s_['qlike_spike']:.4f}  calm {s_['qlike_calm']:.4f}  "
                  f"tailMCB {s_['tail_mcb']:.4f}  cov {s_['n_cov']}/{s_['n_test']}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        if line and all(k in line for k in ("control", "treated", "placebo")):
            recs.append(line)

    if len(recs) < 2:
        print("\nfewer than 2 complete folds -- no verdict"); return 1

    alpha = 0.05 / N_TESTS
    out = {"folds": recs, "cols": COLS, "n_tests": N_TESTS,
           "ci_level": 1.0 - alpha, "all_hours": bool(a.all_hours)}
    print(f"\nCIs at the Bonferroni-adjusted level ({100*(1-alpha):.2f}%, "
          f"{N_TESTS} tests in the IV family)")
    print(f"{'arm':>9} {'quantity':>13} {'delta':>11} {'CI':>26} {'signs':>8}")
    for arm in ("treated", "placebo"):
        out[arm] = {}
        for key in ("qlike_pooled", "qlike_spike", "qlike_calm", "tail_mcb",
                    "cov_err"):
            d = np.array([r[arm][key] - r["control"][key] for r in recs])
            d = d[np.isfinite(d)]
            if len(d) < 2:
                out[arm][key] = {"delta": None, "ci": None}; continue
            ci = mean_ci(d, seed=53, alpha=alpha)
            out[arm][key] = {"delta": ci["mean"], "ci": ci["ci95"],
                             "n_negative": ci["n_negative"],
                             "n_positive": ci["n_positive"]}
            print(f"{arm:>9} {key:>13} {ci['mean']:+11.5f}   "
                  f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]   "
                  f"{ci['n_negative']}-/{ci['n_positive']}+")

    margin = np.array([(r["treated"]["qlike_pooled"] - r["control"]["qlike_pooled"])
                       - (r["placebo"]["qlike_pooled"] - r["control"]["qlike_pooled"])
                       for r in recs])
    ci_m = mean_ci(margin, seed=59, alpha=alpha)
    out["placebo_margin"] = {"delta": ci_m["mean"], "ci": ci_m["ci95"]}
    print(f"\n  placebo margin (treated - placebo, pooled): {ci_m['mean']:+.5f}   "
          f"[{ci_m['ci95'][0]:+.5f}, {ci_m['ci95'][1]:+.5f}]")

    base_calm = float(np.mean([r["control"]["qlike_calm"] for r in recs]))
    calm_pct = 100.0 * out["treated"]["qlike_calm"]["delta"] / max(base_calm, 1e-12)
    primary = ci_excludes_zero(out["treated"]["qlike_pooled"]["ci"], -1)
    g_mcb = not ci_excludes_zero(out["treated"]["tail_mcb"]["ci"], +1)
    g_spike = not ci_excludes_zero(out["treated"]["qlike_spike"]["ci"], +1)
    g_calm = calm_pct <= 1.0
    g_plac = ci_excludes_zero(ci_m["ci95"], -1)
    # Conditional coverage: the treated arm's mean |hit_rate - 5%| must not be
    # significantly worse than the control's. Pre-registered as "must not
    # degrade on either side"; `cov_err` averages the two sides, and the CI is
    # what makes "not degrade" testable rather than a point comparison.
    g_cov = not ci_excludes_zero(out["treated"]["cov_err"]["ci"], +1)
    ok = primary and g_mcb and g_spike and g_calm and g_plac and g_cov
    out["verdict"] = "ADOPT" if ok else "REJECT"
    print(f"\n--- pre-registered rule ---")
    print(f"  PRIMARY pooled QLIKE CI clears zero favourably   : {primary}")
    print(f"  GUARD   deep-tail barrier MCB not worse          : {g_mcb}")
    print(f"  GUARD   spike QLIKE not worse                    : {g_spike}")
    print(f"  GUARD   calm QLIKE within 1% ({calm_pct:+.2f}%)     : {g_calm}")
    print(f"  GUARD   beats the placebo                        : {g_plac}")
    print(f"  GUARD   conditional coverage error not worse     : {g_cov}"
          f"   (delta {out['treated']['cov_err']['delta']:+.5f})")
    print(f"  -> {out['verdict']}")

    rep = P.Report()
    rep.add(P.check_ci_is_defined(out["treated"]["qlike_pooled"]["ci"], "pooled"))
    rep.add(P.check_not_a_coin_flip(
        [r["treated"]["qlike_pooled"] - r["control"]["qlike_pooled"] for r in recs],
        "pooled delta"))
    rep.add(P.check_arms_matched(
        {"control": sum(r["control"]["n_cov"] for r in recs),
         "treated": sum(r["treated"]["n_cov"] for r in recs),
         "placebo": sum(r["placebo"]["n_cov"] for r in recs)}, what="scored episodes"))
    rep.add(P.check_rule_satisfiable(2, len(recs), "folds"))
    print("\n--- research/pitfalls on this experiment ---")
    print(rep.render())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
