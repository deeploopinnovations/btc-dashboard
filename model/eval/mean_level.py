"""
eval/mean_level.py
=====================================================================
P2-mean-level: is NOCTUA's level bias exactly the median-vs-mean gap, and does
the model's OWN per-episode conversion survive the barrier battery that a
fitted constant failed?

THE COINCIDENCE THAT STARTED THIS

  * P2-scorecard-rescaled: NOCTUA has the worst calibration ratio in the zoo at
    every horizon -- 1.4330 / 1.4642 / 1.4569 / 1.1660 -- and the scalar that
    fixes it is c = 1.198 / 1.212 / 1.213 / 1.089. Equalise level and NOCTUA is
    the best teacher in the zoo at H=1 and H=6.
  * `noctua/infer.py` has always returned `sigma_mean` beside `sigma_med`,
    built from the model's own atom grid, and records its mean/median ratio as
    1.205.

1.205 against a fitted 1.198-1.213. If that is not coincidence then the level
bias IS the median-vs-mean gap, the model already computes the correction from
its own predicted quantiles, and the correction is PER EPISODE with zero fitted
parameters.

infer.py's docstring also asserts the mean "scores worse". No run accompanies
that assertion and it is inconsistent with the arithmetic above, so it is
treated here as the hypothesis under test rather than as evidence (R34).

WHY THE CONSTANT ARM IS NOT OPTIONAL

P2-scale-v2 already applied a fitted constant to this model: QLIKE improved
9.7% pooled and 33% on spikes, and ALL SIX barrier metrics degraded. If M1 and
M2 are indistinguishable then this is that experiment again under a new name.
M2 is in the family to force M1 to show it does something a constant does not.

AND WHY THE SHUFFLE CONTROL IS NOT OPTIONAL EITHER

delta_i has a mean and a spread. M3 permutes delta_i across episodes within the
fold, preserving the mean and the marginal distribution exactly and destroying
only the alignment. If M3 matches M1, the gain was the mean and the spread was
decoration.

TWO PASSES, AND EXACTLY TWO

delta_i is read off the model's own prediction, so it does not exist before the
model runs. Pass A predicts unshifted and records sigma_mean/sigma_med; pass B
re-runs with those deltas. The ratio is INVARIANT to a uniform shift -- qa, the
atoms and the median all move together -- so one iteration is exact, not
approximate. run_fold's assertion that the achieved log-level shift equals the
requested one to 1e-6 is retained and is a pass condition.

    python -m model.eval.mean_level --self-test
    python -m model.eval.mean_level
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                          # noqa: E402
from eval.direction import mean_ci                                           # noqa: E402
from eval.scale_adopt import barrier_cols                                    # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                         # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all                                            # noqa: E402

PROD_H = 19                       # the production slice, same as P2-scale-v2
ARMS = ("M1", "M2", "M3")
N_FAMILY = 3
RATIO_LO, RATIO_HI = 0.95, 1.05
SLICE_TOL = 0.02


def deltas(pe) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-episode log(sigma_mean / sigma_med) on the test and calib slices.

    Uses no target. It is a function of the predictive distribution alone, so
    computing it on the test slice is not a fit.
    """
    d_te = np.log(np.maximum(pe["sigma_mean"], 1e-300)
                  / np.maximum(pe["sigma_med"], 1e-300))
    d_ca = np.log(np.maximum(pe["sigma_mean_cal"], 1e-300)
                  / np.maximum(pe["sigma_cal"], 1e-300))
    return d_te, d_ca, float(np.mean(d_ca))       # the constant is CALIB-only


def self_test() -> int:
    ok = []
    rng = np.random.default_rng(0)
    n = 400
    med = np.exp(rng.normal(-4, 0.4, n))
    r = np.exp(rng.normal(0.19, 0.05, n))          # ~1.205 with spread
    pe = {"sigma_med": med, "sigma_mean": med * r,
          "sigma_cal": med, "sigma_mean_cal": med * r}
    d_te, d_ca, k = deltas(pe)
    ok.append(("delta-recovers-ratio",
               float(np.max(np.abs(np.exp(d_te) - r))) < 1e-12,
               "exp(delta) is exactly the mean/median ratio"))
    ok.append(("constant-is-calib-mean", abs(k - float(np.mean(d_ca))) < 1e-15,
               "M2's constant is the calib mean of delta, nothing else"))
    # the shuffle must preserve the mean and change the alignment
    perm = rng.permutation(d_te)
    ok.append(("shuffle-preserves-mean",
               abs(float(perm.mean() - d_te.mean())) < 1e-12
               and not np.array_equal(perm, d_te),
               "M3 keeps the mean exactly and destroys the alignment"))
    # a uniform shift must leave the ratio unchanged -- the two-pass argument
    sh = 0.37
    pe2 = {k2: v * np.exp(sh) for k2, v in pe.items()}
    d2, _, _ = deltas(pe2)
    ok.append(("ratio-shift-invariant",
               float(np.max(np.abs(d2 - d_te))) < 1e-12,
               "a uniform level shift leaves delta unchanged, so one pass is exact"))
    print("mean_level self-test")
    for nm, good, msg in ok:
        print(f"  [{'ok ' if good else 'FAIL'}] {nm}: {msg}")
    bad = [o for o in ok if not o[1]]
    print(f"\n{len(ok)-len(bad)}/{len(ok)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-mean-level")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/mean_level.json"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    Hall = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    def sig_fn(train_mask):
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    alpha = 0.05 / N_FAMILY
    print(f"P2-mean-level   production slice H={PROD_H}   seeds={a.seeds}")
    print(f"family {N_FAMILY} -> {100*(1-alpha):.2f}% intervals")
    print("pass A: no shift, read delta off the model. pass B: three shifts.\n")

    acc = []
    for f in folds:
        t0 = time.time()
        r0 = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn)
        if r0 is None:
            print(f"  fold {f['year']}: skipped"); continue
        pe0 = r0["per_episode"]
        d_te, d_ca, kconst = deltas(pe0)
        rng = np.random.default_rng(7000 + f["year"])
        # Deltas are carried in a FULL-LENGTH array indexed by episode, not
        # matched to a slice by its length. Length matching would work until
        # the day the calib and test slices happened to be the same size, and
        # then it would silently apply one slice's corrections to the other.
        # The control permutes WITHIN each slice, so it preserves that slice's
        # mean exactly and destroys only the alignment.
        d_full = np.full(len(ep), np.nan)
        p_full = np.full(len(ep), np.nan)
        d_full[pe0["cal_idx"]] = d_ca
        d_full[pe0["test_idx"]] = d_te
        p_full[pe0["cal_idx"]] = rng.permutation(d_ca)
        p_full[pe0["test_idx"]] = rng.permutation(d_te)

        def shift_for(arm, mask, m_tr, _d=d_full, _p=p_full, _k=kconst):
            n = int(mask.sum())
            if arm == "M2":
                return np.full(n, _k)
            v = (_d if arm == "M1" else _p)[mask]
            if not np.isfinite(v).all():
                raise SystemExit(
                    f"REFUSING: {int((~np.isfinite(v)).sum()):,} of {n:,} "
                    f"episodes in this slice have no recorded delta. Filling "
                    f"them would be inventing a correction (R43).")
            return v

        row = {"year": f["year"], "rv": pe0["rv"], "s0": pe0["sigma_med"],
               "kconst": kconst, "d_sd": float(np.std(d_te)),
               "vol0": r0["vol"], "bar0": barrier_cols(r0["rows"]),
               "rv_cal": pe0["rv_cal"], "sig_cal": pe0["sigma_cal"]}
        okf = True
        for arm in ARMS:
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, sigma_ref_fn=sig_fn,
                          post_shift_fn=lambda m, mt, _a=arm: shift_for(_a, m, mt))
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], pe0["test_idx"]):
                raise SystemExit(
                    f"REFUSING: fold {f['year']} arm {arm} scored different "
                    f"episodes than the reference. The contrast is not paired.")
            row[f"s_{arm}"] = pe1["sigma_med"]
            row[f"vol_{arm}"] = r1["vol"]
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
        if not okf:
            print(f"  fold {f['year']}: skipped"); continue
        # M1 must have achieved exactly the model's own mean level
        got = float(np.max(np.abs(np.log(row["s_M1"] / pe0["sigma_med"]) - d_te)))
        if got > 1e-6:
            raise SystemExit(
                f"REFUSING: fold {f['year']} asked M1 for the per-episode "
                f"median-to-mean shift and the achieved shift differs by "
                f"{got:.3e}. The arm being scored is not the arm described.")
        acc.append(row)
        print(f"  fold {f['year']}  k={kconst:.4f} (x{np.exp(kconst):.4f})  "
              f"sd(delta)={row['d_sd']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    if not acc:
        print("no usable folds"); return 1

    rv = np.concatenate([r["rv"] for r in acc])
    s0 = np.concatenate([r["s0"] for r in acc])
    q0 = qlike_vec(rv, s0)
    hi = rv >= np.quantile(rv, 0.95)
    L = block_len_for(PROD_H, len(q0))
    kbar = float(np.mean([r["kconst"] for r in acc]))
    sdbar = float(np.mean([r["d_sd"] for r in acc]))

    print("\n" + "=" * 96)
    print(f"PRODUCTION SLICE  {len(q0):,} test episodes  blocks of {L}")
    print(f"mean median-to-mean factor {np.exp(kbar):.4f}   "
          f"mean sd(delta) {sdbar:.4f}   "
          f"(the fitted scalars in P2-scorecard-rescaled were 1.198-1.213)")
    print("=" * 96)
    print(f"  {'arm':>4} {'QLIKE':>9} {'vs M0':>10} {'rel %':>8}   paired CI")
    print(f"  {'M0':>4} {np.nanmean(q0):9.5f} {'-':>10} {'-':>8}   (reference)")
    out = {"family_size": N_FAMILY, "alpha": alpha, "block_len": L,
           "n_test": int(len(q0)), "k_mean": kbar, "k_factor": float(np.exp(kbar)),
           "delta_sd": sdbar, "arms": {}}
    qs = {}
    for arm in ARMS:
        s1 = np.concatenate([r[f"s_{arm}"] for r in acc])
        q1 = qlike_vec(rv, s1)
        qs[arm] = (q1, s1)
        d = q0 - q1
        g = np.isfinite(d)
        ci = mean_ci(d[g], alpha=alpha, block_len=L)
        rel = 100 * float(np.nanmean(d)) / float(np.nanmean(q0))
        tag = {"M1": "per-episode", "M2": "CONSTANT", "M3": "CONTROL"}[arm]
        print(f"  {arm:>4} {np.nanmean(q1):9.5f} {np.nanmean(d):+10.5f} "
              f"{rel:+8.2f}   [{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
              f"  <- {tag}")
        out["arms"][arm] = {"qlike": float(np.nanmean(q1)),
                            "vs_m0": float(np.nanmean(d)), "rel_pct": rel,
                            "ci": list(ci["ci95"]),
                            "clears": bool(ci["ci95"][0] > 0),
                            "calib_ratio": float(np.nanmean(
                                rv ** 2 / np.maximum(s1, 1e-12) ** 2)),
                            "spike": float(np.nanmean(q1[hi])),
                            "calm": float(np.nanmean(q1[~hi]))}

    # M1 against M2 directly: does the per-episode variation buy anything?
    dm = qlike_vec(rv, qs["M2"][1]) - qlike_vec(rv, qs["M1"][1])
    gm = np.isfinite(dm)
    cim = mean_ci(dm[gm], alpha=alpha, block_len=L)
    print(f"\n  M1 vs M2 (per-episode vs the constant): {np.nanmean(dm):+.5f}  "
          f"CI [{cim['ci95'][0]:+.5f}, {cim['ci95'][1]:+.5f}]"
          + ("  M1 BETTER" if cim["ci95"][0] > 0 else "  not separated"))
    out["m1_vs_m2"] = {"gain": float(np.nanmean(dm)), "ci": list(cim["ci95"]),
                       "separated": bool(cim["ci95"][0] > 0)}

    ratio0 = float(np.nanmean(rv ** 2 / np.maximum(s0, 1e-12) ** 2))
    print(f"\n  calib ratio  M0 {ratio0:.4f}  ->  "
          + "  ".join(f"{a_} {out['arms'][a_]['calib_ratio']:.4f}" for a_ in ARMS)
          + f"   guard [{RATIO_LO}, {RATIO_HI}]")
    print(f"  spike        M0 {np.nanmean(q0[hi]):.5f}  ->  "
          + "  ".join(f"{a_} {out['arms'][a_]['spike']:.5f}" for a_ in ARMS))
    print(f"  calm         M0 {np.nanmean(q0[~hi]):.5f}  ->  "
          + "  ".join(f"{a_} {out['arms'][a_]['calm']:.5f}" for a_ in ARMS))

    print("\n  --- THE BARRIER BATTERY, verbatim from P2-scale-v2 ---")
    keys = sorted(set().union(*[set(r["bar0"]) for r in acc]))
    print(f"  {'metric':>8} {'M0':>10} " + " ".join(f"{a_:>10}" for a_ in ARMS))
    bar = {}
    for k in keys:
        b = float(np.mean([r["bar0"][k] for r in acc if k in r["bar0"]]))
        vals = {}
        for arm in ARMS:
            v = float(np.mean([r[f"bar_{arm}"][k] for r in acc
                               if k in r.get(f"bar_{arm}", {})]))
            vals[arm] = {"value": v,
                         "better": bool(v > b) if k == "DSC" else bool(v < b),
                         "rel_pct": 100 * (v - b) / abs(b)}
        bar[k] = {"m0": b, **vals}
        print(f"  {k:>8} {b:10.6f} "
              + " ".join(f"{vals[a_]['value']:10.6f}" for a_ in ARMS)
              + "    " + " ".join(
                  f"{a_}:{'+' if vals[a_]['better'] else '-'}" for a_ in ARMS))

    m1 = out["arms"]["M1"]
    guards = {
        "ratio_in_band": bool(RATIO_LO <= m1["calib_ratio"] <= RATIO_HI),
        "spike_not_worse_2pct":
            bool(m1["spike"] <= float(np.nanmean(q0[hi])) * (1 + SLICE_TOL)),
        "calm_not_worse_2pct":
            bool(m1["calm"] <= float(np.nanmean(q0[~hi])) * (1 + SLICE_TOL)),
        "dsc_not_worse": bool(bar.get("DSC", {}).get("M1", {}).get("better", False)),
        "brier_not_worse": bool(bar.get("brier", {}).get("M1", {}).get("better", True)),
        "control_does_not_clear": bool(not out["arms"]["M3"]["clears"]),
    }
    print("\n  --- pre-registered guards ---")
    for k, v in guards.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    primary = out["arms"]["M1"]["clears"]
    print(f"  [{'PASS' if primary else 'FAIL'}] primary: M1 beats M0")
    verdict = "ADOPTABLE" if primary and all(guards.values()) else "NOT ADOPTABLE"
    print(f"\n  VERDICT: {verdict}")

    out["barriers"] = bar
    out["guards"] = guards
    out["primary_clears"] = bool(primary)
    out["calib_ratio_m0"] = ratio0
    out["verdict"] = verdict
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
