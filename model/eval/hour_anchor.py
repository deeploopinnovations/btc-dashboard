"""
eval/hour_anchor.py
=====================================================================
P4-hour-anchor: give the served anchor the one thing it cannot see -- the
clock.

THE DEFECT

The served forecast is 0.25 * network + 0.75 * log_har_cal in log space
(infer.BLEND_W). log_har_cal regresses on har_1d / har_5d / har_22d, the
horizon and the weekend share -- and NOTHING that says what time of day the
forecast window covers. BTC volatility has a strong intraday cycle, so a 19-hour
window opened at 17:00 UTC (mostly the Asian night) and one opened at 05:00
(mostly the European and US day) get the same anchor. Measured on the fold
calib slices at H = 19, the anchor's median log(RV / sigma) runs from +0.04 at
09:00 to -0.108 at 17:00 -- the product's own anchor hour is where it
over-forecasts most.

Serving's trailing correction (serve/adaptive.volatility_correction) does not
remove it: it takes the median RV/sigma over settled anchors at a 6-hour stride
counted back from anchor - H, which for a 17:00 anchor at H = 19 lands on 22:00,
04:00, 10:00 and 16:00 -- never 17:00. On the calib slices the residual it
leaves at 17:00 is -0.03 to -0.12, largest in the most recent folds.

WHAT IS BUILT

    season_fwd(a, H) = 0.5 * log( mean_{k=0..H-1} v((a + k) mod 24) / mean_h v(h) )

the log of the average seasonal VARIANCE factor over the forecast window's own
clock hours, centred so a whole number of days scores 0. v(h) is estimated on
each fold's TRAINING slice only, from har_1h (the log vol of the hour ending at
the anchor, so an anchor at hour a measures clock hour a - 1). The hour-aware
anchor is log_har_cal's own OLS plus this one column, fitted on the same
training rows, weights and target. It enters the pipeline through
post_shift_fn as (1 - w) * (anchor_new - log_har_cal), so Stage B, the committee
and every barrier curve inherit it -- NOT demeaned: the level at the anchor
hour is the point.

THE CONTROL. `P` uses the SAME 24 hourly values, permuted: of 2,000 seeded
permutations, the one whose profile is least correlated with the true one.
Same values, same amplitude, same fitted-coefficient freedom, no clock. (A
12-hour roll was the first design and was rejected before any run: the
profile is close to one sinusoid, a half-period roll is close to its
negation, and OLS's free sign recovered the clock -- coefficients -0.8 to
-1.2.) If P helps as much as A, the gain is a regressor absorbing level.

SERVING FIDELITY. Every scored arm carries serving's trailing factor, computed
the way serving computes it -- 60 days, 6-hour stride, median of RV/sigma_med,
clip [0.70, 1.40], at least 20 episodes -- from THAT ARM's own forecasts. The
benchmark's M0 has never had it; M0s does, and is the baseline here. One extra
run per fold (`F`, the factor-hour anchors) supplies the M0 forecasts the factor
needs; an arm's forecasts there are M0's times exp(its own shift), exact
because run_fold asserts post_shift_fn moves the median by what it asks.

    python -m model.eval.hour_anchor --selftest
    python -m model.eval.hour_anchor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                    # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                   # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import infer as I                                          # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

HOUR = 3600
PROD_H, PROD_A = 19, 17
FLOOR_LOG = np.log(1e-5)          # above the 1e-6 missing-hour floor
# serve/adaptive.py, restated so a drift there is a visible diff here
FAC_WINDOW_D, FAC_STRIDE_H, FAC_MIN = 60, 6, 20
FAC_LO, FAC_HI = 0.70, 1.40
METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
HIGHER_BETTER = {"DSC"}
ARMS = ("M0", "M0s", "As", "Ps", "Ams")
CONTRASTS = (("As", "M0s"), ("Ps", "M0s"), ("Ams", "M0s"), ("M0s", "M0"))


# ---------------------------------------------------------------- season ----
def hour_profile(har_1h: np.ndarray, anchor_hour: np.ndarray) -> np.ndarray:
    """log-vol seasonal offset s(h) for each CLOCK hour h, centred to mean 0.

    har_1h at an anchor of hour a is the vol of the hour ENDING at a, i.e.
    clock hour a - 1.
    """
    # log(1e-6) is the encoding of a MISSING hour (P2-floor-defect), not a
    # quiet one; averaging it in would drag every affected clock hour down
    ok = np.isfinite(har_1h) & (np.asarray(har_1h, np.float64) > FLOOR_LOG)
    clock = (np.asarray(anchor_hour)[ok] - 1) % 24
    v = np.asarray(har_1h, np.float64)[ok]
    s = np.array([v[clock == h].mean() for h in range(24)])
    return s - s.mean()


def placebo_profile(profile: np.ndarray, n_try: int = 2000, seed: int = 0):
    """The permutation of the 24 values least correlated with the original."""
    rng = np.random.default_rng(seed)
    best, best_c = None, np.inf
    for _ in range(n_try):
        pp = profile[rng.permutation(24)]
        c = abs(float(np.corrcoef(pp, profile)[0, 1]))
        if c < best_c:
            best, best_c = pp, c
    return best


def season_fwd(profile: np.ndarray, anchor_hour: np.ndarray, H: np.ndarray):
    """0.5 log of the mean seasonal variance factor over [a, a + H), centred."""
    var = np.exp(2.0 * np.asarray(profile, np.float64))
    var = var / var.mean()
    a = np.asarray(anchor_hour, np.int64) % 24
    Hh = np.asarray(H, np.int64)
    # cumulative sum over enough periodic copies for the longest window
    reps = int(np.ceil((Hh.max() + 24) / 24)) + 1
    c = np.concatenate([[0.0], np.cumsum(np.tile(var, reps))])
    tot = c[a + Hh] - c[a]
    return 0.5 * np.log(tot / Hh)


# ------------------------------------------------------- served factor ------
def factor_anchor_ts(t: int, H: int = PROD_H) -> np.ndarray:
    """The anchor times serve/adaptive.volatility_correction reads for an
    anchor at t: rows [last - window, last) at the stride, last = t - H."""
    last = int(t) - H * HOUR
    first = last - FAC_WINDOW_D * 24 * HOUR
    return np.arange(first, last, FAC_STRIDE_H * HOUR, dtype=np.int64)


def served_log_factor(t_query, ts_hist, rv_hist, sig_hist) -> np.ndarray:
    """log of serving's trailing factor at each query anchor, from history
    forecasts keyed by anchor time. Missing anchors are simply absent."""
    lut = dict(zip(np.asarray(ts_hist, np.int64).tolist(),
                   zip(np.asarray(rv_hist, float), np.asarray(sig_hist, float))))
    out = np.zeros(len(t_query))
    for i, t in enumerate(np.asarray(t_query, np.int64)):
        pairs = [lut[a] for a in factor_anchor_ts(int(t)).tolist() if a in lut]
        r = np.array([p[0] / p[1] for p in pairs
                      if np.isfinite(p[0]) and np.isfinite(p[1])
                      and p[0] > 0 and p[1] > 0])
        if len(r) < FAC_MIN:
            continue                                  # serving applies 1.0
        f = float(np.median(r))
        out[i] = np.log(np.clip(f, FAC_LO, FAC_HI))
    return out


# ------------------------------------------------------------ anchors -------
def fold_anchors(ep, X, fold):
    """log_har_cal as run_fold fits it, and the hour-aware / placebo anchors
    fitted on the SAME rows, weights and target. Returns log-rate arrays over
    every episode (NaN where features are missing)."""
    fin = np.isfinite(X.to_numpy()).all(1)
    m_tr = np.asarray(fold["train"], bool) & fin
    wtr = S.sample_weights(ep, m_tr)
    H = ep.H.to_numpy(np.float64)
    y = B.har_target(ep.RV.to_numpy(), H)
    bl = B.fit_vol_baselines(X[m_tr], y[m_tr], wtr)
    ah = ep["anchor_hour"].to_numpy()
    # one row per anchor: the table carries H = 6/12/19/24 and every horizon
    # repeats the anchor's features, so any single horizon is one per anchor
    one = m_tr & (ep.H == PROD_H).to_numpy()
    prof = hour_profile(X["har_1h"].to_numpy(np.float64)[one], ah[one])
    cols = B.VOL_BASELINES["log_har_cal"] + ["season_fwd"]
    out = {"lhc": np.full(len(ep), np.nan), "A": np.full(len(ep), np.nan),
           "P": np.full(len(ep), np.nan), "profile": prof, "coef": {}}
    out["lhc"][fin] = bl["log_har_cal"].predict(X[fin])
    for arm, p in (("A", prof), ("P", placebo_profile(prof))):
        X2 = X[cols[:-1]].copy()
        X2["season_fwd"] = season_fwd(p, ah, H)
        ols = B.OLS(cols).fit(X2[m_tr], y[m_tr], wtr)
        out[arm][fin] = ols.predict(X2[fin])
        out["coef"][arm] = float(ols.beta[-1])
    return out


def better(met, a, b):
    return a > b if met in HIGHER_BETTER else a < b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hour-aware anchor")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/hour_anchor.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = {(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}
    fac_mask = at19 & np.isin(ah, sorted(fac_hours))
    n_family = len(CONTRASTS) * len(METRICS)
    alpha = 0.05 / n_family
    w = I.BLEND_W
    print(f"P4-hour-anchor   production slice H={PROD_H} @ {PROD_A}:00   "
          f"seeds={a.seeds}   served-factor hours {sorted(fac_hours)}")
    print(f"family {n_family} -> {100*(1-alpha):.3f}% intervals\n")

    rows = []
    for f in folds:
        anc = fold_anchors(ep, X, f)
        sh = {"M0": np.zeros(len(ep)),
              "A": np.nan_to_num((1 - w) * (anc["A"] - anc["lhc"])),
              "P": np.nan_to_num((1 - w) * (anc["P"] - anc["lhc"])),
              # the MIRROR: the hour-aware shift with its sign flipped
              "Am": -np.nan_to_num((1 - w) * (anc["A"] - anc["lhc"]))}
        rF = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=fac_mask)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if rF is None or r0 is None:
            continue
        peF, pe0 = rF["per_episode"], r0["per_episode"]
        # M0 forecasts at the factor anchors: calib (all hours) + test (F run)
        ci, ti = pe0["cal_idx"], peF["test_idx"]
        h_idx = np.concatenate([ci, ti])
        h_rv = np.concatenate([pe0["rv_cal"], peF["rv"]])
        h_sig = np.concatenate([pe0["sigma_cal"], peF["sigma_med"]])
        test_i = pe0["test_idx"]
        row = {"year": f["year"], "coef": anc["coef"],
               "profile_17": float(anc["profile"][16]),
               "q_M0": qlike_vec(pe0["rv"], pe0["sigma_mean"]),
               "bar_M0": barrier_cols(r0["rows"])}
        okf = True
        for arm, base in (("M0s", "M0"), ("As", "A"), ("Ps", "P"), ("Ams", "Am")):
            sig_arm = h_sig * np.exp(sh[base][h_idx])
            lf = served_log_factor(ts_all[test_i], ts_all[h_idx], h_rv, sig_arm)
            full = sh[base].copy()
            full[test_i] += lf
            row[f"logfac_{arm}"] = [float(lf.mean()), float(lf.min()), float(lf.max())]
            fn = (lambda mask, _mt, _s=full: _s[mask])
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, post_shift_fn=fn)
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], test_i):
                raise SystemExit("REFUSING: arms scored different episodes")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
            row[f"bias_{arm}"] = float(np.median(np.log(pe1["rv"] / pe1["sigma_med"])))
        row["bias_M0"] = float(np.median(np.log(pe0["rv"] / pe0["sigma_med"])))
        if okf:
            rows.append(row)
            print(f"  fold {f['year']}: season coef A {anc['coef']['A']:+.2f} "
                  f"P {anc['coef']['P']:+.2f}   median log(RV/sig_med) "
                  f"M0 {row['bias_M0']:+.3f} M0s {row['bias_M0s']:+.3f} "
                  f"As {row['bias_As']:+.3f} Ps {row['bias_Ps']:+.3f} "
                  f"Ams {row['bias_Ams']:+.3f}", flush=True)

    if not rows:
        print("no complete folds"); return 1

    per_fold = {m: {k: [r[f"bar_{k}"].get(m) for r in rows] for k in ARMS}
                for m in METRICS}
    print(f"\n{'metric':>9} " + " ".join(f"{k:>10}" for k in ARMS))
    for m in METRICS:
        print(f"{m:>9} " + " ".join(f"{np.mean(per_fold[m][k]):10.6f}"
                                    for k in ARMS))
    ci_out, res = {}, {c: {"better": [], "worse": []} for c in CONTRASTS}
    print(f"\n{'metric':>9} {'contrast':>9} {'delta':>12} "
          f"{'CI (corrected)':>28} {'folds':>6}")
    for m in METRICS:
        sgn = 1.0 if m in HIGHER_BETTER else -1.0
        for c, o in CONTRASTS:
            d = sgn * (np.asarray(per_fold[m][c], float)
                       - np.asarray(per_fold[m][o], float))
            lo, hi = mean_ci(d, alpha=alpha)["ci95"]
            ci_out[f"{m}_{c}_vs_{o}"] = {"delta": float(d.mean()),
                                         "ci95": [float(lo), float(hi)],
                                         "n_better": int((d > 0).sum())}
            if lo > 0:
                res[(c, o)]["better"].append(m)
            if hi < 0:
                res[(c, o)]["worse"].append(m)
            print(f"{m:>9} {c+'-'+o:>9} {d.mean():+12.6f} "
                  f"[{lo:+12.6f}, {hi:+12.6f}] {int((d>0).sum()):>3}/{len(d)}")

    L = block_len_for(PROD_H, sum(len(r["q_M0"]) for r in rows))
    print("\nper-episode QLIKE (sigma_mean):")
    qci = {}
    for c, o in CONTRASTS:
        qo = np.concatenate([r[f"q_{o}"] for r in rows])
        qc = np.concatenate([r[f"q_{c}"] for r in rows])
        dd = qo - qc; g = np.isfinite(dd)
        lo, hi = mean_ci(dd[g], alpha=alpha, block_len=L)["ci95"]
        qci[f"{c}_vs_{o}"] = {"pct": float(100 * np.nanmean(dd) / np.nanmean(qo)),
                              "ci95": [float(lo), float(hi)]}
        print(f"   {c:>3} vs {o:<3}: {qci[f'{c}_vs_{o}']['pct']:+6.2f}%  "
              f"CI [{lo:+.5f}, {hi:+.5f}]")

    ra, rp, rm = res[("As", "M0s")], res[("Ps", "M0s")], res[("Ams", "M0s")]
    q_ok = qci["As_vs_M0s"]["ci95"][0] > 0
    met = (len(ra["better"]) >= 4 and not ra["worse"] and len(rp["better"]) < 4
           and len(rm["better"]) < 4 and q_ok)
    print("\n--- pre-registered rule: As clears >= 4 of 6 vs M0s and is worse on none;")
    print("    placebo Ps and mirror Ams each clear < 4; As's per-episode QLIKE")
    print("    interval vs M0s excludes 0 favourably")
    print(f"   As  vs M0s: better {ra['better'] or 'nothing'}, worse {ra['worse'] or 'nothing'}")
    print(f"   Ps  vs M0s: better {rp['better'] or 'nothing'}, worse {rp['worse'] or 'nothing'}")
    print(f"   Ams vs M0s: better {rm['better'] or 'nothing'}, worse {rm['worse'] or 'nothing'}")
    print(f"   QLIKE clears: {q_ok}   ->  {'MET' if met else 'NOT MET'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_family": n_family, "alpha": alpha,
        "folds": [{k: v for k, v in r.items() if not k.startswith(("q_", "bar_"))}
                  for r in rows],
        "barriers": {m: {k: float(np.mean(v)) for k, v in d.items()}
                     for m, d in per_fold.items()},
        "ci": ci_out, "qlike": qci,
        "results": {f"{c}_vs_{o}": v for (c, o), v in res.items()},
        "rule_met": bool(met)}, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    # 1. profile recovers a planted cycle, clock-shifted by one hour
    rng = np.random.default_rng(7)
    n = 24 * 400
    ah = np.tile(np.arange(24), n // 24)
    clock = (ah - 1) % 24
    planted = 0.3 * np.sin(2 * np.pi * clock / 24)
    h1 = -5 + planted + rng.normal(0, 0.2, n)
    prof = hour_profile(h1, ah)
    want = 0.3 * np.sin(2 * np.pi * np.arange(24) / 24)
    ok.append(("profile recovers the planted clock-hour cycle",
               np.max(np.abs(prof - (want - want.mean()))) < 0.02))
    # 2. whole days score zero; one hour scores that hour's offset
    flat = season_fwd(prof, np.arange(24), np.full(24, 24))
    ok.append(("a 24-hour window has zero season", np.max(np.abs(flat)) < 1e-12))
    wk = season_fwd(prof, np.arange(24), np.full(24, 168))
    ok.append(("a 168-hour window has zero season", np.max(np.abs(wk)) < 1e-12))
    v = np.exp(2 * prof); v = v / v.mean()
    one = season_fwd(prof, np.arange(24), np.ones(24))
    ok.append(("a 1-hour window scores its own clock hour",
               np.allclose(one, 0.5 * np.log(v))))
    # 3. the served-factor anchors, exactly as adaptive._settled_anchors
    t = 24 * 41_667 * HOUR + 17 * HOUR           # a whole number of days, then 17:00
    fa = factor_anchor_ts(t)
    hrs = sorted({int((x // HOUR) % 24) for x in fa})
    ok.append(("factor anchors for 17:00/H=19 are 04/10/16/22", hrs == [4, 10, 16, 22]))
    ok.append(("every factor episode has closed by the anchor",
               bool(np.all(fa + PROD_H * HOUR <= t))))
    ok.append(("60 days at a 6-hour stride is 240 anchors", len(fa) == 240))
    # 4. factor: median ratio, clipped, 1.0 below the floor
    ts_h = fa; rv_h = np.full(len(fa), 0.9); sg_h = np.ones(len(fa))
    lf = served_log_factor(np.array([t]), ts_h, rv_h, sg_h)
    ok.append(("factor = median RV/sigma", abs(np.exp(lf[0]) - 0.9) < 1e-12))
    lf2 = served_log_factor(np.array([t]), ts_h[:10], rv_h[:10], sg_h[:10])
    ok.append(("fewer than 20 settled -> no correction", lf2[0] == 0.0))
    lf3 = served_log_factor(np.array([t]), ts_h, np.full(len(fa), 3.0), sg_h)
    ok.append(("clipped at 1.40", abs(np.exp(lf3[0]) - 1.40) < 1e-12))
    # 5. placebo: the same values, no clock
    pp = placebo_profile(prof)
    ok.append(("placebo keeps the same 24 values", np.allclose(np.sort(pp), np.sort(prof))))
    ok.append(("placebo is nearly uncorrelated with the clock",
               abs(np.corrcoef(pp, prof)[0, 1]) < 0.02))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
