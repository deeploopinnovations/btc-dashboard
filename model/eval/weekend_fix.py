"""
eval/weekend_fix.py
=====================================================================
P4-weekend-fix: the anchor's weekend column counts the wrong two days.

THE DEFECT (P4-weekend-bug)

noctua/features.py builds cal_weekend_frac from the forward window's weekday as
((ts // 86400) + 4) % 7 and flags `>= 5`. 1970-01-01 was a Thursday, so the
correct offset is +3: the shipped column counts FRIDAY and SATURDAY. At the
product anchor (17:00, H = 19) it reads Thu 0.632 / Fri 1 / Sat 0.368 / Sun 0,
where the weekend is Fri 0.632 / Sat 1 / Sun 0.368 / Thu 0. Serving computes
the column with the same function, so train and serve agree -- which is also why
features.py is NOT edited here: the shipped network was trained on the buggy
column, and changing it in place would be a train/serve skew.

ARMS (every one carries serving's trailing factor from its OWN forecasts):
  M0s  shipped model -- the baseline
  Ws   log_har_cal's OLS plus ONE column, the true Sat+Sun fraction, fitted on
       the same train rows, weights and target; enters through post_shift_fn
       as (1 - w)(anchor_W - lhc). Network untouched: deployable as an artifact
       increment, like the hour anchor.
  Ps   PLACEBO: the same construction with the Tue+Wed fraction -- same two-day
       structure, no calendar reason. If Ps helps, the gain is freedom.
  Fs   SECONDARY: the column corrected everywhere, network and anchor
       retrained per fold. Its served factor is applied by scaling the test
       curves, which is exact for a test-slice-only shift; the identity is
       VERIFIED per fold against a real post_shift run of M0s.

    python -m model.eval.weekend_fix --selftest
    python -m model.eval.weekend_fix
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

HOUR = 3600
WEEKEND = (5, 6)          # Sat, Sun  (0 = Mon)
PLACEBO = (1, 2)          # Tue, Wed
BUGGY_OFFSET, TRUE_OFFSET = 4, 3
EP_METRICS = ("brier", "logs", "pinball", "crps", "qlike")
BAR_METRICS = ("brier", "logs", "pinball", "crps")
PRIMARY = (("Ws", "M0s"), ("Ps", "M0s"))
SECONDARY = (("Fs", "M0s"), ("Fs", "Ws"))
ARMS = ("M0s", "Ws", "Ps", "Fs")
TOL_BAT, TOL_SCALE = 1e-10, 1e-9


def day_frac(anchor_ts, H, days, offset: int = TRUE_OFFSET) -> np.ndarray:
    """Fraction of the forward window's hours whose UTC weekday is in `days`.
    offset 3 is the true calendar (epoch day 0 is a Thursday); offset 4 is
    what features.py uses."""
    anchor_ts = np.asarray(anchor_ts, np.int64)
    H = np.asarray(H, np.int64)
    offs = np.arange(int(H.max()))
    dow = (((anchor_ts[:, None] + offs[None, :] * HOUR) // 86400) + offset) % 7
    valid = offs[None, :] < H[:, None]
    return (np.isin(dow, days) & valid).sum(1) / np.maximum(H, 1)


def anchor_plus(ep, X, fold, extra: np.ndarray) -> dict:
    """log_har_cal and log_har_cal + one column, same train rows/weights/target."""
    from noctua import baselines as B
    from noctua import splits as S
    fin = np.isfinite(X.to_numpy()).all(1)
    m_tr = np.asarray(fold["train"], bool) & fin
    wtr = S.sample_weights(ep, m_tr)
    y = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))
    cols = B.VOL_BASELINES["log_har_cal"]
    lhc = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], wtr)
    X2 = X[cols].copy()
    X2["extra"] = extra
    ols = B.OLS(cols + ["extra"]).fit(X2[m_tr], y[m_tr], wtr)
    out_l, out_a = np.full(len(ep), np.nan), np.full(len(ep), np.nan)
    out_l[fin] = lhc.predict(X[cols][fin])
    out_a[fin] = ols.predict(X2[fin])
    return {"lhc": out_l, "anchor": out_a, "coef": float(ols.beta[-1])}


def scale_curves(cur: dict, lf: np.ndarray) -> dict:
    return {s: q * np.exp(lf)[:, None] for s, q in cur.items()}


def take(pe: dict, rows: np.ndarray) -> dict:
    """Slice a run_fold per_episode test block to `rows` (positions)."""
    return {"curves": {s: q[rows] for s, q in pe["curves"]["noctua_v2"].items()},
            "M": {s: m[rows] for s, m in pe["M_abs"].items()},
            "rv": pe["rv"][rows], "sigma_med": pe["sigma_med"][rows],
            "sigma_mean": pe["sigma_mean"][rows]}


def main(argv=None) -> int:
    from eval.benchmark import run_fold
    from eval.direction import mean_ci
    from eval.hour_anchor import FAC_STRIDE_H, PROD_A, PROD_H, fold_tests, served_log_factor
    from eval.product_score import battery, per_episode
    from eval.scale_adopt import barrier_cols
    from eval.vol_matrix import block_len_for, qlike_vec
    from noctua import infer as I
    from noctua import splits as S
    from noctua.train import load_all

    ap = argparse.ArgumentParser(description="true weekend column in the anchor")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/weekend_fix.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    Hs = ep.H.to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    wf_bug = day_frac(ts, Hs, WEEKEND, BUGGY_OFFSET)
    if np.max(np.abs(wf_bug - X["cal_weekend_frac"].to_numpy())) > 1e-12:
        raise SystemExit("REFUSING: the shipped column is not the Fri+Sat fraction "
                         "this test was registered against")
    wf_true = day_frac(ts, Hs, WEEKEND)
    wf_plac = day_frac(ts, Hs, PLACEBO)
    true_dow = pd.to_datetime(ts, unit="s", utc=True).dayofweek.to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = sorted({(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)})
    fac_mask = at19 & np.isin(ah, fac_hours)
    prod = np.asarray(S.production_mask(ep), bool)
    w = I.BLEND_W
    XF = X.copy()
    XF["cal_weekend_frac"] = wf_true
    print(f"P4-weekend-fix   production 17:00/H=19, factor hours {fac_hours}, "
          f"seeds {a.seeds}", flush=True)

    pe = {k: {m: [] for m in EP_METRICS} for k in ARMS}
    keep = {"idx": [], "year": [], "dow": []}
    keep.update({f"{k}__{v}": [] for k in ARMS for v in ("sigma_med", "sigma_mean")})
    keep.update({f"M0s__Q_{s}": [] for s in ("up", "dn")})
    keep.update({f"M__{s}": [] for s in ("up", "dn")})
    keep["rv"] = []
    dsc = {k: [] for k in ARMS}
    bias = {k: [] for k in ARMS}
    coefs, years = [], []

    def hist_from(peB):
        te = peB["test_idx"]
        f_rows = np.flatnonzero(fac_mask[te])
        h_idx = np.concatenate([peB["cal_idx"], te[f_rows]])
        h_rv = np.concatenate([peB["rv_cal"], peB["rv"][f_rows]])
        h_sig = np.concatenate([peB["sigma_cal"], peB["sigma_med"][f_rows]])
        return h_idx, h_rv, h_sig

    for f in S.walk_forward_folds(ep):
        t0 = time.time()
        aW = anchor_plus(ep, X, f, wf_true)
        aP = anchor_plus(ep, X, f, wf_plac)
        sh = {"M0s": np.zeros(len(ep)),
              "Ws": np.nan_to_num((1 - w) * (aW["anchor"] - aW["lhc"])),
              "Ps": np.nan_to_num((1 - w) * (aP["anchor"] - aP["lhc"]))}
        rB = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=prod | fac_mask)
        rF = run_fold(ep, XF, f, a.hidden, a.seeds, prod_override=prod | fac_mask)
        if rB is None or rF is None:
            continue
        peB, peF = rB["per_episode"], rF["per_episode"]
        if not np.array_equal(peB["test_idx"], peF["test_idx"]):
            raise SystemExit("REFUSING: base and fixed runs scored different episodes")
        rows = np.flatnonzero(prod[peB["test_idx"]])
        ti = peB["test_idx"][rows]
        h_idx, h_rv, h_sig = hist_from(peB)
        got = {}
        for arm in ("M0s", "Ws", "Ps"):
            lf = served_log_factor(ts[ti], ts[h_idx], h_rv, h_sig * np.exp(sh[arm][h_idx]))
            full = sh[arm].copy()
            full[ti] += lf
            r1 = run_fold(ep, X, f, a.hidden, a.seeds,
                          post_shift_fn=lambda mask, _mt, _s=full: _s[mask])
            p1 = r1["per_episode"]
            if not np.array_equal(p1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            cur, M = p1["curves"]["noctua_v2"], p1["M_abs"]
            b, b_run = battery(cur, M), barrier_cols(r1["rows"])
            err = max(abs(b[k] - b_run[k]) for k in b_run)
            if not err < TOL_BAT:
                raise SystemExit(f"REFUSING: offline battery differs by {err:.2e}")
            got[arm] = {"curves": cur, "M": M, "rv": p1["rv"],
                        "sigma_med": p1["sigma_med"], "sigma_mean": p1["sigma_mean"],
                        "lf": lf, "DSC": b["DSC"]}
        # the scaling identity Fs relies on, checked on M0s
        base = take(peB, rows)
        sc = scale_curves(base["curves"], got["M0s"]["lf"])
        rel = max(float(np.max(np.abs(sc[s] / got["M0s"]["curves"][s] - 1))) for s in sc)
        if not rel < TOL_SCALE:
            raise SystemExit(f"REFUSING: test-slice factor is not a pure curve scaling "
                             f"(max rel err {rel:.2e}); Fs cannot be built this way")
        # Fs: corrected column everywhere, factor from its own forecasts
        hF_idx, hF_rv, hF_sig = hist_from(peF)
        lfF = served_log_factor(ts[ti], ts[hF_idx], hF_rv, hF_sig)
        fx = take(peF, rows)
        curF = scale_curves(fx["curves"], lfF)
        got["Fs"] = {"curves": curF, "M": fx["M"], "rv": fx["rv"],
                     "sigma_med": fx["sigma_med"] * np.exp(lfF),
                     "sigma_mean": fx["sigma_mean"] * np.exp(lfF), "lf": lfF,
                     "DSC": battery(curF, fx["M"])["DSC"]}
        for arm in ARMS:
            g = got[arm]
            p = per_episode(g["curves"], g["M"])
            for m in BAR_METRICS:
                pe[arm][m].append(p[m])
            pe[arm]["qlike"].append(qlike_vec(g["rv"], g["sigma_mean"]))
            dsc[arm].append(g["DSC"])
            lr = np.log(g["rv"] / g["sigma_med"])
            d = true_dow[ti]
            bias[arm].append({"year": f["year"],
                              "Mon-Wed": float(np.median(lr[d <= 2])),
                              "Thu": float(np.median(lr[d == 3])),
                              "Fri": float(np.median(lr[d == 4])),
                              "Sat": float(np.median(lr[d == 5])),
                              "Sun": float(np.median(lr[d == 6]))})
            keep[f"{arm}__sigma_med"].append(g["sigma_med"])
            keep[f"{arm}__sigma_mean"].append(g["sigma_mean"])
        for s in ("up", "dn"):
            keep[f"M0s__Q_{s}"].append(got["M0s"]["curves"][s])
            keep[f"M__{s}"].append(got["M0s"]["M"][s])
        keep["idx"].append(ti); keep["year"].append(np.full(len(ti), f["year"]))
        keep["dow"].append(true_dow[ti]); keep["rv"].append(got["M0s"]["rv"])
        coefs.append({"year": f["year"], "W": aW["coef"], "P": aP["coef"],
                      "logfac_mean": {k: float(got[k]["lf"].mean()) for k in ARMS}})
        years.append(f["year"])
        print(f"  fold {f['year']}: coef W {aW['coef']:+.3f} P {aP['coef']:+.3f}  "
              f"DSC " + " ".join(f"{k} {got[k]['DSC']:.6f}" for k in ARMS)
              + f"  scaling err {rel:.1e}  ({time.time() - t0:.0f}s)", flush=True)

    if not years:
        print("no complete folds"); return 1
    cat = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in pe.items()}
    n = len(cat["M0s"]["brier"])
    L = block_len_for(PROD_H, n)
    out = {"years": years, "n_nights": n, "block_len": L, "coefs": coefs,
           "bias_by_day": bias, "ci": {}, "dsc": {}, "subsets": {}}

    def contrast(c, o, alpha):
        better, worse = [], []
        for m in EP_METRICS:
            d = cat[o][m] - cat[c][m]
            lo, hi = mean_ci(d, alpha=alpha, block_len=L)["ci95"]
            pct = 100 * d.mean() / cat[o][m].mean()
            out["ci"][f"{m}_{c}_vs_{o}"] = {"pct": float(pct), "delta": float(d.mean()),
                                            "ci": [float(lo), float(hi)], "alpha": alpha}
            if lo > 0: better.append(m)
            if hi < 0: worse.append(m)
            print(f"  {m:>8} {c + '-' + o:>8} {pct:+7.3f}%  [{lo:+.6f}, {hi:+.6f}]")
        dd = np.array(dsc[c]) - np.array(dsc[o])
        ft = fold_tests(dd, alpha)
        out["dsc"][f"{c}_vs_{o}"] = {"delta": float(dd.mean()), "t": ft["t_ci"],
                                     "p_flip": ft["p_flip"],
                                     "folds_better": int((dd > 0).sum())}
        print(f"       DSC {c + '-' + o:>8} {dd.mean():+.6f}  {int((dd > 0).sum())}/{len(dd)} "
              f"folds  t [{ft['t_ci'][0]:+.6f}, {ft['t_ci'][1]:+.6f}]")
        return {"better": better, "worse": worse}

    a1 = 0.05 / (len(PRIMARY) * len(EP_METRICS))
    a2 = 0.05 / ((len(PRIMARY) + len(SECONDARY)) * len(EP_METRICS))
    print(f"\n{n} production nights, block {L}; PRIMARY family 10 -> {100 * (1 - a1):.2f}%")
    res = {f"{c}_vs_{o}": contrast(c, o, a1) for c, o in PRIMARY}
    print(f"\nSECONDARY (reported, no rule), family 20 -> {100 * (1 - a2):.2f}%")
    res.update({f"{c}_vs_{o}": contrast(c, o, a2) for c, o in SECONDARY})

    dow = np.concatenate(keep["dow"])
    print("\nex-ante weekday classes, Ws - M0s (reported, no rule):")
    for name, msk in (("Fri/Sat/Sun", dow >= 4), ("Thu", dow == 3), ("Mon-Wed", dow <= 2)):
        cell = {"n": int(msk.sum())}
        for m in ("brier", "logs", "qlike"):
            d = cat["M0s"][m][msk] - cat["Ws"][m][msk]
            lo, hi = mean_ci(d, alpha=0.05, block_len=1)["ci95"]
            cell[m] = {"pct": float(100 * d.mean() / cat["M0s"][m][msk].mean()),
                       "ci95": [float(lo), float(hi)]}
        out["subsets"][name] = cell
        print(f"   {name:>11} n={cell['n']:4d}  " + "  ".join(
            f"{m} {cell[m]['pct']:+.2f}% [{cell[m]['ci95'][0]:+.5f}, {cell[m]['ci95'][1]:+.5f}]"
            for m in ("brier", "logs", "qlike")))
    print("\nmedian log(RV/sigma_med) by anchor day, mean over folds:")
    for arm in ARMS:
        print(f"   {arm:>3} " + "  ".join(
            f"{k} {np.mean([b[k] for b in bias[arm]]):+.3f}"
            for k in ("Mon-Wed", "Thu", "Fri", "Sat", "Sun")))

    rw, rp = res["Ws_vs_M0s"], res["Ps_vs_M0s"]
    n_bar = [m for m in rw["better"] if m in BAR_METRICS]
    met = len(n_bar) >= 2 and not rw["worse"] and not rp["better"]
    print("\n--- registered rule: Ws beats M0s on >= 2 of {brier, logs, pinball, crps},")
    print("    worse on none of the five, and the placebo Ps better on none")
    print(f"   Ws vs M0s: better {rw['better'] or 'nothing'}, worse {rw['worse'] or 'nothing'}")
    print(f"   Ps vs M0s: better {rp['better'] or 'nothing'}, worse {rp['worse'] or 'nothing'}")
    print(f"   -> {'MET (ADVANCE)' if met else 'NOT MET (REJECT)'}")
    out.update(results=res, rule_met=bool(met))
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    np.savez_compressed(a.out.with_suffix(".npz"),
                        **{k: np.concatenate(v) for k, v in keep.items()},
                        **{f"{k}__{m}": cat[k][m] for k in ARMS for m in EP_METRICS})
    print(f"wrote {a.out} and {a.out.with_suffix('.npz')}")
    return 0


def selftest() -> int:
    ok = []

    def at(s):
        return np.array([int(pd.Timestamp(s, tz="UTC").timestamp())], np.int64)

    H19 = np.array([19])
    want = {"2026-09-24 17:00": (0.0, 12 / 19),      # Thu: true 0, shipped 0.632
            "2026-09-25 17:00": (12 / 19, 1.0),      # Fri
            "2026-09-26 17:00": (1.0, 7 / 19),       # Sat
            "2026-09-27 17:00": (7 / 19, 0.0),       # Sun
            "2026-09-28 17:00": (0.0, 0.0)}          # Mon
    good = all(abs(day_frac(at(s), H19, WEEKEND)[0] - t) < 1e-12 and
               abs(day_frac(at(s), H19, WEEKEND, BUGGY_OFFSET)[0] - b) < 1e-12
               for s, (t, b) in want.items())
    ok.append(("true and shipped weekend fractions at the five product nights", good))
    rng = np.random.default_rng(0)
    t = (1_500_000_000 + rng.integers(0, 3 * 10**8, 500)) // HOUR * HOUR
    d = pd.to_datetime(t, unit="s", utc=True).dayofweek.to_numpy()
    one = day_frac(t, np.ones(500, np.int64), WEEKEND)
    ok.append(("one-hour windows agree with pandas dayofweek on 500 dates",
               np.array_equal(one == 1.0, d >= 5)))
    ok.append(("placebo is Tue+Wed, disjoint from both weekend definitions",
               np.array_equal(day_frac(t, np.ones(500, np.int64), PLACEBO) == 1.0,
                              np.isin(d, (1, 2)))))
    wk = day_frac(t, np.full(500, 168), WEEKEND)
    ok.append(("a 168-hour window is 2/7 weekend", np.allclose(wk, 2 / 7)))
    q = {"up": np.ones((3, 4)), "dn": 2 * np.ones((3, 4))}
    s = scale_curves(q, np.log(np.array([1.0, 2.0, 0.5])))
    ok.append(("curve scaling multiplies each episode's quantiles",
               np.allclose(s["up"][1], 2.0) and np.allclose(s["dn"][2], 1.0)))
    for name, g in ok:
        print(f"  [{'ok' if g else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
