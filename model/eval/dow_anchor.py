"""
eval/dow_anchor.py
=====================================================================
P4-dow-anchor: does the FULL day-of-week calendar in the anchor beat the
weekend-only increment (P4-weekend-fix-result)?

P4-weekend-calendar-perm found that the weekend increment works by separating
Friday (busy) from Saturday (quiet) -- which the shipped Fri+Sat column lumps
-- and that Monday carries a further effect. The principled generalisation is
the whole week: the window's fraction on Mon, Tue, Wed, Thu, Fri as five extra
anchor columns (with the shipped Fri+Sat column and the intercept they span all
seven days).

ARMS (each with serving's trailing factor from its OWN forecasts; all through
run_fold, like eval/weekend_fix.py):
  M0s  shipped
  Ws   + true Sat+Sun fraction (the frozen candidate, re-run here)
  Ds   + Mon..Fri fractions (5 columns)
  P6s  PLACEBO: 5 columns from a fake 6-day cycle (epoch day mod 6, phases
       0-4) -- the same freedom, no calendar

    python -m model.eval.dow_anchor --selftest
    python -m model.eval.dow_anchor
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

from eval.weekend_fix import WEEKEND, day_frac                         # noqa: E402

HOUR = 3600
ARMS = ("M0s", "Ws", "Ds", "P6s")
EP_METRICS = ("brier", "logs", "pinball", "crps", "qlike")
BAR = ("brier", "logs", "pinball", "crps")
CONTRASTS = (("Ds", "Ws"), ("Ds", "M0s"), ("P6s", "M0s"))
TOL_BAT = 1e-10


def cycle_frac(anchor_ts, H, period: int, phases) -> np.ndarray:
    """Fraction of window hours whose (epoch day mod period) is in `phases`."""
    anchor_ts = np.asarray(anchor_ts, np.int64)
    H = np.asarray(H, np.int64)
    offs = np.arange(int(H.max()))
    ph = ((anchor_ts[:, None] + offs[None, :] * HOUR) // 86400) % period
    valid = offs[None, :] < H[:, None]
    return (np.isin(ph, phases) & valid).sum(1) / np.maximum(H, 1)


def extra_cols(ts, Hs) -> dict:
    return {"Ws": {"wk": day_frac(ts, Hs, WEEKEND)},
            "Ds": {f"d{d}": day_frac(ts, Hs, (d,)) for d in range(5)},
            "P6s": {f"p{k}": cycle_frac(ts, Hs, 6, (k,)) for k in range(5)}}


def anchor_multi(ep, X, fold, extra: dict) -> dict:
    from noctua import baselines as B
    from noctua import splits as S
    fin = np.isfinite(X.to_numpy()).all(1)
    m_tr = np.asarray(fold["train"], bool) & fin
    wtr = S.sample_weights(ep, m_tr)
    y = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))
    cols = B.VOL_BASELINES["log_har_cal"]
    lhc = B.OLS(cols).fit(X[cols][m_tr], y[m_tr], wtr)
    X2 = X[cols].copy()
    for k, v in extra.items():
        X2[k] = v
    ols = B.OLS(cols + list(extra)).fit(X2[m_tr], y[m_tr], wtr)
    out_l, out_a = np.full(len(ep), np.nan), np.full(len(ep), np.nan)
    out_l[fin] = lhc.predict(X[cols][fin])
    out_a[fin] = ols.predict(X2[fin])
    return {"lhc": out_l, "anchor": out_a,
            "coef": [float(c) for c in ols.beta[-len(extra):]]}


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

    ap = argparse.ArgumentParser(description="full day-of-week anchor")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/dow_anchor.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    ts = ep["anchor_ts"].to_numpy(np.int64)
    Hs = ep.H.to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = sorted({(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)})
    fac_mask = at19 & np.isin(ah, fac_hours)
    prod = np.asarray(S.production_mask(ep), bool)
    w = I.BLEND_W
    extras = extra_cols(ts, Hs)
    true_dow = pd.to_datetime(ts, unit="s", utc=True).dayofweek.to_numpy()
    print(f"P4-dow-anchor   production 17:00/H=19, seeds {a.seeds}", flush=True)

    pe = {k: {m: [] for m in EP_METRICS} for k in ARMS}
    dsc = {k: [] for k in ARMS}
    keep = {"idx": [], "year": [], "dow": []}
    coefs, years = [], []
    for f in S.walk_forward_folds(ep):
        t0 = time.time()
        anc = {k: anchor_multi(ep, X, f, extras[k]) for k in ("Ws", "Ds", "P6s")}
        sh = {"M0s": np.zeros(len(ep))}
        sh.update({k: np.nan_to_num((1 - w) * (v["anchor"] - v["lhc"])) for k, v in anc.items()})
        rB = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=prod | fac_mask)
        if rB is None:
            continue
        peB = rB["per_episode"]
        te = peB["test_idx"]
        rows = np.flatnonzero(prod[te])
        ti = te[rows]
        f_rows = np.flatnonzero(fac_mask[te])
        h_idx = np.concatenate([peB["cal_idx"], te[f_rows]])
        h_rv = np.concatenate([peB["rv_cal"], peB["rv"][f_rows]])
        h_sig = np.concatenate([peB["sigma_cal"], peB["sigma_med"][f_rows]])
        for arm in ARMS:
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
            p = per_episode(cur, M)
            for m in BAR:
                pe[arm][m].append(p[m])
            pe[arm]["qlike"].append(qlike_vec(p1["rv"], p1["sigma_mean"]))
            dsc[arm].append(b["DSC"])
        keep["idx"].append(ti); keep["year"].append(np.full(len(ti), f["year"]))
        keep["dow"].append(true_dow[ti])
        coefs.append({"year": f["year"], **{k: v["coef"] for k, v in anc.items()}})
        years.append(f["year"])
        print(f"  fold {f['year']}: Ds coefs {np.round(anc['Ds']['coef'], 3).tolist()}  "
              f"DSC " + " ".join(f"{k} {dsc[k][-1]:.6f}" for k in ARMS)
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    if not years:
        print("no complete folds"); return 1
    cat = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in pe.items()}
    n = len(cat["M0s"]["brier"])
    L = block_len_for(PROD_H, n)
    alpha = 0.05 / (len(CONTRASTS) * len(EP_METRICS))
    out = {"years": years, "n_nights": n, "block_len": L, "alpha": alpha,
           "coefs": coefs, "ci": {}, "dsc": {}}
    res = {}
    print(f"\n{n} production nights, block {L}, family 15 -> {100 * (1 - alpha):.2f}%")
    for c, o in CONTRASTS:
        better, worse = [], []
        for m in EP_METRICS:
            d = cat[o][m] - cat[c][m]
            lo, hi = mean_ci(d, alpha=alpha, block_len=L)["ci95"]
            pct = 100 * d.mean() / cat[o][m].mean()
            out["ci"][f"{m}_{c}_vs_{o}"] = {"pct": float(pct), "ci": [float(lo), float(hi)]}
            if lo > 0: better.append(m)
            if hi < 0: worse.append(m)
            print(f"  {m:>8} {c + '-' + o:>8} {pct:+7.3f}%  [{lo:+.6f}, {hi:+.6f}]")
        dd = np.array(dsc[c]) - np.array(dsc[o])
        ft = fold_tests(dd, alpha)
        out["dsc"][f"{c}_vs_{o}"] = {"delta": float(dd.mean()), "t": ft["t_ci"],
                                     "folds_better": int((dd > 0).sum())}
        print(f"       DSC {c + '-' + o:>8} {dd.mean():+.6f}  {int((dd > 0).sum())}/{len(dd)} folds")
        res[f"{c}_vs_{o}"] = {"better": better, "worse": worse}
    dw = np.concatenate(keep["dow"])
    print("\nper weekday, Ds - Ws Brier (reported, no rule):")
    for d, name in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        m = dw == d
        dd = cat["Ws"]["brier"][m] - cat["Ds"]["brier"][m]
        print(f"   {name}: {100 * dd.mean() / cat['Ws']['brier'][m].mean():+.3f}%  (n={int(m.sum())})")
    rd, rp = res["Ds_vs_Ws"], res["P6s_vs_M0s"]
    met = (len([m for m in rd["better"] if m in BAR]) >= 2 and not rd["worse"]
           and not rp["better"])
    print("\n--- registered rule: Ds beats Ws on >= 2 of the 4 barrier metrics, worse on")
    print("    none of the five, and the 6-day placebo better than M0s on none")
    print(f"   Ds vs Ws:   better {rd['better'] or 'nothing'}, worse {rd['worse'] or 'nothing'}")
    print(f"   Ds vs M0s:  better {res['Ds_vs_M0s']['better'] or 'nothing'}, "
          f"worse {res['Ds_vs_M0s']['worse'] or 'nothing'}")
    print(f"   P6s vs M0s: better {rp['better'] or 'nothing'}, worse {rp['worse'] or 'nothing'}")
    print(f"   -> {'MET: Ds replaces Ws as the candidate' if met else 'NOT MET: Ws stays; the calendar line closes on walk-forward data'}")
    out.update(results=res, rule_met=bool(met))
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(a.out.with_suffix(".npz"),
                        **{k: np.concatenate(v) for k, v in keep.items()},
                        **{f"{k}__{m}": cat[k][m] for k in ARMS for m in EP_METRICS})
    print(f"wrote {a.out} and {a.out.with_suffix('.npz')}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(0)
    t = (1_500_000_000 + rng.integers(0, 3 * 10**8, 300)) // HOUR * HOUR
    H = rng.choice([6, 19, 24], len(t))
    ex = extra_cols(t, H)
    days = np.stack([day_frac(t, H, (d,)) for d in range(7)], 1)
    ok.append(("seven day fractions sum to 1", np.allclose(days.sum(1), 1.0)))
    bug = day_frac(t, H, WEEKEND, 4)
    A = np.column_stack([np.ones(len(t)), bug] + list(ex["Ds"].values()))
    ok.append(("intercept + shipped Fri+Sat + Mon..Fri span all seven days",
               np.linalg.matrix_rank(np.column_stack([A, days])) == np.linalg.matrix_rank(A) == 7))
    ph = np.stack([cycle_frac(t, H, 6, (k,)) for k in range(6)], 1)
    ok.append(("6-day placebo phases sum to 1", np.allclose(ph.sum(1), 1.0)))
    ok.append(("placebo has the same number of columns", len(ex["P6s"]) == len(ex["Ds"]) == 5))
    for name, g in ok:
        print(f"  [{'ok' if g else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
