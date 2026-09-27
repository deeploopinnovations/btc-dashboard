"""
eval/online_stack.py
=====================================================================
P4-online-stack: let the ensemble weights MOVE during the year.

WHAT EXISTS, AND WHAT IT CANNOT DO

P4-zoo-stack-v2-result: a convex log-space ensemble of the zoo, weights fitted
by QLIKE on each fold's six-month calib slice and shrunk halfway toward NOCTUA
(`stack_half`), beats NOCTUA as served by +6.56 / +2.05 / +1.69% at H = 1 / 6 /
24. Its weights are fitted ONCE, on 1 Jul - 24 Dec of the prior year, and then
held for twelve months. Which teacher is right changes faster than that: the
2022 fold, which the unshrunk stack lost by -36.6%, is a year whose calib half
looked nothing like its test half.

WHAT IS BUILT

The same estimator, refreshed every week on a TRAILING window of the same
length as calib (182 days), so the only thing that changes against stack_half
is RECENCY -- not the estimator, not the shrinkage, not the sample size:

    at refresh time T (weekly from 1 Jan of the test year):
      window  = every episode whose label has MATURED in (T - 182d, T],
                i.e. anchor_ts + H hours <= T, drawn from the fold's calib
                slice and from test episodes already realised
      w_T     = 0.5 * fit_convex(window) + 0.5 * e_NOCTUA      (R82, as v2)
      b_T     = the QLIKE-optimal level for w_T on the window
      forecast every test episode anchored in [T, T + 7d) with (w_T, b_T)

Every forecast in the window comes from the SAME fold's teachers -- the model
set that is live during that test year -- and each is out-of-sample for them
(calib is held out of training; test rows are scored before they enter any
window). A label is used only after it exists.

THE TRAP THIS HAS TO CLEAR. A rolling fit moves two things: the weights and the
LEVEL b. P3-rolling-level and P3-transfer-anatomy measured rolling LEVEL
corrections as drift-chasing that does not transfer. So the design carries a
control, `ref_roll`: NOCTUA alone with ONLY its level re-fitted on the same
rolling window. If roll_half's gain over stack_half is matched by ref_roll's
gain over ref, the online stack is a rolling level in disguise.

    python -m model.eval.online_stack --selftest
    python -m model.eval.online_stack
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                       # noqa: E402
from eval.mz_recalibration import YEARS, qlike_vec                       # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                    # noqa: E402
from eval.teacher_zoo import FoldScopedFit                               # noqa: E402
from eval.zoo_stack import (REF, SHRINK, ZOO, _clean, apply,             # noqa: E402
                            fit_convex, level_for)

HOUR = 3600
STEP_H = 168            # weekly refresh
WINDOW_H = 182 * 24     # the calib slice's own length
MIN_WINDOW = 500        # below this the static fit is used, and counted
ARMS = ("ref", "ref_roll", "stack_half", "roll_half", "roll_full")
CONTRASTS = (("roll_half", "stack_half"), ("roll_half", "ref"),
             ("ref_roll", "ref"))


def refresh_times(t0: int, t1: int, step_h: int = STEP_H) -> np.ndarray:
    """Refresh instants from the first test anchor, every step, covering t1."""
    return np.arange(t0, t1 + 1, step_h * HOUR, dtype=np.int64)


def window_mask(ts: np.ndarray, H: int, T: int, window_h: int = WINDOW_H):
    """Rows whose label matured in (T - window, T]. Matured = anchor + H <= T."""
    mat = ts.astype(np.int64) + int(H) * HOUR
    return (mat <= T) & (mat > T - int(window_h) * HOUR)


def online_fold(Lc, lc2, tsc, Lt, lt2, tst, H, i_ref,
                step_h=STEP_H, window_h=WINDOW_H, shrink=SHRINK):
    """Forecasts for one fold's test slice under every arm.

    Lc/lc2/tsc: calib log-sigma (n, K), log RV^2, anchor ts. Lt/lt2/tst: test.
    Returns {arm: log-sigma forecasts (n_test,)} and the refresh log.
    """
    K = Lc.shape[1]
    e_ref = np.eye(K)[i_ref]
    w_st = fit_convex(Lc, lc2)
    w_hf = shrink * w_st + (1.0 - shrink) * e_ref
    out = {"ref": Lt @ e_ref + level_for(e_ref, Lc, lc2),
           "stack_half": Lt @ w_hf + level_for(w_hf, Lc, lc2),
           "ref_roll": np.full(len(tst), np.nan),
           "roll_half": np.full(len(tst), np.nan),
           "roll_full": np.full(len(tst), np.nan)}
    L_all = np.vstack([Lc, Lt]); l2_all = np.concatenate([lc2, lt2])
    ts_all = np.concatenate([tsc, tst])
    log = {"n_refresh": 0, "n_fallback": 0, "w_ref_path": []}
    for T in refresh_times(int(tst.min()), int(tst.max()), step_h):
        score = (tst >= T) & (tst < T + step_h * HOUR)
        if not score.any():
            continue
        win = window_mask(ts_all, H, T, window_h)
        # the guard the design rests on: nothing scored this week is in the
        # window, and nothing in the window is unrealised at T
        assert not np.any(win[len(tsc):] & score), "scored row inside window"
        assert np.all(ts_all[win] + H * HOUR <= T), "unrealised label in window"
        log["n_refresh"] += 1
        if win.sum() < MIN_WINDOW:
            log["n_fallback"] += 1
            w_on = w_st
            Lw, lw2 = Lc, lc2
        else:
            Lw, lw2 = L_all[win], l2_all[win]
            w_on = fit_convex(Lw, lw2)
        w_oh = shrink * w_on + (1.0 - shrink) * e_ref
        for arm, w in (("roll_half", w_oh), ("roll_full", w_on),
                       ("ref_roll", e_ref)):
            out[arm][score] = Lt[score] @ w + level_for(w, Lw, lw2)
        log["w_ref_path"].append(float(w_oh[i_ref]))
    return out, log


def run_horizon(z, H: int) -> dict:
    pooled = {a: [] for a in ARMS}
    per_fold, logs = [], []
    i_ref = ZOO.index(REF)
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kc, kt = f"{y}/{H}/calib", f"{y}/{H}/test"
            if any(f"{kt}/sigma/{t}" not in z or f"{kc}/sigma/{t}" not in z
                   for t in ZOO):
                continue
            Sc = np.column_stack([np.asarray(sc.calib(z, H, t), np.float64)
                                  for t in ZOO])
            St = np.column_stack([np.asarray(sc.test(z, H, t), np.float64)
                                  for t in ZOO])
        rc = np.asarray(z[f"{kc}/rv"], np.float64)
        rt = np.asarray(z[f"{kt}/rv"], np.float64)
        tsc = np.asarray(z[f"{kc}/anchor_ts"], np.int64)
        tst = np.asarray(z[f"{kt}/anchor_ts"], np.int64)
        okc, okt = _clean(rc, Sc), _clean(rt, St)
        if okc.sum() < 500 or okt.sum() < 500:
            continue
        o = np.argsort(tst[okt], kind="stable")
        Lt = np.log(St[okt])[o]; rt_ = rt[okt][o]; tst_ = tst[okt][o]
        out, log = online_fold(np.log(Sc[okc]), np.log(rc[okc] ** 2), tsc[okc],
                               Lt, np.log(rt_ ** 2), tst_, H, i_ref)
        row = {"year": y, "n": int(len(rt_))}
        for arm in ARMS:
            q = qlike_vec(rt_, np.exp(out[arm]))
            pooled[arm].append(q)
            row[arm] = float(np.nanmean(q))
        per_fold.append(row)
        logs.append({"year": y, "n_refresh": log["n_refresh"],
                     "n_fallback": log["n_fallback"],
                     "w_ref_min": float(np.min(log["w_ref_path"])),
                     "w_ref_max": float(np.max(log["w_ref_path"]))})
    if not per_fold:
        return {}
    return {"q": {a: np.concatenate(v) for a, v in pooled.items()},
            "per_fold": per_fold, "logs": logs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="online (weekly) zoo stack")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/online_stack.json"))
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    n_family = len(CONTRASTS) * len(HORIZONS)
    alpha = 0.05 / n_family
    print("P4-online-stack   weekly refresh, trailing 182-day matured window")
    print(f"family {n_family} -> {100*(1-alpha):.3f}% intervals, block "
          f"max(n^1/3, 2H), {a.boot} reps\n")
    out = {"alpha": alpha, "step_h": STEP_H, "window_h": WINDOW_H,
           "horizons": {}}
    clears = {c: [] for c in CONTRASTS}
    worse = {c: [] for c in CONTRASTS}
    gains = {}
    for H in HORIZONS:
        r = run_horizon(z, H)
        if not r:
            continue
        q = r["q"]
        base = float(np.nanmean(q["ref"]))
        print(f"=== H = {H}   ({len(q['ref']):,} test episodes)")
        for arm in ARMS:
            m = float(np.nanmean(q[arm]))
            print(f"   {arm:>10}  QLIKE {m:.5f}   vs ref {100*(base-m)/base:+6.2f}%")
        L = max(int(round(len(q["ref"]) ** (1 / 3))), 2 * H)
        ci = {}
        for c, o in CONTRASTS:
            d = q[o] - q[c]
            g = np.isfinite(d)
            lo, hi = mean_ci(d[g], n_rep=a.boot, alpha=alpha,
                             block_len=L)["ci95"]
            pct = 100 * float(np.nanmean(d)) / float(np.nanmean(q[o]))
            gains[(c, o, H)] = pct
            if lo > 0:
                clears[(c, o)].append(H)
            if hi < 0:
                worse[(c, o)].append(H)
            tag = (f"{c} BETTER" if lo > 0 else f"{c} WORSE" if hi < 0
                   else "not separated")
            ci[f"{c}_vs_{o}"] = {"pct": pct, "ci95": [float(lo), float(hi)]}
            print(f"   {c:>10} vs {o:<10} {pct:+6.2f}%  "
                  f"[{lo:+.5f}, {hi:+.5f}]  {tag}")
        print("   per fold (roll_half vs stack_half, %): " + "  ".join(
            f"{f['year']}:{100*(f['stack_half']-f['roll_half'])/f['stack_half']:+.1f}"
            for f in r["per_fold"]))
        print()
        out["horizons"][str(H)] = {
            "qlike": {arm: float(np.nanmean(q[arm])) for arm in ARMS},
            "ci": ci, "per_fold": r["per_fold"], "refresh": r["logs"]}

    dec = ("roll_half", "stack_half")
    not_level = [H for H in clears[dec]
                 if gains[dec + (H,)] > gains[("ref_roll", "ref", H)]]
    met = len(clears[dec]) >= 2 and not worse[dec] and \
        len(not_level) == len(clears[dec])
    print("--- pre-registered rule: roll_half beats stack_half at >= 2 of 4 "
          "horizons, is worse at none,")
    print("    and at every clearing horizon its gain exceeds ref_roll's gain "
          "over ref (not a rolling level)")
    print(f"   clears at {clears[dec] or 'none'}, worse at {worse[dec] or 'none'}, "
          f"gain beyond rolling level at {not_level or 'none'}  ->  "
          f"{'MET' if met else 'NOT MET'}")
    out["rule_met"] = bool(met)
    out["clears"] = {f"{c}_vs_{o}": v for (c, o), v in clears.items()}
    out["worse"] = {f"{c}_vs_{o}": v for (c, o), v in worse.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(3)
    # planted REGIME SWITCH: teacher 0 is right for the first half of the
    # test year, teacher 1 for the second half; calib looks like the first
    # half. A static fit must stay on teacher 0; the weekly refit must move.
    H, n_c, n_t = 1, 4000, 8000
    tsc = np.arange(n_c, dtype=np.int64) * HOUR
    tst = (n_c + 200 + np.arange(n_t, dtype=np.int64)) * HOUR
    truth_c = rng.normal(-4, 0.4, n_c)
    truth_t = rng.normal(-4, 0.4, n_t)
    half = np.arange(n_t) >= n_t // 2

    def teachers(truth, good0):
        good = truth + rng.normal(0, 0.05, truth.size)
        bad = truth + rng.normal(0, 0.60, truth.size)
        a0 = np.where(good0, good, bad)
        a1 = np.where(good0, bad, good)
        return np.column_stack([a0, a1])

    Lc = teachers(truth_c, np.ones(n_c, bool))
    Lt = teachers(truth_t, ~half)
    lc2 = 2 * truth_c + rng.normal(0, 0.3, n_c)
    lt2 = 2 * truth_t + rng.normal(0, 0.3, n_t)
    out, log = online_fold(Lc, lc2, tsc, Lt, lt2, tst, H, i_ref=0,
                           shrink=1.0)
    rt = np.exp(lt2 / 2)
    q_st = np.nanmean(qlike_vec(rt[half], np.exp(out["stack_half"][half])))
    q_on = np.nanmean(qlike_vec(rt[half], np.exp(out["roll_half"][half])))
    ok.append(("static fit stays on the calib-era teacher; weekly refit "
               "follows the switch", q_on < 0.7 * q_st))
    ok.append(("first half: both arms near-equal (nothing to adapt to)",
               abs(np.nanmean(qlike_vec(rt[~half], np.exp(out["roll_half"][~half])))
                   - np.nanmean(qlike_vec(rt[~half], np.exp(out["stack_half"][~half]))))
               < 0.02))
    ok.append(("every test row forecast by every arm",
               all(np.isfinite(out[a]).all() for a in ARMS)))
    ok.append(("weekly cadence", log["n_refresh"] == int(np.ceil(n_t / STEP_H))))
    # a label is in the window only once it exists
    w = window_mask(np.array([0, 10, 20], np.int64) * HOUR, H=6, T=16 * HOUR)
    ok.append(("label matures at anchor + H, not before",
               list(w) == [True, True, False]))
    w2 = window_mask(np.array([0], np.int64), H=1, T=(WINDOW_H + 5) * HOUR)
    ok.append(("labels older than the window drop out", not w2[0]))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
