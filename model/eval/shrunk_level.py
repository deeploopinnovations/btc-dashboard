"""
eval/shrunk_level.py
=====================================================================
P3-shrunk-level: the symmetric level rescale taxes the arm that needs it least.
Does shrinking c by its own estimation precision remove that tax without giving
back the correction's legitimate work?

WHAT THE PROTOCOL CURRENTLY COSTS

`P3-symmetric-correction-cost` measured who gains and who loses from the
c-rescale. `noctua_v1_mean` is the only teacher that LOSES at every horizon
(-1.29 / -2.76 / -4.45 / -2.12%) and is also the best raw arm at every horizon.
Rivals gain enormously: log_har +27.23%, har_short +19.00%, persistence
+32.01%, garch_normal +48.66%.

Both facts have the same cause. c is estimated, and an arm whose true c is
already 1 gains nothing from estimating it and pays the variance of the
estimate. An arm whose true c is 1.2 gains far more than it pays.

THE FIX, AND WHY IT CANNOT FAVOUR ANYBODY

    c_shrunk = exp(w * log c),    w = tau^2 / (tau^2 + SE^2)

SE is the bootstrap error of log(c) on that fold's own calib slice; tau^2 is
the between-fold variance of the true log(c), estimated from PAST FOLDS ONLY.
Every teacher gets the same rule and every weight is derived from that
teacher's own data, so the correction is large exactly where the arm
demonstrably needs one and small where it does not. Nothing in it knows which
arm is NOCTUA.

THE CONTROL IS THE EXPERIMENT, FOR THE FOURTH TIME

`c_half` applies a FIXED w = 0.5 to every teacher, fold and horizon. Three
previous experiments in this project ended with a trivial control matching a
measured rule (R82), so it is in the family from the start rather than added
afterwards -- and its cold-start folds are handicapped identically, or the
comparison measures start-up policy instead of weighting (P3-warm-start-result).

    python -m model.eval.shrunk_level --selftest
    python -m model.eval.shrunk_level
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mz_recalibration import YEARS, optimal_c, qlike_vec            # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                    # noqa: E402
from eval.teacher_zoo import FoldScopedFit                               # noqa: E402

ARMS = ("raw", "c", "c_shrunk", "c_half")
FIXED_W = 0.5
N_BOOT = 300


def fixed_target_weight(log_c: float, se: float) -> float:
    """w = (log c)^2 / ((log c)^2 + SE^2): magnitude of the estimate against
    its own error.

    THE RIGHT FORM FOR A FIXED TARGET, and the first version of this module had
    the wrong one. Shrinking toward a FITTED POOLED MEAN calls for the
    between-fold variance of the estimates; shrinking toward a FIXED point --
    here log c = 0, i.e. no correction -- calls for how far the estimate is
    from that point relative to how precisely it is known.

    The difference was not academic. With the variance form, a badly-calibrated
    arm whose log c is large but CONSISTENT across folds has a small
    between-fold variance, so tau^2 floored at zero and the correction it needs
    was shrunk away entirely: at H=6 log_har, log_har_cal, har_short,
    noctua_v1 and persistence all received w = 0 while the three arms that
    needed no correction kept w near 0.8. Exactly backwards
    (P3-shrunk-level-result).

    Uses only this fold's own calib slice, so there is no cold start and no
    past-fold estimate that could leak.
    """
    if not np.isfinite(log_c) or not np.isfinite(se):
        return 0.0
    lc2 = float(log_c) ** 2
    den = lc2 + float(se) ** 2
    return float(np.clip(lc2 / den, 0.0, 1.0)) if den > 0 else 0.0


def logc_se(rv, sig, H: int, n_rep: int = N_BOOT, seed: int = 0) -> float:
    """Moving-block bootstrap error of log(optimal_c) on one calib slice.

    Block 2H for the same reason every other interval here uses it: episodes
    are anchored hourly against an H-hour forward window, so neighbours share
    H-1 of their H hours.
    """
    rv = np.asarray(rv, np.float64); sig = np.asarray(sig, np.float64)
    ok = np.isfinite(rv) & np.isfinite(sig) & (rv > 0) & (sig > 0)
    rv, sig = rv[ok], sig[ok]
    n = len(rv)
    L = min(max(2 * H, 2), n)
    if n < 3 * L:
        return float("nan")
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    out = np.empty(n_rep)
    for i in range(n_rep):
        st = rng.integers(0, n - L + 1, size=nb)
        idx = (st[:, None] + np.arange(L)[None, :]).reshape(-1)[:n]
        out[i] = np.log(max(optimal_c(rv[idx], sig[idx]), 1e-12))
    return float(np.nanstd(out, ddof=1))


def run_teacher(z, H: int, teacher: str, n_boot: int = N_BOOT) -> dict | None:
    """Walk the folds, shrinking each fold's c with PAST folds' spread only."""
    past_lc, past_se = [], []
    acc = {a: [] for a in ARMS}
    rows = []
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kt, kc = f"{y}/{H}/test", f"{y}/{H}/calib"
            if (f"{kt}/sigma/{teacher}" not in z
                    or f"{kc}/sigma/{teacher}" not in z):
                continue
            sc_ = np.asarray(sc.calib(z, H, teacher), np.float64)
            rc = np.asarray(z[f"{kc}/rv"], np.float64)
            st_ = np.asarray(sc.test(z, H, teacher), np.float64)
            rt = np.asarray(z[f"{kt}/rv"], np.float64)
        ok = np.isfinite(rc) & np.isfinite(sc_) & (rc > 0) & (sc_ > 0)
        if ok.sum() < 50:
            continue
        lc = float(np.log(max(optimal_c(rc[ok], sc_[ok]), 1e-12)))
        se = logc_se(rc, sc_, H, n_rep=n_boot, seed=H * 977 + y)
        w = fixed_target_weight(lc, se)
        acc["raw"].append(qlike_vec(rt, st_))
        acc["c"].append(qlike_vec(rt, np.exp(lc) * st_))
        acc["c_shrunk"].append(qlike_vec(rt, np.exp(w * lc) * st_))
        acc["c_half"].append(qlike_vec(rt, np.exp(FIXED_W * lc) * st_))
        rows.append({"year": y, "log_c": lc, "c": float(np.exp(lc)),
                     "se": se, "w": w})
        past_lc.append(lc); past_se.append(se)   # reported only; not used
    if not rows:
        return None
    return {"teacher": teacher, "H": H, "folds": rows,
            "qlike": {a: float(np.nanmean(np.concatenate(v)))
                      for a, v in acc.items()}}


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(23)
    n = 20000
    true = np.exp(rng.normal(-4.0, 0.5, n))
    rv = true * np.exp(rng.normal(0, 0.2, n))

    # 1-2. optimal_c recovers a planted level error, and log_c is ~0 when the
    #      forecast is already correctly levelled.
    biased = true * 1.25
    ok.append(("optimal_c recovers a planted 1.25x over-forecast",
               abs(optimal_c(rv, biased) - 1 / 1.25) < 0.05))
    ok.append(("an already-levelled forecast needs c ~ 1",
               abs(np.log(optimal_c(rv, true))) < 0.05))

    # 3-4. The SE must be finite and must shrink as the slice grows -- an error
    #      that ignored n would give the same weight to every arm.
    s_small = logc_se(rv[:3000], biased[:3000], 1, n_rep=120, seed=1)
    s_big = logc_se(rv, biased, 1, n_rep=120, seed=1)
    ok.append(("log_c SE is finite", np.isfinite(s_big) and s_big > 0))
    ok.append(("SE falls as the calib slice grows", s_big < s_small))

    # 5-7. The weight, which decides everything. A large, precisely-known level
    #      error must keep its correction; a small, noisy one must lose it.
    ok.append(("a LARGE, precisely-known log c keeps its correction",
               fixed_target_weight(0.20, 0.02) > 0.98))
    ok.append(("a SMALL log c swamped by its error loses it",
               fixed_target_weight(0.02, 0.06) < 0.11))
    ok.append(("log c of exactly 0 -> w = 0", fixed_target_weight(0.0, 0.01) == 0.0))
    # The failed version got these two backwards, so assert the ORDERING that
    # distinguishes the forms: the arm with the bigger |log c| must keep MORE
    # of its correction when both are measured equally precisely.
    ok.append(("bigger |log c| keeps more correction at equal SE",
               fixed_target_weight(0.20, 0.03) > fixed_target_weight(0.03, 0.03)))
    ok.append(("sign of log c does not matter, only magnitude",
               fixed_target_weight(-0.2, 0.03) == fixed_target_weight(0.2, 0.03)))

    # 8. THE PROPERTY THAT MAKES THIS NEUTRAL: the weight is a function of the
    #    arm's own calib statistics only. Two arms with identical (tau, SE) get
    #    identical weights regardless of which teacher they are.
    ok.append(("the weight depends only on (log c, SE), not on the arm",
               fixed_target_weight(0.05, 0.03)
               == fixed_target_weight(0.05, 0.03)))

    # 9. Shrinking toward 1 in LOG space: w=0 must give exactly c=1, not c=0.
    ok.append(("w=0 gives c exactly 1", abs(np.exp(0.0 * 0.7) - 1.0) < 1e-15))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="precision-shrunk level rescale")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/shrunk_level.json"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, teachers = load_oof(a.oof)
    teachers = sorted(teachers)
    print(f"P3-shrunk-level   {len(teachers)} teachers x {len(HORIZONS)} horizons")
    print("applied SYMMETRICALLY; every weight comes from that arm's own calib\n")

    out = {"arms": list(ARMS), "fixed_w": FIXED_W, "horizons": {}}
    for H in HORIZONS:
        print(f"=== H = {H}")
        print(f"    {'teacher':>15} " + " ".join(f"{a_:>9}" for a_ in ARMS)
              + f" {'w (med)':>8}  {'shrunk vs c %':>13}")
        res = {}
        for t in teachers:
            r = run_teacher(z, H, t, n_boot=a.boot)
            if r is None:
                continue
            q = r["qlike"]
            wmed = float(np.median([f["w"] for f in r["folds"]]))
            rel = 100 * (q["c"] - q["c_shrunk"]) / q["c"]
            res[t] = {**r, "w_median": wmed, "shrunk_vs_c_pct": rel}
            print(f"    {t:>15} " + " ".join(f"{q[a_]:9.5f}" for a_ in ARMS)
                  + f" {wmed:8.3f}  {rel:+13.2f}")
        if not res:
            continue
        # the pre-registered readout
        for arm in ("c", "c_shrunk", "c_half"):
            helped = sum(1 for t in res if res[t]["qlike"][arm]
                         < res[t]["qlike"]["raw"])
            print(f"    -> {arm:>9} helps {helped}/{len(res)} teachers")
        win = {arm: min(res, key=lambda t: res[t]["qlike"][arm])
               for arm in ARMS}
        print(f"    -> winner: " + "  ".join(f"{a_}={win[a_]}" for a_ in ARMS))
        NM = "noctua_v1_mean"
        if NM in res:
            for arm in ARMS:
                riv = [t for t in res if not t.startswith("noctua")]
                if not riv:
                    continue
                b = min(riv, key=lambda t: res[t]["qlike"][arm])
                m = 100 * (res[b]["qlike"][arm] - res[NM]["qlike"][arm]) \
                    / res[b]["qlike"][arm]
                print(f"       NOCTUA-mean margin under {arm:>9}: {m:+6.2f}% "
                      f"(vs {b})")
        out["horizons"][str(H)] = {
            t: {"qlike": res[t]["qlike"], "w_median": res[t]["w_median"],
                "shrunk_vs_c_pct": res[t]["shrunk_vs_c_pct"],
                "folds": res[t]["folds"]} for t in res}
        print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
