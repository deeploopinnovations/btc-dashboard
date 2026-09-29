"""
eval/transfer_anatomy.py
=====================================================================
P3-transfer-anatomy: WHICH calibration parameter survives its own fold?

WHY THIS EXISTS, AND WHY IT IS NOT A CANDIDATE

Two shrinkage experiments in this project shrink a fitted correction toward a
fixed target -- `eval/shrunk_level.py` toward log c = 0, `eval/shrunk_slope.py`
toward beta = 1 -- and both now use, or are being asked to use, the same
weight: the magnitude of the estimate against its own standard error. That
weight asks ONE question: is the departure from the target real?

In `P3-shrunk-slope-v2` I wrote, as a pre-registered note, that this is the
right question for the level because "c fitted on calib is close to c on test"
and the wrong one for the slope because a slope can be significant and
untransferable. I asserted both halves. Asserting a mechanism instead of
measuring it is the single most repeated error in this project's history
(P3-dispersion-qlike-mechanism, twice in one afternoon), so this module
measures it.

THE STATISTIC

For each teacher, horizon and fold, four arms are fitted and scored on the
SAME test slice:

    raw                     no correction
    c_feasible              level fitted on CALIB           (what is shipped)
    c_oracle                level fitted on TEST            (infeasible)
    mzq_feasible            MZ slope + QLIKE level on CALIB
    mzq_oracle              MZ slope + QLIKE level on TEST   (infeasible)

and the correction's TRANSFER FRACTION is

    value   = QLIKE(raw) - QLIKE(feasible)      what the correction delivers
    ceiling = QLIKE(raw) - QLIKE(oracle)        what it could deliver if the
                                                calib fit held exactly on test
    transfer = value / ceiling

1.0 means the parameter estimated on calib is worth everything it would be
worth if known exactly; 0 means estimating it on calib buys nothing; NEGATIVE
means the calib estimate is worse than not correcting at all -- the correction
is anti-transferring, which is a different failure from being imprecise and is
the one a magnitude-against-error weight cannot see.

THE ORACLE ARMS ARE INFEASIBLE BY CONSTRUCTION AND MUST NEVER LEAVE THIS FILE.
They read the test slice they are scored on. They exist to put a denominator
under the feasible arm and are labelled `_oracle` everywhere so that a later
reader cannot mistake one for a candidate. Nothing here is registered for
adoption and nothing here produces a served quantity.

HOW IT CAME OUT, AND THE PART THAT IS NOT A MEASUREMENT
(`P3-transfer-anatomy-audited`). The module's first version reported
`rho(drift/signal, transfer) = -0.564` with a permutation `p < 0.0001` and
called it one quantity explaining the whole family. For the LEVEL that is an
IDENTITY, not a finding: QLIKE is locally quadratic in `log c`, so
`transfer = 1 - (drift/signal)^2` by construction, and measured against that
prediction the level lands within a median absolute error of 0.020. A
permutation null breaks a pairing the definitions fix, so it tested the
identity.

For the SLOPE the identity FAILS -- rho +(-0.168), median error 0.751 -- and
that failure is the finding. Applying `sigma^beta` moves each forecast by an
amount proportional to that episode's `log sigma`, so the loss is not a
quadratic in `(beta - 1)` alone. `quadratic_transfer` computes the prediction
and `main` prints it directly under the correlation it deflates, so the
number cannot be read again without it.

What survives unchanged: the measured drifts and signals themselves, and the
production arm's level correction anti-transferring at every horizon with a
positive oracle ceiling. The identity says how measured quantities combine; it
does not manufacture them.

    python -m model.eval.transfer_anatomy --selftest
    python -m model.eval.transfer_anatomy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.mz_recalibration import (YEARS, apply_mz, fit_mz,              # noqa: E402
                                   optimal_c, qlike_vec)
from eval.teacher_scorecard import HORIZONS, load_oof                    # noqa: E402
from eval.teacher_zoo import FoldScopedFit                               # noqa: E402

TEACHERS = ("garch_normal", "garch_t", "har_short", "log_har", "log_har_cal",
            "noctua_v1", "noctua_v1_mean", "persistence")


def fit_pair(rv, sig) -> dict | None:
    """(alpha, beta) for the level-only and the slope-and-level correction.

    Both fitted on whatever slice is handed in -- which is what makes the same
    function serve the feasible arm (calib) and the oracle arm (test). The
    caller decides which slice that is, and the caller is the only place the
    distinction between feasible and infeasible lives.
    """
    rv = np.asarray(rv, np.float64); sig = np.asarray(sig, np.float64)
    ok = np.isfinite(rv) & np.isfinite(sig) & (rv > 0) & (sig > 0)
    if ok.sum() < 200:
        return None
    rv, sig = rv[ok], sig[ok]
    p = fit_mz(rv, sig)
    if p is None:
        return None
    beta = float(p[1])
    return {
        "c": (float(np.log(optimal_c(rv, sig))), 1.0),
        "mzq": (float(np.log(optimal_c(rv, sig ** beta))), beta),
        "beta": beta,
    }


def transfer_fraction(q_raw: float, q_feas: float, q_orac: float) -> float:
    """value / ceiling, NaN when the ceiling is not a ceiling.

    The ceiling can fail to be positive -- an oracle correction that does not
    beat `raw` means the correction has no value to transfer at all, and a
    ratio against a zero or negative denominator is not interpretable. Return
    NaN rather than a number whose sign is an artefact of the denominator;
    a ratio that flips sign because its denominator did is exactly how a
    diagnostic starts lying.
    """
    ceiling = q_raw - q_orac
    if not np.isfinite(ceiling) or ceiling <= 0:
        return float("nan")
    return float((q_raw - q_feas) / ceiling)


def quadratic_transfer(drift: float, signal: float) -> float:
    """What `transfer` MUST be if the loss is locally quadratic in the
    parameter around its optimum.

    THE CONTROL THIS MODULE SHIPPED WITHOUT, AND IT DEFLATES THE MODULE'S OWN
    HEADLINE. QLIKE as a function of log c is locally quadratic about the
    QLIKE-optimal level, so with k the curvature,

        q_raw  - q_oracle ~= k * signal^2      (the fitted departure)
        q_feas - q_oracle ~= k * drift^2       (the calib-to-test movement)
        transfer = (q_raw - q_feas)/(q_raw - q_oracle) = 1 - (drift/signal)^2

    That is ARITHMETIC. A correlation between drift/signal and transfer is
    therefore expected by construction for any parameter whose loss surface is
    locally quadratic, and a permutation test against a null that breaks the
    pairing is testing the identity rather than the data.

    It is still worth computing, because it is a PREDICTION that can fail --
    and for the SLOPE it does. Applying sigma^beta moves each forecast by an
    amount proportional to that episode's log sigma, so the loss is not a
    quadratic in (beta - 1) alone and the dispersion of log sigma enters. Where
    the identity holds, the correlation is bookkeeping; where it breaks, the
    departure is the finding.
    """
    if not (np.isfinite(drift) and np.isfinite(signal)) or signal <= 0:
        return float("nan")
    return float(1.0 - (drift / signal) ** 2)


def spearman(x, y) -> float:
    """Rank correlation on the pairs where both are finite.

    Ranks rather than levels because a transfer fraction is unbounded below --
    one anti-transferring teacher at -2.7 would otherwise decide a Pearson
    correlation computed over eight points by itself.
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(cells: list, n_perm: int = 20_000, seed: int = 0) -> dict:
    """Pooled rank correlation with a WITHIN-CELL permutation null.

    `cells` is a list of (x, y) arrays, one per (horizon, parameter) group.
    The pooled correlation over all groups looks like it rests on 64 points; it
    does not. The same eight teachers appear in every group and the four
    horizons are nested forecasts of one series, so the points are strongly
    dependent and the textbook p-value for n = 64 would be a fabrication.

    Permuting `y` WITHIN each cell destroys the association being tested while
    leaving intact everything that creates the dependence -- which teachers
    exist, how transfer is distributed at each horizon, and the fact that a
    horizon's eight values share a scale. The resulting null is the honest one
    for this design, and it is wider than the textbook null by construction.
    """
    xs = [np.asarray(x, np.float64) for x, _ in cells]
    ys = [np.asarray(y, np.float64) for _, y in cells]
    obs = spearman(np.concatenate(xs), np.concatenate(ys))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        perm = [rng.permutation(y) for y in ys]
        r = spearman(np.concatenate(xs), np.concatenate(perm))
        if np.isfinite(r) and abs(r) >= abs(obs):
            hits += 1
    return {"rho": obs, "n_perm": n_perm,
            "p": float((hits + 1) / (n_perm + 1)),
            "n_pairs": int(sum(np.isfinite(x).sum() for x in xs))}


def run_teacher(z, H: int, teacher: str) -> dict | None:
    """Pool the four arms over folds, plus the per-fold parameter movement."""
    acc = {k: [] for k in ("raw", "c_feasible", "c_oracle",
                           "mzq_feasible", "mzq_oracle")}
    rv_acc, rows = [], []
    for y in YEARS:
        with FoldScopedFit(year=y) as sc:
            kt, kc = f"{y}/{H}/test", f"{y}/{H}/calib"
            if f"{kt}/sigma/{teacher}" not in z or f"{kc}/sigma/{teacher}" not in z:
                continue
            sig_c = np.asarray(sc.calib(z, H, teacher), np.float64)
            rv_c = np.asarray(z[f"{kc}/rv"], np.float64)
            sig_t = np.asarray(sc.test(z, H, teacher), np.float64)
            rv_t = np.asarray(z[f"{kt}/rv"], np.float64)
        fc, ft = fit_pair(rv_c, sig_c), fit_pair(rv_t, sig_t)
        if fc is None or ft is None:
            continue
        rv_acc.append(rv_t)
        acc["raw"].append(sig_t)
        for arm, src, key in (("c_feasible", fc, "c"), ("c_oracle", ft, "c"),
                              ("mzq_feasible", fc, "mzq"),
                              ("mzq_oracle", ft, "mzq")):
            al, be = src[key]
            acc[arm].append(apply_mz(sig_t, al, be))
        rows.append({"year": y,
                     "logc_calib": fc["c"][0], "logc_test": ft["c"][0],
                     "beta_calib": fc["beta"], "beta_test": ft["beta"]})
    if not rows:
        return None
    rv = np.concatenate(rv_acc)
    q = {k: float(np.nanmean(qlike_vec(rv, np.concatenate(v))))
         for k, v in acc.items()}
    dl = np.array([r["logc_test"] - r["logc_calib"] for r in rows])
    db = np.array([r["beta_test"] - r["beta_calib"] for r in rows])
    sl = np.array([r["logc_calib"] for r in rows])
    sb = np.array([r["beta_calib"] - 1.0 for r in rows])
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    return {
        "qlike": q, "folds": rows,
        "signal_c": rms(sl), "drift_c": rms(dl),
        "signal_b": rms(sb), "drift_b": rms(db),
        "transfer_c": transfer_fraction(q["raw"], q["c_feasible"], q["c_oracle"]),
        "transfer_b": transfer_fraction(q["raw"], q["mzq_feasible"],
                                        q["mzq_oracle"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="which correction transfers")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/transfer_anatomy.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    z, _ = load_oof(a.oof)
    print("P3-transfer-anatomy   DIAGNOSTIC, no candidate, no pre-registered "
          "decision rule")
    print("transfer = (raw - feasible) / (raw - oracle); 1.0 = the calib fit "
          "is worth everything\n")
    out = {"teachers": TEACHERS, "horizons": {}}
    for H in HORIZONS:
        print(f"=== H = {H}")
        print(f"{'teacher':>16} | {'|log c|':>8} {'drift':>7} | {'xfer_c':>7} "
              f"| {'|b-1|':>7} {'drift':>7} | {'xfer_b':>7}")
        hh = {}
        for t in TEACHERS:
            r = run_teacher(z, H, t)
            if r is None:
                continue
            hh[t] = r
            print(f"{t:>16} | {r['signal_c']:8.4f} {r['drift_c']:7.4f} "
                  f"| {r['transfer_c']:7.3f} | {r['signal_b']:7.3f} "
                  f"{r['drift_b']:7.3f} | {r['transfer_b']:7.3f}")
        out["horizons"][str(H)] = hh
        xc = [v["transfer_c"] for v in hh.values() if np.isfinite(v["transfer_c"])]
        xb = [v["transfer_b"] for v in hh.values() if np.isfinite(v["transfer_b"])]
        if xc and xb:
            print(f"{'median':>16} | {'':>8} {'':>7} | {np.median(xc):7.3f} "
                  f"| {'':>7} {'':>7} | {np.median(xb):7.3f}")
        print()

    # ---- the one claim this module is for -------------------------------
    # Does the MAGNITUDE of a fitted correction predict whether it transfers?
    # That is the question the magnitude-against-error weight silently answers
    # "yes" to, in both shrinkage modules, for both parameters.
    print("does |departure| predict transfer?  (Spearman over 8 teachers; "
          "n = 8, so\nany single row needs |rho| >= 0.74 to clear p = 0.05 "
          "on its own)\n")
    print(f"{'H':>6} {'rho(|log c|, xfer_c)':>22} {'rho(|b-1|, xfer_b)':>22}")
    summ, cells = {}, []
    for H in HORIZONS:
        hh = out["horizons"].get(str(H), {})
        if not hh:
            continue
        rc = spearman([v["signal_c"] for v in hh.values()],
                      [v["transfer_c"] for v in hh.values()])
        rb = spearman([v["signal_b"] for v in hh.values()],
                      [v["transfer_b"] for v in hh.values()])
        summ[str(H)] = {"rho_mag_c": rc, "rho_mag_b": rb}
        print(f"{H:>6} {rc:22.3f} {rb:22.3f}")
        for sk, dk, xk in (("signal_c", "drift_c", "transfer_c"),
                           ("signal_b", "drift_b", "transfer_b")):
            rat = [v[dk] / v[sk] if v[sk] > 0 else np.nan for v in hh.values()]
            xf = [v[xk] for v in hh.values()]
            keep = [i for i, (a_, b_) in enumerate(zip(rat, xf))
                    if np.isfinite(a_) and np.isfinite(b_)]
            if len(keep) >= 4:
                cells.append(([rat[i] for i in keep], [xf[i] for i in keep]))
    if cells:
        pr = permutation_p(cells)
        summ["pooled_drift_ratio"] = pr
        print(f"\npooled rho(drift/signal, transfer) = {pr['rho']:+.3f} over "
              f"{pr['n_pairs']} pairs\n  within-cell permutation p = "
              f"{pr['p']:.4f} ({pr['n_perm']} permutations)")

    # THE IDENTITY CHECK, printed under the correlation it deflates.
    print("\nis that correlation arithmetic?  transfer = 1 - (drift/signal)^2 "
          "holds exactly\nfor a loss that is locally quadratic in its "
          "parameter, so where the prediction\nlands on the measurement the "
          "correlation above is bookkeeping, not evidence.")
    ident = {}
    for key, sk, dk, xk in (("level", "signal_c", "drift_c", "transfer_c"),
                            ("slope", "signal_b", "drift_b", "transfer_b")):
        pred, obs = [], []
        for hh in out["horizons"].values():
            for v in hh.values():
                q = quadratic_transfer(v[dk], v[sk])
                if np.isfinite(q) and np.isfinite(v[xk]):
                    pred.append(q); obs.append(v[xk])
        if len(pred) < 4:
            continue
        r = spearman(pred, obs)
        err = float(np.median(np.abs(np.array(pred) - np.array(obs))))
        ident[key] = {"rho": r, "median_abs_error": err, "n": len(pred)}
        verdict = ("IDENTITY HOLDS -- the correlation is bookkeeping"
                   if err < 0.10 else
                   "IDENTITY BREAKS -- the departure is the finding")
        print(f"  {key:>6}: rho(predicted, observed) {r:+.3f}   "
              f"median |error| {err:.3f}   n {len(pred)}   {verdict}")
    summ["quadratic_identity"] = ident
    out["summary"] = summ

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(11)
    n = 6000

    # 1-2. A correction that is IDENTICAL on calib and test transfers fully.
    #      Same generating process on both slices, so the calib fit is the
    #      test fit up to sampling noise and the transfer fraction is ~1.
    def draw(seed, beta_true=1.0, scale=1.0):
        g = np.random.default_rng(seed)
        true = np.exp(g.normal(-4.0, 0.5, n))
        rv = true * np.exp(g.normal(0, 0.2, n))
        sig = scale * np.exp(np.log(true) / beta_true)
        return rv, sig

    rv_a, sig_a = draw(1, beta_true=1.4, scale=1.3)
    rv_b, sig_b = draw(2, beta_true=1.4, scale=1.3)
    fa, fb = fit_pair(rv_a, sig_a), fit_pair(rv_b, sig_b)
    q_raw = float(np.nanmean(qlike_vec(rv_b, sig_b)))
    q_fe = float(np.nanmean(qlike_vec(rv_b, apply_mz(sig_b, *fa["mzq"]))))
    q_or = float(np.nanmean(qlike_vec(rv_b, apply_mz(sig_b, *fb["mzq"]))))
    xf = transfer_fraction(q_raw, q_fe, q_or)
    ok.append(("a stationary correction transfers ~fully", 0.9 < xf <= 1.02))
    ok.append(("the oracle is never beaten by the feasible fit on its own "
               "slice", q_or <= q_fe + 1e-12))

    # 3. A correction whose SLOPE changes between the slices anti-transfers:
    #    fitting beta on a slice generated with 1/1.4 and applying it to one
    #    generated with 1/0.7 must score WORSE than no correction, i.e. a
    #    NEGATIVE transfer fraction -- the failure mode the whole module is
    #    about, and one that a large, precise beta cannot warn about.
    rv_c, sig_c = draw(3, beta_true=0.7, scale=1.3)
    fcal = fit_pair(*draw(4, beta_true=1.4, scale=1.3))
    forc = fit_pair(rv_c, sig_c)
    q_raw2 = float(np.nanmean(qlike_vec(rv_c, sig_c)))
    q_fe2 = float(np.nanmean(qlike_vec(rv_c, apply_mz(sig_c, *fcal["mzq"]))))
    q_or2 = float(np.nanmean(qlike_vec(rv_c, apply_mz(sig_c, *forc["mzq"]))))
    ok.append(("a moved slope anti-transfers (negative fraction)",
               transfer_fraction(q_raw2, q_fe2, q_or2) < 0.0))

    # 4-5. The guard on the denominator.
    ok.append(("a non-positive ceiling returns NaN, not a signed ratio",
               not np.isfinite(transfer_fraction(1.0, 0.9, 1.0))))
    ok.append(("a real ceiling returns the ratio",
               abs(transfer_fraction(1.0, 0.9, 0.8) - 0.5) < 1e-12))

    # 6-7. fit_pair nests the level arm at beta = 1 exactly, and refuses on a
    #      slice too small to fit rather than returning a fabricated parameter.
    ok.append(("the level arm is beta = 1 exactly", fa["c"][1] == 1.0))
    ok.append(("too few points -> None, not a guess",
               fit_pair(rv_a[:50], sig_a[:50]) is None))

    # 8. A PERFECT forecast needs no correction, so both arms must leave it
    #    alone and the ceiling must collapse -- which makes the fraction NaN
    #    rather than 0/0 or a spurious 1.
    perfect_rv = np.exp(rng.normal(-4.0, 0.5, n))
    fp = fit_pair(perfect_rv, perfect_rv)
    ok.append(("a perfect forecast is left alone",
               abs(fp["beta"] - 1.0) < 0.02 and abs(fp["c"][0]) < 0.02))

    # 9-12. The two statistics the module's claim rests on.
    ok.append(("spearman is 1 on a monotone pair",
               abs(spearman([1, 2, 3, 4, 5], [10, 20, 31, 44, 50]) - 1.0) < 1e-12))
    ok.append(("spearman ignores the non-finite pairs",
               abs(spearman([1, 2, 3, 4, float("nan")], [1, 2, 3, 4, 0.0]) - 1.0)
               < 1e-12))
    ok.append(("too few finite pairs -> NaN, not a correlation of noise",
               not np.isfinite(spearman([1, 2, 3], [3, 2, 1]))))
    # A planted association must be detected and a null one must not be, or
    # the permutation is decoration. Two cells of eight, matching the shape of
    # the real call.
    g = np.random.default_rng(7)
    xa, xb = g.normal(size=8), g.normal(size=8)
    planted = [(xa, -xa + 0.05 * g.normal(size=8)),
               (xb, -xb + 0.05 * g.normal(size=8))]
    null = [(xa, g.normal(size=8)), (xb, g.normal(size=8))]
    ok.append(("permutation detects a planted association",
               permutation_p(planted, n_perm=2000, seed=1)["p"] < 0.01))
    ok.append(("permutation does not fire on noise",
               permutation_p(null, n_perm=2000, seed=1)["p"] > 0.05))

    # 13-15. The identity, and that it is a PREDICTION rather than a tautology
    #     about whatever numbers are handed to it.
    ok.append(("no drift means the estimate transfers fully",
               quadratic_transfer(0.0, 0.2) == 1.0))
    ok.append(("drift equal to signal means it buys nothing",
               abs(quadratic_transfer(0.2, 0.2)) < 1e-12))
    ok.append(("drift larger than signal anti-transfers",
               quadratic_transfer(0.4, 0.2) < -2.9))
    ok.append(("a zero signal has no ceiling, so no prediction",
               not np.isfinite(quadratic_transfer(0.1, 0.0))))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
