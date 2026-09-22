"""
eval/rolling_level.py
=====================================================================
P3-rolling-level: the evaluation protocol fits the level once per fold. The
SERVING path re-fits it every hour from a trailing window. Which one is right?

THE DEFECT THAT MOTIVATES THIS, MEASURED AND NOT ASSUMED

`P3-transfer-anatomy` scored every teacher's level correction against an
infeasible oracle fitted on the test slice. For `noctua_v1_mean` -- the
production arm -- the fraction of the correction's value that survives the
calib fit is

    H = 1       6       24      168
      -1.132  -1.443  -1.442  -0.380

Negative at every horizon: the calib-fitted level is WORSE than not correcting
at all, while the oracle ceiling stays positive, so there is a level worth
correcting and the calib estimate points the wrong way. The cause is visible in
one comparison: |log c| is 0.059-0.072 and its calib-to-test movement is
0.077-0.089, so the estimate is smaller than its own drift, and its sign flips
between folds (2021 -0.076 -> +0.064, 2022 +0.034 -> -0.048).

That is not an argument against correcting the level. It is an argument against
correcting it ONCE PER YEAR. `serve/adaptive.py` already says so in prose --
"the bias is a REGIME property, not a fixed model defect, so the correction has
to move with the regime" -- and re-fits from a trailing 60-day window of
SETTLED episodes at every anchor. The evaluation protocol never adopted that.
So the scorecard has been charging every teacher for a correction the shipped
system does not make.

THE ARMS

    raw          no correction
    c            level fitted on the whole CALIB slice, applied to the whole
                 test year. THE SHIPPED PROTOCOL.
    c_roll       level re-fitted at every test anchor from episodes that have
                 already SETTLED, within a trailing window.
    c_expand     the same, with no window: every settled episode so far. The
                 control for "is it recency, or just more data?"
    c_roll_old   the trailing-window estimator evaluated ONCE at the first test
                 anchor and then frozen. Same estimator, same window length,
                 same quantity of data -- it differs from `c_roll` in RECENCY
                 and in nothing else.

`c_roll_old` IS THE EXPERIMENT. Five times now a trivial control has matched or
beaten a measured rule in this project (R82), and the obvious way for this one
to be fool's gold is that a 60-day window simply beats a 176-day one for
reasons having nothing to do with tracking a regime. If `c_roll` does not
separate from `c_roll_old`, the honest report is "a shorter window helps" and
the regime story is decoration.

WINDOW AND MINIMUM ARE NOT TUNED. Both are taken from `serve/adaptive.py`
unchanged -- 60 days, 20 episodes -- because they were fixed there long before
this question existed. No search is run over either, and a result that depends
on searching them is not a result.

CAUSALITY. An episode enters a window only once its full H-hour horizon has
closed strictly before the anchor being corrected, so an episode can never
contribute to its own correction. Calib episodes are eligible: they precede the
test slice by a 192-hour embargo in every fold.

    python -m model.eval.rolling_level --selftest
    python -m model.eval.rolling_level
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
from eval.scorecard_rescaled import optimal_c                            # noqa: E402
from eval.teacher_scorecard import HORIZONS, load_oof                    # noqa: E402
from eval.teacher_zoo import FoldScopedFit                               # noqa: E402
from eval.transfer_anatomy import TEACHERS, transfer_fraction            # noqa: E402

ARMS = ("raw", "c", "c_roll", "c_expand", "c_roll_old")
CONTRASTS = ("c", "c_roll_old", "c_expand")      # what c_roll is tested against
PRIMARY = "noctua_v1_mean"        # named in advance; see the note in main()
WINDOW_DAYS = 60                  # from serve/adaptive.py, unchanged
MIN_EPISODES = 20                 # from serve/adaptive.py, unchanged
N_BOOT = 4000


def rolling_c(anchor_ts, r, H: int, t_eval, window_days: int | None,
              fallback: float, min_episodes: int = MIN_EPISODES) -> np.ndarray:
    """The QLIKE-optimal level at each evaluation time, from SETTLED history.

    `anchor_ts` and `r = RV^2/sigma^2` describe the history pool, sorted by
    anchor time. `t_eval` are the times at which a correction is needed.
    An episode is eligible at time t when

        anchor_ts + H hours <= t            it has settled
        anchor_ts > t - window              it is inside the window

    and the correction is sqrt(mean(r)) over the eligible set, which is exactly
    `optimal_c` restricted to that set. `window_days=None` drops the lower
    bound, giving the expanding arm.

    Below `min_episodes` the window returns `fallback` rather than a number
    built from three observations. The fallback is passed in rather than
    defaulted to 1.0 because "no estimate" and "no correction" are different
    statements and conflating them is what cost `drift` a third of its sample
    (P3-warm-start-result).
    """
    anchor_ts = np.asarray(anchor_ts, np.float64)
    r = np.asarray(r, np.float64)
    t_eval = np.asarray(t_eval, np.float64)
    good = np.isfinite(r) & (r > 0)
    # cumulative sums over the FULL pool, with invalid entries contributing
    # nothing to either the sum or the count, so a window's mean is over the
    # usable episodes it contains rather than over its length
    cs = np.concatenate([[0.0], np.cumsum(np.where(good, r, 0.0))])
    cn = np.concatenate([[0.0], np.cumsum(good.astype(np.float64))])
    settle = anchor_ts + H * 3600.0
    hi = np.searchsorted(settle, t_eval, side="right")
    if window_days is None:
        lo = np.zeros_like(hi)
    else:
        lo = np.searchsorted(anchor_ts, t_eval - window_days * 86400.0,
                             side="right")
    lo = np.minimum(lo, hi)
    n = cn[hi] - cn[lo]
    s = cs[hi] - cs[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.sqrt(np.maximum(s / np.where(n > 0, n, np.nan), 1e-12))
    return np.where(n >= min_episodes, c, fallback)


def build_arms(z, H: int, teacher: str) -> dict | None:
    """Every arm's corrected test-slice sigma, pooled over folds."""
    acc = {a: [] for a in ARMS}
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
            ts_c = np.asarray(z[f"{kc}/anchor_ts"], np.float64)
            ts_t = np.asarray(z[f"{kt}/anchor_ts"], np.float64)
        ok_c = np.isfinite(rv_c) & np.isfinite(sig_c) & (rv_c > 0) & (sig_c > 0)
        if ok_c.sum() < 200:
            continue
        c_fold = optimal_c(rv_c[ok_c], sig_c[ok_c])
        # the history pool is calib followed by test, both already sorted and
        # separated by the fold's embargo
        ts = np.concatenate([ts_c, ts_t])
        r = np.concatenate([rv_c ** 2 / np.maximum(sig_c, 1e-12) ** 2,
                            rv_t ** 2 / np.maximum(sig_t, 1e-12) ** 2])
        if not np.all(np.diff(ts) >= 0):
            raise SystemExit(f"REFUSING: fold {y} H={H} history pool is not "
                             f"time-ordered; every window bound assumes it is.")
        roll = rolling_c(ts, r, H, ts_t, WINDOW_DAYS, c_fold)
        expand = rolling_c(ts, r, H, ts_t, None, c_fold)
        frozen = float(roll[0])
        rv_acc.append(rv_t)
        acc["raw"].append(sig_t)
        acc["c"].append(sig_t * c_fold)
        acc["c_roll"].append(sig_t * roll)
        acc["c_expand"].append(sig_t * expand)
        acc["c_roll_old"].append(sig_t * frozen)
        rows.append({"year": y, "c_fold": float(c_fold), "c_frozen": frozen,
                     "c_roll_med": float(np.nanmedian(roll)),
                     "c_roll_min": float(np.nanmin(roll)),
                     "c_roll_max": float(np.nanmax(roll)),
                     "n_fallback": int((roll == c_fold).sum())})
    if not rows:
        return None
    rv = np.concatenate(rv_acc)
    q = {a: qlike_vec(rv, np.concatenate(v)) for a, v in acc.items()}
    # LEAK GUARD. A correction fitted on the slice it is scored on drives the
    # post-correction ratio to exactly 1 (eval/scorecard_rescaled.leak_folds).
    # `c` is fitted on calib so it cannot; `c_roll` touches test history, so it
    # is the one that has to be checked, and checked per fold because a pooled
    # ratio of a partially-leaking run says nothing.
    ratio = float(np.nanmean(rv ** 2 /
                             np.maximum(np.concatenate(acc["c_roll"]), 1e-12) ** 2))
    return {"q": q, "folds": rows, "roll_ratio": ratio,
            "qlike": {a: float(np.nanmean(v)) for a, v in q.items()}}


def vs_drift(roll_path: Path, xfer_path: Path) -> dict:
    """Does the MEASURED level drift rank which teachers rolling helps?

    The two artifacts are produced by different modules and the drift was
    measured before this experiment ran, so the correlation is not a curve
    fitted to its own answer. It is still an ORACLE predictor -- `drift_c`
    is a calib-to-test movement and therefore reads the test slice -- so it
    ranks the outcome without being usable to choose an estimator in advance.
    Saying which of those two things a number is worth the one extra
    paragraph it costs.
    """
    from eval.transfer_anatomy import permutation_p, spearman

    R = json.loads(Path(roll_path).read_text())
    X = json.loads(Path(xfer_path).read_text())
    print("does a teacher's MEASURED level drift rank the gain from rolling?")
    print("  (n = 8 teachers per horizon; |rho| >= 0.738 clears p = 0.05)\n")
    print(f"{'H':>6} {'rho(drift_c, roll gain)':>25} {'teachers':>10}")
    out, cells = {}, []
    for H in HORIZONS:
        tt = R["horizons"].get(str(H), {}).get("teachers", {})
        xh = X["horizons"].get(str(H), {})
        xs, gs = [], []
        for t, v in tt.items():
            if t not in xh:
                continue
            q = v["qlike"]
            xs.append(xh[t]["drift_c"])
            gs.append((q["c"] - q["c_roll"]) / q["c"])
        if len(xs) < 4:
            continue
        r = spearman(xs, gs)
        out[str(H)] = {"rho": r, "n": len(xs)}
        cells.append((xs, gs))
        print(f"{H:>6} {r:25.3f} {len(xs):10d}")
    if cells:
        # The POOLED correlation is the weaker statement and is reported as
        # such: drift magnitudes are not comparable across horizons while
        # gains are, so pooling mixes scales. It is here because it carries a
        # permutation p-value the per-horizon rows cannot.
        pr = permutation_p(cells)
        out["pooled"] = pr
        print(f"\npooled rho = {pr['rho']:+.3f} over {pr['n_pairs']} pairs, "
              f"within-cell permutation p = {pr['p']:.4f}")
    for H in HORIZONS:
        tt = R["horizons"].get(str(H), {}).get("teachers", {})
        pref = [t for t, v in tt.items()
                if v["qlike"]["c_roll"] < v["qlike"]["c_roll_old"]]
        if tt:
            print(f"  H={H:>3}: {len(pref)}/{len(tt)} teachers prefer updating "
                  f"to freezing -> {', '.join(pref) or 'none'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rolling vs fold-fitted level")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/rolling_level.json"))
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--vs-drift", action="store_true",
                    help="read both artifacts and rank gain against measured "
                         "level drift; runs no model")
    ap.add_argument("--xfer", type=Path,
                    default=Path("model/artifacts/transfer_anatomy.json"))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.vs_drift:
        if not (a.out.exists() and a.xfer.exists()):
            print(f"REFUSING: needs both {a.out} and {a.xfer}; run this module "
                  f"and eval.transfer_anatomy first")
            return 1
        vs_drift(a.out, a.xfer)
        return 0

    z, _ = load_oof(a.oof)
    # FAMILY. Intervals are computed for ONE teacher -- the production arm,
    # named before the run -- because that is where P3-transfer-anatomy located
    # the defect, and because 96 Bonferroni-corrected intervals would resolve
    # nothing. The other seven teachers are reported as point estimates and win
    # counts, declared SECONDARY here so they cannot be promoted afterwards.
    # A fix that helps only the production arm is not a fix, it is special
    # pleading, and the win counts are what would expose that.
    n_family = len(HORIZONS) * len(CONTRASTS)
    alpha = 0.05 / n_family
    print(f"P3-rolling-level   window {WINDOW_DAYS}d, min {MIN_EPISODES} "
          f"settled episodes (both from serve/adaptive.py)")
    print(f"primary {PRIMARY}: family {n_family} -> {100*(1-alpha):.4f}% "
          f"intervals, {a.boot} bootstrap reps")
    print(f"secondary: the other seven teachers, point estimates only\n")

    out = {"window_days": WINDOW_DAYS, "min_episodes": MIN_EPISODES,
           "primary": PRIMARY, "n_family": n_family, "alpha": alpha,
           "horizons": {}}
    for H in HORIZONS:
        print(f"=== H = {H}")
        hh, wins = {}, {a_: 0 for a_ in ARMS if a_ != "raw"}
        print(f"{'teacher':>16} " + " ".join(f"{a_:>10}" for a_ in ARMS)
              + f" {'roll vs c %':>12}")
        for t in TEACHERS:
            r = build_arms(z, H, t)
            if r is None:
                continue
            hh[t] = {"qlike": r["qlike"], "folds": r["folds"],
                     "roll_ratio": r["roll_ratio"]}
            qr = r["qlike"]
            for a_ in wins:
                wins[a_] += int(qr[a_] < qr["raw"])
            gain = 100 * (qr["c"] - qr["c_roll"]) / qr["c"]
            print(f"{t:>16} " + " ".join(f"{qr[a_]:10.5f}" for a_ in ARMS)
                  + f" {gain:+12.2f}")
        for a_ in ("c", "c_roll", "c_expand", "c_roll_old"):
            print(f"    -> {a_:>11} helps {wins[a_]}/{len(hh)} teachers")

        r = build_arms(z, H, PRIMARY)
        ci = {}
        if r is not None:
            L = max(int(round(len(r['q']['raw']) ** (1 / 3))), 2 * H)
            print(f"\n   PRIMARY {PRIMARY}, block {L}")
            for other in CONTRASTS:
                d = r["q"][other] - r["q"]["c_roll"]     # >0 favours c_roll
                g = np.isfinite(d)
                c95 = mean_ci(d[g], n_rep=a.boot, alpha=alpha,
                              block_len=L)["ci95"]
                ci[f"c_roll_vs_{other}"] = [float(v) for v in c95]
                verdict = ("c_roll BETTER" if c95[0] > 0 else
                           "c_roll WORSE" if c95[1] < 0 else "not separated")
                print(f"   c_roll vs {other:>11}: {np.nanmean(d):+9.5f}  "
                      f"CI [{c95[0]:+.5f}, {c95[1]:+.5f}]  {verdict}")
            print(f"   post-correction test ratio under c_roll: "
                  f"{r['roll_ratio']:.4f}  (1.0000 would mean a leak)")
        out["horizons"][str(H)] = {"teachers": hh, "wins": wins, "ci": ci}
        print()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


def selftest() -> int:
    ok = []

    # 1-3. THE CAUSALITY GUARD, which is the only way this arm can cheat.
    #      One episode per hour, H = 5. At the anchor at hour k, episodes
    #      settling at or before k are hours <= k-5, so hour k itself and the
    #      four before it must be invisible.
    n = 400
    ts = np.arange(n, dtype=np.float64) * 3600.0
    r = np.ones(n)
    r[300:] = 100.0                      # a violent regime starting at hour 300
    c = rolling_c(ts, r, 5, ts, None, fallback=-1.0, min_episodes=1)
    ok.append(("an episode cannot see itself", abs(c[300] - 1.0) < 1e-9))
    ok.append(("nor the H-1 episodes before it", abs(c[304] - 1.0) < 1e-9))
    ok.append(("it sees the regime once those settle", c[306] > 1.0))

    # 4-5. The window is a window: with 10 days of history and a 1-day window,
    #      only the last day counts.
    ts2 = np.arange(240, dtype=np.float64) * 3600.0
    r2 = np.where(np.arange(240) < 216, 4.0, 9.0)
    c_w = rolling_c(ts2, r2, 1, np.array([239 * 3600.0]), 1.0, -1.0,
                    min_episodes=1)
    c_e = rolling_c(ts2, r2, 1, np.array([239 * 3600.0]), None, -1.0,
                    min_episodes=1)
    ok.append(("a 1-day window sees only the recent level",
               abs(float(c_w[0]) - 3.0) < 1e-9))
    ok.append(("the expanding arm sees both levels",
               2.0 < float(c_e[0]) < 3.0))

    # 6-7. The fallback is the CALIB constant, not 1.0, and it fires exactly
    #      when the window is too thin.
    thin = rolling_c(ts2[:5], r2[:5], 1, np.array([4 * 3600.0]), 1.0,
                     fallback=0.5, min_episodes=20)
    ok.append(("a thin window returns the fallback", float(thin[0]) == 0.5))
    fat = rolling_c(ts2, r2, 1, np.array([239 * 3600.0]), 1.0, 0.5,
                    min_episodes=20)
    ok.append(("a full window does not", float(fat[0]) != 0.5))

    # 8. Invalid episodes contribute to neither sum nor count, so a window
    #    half-filled with NaN still returns the mean of what is usable rather
    #    than a NaN or a diluted number.
    r3 = r2.copy(); r3[::2] = np.nan
    c_n = rolling_c(ts2, r3, 1, np.array([239 * 3600.0]), 1.0, -1.0,
                    min_episodes=1)
    ok.append(("NaN episodes are skipped, not counted",
               abs(float(c_n[0]) - 3.0) < 1e-9))

    # 9. On a STATIONARY history the rolling arm must reproduce the constant
    #    it would have been given, or the arm is not nested in the protocol it
    #    is meant to generalise.
    g = np.random.default_rng(4)
    r4 = (g.lognormal(0.0, 0.1, 2000)) * 1.44
    c_s = rolling_c(np.arange(2000, dtype=np.float64) * 3600.0, r4, 1,
                    np.array([1999 * 3600.0]), 30, -1.0)
    ok.append(("a stationary history gives back the constant",
               abs(float(c_s[0]) - 1.2) < 0.02))

    # 10. A window WIDER than the whole history must equal the expanding arm
    #     exactly. The first draft asserted only that the expanding arm was
    #     positive, which is true of every return value this function can
    #     produce and therefore tested nothing. This version would fail on an
    #     off-by-one in the lower bound, which is the error the check is for.
    t_last = np.array([239 * 3600.0])
    ok.append(("a window wider than the history IS the expanding arm",
               float(rolling_c(ts2, r2, 1, t_last, 10_000, -1.0,
                               min_episodes=1)[0])
               == float(rolling_c(ts2, r2, 1, t_last, None, -1.0,
                                  min_episodes=1)[0])))

    # 11. And a window of zero days sees nothing, so it must fall back rather
    #     than return the mean of an empty set.
    ok.append(("a zero-width window falls back",
               float(rolling_c(ts2, r2, 1, t_last, 0, 0.77,
                               min_episodes=1)[0]) == 0.77))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g_ for _, g_ in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
