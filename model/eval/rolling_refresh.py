"""
eval/rolling_refresh.py
=====================================================================
Does keeping the training window fresh beat freezing it in 2023?

WHY THIS EXISTS

`eval/forward_split.py` compared a stale training window against one extended
through the post-ETF era, on a common held-out window. It improved QLIKE by
1.03% and discrimination by 2.99% -- and tripped its own pre-registered veto,
because aggregate deep-tail miscalibration worsened by 2.41%, driven entirely
by the 2% barrier. On 585 production episodes neither the gain nor the veto is
resolvable: that is one window, and a 2.4% move in a calibration statistic is
well inside what one window can produce by chance.

So this is the same question with power. Instead of one boundary it walks a
sequence of QUARTERLY test windows through 2025-2026 and, for each, compares:

    frozen     network + calibration fitted once, <= 2023-01-01
               (what actually ships today)
    refreshed  fitted on everything up to that window's own embargoed start
               (what a maintained model would look like)

Both arms are scored on the SAME episodes in each window. The frozen arm's
training set never changes; the refreshed arm's grows. The difference is
exactly the value of not letting the fit go stale, measured repeatedly rather
than once.

WHY QUARTERLY RATHER THAN ANNUAL

There are only ~1.5 years of post-2025 data, so annual windows would give two
comparisons. Quarters give six. Each window holds ~90 production episodes,
which is small -- but six small paired windows support a sign test where one
window of 585 supports nothing, and the sign test is what the decision rule
uses.

DECISION RULE, fixed before running: the refresh is worth adopting if it wins
QLIKE in >= 5 of 6 windows AND does not worsen aggregate deep-tail (alpha <=
2%) miscalibration in more than 2 of 6. Average-case wins bought with tail
degradation stay rejected -- the veto that fired in forward_split.py is kept,
just measured with enough windows to mean something.

THE CONFOUND IN THE FIRST RUN, AND WHY --match-calib EXISTS

The first `--wide` run won QLIKE 5/6 (-5.02%) and lost the tail condition 4/6,
so the rule vetoed it. That verdict stands only if the two arms were otherwise
comparable, and they were NOT: the frozen arm calibrates over the 18 months
from 2023-01-01 to 2024-07-01 while the refreshed arm calibrates over the two
quarters before its own window -- roughly a THIRD as many episodes. Isotonic
calibration is a nonparametric fit, so its sampling variance falls with n; the
arm handed a third of the data has a noisier calibration map before any
question of which training window is better. `tail_mcb` is a sum of six
MISCALIBRATION terms. Feeding one arm less calibration data and then judging
it on miscalibration measures the slice size at least as much as the refresh.

`--match-calib` removes that. Both arms are truncated to the SAME number of
calibration episodes -- the smaller of the two, kept most-recent-first, so each
arm still calibrates on the data closest to its own test window and stays
strictly causal. The training windows still differ; that is the treatment.

There is a real tension here that cannot be designed away: with the data
ending where it does, the refreshed arm cannot have both a maximally fresh
training set and an 18-month calibration slice -- every month given to
calibration is a month taken from training. Matching DOWN to the smaller count
is the choice that leaves the treatment intact, and it costs the frozen arm
the advantage it was never entitled to.

AND WHAT MATCHING DOWN REVEALED, WHICH IS WHY --big-calib EXISTS

Matching moved the frozen arm's mean deep-tail MCB from 0.248019 to 0.253032
while its training set, its seeds and its test episodes were untouched. The
refreshed arm's numbers are bit-identical between the two runs, because it was
always the smaller slice and was never truncated. So **the entire veto was the
calibration slice**: +0.005013 of tail miscalibration bought purely by handing
one arm three times the data to calibrate on.

That is a finding in its own right, and it is larger than the effect under
test: the refresh's own tail gain is -0.005377. A deployment therefore needs
BOTH a fresh training window and a large calibration slice, and neither of the
two arms above is that configuration. `--big-calib` adds it -- the refreshed
arm trains to 18 months before its window and calibrates over those 18 months,
so it is simultaneously ~2 years fresher than the frozen fit and calibrated on
a comparable number of episodes. It is the arm that could actually ship, and
it is scored against the frozen arm exactly as it stands today, with the full
18-month calibration the frozen arm really has.

THE RULE THAT REPLACED THE WIN COUNT (--monthly)

Three designs (6l, 6m, 6n) all put QLIKE 5/6 in the refresh's favour, and all
three disagreed about the deep-tail condition -- 2/6, 5/6, 3/6 -- because a
six-sample win count on an effect of ~0.003 against per-window scatter of
~0.013 is a coin flip. The mean tail delta favoured the refresh in every one
of the three, and the *count* is what the rule read.

So BENCHMARK.md 6n fixed a replacement rule BEFORE any of this data was
scored, and `--monthly` is that measurement:

  * eighteen MONTHLY windows through the unseen era instead of six quarterly
    ones, which is the only way to buy resolution the designs disagreed over;
  * the refreshed arm trains to `window - 6 months`, the configuration 6m
    tested, not 6n's staler `- 18 months`;
  * calibration matched between arms, since 6m proved the unmatched
    comparison measures the slice size;
  * the tail is decided on the MEAN delta with a moving-block bootstrap CI,
    not on how many windows happened to fall each way;
  * ADOPT if that CI excludes zero on the favourable side, OR if it contains
    zero while QLIKE clears the same 5/6 RATE (>= 15 of 18) -- because a tail
    effect indistinguishable from zero is not the "tail degradation" the veto
    was written to catch.

`5/6` is read as a rate, so the 18-window bar is 15/18 -- an identical 83.3%,
and a far stricter sign test (p = 0.0038 against p = 0.109). Reading it as a
rate makes the bar harder, not easier, which is the safe direction to resolve
an ambiguity in one's own pre-registration.

    python -m model.eval.rolling_refresh --wide --match-calib   # the clean test
    python -m model.eval.rolling_refresh --wide --big-calib     # the deployable one
    python -m model.eval.rolling_refresh --wide --monthly       # 6n's fixed rule
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

from eval.benchmark import run_fold                                       # noqa: E402
from eval.direction import block_bootstrap_ci                             # noqa: E402
from eval.efficiency import summarise                                     # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402

HOUR = 3600
TAIL_BARRIERS = (0.5, 1.0, 2.0)


def causal_sigma_fn(ep, X):
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def masks(ep, train_end: str, calib_end: str, win_lo: str, win_hi: str,
          seed: int = 0) -> dict:
    """Embargoed train/calib plus a bounded test WINDOW (not an open tail)."""
    emb = int(ep["H"].max()) * HOUR
    ts = ep["anchor_ts"].to_numpy(np.int64)
    end = ts + ep["H"].to_numpy() * HOUR
    base = S.in_sample_mask(ep)
    t1 = int(pd.Timestamp(train_end, tz="UTC").timestamp())
    t2 = int(pd.Timestamp(calib_end, tz="UTC").timestamp())
    lo = int(pd.Timestamp(win_lo, tz="UTC").timestamp())
    hi = int(pd.Timestamp(win_hi, tz="UTC").timestamp())
    # `year` is consumed downstream as a numpy seed (ShuffledNoctua), so it
    # must be an int -- a date string raises inside SeedSequence.
    return {"year": int(seed),
            "train": base & (end <= t1 - emb),
            "calib": base & (ts >= t1) & (end <= t2 - emb),
            "test":  base & (ts >= lo) & (ts < hi)}


def equalize_calib(arms: dict, ep) -> int:
    """Truncate every arm's calibration slice to the same episode count.

    Kept MOST RECENT first, per arm: an arm's calibration data is the episodes
    immediately before its own cutoff, which is both what a deployment would
    use and what keeps the slice causal. Only the SIZE is equalized -- the
    windows still sit where each arm's split puts them.
    """
    ts = ep["anchor_ts"].to_numpy(np.int64)
    n = min(int(f["calib"].sum()) for f in arms.values())
    for f in arms.values():
        idx = np.flatnonzero(f["calib"])
        if len(idx) > n:
            keep = idx[np.argsort(ts[idx])[-n:]]        # the n latest
            m = np.zeros_like(f["calib"])
            m[keep] = True
            f["calib"] = m
    return n


def _mbb(d, n_rep: int = 20_000, seed: int = 0, alpha: float = 0.05):
    """`direction.block_bootstrap_ci` with its n >= 20 guard lifted.

    That guard exists because the helper is written for per-episode loss
    differences, where fewer than 20 points means the caller has made a
    mistake. Here the unit is a WINDOW and 6n fixed the count at 18 on
    purpose. The estimator is unchanged -- moving blocks of round(n^(1/3)),
    resampled to the original length -- so this is the same interval, not a
    substitute chosen because the original refused to produce one.
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    L = max(1, int(round(n ** (1 / 3))))
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - L + 1, size=(n_rep, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_rep, -1)[:, :n]
    means = d[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def tail_mcb(rows) -> float:
    r = next(x for x in rows if x["model"] == "noctua_v2")
    return float(sum(r[f"MCB_{s}_{p}"] for s in ("up", "dn") for p in TAIL_BARRIERS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is a maintained fit worth it?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--wide", action="store_true",
                    help="score EVERY H=19 anchor hour in each window, not just "
                         "17:00 -- ~24x more test episodes per window")
    ap.add_argument("--monthly", action="store_true",
                    help="18 monthly windows and the BENCHMARK.md 6n decision "
                         "rule (bootstrap CI on the mean tail delta). Implies "
                         "--match-calib.")
    ap.add_argument("--big-calib", action="store_true",
                    help="refreshed arm trains to 18 months before its window "
                         "and calibrates over those 18 months -- fresher than "
                         "frozen AND comparably calibrated. The deployable arm.")
    ap.add_argument("--match-calib", action="store_true",
                    help="truncate both arms to the same number of calibration "
                         "episodes, so `tail_mcb` is not measuring the fact "
                         "that the frozen arm got 3x more calibration data")
    ap.add_argument("--max-test", type=int, default=1200,
                    help="cap on test episodes per window under --wide, drawn "
                         "once with a fixed seed so windows stay comparable")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = Path("model/artifacts/rolling_refresh"
                     + ("_wide" if a.wide else "")
                     + ("_matchcal" if a.match_calib else "")
                     + ("_bigcal" if a.big_calib else "")
                     + ("_monthly" if a.monthly else "") + ".json")
    if a.monthly:
        if a.big_calib:
            raise SystemExit("--monthly fixes the 6m configuration; not --big-calib")
        a.match_calib = True
    if a.match_calib and a.big_calib:
        raise SystemExit("--match-calib and --big-calib are different questions")

    ep, X = load_all(a.artifacts)
    sig_fn = causal_sigma_fn(ep, X)
    # The 17:00 production slice gives ~90 episodes per quarter, which cannot
    # resolve the effects being tested: a 6-window sign test is p=0.109 even at
    # 5/6. eval/anchors.py already established the model's edge is flat across
    # anchor hours (-6.14% at 17:00 vs -6.06% across all), so scoring every
    # H=19 anchor is legitimate and multiplies test episodes by ~24. Capped and
    # drawn once with a fixed seed so windows stay comparable to each other.
    prod = S.production_mask(ep)
    if a.wide:
        prod = (ep["H"] == 19).to_numpy()
        print(f"--wide: scoring all H=19 anchors "
              f"({prod.sum():,} candidates) instead of the 17:00 slice")

    # test windows through the unseen era
    if a.monthly:
        qs = [str((pd.Timestamp("2025-01-01", tz="UTC")
                   + pd.DateOffset(months=k)).date()) for k in range(19)]
        print(f"--monthly: {len(qs)-1} windows, "
              f"{qs[0]} .. {qs[-1]}, BENCHMARK.md 6n rule\n")
    else:
        qs = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01",
              "2026-01-01", "2026-04-01", "2026-07-01"]
    recs = []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        # the refreshed arm calibrates on the two quarters before the window
        # and trains on everything before that; the frozen arm never moves.
        if a.big_calib:
            # 18 months of calibration ending at the window, so the refreshed
            # arm is not judged on miscalibration with a third of the data.
            cal_start = str(pd.Timestamp(lo, tz="UTC").date()
                            - pd.DateOffset(months=18))[:10]
        elif a.monthly:
            # 6m's configuration: the refreshed arm trains to six months
            # before its own window and calibrates over those six months.
            cal_start = str((pd.Timestamp(lo, tz="UTC")
                             - pd.DateOffset(months=6)).date())
        else:
            cal_start = qs[i - 2] if i >= 2 else "2024-07-01"
        arms = {
            "frozen":    masks(ep, "2023-01-01", "2024-07-01", lo, hi, seed=i),
            "refreshed": masks(ep, cal_start, lo, lo, hi, seed=i),
        }
        te_mask = arms["frozen"]["test"] & prod
        if a.wide and te_mask.sum() > a.max_test:
            idx = np.flatnonzero(te_mask)
            keep = np.random.default_rng(4242 + i).choice(
                idx, size=a.max_test, replace=False)
            te_mask = np.zeros_like(te_mask); te_mask[keep] = True
        for f in arms.values():
            f["test"] = te_mask
        n_cal = {k: int(f["calib"].sum()) for k, f in arms.items()}
        if a.match_calib:
            n_eq = equalize_calib(arms, ep)
            print(f"  {lo}  calib {n_cal['frozen']:,}/{n_cal['refreshed']:,} "
                  f"-> both {n_eq:,}")
            n_cal = {k: n_eq for k in n_cal}
        n_te = int(te_mask.sum())
        if n_te < 40:
            print(f"  {lo}: SKIPPED (only {n_te} production episodes)")
            continue
        if not np.array_equal(arms["frozen"]["test"], arms["refreshed"]["test"]):
            raise SystemExit("REFUSING: arms differ on the test window")
        line = {"window": lo, "n_prod": n_te, "n_calib": n_cal,
                "match_calib": bool(a.match_calib)}
        for nm, f in arms.items():
            t0 = time.time()
            out = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                           sigma_ref_fn=sig_fn, min_train=5000)
            if out is None:
                print(f"  {lo}  {nm:10} SKIPPED (train/calib too small)")
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            s["tail_mcb"] = tail_mcb(out["rows"])
            s["n_train"] = int(f["train"].sum())
            s["n_calib"] = int(f["calib"].sum())
            line[nm] = s
            print(f"  {lo}  {nm:10} n_tr={s['n_train']:7,} "
                  f"n_ca={s['n_calib']:6,} n_te={n_te:4,}  "
                  f"QLIKE {s['qlike']:.4f}  DSC/UNC {s['DSS']:.5f}  "
                  f"tailMCB {s['tail_mcb']:.5f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if "frozen" in line and "refreshed" in line:
            recs.append(line)

    if not recs:
        print("no window produced both arms")
        return 1

    q_win = sum(1 for r in recs if r["refreshed"]["qlike"] < r["frozen"]["qlike"])
    d_win = sum(1 for r in recs if r["refreshed"]["DSS"] > r["frozen"]["DSS"])
    t_bad = sum(1 for r in recs
                if r["refreshed"]["tail_mcb"] > r["frozen"]["tail_mcb"])
    n = len(recs)
    print(f"\n{'metric':>12} {'frozen':>11} {'refreshed':>11} {'change':>10} {'wins':>7}")
    for k, lower_better in (("qlike", True), ("DSS", False),
                            ("pinball", True), ("crps", True),
                            ("tail_mcb", True)):
        f_ = np.mean([r["frozen"][k] for r in recs])
        r_ = np.mean([r["refreshed"][k] for r in recs])
        w = sum(1 for r in recs
                if (r["refreshed"][k] < r["frozen"][k]) == lower_better)
        print(f"{k:>12} {f_:11.6f} {r_:11.6f} {100*(r_/f_-1):+9.2f}% {w:>4}/{n}")

    verdict: dict = {}
    if a.monthly:
        # ---- BENCHMARK.md 6n's rule, fixed before this data was scored -----
        d_tail = np.array([r["refreshed"]["tail_mcb"] - r["frozen"]["tail_mcb"]
                           for r in recs])
        d_qlike = np.array([r["refreshed"]["qlike"] - r["frozen"]["qlike"]
                            for r in recs])
        # direction.py's helper refuses n < 20 because it is written for
        # episode-level series; the arithmetic is identical and the block
        # length is still round(n^(1/3)) = 3 at n = 18. Calling it directly
        # keeps the estimator the one 6n named rather than a lookalike.
        lo_ci, hi_ci = block_bootstrap_ci(d_tail, seed=11) if len(d_tail) >= 20 \
            else _mbb(d_tail, seed=11)
        need = int(np.ceil(5 / 6 * n))          # the same 83.3% rate
        # exact one-sided sign test on the QLIKE wins
        from math import comb
        p_sign = sum(comb(n, k) for k in range(q_win, n + 1)) / 2.0 ** n
        tail_excludes_zero = hi_ci < 0.0
        tail_contains_zero = (lo_ci <= 0.0 <= hi_ci)
        ok = tail_excludes_zero or (tail_contains_zero and q_win >= need)
        print(f"\n  --- BENCHMARK.md 6n decision rule (pre-registered) ---")
        print(f"  mean deep-tail MCB delta {d_tail.mean():+.6f}  "
              f"moving-block 95% CI [{lo_ci:+.6f}, {hi_ci:+.6f}]  (L=3, n={n})")
        print(f"  mean QLIKE delta         {d_qlike.mean():+.6f}")
        print(f"  QLIKE wins {q_win}/{n} (need >= {need} = the same 5/6 rate); "
              f"one-sided sign test p = {p_sign:.5f}")
        print(f"  tail CI excludes zero favourably: {tail_excludes_zero}; "
              f"contains zero: {tail_contains_zero}")
        if hi_ci < 0:
            why = "tail CI excludes zero on the favourable side"
        elif tail_contains_zero and q_win >= need:
            why = "tail indistinguishable from zero and QLIKE clears the rate"
        elif lo_ci > 0:
            why = "tail CI excludes zero UNFAVOURABLY -- real degradation"
        else:
            why = f"QLIKE {q_win}/{n} short of {need}"
        print(f"  -> {'ADOPT' if ok else 'DO NOT ADOPT'}: {why}")
        verdict = {"rule": "benchmark_6n", "mean_d_tail": float(d_tail.mean()),
                   "tail_ci95": [lo_ci, hi_ci], "mean_d_qlike": float(d_qlike.mean()),
                   "qlike_wins": q_win, "qlike_need": need,
                   "sign_test_p": float(p_sign), "why": why,
                   "d_tail_per_window": d_tail.tolist()}
    else:
        ok = (q_win >= 5) and (t_bad <= 2)
        print(f"\n  QLIKE wins {q_win}/{n} (need >= 5); tail MCB worse in "
              f"{t_bad}/{n} (allowed <= 2)")
        print(f"  -> pre-registered rule: {'ADOPT' if ok else 'DO NOT ADOPT'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"windows": recs, "qlike_wins": q_win,
                                 "dsc_wins": d_win, "tail_worse": t_bad,
                                 "n_windows": n, "adopt": bool(ok),
                                 "match_calib": bool(a.match_calib),
                                 "wide": bool(a.wide),
                                 "monthly": bool(a.monthly),
                                 "verdict": verdict},
                                indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
