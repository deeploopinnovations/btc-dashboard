"""
eval/dst_shift.py
=====================================================================
P2-dst-shift: does the H=1 error peak sit exactly ONE HOUR EARLIER in UTC
during US summer than during US winter?

WHY A POINT PREDICTION ABOUT AN INTEGER

`P2-dst-alignment-result` measured Eastern-vs-UTC concentration, got z = +2.04
against a placebo family whose nearest members were near-duplicates of the
truth, and could not conclude. Worse, its pre-registered control compared the
Eastern clock against fixed hour offsets -- a relabelling of the bins, so every
statistic was invariant by construction and the control could not fail (R50).

This tests the MECHANISM instead of a summary statistic. US macro releases and
the equity open are scheduled in Eastern local time, so in UTC they land one
hour EARLIER in summer than in winter. Split the episodes on the real DST state
and cross-correlate the two lift tables: the argmax must be lag +1.

    lag +1  the US schedule drives it  (summer advanced one hour to match winter)
    lag  0  a UTC-anchored effect drives it -- perpetual funding settles at
            00:00, 08:00 and 16:00 UTC and does not move with daylight saving
    anything else  neither story survives

The two live hypotheses make DIFFERENT integer predictions, so the data chooses
between them rather than one being tested against noise.

CONTROLS THAT CAN ACTUALLY RETURN A DIFFERENT NUMBER

  * lag 0 is a real rival, not a null of convenience;
  * H=24, where the footprint is absent (chi-square 3.0, p = 1.000) -- a clean
    +1 there would mean the estimator manufactures its answer;
  * a placebo split on odd-vs-even ISO week, which has a similar marginal and
    no relation to daylight saving.

Significance is a permutation of the SUMMER/WINTER LABEL within each calendar
year, which destroys the split while preserving the hour distribution, the
yearly composition and the worst-5% count exactly.

    python -m model.eval.dst_shift
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dst_alignment import TEACHER, WORST_Q, is_summer, load             # noqa: E402

N_PERM = 2000
PREDICTED_LAG = 1


def lift(hour: np.ndarray, worst: np.ndarray) -> np.ndarray:
    obs = np.bincount(hour[worst], minlength=24).astype(np.float64)
    allc = np.bincount(hour, minlength=24).astype(np.float64)
    exp = allc * (obs.sum() / max(allc.sum(), 1.0))
    return np.divide(obs, np.maximum(exp, 1e-12))


def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular cross-correlation over integer lags 0..23, mean-centred.

    Centred because both lift vectors average 1 by construction, and an
    uncentred product would be dominated by that constant at every lag --
    which would make the profile flat and the argmax meaningless.
    """
    a_, b_ = a - a.mean(), b - b.mean()
    return np.array([float(np.dot(np.roll(a_, k), b_)) for k in range(24)])


def analyse(ts, q, split, label: str, rng, n_perm=N_PERM) -> dict:
    utc_h = ((ts // 3600) % 24).astype(int)
    worst = q >= np.quantile(q, WORST_Q)
    year = np.array([datetime.fromtimestamp(int(t), tz=timezone.utc).year
                     for t in ts])
    la = lift(utc_h[split], worst[split])
    lb = lift(utc_h[~split], worst[~split])
    c = xcorr(la, lb)
    arg = int(np.argmax(c))

    # permute the SPLIT LABEL within each year: the hour distribution, the
    # yearly composition and the worst-5% count are all preserved exactly.
    hits = 0
    for _ in range(n_perm):
        p = split.copy()
        for y in np.unique(year):
            m = year == y
            p[m] = rng.permutation(split[m])
        cp = xcorr(lift(utc_h[p], worst[p]), lift(utc_h[~p], worst[~p]))
        if cp[PREDICTED_LAG] >= c[PREDICTED_LAG]:
            hits += 1
    return {"label": label, "n_a": int(split.sum()), "n_b": int((~split).sum()),
            "argmax_lag": arg, "corr": c.tolist(),
            "corr_at_predicted": float(c[PREDICTED_LAG]),
            "corr_at_zero": float(c[0]),
            "p_perm": float((hits + 1) / (n_perm + 1)),
            "lift_a": la.tolist(), "lift_b": lb.tolist()}


def show(r: dict) -> None:
    c = np.array(r["corr"])
    print(f"  {r['label']}   n = {r['n_a']:,} / {r['n_b']:,}")
    print(f"    argmax lag {r['argmax_lag']:>3}   "
          f"corr@+1 {r['corr_at_predicted']:+.4f}   "
          f"corr@0 {r['corr_at_zero']:+.4f}   "
          f"permutation p (lag +1) {r['p_perm']:.4f}")
    prof = "  ".join(f"{k}:{c[k]:+.3f}" for k in (23, 0, 1, 2, 3))
    print(f"    profile near zero (lag:corr)   {prof}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-dst-shift")
    ap.add_argument("--oof", type=Path,
                    default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/dst_shift.json"))
    a = ap.parse_args(argv)
    z = np.load(a.oof)
    rng = np.random.default_rng(20260902)

    print(f"P2-dst-shift   teacher={TEACHER}   {a.perm} permutations")
    print(f"PREDICTION: argmax lag = +{PREDICTED_LAG} (US schedule).  "
          f"RIVAL: lag 0 (UTC-anchored).\n")

    out = {"n_perm": a.perm, "predicted_lag": PREDICTED_LAG, "runs": {}}
    for H, tag in ((1, "PRIMARY"), (24, "CONTROL, footprint absent")):
        ts, q = load(z, H)
        if ts is None:
            continue
        print("=" * 88)
        print(f"H = {H}h   {len(q):,} episodes   [{tag}]")
        print("=" * 88)
        r = analyse(ts, q, is_summer(ts), "US summer vs US winter", rng, a.perm)
        show(r)
        out["runs"][f"H{H}_dst"] = r

        # placebo split: odd vs even ISO week, similar marginal, unrelated
        wk = np.array([datetime.fromtimestamp(int(t), tz=timezone.utc)
                       .isocalendar().week for t in ts])
        rp = analyse(ts, q, (wk % 2 == 0), "PLACEBO: even vs odd ISO week",
                     rng, a.perm)
        show(rp)
        out["runs"][f"H{H}_week"] = rp
        print()

    prim = out["runs"].get("H1_dst", {})
    plac = out["runs"].get("H1_week", {})
    ctrl = out["runs"].get("H24_dst", {})
    passed = (prim.get("argmax_lag") == PREDICTED_LAG
              and prim.get("p_perm", 1.0) < 0.05
              and plac.get("argmax_lag") != PREDICTED_LAG
              and ctrl.get("argmax_lag") != PREDICTED_LAG)
    print("=" * 88)
    if prim.get("argmax_lag") == 0:
        print("VERDICT: argmax 0 -- the footprint is UTC-ANCHORED, not "
              "US-scheduled. The event\nstory is wrong and perpetual funding "
              "(settles 00:00/08:00/16:00 UTC) is the lead.")
    elif passed:
        print("VERDICT: argmax +1 with the controls clean -- the H=1 footprint "
              "is aligned to the\nUS EASTERN schedule.")
    else:
        print(f"VERDICT: NOT ESTABLISHED. primary argmax "
              f"{prim.get('argmax_lag')}, p {prim.get('p_perm')}, "
              f"placebo argmax {plac.get('argmax_lag')}, "
              f"H=24 argmax {ctrl.get('argmax_lag')}.")
    out["passed"] = bool(passed)
    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
