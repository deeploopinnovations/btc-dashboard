"""
eval/tail_clustering.py
=====================================================================
P2-tail-clustering: is LABELLING TRACTABLE? How many distinct days hold
the worst-case episodes?

WHY THIS EXISTS

`P2-dataset-audit` established that error is extremely concentrated: the worst
5% of episodes carry 52.8% of NOCTUA's pooled QLIKE at H=1 and 52.0% at H=24.
That finding is the entire case for event labelling -- a label is only worth a
column if it speaks to the episodes that carry the loss.

But concentration in the LOSS distribution does not imply concentration in
CALENDAR TIME, and only the second one decides whether labelling is affordable.
If the worst 5% of hours fall on thirty days a year, a human or a retrieval
pipeline can label thirty days. If they fall on three hundred, "label the tail"
means "label everything", and the cost argument for labelling collapses into the
cost argument against it.

WHAT THIS CAN AND CANNOT SEE

The model-error artifacts are gone (see DATA_LOSS_2026-09-12.md), so this CANNOT
rank episodes by NOCTUA's QLIKE. It ranks them by `rv5`, realized variance, which
survives in data/noctua_history.parquet for 9,600 hours. That is a PROXY, and its
direction of bias is knowable: `metrics()` already defines the spike bucket by RV
quantile, and the audit found error concentration and RV concentration to coincide
for every teacher but garch_t. A proxy that the project already uses to define
"spike" is the right proxy to use here, but the result is about RV, not loss, and
is labelled that way.

It also sees 400 days, not six years. Clustering measured on one year cannot be
assumed stationary -- 2025-07 to 2026-08 contains whatever regimes it contains.

THE CONTROL THAT MAKES THE NUMBER MEAN ANYTHING

"The worst 5% fall on N days" is uninterpretable alone. Under random scattering
480 episodes across 400 days would touch nearly every day, so almost any N looks
clustered. The null here is a PERMUTATION of which hours are extreme, preserving
the count; the statistic is the distinct-day count against that null. A second
null preserves the DIURNAL pattern (permute within hour-of-day), because vol has
a known time-of-day shape and clustering that is only diurnal is not an event.

    python -m model.eval.tail_clustering
    python -m model.eval.tail_clustering --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIST = Path("data/noctua_history.parquet")
OUT = Path("model/artifacts/tail_clustering.json")
TAIL_Q = 0.95
N_PERM = 2000
SEED = 11

# ---------------------------------------------------------------- bands
# Fixed BEFORE the number exists. The read is on the distinct-day count for
# the worst 5% of hourly episodes, expressed as a FRACTION of days available,
# so it does not depend on the window length that happens to have survived.
BANDS = (
    (0.00, 0.15, "A", "HIGHLY CLUSTERED. Labelling is tractable: a few dozen "
                      "days per year carry the tail. Build the labelled set."),
    (0.15, 0.38, "B", "MODERATELY CLUSTERED. Tractable only if the label is "
                      "EVENT-specific, not day-specific -- a day flag would "
                      "fire on too many ordinary days to be informative."),
    (0.38, 0.75, "C", "WEAKLY CLUSTERED. Labelling the tail means labelling "
                      "most of the calendar. The 'few big events' premise is "
                      "wrong and the cost case for labelling is gone."),
    (0.75, 1.01, "D", "NOT CLUSTERED. Every day touches the tail. Labelling "
                      "cannot target anything and must be rejected."),
)


def band_for(frac: float) -> tuple[str, str]:
    for lo, hi, tag, text in BANDS:
        if lo <= frac < hi:
            return tag, text
    raise ValueError(frac)


def distinct_days(day_idx: np.ndarray) -> int:
    return int(np.unique(day_idx).size)


def longest_run(mask: np.ndarray) -> int:
    """Longest run of consecutive extreme hours -- one shock or many?"""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def analyse(rv: np.ndarray, day_idx: np.ndarray, hod: np.ndarray,
            q: float = TAIL_Q, n_perm: int = N_PERM,
            seed: int = SEED) -> dict:
    n = rv.size
    ok = np.isfinite(rv)
    thr = np.quantile(rv[ok], q)
    tail = ok & (rv >= thr)
    k = int(tail.sum())
    nd_all = distinct_days(day_idx[ok])
    nd = distinct_days(day_idx[tail])

    rng = np.random.default_rng(seed)
    # null 1: which hours are extreme is arbitrary
    free = np.empty(n_perm, np.int64)
    idx = np.flatnonzero(ok)
    for i in range(n_perm):
        free[i] = distinct_days(day_idx[rng.choice(idx, k, replace=False)])
    # null 2: preserve the diurnal shape -- permute extreme labels within
    # each hour-of-day bucket, so time-of-day structure is held fixed
    diur = np.empty(n_perm, np.int64)
    buckets = [np.flatnonzero(ok & (hod == h)) for h in range(24)]
    per_h = [int(((hod == h) & tail).sum()) for h in range(24)]
    for i in range(n_perm):
        take = [rng.choice(b, c, replace=False)
                for b, c in zip(buckets, per_h) if c]
        diur[i] = distinct_days(day_idx[np.concatenate(take)])

    mass = float(np.nansum(rv[tail]) / np.nansum(rv[ok]))
    # how much of the tail sits on the busiest days
    s = pd.Series(1, index=day_idx[tail]).groupby(level=0).size().sort_values()[::-1]
    top10 = float(s.head(10).sum() / k)
    return {
        "n_episodes": int(ok.sum()),
        "days_available": nd_all,
        "tail_q": q,
        "tail_n": k,
        "tail_days": nd,
        "tail_day_fraction": nd / nd_all,
        "rv_mass_in_tail": mass,
        "tail_share_on_busiest_10_days": top10,
        "longest_consecutive_run_h": longest_run(tail),
        "null_free_mean_days": float(free.mean()),
        "null_free_p05_days": float(np.quantile(free, 0.05)),
        "null_diurnal_mean_days": float(diur.mean()),
        "null_diurnal_p05_days": float(np.quantile(diur, 0.05)),
        "p_free": float((free <= nd).mean()),
        "p_diurnal": float((diur <= nd).mean()),
        "n_perm": n_perm,
    }


def selftest() -> int:
    """The statistic must separate a clustered world from a scattered one."""
    rng = np.random.default_rng(3)
    D, Hh = 400, 24
    day_idx = np.repeat(np.arange(D), Hh)
    hod = np.tile(np.arange(Hh), D)
    checks = []

    # world 1: 20 shock days carry every extreme hour
    rv = rng.lognormal(-9, 0.3, D * Hh)
    shock = rng.choice(D, 20, replace=False)
    rv[np.isin(day_idx, shock)] *= 60.0
    a = analyse(rv, day_idx, hod, n_perm=200)
    checks.append(("clustered-world-reads-A-or-B",
                   band_for(a["tail_day_fraction"])[0] in ("A", "B"),
                   f"{a['tail_days']} days = {a['tail_day_fraction']:.3f}, "
                   f"band {band_for(a['tail_day_fraction'])[0]}"))
    checks.append(("clustered-world-beats-free-null", a["p_free"] < 0.01,
                   f"p={a['p_free']:.4f}, {a['tail_days']} days vs null "
                   f"{a['null_free_mean_days']:.1f}"))

    # world 2: extremes scattered at random, no day structure at all
    rv2 = rng.lognormal(-9, 1.1, D * Hh)
    b = analyse(rv2, day_idx, hod, n_perm=200)
    checks.append(("scattered-world-reads-C-or-D",
                   band_for(b["tail_day_fraction"])[0] in ("C", "D"),
                   f"{b['tail_days']} days = {b['tail_day_fraction']:.3f}, "
                   f"band {band_for(b['tail_day_fraction'])[0]}"))
    checks.append(("scattered-world-does-NOT-beat-null", b["p_free"] > 0.05,
                   f"p={b['p_free']:.4f} -- the statistic must not cry "
                   f"clustering at noise"))

    # world 3: purely DIURNAL -- extremes only at one hour, every day.
    # the free null must be beaten (concentrated in time-of-day) while the
    # diurnal null must NOT be, which is the whole point of having two.
    rv3 = rng.lognormal(-9, 0.25, D * Hh)
    rv3[hod == 14] *= 50.0
    c = analyse(rv3, day_idx, hod, n_perm=200)
    checks.append(("diurnal-world-is-not-mistaken-for-events",
                   c["p_diurnal"] > 0.05,
                   f"p_diurnal={c['p_diurnal']:.4f} (free={c['p_free']:.4f}) "
                   f"-- a time-of-day shape is not an event"))

    # bands must tile [0,1] without a gap or an overlap
    edges_ok = all(BANDS[i][1] == BANDS[i + 1][0] for i in range(len(BANDS) - 1))
    checks.append(("bands-tile-the-range", edges_ok and BANDS[0][0] == 0.0,
                   "contiguous from 0"))

    print("tail_clustering selftest")
    bad = 0
    for name, ok, detail in checks:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-tail-clustering")
    ap.add_argument("--hist", type=Path, default=HIST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    print(__doc__.split("    python")[0].strip()[:0] or "", end="")
    print("P2-tail-clustering   bands fixed before the number exists:")
    for lo, hi, tag, text in BANDS:
        print(f"  {tag}  day fraction [{lo:.2f}, {hi:.2f}):  {text}")
    print("\nPREDICTION, recorded before running: band B. Volatility clusters")
    print("within days -- that is what GARCH and HAR are -- but 400 days of BTC")
    print("contains many separate shocks, so I expect well above the null and")
    print("well short of a few dozen days.\n")

    if not a.hist.exists():
        print(f"MISSING: {a.hist}", file=sys.stderr)
        return 2
    d = pd.read_parquet(a.hist).sort_values("hour_ts").reset_index(drop=True)
    ts = pd.to_datetime(d["hour_ts"], unit="s", utc=True)
    rv = np.asarray(d["rv5"], np.float64)
    day_idx = np.asarray((ts.dt.floor("D") - ts.dt.floor("D").min()).dt.days)
    hod = np.asarray(ts.dt.hour)
    print(f"{a.hist}: {len(d):,} hours, {ts.min().date()} -> {ts.max().date()}")

    res = {}
    for q in (0.95, 0.99):
        r = analyse(rv, day_idx, hod, q=q)
        tag, text = band_for(r["tail_day_fraction"])
        r["band"], r["band_text"] = tag, text
        res[str(q)] = r
        print(f"\n{'=' * 78}\nworst {100*(1-q):.0f}% of hourly episodes by rv5 "
              f"({r['tail_n']:,} of {r['n_episodes']:,})")
        print(f"{'=' * 78}")
        print(f"  distinct days touched     {r['tail_days']:,} of "
              f"{r['days_available']:,}  = {r['tail_day_fraction']:.3f}")
        print(f"  band                      {tag}  {text}")
        print(f"  rv mass inside the tail   {100*r['rv_mass_in_tail']:.1f}%")
        print(f"  tail on busiest 10 days   {100*r['tail_share_on_busiest_10_days']:.1f}%")
        print(f"  longest consecutive run   {r['longest_consecutive_run_h']}h")
        print(f"  null, free permutation    {r['null_free_mean_days']:.1f} days "
              f"(5th pct {r['null_free_p05_days']:.0f})   p = {r['p_free']:.4f}")
        print(f"  null, diurnal-preserving  {r['null_diurnal_mean_days']:.1f} days "
              f"(5th pct {r['null_diurnal_p05_days']:.0f})   p = {r['p_diurnal']:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
