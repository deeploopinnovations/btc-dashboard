"""
eval/oi_levels.py
=====================================================================
The level extraction for `P3-oi-max-strike`, written while the rule is fresh
and validated on the only thing that exists yet: two real option chains.

WHAT THIS IS AND WHAT IT DELIBERATELY IS NOT

`P3-oi-max-strike` is pre-registered and cannot be scored for months -- the
outcome needs a realised price path for each snapshot day, the corpus ends
2026-08-09, and per-strike open interest cannot be backfilled. So this file
implements the registered rule's LEVEL SELECTION and stops there. It contains
no scorer, reads no price, and returns no verdict.

Writing it now is the point. The rule was fixed before any outcome existed; the
code that implements it should be written and checked in the same state, and
its mechanics can be validated against real chains without any possibility of
seeing how the test lands. A bug in the placebo construction found in six
months is six months of episodes scored with the wrong control.

THE RULE, COPIED FROM THE REGISTRATION RATHER THAN RE-DECIDED HERE

  eligibility   an expiry qualifies on a day only if it holds >= 5% of that
                day's whole-chain open interest. Scale-free, fixed a priori,
                and NOT swept.
  level         the strike with the most open interest in that expiry, calls
                and puts POOLED.
  placebo       the listed strike in the SAME expiry closest to the level's
                distance from spot but on the OTHER side, whose open interest
                is at or below that expiry's median.
  horizon       H=168, one week, NOT the production 19-hour anchor.
  band          an episode is scored only if the MODEL'S own touch probability
                for the level is in [0.05, 0.95] -- a property of the forecast,
                not of the outcome.

THE HORIZON AND THE BAND ARE AMENDMENT 1, AND THIS FILE IS WHY

Implementing the rule and printing what it selects is what implementing a rule
is for. On the two chains that exist, ONE of eight qualifying max-OI levels
sits within three sigma of a 19-hour window; the rest are 3.0, 3.9, 4.4, 5.4,
7.6, 15.3 and 16.9 sigma away. Seven of eight episodes would have asked whether
price failed to travel fifteen standard deviations overnight, where "the market
did not break it" is true by arithmetic with no thesis involved. At H=168 sigma
is roughly 7.4% and six of the same eight come inside 2.6 sigma.

No outcome exists for either snapshot -- the corpus ends 2026-08-09 -- so the
amendment could not have been chosen to make a result come out. Once the first
episode is SCORED this rule is frozen (P3-oi-max-strike-v2).

WHY THE PLACEBO HAS TO BE DISTANCE-MATCHED AND ON THE OTHER SIDE

Open interest concentrates where the market thinks price will not go, which is
also exactly where a mis-calibrated tail would show up. A control at a
different distance would compare two different barrier questions. A control on
the same side at the same distance would frequently be the max-OI strike's
neighbour and inherit its open interest. The other side at matched distance is
the one that varies the thing under test -- concentration -- and holds the
barrier question fixed.

    python -m model.eval.oi_levels --selftest
    python -m model.eval.oi_levels          # describe the chains on disk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SHARE_FLOOR = 0.05          # P3-oi-max-strike, fixed a priori, never swept


def qualifying_expiries(df: pd.DataFrame, floor: float = SHARE_FLOOR) -> list:
    """Expiries holding at least `floor` of the WHOLE CHAIN's open interest.

    The denominator is the chain, not the expiry, which is what makes the
    threshold scale-free: it still means "a meaningful share of the market's
    positioning" after the market doubles. A fixed contract count would not.
    """
    tot = float(df["open_interest"].sum())
    if not (tot > 0):
        return []
    g = df.groupby("expiry")["open_interest"].sum() / tot
    return sorted(g[g >= floor].index)


def max_oi_level(df: pd.DataFrame, expiry) -> dict | None:
    """The strike holding the most open interest in one expiry, C and P POOLED.

    Pooling is the registered rule and it changes the answer: a strike can lead
    on calls alone and lose once puts are added. `harvest_oi`'s fixture is
    built so that failing to pool picks a different strike, and this function
    is the one that would fail it.
    """
    e = df[df["expiry"] == expiry]
    if e.empty:
        return None
    g = e.groupby("strike")["open_interest"].sum()
    g = g[g > 0]
    if g.empty:
        return None
    tot = float(g.sum())
    k = float(g.idxmax())
    return {"expiry": expiry, "strike": k, "oi": float(g.max()),
            "share_of_expiry": float(g.max() / tot), "expiry_oi": tot}


def placebo_level(df: pd.DataFrame, expiry, spot: float,
                  level: float) -> dict | None:
    """A distance-matched, LOW-open-interest strike on the other side of spot.

    Returns None rather than a substitute when the expiry lists no such strike
    -- a one-sided chain, or every candidate above the median. The caller drops
    that episode and counts it; silently relaxing the definition to keep an
    episode is how a control stops being one.
    """
    e = df[df["expiry"] == expiry]
    if e.empty or not (spot > 0):
        return None
    g = e.groupby("strike")["open_interest"].sum()
    g = g[g > 0]
    if len(g) < 2:
        return None
    med = float(np.median(g.values))
    d = abs(level - spot)
    above = level > spot
    # the other side of spot, strictly, and below the expiry's median OI
    cand = g[(g.index < spot) if above else (g.index > spot)]
    # AT OR BELOW the median, and the tie matters. The registration says
    # "below that expiry's median"; read strictly, a chain whose thin strikes
    # all carry the SAME open interest -- which is the common case, most
    # strikes sitting at a handful of contracts -- has its median equal to
    # that value and every candidate is excluded, so the control disappears
    # on exactly the chains it is most needed for. `<=` preserves what the
    # rule means, a strike that is not a concentration, and this note is
    # written while no outcome for any snapshot exists, so it cannot be an
    # interpretation chosen to make a result come out.
    cand = cand[cand <= med]
    if cand.empty:
        return None
    k = float(cand.index[int(np.argmin(np.abs(np.abs(cand.index - spot) - d)))])
    return {"expiry": expiry, "strike": k, "oi": float(cand.loc[k]),
            "median_oi": med, "target_distance": float(d),
            "distance": float(abs(k - spot)),
            "side": "dn" if above else "up"}


def episodes(df: pd.DataFrame, floor: float = SHARE_FLOOR) -> pd.DataFrame:
    """One row per qualifying expiry: the level, its placebo, and why not."""
    spot = float(df["underlying_price"].replace(0.0, np.nan).median())
    rows = []
    for x in qualifying_expiries(df, floor):
        lv = max_oi_level(df, x)
        if lv is None:
            continue
        pb = placebo_level(df, x, spot, lv["strike"])
        rows.append({
            "snapshot_ts": df["snapshot_ts"].iloc[0], "expiry": x, "spot": spot,
            "level": lv["strike"], "level_oi": lv["oi"],
            "share_of_expiry": lv["share_of_expiry"],
            "level_dist_pct": 100.0 * (lv["strike"] / spot - 1.0),
            "placebo": None if pb is None else pb["strike"],
            "placebo_oi": None if pb is None else pb["oi"],
            "placebo_dist_pct": (None if pb is None
                                 else 100.0 * (pb["strike"] / spot - 1.0)),
            "dropped": pb is None,
        })
    return pd.DataFrame(rows)


def selftest() -> int:
    ok = []

    # ---- synthetic: the properties the rule depends on --------------------
    def chain(strikes, ois, expiry="E1", spot=100.0, right="C"):
        return pd.DataFrame({
            "snapshot_ts": pd.Timestamp("2026-01-01", tz="UTC"),
            "expiry": expiry, "strike": strikes, "right": right,
            "open_interest": ois, "underlying_price": spot})

    # 1. POOLING. Calls lead at 120; puts make 110 the winner.
    c = chain([110, 120], [30, 50])
    p = chain([110, 120], [40, 5], right="P")
    ok.append(("calls and puts are pooled",
               max_oi_level(pd.concat([c, p]), "E1")["strike"] == 110.0))

    # 2-3. The eligibility floor is a share of the WHOLE CHAIN.
    # 900/950 and 50/950 would put SMALL at 5.3% -- ABOVE the floor. The
    # first version of this fixture asserted it was below and failed, which is
    # the fixture doing its job: a threshold on a share needs its denominator
    # computed, not eyeballed.
    big = chain([100], [1000], expiry="BIG")
    small = chain([100], [40], expiry="SMALL")
    both = pd.concat([big, small])
    q = qualifying_expiries(both, 0.05)
    ok.append(("an expiry above the floor qualifies", "BIG" in q))
    ok.append(("one below it does not", "SMALL" not in q))

    # 4. The placebo is on the OTHER side of spot.
    d = chain([80, 90, 110, 120], [5, 5, 100, 5], spot=100.0)
    pb = placebo_level(d, "E1", 100.0, 110.0)
    ok.append(("placebo crosses spot", pb is not None and pb["strike"] < 100.0))

    # 5. It is distance-matched: level at +10 -> 90, not 80.
    ok.append(("placebo matches the distance", pb["strike"] == 90.0))

    # 6. It must be BELOW the expiry's median open interest.
    ok.append(("placebo open interest is at or below the median",
               pb["oi"] <= pb["median_oi"]))

    # 7. A one-sided chain yields no placebo and is DROPPED, not substituted.
    one = chain([110, 120, 130], [100, 5, 5], spot=100.0)
    ok.append(("no strike on the other side -> None",
               placebo_level(one, "E1", 100.0, 110.0) is None))

    # 8. When the ONLY strike on the other side is itself a concentration,
    #    the episode is dropped rather than controlled by a second
    #    concentration. 90 carries 100 against a median of 5.
    conc = chain([90, 110, 120], [100, 5, 5], spot=100.0)
    ok.append(("the only other-side strike is a concentration -> None",
               placebo_level(conc, "E1", 100.0, 110.0) is None))

    # 9. And the tie case the `<=` note is about: thin strikes all equal, so
    #    the median IS that value. Strictly-below would drop every episode.
    tied = chain([80, 90, 110], [7, 7, 200], spot=100.0)
    ok.append(("tied thin strikes still yield a control",
               placebo_level(tied, "E1", 100.0, 110.0)["strike"] == 90.0))

    # ---- the real chains on disk: MECHANICS ONLY -------------------------
    # No outcome exists for these dates (the corpus ends 2026-08-09), so
    # nothing here can see how the test lands. R81: a fixture checks logic, a
    # real file checks convention, and the conventions this rule depends on --
    # that `expiry` groups, that `strike` is numeric, that both sides are
    # listed -- are exactly what a synthetic frame would assume rather than
    # test.
    files = sorted(Path("data/oi").glob("*.parquet"))
    if not files:
        ok.append(("real chains present to check conventions against", False))
    else:
        for f in files:
            df = pd.read_parquet(f)
            ep = episodes(df)
            ok.append((f"{f.name}: at least one expiry qualifies", len(ep) > 0))
            if len(ep):
                ok.append((f"{f.name}: every level is a real listed strike",
                           bool(ep["level"].isin(df["strike"]).all())))
                ok.append((f"{f.name}: no placebo shares its level's side",
                           bool(((ep["level_dist_pct"] > 0)
                                 != (ep["placebo_dist_pct"] > 0))
                                [~ep["dropped"]].all())))
                ok.append((f"{f.name}: placebo carries less OI than the level",
                           bool((ep["placebo_oi"] < ep["level_oi"])
                                [~ep["dropped"]].all())))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P3-oi-max-strike level selection")
    ap.add_argument("--oi", type=Path, default=Path("data/oi"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    files = sorted(a.oi.glob("*.parquet"))
    if not files:
        print(f"no chains in {a.oi}")
        return 1
    print("P3-oi-max-strike level selection. NO SCORING: the outcome needs a "
          "realised\nprice path and the corpus ends before these dates. "
          f"Stopping rule is 200\ndistinct (expiry, level) episodes.\n")
    seen = set()
    for f in files:
        ep = episodes(pd.read_parquet(f))
        print(f"{f.name}  {len(ep)} qualifying expiries, "
              f"{int(ep['dropped'].sum())} without a placebo")
        for r in ep.itertuples():
            seen.add((str(r.expiry), float(r.level)))
            pb = "-" if r.dropped else f"{r.placebo:,.0f} @ {r.placebo_dist_pct:+.1f}%"
            print(f"   {pd.Timestamp(r.expiry):%Y-%m-%d}  level {r.level:>9,.0f} "
                  f"@ {r.level_dist_pct:+6.1f}%  "
                  f"({100*r.share_of_expiry:4.1f}% of expiry)   placebo {pb}")
    print(f"\ndistinct (expiry, level) episodes so far: {len(seen)} / 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
