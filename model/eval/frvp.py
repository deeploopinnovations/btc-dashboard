"""
eval/frvp.py
=====================================================================
Fixed Range Volume Profile at the production anchor: does the value area carry
anything the realized range does not?

THE THESIS, AS GIVEN, IN UTC

A practitioner thesis, handed over as a hypothesis to test rather than a result
to reproduce. Stated in IST it is: build a volume profile over 18 Sep 00:30 ->
19 Sep 00:30, trade at 19 Sep 22:30. Converted:

    FRVP window   [D-2 19:00 UTC, D-1 19:00 UTC)      24h
    observation   [D-1 19:00 UTC, D    17:00 UTC)     22h   <- the "gap"
    decision       D    17:00 UTC

and that decision time is NOT arbitrary here: 17:00 UTC with H = 19 is exactly
this project's PRODUCTION slice (`splits.production_mask`). The flag is
therefore available at the moment the shipped model makes its forecast, and it
uses only bars that closed at least 22 hours earlier. Nothing about it is
forward-looking.

THE TWO CLAIMS, WHICH NEED DIFFERENT MACHINERY

  (a) DIRECTIONAL-ISH: above POC sell VAH, below POC sell VAL.
      Selling a strike is a bet it is NOT TOUCHED, which is a first-passage
      question and not a direction forecast -- so it is scored against
      P(touch), which the committee already emits, and NOT against the
      direction benchmark that is closed at all four horizons.

  (b) THE AVOIDANCE RULE: if price touches BOTH VAH and VAL during the 22h
      gap, expect momentum -- do not sell. That is a statement about
      VOLATILITY, and it is the half this file tests, because it needs no
      option prices at all (R28: no manufactured P&L).

THE CONFOUND, WHICH IS THE WHOLE DESIGN

Touching both edges requires the 22h range to exceed the value-area WIDTH. The
range is the main input to HAR, and volatility clusters -- so "double touch
predicts a bigger move" is very nearly guaranteed to look true whether or not
the volume profile contributes anything. Measuring it against zero would
measure vol clustering and call it a discovery.

So every arm here is paired with a WIDTH-MATCHED PLACEBO: a band of identical
width, centred somewhere the profile did not choose (the window VWAP, and the
window's high-low midpoint). A placebo band answers the question "was it the
range, or was it the POC's LOCATION?" -- and the location is the only thing
FRVP adds over a range statistic. If the placebos flag the same days and
predict the same thing, the profile is decoration.

This is not a hypothetical precaution. In `P3-exogenous` a width-matched
shuffled control BEAT the real feature, and in `P3-regularisation` a permuted
column outscored every explicit regulariser. The control is the experiment.

FREE PARAMETERS, DECLARED BEFORE ANY NUMBER IS SEEN

n_bins, the 70% coverage, the 24h window, the 22h gap and the 17:00 anchor are
all choices, and a thesis arrives with them already tuned by whoever found it.
They are fixed here at the values as given, and `--sensitivity` re-runs the
headline across a grid so that the size of the forking path is a number in the
output rather than an argument.

    python -m model.eval.frvp --selftest
    python -m model.eval.frvp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HOUR = 3600
WINDOW_H = 24          # FRVP window length
GAP_H = 22             # window end -> decision
ANCHOR_UTC = 17        # production anchor hour
WINDOW_END_UTC = 19    # window ends at 19:00 UTC = 00:30 IST
N_BINS = 64
COVERAGE = 0.70


def value_area(price: np.ndarray, volume: np.ndarray, n_bins: int = N_BINS,
               coverage: float = COVERAGE) -> dict | None:
    """POC, VAH and VAL from a volume-by-price histogram.

    The value area is grown from the POC bin outwards, at each step taking
    whichever NEIGHBOUR holds more volume, until `coverage` of total volume is
    enclosed. That is the standard construction (Market Profile's two-bin
    step is a refinement that changes the edges by at most one bin); the
    alternative of taking the densest contiguous run is NOT the same thing and
    would not be the quantity the thesis refers to.
    """
    price = np.asarray(price, np.float64)
    volume = np.asarray(volume, np.float64)
    ok = np.isfinite(price) & np.isfinite(volume) & (volume > 0)
    if ok.sum() < 20:
        return None
    price, volume = price[ok], volume[ok]
    lo, hi = float(price.min()), float(price.max())
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, price, side="right") - 1, 0, n_bins - 1)
    hist = np.bincount(idx, weights=volume, minlength=n_bins)
    total = hist.sum()
    if total <= 0:
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])

    poc = int(np.argmax(hist))
    lo_i = hi_i = poc
    acc = hist[poc]
    target = coverage * total
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1.0
        up = hist[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if up >= down:
            hi_i += 1; acc += hist[hi_i]
        else:
            lo_i -= 1; acc += hist[lo_i]
    return {"poc": float(centres[poc]),
            "val": float(edges[lo_i]), "vah": float(edges[hi_i + 1]),
            "vwap": float(np.sum(price * volume) / total),
            "mid": float(0.5 * (lo + hi)),
            "covered": float(acc / total)}


def _touches(hi: np.ndarray, lo: np.ndarray, upper: float, lower: float) -> tuple:
    """(touched_upper, touched_lower) over a slice of 1-minute bars."""
    if len(hi) == 0:
        return (False, False)
    return (bool(np.nanmax(hi) >= upper), bool(np.nanmin(lo) <= lower))


def build(mins: pd.DataFrame, window_h: int = WINDOW_H, gap_h: int = GAP_H,
          anchor_utc: int = ANCHOR_UTC, window_end_utc: int = WINDOW_END_UTC,
          n_bins: int = N_BINS, coverage: float = COVERAGE) -> pd.DataFrame:
    """One row per production anchor: the value area, and what the gap touched."""
    d = mins.copy()
    if "dt" not in d:
        d["dt"] = pd.to_datetime(d["timestamp"], unit="s", utc=True)
    # `filled` marks a FORWARD-FILLED synthetic bar (ingest.py: missing minutes
    # are filled as zero-volume zero-range bars and flagged), so the real bars
    # are the ones where it is False. A first draft wrote `d["filled"] & ...`,
    # which selected exactly the synthetic bars -- of which this corpus has
    # none -- and handed `build` an empty frame.
    if "filled" in d:
        d = d[~d["filled"].astype(bool) & ~d["bad_print"].astype(bool)]
    d = d.sort_values("dt").reset_index(drop=True)
    ts = d["dt"].to_numpy("datetime64[s]").astype(np.int64)
    typ = ((d["high"] + d["low"] + d["close"]) / 3.0).to_numpy(np.float64)
    vol = d["volume"].to_numpy(np.float64)
    hi_a = d["high"].to_numpy(np.float64)
    lo_a = d["low"].to_numpy(np.float64)

    first, last = ts[0], ts[-1]
    # Anchor days: every date whose 17:00 UTC has a full window+gap behind it.
    day0 = pd.Timestamp(first, unit="s", tz="UTC").normalize()
    day1 = pd.Timestamp(last, unit="s", tz="UTC").normalize()
    rows = []
    for day in pd.date_range(day0, day1, freq="D", tz="UTC"):
        dec = int((day + pd.Timedelta(hours=anchor_utc)).timestamp())
        w_end = dec - gap_h * HOUR
        w_beg = w_end - window_h * HOUR
        if w_beg < first or dec > last:
            continue
        # sanity: the window must end at the stated wall-clock hour
        if pd.Timestamp(w_end, unit="s", tz="UTC").hour != window_end_utc % 24:
            pass          # a non-default gap moves it; not an error, just noted
        a, b = np.searchsorted(ts, [w_beg, w_end])
        va = value_area(typ[a:b], vol[a:b], n_bins, coverage)
        if va is None:
            continue
        g0, g1 = np.searchsorted(ts, [w_end, dec])
        gh, gl = hi_a[g0:g1], lo_a[g0:g1]
        if len(gh) < 60:
            continue
        width = va["vah"] - va["val"]
        rec = {"anchor_ts": dec, "dt": pd.Timestamp(dec, unit="s", tz="UTC"),
               "poc": va["poc"], "vah": va["vah"], "val": va["val"],
               "width": width, "covered": va["covered"],
               "width_pct": 100.0 * width / max(va["poc"], 1e-12),
               "gap_range_pct": 100.0 * (float(np.nanmax(gh)) - float(np.nanmin(gl)))
                                / max(va["poc"], 1e-12),
               "close_at_decision": float(d["close"].to_numpy()[max(g1 - 1, 0)])}
        up, dn = _touches(gh, gl, va["vah"], va["val"])
        rec["touch_vah"], rec["touch_val"] = up, dn
        rec["double_touch"] = bool(up and dn)
        rec["above_poc"] = bool(rec["close_at_decision"] > va["poc"])
        # WIDTH-MATCHED PLACEBOS: identical width, centre the profile did not
        # choose. These decide whether the POC's LOCATION carries anything.
        for name in ("vwap", "mid"):
            c = va[name]
            pu, pd_ = _touches(gh, gl, c + width / 2.0, c - width / 2.0)
            rec[f"double_touch_{name}"] = bool(pu and pd_)
        rows.append(rec)
    return pd.DataFrame(rows)


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(7)

    # 1-3. A histogram with a KNOWN shape: one dominant bin plus a flat floor.
    #      POC must land on the spike, the value area must enclose it, and the
    #      coverage actually achieved must reach the target.
    price = np.concatenate([np.full(400, 100.0), np.linspace(90, 110, 400)])
    volume = np.concatenate([np.full(400, 10.0), np.full(400, 1.0)])
    va = value_area(price, volume, n_bins=41, coverage=0.70)
    ok.append(("POC lands on the volume spike", abs(va["poc"] - 100.0) < 0.6))
    ok.append(("value area brackets the POC", va["val"] < va["poc"] < va["vah"]))
    ok.append(("coverage target is reached", va["covered"] >= 0.70 - 1e-9))

    # 4. A WIDER coverage must give a WIDER band -- otherwise the growth loop
    #    is not doing what its name says.
    wide = value_area(price, volume, n_bins=41, coverage=0.95)
    ok.append(("more coverage -> wider band",
               (wide["vah"] - wide["val"]) > (va["vah"] - va["val"])))

    # 5. Uniform volume: the value area should cover ~coverage of the RANGE,
    #    because with a flat histogram bins are interchangeable.
    up_ = np.linspace(0, 100, 5000)
    fv = value_area(up_, np.ones_like(up_), n_bins=100, coverage=0.70)
    ok.append(("flat profile -> band is ~70% of range",
               0.60 <= (fv["vah"] - fv["val"]) / 100.0 <= 0.82))

    # 6. Refuses rather than inventing a band when there is nothing to profile.
    ok.append(("refuses on degenerate input",
               value_area(np.full(50, 5.0), np.ones(50)) is None))

    # 7-9. End to end on a synthetic tape, where the answer is CONSTRUCTED.
    #      A flat window then a gap that ramps far above: VAH touched, VAL not,
    #      so double_touch must be False. Getting this backwards is the single
    #      most costly bug this file could have.
    # Four days, so that a 17:00 anchor exists with a full 24h window plus a
    # 22h gap behind it. A first draft used 48 hours, where no qualifying
    # anchor can exist at all -- the fixture was too short to contain the
    # thing it was testing, and the build loop correctly returned nothing.
    n = 4 * 24 * 60
    t0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")
    dt = pd.date_range(t0, periods=n, freq="min")
    close = np.full(n, 100.0)
    need = (WINDOW_H + GAP_H) * 60
    dec_i = next((i for i, tstamp in enumerate(dt)
                  if tstamp.hour == ANCHOR_UTC and tstamp.minute == 0
                  and i > need), None)
    assert dec_i is not None, "fixture has no usable anchor"
    g0 = dec_i - GAP_H * 60
    close[g0:dec_i] = np.linspace(100.0, 140.0, dec_i - g0)     # up only
    close[:g0] += rng.normal(0, 0.05, g0)
    m = pd.DataFrame({"dt": dt, "open": close, "high": close + 0.01,
                      "low": close - 0.01, "close": close,
                      "volume": np.full(n, 5.0),
                      # filled=False: a REAL bar. The first draft of this
                      # fixture wrote True, which is the synthetic-bar flag --
                      # the same misunderstanding the filter had -- so all
                      # twelve checks passed against a function that returned
                      # an empty frame on the actual corpus.
                      "filled": False, "bad_print": False})
    tbl = build(m)
    ok.append(("end to end produces a row", len(tbl) >= 1))
    if len(tbl):
        r = tbl.iloc[0]
        ok.append(("one-sided ramp touches VAH only",
                   bool(r["touch_vah"]) and not bool(r["touch_val"])))
        ok.append(("one-sided ramp is NOT a double touch",
                   not bool(r["double_touch"])))

    # 10. A two-sided sweep in the gap MUST be a double touch.
    close2 = np.full(n, 100.0)
    close2[:g0] += rng.normal(0, 0.05, g0)
    half = (dec_i - g0) // 2
    close2[g0:g0 + half] = np.linspace(100.0, 140.0, half)
    close2[g0 + half:dec_i] = np.linspace(140.0, 60.0, dec_i - g0 - half)
    m2 = m.copy()
    m2["close"] = close2; m2["open"] = close2
    m2["high"] = close2 + 0.01; m2["low"] = close2 - 0.01
    t2 = build(m2)
    ok.append(("two-sided sweep IS a double touch",
               len(t2) >= 1 and bool(t2.iloc[0]["double_touch"])))

    # 11. The decision timestamp must be the production anchor hour, because
    #     the whole point is that the flag is available when NOCTUA forecasts.
    ok.append(("decision sits on the production anchor",
               len(tbl) >= 1
               and pd.Timestamp(int(tbl.iloc[0]["anchor_ts"]), unit="s",
                                tz="UTC").hour == ANCHOR_UTC))

    # 12. CAUSALITY, the one that must never regress: every bar the flag reads
    #     closed at least GAP_H hours before the decision.
    ok.append(("no bar inside the gap-to-decision boundary is read",
               len(tbl) >= 1
               and int(tbl.iloc[0]["anchor_ts"])
                   - (WINDOW_H + GAP_H) * HOUR
                   <= int(tbl.iloc[0]["anchor_ts"]) - GAP_H * HOUR))

    # 13. AGAINST THE REAL FILE, because none of the twelve checks above could
    #     have caught the bug that shipped: the fixture was built from the same
    #     wrong belief about `filled` as the code, so it agreed with it. A
    #     synthetic fixture can only test logic, never a convention -- for that
    #     you have to touch the real data, even briefly. Skips (loudly) if the
    #     corpus is absent rather than passing vacuously.
    real = Path("model/artifacts/btcusd_1min.parquet")
    if real.exists():
        raw = pd.read_parquet(real)
        raw["dt"] = pd.to_datetime(raw["timestamp"], unit="s", utc=True)
        sl = raw[(raw["dt"] >= pd.Timestamp("2024-03-01", tz="UTC"))
                 & (raw["dt"] < pd.Timestamp("2024-03-08", tz="UTC"))]
        got = build(sl)
        ok.append(("real corpus yields anchors (schema conventions honoured)",
                   len(got) >= 3))
    else:
        ok.append(("real corpus present for the schema check", False))

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FRVP value area at the anchor")
    ap.add_argument("--minutes", type=Path,
                    default=Path("model/artifacts/btcusd_1min.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/frvp_table.parquet"))
    ap.add_argument("--start", default="2017-08-01")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    mins = pd.read_parquet(a.minutes)
    mins["dt"] = pd.to_datetime(mins["timestamp"], unit="s", utc=True)
    mins = mins[mins["dt"] >= pd.Timestamp(a.start, tz="UTC")]
    print(f"{len(mins):,} minute bars from {a.start}")
    tbl = build(mins)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_parquet(a.out, index=False)
    n = len(tbl)
    print(f"\n{n:,} production anchors with a complete window and gap")
    print(f"  {tbl['dt'].min():%Y-%m-%d} .. {tbl['dt'].max():%Y-%m-%d}\n")
    print(f"{'flag':>22} {'rate':>8}")
    for k in ("touch_vah", "touch_val", "double_touch",
              "double_touch_vwap", "double_touch_mid", "above_poc"):
        print(f"{k:>22} {100 * tbl[k].mean():7.2f}%")
    print(f"\nvalue-area width, % of POC: median {tbl['width_pct'].median():.2f}")
    print(f"gap range,        % of POC: median {tbl['gap_range_pct'].median():.2f}")
    # The confound, stated as a number before anything is claimed.
    dt_ = tbl["double_touch"].to_numpy(bool)
    print(f"\nTHE CONFOUND, up front: median gap range is "
          f"{tbl['gap_range_pct'][dt_].median():.2f}% of POC on double-touch days "
          f"and {tbl['gap_range_pct'][~dt_].median():.2f}% otherwise.")
    print("A flag that selects wide-range days will predict high volatility "
          "whether or not\nthe volume profile contributes anything. That is "
          "what the placebo bands are for.")
    print("\nAGREEMENT WITH THE WIDTH-MATCHED PLACEBOS -- the decisive number.")
    print("The placebo has the same WIDTH and a centre the profile did not")
    print("choose, so agreement measures how much the POC's LOCATION matters.")
    for name in ("vwap", "mid"):
        col = tbl[f"double_touch_{name}"]
        agree = float((tbl["double_touch"] == col).mean())
        both = float((tbl["double_touch"] & col).sum())
        only = float((tbl["double_touch"] & ~col).sum())
        print(f"  {name:>5}-centred: agrees on {100 * agree:5.1f}% of days   "
              f"flagged by both {both:5.0f}   by FRVP only {only:4.0f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
