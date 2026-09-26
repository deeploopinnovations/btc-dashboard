"""
serve/predict.py
=====================================================================
One forecast, end to end: live bars -> features -> NOCTUA -> JSON.

Emits two payloads.

`legacy` is byte-compatible with what `src/data.js fetchKronos()` already
consumes, so the dead Kronos scrape can be replaced without touching the
dashboard. Two fields need an honest mapping:

  upside  PINNED TO 50.0. Walk-forward found NO directional skill at this
          horizon (log-loss 0.6941 vs 0.6931 for a coin flip -- fractionally
          WORSE than nothing), independently reproducing this repo's own
          SELLER_DIRECTIONAL_ALPHA.md conclusion that direction is a coin
          flip. Since src/data.js pipes this field into strike selection,
          passing the raw number through would trade on noise. The raw value
          is published untouched as `p_up_raw`. See `to_legacy` for the full
          argument.

  volAmp  P(RV over the window > trailing RV of the same length). This one IS
          a real forecast -- it comes from Stage A, which beats Log-HAR by
          2.79% QLIKE (p = 0.043, 5/6 walk-forward folds).

`noctua` is the real product: barrier survival curves and the alpha-safe
strike levels an option seller actually needs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua.features import build_features                    # noqa: E402
from serve.adaptive import (apply_correction, qlike_scale,  # noqa: E501
                            volatility_correction)  # noqa: E402
from serve.fetch import fetch_bars                            # noqa: E402
from serve.history import get_hours, load_bundle              # noqa: E402
from serve.runtime import load_model                          # noqa: E402

_MODEL_TAG = ["NOCTUA-v1"]   # set by main() once the artifact is chosen

PROD_H = 19
PROD_ANCHOR_UTC = 17
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)
BARRIER_GRID_PCT = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)


def _next_anchor(now_ts: int) -> int:
    """The next 17:00 UTC boundary at or before `now` (today's if past)."""
    dt = datetime.fromtimestamp(now_ts, timezone.utc)
    anchor = dt.replace(hour=PROD_ANCHOR_UTC, minute=0, second=0, microsecond=0)
    return int(anchor.timestamp())


# Which functional of the predictive distribution the REPORTED sigma is.
# "mean" is the QLIKE-optimal choice and the adopted default; "median" is the
# historical behaviour, kept so `tests/test_level_report.py` can run both and
# assert that the choice reaches the reported number and nothing else.
REPORT_FUNCTIONAL = "mean"

# How much of the predictive distribution's WIDTH to keep, about its own
# median. 1.0 is a bit-identical no-op (`infer.scale_atoms` returns the same
# array) and is the shipped default: nothing here is adopted.
#
# This exists so the dispersion correction has a serving switch and so
# `tests/test_dispersion_report.py` can be a gate that CAN FAIL. R2 -- a guard
# whose failing branch has never been exercised is not a guard, and five have
# already been found in this project printing reassuring output while unable to
# return the other answer.
#
# UNLIKE `REPORT_FUNCTIONAL`, THIS ONE MOVES THE PRODUCT. The functional choice
# reaches the reported scalar and provably nothing else (P3-barrier-channel);
# this writes `sigma_atoms`, which every barrier curve and safe level is built
# from. "Safe by construction" is therefore not available and the gate asserts
# field by field what moves. Measured at lambda = 0.87 on the production slice
# it improves four of six barrier metrics while its mirror degrades the same
# four (P3-dispersion-barriers-result) -- ADVANCE, not ADOPT.
DISP_LAMBDA = 1.0


def forecast(model, hours: pd.DataFrame, H: int = PROD_H,
             anchor_ts: int | None = None, source: str = "unknown") -> dict:
    """Run one forecast anchored at `anchor_ts` (default: the latest full hour).

    `hours` is the merged hourly history from `serve.history.get_hours` --
    committed bundle plus the freshly fetched tail. It must reach back at least
    365 days, because `reg_rv_vs_year` looks that far and a short history would
    silently fall back to the feature's training mean rather than fail.
    """
    hour_ts = hours["hour_ts"].to_numpy(np.int64)

    if anchor_ts is None:
        anchor_ts = int(hour_ts[-1])          # forecast from the last closed hour
    row = int(np.searchsorted(hour_ts, anchor_ts))
    row = min(row, len(hours) - 1)
    if row < 24 * 22:
        raise RuntimeError("not enough history at the requested anchor")

    dt = pd.to_datetime(hour_ts[row], unit="s", utc=True)
    ep = pd.DataFrame({
        "anchor_ts": [hour_ts[row]], "H": [H], "row": [row],
        "dt": [dt], "anchor_hour": [dt.hour], "dow": [dt.dayofweek],
    })
    X = build_features(hours, ep)
    d = model.prepare(X, np.array([float(H)]))
    pred = model.predict(d, disp_lambda=DISP_LAMBDA)

    # Causal volatility-level recalibration. Measured out of sample on
    # 2024-07 onward, the raw forecast runs high -- realized vol lands below
    # it 66.4% of the time -- which pushes every quoted strike too far out and
    # costs premium. The correction is estimated only from episodes that have
    # already settled, so it carries no look-ahead. See serve/adaptive.py.
    cal = volatility_correction(model, hours, row, H)
    if cal["applied"]:
        pred = apply_correction(pred, cal["factor"])

    # THE REPORTED VOLATILITY AND THE BARRIER CURVE ARE TWO PRODUCTS WITH TWO
    # LOSSES, and this is the line where they part company.
    #
    # QLIKE -- and any squared-error loss on variance -- is minimised by the
    # conditional MEAN of variance. `sigma_med` is a median. On the production
    # slice the gap is a factor of 1.1250 and the published number was 12.5%
    # low against the loss it is scored under.
    #
    # The correction is applied HERE, to the reported scalar, and NOT to
    # `pred`. That is not a shortcut, it is the finding: `P2-scale-v2` put the
    # same constant inside the predictive object and lost all six barrier
    # metrics, and `P2-mean-level`'s SHUFFLED control lost them by the same
    # amount -- so the damage comes from moving the level at all, not from how
    # the shift is obtained. Left here, there is nothing downstream of it.
    #
    # `tests/test_level_report.py` asserts the separation by running this
    # function twice and requiring every touch probability, safe level, p_up
    # and p_vol_amplify to be bit-identical. That test fails the moment this
    # scalar reaches `pred`.
    # ADOPTED 2026-09-13 (P3-functional-parity, P3-functional-scorecard):
    # report the model's OWN conditional mean, not its median times a fitted
    # trailing estimate of the gap between them.
    #
    # `sigma_mean` is sqrt(mean(exp(2*atoms_y)) * H) -- the conditional mean of
    # variance, computed in the same forward pass in runtime.py and never used.
    # `qlike_scale` was a WINDOWED EMPIRICAL ESTIMATE of the same median-to-mean
    # ratio, with a clip rail and a settled-episode lag. Replacing an estimate of
    # a quantity with the quantity removes the window, the lag, the rail, and the
    # requirement that enough episodes have settled; and it is conditional per
    # episode where the trailing scalar was an average.
    #
    # MEASURED, on the research side: reading sigma_mean instead of sigma_med
    # improves raw pooled QLIKE by 13.3% / 20.7% / 26.0% / 11.9% at
    # H=1/6/24/168, and the resulting forecast beats every teacher in the zoo at
    # H=1, H=6 and H=24 under a symmetric two-parameter affine correction, tying
    # the HAR family at H=168. A FURTHER fitted level correction on top of it
    # makes it WORSE at all four horizons (0.54119 -> 0.54815 at H=1), which is
    # why the trailing scalar is dropped rather than kept on top.
    #
    # SCOPE CORRECTION, 2026-09-21 (P3-functional-adopt-scope). THE NUMBERS
    # ABOVE ARE THE RAW NETWORK'S, AND THIS PATH IS BLENDED. The teacher zoo
    # scores `noctua_v1` unblended; the serving runtime applies blend_w = 0.25
    # inside NumpyNoctua.predict, and a uniform log-shift cannot change the
    # mean/median RATIO but does change the level that ratio multiplies. Raw
    # sits at a calibration ratio of 1.43, so a factor near 1.2 lands close to
    # 1.0 -- that is the zoo result quoted above. The BLENDED path already sits
    # at 1.12, and the same factor overshoots to 0.82. Measured on the
    # production slice: median ratio 1.1231, mean ratio 0.8181, so the MEDIAN
    # is the closer of the two to 1 here and the "calibrated to within 1-4%"
    # claim does NOT hold on this pipeline.
    #
    # WHY THE MEAN IS NEVERTHELESS STILL REPORTED: the paired contrast on the
    # served slice is NOT SEPARATED -- median 0.25619 against mean 0.26039,
    # delta -1.64% favouring the median, CI [-0.03105, +0.01798] with 2 of 6
    # folds favouring the mean. n is 2,046 episodes here against 49,000 in the
    # zoo, because the production configuration is one episode per day, so a
    # 1.6% difference cannot resolve. Flipping a level decision on a point
    # estimate is how phase2/level-scale oscillated three times (R80), so this
    # stays put until a properly powered contrast on THIS slice says otherwise.
    #
    # SAFE BY CONSTRUCTION, not merely by test (P3-barrier-channel): the
    # committee builds every barrier curve from `pred["sigma_atoms"]` and never
    # reads the reported scalar -- a 13% change in it moves the curves by
    # 0.000e+00 while the same change to sigma_atoms moves them by 8.9e-03. So
    # this cannot repeat P2-scale-v2, whose damage came from post_shift_fn
    # rewriting the log-vol that FEEDS the quantiles.
    qs = qlike_scale(model, hours, row, H)          # reported for continuity
    spot = float(hours["close"].to_numpy()[row - 1])
    sig_med = float(pred["sigma_med"][0])
    sig_mean = float(pred["sigma_mean"][0])
    sigma = sig_mean if REPORT_FUNCTIONAL == "mean" else sig_med

    # trailing realized vol over the same window length, for volAmp
    rv5 = hours["rv5"].to_numpy(np.float64)
    trailing = float(np.sqrt(rv5[row - H:row].sum()))
    rv_q = np.stack([np.interp([0.5], model.levels, r) for r in pred["qa"]])
    p_amp = float(model_prob_rv_above(model, pred, trailing))

    curves = {"up": [], "dn": []}
    for pct in BARRIER_GRID_PCT:
        u = np.array([np.log1p(pct / 100.0)])
        curves["up"].append({
            "pct": pct, "price": round(spot * (1 + pct / 100.0), 2),
            "touch_prob": round(float(model.touch_prob(pred, u, True)[0]), 4)})
        curves["dn"].append({
            "pct": -pct, "price": round(spot * (1 - pct / 100.0), 2),
            "touch_prob": round(float(model.touch_prob(pred, u, False)[0]), 4)})

    safe = []
    for a in ALPHAS:
        u = float(model.safe_level(pred, a, True)[0])
        l = float(model.safe_level(pred, a, False)[0])
        safe.append({
            "alpha": a,
            "call_strike": round(spot * float(np.exp(u)), 2),
            "put_strike": round(spot * float(np.exp(-l)), 2),
            "call_pct": round(100 * (np.exp(u) - 1), 3),
            "put_pct": round(-100 * (1 - np.exp(-l)), 3),
        })

    p_up = float(model.prob_up(pred)[0])
    settle = int(hour_ts[row] + H * 3600)
    return {
        "anchor_utc": str(dt), "settle_utc": str(pd.to_datetime(settle, unit="s", utc=True)),
        "H_hours": H, "spot": round(spot, 2),
        "sigma_window_pct": round(100 * sigma, 3),
        "sigma_annualized_pct": round(100 * sigma * np.sqrt(365 * 24 / H), 1),
        "trailing_rv_pct": round(100 * trailing, 3),
        "p_up": round(p_up, 4),
        "p_vol_amplify": round(p_amp, 4),
        "safe_levels": safe,
        "barrier_curves": curves,
        "model": model.meta.get("version", "NOCTUA-v1"),
        # The reported sigma carries this; the predictive object does not.
        # The reported sigma is now a FUNCTIONAL CHOICE, not a correction. The
        # trailing scalar is still computed and published so the two can be
        # compared in the field -- it is an empirical estimate of the same
        # median-to-mean ratio this now takes exactly -- but it no longer
        # multiplies anything.
        "sigma_functional": {
            "reported": REPORT_FUNCTIONAL,
            "sigma_med_pct": round(100 * sig_med, 3),
            "sigma_mean_pct": round(100 * sig_mean, 3),
            "mean_over_median": round(sig_mean / max(sig_med, 1e-12), 4),
            "why": "QLIKE is minimised by the conditional MEAN of variance; "
                   "sigma_mean is that mean, from the same forward pass. See "
                   "P3-functional-parity.",
            "applies_to": "sigma_window_pct and sigma_annualized_pct only; "
                          "barrier_curves, safe_levels and p_up are built from "
                          "pred['sigma_atoms'] and p_vol_amplify from "
                          "pred['qa'], so none of them can see this choice "
                          "(P3-barrier-channel)",
        },
        "dispersion": {
            "lambda": round(float(DISP_LAMBDA), 4),
            "applied": bool(DISP_LAMBDA != 1.0),
            "note": "scales the predictive distribution's WIDTH about its own "
                    "median. 1.0 is a bit-identical no-op and is the shipped "
                    "default. See P3-dispersion-barriers-result (ADVANCE).",
            "applies_to": "sigma_atoms, and therefore sigma_window_pct and "
                          "sigma_annualized_pct (the mean functional), "
                          "barrier_curves' touch probabilities, safe_levels "
                          "and p_up. NOT sigma_med, NOT the barrier price "
                          "grid, and NOT p_vol_amplify, which reads pred['qa'] "
                          "rather than the atoms. Asserted field by field in "
                          "tests/test_dispersion_report.py.",
        },
        "sigma_scale": {
            "scale": round(float(qs["scale"]), 4),
            "applied": False,
            "n_settled_episodes": int(qs["n_episodes"]),
            "note": "NO LONGER APPLIED. Superseded by the exact conditional "
                    "mean; reported for comparison only. " + str(qs["reason"]),
            "applies_to": "nothing -- retained as a field for continuity",
        },
        "vol_calibration": {
            "factor": round(float(cal["factor"]), 4),
            "applied": bool(cal["applied"]),
            "n_settled_episodes": int(cal["n_episodes"]),
            "window_days": int(cal["window_days"]),
            "note": cal["reason"],
        },
        "source": source,
        "history_hours": int(len(hours)),
    }


def model_prob_rv_above(model, pred: dict, threshold: float) -> float:
    """P(realized vol over the window exceeds `threshold`), from Stage A."""
    qa = pred["qa"][0]
    H = pred["H"][0]
    tot = np.exp(qa) * np.sqrt(H)             # window vol at each quantile level
    return float(1.0 - np.interp(threshold, tot, model.levels, left=0.0, right=1.0))


def to_legacy(f: dict) -> dict:
    """The exact JSON shape `src/data.js fetchKronos()` already handles.

    `upside` is deliberately pinned to 50.0 rather than passed through.

    This is not timidity, it is the only defensible option. `src/data.js` feeds
    `kronos.upside` into `UI.computeStrikes()` to SKEW the recommended call and
    put strikes, and into the conviction score. Publishing a directional number
    with no validated skill (walk-forward log-loss 0.6941 vs 0.6931 for a coin
    flip) would push real strike recommendations around on noise -- strictly
    worse than staying neutral. The dashboard already handles a neutral reading
    correctly: `Math.abs(upside - 50) < 5` raises "no directional edge for
    asymmetric wing", which is precisely the right conclusion.

    The model's raw P(R > 0) is still published as `p_up_raw` (and in
    noctua.json) so nothing is hidden -- it simply is not wired to anything
    that trades.
    """
    now_ms = int(time.time() * 1000)
    return {
        "upside": 50.0,
        "p_up_raw": round(100 * f["p_up"], 1),
        "volAmp": round(100 * f["p_vol_amplify"], 1),
        "sourceTs": f["anchor_utc"][:19],
        "sourceMs": int(pd.Timestamp(f["anchor_utc"]).timestamp() * 1000),
        "tz": "UTC",
        "ageHrs": 0.0,
        "freshness": "fresh",
        "fetchedAt": now_ms,
        "proxy": "noctua-local",
        "_updatedMs": now_ms,
        "model": f.get("model", _MODEL_TAG[0]),
        "upside_is_informative": False,
        "warning": (
            "upside is pinned to 50.0 on purpose: direction is NOT predictable at "
            "this horizon (walk-forward log-loss 0.6941 vs 0.6931 for a coin flip), "
            "and src/data.js uses this field to skew strike recommendations. The "
            "model's raw P(up) is in p_up_raw, wired to nothing. volAmp and the "
            "barrier levels in noctua.json ARE validated."
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run one NOCTUA forecast")
    p.add_argument("--weights", type=Path, default=None,
                   help="explicit weights file; defaults to v2 if present, else v1")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--offline", action="store_true",
                   help="use the committed history bundle only, no network fetch")
    p.add_argument("--anchor", type=int, help="unix ts of the anchor hour")
    p.add_argument("--H", type=int, default=PROD_H)
    a = p.parse_args(argv)

    # Prefer the v2 committee artifact; fall back to v1 if it is absent. The
    # runtime is chosen from the artifact's own metadata, not its filename.
    model = load_model(a.weights)
    print(f"[predict] model = {model.meta.get('version', 'NOCTUA-v1')} "
          f"({model.meta.get('n_params_total', model.meta.get('n_params')):,} params)")
    if a.offline:
        hours, src = load_bundle(), "offline:bundle"
    else:
        hours, info = get_hours(fetch_bars)
        src = info["source"]

    _MODEL_TAG[0] = model.meta.get("version", "NOCTUA-v1")
    f = forecast(model, hours, H=a.H, anchor_ts=a.anchor, source=src)
    legacy = to_legacy(f)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "noctua.json").write_text(json.dumps(f, indent=2) + "\n")
    (a.out_dir / "kronos.json").write_text(json.dumps(legacy, indent=2) + "\n")

    print(json.dumps(f, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
