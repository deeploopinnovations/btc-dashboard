"""
eval/worked_example.py
=====================================================================
What the shipped model actually says, and what actually happened.

WHY THIS EXISTS

Every other file here reports aggregate scores. Those are the right way to
judge a forecaster and the wrong way to understand one. This prints the
model's full output for individual episodes alongside the realized outcome,
so the numbers can be read the way they are traded.

THE CONFIGURATION IS THE DEPLOYED ONE

An option seller opening at 17:00 UTC (22:30 IST) and holding to 12:00 UTC the
next day (17:30 IST) is holding a 19-hour window -- exactly
`splits.production_mask`, and exactly what every benchmark number in
BENCHMARK.md is measured on. So these episodes need no translation: they are
the product.

WHAT IS SHOWN, AND HOW MUCH EACH NUMBER IS WORTH

  sigma           the volatility forecast. Beats a calibrated Log-HAR by
                  ~6% QLIKE across anchors (BENCHMARK.md 6e).
  p_amplify       P(this window is wilder than the trailing one) -- the
                  sell-premium versus buy-the-straddle call. DSC/UNC 20.3%,
                  beats climatology 6/6 folds (6g). The strongest number
                  here.
  touch probs     P(price reaches +/- x% before settlement) -- whether a sold
                  strike breaks. DSC/UNC ~5%, deep-tail calibration 1.09pp
                  against the Gaussian's 3.33pp (2, 6f).
  safe levels     the strike at which touch probability falls to alpha.
  p_up            DELIBERATELY NOT SHOWN. Direction has no measurable skill
                  (2), the served field is pinned to 50, and printing it here
                  would invite exactly the use the pinning exists to prevent.

    python -m model.eval.worked_example --n 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua import splits as S                                            # noqa: E402
from noctua.features import build_features                                # noqa: E402
from serve.runtime import load_model                                      # noqa: E402

GRID_PCT = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
ALPHAS = (0.01, 0.05, 0.10, 0.20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Show the model on real episodes")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--n", type=int, default=6, help="most recent episodes")
    a = ap.parse_args(argv)

    hours = pd.read_parquet(a.artifacts / "btcusd_1h.parquet")
    ep = pd.read_parquet(a.artifacts / "episodes.parquet")
    m = load_model()

    prod = S.production_mask(ep)
    e = ep[prod].sort_values("anchor_ts").tail(a.n).reset_index(drop=True)
    X = build_features(hours, e)
    d = m.prepare(X, e["H"].to_numpy(np.float64))
    pred = m.predict(d)

    close = hours["close"].to_numpy(np.float64)
    rv5 = hours["rv5"].to_numpy(np.float64)
    rows = e["row"].to_numpy(np.int64)
    H = e["H"].to_numpy(np.int64)

    print(f"NOCTUA {m.meta.get('version')} | {a.n} most recent production "
          f"episodes (H=19 anchored 17:00 UTC)")
    print("This is the same window as an open at 22:30 IST held to 17:30 IST.\n")

    n_hit_tot = n_ep = 0
    for i in range(len(e)):
        r = int(rows[i])
        spot = float(close[r - 1])
        trail = float(np.sqrt(rv5[r - H[i]:r].sum()))
        sig = float(pred["sigma_med"][i])
        # realized
        rv_fwd = float(e["RV"].iloc[i])
        mu = float(abs(e["M_up"].iloc[i]))
        md = float(abs(e["M_dn"].iloc[i]))
        ret = float(e["R"].iloc[i])

        qa = pred["qa"][i]
        tot = np.exp(qa) * np.sqrt(H[i])
        p_amp = float(1.0 - np.interp(trail, tot, m.levels, left=0.0, right=1.0))
        amp_real = rv_fwd > trail

        print(f"=== anchor {e['dt'].iloc[i]}  (settles +19h) ===")
        print(f"  spot at open        {spot:,.0f}")
        print(f"  sigma forecast      {100*sig:.3f}% of spot over the window   "
              f"| REALIZED {100*rv_fwd:.3f}%   ratio {rv_fwd/max(sig,1e-9):.2f}")
        print(f"  trailing 19h vol    {100*trail:.3f}%")
        print(f"  P(wilder than trailing) {100*p_amp:5.1f}%"
              f"   -> ACTUALLY {'WILDER' if amp_real else 'calmer'}"
              f"   [{'correct side' if (p_amp > 0.5) == amp_real else 'wrong side'}]")
        print(f"  realized move       up {100*mu:.2f}%  down {100*md:.2f}%  "
              f"close {100*ret:+.2f}%")
        print(f"  {'strike':>8} {'P(touch up)':>12} {'P(touch dn)':>12}  outcome")
        for pct in GRID_PCT:
            # the runtime scores a whole batch at once, so the barrier has to
            # be supplied per episode rather than as a scalar
            u = np.full(len(e), np.log1p(pct / 100.0))
            pu = float(m.touch_prob(pred, u, True)[i])
            pd_ = float(m.touch_prob(pred, u, False)[i])
            hu, hd = mu >= u[0], md >= u[0]
            n_hit_tot += int(hu) + int(hd); n_ep += 2
            tag = ("BOTH broke" if hu and hd else "up broke" if hu
                   else "down broke" if hd else "held")
            print(f"  {pct:6.1f}%  {100*pu:11.1f}% {100*pd_:11.1f}%  {tag}")
        lv = []
        for al in ALPHAS:
            up = float(m.safe_level(pred, al, True)[i])
            dn = float(m.safe_level(pred, al, False)[i])
            lv.append(f"a={al:.2f}: call {spot*np.exp(up):,.0f} "
                      f"({100*(np.exp(up)-1):+.2f}%) / put {spot*np.exp(-dn):,.0f} "
                      f"({-100*(1-np.exp(-dn)):+.2f}%)")
        print("  safe levels (touch prob <= alpha):")
        for s in lv:
            print(f"    {s}")
        print()

    print(f"across these episodes: {n_hit_tot} of {n_ep} barrier-sides broke")
    print("\nReliability of each number, from the walk-forward benchmark:")
    print("  sigma          ~6% better QLIKE than a calibrated Log-HAR")
    print("  P(wilder)      DSC/UNC 20.3%, beats climatology 6/6 folds  <- strongest")
    print("  touch probs    DSC/UNC ~5%; deep-tail calibration 1.09pp vs "
          "Gaussian 3.33pp")
    print("  direction      no measurable skill -- not shown, and pinned to 50 "
          "in the served payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
