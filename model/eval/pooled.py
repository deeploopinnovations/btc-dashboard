"""
eval/pooled.py
=====================================================================
Does training on OTHER crypto help Bitcoin, hurt it, or do nothing?

The question the whole 3300-day harvest existed to answer, and it needed that
harvest for one reason: leakage. With only 900 days of altcoin history every
altcoin bar started 2024-02-27, inside BTC's test era (from 2024-07-01), and
pooling it would have put ETH's 2025 volatility -- roughly 0.8 correlated with
BTC's -- into a model whose out-of-sample claim is measured on BTC in 2025.
Contemporaneous cross-sectional leakage, and the kind that produces a
convincing number.

BTC, ETH, LTC and XRP now reach back to 2017-08-02. SOL does not (it lists
later, 899 days, all of it inside the test era) and is therefore EXCLUDED from
training entirely -- it stays a zero-shot evaluation asset in
eval/cross_asset.py, which is unaffected by any of this because it refits
nothing.

HOW THE POOL IS BUILT

Altcoin episodes are constructed exactly like BTC's -- same horizons, same
anchor grid, same feature builder -- and then passed through the SAME
`splits.time_splits` boundaries. An altcoin episode is admitted to training
only if its window closes before the same embargoed boundary a BTC episode
would have to clear. Nothing about being a different asset relaxes the
calendar.

WHAT IS SCORED

The BTC production slice, and only that. Adding assets must be judged on the
instrument that ships, not on an average across the pool that a strong altcoin
could carry.

    python -m model.eval.pooled
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

from noctua import splits as S                                      # noqa: E402
from noctua.episodes import build_episodes                          # noqa: E402
from noctua.features import build_features                          # noqa: E402
from noctua.train import load_all                                   # noqa: E402

from .benchmark import run_fold                                     # noqa: E402
from .efficiency import paired, summarise                           # noqa: E402

ASSET_DIR = Path("data/assets")
# SOL is deliberately absent: 899 days, every one of them inside BTC's test
# era, so it cannot enter training without reintroducing the exact leak the
# long harvest was run to avoid.
POOL = ("eth", "ltc", "xrp")
HORIZONS = (6, 12, 19, 24)
MIN_DAYS_BEFORE_TRAIN_END = 400


def asset_episodes(path: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Episodes + features for one asset, built exactly as BTC's are."""
    hours = pd.read_parquet(path)
    ts = pd.to_datetime(hours["hour_ts"], unit="s", utc=True)
    train_end = pd.Timestamp(S.TRAIN_END, tz="UTC")
    usable = (train_end - ts.min()).days
    if usable < MIN_DAYS_BEFORE_TRAIN_END:
        print(f"  {label}: SKIPPED -- only {usable} days before TRAIN_END "
              f"({ts.min().date()}); it would contribute test-era data only")
        return None

    ep = build_episodes(hours, tuple(HORIZONS))
    X = build_features(hours, ep)
    ok = np.isfinite(X.to_numpy()).all(1)
    ep, X = ep[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    ep = ep.assign(asset=label)
    print(f"  {label}: {len(ep):,} episodes  {ts.min().date()} -> {ts.max().date()}")
    return ep, X


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Does pooling other crypto help BTC?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--assets", type=Path, default=ASSET_DIR)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/pooled.json"))
    a = ap.parse_args(argv)

    ep_btc, X_btc = load_all(a.artifacts)
    ep_btc = ep_btc.assign(asset="btc")
    print(f"BTC: {len(ep_btc):,} episodes\n\npooling candidates:")

    extra_ep, extra_X = [], []
    for sym in POOL:
        p = a.assets / f"{sym}_history.parquet"
        if not p.exists():
            print(f"  {sym}: no bundle at {p}")
            continue
        got = asset_episodes(p, sym)
        if got is not None:
            extra_ep.append(got[0])
            extra_X.append(got[1])

    if not extra_ep:
        print("\nno altcoin reaches far enough back to train on. Nothing to test.")
        return 1

    # Column alignment is non-negotiable: a pooled frame with a different
    # column ORDER would silently feed the network permuted features.
    cols = list(X_btc.columns)
    for i, Xa in enumerate(extra_X):
        missing = [c for c in cols if c not in Xa.columns]
        if missing:
            print(f"feature mismatch on pooled asset {i}: missing {missing}")
            return 2
        extra_X[i] = Xa[cols]

    ep_all = pd.concat([ep_btc] + extra_ep, ignore_index=True)
    X_all = pd.concat([X_btc] + extra_X, ignore_index=True)
    is_btc = (ep_all.asset == "btc").to_numpy()
    print(f"\npooled: {len(ep_all):,} episodes "
          f"({is_btc.sum():,} BTC + {(~is_btc).sum():,} altcoin)")

    # The test slice must be BTC-only in BOTH arms, or the arms are not
    # comparable. run_fold intersects fold["test"] with the production mask;
    # production_mask is asset-blind, so the altcoin rows are removed here.
    folds_all = S.walk_forward_folds(ep_all)
    for f in folds_all:
        f["test"] = f["test"] & is_btc
        f["calib"] = f["calib"] & is_btc          # calibrate on BTC too

    # BTC-only arm: identical folds, with the altcoin rows removed from TRAIN.
    print(f"\n{len(folds_all)} folds x 2 arms x {a.seeds} seeds\n")
    recs = []
    for f in folds_all:
        line = {"year": f["year"]}
        for name, filt in (("btc_only", is_btc), ("pooled", None)):
            t0 = time.time()
            out = run_fold(ep_all, X_all, f, hidden=a.hidden, seeds=a.seeds,
                           train_filter=filt)
            if out is None:
                print(f"  {f['year']}  {name:9} SKIPPED")
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            s["DSS_har"] = summarise(out["rows"], "log_har_gauss")["DSS"]
            s["n_train"] = int((f["train"] & (filt if filt is not None
                                              else np.ones(len(ep_all), bool))).sum())
            line[name] = s
            print(f"  {f['year']}  {name:9} n_tr={s['n_train']:7,}  "
                  f"DSC/UNC {s['DSS']:.5f}  pinball {s['pinball']:.6f}  "
                  f"CRPS {s['crps']:.6f}  QLIKE {s['qlike']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if "btc_only" in line and "pooled" in line:
            recs.append(line)

    if not recs:
        print("no fold produced both arms")
        return 1

    print(f"\n{'metric':>14} {'btc_only':>10} {'pooled':>10} {'delta':>11} "
          f"{'wins':>6} {'t-like':>7}")
    report = {}
    for key, sgn in (("DSS", +1), ("pinball", -1), ("crps", -1), ("qlike", -1)):
        a0 = np.array([r["btc_only"][key] for r in recs])
        a1 = np.array([r["pooled"][key] for r in recs])
        p = paired(sgn * (a1 - a0))
        report[key] = {"btc_only": float(a0.mean()), "pooled": float(a1.mean()), **p}
        print(f"{key:>14} {a0.mean():10.6f} {a1.mean():10.6f} "
              f"{a1.mean()-a0.mean():+11.6f} {p['wins']:>3}/{len(recs):<2} "
              f"{p['t_like']:+7.2f}")

    print("\n  n = 6 folds; 't-like' is descriptive, not a p-value.")
    a.out.write_text(json.dumps({"pool": list(POOL), "seeds": a.seeds,
                                 "summary": report, "folds": recs},
                                indent=2, default=float) + "\n")
    print(f"\nwritten -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
