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


def by_expiry(ep_all, X_all, is_btc, a) -> int:
    """Does the pooling gain depend on the EXPIRY being asked about?

    ONE train/test split, not the six walk-forward folds. The horizon breakdown
    needs the test slice widened past the production H = 19, and refitting that
    six times over 1.36M episodes buys precision this question does not need --
    the aggregate result is already established in the walk-forward run above.
    Read this as a decomposition of that result, not as independent evidence.
    """
    import numpy as _np
    from noctua import baselines as _B
    from noctua import infer as _I
    from noctua.model import BASE_COLS as _BC
    from noctua.train import prepare as _prep, train_model as _tm
    from sklearn.isotonic import IsotonicRegression

    fin = _np.isfinite(X_all.to_numpy()).all(1)
    sp = S.time_splits(ep_all)
    m_va = sp["calib"] & fin & is_btc          # calibrate on BTC in both arms
    H_all = ep_all.H.to_numpy(_np.float64)
    alphas = _np.array([1.0, 2.0, 3.0, 5.0])
    bu = _np.log1p(alphas / 100.0)

    print("\nPER-EXPIRY, single split (see docstring: a decomposition, not "
          "independent evidence)\n")
    out = {}
    for arm, filt in (("btc_only", is_btc), ("pooled", _np.ones(len(ep_all), bool))):
        m_tr = sp["train"] & fin & filt
        tr, stds = _prep(ep_all, X_all, m_tr)
        wtr = S.sample_weights(ep_all, m_tr)
        va, _ = _prep(ep_all, X_all, m_va, *stds)
        ols = _B.OLS(_BC).fit(pd.DataFrame(tr["Xb"], columns=_BC),
                              tr["y"].astype(_np.float64), wtr)
        bl = _B.fit_vol_baselines(X_all[m_tr],
                                  _B.har_target(ep_all.RV.to_numpy()[m_tr], H_all[m_tr]),
                                  wtr)
        models = [_tm(tr, wtr, va, hidden=a.hidden, epochs=40, seed=s_,
                      verbose=False, ols_beta=ols.beta)[0] for s_ in range(a.seeds)]
        print(f"  {arm}: trained on {m_tr.sum():,} episodes")

        rows = []
        for H in sorted(ep_all.H.unique()):
            m = (sp["test"] & fin & is_btc
                 & (ep_all.H == H).to_numpy() & (ep_all.anchor_hour == 17).to_numpy())
            if m.sum() < 60:
                continue
            d, _ = _prep(ep_all, X_all, m, *stds)
            lp = bl["log_har_cal"].predict(X_all[m])
            preds = [_I.predict(mm, d, har_logvol=lp) for mm in models]
            pr = dict(preds[0])
            for k in ("qa", "sigma_atoms", "sigma_med", "q_r", "q_up", "q_dn"):
                pr[k] = _np.mean([q[k] for q in preds], axis=0)
            e = ep_all[m]
            rv = e.RV.to_numpy(); Mu = _np.abs(e.M_up.to_numpy())
            sig = _np.asarray(pr["sigma_med"], _np.float64)
            r = _np.maximum(rv**2, 1e-18) / _np.maximum(sig**2, 1e-18)
            dsc = unc = 0.0
            for j, u in enumerate(bu):
                y = (Mu >= u).astype(float)
                if y.mean() <= 0 or y.mean() >= 1:
                    continue
                pv = _I.touch_prob(pr, _np.full(int(m.sum()), u), True) \
                    if hasattr(_I, "touch_prob") else None
                if pv is None:
                    continue
                base = float(y.mean())
                pc = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip"
                                        ).fit_transform(pv, y)
                dsc += float(_np.mean((base - y)**2) - _np.mean((pc - y)**2))
                unc += float(_np.mean((base - y)**2))
            rows.append({"H": int(H), "n": int(m.sum()),
                         "qlike": float(_np.mean(r - _np.log(r) - 1.0)),
                         "DSS": dsc / unc if unc > 0 else float("nan")})
        out[arm] = rows

    print(f"\n{'H':>4} {'n':>5} {'QLIKE btc':>10} {'QLIKE pool':>11} {'d%':>7} "
          f"{'DSS btc':>9} {'DSS pool':>9} {'d%':>7}")
    for rb, rp in zip(out["btc_only"], out["pooled"]):
        dq = 100 * (rp["qlike"] / rb["qlike"] - 1)
        ds = (100 * (rp["DSS"] / rb["DSS"] - 1)) if rb["DSS"] == rb["DSS"] else float("nan")
        print(f"{rb['H']:>4} {rb['n']:>5} {rb['qlike']:10.4f} {rp['qlike']:11.4f} "
              f"{dq:+6.1f}% {rb['DSS']:9.5f} {rp['DSS']:9.5f} {ds:+6.1f}%")
    a.out.with_name("pooled_by_expiry.json").write_text(
        json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwritten -> {a.out.with_name('pooled_by_expiry.json')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Does pooling other crypto help BTC?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--assets", type=Path, default=ASSET_DIR)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/pooled.json"))
    ap.add_argument("--by-expiry", action="store_true",
                    help="score per horizon on one split instead of walk-forward")
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

    if a.by_expiry:
        return by_expiry(ep_all, X_all, is_btc, a)

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
