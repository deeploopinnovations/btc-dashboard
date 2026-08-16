"""
eval/freshness.py
=====================================================================
Does giving the model the hour it currently throws away make it better?

THE DEFECT

`features.py` states its contract as "every feature is a function of hourly
rows with index <= a-1". The implementation satisfies it twice:

    _trailing_sum(x, k)[i] = sum(x[i-k : i])      # already excludes i
    out = F[k][rows - 1]                          # then shifted again

so the freshest bar the model ever sees is `a-2`. Hour `a-1` -- the last
COMPLETE hour before the anchor, the single most recent observation available
and, for a volatility cascade, the most informative one -- is discarded on
every episode.

This was defensive, not accidental: the contract note says the double shift
means "an off-by-one cannot silently leak the anchor hour itself". The
direction of the error is the safe one. But the safety is redundant --
`audit_lookahead()` verifies the contract NUMERICALLY by corrupting the future
and checking that no feature moves, and it passes at both settings (max
feature change 0.000e+00 over 200,000 probed episodes). So the second shift
buys no protection that is not already checked, and it costs real information.

WHY THIS IS WORTH MEASURING RATHER THAN JUST FIXING

The cost is not obviously large. `har_1d` averages 24 hours; dropping one and
adding another shifts it by ~4%. But `har_1h` IS that single hour, so at the
default it is not "realized vol over the last hour" at all -- it is realized
vol over the hour before last, a feature whose name has been wrong since it
was written. Short horizons should feel this most, which the per-expiry
breakdown can confirm or refute.

Against that, the honest possibility is that the freshest hour is mostly
microstructure noise and the model is better off without it. Announcing an
improvement before measuring it is exactly the error that produced the
retracted pooling prediction in BENCHMARK.md section 6c. So: same six
walk-forward folds, same seeds, same committee, one variable changed.

    python -m model.eval.freshness --out model/artifacts/freshness.json
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

from eval.benchmark import run_fold, summarise                            # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.features import audit_lookahead, build_features               # noqa: E402
from noctua.train import load_all                                         # noqa: E402


def causal_sigma_fn(ep, X):
    """The shipped stage-B reference, rebuilt from whichever X is in play.

    Each arm must get the reference built from its OWN features, otherwise the
    comparison would confound the feature change with a change of target.
    """
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def paired(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(d.mean()), "wins": int((d > 0).sum()), "n": n,
            "t_like": float(d.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is the discarded hour worth anything?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hours", type=Path,
                    default=Path("model/artifacts/btcusd_1h.parquet"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--by-expiry", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/freshness.json"))
    a = ap.parse_args(argv)

    ep, X_lag1 = load_all(a.artifacts)
    hours = pd.read_parquet(a.hours)

    print("rebuilding features with extra_lag_hours=0 ...", flush=True)
    X_lag0 = build_features(hours, ep, extra_lag_hours=0)[list(X_lag1.columns)]

    # The whole claim rests on lag=0 still being causal. Verify it here rather
    # than trusting the argument, and refuse to report scores if it is not.
    aud = audit_lookahead(hours, ep, n_probe=200, extra_lag_hours=0)
    print(f"lookahead audit at lag 0: leak_free={aud['leak_free']} "
          f"max_change={aud['max_abs_feature_change']:.3e} "
          f"episodes={aud['episodes_checked']:,}")
    if not aud["leak_free"]:
        raise SystemExit(f"REFUSING: lag=0 leaks -> {aud['offending_features']}")

    d = (X_lag0 - X_lag1).abs()
    moved = {c: float(d[c].mean()) for c in X_lag1.columns if float(d[c].mean()) > 0}
    print(f"{len(moved)} of {X_lag1.shape[1]} features change; "
          f"largest mean |delta|: "
          f"{sorted(moved.items(), key=lambda kv: -kv[1])[:4]}\n")

    folds = S.walk_forward_folds(ep)
    recs = []
    for f in folds:
        line = {"year": f["year"]}
        for name, Xa in (("lag1_shipped", X_lag1), ("lag0_fresh", X_lag0)):
            t0 = time.time()
            out = run_fold(ep, Xa, f, hidden=a.hidden, seeds=a.seeds,
                           sigma_ref_fn=causal_sigma_fn(ep, Xa))
            if out is None:
                print(f"  {f['year']}  {name:13} SKIPPED")
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            line[name] = s
            print(f"  {f['year']}  {name:13} DSC/UNC {s['DSS']:.5f}  "
                  f"pinball {s['pinball']:.6f}  CRPS {s['crps']:.6f}  "
                  f"QLIKE {s['qlike']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if "lag1_shipped" in line and "lag0_fresh" in line:
            recs.append(line)

    if not recs:
        print("no fold produced both arms")
        return 1

    print(f"\n{'metric':>14} {'lag1 (shipped)':>15} {'lag0 (fresh)':>13} "
          f"{'delta':>11} {'wins':>6} {'t-like':>7}")
    report = {}
    for key, sgn in (("DSS", +1), ("pinball", -1), ("crps", -1), ("qlike", -1)):
        a0 = np.array([r["lag1_shipped"][key] for r in recs])
        a1 = np.array([r["lag0_fresh"][key] for r in recs])
        p = paired(sgn * (a1 - a0))
        report[key] = {"lag1": float(a0.mean()), "lag0": float(a1.mean()), **p}
        print(f"{key:>14} {a0.mean():15.6f} {a1.mean():13.6f} "
              f"{a1.mean()-a0.mean():+11.6f} {p['wins']:>3}/{len(recs):<2} "
              f"{p['t_like']:+7.2f}")
    print("\n  n = 6 folds; 't-like' is descriptive, not a p-value.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"audit": aud, "features_changed": moved, "folds": recs,
         "summary": report, "seeds": a.seeds, "hidden": a.hidden},
        indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
