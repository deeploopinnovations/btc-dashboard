"""
eval/anchors.py
=====================================================================
Every published number describes an anchor the product almost never uses.

THE FINDING THAT PROMPTED THIS

`splits.production_mask` is `H == 19 AND anchor_hour == 17`, and every headline
in BENCHMARK.md -- QLIKE, DSC/UNC, barrier calibration, the BLEND_W sweep, the
committee weighting, the training-method ablation -- is measured on it and
nothing else.

`serve/predict.py:74` picks its anchor differently:

    if anchor_ts is None:
        anchor_ts = int(hour_ts[-1])          # the last CLOSED hour

and the cron that publishes (`fetch-data.yml`, every 30 minutes) never passes
`--anchor`. `_next_anchor()`, the function that would pin serving to 17:00
UTC, is defined at `predict.py:56` and called from nowhere in the repository.

Measured over the 102 forecasts in this repo's own git history:

    published at anchor_hour == 17 ..... 5 of 102  (4.9%)
    anchor hours actually used ......... all 24, roughly uniform

So ~95% of everything ever shown to a user was produced at a configuration
that has never been scored. That is not a leak and not a bug in the model --
it is a claim-versus-product mismatch, and the honest response is to find out
whether the claim survives at the anchors that ship.

WHAT COULD GO WRONG, MECHANICALLY

BTC volatility has a strong and well-documented intraday seasonal: the US
session is livelier than the Asian pre-dawn, and 17:00 UTC sits just after the
US equity open, near the daily peak. The model is given `cal_hour_sin/cos` and
is trained on all 24 anchors, so it has the means to adapt. But three of its
downstream constants were TUNED on the 17:00 slice alone -- the Log-HAR
ensemble weight, the committee's equal weighting, and the adaptive volatility
correction -- and a constant tuned at the seasonal peak need not be right at
the trough.

DESIGN

`run_fold` is deterministic given its seeds, so calling it twice on one fold
with different `prod_override` masks trains identical models and changes only
the slice they are scored on. That makes the arms exactly paired: any
difference is the anchor, not the fit.

    python -m model.eval.anchors --out model/artifacts/anchors.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                       # noqa: E402
from eval.efficiency import summarise                                      # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402


def causal_sigma_fn(ep, X):
    """The shipped stage-B reference (`serve_consistent`)."""
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
    ap = argparse.ArgumentParser(description="Does the benchmark hold at served anchors?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--max-episodes", type=int, default=6000,
                    help="cap on episodes per non-reference arm; the "
                         "17:00 arm is never subsampled")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/anchors.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    hour = ep["anchor_hour"].to_numpy()
    h19 = (ep["H"] == 19).to_numpy()

    arms = {
        "anchor_17_benchmarked": h19 & (hour == 17),
        "anchor_all_served":     h19,
        "anchor_not_17":         h19 & (hour != 17),
        "anchor_00_06_quiet":    h19 & (hour >= 0) & (hour < 6),
        "anchor_12_18_active":   h19 & (hour >= 12) & (hour < 18),
    }

    # Scoring cost is linear in episodes and the committee is not cheap: every
    # episode costs a 32-atom mixing integral across 8 barrier cells. The wide
    # arms hold 128,000 episodes against the benchmarked slice's 5,325, which
    # turns a 40-second fold into a many-minute one for no statistical gain --
    # the question here is whether the anchor MOVES the score, and a few
    # thousand episodes answers that as well as a hundred thousand.
    #
    # Subsample deterministically (fixed seed, drawn once over the whole
    # panel) so the arms stay directly comparable across folds, and keep the
    # 17:00 arm whole so the reference is exactly the published slice.
    rng = np.random.default_rng(20260816)
    for k in list(arms):
        m = arms[k]
        if k != "anchor_17_benchmarked" and m.sum() > a.max_episodes:
            idx = np.flatnonzero(m)
            keep = rng.choice(idx, size=a.max_episodes, replace=False)
            m2 = np.zeros_like(m)
            m2[keep] = True
            arms[k] = m2
            print(f"  {k:24} {m.sum():7,} -> {m2.sum():7,} episodes (subsampled)")
        else:
            print(f"  {k:24} {m.sum():7,} episodes")
    print()

    folds = S.walk_forward_folds(ep)
    sig_fn = causal_sigma_fn(ep, X)
    recs = []
    for f in folds:
        line = {"year": f["year"]}
        for name, msk in arms.items():
            t0 = time.time()
            out = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                           sigma_ref_fn=sig_fn, prod_override=msk)
            if out is None:
                print(f"  {f['year']}  {name:24} SKIPPED")
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            s["qlike_har"] = out["vol"]["log_har"]
            s["n"] = int(out["rows"][0]["n"])
            line[name] = s
            print(f"  {f['year']}  {name:24} n={s['n']:6,}  "
                  f"DSC/UNC {s['DSS']:.5f}  QLIKE {s['qlike']:.4f}  "
                  f"(HAR {s['qlike_har']:.4f})  ({time.time()-t0:.0f}s)", flush=True)
        recs.append(line)

    base = "anchor_17_benchmarked"
    print(f"\n{'arm':>24} {'DSC/UNC':>10} {'QLIKE':>9} {'vs HAR %':>9} "
          f"{'dDSC vs 17':>11} {'dQLIKE %':>9} {'t-like':>7}")
    report = {}
    for name in arms:
        ok = [r for r in recs if name in r and base in r]
        if not ok:
            continue
        dss = np.array([r[name]["DSS"] for r in ok])
        ql = np.array([r[name]["qlike"] for r in ok])
        qh = np.array([r[name]["qlike_har"] for r in ok])
        b_dss = np.array([r[base]["DSS"] for r in ok])
        b_ql = np.array([r[base]["qlike"] for r in ok])
        # the product claim is "better than Log-HAR", so carry that per arm
        vs_har = 100.0 * (ql / qh - 1.0)
        p = paired(-(ql / b_ql - 1.0))
        report[name] = {"DSS": float(dss.mean()), "qlike": float(ql.mean()),
                        "qlike_har": float(qh.mean()),
                        "vs_har_pct": float(vs_har.mean()),
                        "d_dss_vs_17": float(dss.mean() - b_dss.mean()),
                        "d_qlike_pct_vs_17": float(100 * (ql.mean() / b_ql.mean() - 1)),
                        **p}
        print(f"{name:>24} {dss.mean():10.5f} {ql.mean():9.4f} "
              f"{vs_har.mean():+9.2f} {dss.mean()-b_dss.mean():+11.5f} "
              f"{100*(ql.mean()/b_ql.mean()-1):+9.2f} {p['t_like']:+7.2f}")
    print("\n  'vs HAR %' is the product's actual claim, recomputed per arm; "
          "negative = NOCTUA better.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"folds": recs, "summary": report,
                                 "seeds": a.seeds, "hidden": a.hidden},
                                indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
