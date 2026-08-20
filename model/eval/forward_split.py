"""
eval/forward_split.py
=====================================================================
Does letting the model see the post-ETF regime actually fix it?

THE PROBLEM, RESTATED FROM 6i

100% of the shipped model's fitted weight comes from before 2023-01-01, and
the market changed in January 2024: median 19-hour realized volatility fell
from 2.975% to 1.736% (ratio 0.584, Mann-Whitney p = 3.1e-156), with return
kurtosis collapsing from 15.04 to 0.41. Every volatility-scale feature is
served on a distribution 25-40% narrower than the one its weights were fitted
against (har_22d KS 0.614). The model over-forecasts, and `serve/adaptive.py`
has been holding production together with a nightly 0.93-0.96 shrink.

The obvious fix is to move the training window forward. This measures whether
that works, rather than assuming it.

THE DESIGN, AND THE ONE THING IT MUST NOT GET WRONG

Both arms are scored on **exactly the same held-out window** -- production
anchors from 2025-01-01 onward, which neither arm has seen in training or
calibration. Without that, "the model trained on more recent data does better
on more recent data" is unfalsifiable: a later training cutoff also moves the
test period, and the comparison would be measuring the easiness of the test
years rather than the value of the extra data.

    arm            network + calibration fitted on      tested on
    shipped        <= 2023-01-01, calib -> 2024-07-01    2025-01-01 ->
    forward        <= 2024-07-01, calib -> 2025-01-01    2025-01-01 ->

The forward arm's training window therefore contains roughly six months of
post-ETF episodes (2024-01-11 to 2024-07-01) and its calibration slice another
six, while the shipped arm contains none. Everything downstream -- seeds,
committee, causal stage-B reference, embargo -- is identical.

Embargoes are applied at both internal boundaries at `max(H)` hours, as
`splits.time_splits` does, so no training episode's forward window can overlap
a calibration or test episode.

WHAT WOULD MAKE THIS A FAILURE

More data from the right regime is not automatically better: the forward arm
trains on ~30% fewer episodes (it gains 18 months at the recent end but the
900-day half-life was already concentrating weight there, and the extra recent
data is drawn from a *narrower* distribution, which can shrink the model's
effective support in the tails an option seller cares about most). A model
that has never seen a 20% night may not extrapolate to one.

So the tail is scored separately, not just the average.

DECISION RULE, fixed before running: adopt the forward split if it improves
QLIKE **and** does not worsen deep-tail barrier calibration (alpha <= 2%) on
the common window. An average-case win bought with tail degradation is the
wrong trade for a seller and gets rejected.

    python -m model.eval.forward_split
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

from eval.benchmark import BARRIER_U, run_fold                            # noqa: E402
from eval.efficiency import summarise                                     # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402

HOUR = 3600


def build_split(ep, train_end: str, calib_end: str, test_start: str) -> dict:
    """Train/calib/test masks with the same embargo rule as `time_splits`."""
    emb = int(ep["H"].max()) * HOUR
    ts = ep["anchor_ts"].to_numpy(np.int64)
    end = ts + ep["H"].to_numpy() * HOUR
    base = S.in_sample_mask(ep)
    t1 = int(pd.Timestamp(train_end, tz="UTC").timestamp())
    t2 = int(pd.Timestamp(calib_end, tz="UTC").timestamp())
    t3 = int(pd.Timestamp(test_start, tz="UTC").timestamp())
    return {"year": 0,
            "train": base & (end <= t1 - emb),
            "calib": base & (ts >= t1) & (end <= t2 - emb),
            "test":  base & (ts >= t3)}


def causal_sigma_fn(ep, X):
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train through the post-ETF era?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/forward_split.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    sig_fn = causal_sigma_fn(ep, X)
    ETF = int(pd.Timestamp("2024-01-11", tz="UTC").timestamp())
    ts = ep["anchor_ts"].to_numpy(np.int64)

    arms = {
        "shipped_split": build_split(ep, "2023-01-01", "2024-07-01", a.test_start),
        "forward_split": build_split(ep, "2024-07-01", "2025-01-01", a.test_start),
    }
    for nm, f in arms.items():
        n_post = int((f["train"] & (ts >= ETF)).sum())
        print(f"  {nm:14} train {f['train'].sum():7,} "
              f"({n_post:,} post-ETF)  calib {f['calib'].sum():6,}  "
              f"test {f['test'].sum():6,}")
    same_test = np.array_equal(arms["shipped_split"]["test"],
                               arms["forward_split"]["test"])
    print(f"  identical test masks: {same_test}")
    if not same_test:
        raise SystemExit("REFUSING: arms must be scored on the same episodes")
    prod_test = arms["shipped_split"]["test"] & S.production_mask(ep)
    print(f"  production episodes in the common window: {prod_test.sum():,}\n")

    out = {}
    for nm, f in arms.items():
        t0 = time.time()
        r = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds, sigma_ref_fn=sig_fn)
        if r is None:
            print(f"  {nm}: SKIPPED (insufficient data)")
            continue
        s = summarise(r["rows"])
        s["qlike"] = r["vol"]["noctua"]
        s["qlike_har"] = r["vol"]["log_har"]
        # per-barrier calibration error, so the tail can be judged separately
        row = next(x for x in r["rows"] if x["model"] == "noctua_v2")
        s["barrier_cal"] = {
            f"{p}%": {"up_MCB": row[f"MCB_up_{p}"], "dn_MCB": row[f"MCB_dn_{p}"],
                      "up_DSC": row[f"DSC_up_{p}"], "dn_DSC": row[f"DSC_dn_{p}"]}
            for p in (0.5, 1.0, 2.0, 3.0, 5.0)}
        out[nm] = s
        print(f"  {nm:14} DSC/UNC {s['DSS']:.5f}  pinball {s['pinball']:.6f}  "
              f"CRPS {s['crps']:.6f}  QLIKE {s['qlike']:.4f} "
              f"(HAR {s['qlike_har']:.4f})  ({time.time()-t0:.0f}s)", flush=True)

    if len(out) == 2:
        sh, fw = out["shipped_split"], out["forward_split"]
        print(f"\n{'metric':>12} {'shipped':>11} {'forward':>11} {'change':>11}")
        for k, better_low in (("DSS", False), ("pinball", True),
                              ("crps", True), ("qlike", True)):
            d = fw[k] - sh[k]
            good = (d < 0) if better_low else (d > 0)
            print(f"{k:>12} {sh[k]:11.6f} {fw[k]:11.6f} {d:+11.6f}  "
                  f"{'BETTER' if good else 'worse'}")
        print(f"\n  deep-tail calibration (MCB, lower better) -- the veto condition:")
        print(f"  {'barrier':>8} {'shipped up/dn':>22} {'forward up/dn':>22}")
        for p in ("0.5%", "1.0%", "2.0%"):
            s_, f_ = sh["barrier_cal"][p], fw["barrier_cal"][p]
            print(f"  {p:>8} {s_['up_MCB']:10.6f}/{s_['dn_MCB']:10.6f} "
                  f"{f_['up_MCB']:10.6f}/{f_['dn_MCB']:10.6f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"test_start": a.test_start, "arms": out,
                                 "seeds": a.seeds, "hidden": a.hidden},
                                indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    print("Decision rule fixed before the run: adopt only if QLIKE improves AND "
          "deep-tail (alpha <= 2%) calibration does not worsen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
