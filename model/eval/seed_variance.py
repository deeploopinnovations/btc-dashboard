"""
eval/seed_variance.py
=====================================================================
P2-seed-variance: how big is the training variance the paired bootstrap omits?

WHY THIS EXISTS

`P2-event-window-result` produced a matched pair that cannot both be right:

    H=1   E1 vs C2  +0.00136  (+0.19% of base)  CI [+0.00086, +0.00186]
    H=24  E1 vs C2  -0.00037  (-0.14% of base)  CI [-0.00083, -0.00008]

Both exclude zero. But at H=24 there is no intraday footprint at all --
`P2-event-footprint` measured chi-square 3.0, p = 1.000 -- so the two columns
there are arbitrary 4-of-24-hour partitions carrying nothing. An estimator that
finds a significant effect in provable noise is measuring itself.

R51 was written from that pair, and R51 is currently an ARGUMENT: a
moving-block bootstrap resamples EPISODES while holding both fitted networks
fixed, so it answers "is the difference between these two fits consistent
across episodes", never "would a re-run agree". This file converts the argument
into a number.

THE MEASUREMENT

Re-run ONLY E1 and C2, unchanged in every other respect, with THREE INDEPENDENT
SEED SETS (0-2, 3-5, 6-8). Each is a complete 3-seed ensemble exactly as the
original ran -- not one wider ensemble, which would average the quantity away
rather than measure it.

    PRIMARY = across-seed-set sd of the gap / bootstrap CI half-width

Near zero: the bootstrap was adequate and R51 is overstated. Near or above one:
the omitted component is as large as the reported uncertainty, and every paired
comparison in this project between separately-trained arms has been quoted with
an interval that is too narrow.

    python -m model.eval.seed_variance
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.event_window import eastern_hour, run_arm                          # noqa: E402
from eval.vol_matrix import build_h4_table, qlike_vec                        # noqa: E402
from noctua import splits as S                                              # noqa: E402

ARMS = ("E1", "C2")
SEED_SETS = (0, 3, 6)
# the realised bootstrap CI half-widths from P2-event-window-result
BOOT_HALFWIDTH = {1: (0.00186 - 0.00086) / 2, 24: (0.00083 - 0.00008) / 2}
OBSERVED_GAP = {1: +0.00136, 24: -0.00037}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-seed-variance")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/seed_variance.json"))
    a = ap.parse_args(argv)

    ep, X = build_h4_table(a.artifacts)
    ts = ep.anchor_ts.to_numpy(np.int64)
    et_h, utc_h = eastern_hour(ts), ((ts // 3600) % 24).astype(int)
    folds = S.walk_forward_folds(ep)

    print("P2-seed-variance   E1 vs C2, three independent 3-seed ensembles")
    print(f"seed sets {SEED_SETS}   arms {ARMS}\n")
    out = {"seed_sets": list(SEED_SETS), "horizons": {}}

    for H in (1, 24):
        tag = "the effect" if H == 1 else "the artifact (footprint absent)"
        gaps = []
        for s0 in SEED_SETS:
            acc = {k: [] for k in ARMS}
            for f in folds:
                t0 = time.time()
                rng = np.random.default_rng(5000 + f["year"])
                for arm in ARMS:
                    r = run_arm(ep, X, f, H, arm, et_h, utc_h, a.hidden,
                                a.seeds, rng, seed0=s0)
                    if r is not None:
                        acc[arm].append(r)
                print(f"  H={H:>3} seeds {s0}-{s0+2} fold {f['year']}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if not acc["E1"] or not acc["C2"]:
                continue
            rv = np.concatenate([x["rv"] for x in acc["E1"]])
            q1 = qlike_vec(rv, np.concatenate([x["sigma"] for x in acc["E1"]]))
            q2 = qlike_vec(rv, np.concatenate([x["sigma"] for x in acc["C2"]]))
            gaps.append(float(np.nanmean(q2 - q1)))
            print(f"    -> gap {gaps[-1]:+.5f}", flush=True)

        if not gaps:
            continue
        g = np.array(gaps)
        sd = float(g.std(ddof=1)) if len(g) > 1 else float("nan")
        hw = BOOT_HALFWIDTH[H]
        ratio = sd / hw if hw else float("nan")
        same_sign = bool(np.all(g > 0) or np.all(g < 0))
        print("\n" + "=" * 88)
        print(f"H = {H}h   [{tag}]")
        print("=" * 88)
        print(f"  gap per seed set   " + "  ".join(f"{v:+.5f}" for v in g))
        print(f"  mean {g.mean():+.5f}   across-seed-set sd {sd:.5f}   "
              f"all same sign: {same_sign}")
        print(f"  original run       gap {OBSERVED_GAP[H]:+.5f}   "
              f"bootstrap CI half-width {hw:.5f}")
        print(f"  PRIMARY  sd / half-width = {ratio:.2f}"
              + ("   <- the omitted variance is the larger one"
                 if ratio >= 1 else "   <- the bootstrap was the larger one"))
        out["horizons"][str(H)] = {
            "gaps": [float(v) for v in g], "mean": float(g.mean()),
            "sd_across_seed_sets": sd, "boot_half_width": hw,
            "ratio": float(ratio), "all_same_sign": same_sign,
            "original_gap": OBSERVED_GAP[H]}
        print()

    h1, h24 = out["horizons"].get("1"), out["horizons"].get("24")
    if h1 and h24:
        restored = (h1["all_same_sign"]
                    and min(abs(np.array(h1["gaps"]))) > h1["sd_across_seed_sets"]
                    and not h24["all_same_sign"])
        print("=" * 88)
        print("PRE-REGISTERED DECISION on the withdrawn H=1 headline:")
        print(f"  all three H=1 gaps same sign      {h1['all_same_sign']}")
        print(f"  smallest |gap| > seed-set sd      "
              f"{min(abs(np.array(h1['gaps']))) > h1['sd_across_seed_sets']}")
        print(f"  H=24 gaps NOT all same sign       {not h24['all_same_sign']}")
        print(f"\n  {'RESTORED' if restored else 'STAYS BURIED'}")
        out["headline_restored"] = bool(restored)

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
