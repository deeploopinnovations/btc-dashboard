"""
eval/rolling_refresh.py
=====================================================================
Does keeping the training window fresh beat freezing it in 2023?

WHY THIS EXISTS

`eval/forward_split.py` compared a stale training window against one extended
through the post-ETF era, on a common held-out window. It improved QLIKE by
1.03% and discrimination by 2.99% -- and tripped its own pre-registered veto,
because aggregate deep-tail miscalibration worsened by 2.41%, driven entirely
by the 2% barrier. On 585 production episodes neither the gain nor the veto is
resolvable: that is one window, and a 2.4% move in a calibration statistic is
well inside what one window can produce by chance.

So this is the same question with power. Instead of one boundary it walks a
sequence of QUARTERLY test windows through 2025-2026 and, for each, compares:

    frozen     network + calibration fitted once, <= 2023-01-01
               (what actually ships today)
    refreshed  fitted on everything up to that window's own embargoed start
               (what a maintained model would look like)

Both arms are scored on the SAME episodes in each window. The frozen arm's
training set never changes; the refreshed arm's grows. The difference is
exactly the value of not letting the fit go stale, measured repeatedly rather
than once.

WHY QUARTERLY RATHER THAN ANNUAL

There are only ~1.5 years of post-2025 data, so annual windows would give two
comparisons. Quarters give six. Each window holds ~90 production episodes,
which is small -- but six small paired windows support a sign test where one
window of 585 supports nothing, and the sign test is what the decision rule
uses.

DECISION RULE, fixed before running: the refresh is worth adopting if it wins
QLIKE in >= 5 of 6 windows AND does not worsen aggregate deep-tail (alpha <=
2%) miscalibration in more than 2 of 6. Average-case wins bought with tail
degradation stay rejected -- the veto that fired in forward_split.py is kept,
just measured with enough windows to mean something.

    python -m model.eval.rolling_refresh
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

from eval.benchmark import run_fold                                       # noqa: E402
from eval.efficiency import summarise                                     # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402

HOUR = 3600
TAIL_BARRIERS = (0.5, 1.0, 2.0)


def causal_sigma_fn(ep, X):
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def masks(ep, train_end: str, calib_end: str, win_lo: str, win_hi: str,
          seed: int = 0) -> dict:
    """Embargoed train/calib plus a bounded test WINDOW (not an open tail)."""
    emb = int(ep["H"].max()) * HOUR
    ts = ep["anchor_ts"].to_numpy(np.int64)
    end = ts + ep["H"].to_numpy() * HOUR
    base = S.in_sample_mask(ep)
    t1 = int(pd.Timestamp(train_end, tz="UTC").timestamp())
    t2 = int(pd.Timestamp(calib_end, tz="UTC").timestamp())
    lo = int(pd.Timestamp(win_lo, tz="UTC").timestamp())
    hi = int(pd.Timestamp(win_hi, tz="UTC").timestamp())
    # `year` is consumed downstream as a numpy seed (ShuffledNoctua), so it
    # must be an int -- a date string raises inside SeedSequence.
    return {"year": int(seed),
            "train": base & (end <= t1 - emb),
            "calib": base & (ts >= t1) & (end <= t2 - emb),
            "test":  base & (ts >= lo) & (ts < hi)}


def tail_mcb(rows) -> float:
    r = next(x for x in rows if x["model"] == "noctua_v2")
    return float(sum(r[f"MCB_{s}_{p}"] for s in ("up", "dn") for p in TAIL_BARRIERS))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is a maintained fit worth it?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/rolling_refresh.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    sig_fn = causal_sigma_fn(ep, X)
    prod = S.production_mask(ep)

    # quarterly test windows through the unseen era
    qs = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01",
          "2026-01-01", "2026-04-01", "2026-07-01"]
    recs = []
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        # the refreshed arm calibrates on the two quarters before the window
        # and trains on everything before that; the frozen arm never moves.
        cal_start = qs[i - 2] if i >= 2 else "2024-07-01"
        arms = {
            "frozen":    masks(ep, "2023-01-01", "2024-07-01", lo, hi, seed=i),
            "refreshed": masks(ep, cal_start, lo, lo, hi, seed=i),
        }
        n_te = int((arms["frozen"]["test"] & prod).sum())
        if n_te < 40:
            print(f"  {lo}: SKIPPED (only {n_te} production episodes)")
            continue
        if not np.array_equal(arms["frozen"]["test"], arms["refreshed"]["test"]):
            raise SystemExit("REFUSING: arms differ on the test window")
        line = {"window": lo, "n_prod": n_te}
        for nm, f in arms.items():
            t0 = time.time()
            out = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                           sigma_ref_fn=sig_fn, min_train=5000)
            if out is None:
                print(f"  {lo}  {nm:10} SKIPPED (train/calib too small)")
                continue
            s = summarise(out["rows"])
            s["qlike"] = out["vol"]["noctua"]
            s["tail_mcb"] = tail_mcb(out["rows"])
            s["n_train"] = int(f["train"].sum())
            line[nm] = s
            print(f"  {lo}  {nm:10} n_tr={s['n_train']:7,} n_te={n_te:3}  "
                  f"QLIKE {s['qlike']:.4f}  DSC/UNC {s['DSS']:.5f}  "
                  f"tailMCB {s['tail_mcb']:.5f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if "frozen" in line and "refreshed" in line:
            recs.append(line)

    if not recs:
        print("no window produced both arms")
        return 1

    q_win = sum(1 for r in recs if r["refreshed"]["qlike"] < r["frozen"]["qlike"])
    d_win = sum(1 for r in recs if r["refreshed"]["DSS"] > r["frozen"]["DSS"])
    t_bad = sum(1 for r in recs
                if r["refreshed"]["tail_mcb"] > r["frozen"]["tail_mcb"])
    n = len(recs)
    print(f"\n{'metric':>12} {'frozen':>11} {'refreshed':>11} {'change':>10} {'wins':>7}")
    for k, lower_better in (("qlike", True), ("DSS", False),
                            ("pinball", True), ("crps", True),
                            ("tail_mcb", True)):
        f_ = np.mean([r["frozen"][k] for r in recs])
        r_ = np.mean([r["refreshed"][k] for r in recs])
        w = sum(1 for r in recs
                if (r["refreshed"][k] < r["frozen"][k]) == lower_better)
        print(f"{k:>12} {f_:11.6f} {r_:11.6f} {100*(r_/f_-1):+9.2f}% {w:>4}/{n}")

    ok = (q_win >= 5) and (t_bad <= 2)
    print(f"\n  QLIKE wins {q_win}/{n} (need >= 5); tail MCB worse in "
          f"{t_bad}/{n} (allowed <= 2)")
    print(f"  -> pre-registered rule: {'ADOPT' if ok else 'DO NOT ADOPT'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"windows": recs, "qlike_wins": q_win,
                                 "dsc_wins": d_win, "tail_worse": t_bad,
                                 "n_windows": n, "adopt": bool(ok)},
                                indent=2, default=float) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
