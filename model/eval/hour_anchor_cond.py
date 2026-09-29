"""
eval/hour_anchor_cond.py
=====================================================================
P4-hour-anchor-cond: does the clock-aware anchor's average gain HIDE damage
where a barrier seller is hurt -- on spike nights?

Lowering sigma at 17:00 tightens every barrier curve. On calm nights that is
the calibration gain P4-hour-anchor-pe-result measured; on the nights that
break out it makes touches MORE likely than the curve says, which is the loss a
seller of barriers actually takes. An average over 2,046 nights can improve
while the 5% that matter get worse.

Same arms and construction as eval/hour_anchor_pe.py (M0s, As, each with
serving's trailing factor from its own forecasts), scored per episode from
run_fold's saved curves (offline battery must match run_fold to 1e-10), and --
new -- every per-episode array is SAVED (hour_anchor_cond.npz) so the next
attack needs no retraining.

Subsets, all defined on the OUTCOME for REPORTING only (neither arm sees
them; both are scored on identical nights): SPIKE = realised vol in the top 5%
of the fold's production test slice, CALM its complement; WEEKEND = the window
touches Saturday or Sunday; per fold.

    python -m model.eval.hour_anchor_cond
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.benchmark import run_fold                                    # noqa: E402
from eval.direction import mean_ci                                     # noqa: E402
from eval.hour_anchor import (FAC_STRIDE_H, PROD_A, PROD_H,            # noqa: E402
                              fold_anchors, served_log_factor)
from eval.product_score import battery, per_episode, touch             # noqa: E402
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.benchmark import BARRIER_U                                   # noqa: E402
from noctua import infer as I                                          # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

EP_METRICS = ("brier", "logs", "pinball", "crps", "brier_far")
SPIKE_Q = 0.95
TOL = 1e-10


def far_brier(cur, M) -> np.ndarray:
    """Per-episode Brier at the FARTHEST barrier on each side -- the rung a
    breakout hits and a calm night never does."""
    k = len(BARRIER_U) - 1
    out = []
    for side in ("up", "dn"):
        p = np.clip(touch(cur[side])[:, k], 1e-6, 1 - 1e-6)
        o = (M[side] >= BARRIER_U[k]).astype(float)
        out.append((p - o) ** 2)
    return np.mean(out, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="hour anchor, conditional")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/hour_anchor_cond.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    ts_all = ep["anchor_ts"].to_numpy(np.int64)
    ah = ep["anchor_hour"].to_numpy()
    wkd = X["cal_weekend_frac"].to_numpy(np.float64)
    at19 = (ep.H == PROD_H).to_numpy()
    fac_hours = {(PROD_A - PROD_H + FAC_STRIDE_H * k) % 24 for k in range(4)}
    fac_mask = at19 & np.isin(ah, sorted(fac_hours))
    w = I.BLEND_W

    store = {k: [] for k in ("idx", "year", "rv", "spike", "weekend")}
    loss = {arm: {m: [] for m in EP_METRICS} for arm in ("M0s", "As")}
    for f in folds:
        anc = fold_anchors(ep, X, f)
        sh = {"M0s": np.zeros(len(ep)),
              "As": np.nan_to_num((1 - w) * (anc["A"] - anc["lhc"]))}
        rF = run_fold(ep, X, f, a.hidden, a.seeds, prod_override=fac_mask)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if rF is None or r0 is None:
            continue
        peF, pe0 = rF["per_episode"], r0["per_episode"]
        ci, ti = pe0["cal_idx"], pe0["test_idx"]
        h_idx = np.concatenate([ci, peF["test_idx"]])
        h_rv = np.concatenate([pe0["rv_cal"], peF["rv"]])
        h_sig = np.concatenate([pe0["sigma_cal"], peF["sigma_med"]])
        got = {}
        for arm in ("M0s", "As"):
            lf = served_log_factor(ts_all[ti], ts_all[h_idx], h_rv,
                                   h_sig * np.exp(sh[arm][h_idx]))
            full = sh[arm].copy(); full[ti] += lf
            r1 = run_fold(ep, X, f, a.hidden, a.seeds,
                          post_shift_fn=lambda mask, _mt, _s=full: _s[mask])
            p1 = r1["per_episode"]
            if not np.array_equal(p1["test_idx"], ti):
                raise SystemExit("REFUSING: arms scored different episodes")
            cur, M = p1["curves"]["noctua_v2"], p1["M_abs"]
            b, b_run = battery(cur, M), barrier_cols(r1["rows"])
            if not max(abs(b[k] - b_run[k]) for k in b_run) < TOL:
                raise SystemExit("REFUSING: offline battery differs from run_fold")
            pe = per_episode(cur, M)
            pe["brier_far"] = far_brier(cur, M)
            got[arm] = pe
        rv = pe0["rv"]
        store["idx"].append(ti); store["year"].append(np.full(len(ti), f["year"]))
        store["rv"].append(rv)
        store["spike"].append(rv >= np.quantile(rv, SPIKE_Q))
        store["weekend"].append(wkd[ti] > 0)
        for arm in ("M0s", "As"):
            for m in EP_METRICS:
                loss[arm][m].append(got[arm][m])
        print(f"  fold {f['year']}: guards ok, {len(ti)} nights", flush=True)

    cat = {k: np.concatenate(v) for k, v in store.items()}
    L = {arm: {m: np.concatenate(v) for m, v in d.items()} for arm, d in loss.items()}
    np.savez_compressed(a.out.with_suffix(".npz"), **cat,
                        **{f"{arm}__{m}": L[arm][m] for arm in L for m in EP_METRICS})

    subsets = {"all": np.ones(len(cat["rv"]), bool), "spike": cat["spike"],
               "calm": ~cat["spike"], "weekend": cat["weekend"],
               "weekday": ~cat["weekend"]}
    for y in sorted(set(cat["year"].tolist())):
        subsets[f"year_{y}"] = cat["year"] == y
    n_family = len(EP_METRICS) * 5            # the five named subsets decide
    alpha = 0.05 / n_family
    out = {"alpha": alpha, "n_family": n_family, "rows": {}}
    print(f"\nper-episode As minus M0s (+ = As better), family {n_family} -> "
          f"{100*(1-alpha):.2f}%, block 38 for pooled subsets\n")
    print(f"{'subset':>10} {'n':>5} " + " ".join(f"{m:>26}" for m in EP_METRICS))
    worse_spike = []
    for name, msk in subsets.items():
        cells = []
        for m in EP_METRICS:
            d = L["M0s"][m][msk] - L["As"][m][msk]
            bl = 38 if msk.sum() > 600 else 1     # thin subsets: nights ~independent
            lo, hi = mean_ci(d, alpha=alpha, block_len=bl)["ci95"]
            pct = 100 * d.mean() / max(L["M0s"][m][msk].mean(), 1e-12)
            out["rows"].setdefault(name, {})[m] = {"n": int(msk.sum()),
                "delta": float(d.mean()), "ci": [float(lo), float(hi)], "pct": float(pct)}
            tag = "+" if lo > 0 else "-" if hi < 0 else " "
            cells.append(f"{pct:+6.2f}% [{lo:+.5f},{hi:+.5f}]{tag}")
            if name == "spike" and hi < 0:
                worse_spike.append(m)
        print(f"{name:>10} {int(msk.sum()):>5} " + " ".join(f"{c:>26}" for c in cells))
    ship_block = bool(worse_spike)
    print(f"\n--- registered rule: SHIP-BLOCKING if As is significantly worse than M0s")
    print(f"    on SPIKE nights on any per-episode metric -> "
          f"{'BLOCKED on ' + str(worse_spike) if ship_block else 'NOT BLOCKED'}")
    out["ship_blocked"] = ship_block
    out["worse_on_spike"] = worse_spike
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out} and {a.out.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
