"""
eval/levers.py
=====================================================================
Does upweighting spike episodes in training survive a proper walk-forward?

THE PRE-REGISTERED RULE (BENCHMARK.md §12, fixed before this ran)

    adopt a lever only if, on the standard 6-fold walk-forward at 3 seeds, it
    improves pooled QLIKE in >= 5 of 6 folds AND improves spike-episode
    RV/sigma toward 1.0 AND worsens calm-episode QLIKE by no more than 3%

The third condition is the one that can veto a headline win, and it is there on
purpose: an option seller who stops being able to trust calm nights has not been
helped by a model that got better at wild ones.

WHY THIS NEEDED A PROPER TEST

§12 measured, on a single seed over the served population, that upweighting
high-volatility training episodes 3x moves pooled QLIKE -5.6%, spike QLIKE
-25.9%, and spike RV/sigma from 1.4643 toward 1.3484, at a cost of +1.7% on
calm nights. That is a large and plausible effect. It is also one seed on one
population, and §10 of this document records a finding just as clean that
REVERSED SIGN on 24x the data. So it does not get adopted on that basis.

WHAT IS BEING WEIGHTED, AND WHY IT IS NOT LOOK-AHEAD

The spike flag is `RV above the trailing 180-day 95th percentile of production
RV, computed from strictly prior days`. It is applied only to TRAINING
episodes, whose outcomes are known by construction -- this is class weighting,
not a feature. Nothing at inference time sees it. The trailing percentile is
still built causally so that the flag means the same thing in every fold rather
than drifting with the sample.

`run_fold`'s `extra_w` hook does the weighting, so the network, committee,
calibration, embargo and scoring are byte-identical to the headline benchmark
in every arm. The only difference between arms is one multiplier.

WHAT A NEGATIVE RESULT LOOKS LIKE

Either pooled QLIKE fails to clear 5/6 -- the single-seed gain was noise -- or
calm-episode QLIKE degrades past 3%, meaning the lever buys spike accuracy by
making the other 92% of nights worse. Both are useful, and the second is the
more likely trap: the loss is dominated by spike episodes, so a weighting that
sacrifices calm nights can still look good pooled.

    python -m model.eval.levers --weights 3.0 8.0
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
from eval.direction import block_bootstrap_ci                             # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402
from research import pitfalls as P                                        # noqa: E402

TRAIL_DAYS = 180
SPIKE_Q = 0.95


def causal_spike_flag(ep) -> np.ndarray:
    """Trailing 180-day 95th percentile of production RV, strictly prior days.

    Same definition as eval/anatomy.py's, so the populations these numbers
    describe are the ones §7a and §12 describe. `shift(1)` before the rolling
    window is what makes it strictly prior; without it an episode helps set the
    threshold it is then compared against.
    """
    prod = S.production_mask(ep)
    ts = pd.to_datetime(ep["anchor_ts"], unit="s", utc=True)
    df = pd.DataFrame({"ts": ts, "rv": ep["RV"].to_numpy(np.float64),
                       "prod": prod}).sort_values("ts")
    p = df[df["prod"]].copy()
    thr = p["rv"].shift(1).rolling(TRAIL_DAYS, min_periods=60).quantile(SPIKE_Q)
    p["spike"] = (p["rv"] > thr).fillna(False)
    # broadcast the daily threshold back to every episode by timestamp order
    df = df.merge(p[["ts", "spike"]], on="ts", how="left")
    df["spike"] = df["spike"].ffill().fillna(False)
    out = np.zeros(len(ep), dtype=bool)
    out[df.index.to_numpy()] = df["spike"].to_numpy(bool)
    return out


def causal_sigma_fn(ep, X):
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)

    def fn(train_mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(raw[train_mask], [0.005, 0.995])
        return np.maximum(np.clip(raw, lo, hi), 1e-12)

    return fn


def qlike(rv, sig) -> np.ndarray:
    pv = np.maximum(sig, 1e-12) ** 2
    r = np.maximum(rv ** 2, 1e-18) / np.maximum(pv, 1e-18)
    return r - np.log(r) - 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spike upweighting, walk-forward")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--weights", type=float, nargs="+", default=[3.0, 8.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/levers.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    spike = causal_spike_flag(ep)
    sig_fn = causal_sigma_fn(ep, X)
    folds = S.walk_forward_folds(ep)
    print(f"spike episodes: {spike.sum():,} of {len(ep):,} "
          f"({100*spike.mean():.2f}%)   folds: {len(folds)}   seeds: {a.seeds}\n")

    arms = {"1.0x_control": 1.0}
    for w in a.weights:
        arms[f"{w:g}x"] = w

    recs = []
    for f in folds:
        line = {"year": f["year"]}
        for nm, w in arms.items():
            t0 = time.time()
            extra = np.where(spike, w, 1.0) if w != 1.0 else None
            r = run_fold(ep, X, f, hidden=a.hidden, seeds=a.seeds,
                         sigma_ref_fn=sig_fn, extra_w=extra)
            if r is None:
                print(f"  {f['year']}  {nm:12} SKIPPED")
                continue
            pe = r["per_episode"]
            rv, sg, idx = pe["rv"], pe["sigma_med"], pe["test_idx"]
            q = qlike(rv, sg)
            sp = spike[idx]
            ratio = rv / np.maximum(sg, 1e-12)
            line[nm] = {
                "qlike_pooled": float(q.mean()),
                "qlike_spike": float(q[sp].mean()) if sp.any() else float("nan"),
                "qlike_calm": float(q[~sp].mean()),
                "rv_sigma_spike": float(np.median(ratio[sp])) if sp.any() else float("nan"),
                "rv_sigma_calm": float(np.median(ratio[~sp])),
                "n_test": int(len(idx)), "n_spike": int(sp.sum()),
                "dsc_unc": float(next(x for x in r["rows"]
                                      if x["model"] == "noctua_v2")["DSC_up_2.0"]
                                 / max(next(x for x in r["rows"]
                                            if x["model"] == "noctua_v2")["UNC_up_2.0"], 1e-12)),
            }
            s = line[nm]
            print(f"  {f['year']}  {nm:12} pooled {s['qlike_pooled']:.4f}  "
                  f"spike {s['qlike_spike']:.4f}  calm {s['qlike_calm']:.4f}  "
                  f"RV/sig(spike) {s['rv_sigma_spike']:.4f}  "
                  f"n_sp={s['n_spike']:3d}  ({time.time()-t0:.0f}s)", flush=True)
        if all(k in line for k in arms):
            recs.append(line)

    if not recs:
        print("no fold produced every arm")
        return 1

    print(f"\n{'arm':>13} {'pooled':>9} {'spike':>9} {'calm':>9} {'RV/sig sp':>10} "
          f"{'pooled wins':>12} {'calm cost':>10}")
    ctl = "1.0x_control"
    report = {}
    for nm in arms:
        pooled = np.array([r[nm]["qlike_pooled"] for r in recs])
        spk = np.array([r[nm]["qlike_spike"] for r in recs])
        calm = np.array([r[nm]["qlike_calm"] for r in recs])
        rs = np.array([r[nm]["rv_sigma_spike"] for r in recs])
        c_pool = np.array([r[ctl]["qlike_pooled"] for r in recs])
        c_calm = np.array([r[ctl]["qlike_calm"] for r in recs])
        c_rs = np.array([r[ctl]["rv_sigma_spike"] for r in recs])
        wins = int((pooled < c_pool).sum())
        calm_cost = 100.0 * (np.nanmean(calm) / np.nanmean(c_calm) - 1.0)
        toward_one = bool(np.nanmean(np.abs(rs - 1.0)) < np.nanmean(np.abs(c_rs - 1.0)))
        report[nm] = {"qlike_pooled": float(np.nanmean(pooled)),
                      "qlike_spike": float(np.nanmean(spk)),
                      "qlike_calm": float(np.nanmean(calm)),
                      "rv_sigma_spike": float(np.nanmean(rs)),
                      "pooled_wins": wins, "n_folds": len(recs),
                      "calm_cost_pct": float(calm_cost),
                      "rv_sigma_toward_one": toward_one}
        print(f"{nm:>13} {np.nanmean(pooled):9.4f} {np.nanmean(spk):9.4f} "
              f"{np.nanmean(calm):9.4f} {np.nanmean(rs):10.4f} "
              f"{wins:>8}/{len(recs)} {calm_cost:+9.2f}%")

    print(f"\n--- pre-registered rule (§12): pooled QLIKE >= 5/6 folds AND "
          f"RV/sigma toward 1.0 AND calm cost <= 3% ---")
    verdicts = {}
    for nm in arms:
        if nm == ctl:
            continue
        r = report[nm]
        need = int(np.ceil(5 / 6 * r["n_folds"]))
        c1 = r["pooled_wins"] >= need
        c2 = r["rv_sigma_toward_one"]
        c3 = r["calm_cost_pct"] <= 3.0
        ok = c1 and c2 and c3
        verdicts[nm] = {"pooled_ok": c1, "rv_sigma_ok": c2, "calm_ok": c3,
                        "adopt": ok, "need_wins": need}
        print(f"  {nm:>8}: pooled {r['pooled_wins']}/{r['n_folds']} (need {need}) "
              f"{'OK' if c1 else 'FAIL'} | RV/sigma toward 1.0 "
              f"{'OK' if c2 else 'FAIL'} | calm cost {r['calm_cost_pct']:+.2f}% "
              f"{'OK' if c3 else 'FAIL'}  ->  {'ADOPT' if ok else 'DO NOT ADOPT'}")

    # execution feedback: run the catalogued checks against this experiment
    print("\n--- research/pitfalls checks on this experiment ---")
    rep = P.Report()
    rep.add(P.check_rule_satisfiable(int(np.ceil(5/6*len(recs))), len(recs), "folds"))
    d_pool = np.array([recs[i][list(arms)[1]]["qlike_pooled"]
                       - recs[i][ctl]["qlike_pooled"] for i in range(len(recs))])
    rep.add(P.check_not_a_coin_flip(d_pool, "pooled QLIKE delta"))
    rep.add(P.check_arms_matched({k: len(recs) for k in arms}, what="folds scored"))
    print(rep.render())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"folds": recs, "summary": report,
                                 "verdicts": verdicts, "seeds": a.seeds,
                                 "spike_pct": float(100*spike.mean())},
                                indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
