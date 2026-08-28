"""
eval/teacher_scorecard.py
=====================================================================
The per-horizon teacher scorecard. TEACHER_ZOO section 5.

THERE IS NO UNIVERSAL WINNER AND NO COMPOSITE SCORE

Phase 1 already demonstrated why a single number would destroy the thing
Phase 2 needs: at H = 168 GARCH-t has the BEST spike QLIKE of any arm (0.1917)
and the WORST calm QLIKE (0.5565). Averaging those into one figure of merit
would report a mediocre model and hide the fact that it is the best tail
forecaster in the zoo -- which is exactly the kind of mechanism the mining is
looking for.

So every metric is reported side by side, per horizon, and where they disagree
THE DISAGREEMENT IS THE FINDING.

SELECTION IS ON CALIB. REPORTING IS ON TEST. THEY ARE NEVER MIXED.

Every "best teacher" statement here means best by CALIBRATION QLIKE. The
test-side ranking is printed beside it as a separate, explicitly
non-actionable observation, because Phase 1 measured that the two disagree: at
H = 1 and H = 6 the best teacher on test is garch_t while calib chooses
persistence and har_short. Selecting on test would import that gap into every
downstream arm, which would then be scored on the data that chose it.

The columns:

    pooled QLIKE    the headline loss
    spike QLIKE     top 5% of realized vol, by the test slice's own quantile
    calm QLIKE      the other 95%
    deep tail       top 1% -- separated from `spike` because the mechanisms
                    that help at the 95th percentile and at the 99th are not
                    obviously the same one
    calib ratio     mean(RV^2 / sigma^2). 1.0 means the forecast variance
                    equals the average realized variance. Above 1 means the
                    forecast was too LOW on average -- the overlay measured
                    exactly this and it is the E-scale signature.
    worst fold      the maximum per-fold QLIKE, because a teacher that is good
                    on average and catastrophic in one regime is not a teacher
    cost            measured, not estimated

    python -m model.eval.teacher_scorecard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.vol_matrix import qlike_vec                                        # noqa: E402

HORIZONS = (1, 6, 24, 168)
YEARS = (2021, 2022, 2023, 2024, 2025, 2026)


def load_oof(path: Path):
    z = np.load(path)
    keys = list(z.keys())
    teachers = sorted({k.rsplit("/", 1)[1] for k in keys if "/sigma/" in k})
    return z, teachers


def gather(z, H: int, sl: str, teacher: str):
    """Concatenate one teacher's predictions across folds for one horizon."""
    rv, sig, fold_q, years = [], [], [], []
    for y in YEARS:
        pre = f"{y}/{H}/{sl}"
        k = f"{pre}/sigma/{teacher}"
        if k not in z:
            continue
        r = z[f"{pre}/rv"]
        s = z[k]
        ok = np.isfinite(s) & (s > 0)
        if ok.mean() < 0.95:
            continue
        s = np.where(ok, s, np.nan)
        rv.append(r); sig.append(s)
        fold_q.append(float(np.nanmean(qlike_vec(r, s))))
        years.append(y)
    if not rv:
        return None
    rv = np.concatenate(rv); sig = np.concatenate(sig)
    return {"rv": rv, "sigma": sig, "q": qlike_vec(rv, sig),
            "per_fold": fold_q, "years": years}


def metrics(d: dict) -> dict:
    rv, q, sig = d["rv"], d["q"], d["sigma"]
    hi5 = rv >= np.quantile(rv, 0.95)
    hi1 = rv >= np.quantile(rv, 0.99)
    ratio = np.nanmean(rv ** 2 / np.maximum(sig, 1e-12) ** 2)
    return {
        "pooled": float(np.nanmean(q)),
        "spike": float(np.nanmean(q[hi5])),
        "calm": float(np.nanmean(q[~hi5])),
        "deep_tail": float(np.nanmean(q[hi1])),
        "calib_ratio": float(ratio),
        "worst_fold": float(max(d["per_fold"])),
        "per_fold": d["per_fold"],
        "years": d["years"],
        "n": int(np.isfinite(q).sum()),
    }


def measure_cost(artifacts: Path) -> dict:
    """Wall-clock per fold, measured here rather than guessed.

    OLS families are refitted and timed directly. GARCH is timed on a cold fit
    including its nine multi-starts. NOCTUA's cost is read from the run that
    produced the OOF artifact rather than re-measured, and is labelled as such.
    """
    import pandas as pd

    from eval.vol_matrix import build_h4_table
    from noctua import baselines as B
    from noctua import splits as S

    ep, X = build_h4_table(artifacts)
    yall = B.har_target(ep.RV.to_numpy(), ep.H.to_numpy(np.float64))
    f = S.walk_forward_folds(ep)[-1]
    m = f["train"] & (ep.H == 24).to_numpy()
    w = S.sample_weights(ep, m)
    t0 = time.time(); B.fit_vol_baselines(X[m], yall[m], w)
    ols = time.time() - t0

    garch = None
    try:
        from eval.garch import _FIT_CACHE, fit_and_forecast, hourly_returns
        ret = hourly_returns(artifacts)
        _FIT_CACHE.clear()
        a = ep.anchor_ts.to_numpy(np.int64)[m][:500]
        t0 = time.time()
        fit_and_forecast(ret, int(ep.anchor_ts.to_numpy()[f["train"]].max()),
                         a, np.full(len(a), 24.0), dist="t", verbose=False)
        garch = time.time() - t0
    except Exception as e:                                        # noqa: BLE001
        garch = None
        print(f"  (GARCH cost not measured: {e})")

    return {"ols_family_fit_s": round(ols, 3),
            "garch_t_cold_fit_s": None if garch is None else round(garch, 2),
            "noctua_v1_fold_s": "42-70 (from the teacher_zoo run that produced the OOF artifact)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="per-horizon teacher scorecard")
    ap.add_argument("--oof", type=Path, default=Path("model/artifacts/teacher_oof.npz"))
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--no-cost", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/teacher_scorecard.json"))
    a = ap.parse_args(argv)

    z, teachers = load_oof(a.oof)
    side = json.loads(a.oof.with_suffix(".json").read_text())
    print(f"OOF artifact sha256 {side['sha256'][:16]}…  teachers: {len(teachers)}")
    print(f"slices in the artifact: {side['emitted_slices']} "
          f"(train is absent by construction)\n")

    cost = {} if a.no_cost else measure_cost(a.artifacts)
    out = {"oof_sha256": side["sha256"], "cost": cost, "horizons": {}}

    for H in HORIZONS:
        cal, tst = {}, {}
        for t in teachers:
            dc, dt = gather(z, H, "calib", t), gather(z, H, "test", t)
            if dc is not None:
                cal[t] = metrics(dc)
            if dt is not None:
                tst[t] = metrics(dt)
        if not cal or not tst:
            continue
        best_calib = min(cal, key=lambda k: cal[k]["pooled"])
        best_test = min(tst, key=lambda k: tst[k]["pooled"])

        print("=" * 104)
        print(f"H = {H}h    selection on CALIB → **{best_calib}**"
              + (f"    (test-best would have been {best_test} — "
                 f"NOT actionable, reported for contrast)"
                 if best_test != best_calib else
                 "    (test agrees, which is a coincidence and not a licence)"))
        print("=" * 104)
        print(f"{'teacher':>14} {'calibQ':>8} │ {'pooled':>8} {'spike':>8} {'calm':>8} "
              f"{'tail1%':>8} {'RV²/σ²':>8} {'worst':>8}   per-fold")
        order = sorted(tst, key=lambda k: cal.get(k, {}).get("pooled", 9e9))
        for t in order:
            m = tst[t]
            cq = cal.get(t, {}).get("pooled", float("nan"))
            mark = " ←" if t == best_calib else ""
            print(f"{t:>14} {cq:8.5f} │ {m['pooled']:8.5f} {m['spike']:8.4f} "
                  f"{m['calm']:8.5f} {m['deep_tail']:8.4f} {m['calib_ratio']:8.3f} "
                  f"{m['worst_fold']:8.5f}   "
                  + " ".join(f"{v:.3f}" for v in m["per_fold"]) + mark)

        # where the metrics disagree -- the point of not compositing
        winners = {
            "pooled": min(tst, key=lambda k: tst[k]["pooled"]),
            "spike": min(tst, key=lambda k: tst[k]["spike"]),
            "calm": min(tst, key=lambda k: tst[k]["calm"]),
            "deep_tail": min(tst, key=lambda k: tst[k]["deep_tail"]),
            "worst_fold": min(tst, key=lambda k: tst[k]["worst_fold"]),
            "calibration": min(tst, key=lambda k: abs(tst[k]["calib_ratio"] - 1.0)),
        }
        uniq = sorted(set(winners.values()))
        print(f"\n   metric winners (test): "
              + " · ".join(f"{k}={v}" for k, v in winners.items()))
        print(f"   {len(uniq)} distinct winner(s) across 6 metrics"
              + ("  — the disagreement IS the finding, and is why nothing is composited"
                 if len(uniq) > 1 else "  — the metrics agree here"))
        print()
        out["horizons"][str(H)] = {
            "best_by_calib": best_calib, "best_by_test": best_test,
            "metric_winners": winners, "n_distinct_winners": len(uniq),
            "calib": cal, "test": tst,
        }

    a.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {a.out}")
    if cost:
        print(f"cost: {cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
