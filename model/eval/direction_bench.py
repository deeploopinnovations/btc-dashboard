"""
eval/direction_bench.py
=====================================================================
D1: does ANY direction model beat the rolling base rate as a PROBABILITY?

Pre-registered in ledger `D1-direction-bench` and committed BEFORE this file
existed. The rule is reproduced here so it can be checked against the ledger
rather than trusted.

WHY THE BASELINE IS NOT 0.5

P(R > 0) is 0.5089 / 0.5177 / 0.5328 / 0.5506 at H = 1 / 6 / 24 / 168, rising
with horizon as BTC's drift accumulates (`horizons-built`). A model reporting
53% accuracy at H = 24 has reproduced the base rate and demonstrated nothing.
Worse, the base rate itself MOVES: it ranges 0.4467 to 0.6360 across years at
H = 168. A model that merely tracks the current regime's base rate would post
large apparent skill against a FIXED reference while carrying no direction
information at all.

So the baseline to beat is the CALIB-WINDOW base rate — the most recent estimate
available without touching test — and not the unconditional one.

WHY THE PRIMARY IS PAIRED PER-EPISODE AND NOT FOLD-LEVEL

`direction-power` measured the fold-level noise floor from a comparison
containing no model, and found three of four horizons cannot resolve the effect
size prior work reported (BSS 0.0018):

    fold-level MDE at 80% power     H=1 0.0008   H=6 0.0042
                                    H=24 0.0168  H=168 0.0517

Only H = 1 is fold-level powered. The paired per-episode bootstrap has n in the
tens of thousands and is therefore the primary; the fold-level interval is
reported beside it and LABELLED UNDERPOWERED at H = 6, 24 and 168. This is
E-power's lesson applied before the fact rather than after: the two estimators
answer different questions, and the fold-level one is dominated by between-year
heterogeneity that no amount of extra episodes reduces.

THE ARMS

    base_unc     unconditional base rate from the TRAIN slice
    base_calib   base rate from the CALIB slice -- THE BASELINE TO BEAT
    logistic     logistic regression on the existing causal features
    gbm          gradient boosting (sklearn HistGradientBoosting)

and two negative controls that must NOT succeed:

    placebo      the same fit on features circularly rotated in time, so their
                 alignment with the episode is destroyed while every marginal
                 and most of the autocorrelation survives
    shuffled     the same fit on row-shuffled LABELS, which is the positive
                 falsification control -- if this scores, the harness is broken

CALIBRATION IS A PASS CONDITION. AUC IS NOT.

A model can rank episodes correctly and still emit probabilities that are wrong,
which is useless for anything that has to size a position. Slope must be in
[0.8, 1.2] and intercept within 0.02 of zero, both measured on test. AUC is
reported and is explicitly not a condition.

    python -m model.eval.direction_bench
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

from eval.direction import mean_ci                                        # noqa: E402
from noctua import splits as S                                            # noqa: E402
from research import pitfalls as P                                        # noqa: E402

HORIZONS = (1, 6, 24, 168)
N_FAMILY = 16                    # 4 model classes x 4 horizons, fixed a priori
CAL_SLOPE_LO, CAL_SLOPE_HI = 0.8, 1.2
CAL_INTERCEPT_TOL = 0.02
EPS = 1e-12


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    """Rank AUC via the Mann-Whitney identity. Ties get their average rank,
    which matters here because the constant-probability arms are ALL ties and
    must come out at exactly 0.5 rather than at whatever the sort happened to
    produce."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calibration(y, p):
    """Slope and intercept of a logistic recalibration on the LOGIT of p.

    Perfect calibration is slope 1, intercept 0. Slope < 1 means the forecast is
    over-confident (probabilities too far from the base rate); slope > 1 means
    under-confident. A constant forecast has no slope to estimate and returns
    NaN rather than a fabricated 1.0.
    """
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    if np.ptp(p) < 1e-9:
        return float("nan"), float("nan")
    x = np.log(p / (1 - p))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(x[:, None], y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def bucket_table(y, p, n_bins: int = 10):
    """Reliability table: predicted vs observed frequency per probability bin."""
    q = np.clip(np.asarray(p, np.float64), 0, 1)
    edges = np.quantile(q, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    idx = np.clip(np.digitize(q, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 20:
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "p_mean": float(q[m].mean()), "y_mean": float(y[m].mean())})
    return out


def fit_arm(name, Xtr, ytr, Xca, yca, Xte, rng):
    """Return test-slice probabilities for one arm. Every arm sees TRAIN for
    fitting and CALIB for calibration, and TEST only to predict."""
    if name == "base_unc":
        return np.full(len(Xte), float(ytr.mean()))
    if name == "base_calib":
        return np.full(len(Xte), float(yca.mean()))

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic":
        base = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0))
    elif name == "gbm":
        base = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4,
            min_samples_leaf=200, l2_regularization=1.0,
            random_state=0, early_stopping=False)
    else:
        raise ValueError(name)

    base.fit(Xtr, ytr)
    # Calibration is fitted on CALIB ONLY -- never on test. `cv="prefit"` is
    # what makes that true; refitting here would leak the calibration slice into
    # the fit and the guard would be measuring its own training data.
    cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    cal.fit(Xca, yca)
    return cal.predict_proba(Xte)[:, 1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D1 direction benchmark")
    ap.add_argument("--episodes", type=Path,
                    default=Path("model/artifacts/episodes_h4.parquet"))
    ap.add_argument("--features", type=Path,
                    default=Path("model/artifacts/features.parquet"))
    ap.add_argument("--ref-episodes", type=Path,
                    default=Path("model/artifacts/episodes.parquet"))
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/direction_bench.json"))
    a = ap.parse_args(argv)

    ep = pd.read_parquet(a.episodes)
    feat = pd.read_parquet(a.features)
    ref = pd.read_parquet(a.ref_episodes, columns=["anchor_ts", "H"])
    if len(feat) != len(ref):
        raise SystemExit("REFUSING: features.parquet is not aligned with episodes.parquet")

    # Only `cal_H` varies with H at a fixed anchor (verified); every other
    # column is a function of the anchor alone. So the per-anchor table is
    # well defined, and cal_H is DROPPED rather than recomputed -- within a
    # single-horizon model it is a constant and carries no information.
    cols = [c for c in feat.columns if c != "cal_H"]
    per_anchor = feat[cols].copy()
    per_anchor["anchor_ts"] = ref["anchor_ts"].to_numpy(np.int64)
    per_anchor = per_anchor.drop_duplicates("anchor_ts", keep="first")
    print(f"per-anchor feature table: {len(per_anchor):,} anchors x {len(cols)} features")

    alpha = 0.05 / N_FAMILY
    print(f"Bonferroni within the direction family: {N_FAMILY} arms -> "
          f"{100*(1-alpha):.2f}% intervals\n")

    rng = np.random.default_rng(0)
    results = {}
    for H in a.horizons:
        e = ep[ep.H == H].merge(per_anchor, on="anchor_ts", how="inner")
        e = e.sort_values("anchor_ts").reset_index(drop=True)
        fin = np.isfinite(e[cols].to_numpy(np.float64)).all(axis=1)
        e = e[fin].reset_index(drop=True)
        y_all = (e["R"].to_numpy(np.float64) > 0).astype(np.float64)
        X_all = e[cols].to_numpy(np.float64)
        folds = S.walk_forward_folds(e)
        print(f"H={H}: {len(e):,} episodes with complete features, {len(folds)} folds, "
              f"base rate {y_all.mean():.4f}")

        arms = ("base_unc", "base_calib", "logistic", "gbm", "placebo", "shuffled")
        per_fold = {k: [] for k in arms}
        pooled = {k: {"y": [], "p": []} for k in arms}
        for f in folds:
            tr, ca, te = f["train"], f["calib"], f["test"]
            if tr.sum() < 5000 or ca.sum() < 500 or te.sum() < 200:
                continue
            Xtr, ytr = X_all[tr], y_all[tr]
            Xca, yca = X_all[ca], y_all[ca]
            Xte, yte = X_all[te], y_all[te]
            t0 = time.time()
            for arm in arms:
                if arm == "placebo":
                    # circular rotation of the feature block: destroys alignment
                    # with the episode, preserves marginals and most structure
                    k = len(Xtr) // 3
                    p = fit_arm("logistic", np.roll(Xtr, k, axis=0), ytr,
                                np.roll(Xca, k // 4, axis=0), yca, Xte, rng)
                elif arm == "shuffled":
                    p = fit_arm("logistic", Xtr, rng.permutation(ytr),
                                Xca, yca, Xte, rng)
                else:
                    p = fit_arm(arm, Xtr, ytr, Xca, yca, Xte, rng)
                per_fold[arm].append(brier(yte, p))
                pooled[arm]["y"].append(yte)
                pooled[arm]["p"].append(p)
            print(f"   fold {f['year']}  n_te={int(te.sum()):6d}  ({time.time()-t0:.0f}s)",
                  flush=True)

        if not per_fold["base_calib"]:
            print(f"   H={H}: no usable folds\n"); continue

        ref_arm = "base_calib"
        yv = {k: np.concatenate(v["y"]) for k, v in pooled.items()}
        pv = {k: np.concatenate(v["p"]) for k, v in pooled.items()}
        base_brier_ep = (yv[ref_arm] - pv[ref_arm]) ** 2
        row = {}
        print(f"\n{'arm':>11} {'Brier':>9} {'BSS vs calib':>13} {'paired CI':>24} "
              f"{'logloss':>9} {'AUC':>7} {'cal slope':>10} {'cal int':>9}")
        for arm in arms:
            d_ep = (yv[arm] - pv[arm]) ** 2 - base_brier_ep     # paired, per episode
            ci = mean_ci(d_ep, seed=71, alpha=alpha) if arm != ref_arm else None
            sl, ic = calibration(yv[arm], pv[arm])
            bss = 1.0 - brier(yv[arm], pv[arm]) / max(brier(yv[ref_arm], pv[ref_arm]), EPS)
            row[arm] = {
                "brier": brier(yv[arm], pv[arm]), "bss_vs_calib": bss,
                "logloss": logloss(yv[arm], pv[arm]), "auc": auc(yv[arm], pv[arm]),
                "cal_slope": sl, "cal_intercept": ic,
                "paired_delta": None if ci is None else ci["mean"],
                "paired_ci": None if ci is None else ci["ci95"],
                "fold_brier": per_fold[arm],
                "buckets": bucket_table(yv[arm], pv[arm]),
                "n": int(len(yv[arm]))}
            cis = "—" if ci is None else f"[{ci['ci95'][0]:+.6f}, {ci['ci95'][1]:+.6f}]"
            print(f"{arm:>11} {row[arm]['brier']:>9.5f} {bss:>+13.5f} {cis:>24} "
                  f"{row[arm]['logloss']:>9.5f} {row[arm]['auc']:>7.4f} "
                  f"{sl:>10.3f} {ic:>9.3f}")

        # the pre-registered rule, applied to the two real model arms
        print(f"\n   --- pre-registered rule, H={H} ---")
        for arm in ("logistic", "gbm"):
            r = row[arm]
            lo, hi = r["paired_ci"]
            beats = hi < 0                                   # lower Brier is better
            sl, ic = r["cal_slope"], r["cal_intercept"]
            cal_ok = (np.isfinite(sl) and CAL_SLOPE_LO <= sl <= CAL_SLOPE_HI
                      and abs(ic) <= CAL_INTERCEPT_TOL)
            beats_plac = r["brier"] < row["placebo"]["brier"]
            ok = beats and cal_ok and beats_plac
            r["verdict"] = "PASS" if ok else "FAIL"
            print(f"   {arm:>9}: beats calib base rate {str(beats):>5} | "
                  f"calibrated {str(cal_ok):>5} (slope {sl:.3f}, int {ic:+.3f}) | "
                  f"beats placebo {str(beats_plac):>5}  ->  {r['verdict']}")
        # the falsification control must NOT succeed
        sh = row["shuffled"]
        print(f"   {'shuffled':>9}: BSS {sh['bss_vs_calib']:+.5f}  "
              f"(a positive value here would mean the harness is broken)")
        results[str(H)] = row

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
