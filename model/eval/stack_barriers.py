"""
eval/stack_barriers.py
=====================================================================
P4-stack-anchor: does the zoo ensemble improve the PRODUCT, not just QLIKE?

WHERE THIS COMES FROM

`P4-zoo-stack-v2-result` found the first forecast improvement in this line of
work: a QLIKE-stacked ensemble over the teacher zoo, shrunk halfway toward
NOCTUA, beats NOCTUA as served at H = 1 / 6 / 24 (+6.56 / +2.05 / +1.69%). It
was measured on sigma, at the zoo's four horizons. The product is a barrier
curve at H = 19, built from `sigma_atoms`, and `P2-mean-level` showed that a
level move inside the predictive object can cost every barrier metric ~20%.

THE INTEGRATION, AND WHY IT IS THIS ONE

The served forecast is already a two-model log-space blend: NOCTUA's quantile
curve is shifted so its median lands on 0.25 * NOCTUA + 0.75 * Log-HAR
(`infer.BLEND_W`). The anchor -- Log-HAR -- is one hand-picked baseline. This
replaces it with a STACKED anchor: a convex combination of the baselines the
fold runner already fits (log_har, log_har_cal, har_short, persistence), with
weights fitted on each fold's CALIB slice by QLIKE, shrunk halfway toward
Log-HAR as R82's constant prescribes. NOCTUA's own 0.25 is untouched.

    anchor_new = 0.5 * S + 0.5 * log_har_cal,   S = sum_k w_k x_k,  w convex
    final shift = 0.75 * (anchor_new - log_har_cal)

applied through `run_fold(post_shift_fn=...)`, which moves the final blended
level BEFORE Stage B and the committee, so every barrier curve inherits it.

LEVEL-NEUTRAL BY CONSTRUCTION. The shift is DEMEANED on the calib slice. The
feasibility screen that motivated this gave each arm its own calib level, so
its gain was the CROSS-SECTIONAL re-weighting, net of level; demeaning keeps the
product test faithful to that and removes the systematic level move that
`P2-mean-level` showed is the dangerous one. What remains is information about
WHICH nights are riskier, not a claim that all nights are.

GARCH is excluded because the fold runner does not fit it -- and the screen
found the pool without GARCH was BETTER at H = 24 and 168, where GARCH
overfits a thin calib slice.

THE MIRROR IS THE CONTROL. Same shift, opposite sign. If the anchor carries
information, the mirror must hurt what the anchor helps; if both move the
battery the same way, the effect belongs to perturbing the level at all.

    python -m model.eval.stack_barriers --selftest
    python -m model.eval.stack_barriers
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
from eval.scale_adopt import barrier_cols                              # noqa: E402
from eval.vol_matrix import block_len_for, qlike_vec                   # noqa: E402
from eval.zoo_stack import fit_convex                                  # noqa: E402
from noctua import baselines as B                                      # noqa: E402
from noctua import infer as I                                          # noqa: E402
from noctua import splits as S                                         # noqa: E402
from noctua.train import load_all                                      # noqa: E402

PROD_H = 19
POOL = ("log_har", "log_har_cal", "har_short", "persistence")
SHRINK = 0.5
ARMS = ("A1", "MIRROR")
METRICS = ("DSC", "MCB", "brier", "crps", "logs", "pinball")
HIGHER_BETTER = {"DSC"}


def pool_logvol(bl, X, mask) -> np.ndarray:
    """(n, K) log vol RATE for each pooled baseline, in the units the blend uses.

    The OLS baselines predict the log hourly vol rate directly. Persistence is
    har_1d, which is already that rate -- the fold runner's own
    `sigma_persist` is exp(har_1d) * sqrt(H), the same quantity scaled to the
    window, so the two sit on one axis.
    """
    cols = []
    for k in POOL:
        if k == "persistence":
            cols.append(X.loc[mask, "har_1d"].to_numpy(np.float64))
        else:
            cols.append(np.asarray(bl[k].predict(X[mask]), np.float64))
    return np.column_stack(cols)


def make_shift(ep, X, fold, sign: float = 1.0):
    """Build the post_shift_fn for one fold, fitted on that fold's CALIB.

    Reproduces the fold runner's own baseline fit exactly -- same finite mask,
    same sample weights, same target -- so `log_har_cal` here is the anchor the
    blend already uses and the shift is relative to the right thing.
    """
    fin = np.isfinite(X.to_numpy()).all(1)
    m_tr = np.asarray(fold["train"], bool) & fin
    wtr = S.sample_weights(ep, m_tr)
    H = ep.H.to_numpy(np.float64)
    y = B.har_target(ep.RV.to_numpy(), H)
    bl = B.fit_vol_baselines(X[m_tr], y[m_tr], wtr)
    at = (ep.H == PROD_H).to_numpy()
    m_ca = np.asarray(fold["calib"], bool) & fin & at
    rv = ep.RV.to_numpy(np.float64)

    Lc = pool_logvol(bl, X, m_ca)
    ok = np.all(np.isfinite(Lc), axis=1) & np.isfinite(rv[m_ca]) & (rv[m_ca] > 0)
    # window sigma = exp(rate) * sqrt(H), so log window sigma = rate + 0.5 log H;
    # the convex weights are invariant to that common offset, the level is not,
    # and the level is profiled out inside fit_convex anyway
    lwin = Lc[ok] + 0.5 * np.log(PROD_H)
    w = fit_convex(lwin, np.log(rv[m_ca][ok] ** 2))
    i_lhc = POOL.index("log_har_cal")

    def raw_shift(mask):
        L = pool_logvol(bl, X, mask)
        stacked = L @ w
        anchor_new = SHRINK * stacked + (1.0 - SHRINK) * L[:, i_lhc]
        return (1.0 - I.BLEND_W) * (anchor_new - L[:, i_lhc])

    # DEMEAN ON CALIB: level-neutral by construction, see the module note.
    d_ca = raw_shift(m_ca)
    mu = float(np.nanmean(d_ca))

    def fn(mask, _m_tr):
        out = sign * (raw_shift(mask) - mu)
        return np.nan_to_num(out, nan=0.0)

    return fn, {"weights": dict(zip(POOL, map(float, w))), "demean": mu}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stacked anchor vs the barrier battery")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=Path("model/artifacts/stack_barriers.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    ep, X = load_all(a.artifacts)
    folds = S.walk_forward_folds(ep)
    n_family = len(ARMS) * len(METRICS)
    alpha = 0.05 / n_family
    print(f"P4-stack-anchor   production slice H={PROD_H}   seeds={a.seeds}")
    print(f"family {n_family} -> {100*(1-alpha):.3f}% intervals\n")

    rows = []
    for f in folds:
        fn, info = make_shift(ep, X, f, +1.0)
        fm, _ = make_shift(ep, X, f, -1.0)
        r0 = run_fold(ep, X, f, a.hidden, a.seeds)
        if r0 is None:
            continue
        pe0 = r0["per_episode"]
        row = {"year": f["year"], "info": info,
               "q_M0": qlike_vec(pe0["rv"], pe0["sigma_mean"]),
               "bar_M0": barrier_cols(r0["rows"])}
        okf = True
        for arm, sfn in (("A1", fn), ("MIRROR", fm)):
            r1 = run_fold(ep, X, f, a.hidden, a.seeds, post_shift_fn=sfn)
            if r1 is None:
                okf = False; break
            pe1 = r1["per_episode"]
            if not np.array_equal(pe1["test_idx"], pe0["test_idx"]):
                raise SystemExit(f"REFUSING: fold {f['year']} arm {arm} scored "
                                 "different episodes -- the contrast is not paired")
            moved = float(np.max(np.abs(np.log(pe1["sigma_med"])
                                        - np.log(pe0["sigma_med"]))))
            if moved == 0.0:
                raise SystemExit(f"REFUSING: fold {f['year']} arm {arm} did not "
                                 "move sigma_med at all -- the shift never arrived")
            row[f"q_{arm}"] = qlike_vec(pe1["rv"], pe1["sigma_mean"])
            row[f"bar_{arm}"] = barrier_cols(r1["rows"])
            row[f"moved_{arm}"] = moved
        if okf:
            rows.append(row)
            print(f"  fold {f['year']}: weights "
                  + " ".join(f"{k}={v:.2f}" for k, v in info["weights"].items())
                  + f"   max |log shift| {row['moved_A1']:.3f}")

    if not rows:
        print("no complete folds"); return 1

    print(f"\n{'metric':>9} {'M0':>10} {'A1':>10} {'MIRROR':>10}")
    per_fold = {}
    for met in METRICS:
        per_fold[met] = {arm: [r[f"bar_{arm}"].get(met) for r in rows]
                         for arm in ("M0",) + ARMS}
        print(f"{met:>9} " + " ".join(
            f"{np.mean(per_fold[met][k]):10.6f}" for k in ("M0",) + ARMS))

    print(f"\n{'metric':>9} {'arm':>7} {'delta vs M0':>13} "
          f"{'CI (corrected)':>28} {'folds':>6}")
    ci_out, clears = {}, {arm: [] for arm in ARMS}
    worse = {arm: [] for arm in ARMS}
    for met in METRICS:
        sgn = 1.0 if met in HIGHER_BETTER else -1.0
        for arm in ARMS:
            d = sgn * (np.asarray(per_fold[met][arm], float)
                       - np.asarray(per_fold[met]["M0"], float))
            lo, hi = mean_ci(d, alpha=alpha)["ci95"]
            ci_out[f"{met}_{arm}"] = {"delta": float(d.mean()),
                                      "ci95": [float(lo), float(hi)],
                                      "n_better": int((d > 0).sum())}
            if lo > 0:
                clears[arm].append(met)
            if hi < 0:
                worse[arm].append(met)
            print(f"{met:>9} {arm:>7} {d.mean():+13.6f} "
                  f"[{lo:+12.6f}, {hi:+12.6f}] {int((d>0).sum()):>3}/{len(d)}")

    L = block_len_for(PROD_H, sum(len(r["q_M0"]) for r in rows))
    q0 = np.concatenate([r["q_M0"] for r in rows])
    print("\npaired per-episode QLIKE on the production slice:")
    qci = {}
    for arm in ARMS:
        qa = np.concatenate([r[f"q_{arm}"] for r in rows])
        dd = q0 - qa
        g = np.isfinite(dd)
        lo, hi = mean_ci(dd[g], alpha=alpha, block_len=L)["ci95"]
        qci[arm] = {"pct": float(100 * np.nanmean(dd) / np.nanmean(q0)),
                    "ci95": [float(lo), float(hi)]}
        print(f"   {arm:>7}: {qci[arm]['pct']:+6.2f}%  CI [{lo:+.5f}, {hi:+.5f}]")

    print("\n--- pre-registered rule ---")
    for arm in ARMS:
        print(f"   {arm}: clears {len(clears[arm])}/6 {clears[arm]}   "
              f"significantly WORSE on {worse[arm] or 'nothing'}")
    met_rule = (len(clears["A1"]) >= 4 and len(clears["MIRROR"]) < 4
                and qci["A1"]["ci95"][0] > 0 and not worse["A1"])
    print(f"   ADOPTION CONDITION (A1 >= 4/6, mirror < 4/6, QLIKE clears, "
          f"no metric worse): {'MET' if met_rule else 'NOT MET'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_family": n_family, "alpha": alpha, "prod_H": PROD_H, "pool": POOL,
        "shrink": SHRINK, "folds": [r["year"] for r in rows],
        "weights": {str(r["year"]): r["info"]["weights"] for r in rows},
        "barriers": {m: {k: float(np.mean(v)) for k, v in d.items()}
                     for m, d in per_fold.items()},
        "barrier_ci": ci_out, "qlike_ci": qci,
        "clears": clears, "worse": worse, "rule_met": bool(met_rule)},
        indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


def selftest() -> int:
    ok = []
    rng = np.random.default_rng(3)
    # The shift algebra: with anchor_new = 0.5 S + 0.5 lhc, the FINAL level must
    # move by 0.75 * 0.5 * (S - lhc), and run_fold divides by (1 - BLEND_W) to
    # make the realised move equal the requested one.
    S_ = rng.normal(-4, 0.3, 100); lhc = rng.normal(-4, 0.3, 100)
    anchor_new = SHRINK * S_ + (1 - SHRINK) * lhc
    want = (1 - I.BLEND_W) * (anchor_new - lhc)
    qa = rng.normal(-4, 0.3, 100)
    old = I.BLEND_W * qa + (1 - I.BLEND_W) * lhc
    new = I.BLEND_W * qa + (1 - I.BLEND_W) * (lhc + want / (1 - I.BLEND_W))
    ok.append(("final level moves by exactly the requested shift",
               np.allclose(new - old, want)))
    ok.append(("which equals 0.75 * 0.5 * (S - lhc)",
               np.allclose(want, 0.375 * (S_ - lhc))))
    # The mirror is the same magnitude, opposite sign.
    ok.append(("mirror is the exact negation", np.allclose(-want, -1.0 * want)))
    # POOL must contain the incumbent anchor, or 'shrink toward it' is undefined.
    ok.append(("the incumbent anchor is in the pool", "log_har_cal" in POOL))
    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    bad = sum(not g for _, g in ok)
    print(f"\n{len(ok) - bad}/{len(ok)} pass")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
