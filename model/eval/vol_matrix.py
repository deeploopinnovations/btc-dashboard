"""
eval/vol_matrix.py
=====================================================================
The four-horizon volatility matrix. Pre-registered as ledger `vol-matrix`
BEFORE this file existed.

WHAT THIS ANSWERS

Every volatility number this repository has published so far lives on ONE
slice: H = 19 hours, anchored at 17:00 UTC. That is the production
configuration, and it is also a single point in a two-dimensional space. The
protocol asks for four horizons -- 1h, 6h, 1d, 1w -- with the mandatory
baseline family scored on each, so that "NOCTUA beats the baselines" stops
being a claim about one cell.

WHY THE 1h ROW HAS A DIFFERENT BASELINE FAMILY

`vol-matrix-power` measured, before any of this was built, that the sign of
the Log-HAR-minus-persistence difference FLIPS with horizon:

    H=1    +0.18831   Log-HAR is WORSE than doing nothing
    H=6    -0.00251   indistinguishable
    H=24   -0.18126   Log-HAR wins
    H=168  -0.43014   Log-HAR wins decisively

That is mechanically sensible: the HAR cascade's shortest component is DAILY,
so at a one-hour horizon it is being asked about a frequency it does not
represent. Quoting Log-HAR as "the baseline to beat" at H=1 would be beating a
model that loses to doing nothing. `har_short` -- Corsi's cascade extended
downward with har_1h and har_6h -- is the family that row needs, and it
already exists in `noctua/baselines.py`.

WHY THE PRIMARY IS PER-EPISODE

`vol-matrix-power` also measured the fold-level MDE at 80% power, as a
percentage of the persistence baseline:

    H=1 5.21%   H=6 11.76%   H=24 31.68%   H=168 65.48%

against a 4.98% reference effect. Not one row is comfortably powered at the
fold level; H=1 is marginal and the rest are not powered by factors of 2.4,
6.4 and 13. So the paired per-episode moving-block bootstrap is the primary
and the fold-level interval is reported beside it, LABELLED, exactly as
STATS_PROTOCOL section 1 prescribes.

WHAT THE MODEL BEING SCORED IS

One network per fold per seed, trained on ALL FOUR horizons together with the
horizon as an input feature -- which is the shipped architecture, not a
variant of it. Training a separate network per horizon would have been a
different model family, and scoring the SHIPPED artifact (trained on
H in {6,12,19,24}) at H=1 and H=168 would have been extrapolation dressed up
as a horizon study. This design avoids both.

THE FEATURE TABLE, AND THE TWO COLUMNS THAT ARE NOT PER-ANCHOR

`features.parquet` is aligned to `episodes.parquet` (H in {6,12,19,24}), not
to `episodes_h4.parquet` (H in {1,6,24,168}). Of its 42 columns, exactly two
depend on H: `cal_H` = H/24, and `cal_weekend_frac` = the weekend share of
the forward window. Both are recomputed here from the h4 horizons using the
same arithmetic as `noctua/features.py`, and the rest are joined per anchor.
`_verify_per_anchor` asserts on real data that no OTHER column varies with H
at a fixed anchor -- if that ever stops being true the run refuses rather
than silently joining the wrong number.

    python -m model.eval.vol_matrix
    python -m model.eval.vol_matrix --horizons 1 6 --seeds 1     # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.direction import mean_ci                                       # noqa: E402
from noctua import baselines as B                                        # noqa: E402
from noctua import infer as I                                            # noqa: E402
from noctua import splits as S                                           # noqa: E402
from noctua.model import BASE_COLS                                       # noqa: E402
from noctua.train import prepare, train_model                            # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

HOUR = 3600
HORIZONS = (1, 6, 24, 168)
N_FAMILY = 4                    # 4 horizons x 1 primary contrast, fixed a priori
BASELINE_ARMS = ("persistence", "log_har", "har_short", "garch_normal", "garch_t")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def qlike_vec(rv: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """QLIKE per episode. Identical expression to eval/garch.py and to
    benchmark.run_fold's `ql`, so the three are comparable by construction."""
    pv = np.maximum(sig, 1e-12) ** 2
    r = np.maximum(rv ** 2, 1e-18) / np.maximum(pv, 1e-18)
    return r - np.log(r) - 1.0


# --------------------------------------------------------------------------
# the h4 feature table
# --------------------------------------------------------------------------
H_DEPENDENT = ("cal_H", "cal_weekend_frac")


def _weekend_frac(anchor_ts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Weekend share of the forward window. Transcribed from
    noctua/features.py; `_verify_per_anchor` checks it reproduces the shipped
    column on the episodes where the two tables overlap."""
    maxH = int(H.max())
    offs = np.arange(maxH)
    fut_dow = (((anchor_ts[:, None] + offs[None, :] * HOUR) // 86400) + 4) % 7
    valid = offs[None, :] < H[:, None]
    is_we = ((fut_dow >= 5) & valid).sum(axis=1)
    return is_we / np.maximum(H, 1)


def _verify_per_anchor(feat: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    """Assert on real data that only H_DEPENDENT varies with H at a fixed
    anchor, and that the recomputation reproduces the shipped columns.

    This is a guard that can fail: it is checked against `episodes.parquet`,
    where multiple H share an anchor, so a column that secretly depends on H
    would be caught here rather than after a join has averaged it away.
    """
    a = ref["anchor_ts"].to_numpy(np.int64)
    h = ref["H"].to_numpy(np.float64)
    cols = list(feat.columns)
    varying = []
    g = pd.DataFrame({"anchor_ts": a})
    for c in cols:
        g[c] = feat[c].to_numpy(np.float64)
    spread = g.groupby("anchor_ts").agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
    for c in cols:
        if float(np.nanmax(spread[c].to_numpy())) > 1e-9:
            varying.append(c)
    unexpected = sorted(set(varying) - set(H_DEPENDENT))
    if unexpected:
        raise SystemExit(
            "REFUSING: these feature columns vary with H at a fixed anchor and "
            f"cannot be joined per anchor: {unexpected}")
    # and the recomputation must reproduce what ships
    got_H = h / 24.0
    err_H = float(np.max(np.abs(got_H - feat["cal_H"].to_numpy(np.float64))))
    got_we = _weekend_frac(a, h.astype(np.int64))
    err_we = float(np.max(np.abs(got_we - feat["cal_weekend_frac"].to_numpy(np.float64))))
    if err_H > 1e-12 or err_we > 1e-12:
        raise SystemExit(
            f"REFUSING: the H-dependent columns were not reproduced "
            f"(cal_H err {err_H:.3e}, cal_weekend_frac err {err_we:.3e}). "
            f"The recomputation in this file has drifted from noctua/features.py.")
    return varying


def build_h4_table(artifacts: Path):
    """episodes_h4 joined to a per-anchor feature block, with the two
    H-dependent columns recomputed for the h4 horizons."""
    ep4 = pd.read_parquet(artifacts / "episodes_h4.parquet")
    feat = pd.read_parquet(artifacts / "features.parquet")
    ref = pd.read_parquet(artifacts / "episodes.parquet",
                          columns=["anchor_ts", "H"])
    if len(feat) != len(ref):
        raise SystemExit("REFUSING: features.parquet is not aligned with episodes.parquet")

    varying = _verify_per_anchor(feat, ref)
    print(f"columns varying with H at a fixed anchor: {varying}  (expected {list(H_DEPENDENT)})")

    per_anchor_cols = [c for c in feat.columns if c not in H_DEPENDENT]
    pa = feat[per_anchor_cols].copy()
    pa["anchor_ts"] = ref["anchor_ts"].to_numpy(np.int64)
    pa = pa.drop_duplicates("anchor_ts", keep="first")

    ep4 = ep4.merge(pa, on="anchor_ts", how="inner")
    ep4 = ep4.sort_values(["anchor_ts", "H"]).reset_index(drop=True)
    Hv = ep4["H"].to_numpy(np.int64)
    ep4["cal_H"] = Hv / 24.0
    ep4["cal_weekend_frac"] = _weekend_frac(ep4["anchor_ts"].to_numpy(np.int64), Hv)

    X = ep4[list(feat.columns)].copy()
    ep = ep4[[c for c in ep4.columns if c not in feat.columns]].copy()
    fin = np.isfinite(X.to_numpy(np.float64)).all(axis=1)
    ep, X = ep[fin].reset_index(drop=True), X[fin].reset_index(drop=True)
    print(f"h4 table: {len(ep):,} episodes x {X.shape[1]} features, "
          f"H counts {dict(ep.H.value_counts().sort_index())}")
    return ep, X


# --------------------------------------------------------------------------
# one fold
# --------------------------------------------------------------------------
def run_fold(ep, X, fold, ret, hidden=32, seeds=3, verbose=False):
    """Train the multi-horizon network on this fold and score every arm on
    every horizon. Returns per-episode QLIKE arrays keyed by (H, arm)."""
    fin = np.isfinite(X.to_numpy(np.float64)).all(1)
    m_tr, m_va, m_te = fold["train"] & fin, fold["calib"] & fin, fold["test"] & fin
    if m_tr.sum() < 5000 or m_te.sum() < 200 or m_va.sum() < 500:
        return None

    Hall = ep.H.to_numpy(np.float64)
    yall = B.har_target(ep.RV.to_numpy(), Hall)

    # Stage B's causal volatility reference, clip bounds refit on THIS fold's
    # training episodes -- the same treatment benchmark.main uses.
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)
    lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
    sigma_ref_all = np.maximum(np.clip(raw, lo, hi), 1e-12)

    tr, stds = prepare(ep, X, m_tr, sigma_ref=sigma_ref_all[m_tr])
    wtr = S.sample_weights(ep, m_tr)
    va, _ = prepare(ep, X, m_va, *stds, sigma_ref=sigma_ref_all[m_va])

    ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                               tr["y"].astype(np.float64), wtr)
    bl = B.fit_vol_baselines(X[m_tr], yall[m_tr], wtr)
    models = [train_model(tr, wtr, va, hidden=hidden, epochs=40, seed=s,
                          verbose=verbose, ols_beta=ols.beta)[0]
              for s in range(seeds)]

    out = {}
    for H in sorted(ep.H.unique()):
        mh = m_te & (ep.H == H).to_numpy()
        if mh.sum() < 100:
            continue
        Ht = Hall[mh]
        rv = ep.RV.to_numpy()[mh]
        sq = np.sqrt(Ht)

        # NOCTUA: seeds averaged in the SAME place the shipped path averages
        # them -- on the atom outputs, not on the final QLIKE.
        d, _ = prepare(ep, X, mh, *stds)
        lp = bl["log_har_cal"].predict(X[mh])
        preds = [I.predict(m, d, har_logvol=lp) for m in models]
        sig_noctua = np.mean([p["sigma_med"] for p in preds], axis=0)

        arms = {
            "noctua": sig_noctua,
            # persistence uses the FEATURE har_1d, never episodes.RV1: RV1 is
            # built from fwd_rv1 and looks forward from the anchor.
            "persistence": np.maximum(np.exp(X.loc[mh, "har_1d"].to_numpy()) * sq, 1e-12),
            "log_har": np.exp(bl["log_har"].predict(X[mh])) * sq,
            "har_short": np.exp(bl["har_short"].predict(X[mh])) * sq,
        }
        if ret is not None:
            from eval.garch import fit_and_forecast
            anchors = ep.anchor_ts.to_numpy(np.int64)[mh]
            for dist, name in (("normal", "garch_normal"), ("t", "garch_t")):
                key = (fold["year"], dist)
                if key not in run_fold._garch:
                    run_fold._garch[key] = {}
                cache = run_fold._garch[key]
                if H not in cache:
                    cache[H] = fit_and_forecast(
                        ret, int(fold["train_end_ts"]), anchors, Ht, dist=dist,
                        verbose=False)
                arms[name] = cache[H]

        rec = {}
        for k, sig in arms.items():
            sig = np.asarray(sig, np.float64)
            ok = np.isfinite(sig) & (sig > 0)
            if ok.mean() < 0.95:
                continue
            q = qlike_vec(rv, np.where(ok, sig, np.nan))
            rec[k] = q
        # the train/calib-side QLIKE of each baseline, which is how the
        # BEST BASELINE is chosen -- never on test.
        sel = {}
        mv = m_va & (ep.H == H).to_numpy()
        if mv.sum() >= 50:
            rvv = ep.RV.to_numpy()[mv]
            sqv = np.sqrt(Hall[mv])
            sel["persistence"] = float(np.nanmean(qlike_vec(
                rvv, np.exp(X.loc[mv, "har_1d"].to_numpy()) * sqv)))
            sel["log_har"] = float(np.nanmean(qlike_vec(
                rvv, np.exp(bl["log_har"].predict(X[mv])) * sqv)))
            sel["har_short"] = float(np.nanmean(qlike_vec(
                rvv, np.exp(bl["har_short"].predict(X[mv])) * sqv)))
        out[int(H)] = {"qlike": rec, "n": int(mh.sum()),
                       "anchor_ts": ep.anchor_ts.to_numpy(np.int64)[mh],
                       "rv": rv, "sigma_persist": arms["persistence"],
                       "calib_qlike": sel}
    return out


run_fold._garch = {}


# --------------------------------------------------------------------------
def spike_mask(rv: np.ndarray, sig_persist: np.ndarray, q: float = 0.95) -> np.ndarray:
    """Spike = realized vol in the top 5% of the test slice's own distribution.
    Defined on the OUTCOME, so it is a conditioning variable for reporting and
    explicitly not something any arm is allowed to use."""
    return rv >= np.quantile(rv, q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="four-horizon volatility matrix")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/vol_matrix.json"))
    a = ap.parse_args(argv)

    ep, X = build_h4_table(a.artifacts)
    keep = ep.H.isin(a.horizons).to_numpy()
    ep, X = ep[keep].reset_index(drop=True), X[keep].reset_index(drop=True)

    ret = None
    if not a.no_garch:
        try:
            from eval.garch import hourly_returns
            ret = hourly_returns(a.artifacts)
            print(f"hourly returns for GARCH: {len(ret):,} rows")
        except Exception as e:                                     # noqa: BLE001
            print(f"GARCH unavailable ({e}); the row will be omitted, not faked")

    folds = S.walk_forward_folds(ep)
    alpha = 0.05 / N_FAMILY
    print(f"{len(folds)} walk-forward folds, hidden={a.hidden}, seeds={a.seeds}")
    print(f"Bonferroni within the volatility-matrix family: {N_FAMILY} rows -> "
          f"{100*(1-alpha):.2f}% intervals\n")

    acc = {}
    for f in folds:
        t0 = time.time()
        # the fold's train_end timestamp, needed by the GARCH fit
        f = dict(f)
        f["train_end_ts"] = int(ep.anchor_ts.to_numpy()[f["train"]].max())
        r = run_fold(ep, X, f, ret, hidden=a.hidden, seeds=a.seeds)
        if r is None:
            print(f"  fold {f['year']}: skipped (too few episodes)"); continue
        for H, d in r.items():
            acc.setdefault(H, []).append({"year": f["year"], **d})
        ns = " ".join(f"H{H}:{d['n']}" for H, d in sorted(r.items()))
        print(f"  fold {f['year']}  {ns}  ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for H in sorted(acc):
        rows = acc[H]
        arms = sorted(set().union(*[set(r["qlike"]) for r in rows]))
        # BEST BASELINE, chosen on calib only
        cal = {}
        for k in ("persistence", "log_har", "har_short"):
            vs = [r["calib_qlike"][k] for r in rows if k in r["calib_qlike"]]
            if vs:
                cal[k] = float(np.mean(vs))
        best = min(cal, key=cal.get) if cal else "persistence"

        print("\n" + "=" * 96)
        print(f"H = {H}h   {sum(r['n'] for r in rows):,} test episodes over "
              f"{len(rows)} folds")
        print(f"best baseline by CALIB QLIKE (never test): {best}   "
              + "  ".join(f"{k} {v:.4f}" for k, v in sorted(cal.items(), key=lambda kv: kv[1])))
        print("=" * 96)
        print(f"{'arm':>14} {'QLIKE':>9} {'vs best':>10} {'worst fold':>11} "
              f"{'spike':>9} {'calm':>9}   per-episode paired CI vs best")

        pooled = {k: np.concatenate([r["qlike"][k] for r in rows if k in r["qlike"]])
                  for k in arms}
        sp = np.concatenate([spike_mask(r["rv"], r["sigma_persist"]) for r in rows])
        per_fold = {k: [float(np.nanmean(r["qlike"][k])) for r in rows if k in r["qlike"]]
                    for k in arms}
        row_out = {}
        for k in ["noctua"] + [x for x in arms if x != "noctua"]:
            v = pooled[k]
            if len(v) != len(pooled[best]):
                continue
            d = pooled[best] - v          # positive = k is BETTER than best
            good = np.isfinite(d)
            ci = mean_ci(d[good], alpha=alpha) if k != best else None
            worst = max(per_fold[k]) if per_fold[k] else float("nan")
            cis = ("—" if ci is None else
                   f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]")
            print(f"{k:>14} {np.nanmean(v):9.5f} {np.nanmean(d):+10.5f} "
                  f"{worst:11.5f} {np.nanmean(v[sp]):9.4f} {np.nanmean(v[~sp]):9.4f}   {cis}")
            row_out[k] = {
                "qlike": float(np.nanmean(v)),
                "delta_vs_best": float(np.nanmean(d)),
                "worst_fold": float(worst),
                "spike": float(np.nanmean(v[sp])),
                "calm": float(np.nanmean(v[~sp])),
                "per_fold": per_fold[k],
                "paired_ci": None if ci is None else ci["ci95"],
                "paired_n": None if ci is None else ci["n"],
            }

        # the pre-registered rule, applied verbatim
        n = row_out.get("noctua")
        if n and n["paired_ci"]:
            clears = n["paired_ci"][0] > 0.0
            verdict = "CLEARS" if clears else "DOES NOT CLEAR"
        else:
            verdict = "NO INTERVAL"
        print(f"\n   --- pre-registered rule, H={H} ---")
        print(f"   NOCTUA vs {best}: paired per-episode CI at "
              f"{100*(1-alpha):.2f}% {verdict} zero favourably")
        print(f"   fold-level interval is NOT the primary and is UNDERPOWERED "
              f"at this horizon (vol-matrix-power)")
        results[str(H)] = {"best_baseline": best, "calib_qlike": cal,
                           "n_test": sum(r["n"] for r in rows),
                           "years": [r["year"] for r in rows],
                           "arms": row_out, "verdict": verdict}

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha,
        "seeds": a.seeds, "hidden": a.hidden,
        "horizons": results,
    }, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
