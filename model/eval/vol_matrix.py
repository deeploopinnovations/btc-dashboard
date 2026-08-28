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

THE FEATURE TABLE, AND THE GUARD THAT FIRED

`features.parquet` is aligned to `episodes.parquet` (H in {6,12,19,24}), not
to `episodes_h4.parquet` (H in {1,6,24,168}), so the h4 horizons need their
own feature matrix.

The first version of this file joined the features PER ANCHOR and recomputed
only the two columns it believed depended on H -- `cal_H` = H/24 and
`cal_weekend_frac`. `_verify_per_anchor` was written to CHECK that belief
against real data rather than assert it, and on its first run it refused:

    REFUSING: these feature columns vary with H at a fixed anchor and cannot
    be joined per anchor: ['seas_1d', 'seas_22d', 'seas_5d']

It was right. `seas_{d}d` is the realized volatility of [a - 24d, a - 24d + H)
-- the same clock window on a prior day, whose LENGTH is the horizon. Three
more columns depend on H, and at H = 168 two of them are not merely different
but UNDEFINED, because the window would run past the anchor.

So the per-anchor join is abandoned. `noctua.features.build_features` is
called directly on `episodes_h4`, which computes every column at the right
horizon by construction and cannot drift from what training and serving use.
A second guard cross-checks the result against `features.parquet` at H = 6,
the one horizon the two tables share, and refuses on any disagreement above
1e-9 or any NaN-pattern mismatch.

`_verify_per_anchor` is kept, and still runs, with the empirically discovered
set of H-dependent columns baked into it -- so it will fire again if a
SIXTH such column ever appears.

THE SAME DEFECT IS PRESENT IN eval/direction_bench.py, which made the
per-anchor join on this table and asserted in a comment that "only cal_H
varies with H at a fixed anchor (verified)". That comment is wrong. It is a
MISSPECIFICATION and not a leak -- every substituted value is a function of
the anchor and of hours strictly before it -- but the direction benchmark is
re-run against corrected features rather than defended. See BENCHMARK.md.

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

# THE BLOCK LENGTH, and why it is not left at the default.
#
# `mean_ci` blocks at round(n^(1/3)) by default. That is a rule of thumb about
# SAMPLE SIZE and knows nothing about the dependence it exists to absorb. The
# episode table is anchored HOURLY with an H-hour forward window, so two
# consecutive episodes share H-1 of their H hours and the dependence range is
# H no matter how large n is. At H = 168 and n ~ 49,000 the default gives 37 --
# about a fifth of the overlap -- and the resulting interval treats four fifths
# of a shared window as independent evidence.
#
# So the primary interval blocks at 2H (a block spanning two full windows,
# which is the conservative reading), floored at the n^(1/3) value so it never
# becomes SHORTER than the default. The default interval is reported beside it
# as a sensitivity, because the choice is a real degree of freedom and hiding
# it would be a way to pick the answer.
def block_len_for(H: int, n: int) -> int:
    return max(int(round(n ** (1 / 3))), 2 * int(H))
BASELINE_ARMS = ("persistence", "log_har", "har_short", "garch_normal", "garch_t")

# The two feature columns that are UNDEFINED at H = 168, because `seas_{d}d`
# reads [a - 24d, a - 24d + H) and that window runs past the anchor once
# H > 24d. Dropping them yields a 40-column set that is defined at every
# horizon, which is what makes a four-row matrix possible at all. Both arms
# are scored: `noctua` on the full 42 columns wherever they exist, `noctua40`
# on the reduced set everywhere. The difference between them at H = 1/6/24 is
# a free ablation of exactly what those two columns are worth.
UNDEFINED_AT_1W = ("seas_1d", "seas_5d")


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
# Columns that depend on the HORIZON, not only on the anchor. This list was
# not written from the source -- it was DISCOVERED by `_verify_per_anchor`
# refusing on the first run, and the seas_* entries are the ones that were
# missed. Anything outside this set that varies with H still stops the run.
H_DEPENDENT = ("cal_H", "cal_weekend_frac", "seas_1d", "seas_5d", "seas_22d")


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
    g = pd.DataFrame({c: feat[c].to_numpy(np.float64) for c in cols})
    g["anchor_ts"] = a
    gb = g.groupby("anchor_ts", sort=False)
    # max - min per anchor, vectorised. `agg(lambda ...)` here is a per-group
    # Python call over 42 columns and 500k rows and takes minutes; this is the
    # same quantity in one pass.
    spread = gb.max() - gb.min()
    for c in cols:
        if float(np.nanmax(spread[c].to_numpy())) > 1e-9:
            varying.append(c)
    unexpected = sorted(set(varying) - set(H_DEPENDENT))
    if unexpected:
        raise SystemExit(
            "REFUSING: these feature columns vary with H at a fixed anchor and "
            f"are not in H_DEPENDENT: {unexpected}. Every H-dependent column has "
            "to be computed at the episode's own horizon, so this list going "
            "stale is exactly the failure this guard exists to catch.")
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
    """The h4 episodes with a feature matrix built AT THEIR OWN HORIZONS.

    `build_features` is called rather than reimplemented, so this table cannot
    drift from the one training and serving use. `_verify_per_anchor` still
    runs against `episodes.parquet` -- it is the guard that rejected the
    per-anchor shortcut, and leaving it in place is what keeps the reason
    visible.
    """
    from noctua.features import build_features

    ep4 = pd.read_parquet(artifacts / "episodes_h4.parquet")
    feat = pd.read_parquet(artifacts / "features.parquet")
    ref = pd.read_parquet(artifacts / "episodes.parquet", columns=["anchor_ts", "H"])
    if len(feat) != len(ref):
        raise SystemExit("REFUSING: features.parquet is not aligned with episodes.parquet")
    varying = _verify_per_anchor(feat, ref)
    print(f"columns that vary with H at a fixed anchor: {varying}")

    hours = pd.read_parquet(artifacts / "btcusd_1h.parquet")
    ep4 = ep4.sort_values(["anchor_ts", "H"]).reset_index(drop=True)
    X = build_features(hours, ep4)
    if len(X) != len(ep4):
        raise SystemExit("REFUSING: build_features did not return one row per episode")
    if sorted(X.columns) != sorted(feat.columns):
        raise SystemExit(
            "REFUSING: the h4 feature matrix has different columns from the shipped one: "
            f"{sorted(set(X.columns) ^ set(feat.columns))}")
    X = X[list(feat.columns)]

    # An exactness check that can fail: at the ONE horizon the two tables
    # share (H=6), every column must agree to floating point. If it does not,
    # this file is building different features from the ones that ship.
    shared = ep4.H.to_numpy() == 6
    ref_key = pd.DataFrame({"anchor_ts": ref.anchor_ts.to_numpy(np.int64),
                            "H": ref.H.to_numpy(np.int64), "i": np.arange(len(ref))})
    ref6 = ref_key[ref_key.H == 6].drop_duplicates("anchor_ts").set_index("anchor_ts")["i"]
    a6 = ep4.anchor_ts.to_numpy(np.int64)[shared]
    have = np.isin(a6, ref6.index.to_numpy())
    if have.sum() > 1000:
        j = ref6.loc[a6[have]].to_numpy()
        A = X.loc[shared].to_numpy(np.float64)[have]
        Bv = feat.to_numpy(np.float64)[j]
        both = np.isfinite(A) & np.isfinite(Bv)
        err = float(np.max(np.abs(A[both] - Bv[both])))
        nan_mismatch = int((np.isfinite(A) != np.isfinite(Bv)).sum())
        print(f"H=6 cross-check against features.parquet on {int(have.sum()):,} episodes: "
              f"max |diff| {err:.3e}, NaN-pattern mismatches {nan_mismatch}")
        if err > 1e-9 or nan_mismatch:
            raise SystemExit(
                "REFUSING: the h4 feature matrix disagrees with features.parquet at the "
                "one horizon they share. This file is not building the shipped features.")
    else:
        raise SystemExit("REFUSING: too few shared H=6 episodes to cross-check")

    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    fin42 = np.isfinite(X.to_numpy(np.float64)).all(axis=1)
    fin40 = np.isfinite(X[cols40].to_numpy(np.float64)).all(axis=1)
    tab = pd.DataFrame({"H": ep4.H.to_numpy(), "f42": fin42, "f40": fin40})
    print(tab.groupby("H")[["f42", "f40"]].sum().to_string())
    print("  H=168 has NO complete 42-column row: seas_1d and seas_5d read a window "
          "[a-24d, a-24d+H) that runs past the anchor once H > 24d.")
    # The two sets must coincide wherever the dropped columns are defined; if
    # they do not, the 42- and 40-column arms would be scored on different
    # episodes at the same horizon and the ablation would measure the sample,
    # not the features.
    short = ep4.H.to_numpy() <= 24
    if not np.array_equal(fin42[short], fin40[short]):
        raise SystemExit("REFUSING: the 42- and 40-column complete-row sets differ at H<=24")

    ep, X = ep4[fin40].reset_index(drop=True), X[fin40].reset_index(drop=True)
    print(f"h4 table: {len(ep):,} episodes x {X.shape[1]} features "
          f"({len(cols40)} of them defined at every horizon)")
    return ep, X


# --------------------------------------------------------------------------
# one fold
# --------------------------------------------------------------------------
def run_fold(ep, X, fold, ret, hidden=32, seeds=3, verbose=False):
    """Train the multi-horizon networks on this fold and score every arm at
    every horizon. Returns per-episode QLIKE arrays keyed by (H, arm).

    TWO NOCTUA ARMS, and the reason is structural rather than a design choice:
    `noctua` uses the full 42-column set and therefore CANNOT be evaluated at
    H = 168, where two of those columns do not exist. `noctua40` drops exactly
    those two everywhere, so it spans all four horizons and the row-to-row
    comparison is between the same model. At H = 1/6/24 both are scored on the
    identical episodes, which makes their difference an ablation of the two
    dropped columns rather than a difference of sample.
    """
    Hall = ep.H.to_numpy(np.float64)
    cols40 = [c for c in X.columns if c not in UNDEFINED_AT_1W]
    fin42 = np.isfinite(X.to_numpy(np.float64)).all(1)
    fin40 = np.isfinite(X[cols40].to_numpy(np.float64)).all(1)

    yall = B.har_target(ep.RV.to_numpy(), Hall)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(Hall)

    variants = {}
    for name, Xv, fv in (("noctua", X, fin42), ("noctua40", X[cols40], fin40)):
        m_tr, m_va = fold["train"] & fv, fold["calib"] & fv
        if m_tr.sum() < 5000 or m_va.sum() < 500:
            continue
        # Stage B's causal volatility reference, clip bounds refit on THIS
        # fold's training episodes -- the treatment benchmark.main uses.
        lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
        sref = np.maximum(np.clip(raw, lo, hi), 1e-12)
        tr, stds = prepare(ep, Xv, m_tr, sigma_ref=sref[m_tr])
        wtr = S.sample_weights(ep, m_tr)
        va, _ = prepare(ep, Xv, m_va, *stds, sigma_ref=sref[m_va])
        ols = B.OLS(BASE_COLS).fit(pd.DataFrame(tr["Xb"], columns=BASE_COLS),
                                   tr["y"].astype(np.float64), wtr)
        bl = B.fit_vol_baselines(Xv[m_tr], yall[m_tr], wtr)
        models = [train_model(tr, wtr, va, hidden=hidden, epochs=40, seed=s,
                              verbose=verbose, ols_beta=ols.beta)[0]
                  for s in range(seeds)]
        variants[name] = {"X": Xv, "fin": fv, "stds": stds, "bl": bl,
                          "models": models, "m_va": m_va}
    if "noctua40" not in variants:
        return None

    ref40 = variants["noctua40"]
    out = {}
    for H in sorted(ep.H.unique()):
        mh = fold["test"] & ref40["fin"] & (ep.H == H).to_numpy()
        if mh.sum() < 100:
            continue
        Ht = Hall[mh]
        rv = ep.RV.to_numpy()[mh]
        sq = np.sqrt(Ht)
        bl = ref40["bl"]

        arms = {
            # persistence uses the FEATURE har_1d, never episodes.RV1: RV1 is
            # built from fwd_rv1 and looks FORWARD from the anchor.
            "persistence": np.maximum(np.exp(X.loc[mh, "har_1d"].to_numpy()) * sq, 1e-12),
            "log_har": np.exp(bl["log_har"].predict(X[mh])) * sq,
            "har_short": np.exp(bl["har_short"].predict(X[mh])) * sq,
        }
        for name, v in variants.items():
            mv = mh & v["fin"]
            if mv.sum() != mh.sum():
                continue                      # arm not defined on this slice
            d, _ = prepare(ep, v["X"], mh, *v["stds"])
            lp = v["bl"]["log_har_cal"].predict(v["X"][mh])
            preds = [I.predict(m, d, har_logvol=lp) for m in v["models"]]
            # Seeds are averaged on the ATOM outputs, where the shipped path
            # averages them -- not on the final QLIKE.
            arms[name] = np.mean([p["sigma_med"] for p in preds], axis=0)

        if ret is not None:
            from eval.garch import fit_and_forecast
            anchors = ep.anchor_ts.to_numpy(np.int64)[mh]
            for dist, nm in (("normal", "garch_normal"), ("t", "garch_t")):
                key = (fold["year"], dist, int(H))
                if key not in run_fold._garch:
                    run_fold._garch[key] = fit_and_forecast(
                        ret, int(fold["train_end_ts"]), anchors, Ht,
                        dist=dist, verbose=False)
                arms[nm] = run_fold._garch[key]

        rec, sigmas = {}, {}
        for k, sig in arms.items():
            sig = np.asarray(sig, np.float64)
            ok = np.isfinite(sig) & (sig > 0)
            if ok.mean() < 0.95:
                continue
            sig = np.where(ok, sig, np.nan)
            rec[k] = qlike_vec(rv, sig)
            # kept so a caller can score a DIFFERENT objective on the SAME
            # forecasts -- eval/econ_voltarget.py needs sigma, not QLIKE, and
            # retraining it there would be R18's mistake with extra steps.
            sigmas[k] = sig

        # The train/calib-side QLIKE of each baseline. This is how the BEST
        # BASELINE is chosen -- on calib, never on test.
        sel = {}
        mv = ref40["m_va"] & (ep.H == H).to_numpy()
        if mv.sum() >= 50:
            rvv = ep.RV.to_numpy()[mv]
            sqv = np.sqrt(Hall[mv])
            sel["persistence"] = float(np.nanmean(qlike_vec(
                rvv, np.exp(X.loc[mv, "har_1d"].to_numpy()) * sqv)))
            for k in ("log_har", "har_short"):
                sel[k] = float(np.nanmean(qlike_vec(
                    rvv, np.exp(bl[k].predict(X[mv])) * sqv)))
        out[int(H)] = {"qlike": rec, "n": int(mh.sum()),
                       "rv": rv, "sigma_persist": arms["persistence"],
                       "sigma": sigmas, "calib_qlike": sel,
                       "anchor_ts": ep.anchor_ts.to_numpy(np.int64)[mh],
                       "R": ep.R.to_numpy(np.float64)[mh]}
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
        # An arm counts only if it produced a score in EVERY fold at this
        # horizon. A partially present arm would otherwise be pooled over a
        # different set of years from the baseline it is compared against.
        present = [set(r["qlike"]) for r in rows]
        arms = sorted(set.intersection(*present))
        dropped = sorted(set.union(*present) - set(arms))

        cal = {}
        for k in ("persistence", "log_har", "har_short"):
            vs = [r["calib_qlike"][k] for r in rows if k in r["calib_qlike"]]
            if vs:
                cal[k] = float(np.mean(vs))
        best = min(cal, key=cal.get) if cal else "persistence"

        print("\n" + "=" * 100)
        print(f"H = {H}h   {sum(r['n'] for r in rows):,} test episodes over {len(rows)} folds")
        print(f"best baseline by CALIB QLIKE (never test): {best}    "
              + "  ".join(f"{k} {v:.4f}" for k, v in sorted(cal.items(), key=lambda kv: kv[1])))
        if dropped:
            print(f"arms not scored at this horizon (absent in at least one fold): {dropped}")
        print("=" * 100)
        print(f"{'arm':>13} {'QLIKE':>9} {'vs best':>10} {'worst fold':>11} "
              f"{'spike':>9} {'calm':>9}   paired per-episode CI vs {best}")

        pooled = {k: np.concatenate([r["qlike"][k] for r in rows]) for k in arms}
        sp = np.concatenate([spike_mask(r["rv"], r["sigma_persist"]) for r in rows])
        per_fold = {k: [float(np.nanmean(r["qlike"][k])) for r in rows] for k in arms}

        order = [k for k in ("noctua", "noctua40") if k in arms]
        order += [k for k in arms if k not in order]
        row_out = {}
        for k in order:
            v = pooled[k]
            d = pooled[best] - v          # positive => k is BETTER than best
            good = np.isfinite(d)
            ci = ci_thumb = None
            if k != best:
                L = block_len_for(H, int(good.sum()))
                ci = mean_ci(d[good], alpha=alpha, block_len=L)
                # the rule-of-thumb interval, for the NOCTUA arms only -- they
                # are the ones a verdict depends on, and the comparison is
                # there to show whether the verdict survives the choice.
                if k in ("noctua", "noctua40"):
                    ci_thumb = mean_ci(d[good], alpha=alpha)
            cis = "—" if ci is None else f"[{ci['ci95'][0]:+.5f}, {ci['ci95'][1]:+.5f}]"
            if ci_thumb is not None:
                cis += (f"   [n^(1/3), L={ci_thumb['block_len']}: "
                        f"{ci_thumb['ci95'][0]:+.5f}, {ci_thumb['ci95'][1]:+.5f}]")
            print(f"{k:>13} {np.nanmean(v):9.5f} {np.nanmean(d):+10.5f} "
                  f"{max(per_fold[k]):11.5f} {np.nanmean(v[sp]):9.4f} "
                  f"{np.nanmean(v[~sp]):9.4f}   {cis}")
            row_out[k] = {
                "block_len": None if ci is None else ci["block_len"],
                "paired_ci_cuberoot": None if ci_thumb is None else list(ci_thumb["ci95"]),
                "qlike": float(np.nanmean(v)),
                "delta_vs_best": float(np.nanmean(d)),
                "worst_fold": float(max(per_fold[k])),
                "spike": float(np.nanmean(v[sp])),
                "calm": float(np.nanmean(v[~sp])),
                "per_fold": per_fold[k],
                "per_fold_years": [r["year"] for r in rows],
                "paired_ci": None if ci is None else list(ci["ci95"]),
                "paired_n": None if ci is None else ci["n"],
            }

        print(f"\n   --- pre-registered rule, H={H} ---")
        verdicts = {}
        for k in ("noctua", "noctua40"):
            if k not in row_out:
                verdicts[k] = "NOT EVALUABLE"
                print(f"   {k:>9} vs {best}: NOT EVALUABLE at this horizon")
                continue
            ci = row_out[k]["paired_ci"]
            v = "CLEARS" if (ci and ci[0] > 0.0) else "DOES NOT CLEAR"
            verdicts[k] = v
            print(f"   {k:>9} vs {best}: paired per-episode CI at "
                  f"{100*(1-alpha):.2f}% (blocks of {row_out[k]['block_len']}) "
                  f"{v} zero favourably")
            ct = row_out[k]["paired_ci_cuberoot"]
            if ct is not None:
                v2 = "CLEARS" if ct[0] > 0.0 else "DOES NOT CLEAR"
                if v2 != v:
                    print(f"   {'':>9}    SENSITIVE TO THE BLOCK LENGTH: the "
                          f"n^(1/3) interval {v2}. Reported, and the longer "
                          f"block governs.")
                else:
                    print(f"   {'':>9}    same verdict at the n^(1/3) block length")
        print("   the fold-level spread is reported as `per_fold` and is NOT the primary: "
              "vol-matrix-power measured it UNDERPOWERED at this horizon")
        results[str(H)] = {"best_baseline": best, "calib_qlike": cal,
                           "n_test": sum(r["n"] for r in rows),
                           "years": [r["year"] for r in rows],
                           "arms": row_out, "verdicts": verdicts,
                           "arms_absent": dropped}

    a.out.write_text(json.dumps({
        "family_size": N_FAMILY, "alpha": alpha,
        "seeds": a.seeds, "hidden": a.hidden,
        "horizons": results,
    }, indent=1, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
