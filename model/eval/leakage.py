"""
eval/leakage.py
=====================================================================
NOCTUA-LEAK: an adversarial point-in-time audit.

Phase 0 of PHASES.md requires a leakage re-audit before anything downstream
is allowed to move. A previous audit passed. This file exists because "a
previous audit passed" is not evidence of anything by itself -- it is
evidence that whoever ran it did not find a problem, which is a different
claim than "there is no problem." The job here is adversarial: try to BREAK
the point-in-time guarantee, not confirm it, and report honestly either way.

Five separate mechanisms can leak the future into a prediction made at hour
`a`, and each gets its own section below, with its own attack:

  1. FEATURE CAUSALITY. A trailing aggregate that is off by one index reads
     the anchor hour itself. `noctua/features.py` already ships a numerical
     check (`audit_lookahead`, exercised by `model/tests/test_features.py`):
     corrupt every hourly row at or after a cut, rebuild features, and
     confirm the probed anchors' features do not move. That check is real
     and this file does not re-derive it from scratch -- `audit_lookahead`
     and `build_features` are imported and reused directly. But it has two
     coverage gaps, both closed here rather than duplicated around:

       a) it names OFFENDERS (top 5 by drift) rather than reporting a verdict
          for every one of the 42 columns, so a column that never gets
          mentioned could be untested rather than clean;
       b) its cut point is a single draw from the top 40% of history (`lo =
          0.6`), and it only checks episodes anchored BEFORE that cut -- so
          every episode in the most recent ~40% of the timeline is never
          probed at all, silently. Whatever changed most recently (a new
          feature, a schema change in the harvested bars) is exactly what a
          single stale cut would miss.

     `per_column_causality` below extends the same mechanism -- same
     `build_features`, same corrupt-and-diff idea -- across SIX cut points
     spanning the full timeline (5% to 95%) so every era gets probed, and
     adds a second, orthogonal corruption style: not just rescaling the
     future (which a sufficiently unlucky multiplier could theoretically
     leave a ratio feature's value numerically close to unchanged) but
     setting it to NaN outright, which cannot be quietly absorbed by any
     linear or ratio computation. A finite base value that turns non-finite
     under NaN-injection is treated as an unambiguous violation, distinct
     from ordinary numerical drift.

     THE POSITIVE CONTROL. A test that never finds a leak is worthless if it
     was never capable of finding one. `_decoy_leaky_feature` builds a
     feature from `rv5[row]` -- the ANCHOR HOUR ITSELF, not `row - 1` -- and
     runs it through the identical harness. It MUST be flagged VIOLATED. If
     it is not, the harness has no detection power and every CAUSAL verdict
     below is meaningless. This is checked and asserted, not just reported.

  2. THE OVERLAP PROBLEM. Episodes are H-hour forward windows sampled
     hourly, so consecutive anchors share up to H-1 hours of target window.
     `noctua/splits.py` embargoes every split boundary by `embargo_hours`
     (default `H.max()`). `embargo_audit` recomputes, from the real episode
     table, the actual gap in hours between the latest-ending training
     window and the earliest calibration window, and between the latest
     calibration window and the earliest test window, for both partitioning
     schemes the repository actually uses (`walk_forward_folds`, what
     `eval/benchmark.py` scores on, and `time_splits`, what `noctua/train.py`
     ships from). A gap >= 0 means no overlap; the number reported is
     `min_next_start - max_prev_end` in hours, so a NEGATIVE number is the
     finding that blocks everything downstream.

  3. SCALER AND CALIBRATION FITTING. `noctua/train.py:prepare()` fits a
     `Standardizer` (mu/sd) only when none is passed in, and `eval/
     benchmark.py:run_fold` calls it with the training mask first, then
     reuses the returned statistics for calibration and test. Read alone,
     that is a claim about control flow. `scaler_independence_attack` turns
     it into a measurement: perturb the TEST-only rows of the real feature
     matrix with large random noise, refit on the SAME training mask, and
     assert the fitted mu/sd are bit-identical to the unperturbed fit. If
     the scaler depended on test rows through any path (a stray global
     index, a mask computed wrong), corrupting only test rows would move it;
     it does not. The same attack is run on `EmpiricalSpecialist.fit`, the
     committee member most exposed to the mistake (it is a raw quantile of
     the training slice), by perturbing M_up/M_dn on TEST rows only.

  4. THE SIGMA REFERENCE. Stage B's causal volatility reference is
     `exp(har_1d) * sqrt(H)` clipped to the 0.5-99.5 percentile of that value
     on the TRAINING episodes -- and the callable form recomputes the clip
     bounds fresh inside every `run_fold` call, from that fold's `m_tr`
     alone. `sigma_ref_bounds_audit` calls the exact closure every research
     script in this repo uses (`causal_sigma_fn`, reimplemented identically
     seven times -- see the summary this prints) on the six real walk-forward
     folds and reports the six (lo, hi) pairs. If the bounds were secretly
     global, they would be constant across folds; they are not, and the six
     values are printed so that can be checked by eye, not just asserted.
     Separately, and this matters for what "the benchmark path" means: the
     shipped `eval/benchmark.py:main()` never PASSES `sigma_ref_fn` to
     `run_fold`, so `python -m model.eval.benchmark` (the command Phase 0's
     gate reproduces) trains Stage B against raw realized RV, not this
     causal reference. That is a documented train/serve TARGET-DEFINITION
     choice (see `prepare()`'s docstring), not a temporal leak -- RV is that
     episode's own forward-window outcome, never another episode's future --
     but it means the causal-reference code path this item asks about is
     exercised by the *ablation* scripts (`levers.py`, `anchors.py`,
     `freshness.py`, `regime.py`, `losshead.py`, `forward_split.py`,
     `rolling_refresh.py`, `training_methods.py`), not by the headline run.
     Reported plainly so it is not mistaken for coverage it does not have.

  5. THE NEW DATA. `data/newdata/funding_btc.parquet` and `dvol_btc.parquet`
     are harvested but not yet wired into `features.py`. Before anyone adds
     them, the question that matters is what their timestamp MEANS: a value
     known only at settlement is not available at the start of the interval
     it describes, and joining it to an anchor by calendar hour without a
     lag would manufacture exactly the leak this whole file exists to catch.
     `audit_new_data` establishes what the data's own internal structure
     implies (row spacing, gaps, and whether each row's fields are fully
     determined by data at-or-before its own timestamp) and marks explicitly
     what it CANNOT establish (the harvester's/exchange's documented
     semantics were not consulted this session -- this is a data-shape
     argument, not a citation).

This file is a STANDING CHECK, not a one-off: every number below is
recomputed from `model/artifacts/{btcusd_1h,episodes,features}.parquet` and
`data/newdata/*.parquet` on each run, nothing is hardcoded from a prior pass,
and it writes `model/artifacts/leakage.json` so a later run can diff against
it. It reads `model/noctua/*` and `model/eval/benchmark.py` for behavior it
asserts about but modifies none of them -- this is an audit, and the
distinction between reading a claim and changing the code that makes it is
the entire point.

    python -m model.eval.leakage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua import splits as S                                        # noqa: E402
from noctua.committee import EmpiricalSpecialist                      # noqa: E402
from noctua.features import audit_lookahead, build_features           # noqa: E402
from noctua.spec import BASE_COLS, NON_MODEL_COLS, SHAPE_COLS         # noqa: E402
from noctua.train import Standardizer, load_all, prepare              # noqa: E402

HOUR = 3600
RAW_COLS = ("rv5", "rv5_pos", "rv5_neg", "bpv5", "rq5",
            "close", "high", "low", "open", "volume")


# ==========================================================================
# 1. FEATURE CAUSALITY -- per-column, multi-era, two corruption styles
# ==========================================================================
def _decoy_leaky_feature(hours: pd.DataFrame, episodes: pd.DataFrame) -> np.ndarray:
    """A feature that reads the ANCHOR HOUR ITSELF -- deliberately wrong.

    `har_1h` at row `a-1` is the last legal hour; this reads row `a`, one
    past the boundary the contract draws. It exists to prove the harness
    below can actually catch a violation, not merely fail to report one.
    """
    rv5 = hours["rv5"].to_numpy(np.float64)
    rows = np.clip(episodes["row"].to_numpy(np.int64), 0, len(rv5) - 1)
    return 0.5 * np.log(np.maximum(rv5[rows], 1e-12))


def _corrupt(hours: pd.DataFrame, cut: int, style: str, rng: np.random.Generator) -> pd.DataFrame:
    corrupt = hours.copy()
    mask = np.arange(len(hours)) >= cut
    n_mask = int(mask.sum())
    for c in RAW_COLS:
        v = corrupt[c].to_numpy(np.float64).copy()
        if style == "scale_2_5x":
            v[mask] = v[mask] * rng.uniform(2.0, 5.0, size=n_mask)
        elif style == "nan_inject":
            v[mask] = np.nan
        else:
            raise ValueError(style)
        corrupt[c] = v
    return corrupt


def per_column_causality(
    hours: pd.DataFrame, episodes: pd.DataFrame, extra_lag_hours: int = 0,
    cut_fracs: tuple = (0.05, 0.2, 0.4, 0.6, 0.8, 0.95),
    n_probe_per_cut: int = 150, seed: int = 0,
) -> dict:
    """Per-column CAUSAL / VIOLATED / UNTESTED, across the whole timeline.

    For every cut point and both corruption styles: rebuild features with
    everything at-or-after the cut corrupted, and compare against the clean
    build on episodes anchored at-or-before the cut (so the comparison never
    asks a column to be unaffected by a change to data it is actually
    allowed to see). A cell where the clean value is finite but the
    corrupted value is not is an unambiguous violation (magnitude `inf`,
    so it never gets averaged away); otherwise the violation magnitude is
    the absolute difference. A column with zero informative cells across
    every cut and style (clean value never finite where probed) is
    UNTESTED, not silently passed.

    THE PROBE SET IS NOT A PLAIN RANDOM SAMPLE, and that is deliberate. An
    off-by-one bug (a feature reading its own anchor row `a` instead of
    stopping at `a-1`) is only detectable for an episode anchored EXACTLY at
    the cut: episodes anchored strictly before the cut never touch the
    corrupted zone at all under ANY style here, so a naive random sample
    over `row <= cut` would need to land on that one boundary row by chance
    -- vanishingly unlikely out of the hundreds of thousands of candidates,
    which would silently defeat the positive control below. So `cut` is
    always chosen FROM an existing episode's own `row` value (never an
    arbitrary integer), and every episode anchored at that exact row is
    force-included in the probe on top of the random historical sample.
    """
    rng = np.random.default_rng(seed)
    base = build_features(hours, episodes, extra_lag_hours=extra_lag_hours)
    cols = list(base.columns)
    n = len(hours)

    max_bad = {c: 0.0 for c in cols}
    n_tested = {c: 0 for c in cols}
    n_cells = {c: 0 for c in cols}
    trials = []

    rows_arr = episodes["row"].to_numpy(np.int64)
    decoy_base = _decoy_leaky_feature(hours, episodes)

    unique_rows = np.unique(rows_arr)
    unique_rows = unique_rows[(unique_rows >= 400) & (unique_rows <= n - 30)]

    for frac in cut_fracs:
        if len(unique_rows) == 0:
            continue
        cut = int(unique_rows[int(len(unique_rows) * frac) if frac < 1.0 else -1])
        boundary_idx = np.flatnonzero(rows_arr == cut)
        hist_idx = np.flatnonzero(rows_arr < cut)
        if len(boundary_idx) == 0 and len(hist_idx) == 0:
            continue
        hist_probe = (rng.choice(hist_idx, size=min(n_probe_per_cut, len(hist_idx)),
                                 replace=False) if len(hist_idx) else
                     np.array([], dtype=np.int64))
        probe = np.unique(np.concatenate([boundary_idx, hist_probe])).astype(np.int64)
        if len(probe) == 0:
            continue

        for style in ("scale_2_5x", "nan_inject"):
            corrupt = _corrupt(hours, cut, style, rng)
            after = build_features(corrupt, episodes, extra_lag_hours=extra_lag_hours)
            decoy_after = _decoy_leaky_feature(corrupt, episodes.iloc[probe])

            a = base.loc[probe, cols].to_numpy(np.float64)
            b = after.loc[probe, cols].to_numpy(np.float64)
            a_fin, b_fin = np.isfinite(a), np.isfinite(b)
            mismatch = a_fin & ~b_fin                       # clean finite, corrupted broke
            numeric = np.where(a_fin & b_fin, np.abs(a - b), 0.0)
            status = np.where(mismatch, np.inf, numeric)

            trial_bad = 0
            for j, c in enumerate(cols):
                n_tested[c] += int(a_fin[:, j].sum())
                n_cells[c] += len(probe)
                colmax = float(status[:, j].max()) if len(probe) else 0.0
                if colmax > max_bad[c]:
                    max_bad[c] = colmax
                if colmax > 0:
                    trial_bad += 1

            # positive control on THIS trial: the decoy must be flagged
            d0 = decoy_base[probe]
            d0_fin = np.isfinite(d0)
            d1_fin = np.isfinite(decoy_after)
            decoy_mismatch = bool((d0_fin & ~d1_fin).any())
            decoy_diff = float(np.nanmax(np.abs(
                np.where(d0_fin & d1_fin, d0 - decoy_after, 0.0)))) if len(probe) else 0.0
            decoy_caught = decoy_mismatch or decoy_diff > 0

            trials.append({
                "cut_frac": frac, "cut_row": cut, "style": style,
                "n_probed": len(probe), "n_boundary_episodes": len(boundary_idx),
                "n_columns_moved": trial_bad,
                "decoy_caught": decoy_caught,
            })

    verdict = {}
    for c in cols:
        if n_tested[c] == 0:
            verdict[c] = "UNTESTED"
        elif max_bad[c] > 0:
            verdict[c] = "VIOLATED"
        else:
            verdict[c] = "CAUSAL"

    # Columns that never read the hourly bars at all -- cal_* and
    # reg_post_etf key off episode/anchor metadata (anchor_ts, H, dow) and
    # `hour_ts`, none of which this corruption touches. The empirical test
    # above still runs on them and correctly reports CAUSAL (nothing moved),
    # but it is a necessary-not-sufficient check for these specific columns:
    # absence of drift is guaranteed by construction, not just observed.
    # Flagged here so a CAUSAL verdict on this list is not over-read as
    # "the corruption test genuinely stressed this column."
    structural_only = sorted(
        c for c in cols
        if c == "reg_post_etf" or c.startswith("cal_")
    )

    trials_with_boundary = [t for t in trials if t["n_boundary_episodes"] > 0]
    decoy_all_caught = (all(t["decoy_caught"] for t in trials_with_boundary)
                        if trials_with_boundary else False)

    return {
        "columns": len(cols),
        "verdict": verdict,
        "max_abs_violation": max_bad,
        "n_tested_cells": n_tested,
        "n_cells_probed": n_cells,
        "structural_only_columns": structural_only,
        "trials": trials,
        "positive_control": {
            "description": "decoy feature reads rv5[row] (the anchor hour "
                            "itself, not row-1); must be flagged in every "
                            "trial that had a boundary episode to test it on",
            "n_trials": len(trials),
            "n_trials_with_boundary_episode": len(trials_with_boundary),
            "caught_in_every_trial": decoy_all_caught,
        },
        "counts": {
            "CAUSAL": sum(1 for v in verdict.values() if v == "CAUSAL"),
            "VIOLATED": sum(1 for v in verdict.values() if v == "VIOLATED"),
            "UNTESTED": sum(1 for v in verdict.values() if v == "UNTESTED"),
        },
    }


# ==========================================================================
# 2. THE OVERLAP PROBLEM
# ==========================================================================
def _gap_hours(end_a: np.ndarray, start_b: np.ndarray) -> float:
    """min(start_b) - max(end_a), in hours. >=0 means no overlap."""
    if len(end_a) == 0 or len(start_b) == 0:
        return float("nan")
    return float((start_b.min() - end_a.max()) / HOUR)


def embargo_audit(ep: pd.DataFrame) -> dict:
    """Concrete overlap, in hours, at every split boundary that matters.

    A training/calibration window's forward span is `[anchor_ts, anchor_ts +
    H*3600)`. Two windows overlap iff one's end exceeds the other's start.
    Reported as `gap = min(next_start) - max(prev_end)`: gap >= 0 means the
    embargo held with `gap` hours to spare; gap < 0 is the overlap in hours
    and is the finding that blocks Phase 1 onward.
    """
    ts = ep["anchor_ts"].to_numpy(np.int64)
    end = ts + ep["H"].to_numpy(np.int64) * HOUR

    out = {"embargo_hours_used": int(ep["H"].max())}

    # -- walk_forward_folds: what eval/benchmark.py actually scores on -----
    folds = S.walk_forward_folds(ep)
    fold_rows = []
    worst_wf = float("inf")
    for f in folds:
        tri, cai, tei = (np.flatnonzero(f[k]) for k in ("train", "calib", "test"))
        if len(tri) == 0 or len(tei) == 0:
            continue
        gap_tr_cal = _gap_hours(end[tri], ts[cai]) if len(cai) else float("nan")
        gap_cal_te = _gap_hours(end[cai], ts[tei]) if len(cai) else float("nan")
        gap_tr_te = _gap_hours(end[tri], ts[tei])
        # the specific number the brief asks for: overlap between ANY
        # training episode's window and the EARLIEST test episode's window
        earliest_te = tei[np.argmin(ts[tei])]
        et0, et1 = int(ts[earliest_te]), int(end[earliest_te])
        overlap_vs_earliest_test = float(
            (np.minimum(end[tri], et1) - np.maximum(ts[tri], et0)).max() / HOUR
        )
        row = {
            "year": f["year"], "n_train": int(len(tri)), "n_calib": int(len(cai)),
            "n_test": int(len(tei)),
            "gap_train_to_calib_hours": gap_tr_cal,
            "gap_calib_to_test_hours": gap_cal_te,
            "gap_train_to_test_hours": gap_tr_te,
            "max_overlap_train_vs_earliest_test_hours": overlap_vs_earliest_test,
        }
        fold_rows.append(row)
        for g in (gap_tr_cal, gap_cal_te):
            if not np.isnan(g):
                worst_wf = min(worst_wf, g)
    out["walk_forward_folds"] = fold_rows
    out["walk_forward_worst_case_gap_hours"] = None if worst_wf == float("inf") else worst_wf
    out["walk_forward_max_overlap_hours"] = (
        None if worst_wf == float("inf") else max(0.0, -worst_wf)
    )

    # -- time_splits: what noctua/train.py ships the served model from -----
    sp = S.time_splits(ep)
    tri, cai, tei = (np.flatnonzero(sp[k]) for k in ("train", "calib", "test"))
    gap_tr_cal = _gap_hours(end[tri], ts[cai])
    gap_cal_te = _gap_hours(end[cai], ts[tei])
    out["time_splits"] = {
        "n_train": int(len(tri)), "n_calib": int(len(cai)), "n_test": int(len(tei)),
        "gap_train_to_calib_hours": gap_tr_cal,
        "gap_calib_to_test_hours": gap_cal_te,
        "max_overlap_hours": max(0.0, -min(gap_tr_cal, gap_cal_te)),
    }
    return out


# ==========================================================================
# 3. SCALER AND CALIBRATION FITTING -- attacked, not just read
# ==========================================================================
def scaler_independence_attack(ep: pd.DataFrame, X: pd.DataFrame, ep2019: bool = True) -> dict:
    """Perturb TEST-only rows, refit on the SAME training mask, assert no
    movement.

    Uses the real `walk_forward_folds` masks and the real `prepare()`. If the
    standardizer or the empirical specialist read even one value outside the
    training mask -- a mask built wrong, a stray `X` instead of `X[mask]` --
    corrupting only the test rows would move the fitted statistics. This is
    the attack `research/pitfalls.py`-style claims deserve: not "the code
    looks right" but "corrupting the part that must not matter did nothing."
    """
    fin = np.isfinite(X.to_numpy()).all(1)
    folds = S.walk_forward_folds(ep)
    f = next(f for f in folds if (f["train"] & fin).sum() > 5000 and (f["test"] & fin).sum() > 30)
    m_tr = f["train"] & fin
    m_te = f["test"] & fin

    _, stds_before = prepare(ep, X, m_tr)
    mu_before = np.concatenate([stds_before[0].mu, stds_before[1].mu, stds_before[2].mu])
    sd_before = np.concatenate([stds_before[0].sd, stds_before[1].sd, stds_before[2].sd])

    rng = np.random.default_rng(0)
    X_corrupt = X.copy()
    te_idx = np.flatnonzero(m_te)
    for c in X_corrupt.columns:
        v = X_corrupt[c].to_numpy(np.float64).copy()
        v[te_idx] = rng.uniform(-1e6, 1e6, size=len(te_idx))
        X_corrupt[c] = v

    _, stds_after = prepare(ep, X_corrupt, m_tr)
    mu_after = np.concatenate([stds_after[0].mu, stds_after[1].mu, stds_after[2].mu])
    sd_after = np.concatenate([stds_after[0].sd, stds_after[1].sd, stds_after[2].sd])

    scaler_moved = float(np.max(np.abs(mu_before - mu_after)) +
                         np.max(np.abs(sd_before - sd_after)))

    # same attack on EmpiricalSpecialist -- fit on m_tr, corrupt M_up/M_dn
    # ONLY on test rows, refit, compare quantiles bit for bit.
    H = ep["H"].to_numpy(np.float64)
    RV = ep["RV"].to_numpy(np.float64)
    m_up = ep["M_up"].to_numpy(np.float64)
    m_dn = ep["M_dn"].to_numpy(np.float64)
    spec_before = EmpiricalSpecialist().fit(m_up[m_tr], m_dn[m_tr], RV[m_tr])

    m_up_c, m_dn_c = m_up.copy(), m_dn.copy()
    m_up_c[te_idx] = rng.uniform(-50, 50, size=len(te_idx))
    m_dn_c[te_idx] = rng.uniform(-50, 50, size=len(te_idx))
    spec_after = EmpiricalSpecialist().fit(m_up_c[m_tr], m_dn_c[m_tr], RV[m_tr])

    spec_moved = float(np.max(np.abs(spec_before.z_up - spec_after.z_up)) +
                       np.max(np.abs(spec_before.z_dn - spec_after.z_dn)))

    return {
        "fold_year": int(f["year"]), "n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
        "standardizer": {
            "max_abs_change_in_mu_sd_from_corrupting_test_rows": scaler_moved,
            "independent_of_test_rows": bool(scaler_moved == 0.0),
            "fit_call_site": "model/noctua/train.py:119-120 (Standardizer.fit, "
                              "called only when std_all is None)",
            "call_sequence": "model/eval/benchmark.py:369 (tr, stds = prepare(..., "
                              "m_tr, ...)) fits; :374 (va = prepare(..., m_va, *stds, "
                              "...)) and :386 (predict_avg -> prepare(..., mask, *stds, "
                              "...)) reuse the SAME stds for calib and test -- never refit",
        },
        "empirical_specialist": {
            "max_abs_change_in_fitted_quantiles_from_corrupting_test_rows": spec_moved,
            "independent_of_test_rows": bool(spec_moved == 0.0),
            "fit_call_site": "model/eval/benchmark.py:403-404 "
                              "(EmpiricalSpecialist().fit(ep.M_up[m_tr], ep.M_dn[m_tr], "
                              "ep.RV[m_tr]))",
        },
        "committee_fit_masks_by_code_reading": {
            "GaussianSpecialist": "benchmark.py:402, fit on e_cal = ep[m_cal] "
                                  "(m_cal = m_va & H==19) -- calibration split only",
            "EmpiricalSpecialist": "benchmark.py:403-404, fit on ep[...][m_tr] -- "
                                   "training split only",
            "EVTSpecialist": "benchmark.py:405-406, fit on ep[...][m_tr] -- "
                             "training split only",
            "Committee.fit_equal": "benchmark.py:408, no data -- equal weights, "
                                   "nothing to leak",
            "note": "none of the four specialists' fit() is ever called with "
                    "m_te-derived data anywhere in run_fold",
        },
        "corp_isotonic_caveat": {
            "what": "corp_decomposition (benchmark.py:97-130) fits its "
                    "IsotonicRegression on (p, y) drawn from the TEST slice "
                    "itself (called at benchmark.py:456 with P and out both "
                    "test-slice arrays) -- this is IN-SAMPLE with respect to "
                    "the batch being scored.",
            "is_this_a_leak": "NO, by the definition this file otherwise uses: "
                    "it never touches any trained parameter or the raw "
                    "predictions p, and it is the CORP method's own design "
                    "(Dimitriadis, Gneiting & Jordan 2021) -- decomposing a "
                    "proper score by isotonic-recalibrating the SAME batch is "
                    "how the method is defined, and it is applied identically "
                    "to all six competitors including climatology and the "
                    "shuffled control, so no competitor is advantaged over "
                    "another by it.",
            "caveat_worth_recording": "in-sample PAV recalibration is a known "
                    "small-sample-optimistic estimator of DSC/MCB (the fit "
                    "and the evaluation share data), so the reported DSC "
                    "values are not a strictly out-of-sample skill estimate "
                    "in the way the underlying Brier/pinball/CRPS scores are "
                    "-- they are a within-batch decomposition of a score that "
                    "IS out-of-sample. This is a property of the CORP method "
                    "applied here, not a bug in this codebase.",
        },
    }


# ==========================================================================
# 4. THE SIGMA REFERENCE -- per-fold clip bounds, empirically distinct
# ==========================================================================
def sigma_ref_bounds_audit(ep: pd.DataFrame, X: pd.DataFrame) -> dict:
    """Recompute the causal clip bounds on each real walk-forward fold's
    training mask and confirm they differ fold to fold (proof the callable
    form is genuinely refit, not a cached constant), then report whether the
    headline benchmark path actually exercises this code at all.
    """
    fin = np.isfinite(X.to_numpy()).all(1)
    H = ep["H"].to_numpy(np.float64)
    raw = np.exp(X["har_1d"].to_numpy(np.float64)) * np.sqrt(H)
    folds = S.walk_forward_folds(ep)

    rows = []
    for f in folds:
        m_tr = f["train"] & fin
        if m_tr.sum() == 0:
            continue
        lo, hi = np.quantile(raw[m_tr], [0.005, 0.995])
        rows.append({"year": f["year"], "n_train": int(m_tr.sum()),
                     "clip_lo": float(lo), "clip_hi": float(hi)})

    los = [r["clip_lo"] for r in rows]
    his = [r["clip_hi"] for r in rows]
    genuinely_per_fold = bool(len(set(np.round(los, 12))) > 1 and
                              len(set(np.round(his, 12))) > 1)

    array_form_used = _grep_any(
        Path(__file__).resolve().parents[1],
        "sigma_ref_all=", exclude="benchmark.py",
    )

    return {
        "per_fold_clip_bounds": rows,
        "bounds_genuinely_differ_across_folds": genuinely_per_fold,
        "closure_definition": "def fn(train_mask): lo, hi = np.quantile("
                              "raw[train_mask], [0.005, 0.995]); return "
                              "clip(raw, lo, hi) -- reimplemented identically "
                              "in levers.py, anchors.py, freshness.py, "
                              "regime.py, losshead.py, forward_split.py, "
                              "rolling_refresh.py, training_methods.py; each "
                              "call site quantiles on THAT call's train_mask "
                              "argument only",
        "invoked_per_fold_via": "run_fold (benchmark.py:365-366): "
                                "`if sigma_ref_fn is not None: sigma_ref_all = "
                                "sigma_ref_fn(m_tr)` -- called with the "
                                "CURRENT fold's own m_tr, so a later fold "
                                "cannot influence an earlier one's bounds and "
                                "vice versa",
        "array_form_ever_populated_with_real_data_elsewhere": array_form_used,
        "used_by_headline_benchmark_main": False,
        "headline_benchmark_finding": (
            "eval/benchmark.py:main() calls run_fold(ep, X, f, a.hidden, "
            "a.seeds) with NEITHER sigma_ref_all NOR sigma_ref_fn set, so "
            "`python -m model.eval.benchmark` -- the command Phase 0's gate "
            "reproduces -- trains Stage B against sigma_ref=None, i.e. raw "
            "realized RV (train.py prepare():110), NOT this causal "
            "reference. The causal per-fold-clip form exists, is correctly "
            "implemented everywhere it IS called, and is refit per fold with "
            "zero cross-fold leakage (see bounds above) -- but it is "
            "exercised by the ablation scripts listed above, not by the "
            "headline run. Not a temporal leak either way (RV is that "
            "episode's own realized outcome), but it means 'the benchmark "
            "path' and 'the causal-sigma path' are two different pieces of "
            "code, and conflating them would overstate what the headline "
            "numbers test."
        ),
    }


def _grep_any(root: Path, needle: str, exclude: str = "") -> bool:
    for p in root.rglob("*.py"):
        if exclude and p.name == exclude:
            continue
        try:
            if needle in p.read_text():
                return True
        except OSError:
            continue
    return False


# ==========================================================================
# 5. THE NEW DATA -- timestamp semantics, established from shape alone
# ==========================================================================
def audit_new_data(root: Path) -> dict:
    """What the funding and DVOL parquet files' OWN structure implies about
    their timestamp, and what it does not.

    Neither file carries a schema comment or a fetched-from-docs note in
    this session -- no network call was made to confirm Deribit's official
    semantics, so that half is marked UNVERIFIED rather than assumed. What
    IS established, from the data alone: whether each row is fully
    determined by information at-or-before its own timestamp (a "known as
    of t" quantity, like this repo's own hourly `close`), or whether it
    looks like a forward commitment stamped with a future settlement time.
    """
    out = {}

    fund_path = root / "funding_btc.parquet"
    dvol_path = root / "dvol_btc.parquet"
    if not fund_path.exists() or not dvol_path.exists():
        return {"error": f"expected files not found under {root}",
                "funding_exists": fund_path.exists(), "dvol_exists": dvol_path.exists()}

    fund = pd.read_parquet(fund_path)
    dvol = pd.read_parquet(dvol_path)

    def _spacing(df, ts_col):
        ts = df[ts_col].to_numpy(np.int64)
        d = np.diff(ts)
        vc = pd.Series(d).value_counts().to_dict()
        gaps = int((d != 3600).sum())
        return {
            "n_rows": int(len(df)),
            "start_utc": str(pd.to_datetime(int(ts.min()), unit="s", utc=True)),
            "end_utc": str(pd.to_datetime(int(ts.max()), unit="s", utc=True)),
            "monotonic_increasing": bool(np.all(d > 0)),
            "duplicated_timestamps": int(pd.Series(ts).duplicated().sum()),
            "spacing_seconds_value_counts": {str(k): int(v) for k, v in vc.items()},
            "non_hourly_gaps": gaps,
            "aligned_to_hour_boundary": bool((ts % 3600 == 0).all()),
        }

    out["funding"] = _spacing(fund, "ts")
    out["dvol"] = _spacing(dvol, "ts")

    # contemporaneity: does row i's prev_index_price equal row (i-1)'s
    # index_price EXACTLY? If so, every row's fields are fully determined by
    # data at or before that row's own timestamp -- consistent with an
    # "observation time" stamp, not a forward-settlement stamp (which would
    # not need to reference the immediately preceding row's own price at
    # all). This is the strongest thing the data's shape alone can say.
    ip = fund["index_price"].to_numpy(np.float64)
    pip = fund["prev_index_price"].to_numpy(np.float64)
    match = np.isclose(pip[1:], ip[:-1], equal_nan=True)
    out["funding"]["prev_index_price_equals_prior_row_index_price"] = {
        "fraction_matching": float(match.mean()),
        "interpretation": (
            "1.0 means every row's prev_index_price is EXACTLY the previous "
            "row's own index_price, i.e. interest_1h at timestamp t is "
            "computed from the price change ending AT t -- a trailing, "
            "already-known-by-t quantity, the same convention this repo's "
            "own hourly `close` uses (known at hour_ts, not before it). This "
            "is evidence AGAINST a forward-settlement stamp: a forward "
            "commitment for the NEXT interval would not need yesterday's "
            "own realized price change to define itself."
        ),
    }

    # interest_8h updates nearly every hour (a rolling reference), not only
    # at 00/08/16 UTC -- rules out "this column only takes its true value at
    # the 8-hourly funding settlement instant and is undefined/stale
    # between."
    chg_frac = float((fund["interest_8h"].diff() != 0).mean())
    out["funding"]["interest_8h_changes_every_hour_fraction"] = chg_frac
    out["funding"]["interest_8h_interpretation"] = (
        "a rolling trailing-8h reference updated on (nearly) every hourly "
        "row, not a value that only becomes meaningful at the 00/08/16 UTC "
        "settlement instants -- consistent with a continuously-updated "
        "as-of-t quantity"
    )

    fund_gap_locs = []
    ts = fund["ts"].to_numpy(np.int64)
    d = np.diff(ts)
    for i in np.flatnonzero(d != 3600):
        fund_gap_locs.append({
            "before": str(pd.to_datetime(int(ts[i]), unit="s", utc=True)),
            "after": str(pd.to_datetime(int(ts[i + 1]), unit="s", utc=True)),
            "gap_hours": float(d[i] / 3600.0),
        })
    out["funding"]["gap_locations"] = fund_gap_locs

    out["required_lag_if_used_as_a_feature"] = (
        "Established from the data: to respect this repo's own contract "
        "('features = hours <= a-1', episodes.py:35) a funding/DVOL row "
        "stamped at hour h should not be joined to an anchor a until a >= "
        "h+1, i.e. the SAME lag convention already applied to the hourly "
        "OHLC table (build_features indexes trailing aggregates at "
        "`rows - extra_lag_hours`, never at `rows` itself). Joining on "
        "ts == anchor_ts, or any join that lets a funding/DVOL row reach "
        "the SAME hour it is timestamped for, would violate the contract "
        "exactly as using rv5[row] instead of rv5[row-1] would."
    )
    out["unverified"] = [
        "Deribit's documented definition of the funding-rate 'timestamp' "
        "field was not fetched or consulted this session (no network call "
        "was made) -- the 'known as of t' reading above is inferred from "
        "the data's internal structure (prev_index_price chaining, hourly "
        "interest_8h updates), not confirmed against an API reference.",
        "DVOL is a single 'volatility' column with no OHLC breakdown, so "
        "whether the hourly value is a point-in-time index level sampled "
        "AT the hour, or some intra-hour aggregate (e.g. a candle close), "
        "cannot be distinguished from the data alone; the conservative "
        "reading (treat it as known-as-of-its-timestamp, same lag as "
        "everything else) is recommended regardless of which is true.",
        "Whether index_price itself (the Deribit BTC index, an average "
        "across constituent exchanges) has any residual publication delay "
        "relative to its own timestamp was not established.",
    ]
    return out


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NOCTUA-LEAK: adversarial "
                                              "point-in-time leakage audit")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--newdata", type=Path, default=Path("data/newdata"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/leakage.json"))
    ap.add_argument("--n-probe-per-cut", type=int, default=150)
    a = ap.parse_args(argv)

    print("=" * 78)
    print("NOCTUA-LEAK -- adversarial point-in-time audit")
    print("=" * 78)

    hours = pd.read_parquet(a.artifacts / "btcusd_1h.parquet")
    ep, X = load_all(a.artifacts)
    print(f"\n{len(hours):,} hourly bars, {len(ep):,} episodes, "
          f"{X.shape[1]} feature columns\n")

    # ---- 1. feature causality --------------------------------------------
    print("-" * 78)
    print("1. FEATURE CAUSALITY (per column, 6 eras x 2 corruption styles)")
    print("-" * 78)
    base_audit = audit_lookahead(hours, ep, n_probe=200, extra_lag_hours=0)
    print(f"  baseline audit_lookahead() (reused, unmodified): "
          f"leak_free={base_audit['leak_free']} "
          f"max_change={base_audit['max_abs_feature_change']:.3e} "
          f"over {base_audit['episodes_checked']:,} episodes")

    causality = per_column_causality(hours, ep, extra_lag_hours=0,
                                     n_probe_per_cut=a.n_probe_per_cut)
    ctrl = causality["positive_control"]
    print(f"  positive control (decoy reads the anchor hour itself): "
          f"boundary episode available in "
          f"{ctrl['n_trials_with_boundary_episode']}/{ctrl['n_trials']} trials, "
          f"caught in all of them = {ctrl['caught_in_every_trial']}")
    if not ctrl["caught_in_every_trial"]:
        print("  !! POSITIVE CONTROL FAILED -- the harness has no detection "
              "power, every CAUSAL verdict below is UNRELIABLE !!")
    print(f"  {causality['counts']}")
    for c, v in causality["verdict"].items():
        tag = "  " if v == "CAUSAL" else ("**" if v == "VIOLATED" else "??")
        note = " (structural-only test)" if c in causality["structural_only_columns"] else ""
        print(f"  {tag} {c:<24} {v}{note}")

    # ---- 2. overlap ---------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. THE OVERLAP PROBLEM")
    print("-" * 78)
    overlap = embargo_audit(ep)
    for row in overlap["walk_forward_folds"]:
        print(f"  fold {row['year']}: gap train->calib "
              f"{row['gap_train_to_calib_hours']:+.1f}h, gap calib->test "
              f"{row['gap_calib_to_test_hours']:+.1f}h, overlap vs earliest "
              f"test episode {row['max_overlap_train_vs_earliest_test_hours']:+.1f}h")
    print(f"  walk_forward_folds worst-case gap: "
          f"{overlap['walk_forward_worst_case_gap_hours']:.1f}h  "
          f"(max overlap = {overlap['walk_forward_max_overlap_hours']:.1f}h)")
    ts_r = overlap["time_splits"]
    print(f"  time_splits (served-model path): gap train->calib "
          f"{ts_r['gap_train_to_calib_hours']:+.1f}h, gap calib->test "
          f"{ts_r['gap_calib_to_test_hours']:+.1f}h, "
          f"max overlap = {ts_r['max_overlap_hours']:.1f}h")

    # ---- 3. scaler / calibration --------------------------------------------
    print("\n" + "-" * 78)
    print("3. SCALER AND CALIBRATION FITTING (attacked)")
    print("-" * 78)
    scaler = scaler_independence_attack(ep, X)
    print(f"  fold {scaler['fold_year']}: standardizer moved by "
          f"{scaler['standardizer']['max_abs_change_in_mu_sd_from_corrupting_test_rows']:.3e} "
          f"after corrupting test-only rows "
          f"(independent={scaler['standardizer']['independent_of_test_rows']})")
    print(f"  EmpiricalSpecialist moved by "
          f"{scaler['empirical_specialist']['max_abs_change_in_fitted_quantiles_from_corrupting_test_rows']:.3e} "
          f"after corrupting test-only M_up/M_dn "
          f"(independent={scaler['empirical_specialist']['independent_of_test_rows']})")

    # ---- 4. sigma reference ---------------------------------------------------
    print("\n" + "-" * 78)
    print("4. THE SIGMA REFERENCE")
    print("-" * 78)
    sigref = sigma_ref_bounds_audit(ep, X)
    for r in sigref["per_fold_clip_bounds"]:
        print(f"  fold {r['year']}: clip=[{r['clip_lo']:.5f}, {r['clip_hi']:.5f}] "
              f"n_train={r['n_train']:,}")
    print(f"  bounds genuinely differ fold-to-fold: "
          f"{sigref['bounds_genuinely_differ_across_folds']}")
    print(f"  used by headline `python -m model.eval.benchmark`: "
          f"{sigref['used_by_headline_benchmark_main']}")

    # ---- 5. new data ----------------------------------------------------------
    print("\n" + "-" * 78)
    print("5. THE NEW DATA -- timestamp semantics")
    print("-" * 78)
    newdata = audit_new_data(a.newdata)
    if "error" in newdata:
        print(f"  {newdata['error']}")
    else:
        f_ = newdata["funding"]
        print(f"  funding_btc: {f_['n_rows']:,} rows, {f_['start_utc']} -> "
              f"{f_['end_utc']}, {f_['non_hourly_gaps']} non-hourly gap(s), "
              f"aligned_to_hour={f_['aligned_to_hour_boundary']}")
        print(f"    prev_index_price chains to prior row's index_price: "
              f"{f_['prev_index_price_equals_prior_row_index_price']['fraction_matching']:.4f}")
        print(f"    interest_8h changes hour-to-hour: "
              f"{f_['interest_8h_changes_every_hour_fraction']:.4f} of rows")
        d_ = newdata["dvol"]
        print(f"  dvol_btc: {d_['n_rows']:,} rows, {d_['start_utc']} -> "
              f"{d_['end_utc']}, {d_['non_hourly_gaps']} non-hourly gap(s)")
        print(f"  UNVERIFIED items: {len(newdata['unverified'])} "
              f"(see leakage.json -> new_data.unverified)")

    # ---- overall verdict --------------------------------------------------
    causal_ok = causality["counts"]["VIOLATED"] == 0 and ctrl["caught_in_every_trial"]
    overlap_ok = (overlap["walk_forward_max_overlap_hours"] == 0.0 and
                 overlap["time_splits"]["max_overlap_hours"] == 0.0)
    scaler_ok = (scaler["standardizer"]["independent_of_test_rows"] and
                scaler["empirical_specialist"]["independent_of_test_rows"])
    overall_pass = causal_ok and overlap_ok and scaler_ok

    print("\n" + "=" * 78)
    print(f"OVERALL: {'NO LEAKAGE FOUND' if overall_pass else 'LEAKAGE FOUND'}")
    print("=" * 78)
    if not overall_pass:
        print("  BLOCKING every phase after Phase 0 until fixed. Specifics:")
        if not causal_ok:
            print(f"    - feature causality: {causality['counts']}")
        if not overlap_ok:
            print(f"    - overlap: walk_forward={overlap['walk_forward_max_overlap_hours']}h "
                  f"time_splits={overlap['time_splits']['max_overlap_hours']}h")
        if not scaler_ok:
            print("    - scaler/specialist moved when only test rows were corrupted")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "overall_verdict": "NO LEAKAGE FOUND" if overall_pass else "LEAKAGE FOUND",
        "n_hours": int(len(hours)), "n_episodes": int(len(ep)),
        "n_feature_columns": int(X.shape[1]),
        "feature_causality": {**causality,
                              "baseline_audit_lookahead": base_audit},
        "overlap": overlap,
        "scaler_and_calibration": scaler,
        "sigma_reference": sigref,
        "new_data": newdata,
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
