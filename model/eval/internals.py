"""
eval/internals.py
=====================================================================
Every other file in `eval/` scores NOCTUA against something outside it --
Log-HAR, Kronos, the base rate, a perfect oracle. None of them open the box.
This file does. It asks, of the object that actually ships
(`model/serve/noctua_v2.npz`), five questions about what FUNCTION it
computes, not how well that function scores.

Why that is a different question from everything else in BENCHMARK.md: a
model can win every scoreboard while doing something an option seller should
not trust -- responding to an input with the wrong sign, being a relabelled
copy of the linear bar it was seeded from, emitting quantile curves that
cross (which `MonotoneQuantileHead` is supposed to make impossible), or being
asked about conditions its Stage B head never saw in training. None of those
show up in a QLIKE table. All of them show up here.

THE FIVE QUESTIONS

1. RESPONSE CURVES. Hold every input at its test-split median, sweep one HAR
   feature across its observed 5th-95th percentile range, read off the
   shipped predicted sigma (the full 32-atom, blend_w-mixed pipeline, exactly
   what `serve/predict.py` would return). The HAR literature's prediction is
   unambiguous: more recent realized vol -> a higher forecast, monotonically,
   for all five cascade terms. A negative result -- monotone, correctly
   signed, mildly concave the way a `log`-linear cascade blended with a
   saturating residual should be -- is a full pass and is reported as one.
   Any input with the WRONG sign anywhere in its observed range is not a
   subtle finding; it means the network learned an association that will
   lose money the day it is acted on.

2. IS IT JUST THE LINEAR MODEL WEARING A COSTUME. `noctua/model.py` seeds
   Stage A's linear base at the OLS/Log-HAR solution and zero-inits the
   residual, explicitly so the network "degrades gracefully" to the bar if it
   learns nothing on top. `eval/learning.py` already checked whether the
   RESIDUAL moves during training (it does, by design). This file checks the
   question a seller actually cares about at the shipped checkpoint: does the
   whole Stage A OUTPUT still just track Log-HAR? If Pearson r between
   Stage A's pre-blend median and the stored `har_beta` prediction is above
   ~0.98 on test episodes, the nonlinearity is decorative -- report that as a
   major finding, not a disappointment. Either way, the episodes where they
   disagree most are characterised: are they the volatile nights, the calm
   nights, a particular hour?

3. STRUCTURAL CONSTRAINTS. `MonotoneQuantileHead` is supposed to make
   quantile crossing IMPOSSIBLE by construction (cumulative softplus steps
   walking outward from a free median). `coupling_penalty` is a SOFT loss
   term, not a hard constraint, enforcing two pathwise identities that hold
   on every real price path: M_up >= max(0, R) and M_dn <= min(0, R). Nothing
   stops training from leaving small violations on the table. Both are
   checked empirically, on the object that ships, across every test episode
   and every one of the 32 sigma atoms `infer.predict` integrates over. A
   "zero violations" result is a complete, positive answer to "does the
   architecture do what its own docstring claims" -- not a null result.
   NOTE: the shipped artifact's `has_mx()` is False (see
   `serve/runtime.py:NumpyNoctua.has_mx`, which documents exactly this) --
   there is no `q_mx` head in `noctua_v2.npz` to check. That is reported as
   what it is: a head the architecture supports and this artifact does not
   carry, not a defect in the check.

4. SATURATION AND EXTRAPOLATION. Stage B is conditioned on `log_sigma`, and
   at TRAINING time that scale is `sigma_ref = clip(exp(har_1d)*sqrt(H), lo,
   hi)` with `[lo, hi]` the 0.5th/99.5th percentile of the causal proxy on
   the TRAIN split (`noctua/train_v2.py`). At SERVE time the conditioner is
   Stage A's own 32 forecast atoms, unclipped. `model/AUDIT.md` section 3.4
   measured this gap at ~2.01% of the test split falling below the trained
   floor. That number is reproduced here from scratch, on the CURRENT shipped
   artifact and the CURRENT test split, at both an atom level and an episode
   level, because "verify" means recomputing it, not quoting it. Then, using
   `oracle_sigma.stage_b_at_sigma` to drive Stage B directly at a chosen
   sigma (the reuse this file was told to make), the boundary is crossed on
   purpose: does the predicted excursion keep responding to sigma outside the
   trained range, or does it flatten (a saturated network refusing to
   extrapolate, understating risk on the calm nights that matter) or blow up
   (a network extrapolating in an uncontrolled direction)? This is not
   abstract -- an option seller who prices every single calm night that ever
   comes along is, by definition, always operating exactly at the part of
   the input space the training procedure clipped away.

5. WHAT DOES IT DO ON A SPIKE. The highest-realized-vol test episodes, read
   individually: predicted sigma next to realized RV, and next to what every
   HAR/shape input was actually doing that night. The aggregate numbers
   elsewhere (QLIKE, deep-tail MCB) already say the tail is where the error
   concentrates; this section is the readable, per-episode account of WHY --
   which inputs moved, which stayed asleep, and whether the shortfall looks
   like a model that missed a warning sign already present in its own inputs
   or one that had no warning sign to see.

WHAT THIS FILE DELIBERATELY DOES NOT DO
It does not retrain anything (a CPU-heavy training job runs concurrently in
this repo; this file only loads the SHIPPED `.npz` and runs inference), it
does not re-derive the §7c training-dynamics numbers or the §11a
effective-rank numbers (both already measured, see `eval/learning.py`), and
it does not touch `model/serve/` or `model/noctua/` -- it imports from them
read-only, the same way every other file in this directory does.

    python -m model.eval.internals
    python -m model.eval.internals --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.oracle_sigma import stage_b_at_sigma                             # noqa: E402
from noctua import splits as S                                             # noqa: E402
from noctua.train import load_all                                          # noqa: E402
from serve import runtime as R                                             # noqa: E402

HAR_COLS = ["har_1h", "har_6h", "har_1d", "har_5d", "har_22d"]
EPS = 1e-12


# ==========================================================================
# shared plumbing
# ==========================================================================
def load_everything(artifacts: Path, model_path: Path):
    """The shipped artifact, the data, and the test-production mask.

    Mirrors `oracle_sigma.main`'s setup exactly, so the "test production
    episodes" this file scores are the same 769 rows every other eval in this
    directory scores.
    """
    ep, X = load_all(artifacts)
    sp = S.time_splits(ep)
    fin = np.isfinite(X.to_numpy()).all(1)
    tr_mask = sp["train"] & fin
    te_mask = sp["test"] & fin & S.production_mask(ep)
    m = R.load_model(str(model_path))
    return ep, X, sp, fin, tr_mask, te_mask, m


def stage_a_committee(model, d: dict) -> np.ndarray:
    """Seed-averaged, PRE-BLEND Stage A output (n, K quantile levels).

    Same seed-scoping pattern as `oracle_sigma.stage_b_at_sigma` -- swap
    `model.w` to one seed's slice plus the shared `har_beta`, call the base
    `NumpyNoctua` method, average across seeds, restore. This is Stage A
    BEFORE the `blend_w` mix with Log-HAR (`NumpyNoctua.predict` applies that
    shift after calling `stage_a`), which is the version question 2 needs --
    scoring the post-blend output against Log-HAR would be comparing Log-HAR
    to 75% Log-HAR by construction (`infer.BLEND_W = 0.25`) and would answer
    nothing.
    """
    full = model.w
    outs = []
    try:
        for s in range(model.n_seeds):
            model.w = {**model._seed_scope(full, s), "har_beta": full["har_beta"]}
            outs.append(R.NumpyNoctua.stage_a(model, d["Xa"], d["Xb"]))
    finally:
        model.w = full
    return np.mean(outs, axis=0)


def pct_rank(value: float, ref: np.ndarray) -> float:
    """Percentile rank of `value` within `ref` (0-100), NaN-safe."""
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return float("nan")
    return 100.0 * float((ref < value).mean())


# ==========================================================================
# 1. response curves
# ==========================================================================
def response_curves(ep, X, te_mask, m, n_grid: int = 25, verbose: bool = False) -> dict:
    Xte = X[te_mask]
    H_prod = float(ep.loc[te_mask, "H"].iloc[0])   # production slice is H==19 by construction
    assert (ep.loc[te_mask, "H"].to_numpy() == H_prod).all()

    median_row = Xte[m.feat_cols].median()
    out = {}
    for col in HAR_COLS:
        vals = Xte[col].to_numpy(np.float64)
        p5, p95 = np.nanpercentile(vals, [5, 95])
        grid = np.linspace(p5, p95, n_grid)

        df = pd.DataFrame(np.tile(median_row.to_numpy(), (n_grid, 1)), columns=m.feat_cols)
        df[col] = grid
        Harr = np.full(n_grid, H_prod)
        d = m.prepare(df, Harr)
        pred = m.predict(d)
        sigma = pred["sigma_med"]

        diffs = np.diff(sigma)
        n_up, n_dn = int((diffs > 1e-12).sum()), int((diffs < -1e-12).sum())
        monotone_inc = n_dn == 0
        pear = float(np.corrcoef(grid, sigma)[0, 1])
        spear = float(np.corrcoef(np.argsort(np.argsort(grid)),
                                  np.argsort(np.argsort(sigma)))[0, 1])
        # linear-vs-saturating: compare the slope over the first half of the
        # grid to the slope over the second half. Ratio << 1 => saturating
        # (flattening as the input grows); ~1 => linear.
        mid = n_grid // 2
        slope_lo = (sigma[mid] - sigma[0]) / max(grid[mid] - grid[0], EPS)
        slope_hi = (sigma[-1] - sigma[mid]) / max(grid[-1] - grid[mid], EPS)
        curvature_ratio = float(slope_hi / slope_lo) if slope_lo != 0 else float("nan")

        wrong_sign = int((diffs < -1e-12).sum())
        out[col] = {
            "p5": float(p5), "p95": float(p95),
            "sigma_at_p5": float(sigma[0]), "sigma_at_p95": float(sigma[-1]),
            "monotone_increasing": bool(monotone_inc),
            "n_decreasing_steps": n_dn, "n_increasing_steps": n_up,
            "pearson_r_grid_vs_sigma": pear, "spearman_grid_vs_sigma": spear,
            "slope_first_half": float(slope_lo), "slope_second_half": float(slope_hi),
            "curvature_ratio_hi_over_lo": curvature_ratio,
            "has_wrong_sign_step": wrong_sign > 0,
            "grid": grid.tolist(), "sigma_med": sigma.tolist(),
        }
        print(f"  {col:>8}: [{p5:.5f},{p95:.5f}] -> sigma [{sigma[0]:.5f},{sigma[-1]:.5f}]"
              f"  monotone_inc={monotone_inc}  r={pear:+.4f}  curvature(hi/lo)={curvature_ratio:.3f}"
              + ("  *** WRONG-SIGN STEP ***" if wrong_sign else ""))
        if verbose:
            for g, s in zip(grid, sigma):
                print(f"      {col}={g:+.5f}  sigma_med={s:.6f}")
    return out


# ==========================================================================
# 2. Stage A vs Log-HAR
# ==========================================================================
def stage_a_vs_har(ep, X, te_mask, m, verbose: bool = False) -> dict:
    Xte = X[te_mask]
    H = ep.loc[te_mask, "H"].to_numpy(np.float64)
    d = m.prepare(Xte, H)

    qa_pre = stage_a_committee(m, d)                 # (n, K), PRE-blend
    stage_a_med = qa_pre[:, m.median_idx]
    har = m.har_logvol(d)                              # stored har_beta prediction

    r = float(np.corrcoef(stage_a_med, har)[0, 1])
    resid = stage_a_med - har
    rmse = float(np.sqrt(np.mean(resid**2)))

    RV = ep.loc[te_mask, "RV"].to_numpy(np.float64)
    Hh = ep.loc[te_mask, "H"].to_numpy(np.float64)
    log_rv_rate = np.log(np.maximum(RV, EPS)) - 0.5 * np.log(np.maximum(Hh, 1))
    rv_pct = np.array([pct_rank(v, log_rv_rate) for v in log_rv_rate])

    order = np.argsort(-np.abs(resid))
    top_n = min(15, len(order))
    top_idx = order[:top_n]
    dt = ep.loc[te_mask, "dt"].to_numpy()[top_idx]
    hour = ep.loc[te_mask, "anchor_hour"].to_numpy()[top_idx]
    disagreements = []
    for i, idx in enumerate(top_idx):
        disagreements.append({
            "dt": str(pd.Timestamp(dt[i])),
            "anchor_hour": int(hour[i]),
            "stage_a_med": float(stage_a_med[idx]),
            "har_logvol": float(har[idx]),
            "residual": float(resid[idx]),
            "realized_log_vol_rate_pct_rank": float(rv_pct[idx]),
        })

    print(f"  n = {len(stage_a_med)}   Pearson r(Stage A pre-blend median, Log-HAR) = {r:.5f}"
          f"   RMSE(log-rate) = {rmse:.5f}")
    print(f"  {'threshold':>10}   {'>= r means largely reproducing Log-HAR'}")
    print(f"  top {top_n} disagreements (|Stage A - Log-HAR| largest):")
    print(f"  {'when':>22} {'anchor':>6} {'stageA':>9} {'logHAR':>9} {'resid':>9} {'RV%rank':>8}")
    for row in disagreements:
        print(f"  {row['dt']:>22} {row['anchor_hour']:>6} {row['stage_a_med']:>9.4f} "
              f"{row['har_logvol']:>9.4f} {row['residual']:>+9.4f} "
              f"{row['realized_log_vol_rate_pct_rank']:>7.1f}%")
    hi_share = float((rv_pct[order[:top_n]] >= 50).mean())
    print(f"  {hi_share*100:.0f}% of the top-{top_n} disagreements sit above the "
          f"MEDIAN realized log-vol-rate (i.e. skew toward volatile nights)"
          if not np.isnan(hi_share) else "")

    return {
        "n": int(len(stage_a_med)), "pearson_r": r, "rmse_log_rate": rmse,
        "mean_abs_residual": float(np.mean(np.abs(resid))),
        "top_disagreements": disagreements,
        "top_disagreements_share_above_median_rv": hi_share,
    }


# ==========================================================================
# 3. structural constraints
# ==========================================================================
def structural_constraints(ep, X, te_mask, m, verbose: bool = False) -> dict:
    Xte = X[te_mask]
    H = ep.loc[te_mask, "H"].to_numpy(np.float64)
    d = m.prepare(Xte, H)
    pred = m.predict(d)                              # shipped, committee-averaged, 32 atoms

    def monotone_check(q, name):
        # q: (n, A, K). Non-decreasing along the LAST axis (quantile levels).
        diffs = np.diff(q, axis=-1)
        n_viol = int((diffs < -1e-9).sum())
        n_cells = diffs.size
        worst = float(diffs.min()) if diffs.size else float("nan")
        print(f"  {name:>6}: {n_viol:,} / {n_cells:,} adjacent-level steps negative "
              f"(worst = {worst:+.3e})")
        return {"n_violations": n_viol, "n_cells": n_cells,
                "violation_rate": n_viol / max(n_cells, 1), "worst_step": worst}

    print("monotonicity of emitted quantile curves (per episode x atom x level-step):")
    mono = {
        "q_r": monotone_check(pred["q_r"], "q_r"),
        "q_up": monotone_check(pred["q_up"], "q_up"),
        "q_dn": monotone_check(pred["q_dn"], "q_dn"),
    }
    has_mx = m.has_mx()
    if has_mx and "q_mx" in pred:
        mono["q_mx"] = monotone_check(pred["q_mx"], "q_mx")
    else:
        print("  q_mx  : NOT PRESENT in this artifact -- has_mx() is False for "
              "noctua_v2.npz (see serve/runtime.py). Cannot be checked.")
        mono["q_mx"] = None

    # pathwise identity, exactly as `noctua.model.coupling_penalty` states it:
    #   Q_up(tau)   >= max(0,  Q_r(tau))
    #   Q_dn(tau)   >= max(0, -Q_r(1 - tau))      (Q_dn is the nonneg, sign-flipped m_dn)
    # which correspond in RAW units to M_up >= max(0, R) and M_dn <= min(0, R).
    q_r, q_up, q_dn = pred["q_r"], pred["q_up"], pred["q_dn"]
    lo_up = np.maximum(q_r, 0.0)
    lo_dn = np.maximum(-q_r[:, :, ::-1], 0.0)
    tol = 1e-6
    viol_up = q_up < lo_up - tol
    viol_dn = q_dn < lo_dn - tol
    n_up, n_dn = int(viol_up.sum()), int(viol_dn.sum())
    n_cells = q_r.size
    print(f"\npathwise identity  M_up >= max(0,R):  {n_up:,} / {n_cells:,} cells violate"
          f"  (worst {float((lo_up - q_up).max()):.4e})" if n_up else
          f"\npathwise identity  M_up >= max(0,R):  0 / {n_cells:,} cells violate")
    print(f"pathwise identity  M_dn <= min(0,R):  {n_dn:,} / {n_cells:,} cells violate"
          f"  (worst {float((lo_dn - q_dn).max()):.4e})" if n_dn else
          f"pathwise identity  M_dn <= min(0,R):  0 / {n_cells:,} cells violate")

    return {
        "monotonicity": mono,
        "coupling_up_violations": n_up,
        "coupling_dn_violations": n_dn,
        "coupling_n_cells": n_cells,
        "coupling_up_violation_rate": n_up / max(n_cells, 1),
        "coupling_dn_violation_rate": n_dn / max(n_cells, 1),
        "has_mx_head": bool(has_mx),
    }


# ==========================================================================
# 4. saturation and extrapolation
# ==========================================================================
def saturation(ep, X, tr_mask, te_mask, m, verbose: bool = False) -> dict:
    H_tr = ep.loc[tr_mask, "H"].to_numpy(np.float64)
    raw_sig_tr = np.exp(X.loc[tr_mask, "har_1d"].to_numpy(np.float64)) * np.sqrt(H_tr)
    lo, hi = np.quantile(raw_sig_tr, [0.005, 0.995])
    print(f"  Stage B training sigma range (0.5/99.5 pct of causal proxy on TRAIN, "
          f"n={tr_mask.sum():,}): [{lo:.6f}, {hi:.6f}]")

    Xte = X[te_mask]
    H_te = ep.loc[te_mask, "H"].to_numpy(np.float64)
    d = m.prepare(Xte, H_te)
    pred = m.predict(d)
    atoms = pred["sigma_atoms"]                        # (n, 32)

    atom_below = atoms < lo
    atom_above = atoms > hi
    n_atom = atoms.size
    ep_any_below = atom_below.any(axis=1)
    ep_any_above = atom_above.any(axis=1)
    ep_med_below = pred["sigma_med"] < lo
    ep_med_above = pred["sigma_med"] > hi

    print(f"  n test production episodes = {len(atoms)}, {n_atom} (episode x atom) cells")
    print(f"  atom-level:    below floor {100*atom_below.mean():.3f}%   "
          f"above ceiling {100*atom_above.mean():.3f}%")
    print(f"  episode, ANY of 32 atoms outside:  below {100*ep_any_below.mean():.2f}%   "
          f"above {100*ep_any_above.mean():.2f}%")
    print(f"  episode, MEDIAN atom outside:      below {100*ep_med_below.mean():.2f}%   "
          f"above {100*ep_med_above.mean():.2f}%")
    print("  (model/AUDIT.md 3.4 quoted ~2.01% test-below-floor / 0.00% above-ceiling; "
          "the precise figure depends on whether 'outside' is measured per atom, per "
          "episode-any-atom, or per episode-median -- all three are reported above so "
          "the comparison is auditable rather than asserted.)")

    # ---- drive Stage B directly across a range that spans well outside
    # [lo, hi], reusing oracle_sigma.stage_b_at_sigma exactly as instructed.
    sweep = np.geomspace(lo / 8.0, hi * 8.0, 17)
    n = len(Xte)
    med_up_at = []
    for s in sweep:
        sig = np.full(n, s)
        out = stage_b_at_sigma(m, d, sig)
        med_up = out["q_up"][:, 0, m.median_idx]        # (n,) standardized median m_up
        med_up_at.append(float(np.median(med_up)))
    med_up_at = np.array(med_up_at)
    raw_up_at = med_up_at * sweep                        # de-standardize: M_up_med = m_up * sigma

    print("\n  driving Stage B directly at sigma across "
          f"[{sweep[0]:.5f}, {sweep[-1]:.5f}] (trained range was [{lo:.5f},{hi:.5f}]):")
    print(f"  {'sigma':>10} {'in-range?':>10} {'std m_up_med':>13} {'raw M_up_med':>13}")
    for s, mu, ru in zip(sweep, med_up_at, raw_up_at):
        tag = "yes" if lo <= s <= hi else ("BELOW" if s < lo else "ABOVE")
        print(f"  {s:>10.5f} {tag:>10} {mu:>13.5f} {ru:>13.5f}")

    diffs = np.diff(raw_up_at)
    any_nonfinite = bool(not np.all(np.isfinite(raw_up_at)))
    monotone_response = bool(np.all(diffs > -1e-9))
    print(f"\n  raw M_up_med response monotone in sigma across the FULL swept range "
          f"(including outside-training): {monotone_response}")
    print(f"  any non-finite output while extrapolating: {any_nonfinite}")

    return {
        "train_sigma_lo": float(lo), "train_sigma_hi": float(hi),
        "n_test_episodes": int(len(atoms)), "n_atom_cells": int(n_atom),
        "atom_below_floor_pct": float(100 * atom_below.mean()),
        "atom_above_ceiling_pct": float(100 * atom_above.mean()),
        "episode_any_atom_below_pct": float(100 * ep_any_below.mean()),
        "episode_any_atom_above_pct": float(100 * ep_any_above.mean()),
        "episode_median_below_pct": float(100 * ep_med_below.mean()),
        "episode_median_above_pct": float(100 * ep_med_above.mean()),
        "extrapolation_sweep_sigma": sweep.tolist(),
        "extrapolation_sweep_raw_m_up_median": raw_up_at.tolist(),
        "extrapolation_monotone": monotone_response,
        "extrapolation_any_nonfinite": any_nonfinite,
    }


# ==========================================================================
# 5. spike episodes
# ==========================================================================
SPIKE_INPUT_COLS = HAR_COLS + [
    "vov_5d", "vov_22d", "jump_share_1d", "semi_neg_share_1d",
    "mom_ret_1d", "mom_drawdown_90d", "reg_rv_vs_year",
]


def spike_episodes(ep, X, tr_mask, te_mask, m, n_top: int = 8, verbose: bool = False) -> dict:
    Xte = X[te_mask]
    ep_te = ep[te_mask]
    H_te = ep_te["H"].to_numpy(np.float64)
    d = m.prepare(Xte, H_te)
    pred = m.predict(d)
    sigma_med = pred["sigma_med"]
    RV = ep_te["RV"].to_numpy(np.float64)

    order = np.argsort(-RV)[:n_top]

    ref = X.loc[tr_mask, SPIKE_INPUT_COLS]              # train-split reference distribution
    rows = []
    for rank, idx in enumerate(order):
        i_loc = Xte.index[idx]
        under_ratio = float(sigma_med[idx] / max(RV[idx], EPS))
        row = {
            "rank": rank + 1,
            "dt": str(pd.Timestamp(ep_te["dt"].to_numpy()[idx])),
            "RV": float(RV[idx]),
            "sigma_med": float(sigma_med[idx]),
            "under_forecast_ratio_sigma_over_RV": under_ratio,
            "inputs": {},
        }
        for col in SPIKE_INPUT_COLS:
            v = float(Xte.loc[i_loc, col])
            row["inputs"][col] = {
                "value": v, "train_pct_rank": pct_rank(v, ref[col].to_numpy(np.float64)),
            }
        rows.append(row)

    print(f"  top {n_top} highest-RV test episodes (RV = realized total window vol):")
    for row in rows:
        print(f"\n  #{row['rank']}  {row['dt']}   RV={row['RV']:.5f}  "
              f"sigma_med={row['sigma_med']:.5f}  "
              f"(sigma/RV = {row['under_forecast_ratio_sigma_over_RV']:.3f}, "
              f"{'UNDER' if row['under_forecast_ratio_sigma_over_RV'] < 1 else 'over'}-forecast)")
        for col, info in row["inputs"].items():
            flag = " <<< elevated (>=p80 train)" if info["train_pct_rank"] >= 80 else (
                   " <<< depressed (<=p20 train)" if info["train_pct_rank"] <= 20 else "")
            print(f"      {col:>17} = {info['value']:>+9.5f}  "
                  f"(train pct rank {info['train_pct_rank']:5.1f}%){flag}")

    n_under = int(sum(r["under_forecast_ratio_sigma_over_RV"] < 1 for r in rows))
    print(f"\n  {n_under}/{n_top} of the highest-RV episodes were UNDER-forecast "
          f"(sigma_med < RV).")
    return {"top": rows, "n_under_forecast": n_under, "n_top": n_top}


# ==========================================================================
# main
# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="What has NOCTUA actually learned?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--model", type=Path, default=Path("model/serve/noctua_v2.npz"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/internals.json"))
    ap.add_argument("--verbose", action="store_true",
                    help="print per-grid-point / per-episode detail")
    a = ap.parse_args(argv)

    ep, X, sp, fin, tr_mask, te_mask, m = load_everything(a.artifacts, a.model)
    print(f"model: {a.model.name}   test production episodes: {int(te_mask.sum()):,}   "
          f"train episodes (all anchors): {int(tr_mask.sum()):,}\n")

    print("=" * 78)
    print("1. RESPONSE CURVES (HAR cascade, others held at test-split median)")
    print("=" * 78)
    q1 = response_curves(ep, X, te_mask, m, verbose=a.verbose)

    print("\n" + "=" * 78)
    print("2. IS IT JUST THE LINEAR MODEL? (Stage A pre-blend vs stored Log-HAR)")
    print("=" * 78)
    q2 = stage_a_vs_har(ep, X, te_mask, m, verbose=a.verbose)

    print("\n" + "=" * 78)
    print("3. STAGE B STRUCTURAL CONSTRAINTS (monotonicity + pathwise identity)")
    print("=" * 78)
    q3 = structural_constraints(ep, X, te_mask, m, verbose=a.verbose)

    print("\n" + "=" * 78)
    print("4. SATURATION AND EXTRAPOLATION (trained sigma range vs served atoms)")
    print("=" * 78)
    q4 = saturation(ep, X, tr_mask, te_mask, m, verbose=a.verbose)

    print("\n" + "=" * 78)
    print("5. WHAT DOES IT DO ON A SPIKE (highest-RV test episodes)")
    print("=" * 78)
    q5 = spike_episodes(ep, X, tr_mask, te_mask, m, verbose=a.verbose)

    result = {
        "model": str(a.model), "n_test_production": int(te_mask.sum()),
        "n_train": int(tr_mask.sum()),
        "response_curves": q1,
        "stage_a_vs_har": q2,
        "structural_constraints": q3,
        "saturation": q4,
        "spike_episodes": q5,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
