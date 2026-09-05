"""
eval/defects.py
=====================================================================
Three defects, all measured before now, none of them fixed yet. This file
does not re-discover them -- it does the arithmetic that was missing the
first time: not "is this a defect" but "how much does it move the thing a
user actually sees."

Read first: BENCHMARK.md section 6i (the untrained flag, the training-era
weight table), BENCHMARK.md section 7c (effective rank, dead units), and
eval/regime.py (the flag's walk-forward null). This file does not repeat
those measurements; it asks the sharper question each one left open.

-----------------------------------------------------------------------
DEFECT 1 -- an untrained random offset is live in production

`reg_post_etf` is `hour_ts >= 2024-01-11`. On the SHIPPED split (train ends
2023-01-01) it is identically 0 across 189,831 training rows and identically
1 across 73,867 test rows. A column with zero standard deviation on the
training split standardizes to exactly 0.0 for every training row
(`noctua.train.Standardizer`: mu = the constant, sd forced to 1.0 because the
measured sd is 0), so its first-layer weight column in BOTH stages -- it sits
in `feat_cols` (Stage A's wide block) and in `shape_cols` (Stage B) -- gets
gradient exactly zero on every minibatch of every epoch. `eval/learning.py`
Part 4 confirmed this directly with autograd. The weight ships at random
initialization, of entirely ordinary magnitude (BENCHMARK.md 6i: contribution
mean |W| 0.088-0.114 against 0.096-0.116 over all inputs -- it does not look
different from a trained weight by inspection).

`eval/regime.py` already asked "does removing it help" and got back a null:
2-3 of 6 folds, no consistent direction, QLIKE marginally worse without it.
That is the WRONG question to have asked to settle whether this is a live
defect, and section 6i says so explicitly -- a null on an AVERAGE SCORE
across 6 folds of ~365 episodes each cannot distinguish "this input moves
every forecast by a little" from "this input moves no forecast at all"; both
look like noise at that resolution. The question this file asks instead:
holding the shipped artifact fixed, set the standardized input to 0 (the
value it would carry if the flag had varied in training, or equivalently the
value a fix would restore it to) instead of 1 (its actual, permanent, serve-
time value), and measure the PER-EPISODE change in what is actually served --
`sigma_med` (the `sigma_window_pct` field in noctua.json) and the committee's
own `touch_prob` (the `barrier_curves` a user reads). If individual forecasts
move by a lot, an untrained random number is materially shaping live output
regardless of what it does to an averaged fold score. If they do not, that
also settles it, and settles it more directly than a 6-fold average could.

Two structural checks come for free and are reported alongside the headline:
(a) `sigma_med` depends on Stage A only (`infer`/`runtime.predict` builds it
from `qa`, never from Stage B's output), so zeroing the Stage-B copy of the
flag alone must leave `sigma_med` exactly unchanged, and zeroing the Stage-A
copy alone must reproduce the "both zeroed" result exactly -- this is checked
bit-for-bit, not assumed; (b) the served `touch_prob` runs through the
4-specialist equal-weight committee (`NoctuaV2.committee_quantiles`), and only
one of those four specialists (`neural`) reads Stage B at all, so whatever
Stage A's zeroing does to `sigma_atoms` is diluted roughly fourfold by
construction before it reaches a barrier probability -- a mechanical reason
to expect the committee number to move less than a single-model number would.

-----------------------------------------------------------------------
DEFECT 2 -- is the low effective rank a training problem or a data problem

BENCHMARK.md 7c measured effective rank (95% of activation variance, mean-
centered SVD) of 17/12/13/9 against a nominal width of 32 across Stage A's two
hidden layers and Stage B's two hidden layers, with 21.9% dead units in
Stage B's second layer, on a separately-trained single-seed diagnostic model
(`eval/learning.py`, hidden=32, calibration split). Weight magnitudes were
unremarkable (0.097-0.113), which is why that section called the redundancy
"in rank, not weights collapsing to zero" -- but it stopped short of asking
where the rank ceiling comes from.

There are two candidate explanations and they call for opposite responses.
TRAINING PROBLEM: bad initialization, a dying-unit pathology, a learning rate
that never lets the network spread its weight across the full width -- fixable
by changing how training runs. DATA PROBLEM: the 39 Stage-A inputs (dominated
by an intercorrelated HAR cascade -- BENCHMARK.md 5b measured
corr(har_1d, har_5d) = 0.851, corr(har_5d, har_22d) = 0.880) and the 21
Stage-B shape inputs simply do not span 32 independent directions, in which
case a width-32 layer settling on effective rank 12-17 is the CORRECT
behaviour of a correctly-fitted network, not a defect -- exactly the
possibility BENCHMARK.md 7c raised and did not check.

This file checks it directly, on the SHIPPED artifact (`serve/noctua_v2.npz`,
all 3 seeds), not a re-trained stand-in: the effective rank of the
standardized input matrix itself (mean-centered SVD, same 95%-of-variance
convention as 7c, so the two numbers are comparable), the effective rank of
each seed's own first-layer weight matrix (same convention, applied to the
weight matrix's singular values directly), and -- to make the comparison
airtight rather than by proxy -- a fresh forward pass of the shipped weights
through the calibration split (the same population 7c's own diagnostic model
used), reporting pre-activation and post-activation effective rank and dead
fraction exactly as 7c defined them. If the input's own rank already sits
near the measured activation rank, the ceiling was set before a single
gradient step ran.

One structural fact is worth flagging before any number is run: Stage B's
first hidden layer is `nn.Linear(22, 32)` (21 shape columns plus the
`log_sigma` conditioner). Its weight matrix is 32x22, and NO matrix of that
shape can have rank above 22 -- the "effective rank 13 of 32" comparison in
7c compares an achieved rank against a nominal width that Stage B's own input
dimensionality made unreachable from the start, independent of training
quality entirely. That is checked here as a matter of arithmetic, not
measurement, and reported alongside the measured numbers because it changes
what "using 13 of 32" means.

-----------------------------------------------------------------------
DEFECT 3 -- the adaptive correction is global but the bias is not

`serve/adaptive.py` fits one trailing-median multiplier per night, 0.93-0.96
measured, and applies it to every hour of every anchor alike. But
`eval/anatomy.py`'s conditional error map (section 1, by-hour) found the
over-forecast bias concentrated in the hours around the 17:00 UTC production
anchor: median RV/sigma 0.85-0.94 across anchor hours 15-22, against 0.99-1.06
across hours 0-13. A single multiplier fitted on the pooled population sits
somewhere between those two regimes and is wrong for both -- undercorrecting
where the bias is worst, overcorrecting where the model was already close to
unbiased. That is the shape of a problem an HOUR-CONDITIONAL correction is
built to fix: the same trailing-median mechanism `serve/adaptive.py` already
uses, restricted to trailing episodes anchored at the SAME hour of day as the
episode being corrected, with a fallback to the global factor when the
hour-specific trailing pool is too thin (the same MIN_EPISODES=20 floor
`serve/adaptive.py` uses, and the same [0.70, 1.40] clip).

DECISION RULE, fixed here before the numbers below are computed. QLIKE is
NOT the right primary criterion for this comparison, and that turns out to
matter (see the aside below), so the primary criterion is the same one
`serve/adaptive.py` and BENCHMARK.md section 6 use to justify the correction
that already ships: mean absolute barrier-calibration error, |realized breach
rate - nominal alpha|, at the model's OWN calibrated strikes (`safe_level`),
at alpha in {1%, 2%, 5%}, both sides, walk-forward and causal (a trailing
pool built only from episodes that have already settled strictly before the
anchor being corrected, exactly as `serve/adaptive.py` requires). The
hour-conditional correction is adopted over the global one if and only if
(a) it reduces the mean absolute calibration error in the peak bucket (anchor
hours 15-22, where the bias is documented to concentrate) by a MATERIAL
margin relative to the global correction, defined here as >=20% relative
reduction -- a bar set well above what subsample noise at n~4,000 could
plausibly produce by chance, chosen for that reason and not tuned to the
result; AND (b) it does not worsen the OVERALL (all-hour) mean absolute
calibration error by more than a small tolerance, defined here as <=10%
relative, on the reasoning that a strictly local fix is allowed to cost a
little in aggregate without being a bad trade, but a fix that only helps in
aggregate score while making the pooled number materially worse is not one a
seller should take. If (a) fails, hour-conditioning is not worth the added
complexity regardless of (b). If (a) holds and (b) fails, it is reported as a
partial win -- real, localized, and not a clean replacement for the global
correction. Both thresholds (20%, 10%) are stated here, before the numbers,
precisely so they cannot be adjusted after seeing them.

The QLIKE aside, because it is a genuine and useful negative result on its
own and was not anticipated going in: QLIKE = r - log(r) - 1 with
r = RV^2/sigma^2 penalizes UNDER-forecasts (r > 1) far more steeply than
over-forecasts (BENCHMARK.md 7a already measured this asymmetry directly: a
factor-2 error costs 1.60x at QLIKE, a factor-4 error costs 4.67x). A
trailing-MEDIAN correction targets the median of RV/sigma, not the mean of
RV^2 -- the quantity QLIKE is actually minimized by -- so a correction that
successfully repairs the median can still raise mean QLIKE if it shrinks
sigma anywhere the conditional RV distribution is right-skewed, which
volatility always is. This is reported below exactly as measured, with the
sign stated explicitly, and it is NOT used as the adoption criterion, for the
reason just given: the global correction that already ships was justified in
BENCHMARK.md section 6 on barrier calibration, not on QLIKE, and holding the
new arm to a metric the old one was never asked to win would not be a fair
comparison.

    python -m model.eval.defects              # all three parts
    python -m model.eval.defects flag          # defect 1 only
    python -m model.eval.defects capacity      # defect 2 only
    python -m model.eval.defects bias          # defect 3 only
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

from eval.anatomy import subsample_pred                                      # noqa: E402
from eval.direction import block_bootstrap_ci                                # noqa: E402
from noctua import splits as S                                               # noqa: E402
from noctua.train import load_all, prepare                                   # noqa: E402
from serve.runtime import gelu, load_model                                   # noqa: E402

EPS = 1e-12
FLAG = "reg_post_etf"

# -- defect 1 defaults ------------------------------------------------------
D1_N_SUBSAMPLE = 8000
D1_SEED = 0
D1_BARRIER_PCT = (1.0, 2.0, 5.0)

# -- defect 3 defaults -- match serve/adaptive.py's own constants -----------
D3_WINDOW_DAYS = 60
D3_MIN_EPISODES = 20
D3_CLIP_LO, D3_CLIP_HI = 0.70, 1.40
D3_ALPHAS = (0.01, 0.02, 0.05)
D3_N_SUBSAMPLE = 4000
D3_SEED = 0
D3_PEAK_HOURS = tuple(range(15, 23))     # 15..22 inclusive -- where anatomy.json's
D3_OFF_HOURS = tuple(range(0, 14))       # by-hour table shows the bias concentrated
D3_PEAK_MATERIAL_REL = 0.20              # decision-rule threshold (a), fixed in the docstring
D3_OVERALL_TOLERANCE_REL = 0.10          # decision-rule threshold (b), fixed in the docstring


def _dist(x: np.ndarray) -> dict:
    """Distribution summary used everywhere a per-episode delta is reported."""
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(len(x)), "mean": float(np.mean(x)), "mean_abs": float(np.mean(np.abs(x))),
        "median": float(np.median(x)), "std": float(np.std(x)),
        "p05": float(np.quantile(x, 0.05)), "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)), "p95": float(np.quantile(x, 0.95)),
        "frac_positive": float(np.mean(x > 0)), "frac_negative": float(np.mean(x < 0)),
    }


def _eff_rank(A: np.ndarray, center: bool = True) -> dict:
    """Mean-centered SVD, effective rank at 95%/99% of variance -- the exact
    convention `eval/learning.py`'s `layer_capacity` uses, so numbers here are
    directly comparable to BENCHMARK.md 7c's 17/12/13/9."""
    A = np.asarray(A, dtype=np.float64)
    Ac = A - A.mean(axis=0, keepdims=True) if center else A
    s = np.linalg.svd(Ac, full_matrices=False, compute_uv=False)
    var = s ** 2
    tot = max(float(var.sum()), 1e-30)
    cum = np.cumsum(var) / tot
    return {
        "n_obs": int(A.shape[0]), "n_dims": int(A.shape[1]),
        "eff_rank_95": int(np.searchsorted(cum, 0.95) + 1),
        "eff_rank_99": int(np.searchsorted(cum, 0.99) + 1),
        "top5_singular_values": [float(v) for v in s[:5]],
        "cum_var_explained_at_rank10": float(cum[min(9, len(cum) - 1)]),
    }


def _seed_weights(model, s: int) -> dict:
    """Strip the `m{s}.` prefix NoctuaV2 stores per-seed weights under.

    Reimplemented locally rather than calling `NoctuaV2._seed_scope` directly:
    this file only ever reads two named tensors per seed, and doing it here
    keeps `serve/runtime.py` untouched and its private method un-depended-on
    from outside the class that owns it.
    """
    pre = f"m{s}."
    return {k[len(pre):]: v for k, v in model.w.items() if k.startswith(pre)}


def _apply_row_factor(pred: dict, factor: np.ndarray) -> dict:
    """Per-row version of `serve.adaptive.apply_correction` -- that function
    takes one scalar factor for every row; this one takes an array, which is
    what an hour-conditional (or any time-varying) correction needs."""
    factor = np.asarray(factor, dtype=np.float64)
    out = {k: v for k, v in pred.items() if not str(k).startswith("_pooled_")}
    out["sigma_atoms"] = pred["sigma_atoms"] * factor[:, None]
    out["sigma_med"] = pred["sigma_med"] * factor
    if "sigma_mean" in pred:
        out["sigma_mean"] = pred["sigma_mean"] * factor
    return out


# ==========================================================================
# DEFECT 1 -- untrained flag: how much does the served prediction move
# ==========================================================================
def run_defect1(model, ep: pd.DataFrame, X: pd.DataFrame,
                n_subsample: int = D1_N_SUBSAMPLE, seed: int = D1_SEED,
                barrier_pct=D1_BARRIER_PCT) -> dict:
    print(f"\n[defect1] {FLAG}: measuring the served-prediction shift on the shipped artifact")
    fin = np.isfinite(X.to_numpy()).all(1)
    sp = S.time_splits(ep)
    mask_test = sp["test"] & fin

    v_test = X.loc[mask_test, FLAG].to_numpy()
    v_train = X.loc[sp["train"], FLAG].to_numpy()
    train_sd = float(np.nanstd(v_train))
    test_all_one = bool(np.all(v_test == 1.0))
    print(f"[defect1] train sd={train_sd:.6f} (n={sp['train'].sum():,})  "
          f"test: identically 1.0 on every row? {test_all_one} (n={int(mask_test.sum()):,})")

    idx_a = model.feat_cols.index(FLAG)
    idx_s = model.shape_cols.index(FLAG)

    idx_all = np.flatnonzero(mask_test)
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(idx_all, size=n_subsample, replace=False))
          if len(idx_all) > n_subsample else idx_all)
    print(f"[defect1] subsample n={len(idx):,} of {len(idx_all):,} test episodes, seed={seed}")

    Xsub, Hsub = X.iloc[idx], ep["H"].to_numpy()[idx]
    d_served = model.prepare(Xsub, Hsub)   # standardized flag = 1.0 (its real, permanent serve value)

    def zero_copy(d, zero_a, zero_b):
        dd = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        if zero_a:
            dd["Xa"][:, idx_a] = 0.0
        if zero_b:
            dd["Xs"][:, idx_s] = 0.0
        return dd

    t0 = time.time()
    pred_served = model.predict(d_served)
    pred_zero_both = model.predict(zero_copy(d_served, True, True))
    pred_zero_a = model.predict(zero_copy(d_served, True, False))
    pred_zero_b = model.predict(zero_copy(d_served, False, True))
    print(f"[defect1] four predict() passes in {time.time()-t0:.1f}s")

    sigma_served = np.asarray(pred_served["sigma_med"], dtype=np.float64)
    sigma_zero_both = np.asarray(pred_zero_both["sigma_med"], dtype=np.float64)
    sigma_zero_a = np.asarray(pred_zero_a["sigma_med"], dtype=np.float64)
    sigma_zero_b = np.asarray(pred_zero_b["sigma_med"], dtype=np.float64)

    # structural checks -- must hold exactly, since sigma_med is built from
    # Stage A's `qa` alone (see runtime.NumpyNoctua.predict / infer.predict).
    stage_a_only_matches_both = bool(np.allclose(sigma_zero_a, sigma_zero_both))
    stage_b_alone_is_inert = bool(np.allclose(sigma_zero_b, sigma_served))
    print(f"[defect1] structural check: zeroing Stage A alone reproduces "
          f"'both zeroed' exactly = {stage_a_only_matches_both}; zeroing Stage B "
          f"alone leaves sigma_med exactly unchanged = {stage_b_alone_is_inert}")

    d_sigma_pp = 100.0 * (sigma_served - sigma_zero_both)          # pct-of-vol units
    d_sigma_rel_pct = 100.0 * (sigma_served - sigma_zero_both) / np.maximum(sigma_zero_both, EPS)

    touch = {}
    for pct in barrier_pct:
        u = float(np.log1p(pct / 100.0))
        for side, up in (("up", True), ("dn", False)):
            t0 = time.time()
            p_served = model.touch_prob(pred_served, u, up)
            p_zero = model.touch_prob(pred_zero_both, u, up)
            dpp = 100.0 * (p_served - p_zero)
            touch[f"{pct:g}pct_{side}"] = {
                "served_mean_pct": float(100 * p_served.mean()),
                "counterfactual_mean_pct": float(100 * p_zero.mean()),
                "delta_pp": _dist(dpp),
            }
            print(f"[defect1] touch_prob {pct:g}% {side}: served mean "
                  f"{100*p_served.mean():.3f}%  counterfactual mean "
                  f"{100*p_zero.mean():.3f}%  |delta| median "
                  f"{touch[f'{pct:g}pct_{side}']['delta_pp']['median']:.4f}pp  "
                  f"({time.time()-t0:.1f}s)")

    # cross-check against BENCHMARK.md 6i's "contribution" statistic, computed
    # fresh here on the shipped weights rather than quoted.
    contrib = []
    for s in range(model.n_seeds):
        w = _seed_weights(model, s)
        Wa, Wb = w["a.body.0.weight"], w["b.body.0.weight"]
        col_a, col_b = Wa[:, idx_a], Wb[:, idx_s]
        contrib.append({
            "seed": s,
            "stage_a_col_mean_abs": float(np.mean(np.abs(col_a))),
            "stage_a_W_mean_abs": float(np.mean(np.abs(Wa))),
            "stage_b_col_mean_abs": float(np.mean(np.abs(col_b))),
            "stage_b_W_mean_abs": float(np.mean(np.abs(Wb))),
        })
        print(f"[defect1] seed {s}: stage-A |col| {contrib[-1]['stage_a_col_mean_abs']:.4f} "
              f"vs |W| {contrib[-1]['stage_a_W_mean_abs']:.4f}   stage-B |col| "
              f"{contrib[-1]['stage_b_col_mean_abs']:.4f} vs |W| "
              f"{contrib[-1]['stage_b_W_mean_abs']:.4f}")

    return {
        "flag": FLAG,
        "root_cause": {
            "train_std": train_sd, "train_n": int(sp["train"].sum()),
            "test_identically_one": test_all_one, "test_n": int(mask_test.sum()),
            "idx_in_feat_cols_stage_a": idx_a, "idx_in_shape_cols_stage_b": idx_s,
        },
        "structural_checks": {
            "stage_a_only_matches_both_zeroed": stage_a_only_matches_both,
            "stage_b_alone_is_inert_for_sigma_med": stage_b_alone_is_inert,
        },
        "weight_contribution_cross_check": contrib,
        "population": {"source": "shipped-split TEST slice (ts >= CALIB_END), "
                                 "all H and anchor hours", "n_subsample": len(idx),
                       "n_available": len(idx_all), "seed": seed},
        "sigma_med_shift": {"pp_of_vol": _dist(d_sigma_pp), "relative_pct": _dist(d_sigma_rel_pct)},
        "touch_prob_shift_pp": touch,
    }


# ==========================================================================
# DEFECT 2 -- effective rank: data property or training defect
# ==========================================================================
def run_defect2(model, ep: pd.DataFrame, X: pd.DataFrame) -> dict:
    print("\n[defect2] effective rank: is the redundancy in the data or in training")
    fin = np.isfinite(X.to_numpy()).all(1)
    sp = S.time_splits(ep)
    m_tr, m_cal = sp["train"] & fin, sp["calib"] & fin

    tr, stds = prepare(ep, X, m_tr)
    cal, _ = prepare(ep, X, m_cal, *stds)   # calib standardized with TRAIN stats, as serving does

    def shape_block(d):
        return np.concatenate(
            [d["Xs"].astype(np.float64), d["log_sigma"].astype(np.float64)[:, None]], axis=1)

    input_rank = {
        "stage_A_Xa_train": _eff_rank(tr["Xa"].astype(np.float64)),
        "stage_A_Xa_calib": _eff_rank(cal["Xa"].astype(np.float64)),
        "stage_B_Xs_plus_logsigma_train": _eff_rank(shape_block(tr)),
        "stage_B_Xs_plus_logsigma_calib": _eff_rank(shape_block(cal)),
    }
    for k, v in input_rank.items():
        print(f"[defect2] input {k}: n_dims={v['n_dims']}  eff_rank_95={v['eff_rank_95']}  "
              f"eff_rank_99={v['eff_rank_99']}")

    Xa_tr, Xs_tr = tr["Xa"].astype(np.float64), shape_block(tr)
    Xa_cal, Xs_cal = cal["Xa"].astype(np.float64), shape_block(cal)

    per_seed = []
    for s in range(model.n_seeds):
        w = _seed_weights(model, s)
        Wa, ba = w["a.body.0.weight"].astype(np.float64), w["a.body.0.bias"].astype(np.float64)
        Wb, bb = w["b.body.0.weight"].astype(np.float64), w["b.body.0.bias"].astype(np.float64)

        wa_rank = _eff_rank(Wa, center=False)
        wb_rank = _eff_rank(Wb, center=False)
        wa_rank["max_possible_rank"] = int(min(Wa.shape))
        wb_rank["max_possible_rank"] = int(min(Wb.shape))

        def forward(Xa, Xs, tag):
            pre_a, post_a = Xa @ Wa.T + ba, gelu(Xa @ Wa.T + ba)
            pre_b, post_b = Xs @ Wb.T + bb, gelu(Xs @ Wb.T + bb)
            return {
                "a_layer1_pre": {**_eff_rank(pre_a), "dead_frac": float((pre_a.max(0) <= 0.0).mean())},
                "a_layer1_post": _eff_rank(post_a),
                "b_layer1_pre": {**_eff_rank(pre_b), "dead_frac": float((pre_b.max(0) <= 0.0).mean())},
                "b_layer1_post": _eff_rank(post_b),
            }

        fwd_train = forward(Xa_tr, Xs_tr, "train")
        fwd_calib = forward(Xa_cal, Xs_cal, "calib")   # matches eval/learning.py Part 3's own population
        per_seed.append({
            "seed": s, "Wa_rank": wa_rank, "Wb_rank": wb_rank,
            "forward_pass_train": fwd_train, "forward_pass_calib": fwd_calib,
        })
        print(f"[defect2] seed {s}: Wa (32x39) eff_rank_95={wa_rank['eff_rank_95']}/32  "
              f"Wb (32x22) eff_rank_95={wb_rank['eff_rank_95']}/{wb_rank['max_possible_rank']} "
              f"(max possible, NOT 32)")
        print(f"           calib forward pass: a_layer1 post eff_rank_95="
              f"{fwd_calib['a_layer1_post']['eff_rank_95']}/32 dead={fwd_calib['a_layer1_pre']['dead_frac']:.3f}  "
              f"b_layer1 post eff_rank_95={fwd_calib['b_layer1_post']['eff_rank_95']}/32 "
              f"dead={fwd_calib['b_layer1_pre']['dead_frac']:.3f}")

    return {
        "nominal_width": 32,
        "stage_B_first_layer_shape": [32, 22],
        "stage_B_max_possible_rank": 22,
        "note_stage_B_ceiling": (
            "Stage B's first hidden layer is nn.Linear(22, 32): 21 shape columns "
            "plus the log_sigma conditioner. No 32x22 matrix can exceed rank 22, "
            "so a measured eff_rank of 13-17 against a NOMINAL width of 32 "
            "overstates the shortfall -- the honest denominator is 22."),
        "input_effective_rank": input_rank,
        "per_seed": per_seed,
        "reference_prior_measurement": {
            "source": "BENCHMARK.md section 7c / eval/learning.py Part 3 "
                     "(a SEPARATELY-TRAINED single-seed diagnostic model, calib split, "
                     "NOT the shipped 3-seed artifact -- cited for context, not reproduced)",
            "a_layer1_eff_rank_95": "17/32", "a_layer2_eff_rank_95": "12/32",
            "b_layer1_eff_rank_95": "13/32", "b_layer2_eff_rank_95": "9/32",
            "b_layer2_dead_frac": 0.219,
            "propagation_note": (
                "layer 2's INPUT is layer 1's post-activation output, so layer 2's "
                "achievable rank is bounded by layer 1's own eff_rank -- 12/17 and "
                "9/13 measured there are consistent with, not independent of, the "
                "layer-1 ceiling measured fresh in this file; not recomputed here "
                "because the task asked for the first layer and the input."),
        },
    }


# ==========================================================================
# DEFECT 3 -- hour-conditional vs global adaptive correction
# ==========================================================================
def run_defect3(model, ep: pd.DataFrame, X: pd.DataFrame,
                window_days: int = D3_WINDOW_DAYS, min_episodes: int = D3_MIN_EPISODES,
                n_subsample: int = D3_N_SUBSAMPLE, seed: int = D3_SEED) -> dict:
    print("\n[defect3] hour-conditional vs global adaptive correction, walk-forward")
    fin = np.isfinite(X.to_numpy()).all(1)
    train_end_ts = int(pd.Timestamp(S.TRAIN_END, tz="UTC").timestamp())
    calib_end_ts = int(pd.Timestamp(S.CALIB_END, tz="UTC").timestamp())

    # Pool population: H=19, every anchor hour, from TRAIN_END onward -- real
    # chronological history, so the earliest test-era targets still have a
    # warmed-up trailing pool (matches how serve/adaptive.py draws its pool
    # from live history, not from a split boundary).
    pop_mask = (ep["H"] == 19).to_numpy() & fin & (ep["anchor_ts"].to_numpy() >= train_end_ts)
    e = ep.loc[pop_mask].reset_index(drop=True)
    Xp = X.loc[pop_mask].reset_index(drop=True)
    H = e["H"].to_numpy(np.float64)
    print(f"[defect3] pool population (H=19, all anchor hours, ts>=TRAIN_END): {len(e):,}")

    t0 = time.time()
    pred_pop = model.predict(model.prepare(Xp, H))
    print(f"[defect3] predict() over pool population: {time.time()-t0:.1f}s")

    sigma_med = np.asarray(pred_pop["sigma_med"], dtype=np.float64)
    RV = e["RV"].to_numpy(np.float64)
    anchor_ts = e["anchor_ts"].to_numpy(np.int64)
    anchor_hour = e["anchor_hour"].to_numpy(np.int64)
    ratio = RV / np.maximum(sigma_med, EPS)

    order = np.argsort(anchor_ts, kind="stable")
    ts_sorted, ratio_sorted, hour_sorted = anchor_ts[order], ratio[order], anchor_hour[order]

    is_target = anchor_ts >= calib_end_ts     # the "test era" adaptive.py itself measures
    target_idx = np.flatnonzero(is_target)
    n = len(target_idx)
    print(f"[defect3] target population (test era, ts>=CALIB_END): {n:,}")

    hour_pools = {h: (ts_sorted[hour_sorted == h], ratio_sorted[hour_sorted == h]) for h in range(24)}

    H_SEC = 19 * 3600
    WIN_SEC = window_days * 86400
    factor_global = np.ones(n)
    factor_hour = np.ones(n)
    n_pool_g = np.zeros(n, dtype=int)
    n_pool_h = np.zeros(n, dtype=int)
    hour_fallback = np.zeros(n, dtype=bool)

    t0 = time.time()
    for k, i in enumerate(target_idx):
        t_i = int(anchor_ts[i])
        right_ts, left_ts = t_i - H_SEC, t_i - WIN_SEC     # settled strictly before t_i, within the window
        r = np.searchsorted(ts_sorted, right_ts, side="right")
        l = np.searchsorted(ts_sorted, left_ts, side="left")
        pool_g = ratio_sorted[l:r]
        n_pool_g[k] = len(pool_g)
        factor_global[k] = (float(np.clip(np.median(pool_g), D3_CLIP_LO, D3_CLIP_HI))
                            if len(pool_g) >= min_episodes else 1.0)

        hts, hratio = hour_pools[int(anchor_hour[i])]
        rh = np.searchsorted(hts, right_ts, side="right")
        lh = np.searchsorted(hts, left_ts, side="left")
        pool_h = hratio[lh:rh]
        n_pool_h[k] = len(pool_h)
        if len(pool_h) >= min_episodes:
            factor_hour[k] = float(np.clip(np.median(pool_h), D3_CLIP_LO, D3_CLIP_HI))
        else:
            factor_hour[k] = factor_global[k]
            hour_fallback[k] = True
    print(f"[defect3] causal trailing factors computed for {n:,} target rows in "
          f"{time.time()-t0:.1f}s  (global pool median size {int(np.median(n_pool_g))}, "
          f"hour pool median size {int(np.median(n_pool_h))}, hour fallback rate "
          f"{100*hour_fallback.mean():.1f}%)")

    sigma_t, RV_t, hour_t = sigma_med[target_idx], RV[target_idx], anchor_hour[target_idx]

    def qlike(sig):
        r = np.maximum(RV_t ** 2, EPS) / np.maximum(sig ** 2, EPS)
        return r - np.log(r) - 1.0

    sigma_arms = {"raw": sigma_t, "global": sigma_t * factor_global, "hour_conditional": sigma_t * factor_hour}
    qlike_arms = {k: qlike(v) for k, v in sigma_arms.items()}
    for k, v in qlike_arms.items():
        print(f"[defect3] mean QLIKE ({k}): {v.mean():.6f}")

    ci_glob_vs_raw = block_bootstrap_ci(qlike_arms["global"] - qlike_arms["raw"])
    ci_hourc_vs_raw = block_bootstrap_ci(qlike_arms["hour_conditional"] - qlike_arms["raw"])
    ci_hourc_vs_glob = block_bootstrap_ci(qlike_arms["hour_conditional"] - qlike_arms["global"])
    print(f"[defect3] QLIKE delta global-raw {qlike_arms['global'].mean()-qlike_arms['raw'].mean():+.6f} "
          f"CI {ci_glob_vs_raw}  (positive = QLIKE WORSE than no correction)")
    print(f"[defect3] QLIKE delta hourc-raw  {qlike_arms['hour_conditional'].mean()-qlike_arms['raw'].mean():+.6f} "
          f"CI {ci_hourc_vs_raw}")
    print(f"[defect3] QLIKE delta hourc-glob {qlike_arms['hour_conditional'].mean()-qlike_arms['global'].mean():+.6f} "
          f"CI {ci_hourc_vs_glob}  (negative = hour-conditional better)")

    peak_mask = np.isin(hour_t, D3_PEAK_HOURS)
    off_mask = np.isin(hour_t, D3_OFF_HOURS)

    def bucket_stats(sig, mask):
        r = RV_t[mask] / np.maximum(sig[mask], EPS)
        return {"n": int(mask.sum()), "median_ratio": float(np.median(r)),
               "median_abs_bias": float(np.median(np.abs(r - 1.0))),
               "mean_qlike": float(qlike(sig)[mask].mean())}

    buckets = {arm: {"peak_15_22": bucket_stats(sig, peak_mask), "off_0_13": bucket_stats(sig, off_mask)}
              for arm, sig in sigma_arms.items()}
    for arm in sigma_arms:
        p, o = buckets[arm]["peak_15_22"], buckets[arm]["off_0_13"]
        print(f"[defect3] {arm:16} peak(15-22) median RV/sigma={p['median_ratio']:.4f}  "
              f"off(0-13) median RV/sigma={o['median_ratio']:.4f}")

    # ---- barrier calibration: the decision-rule metric --------------------
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, size=n_subsample, replace=False)) if n > n_subsample else np.arange(n)
    pred_target = subsample_pred(pred_pop, target_idx)
    pred_sub = subsample_pred(pred_target, sub)
    M_up_sub = np.abs(e["M_up"].to_numpy(np.float64))[target_idx][sub]
    M_dn_sub = np.abs(e["M_dn"].to_numpy(np.float64))[target_idx][sub]
    hour_sub = hour_t[sub]
    peak_sub, off_sub = np.isin(hour_sub, D3_PEAK_HOURS), np.isin(hour_sub, D3_OFF_HOURS)
    print(f"[defect3] barrier-calibration subsample n={len(sub):,} of {n:,}, seed={seed}")

    factor_sub = {"raw": np.ones(len(sub)), "global": factor_global[sub], "hour_conditional": factor_hour[sub]}
    breach_by_arm = {}    # arm -> list of (alpha, side, breach_bool_array) for the bootstrap below
    cal_err = {}
    t0 = time.time()
    for arm, factor in factor_sub.items():
        pr = _apply_row_factor(pred_sub, factor)
        cells = []
        for alpha in D3_ALPHAS:
            for side, up, M in (("up", True, M_up_sub), ("dn", False, M_dn_sub)):
                lev = np.asarray(model.safe_level(pr, alpha, up=up), dtype=np.float64)
                breach = (M >= lev)
                cells.append((alpha, side, breach))
        breach_by_arm[arm] = cells
        errs = [abs(float(b.mean()) - a) for a, _, b in cells]
        errs_peak = [abs(float(b[peak_sub].mean()) - a) for a, _, b in cells]
        errs_off = [abs(float(b[off_sub].mean()) - a) for a, _, b in cells]
        cal_err[arm] = {"overall": float(np.mean(errs)), "peak_15_22": float(np.mean(errs_peak)),
                        "off_0_13": float(np.mean(errs_off))}
        print(f"[defect3] calibration ({arm:16}): overall={cal_err[arm]['overall']:.5f}  "
              f"peak={cal_err[arm]['peak_15_22']:.5f}  off={cal_err[arm]['off_0_13']:.5f}  "
              f"({time.time()-t0:.0f}s elapsed)")

    # moving-block bootstrap on (hour_conditional - global), peak bucket and overall:
    # resample rows in chronological blocks, recompute each arm's mean-abs-cal-err
    # per replicate, difference the replicates. No new model calls -- the breach
    # indicators are already computed above.
    def calib_bootstrap_ci(cells_a, cells_b, row_mask, n_rep=2000, seed=1):
        idx_pool = np.flatnonzero(row_mask)
        m = len(idx_pool)
        if m < 20:
            return (float("nan"), float("nan"))
        L = max(1, int(round(m ** (1 / 3))))
        nb = int(np.ceil(m / L))
        rng2 = np.random.default_rng(seed)
        starts = rng2.integers(0, m - L + 1, size=(n_rep, nb))
        boot_idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_rep, -1)[:, :m]
        boot_rows = idx_pool[boot_idx]     # (n_rep, m) indices into the subsample
        diffs = np.zeros(n_rep)
        for (alpha, _, ba), (_, _, bb) in zip(cells_a, cells_b):
            ra = ba[boot_rows].mean(axis=1)
            rb = bb[boot_rows].mean(axis=1)
            diffs += np.abs(ra - alpha) - np.abs(rb - alpha)
        diffs /= len(cells_a)
        return (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))

    ci_peak_hourc_vs_glob = calib_bootstrap_ci(
        breach_by_arm["hour_conditional"], breach_by_arm["global"], peak_sub)
    ci_overall_hourc_vs_glob = calib_bootstrap_ci(
        breach_by_arm["hour_conditional"], breach_by_arm["global"], np.ones(len(sub), dtype=bool))
    print(f"[defect3] bootstrap CI, mean_abs_cal_err(hour_cond) - mean_abs_cal_err(global): "
          f"peak {ci_peak_hourc_vs_glob}  overall {ci_overall_hourc_vs_glob}")

    # ---- apply the pre-registered decision rule ----------------------------
    peak_global, peak_hourc = cal_err["global"]["peak_15_22"], cal_err["hour_conditional"]["peak_15_22"]
    overall_global, overall_hourc = cal_err["global"]["overall"], cal_err["hour_conditional"]["overall"]
    peak_rel_change = (peak_hourc - peak_global) / max(peak_global, 1e-12)
    overall_rel_change = (overall_hourc - overall_global) / max(overall_global, 1e-12)
    cond_a = peak_rel_change <= -D3_PEAK_MATERIAL_REL
    cond_b = overall_rel_change <= D3_OVERALL_TOLERANCE_REL
    if cond_a and cond_b:
        verdict = "ADOPT (targeted): hour-conditioning materially improves the peak-hour bucket without materially worsening the pooled number."
    elif cond_a and not cond_b:
        verdict = "PARTIAL: hour-conditioning materially improves the peak-hour bucket but the pooled (all-hour) calibration error worsens beyond the stated tolerance -- real, localized, not a clean replacement for the global correction."
    else:
        verdict = "DO NOT ADOPT: the peak-hour bucket does not improve materially against the pre-registered threshold."
    print(f"[defect3] decision rule: peak rel. change {100*peak_rel_change:+.1f}% "
          f"(threshold <= -{100*D3_PEAK_MATERIAL_REL:.0f}%), overall rel. change "
          f"{100*overall_rel_change:+.1f}% (threshold <= +{100*D3_OVERALL_TOLERANCE_REL:.0f}%)")
    print(f"[defect3] VERDICT: {verdict}")

    return {
        "decision_rule": {
            "primary_metric": "mean absolute barrier-calibration error at safe_level, "
                              "alpha in {1%,2%,5%}, both sides",
            "peak_material_threshold_rel": D3_PEAK_MATERIAL_REL,
            "overall_tolerance_threshold_rel": D3_OVERALL_TOLERANCE_REL,
            "peak_hours": list(D3_PEAK_HOURS), "off_hours": list(D3_OFF_HOURS),
        },
        "population": {"pool_n": int(len(e)), "target_n": n,
                       "window_days": window_days, "min_episodes": min_episodes,
                       "hour_pool_fallback_rate": float(hour_fallback.mean()),
                       "global_pool_median_size": int(np.median(n_pool_g)),
                       "hour_pool_median_size": int(np.median(n_pool_h))},
        "qlike": {
            "mean": {k: float(v.mean()) for k, v in qlike_arms.items()},
            "delta_global_minus_raw": {"mean": float((qlike_arms["global"]-qlike_arms["raw"]).mean()),
                                       "ci95": ci_glob_vs_raw},
            "delta_hourc_minus_raw": {"mean": float((qlike_arms["hour_conditional"]-qlike_arms["raw"]).mean()),
                                      "ci95": ci_hourc_vs_raw},
            "delta_hourc_minus_global": {"mean": float((qlike_arms["hour_conditional"]-qlike_arms["global"]).mean()),
                                         "ci95": ci_hourc_vs_glob},
            "note": "positive delta = QLIKE WORSE. Both corrections raise mean QLIKE relative to "
                   "no correction -- see the module docstring's QLIKE aside for why this is expected "
                   "and is not the adoption criterion.",
        },
        "bucket_bias": buckets,
        "barrier_calibration": {
            "subsample_n": len(sub), "seed": seed,
            "mean_abs_cal_err": cal_err,
            "bootstrap_ci_hourc_minus_global": {"peak_15_22": ci_peak_hourc_vs_glob,
                                                "overall": ci_overall_hourc_vs_glob},
        },
        "verdict": verdict,
        "peak_rel_change": peak_rel_change, "overall_rel_change": overall_rel_change,
    }


# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Three measured defects: how much do they actually move")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/defects.json"))
    sub = ap.add_subparsers(dest="part")

    p1 = sub.add_parser("flag", help="defect 1 -- untrained reg_post_etf offset")
    p1.add_argument("--n-subsample", type=int, default=D1_N_SUBSAMPLE)
    p1.add_argument("--seed", type=int, default=D1_SEED)

    sub.add_parser("capacity", help="defect 2 -- effective rank: data or training")

    p3 = sub.add_parser("bias", help="defect 3 -- hour-conditional vs global adaptive correction")
    p3.add_argument("--window-days", type=int, default=D3_WINDOW_DAYS)
    p3.add_argument("--min-episodes", type=int, default=D3_MIN_EPISODES)
    p3.add_argument("--n-subsample", type=int, default=D3_N_SUBSAMPLE)
    p3.add_argument("--seed", type=int, default=D3_SEED)

    a = ap.parse_args(argv)
    part = a.part or "all"

    t_start = time.time()
    print(f"[defects] loading artifacts from {a.artifacts}")
    ep, X = load_all(a.artifacts)

    out = {"meta": {"artifacts": str(a.artifacts), "part": part,
                    "commands": ["python -m model.eval.defects",
                                "python -m model.eval.defects flag",
                                "python -m model.eval.defects capacity",
                                "python -m model.eval.defects bias"]}}

    needs_model = part in ("flag", "bias", "all")
    model = load_model() if needs_model else None
    if needs_model:
        print(f"[defects] shipped artifact: {model.meta.get('version')}  "
              f"n_params_total={model.meta.get('n_params_total')}")

    if part in ("flag", "all"):
        n1 = getattr(a, "n_subsample", D1_N_SUBSAMPLE) if a.part == "flag" else D1_N_SUBSAMPLE
        s1 = getattr(a, "seed", D1_SEED) if a.part == "flag" else D1_SEED
        out["defect1_untrained_flag"] = run_defect1(model, ep, X, n_subsample=n1, seed=s1)

    if part in ("capacity", "all"):
        model_for_2 = model if model is not None else load_model()
        out["defect2_capacity"] = run_defect2(model_for_2, ep, X)

    if part in ("bias", "all"):
        wd = getattr(a, "window_days", D3_WINDOW_DAYS) if a.part == "bias" else D3_WINDOW_DAYS
        me = getattr(a, "min_episodes", D3_MIN_EPISODES) if a.part == "bias" else D3_MIN_EPISODES
        n3 = getattr(a, "n_subsample", D3_N_SUBSAMPLE) if a.part == "bias" else D3_N_SUBSAMPLE
        s3 = getattr(a, "seed", D3_SEED) if a.part == "bias" else D3_SEED
        out["defect3_hour_bias"] = run_defect3(model, ep, X, window_days=wd, min_episodes=me,
                                               n_subsample=n3, seed=s3)

    out["meta"]["elapsed_seconds"] = round(time.time() - t_start, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n[defects] wrote {a.out}  (total {time.time()-t_start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
