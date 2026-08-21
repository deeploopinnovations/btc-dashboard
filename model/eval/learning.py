"""
eval/learning.py
=====================================================================
Is NOCTUA learning anything, or is it a linear OLS fit wearing a 6,939-
parameter costume?

WHY THIS FILE EXISTS

Every other file in this directory asks whether the model is RIGHT --
calibrated, discriminating, better than a baseline. None of them ask whether
the network is doing any WORK to get there. Stage A's linear base is seeded
at the OLS/Log-HAR solution (`init_base_from_ols`) and the residual head is
zero-initialised, by explicit design (model.py, StageA docstring): "training
therefore STARTS at the Log-HAR benchmark". That is a defensible safety
property -- the model can't finish worse than the linear bar -- but it also
means a network that never moves would still look identical to Log-HAR at
epoch 0 and would still SHIP, because nothing here checks that the residual
actually residualises. This file is that check, in five parts.

PART 1 -- LEARNING CURVES, PER HEAD
The objective sums five pinball terms (stage-A volatility `a`, terminal
return `r`, upside/downside excursion `up`/`dn`, max excursion `mx`) plus a
coupling penalty. `eval/losshead.py` already found that `r` is the largest
term and the one with no measurable predictive signal (`eval/direction.py`).
If a head's validation loss is flat from epoch 1, that head's gradient signal
is not being converted into a better fit -- for `r` that would be consistent
with (not proof of) losshead's finding; for `a`, `up`, `dn`, or `mx` it would
be a genuine defect, since those are the heads the product is sold on.
DECISION RULE, fixed before running: a head is called FLATLINED if its
calibration-split loss falls by less than 5% from epoch 1 to the final
epoch. This is a description, not a p-value -- it is reported alongside the
raw trajectory so a reader can disagree with the cutoff.

PART 2 -- HOW MUCH IS THE INIT
Two arms, identical in every respect (data, split, seed, architecture,
optimizer, LR schedule) except one thing: whether `Noctua.a.base` starts at
the OLS/Log-HAR coefficients or at PyTorch's default `nn.Linear` init. Because
`torch.manual_seed(seed)` is called once before model construction and the
OLS overwrite happens via `@torch.no_grad()` in-place copies (no RNG state
consumed), the two arms see byte-identical minibatch order for the whole run
-- the ONLY thing that differs is six numbers (5 slope + 1 intercept) at
epoch 0. This is about as clean a counterfactual as a stochastic optimizer
allows.
DECISION RULE, fixed before running: if the random-init arm's calibration
loss at the shipped epoch budget (40) is within 2% of the OLS-init arm's,
the OLS init is a warm start with no lasting effect -- the network is doing
the fitting. If the gap exceeds 10%, the network has not matched from-scratch
what OLS handed it for free in 40 epochs, and the OLS seed -- not the
training -- is carrying the volatility forecast.

PART 3 -- IS THE CAPACITY USED
On the OLS-init arm's best-validation checkpoint (i.e. exactly what
`train_model` would ship for this budget), every hidden layer of width 32 in
both stages is examined on the calibration split:
  * DEAD UNITS: pre-activation <= 0 for every sample. GELU keeps a small
    gradient alive even there, unlike ReLU, so a dead-by-this-definition unit
    is not necessarily *frozen* -- but it is contributing nothing positive to
    any downstream feature, which is the operational meaning of "unused
    capacity" for this check.
  * EFFECTIVE RANK: mean-centered SVD of the post-activation matrix; report
    the number of components needed for 95% of variance, out of 32 possible.
  * WEIGHT MAGNITUDE: |W| summary stats for each layer's first linear map.
DECISION RULE: effective rank below half the nominal width (< 16 of 32) on
either stage's first hidden layer is reported as "this width-32 layer behaves
like a much narrower one" -- a description of redundancy, not evidence of a
bug (a network can legitimately not need all its width).

PART 4 -- GRADIENT HEALTH AND THE reg_post_etf PRECEDENT
`eval/regime.py` already found one input, `reg_post_etf`, that is IDENTICALLY
ZERO on the shipped training split (all training rows predate the 2024-01-11
ETF launch) and therefore standardizes to exactly 0.0 for every training row
-- which means its first-layer weight COLUMN receives an exactly-zero
gradient on every minibatch, every epoch, and ships as an untrained random
number that only activates in production. That is a mechanical fact about
`Standardizer`: a column with zero standard deviation on the training split
maps to `(x - mu) / 1.0 = 0` for every training row by construction, so *any*
column constant-on-train, not just this one, gets a permanently-random
weight column. This file (a) recomputes that mechanism directly by checking
raw column standard deviation on the training split for every input the
model consumes, and (b) confirms it empirically by inspecting the actual
autograd gradient of the relevant weight columns during training, rather
than taking the theoretical argument on faith. Per-epoch gradient L2 norms
are also recorded per parameter group, to look for vanishing (norms decaying
toward the numerical floor) or exploding (norms pinned at the clip threshold
of 5.0, set in `train_model`) behaviour.

PART 5 -- DOES 40 EPOCHS CONVERGE
A separate OLS-init arm is trained for 150 epochs with the OneCycleLR
schedule shaped for 150 epochs from the start (this matters: OneCycleLR's
shape depends on `total_steps`, so a 150-epoch run's value AT epoch 40 is NOT
comparable to a 40-epoch run's value at epoch 40 -- the two schedules have
annealed to different places by then. This is why Part 2's comparison uses
two independently-scheduled 40-epoch runs rather than slicing this one.) What
is asked here is narrower and still answerable from one run: given a schedule
that is allowed to run to 150 epochs, has calibration loss already bottomed
out by epoch 40, or is it still falling? If the minimum lands past epoch 40,
that is a cheap, real improvement available to the shipped recipe.

A FIDELITY CHECK
The per-epoch instrumentation below is a second implementation of the
training loop in `noctua/train.py::train_model`, not a wrapper around it --
duplicated so that per-head losses, per-group gradient norms and per-column
gradient zero-checks can be pulled out without touching the file the rest of
the project depends on. To catch a reimplementation bug before it becomes a
false finding, the same data/split/seed/hyperparameters are also run through
the REAL, unmodified `train_model`, and the two best-validation losses are
compared; a mismatch beyond floating-point noise means the instrumented
numbers below should not be trusted until the harness is fixed.

WHAT THIS FILE DOES NOT DO
It does not touch `model/serve/`, does not change any decision rule anywhere
else in the project, and does not decide anything -- the decision rules above
are stated so the reader can check the arithmetic against them, not so a
number gets silently rounded into a verdict.

SCALE, ON PURPOSE
One fold (the shipped `noctua.splits.time_splits` split: train through
2023-01-01, calibrate through 2024-07-01), one seed, hidden=32. This is a
diagnostic, not a benchmark re-run -- `eval/benchmark.py` already averages 3
seeds across 6 walk-forward folds elsewhere, and duplicating that here would
just burn CPU competing with it.

    python -m model.eval.learning
    python -m model.eval.learning --verbose
    python -m model.eval.learning --epochs 40 --long-epochs 150 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua import baselines as B                                          # noqa: E402
from noctua import splits as S                                             # noqa: E402
from noctua.model import BASE_COLS, LEVELS, Noctua, coupling_penalty, pinball_loss  # noqa: E402
from noctua.train import load_all, prepare, train_model                   # noqa: E402

EPS = 1e-12
FLATLINE_REL_DROP = 0.05          # < 5% drop epoch-1 -> final = "flatlined"
INIT_GAP_WARM_START_PCT = 2.0     # < 2% gap = init is just a warm start
INIT_GAP_DOMINANT_PCT = 10.0      # > 10% gap = init is doing the work
EFF_RANK_REDUNDANT_FRAC = 0.5     # eff_rank < 50% of width = "acts narrower"
GRAD_CLIP = 5.0                    # must match train_model's clip_grad_norm_

DATA_KEYS = ("Xa", "Xb", "Xs", "y", "log_sigma", "r", "m_up", "m_dn", "m_mx")

PARAM_GROUPS = [
    "a.base", "a.body", "a.head",
    "b.body", "b.q_r", "b.q_up", "b.q_dn", "b.q_mx",
]


def _group_params(model: Noctua, group: str):
    stage, sub = group.split(".")
    mod = getattr(getattr(model, stage), sub)
    return list(mod.parameters())


# --------------------------------------------------------------------------
# instrumentation helpers
# --------------------------------------------------------------------------
def head_losses(model: Noctua, D: dict, lv: torch.Tensor, lam_mx: float) -> dict:
    """Full-batch (no minibatch noise), per-head pinball losses, unweighted --
    matching exactly what `train_model`'s own validation metric computes, so
    `total` here is bit-comparable to `train_model`'s `best`/`vl`.
    """
    with torch.no_grad():
        qa, res_med = model.a(D["Xa"], D["Xb"], return_parts=True)
        qr, qu, qd, qm = model.b(D["Xs"], D["log_sigma"][:, None])
        out = {
            "a": float(pinball_loss(qa, D["y"], lv)),
            "r": float(pinball_loss(qr, D["r"], lv)),
            "up": float(pinball_loss(qu, D["m_up"], lv)),
            "dn": float(pinball_loss(qd, D["m_dn"], lv)),
            "mx": float(lam_mx * pinball_loss(qm, D["m_mx"], lv)),
            "coupling": float(coupling_penalty(qr, qu, qd)),
            "anchor_res_med_sq": float((res_med ** 2).mean()),
        }
    out["total"] = out["a"] + out["r"] + out["up"] + out["dn"] + out["mx"]
    return out


def train_instrumented(
    tr, wtr, va, *, hidden=32, epochs=40, bs=4096, lr=2e-3, lam_couple=1.0,
    lam_anchor=0.0, lam_r=1.0, lam_mx=1.0, seed=0, ols_beta=None, verbose=False,
):
    """Re-derives `train_model`'s training loop (same hyperparameters, same
    optimizer/scheduler/clip, same loss) with two additions: (1) no early
    stopping -- every arm runs its FULL requested epoch budget so the curve
    is not truncated, and (2) per-epoch logging of per-head losses (train
    AND calibration), per-parameter-group gradient norms, and a per-input-
    column check of whether the gradient into that column was EXACTLY zero
    for every minibatch of every epoch (the `reg_post_etf` mechanism).

    Best-validation checkpointing IS kept (same '>1e-5 improvement' rule as
    `train_model`), so `best_state`/`best_epoch`/`best_val` reproduce what
    `train_model(epochs=N)` would ship even though this function does not
    stop early -- unless `train_model`'s patience-8 rule would have fired
    first, which is checked separately (see `would_have_stopped_early`).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Noctua(tr["Xa"].shape[1], tr["Xb"].shape[1], tr["Xs"].shape[1], hidden)
    if ols_beta is not None:
        model.a.init_base_from_ols(ols_beta)

    dev = torch.device("cpu")
    lv = torch.tensor(LEVELS, dtype=torch.float32, device=dev)
    T = {k: torch.tensor(v, device=dev) for k, v in tr.items() if k in DATA_KEYS}
    W = torch.tensor(wtr.astype(np.float32), device=dev)
    V = {k: torch.tensor(v, device=dev) for k, v in va.items() if k in DATA_KEYS}

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(T["y"])
    steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)

    # first-layer weight matrices to watch for zero-gradient input columns.
    watch = {
        "a.base.weight": (model.a.base.weight, list(BASE_COLS)),
        "a.body0.weight": (model.a.body[0].weight, list(tr["cols"]["all"])),
        "b.body0.weight": (model.b.body[0].weight, list(tr["cols"]["shape"]) + ["log_sigma"]),
    }
    ever_nonzero = {k: np.zeros(w.shape[1], dtype=bool) for k, (w, _) in watch.items()}

    history = []
    best_val, best_epoch, best_state, bad_streak, first_bad_at = np.inf, -1, None, 0, None
    t0 = time.time()
    for ep_i in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        grad_sums = {g: 0.0 for g in PARAM_GROUPS}
        for i in range(steps):
            idx = perm[i * bs: (i + 1) * bs]
            qa, res_med = model.a(T["Xa"][idx], T["Xb"][idx], return_parts=True)
            qr, qu, qd, qm = model.b(T["Xs"][idx], T["log_sigma"][idx][:, None])
            loss = (
                pinball_loss(qa, T["y"][idx], lv, W[idx])
                + lam_anchor * (res_med ** 2).mean()
                + lam_r * pinball_loss(qr, T["r"][idx], lv, W[idx])
                + pinball_loss(qu, T["m_up"][idx], lv, W[idx])
                + pinball_loss(qd, T["m_dn"][idx], lv, W[idx])
                + lam_mx * pinball_loss(qm, T["m_mx"][idx], lv, W[idx])
                + lam_couple * coupling_penalty(qr, qu, qd)
            )
            opt.zero_grad()
            loss.backward()

            for g in PARAM_GROUPS:
                sq = sum(float((p.grad ** 2).sum()) for p in _group_params(model, g)
                         if p.grad is not None)
                grad_sums[g] += sq ** 0.5

            for key, (w, _) in watch.items():
                gcol = w.grad
                if gcol is not None:
                    nz = (gcol.abs().sum(dim=0) > 0).cpu().numpy()
                    ever_nonzero[key] |= nz

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            sched.step()
            tot += float(loss.detach())

        model.eval()
        val_head = head_losses(model, V, lv, lam_mx)
        train_head = head_losses(model, T, lv, lam_mx)
        grad_norm = {g: grad_sums[g] / steps for g in PARAM_GROUPS}
        rec = {"epoch": ep_i, "train_loss_minibatch_avg": tot / steps,
               "train_head": train_head, "val_head": val_head, "grad_norm": grad_norm}
        history.append(rec)

        improved = val_head["total"] < best_val - 1e-5
        if improved:
            best_val, best_epoch = val_head["total"], ep_i
            best_state = copy.deepcopy(model.state_dict())
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= 8 and first_bad_at is None:
                first_bad_at = ep_i    # where train_model's early stop WOULD have fired

        if verbose and (ep_i % 5 == 0 or ep_i == epochs - 1):
            vh = val_head
            print(f"  epoch {ep_i:3d}  train {tot/steps:.5f}  val {vh['total']:.5f}  "
                  f"[a {vh['a']:.4f} r {vh['r']:.4f} up {vh['up']:.4f} dn {vh['dn']:.4f} "
                  f"mx {vh['mx']:.4f} coupling {vh['coupling']:.5f}]  "
                  f"|grad| a.body {grad_norm['a.body']:.3f} b.body {grad_norm['b.body']:.3f}")

    dt = time.time() - t0
    zero_cols = {}
    for key, (_, colnames) in watch.items():
        zero_cols[key] = [c for c, nz in zip(colnames, ever_nonzero[key]) if not nz]

    return dict(
        model=model, history=history, best_val=float(best_val), best_epoch=int(best_epoch),
        best_state=best_state, train_seconds=dt, zero_grad_input_cols=zero_cols,
        would_have_early_stopped_at=first_bad_at, n_params=model.n_params(),
    )


def weight_stats(w: torch.Tensor) -> dict:
    a = w.detach().abs().cpu().numpy().ravel()
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "n_weights": int(a.size)}


def layer_capacity(pre: torch.Tensor, post: torch.Tensor) -> dict:
    """Dead-unit fraction + effective rank (95% variance) of one hidden layer,
    measured on whatever batch `pre`/`post` were computed from.
    """
    pre_np = pre.detach().cpu().numpy()
    post_np = post.detach().cpu().numpy()
    dead = pre_np.max(axis=0) <= 0.0
    A = post_np - post_np.mean(axis=0, keepdims=True)
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    var = s ** 2
    cum = np.cumsum(var) / max(float(var.sum()), 1e-30)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    return {
        "n_units": int(pre_np.shape[1]),
        "dead_frac": float(dead.mean()),
        "dead_unit_idx": [int(i) for i in np.where(dead)[0]],
        "eff_rank_95": k95,
        "singular_values": [float(x) for x in s],
        "cum_var_explained_95_at": k95,
    }


def capacity_report(model: Noctua, V: dict) -> dict:
    model.eval()
    with torch.no_grad():
        a_pre1 = model.a.body[0](V["Xa"]);  a_post1 = model.a.body[1](a_pre1)
        a_pre2 = model.a.body[2](a_post1);  a_post2 = model.a.body[3](a_pre2)
        xb_in = torch.cat([V["Xs"], V["log_sigma"][:, None]], dim=1)
        b_pre1 = model.b.body[0](xb_in);    b_post1 = model.b.body[1](b_pre1)
        b_pre2 = model.b.body[2](b_post1);  b_post2 = model.b.body[3](b_pre2)

    out = {
        "a_layer1": layer_capacity(a_pre1, a_post1),
        "a_layer2": layer_capacity(a_pre2, a_post2),
        "b_layer1": layer_capacity(b_pre1, b_post1),
        "b_layer2": layer_capacity(b_pre2, b_post2),
    }
    out["a_layer1"]["weight_abs"] = weight_stats(model.a.body[0].weight)
    out["a_layer2"]["weight_abs"] = weight_stats(model.a.body[2].weight)
    out["b_layer1"]["weight_abs"] = weight_stats(model.b.body[0].weight)
    out["b_layer2"]["weight_abs"] = weight_stats(model.b.body[2].weight)
    out["a_base_weight_abs"] = weight_stats(model.a.base.weight)
    return out


def near_constant_inputs(ep, X, mask, cols) -> list:
    """Raw (pre-standardization) std of every model input on the TRAINING
    split. `Standardizer` maps a column with std==0 on train to a column of
    EXACT zeros for every training row (mu equals the constant, sd forced to
    1.0) -- that is the reg_post_etf mechanism, and this reproduces the check
    for every input the model consumes, not just the one already found.
    """
    rows = []
    for c in cols:
        v = X.loc[mask, c].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        sd = float(np.std(v)) if len(v) else float("nan")
        mean = float(np.mean(v)) if len(v) else float("nan")
        rows.append({"col": c, "train_std": sd, "train_mean": mean,
                     "constant_on_train": bool(sd == 0.0)})
    rows.sort(key=lambda r: r["train_std"])
    return rows


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Is NOCTUA learning, or coasting on its OLS init?")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/learning.json"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40, help="shipped-recipe epoch budget")
    ap.add_argument("--long-epochs", type=int, default=150, help="Part 5 convergence run")
    ap.add_argument("--threads", type=int, default=2,
                    help="torch CPU threads (kept low: another job shares this box)")
    ap.add_argument("--skip-long", action="store_true", help="skip the 150-epoch run (Part 5)")
    ap.add_argument("--skip-fidelity", action="store_true",
                    help="skip the real train_model() cross-check")
    ap.add_argument("--verbose", action="store_true", help="print per-epoch detail")
    a = ap.parse_args(argv)

    torch.set_num_threads(max(1, a.threads))

    print(f"[learning] loading artifacts from {a.artifacts}")
    t0 = time.time()
    ep, X = load_all(a.artifacts)
    fin = np.isfinite(X.to_numpy()).all(1)
    sp = S.time_splits(ep)
    m_tr, m_va = sp["train"] & fin, sp["calib"] & fin
    print(f"[learning] shipped split: train={int(m_tr.sum()):,}  calib={int(m_va.sum()):,}  "
          f"(loaded in {time.time()-t0:.1f}s)")

    tr, stds = prepare(ep, X, m_tr)
    va, _ = prepare(ep, X, m_va, *stds)
    wtr = S.sample_weights(ep, m_tr)
    print(f"[learning] n_feat={tr['Xa'].shape[1]}  n_base={tr['Xb'].shape[1]}  "
          f"n_shape={tr['Xs'].shape[1]}  hidden={a.hidden}  seed={a.seed}")

    Xb_std = pd.DataFrame(tr["Xb"], columns=BASE_COLS)
    ols = B.OLS(BASE_COLS).fit(Xb_std, tr["y"].astype(np.float64), wtr)
    print(f"[learning] OLS/Log-HAR base beta: {np.round(ols.beta, 4).tolist()}")

    out = {"meta": {
        "artifacts": str(a.artifacts), "hidden": a.hidden, "seed": a.seed,
        "epochs": a.epochs, "long_epochs": a.long_epochs,
        "n_train": int(m_tr.sum()), "n_calib": int(m_va.sum()),
        "n_feat": tr["Xa"].shape[1], "n_base": tr["Xb"].shape[1], "n_shape": tr["Xs"].shape[1],
        "split": "noctua.splits.time_splits (shipped): "
                 f"train_end={S.TRAIN_END}  calib_end={S.CALIB_END}",
        "ols_beta": ols.beta.tolist(),
        "commands": [
            "python -m model.eval.learning",
            "python -m model.eval.learning --verbose",
        ],
    }}

    # ---- Part 1 + Part 2 (OLS arm) ---------------------------------------
    print(f"\n[learning] PART 1+2 -- OLS-init arm, {a.epochs} epochs "
          f"(OneCycleLR scheduled for {a.epochs} epochs)")
    resA = train_instrumented(tr, wtr, va, hidden=a.hidden, epochs=a.epochs, seed=a.seed,
                              ols_beta=ols.beta, verbose=a.verbose)
    print(f"  done in {resA['train_seconds']:.1f}s  best_val={resA['best_val']:.6f} "
          f"@epoch {resA['best_epoch']}  final_val={resA['history'][-1]['val_head']['total']:.6f}")
    if resA["would_have_early_stopped_at"] is not None:
        print(f"  NOTE: train_model's patience-8 early stop would have fired at epoch "
              f"{resA['would_have_early_stopped_at']} (this run kept going anyway)")

    # ---- Part 2 (random arm) ---------------------------------------------
    print(f"\n[learning] PART 2 -- random-init arm, {a.epochs} epochs, same seed/data order")
    resB = train_instrumented(tr, wtr, va, hidden=a.hidden, epochs=a.epochs, seed=a.seed,
                              ols_beta=None, verbose=a.verbose)
    print(f"  done in {resB['train_seconds']:.1f}s  best_val={resB['best_val']:.6f} "
          f"@epoch {resB['best_epoch']}  final_val={resB['history'][-1]['val_head']['total']:.6f}")

    ols_final = resA["history"][-1]["val_head"]["total"]
    rnd_final = resB["history"][-1]["val_head"]["total"]
    gap_pct = 100.0 * (rnd_final - ols_final) / ols_final
    if gap_pct < INIT_GAP_WARM_START_PCT:
        verdict2 = "WARM START -- random-init matched OLS-init within 2%: the network is fitting."
    elif gap_pct > INIT_GAP_DOMINANT_PCT:
        verdict2 = ("INIT-DOMINATED -- random-init has NOT caught up in the training budget: "
                    "the OLS seed, not the training, is carrying the fit.")
    else:
        verdict2 = "IN BETWEEN the two decision thresholds -- reported as-is, no verdict claimed."
    print(f"  gap at final epoch: OLS {ols_final:.6f} vs random {rnd_final:.6f} "
          f"({gap_pct:+.2f}%)  -> {verdict2}")

    # ---- Part 1 flatline / overfit determination (OLS arm) ---------------
    heads = ["a", "r", "up", "dn", "mx"]
    flat = {}
    for h in heads:
        v1 = resA["history"][0]["val_head"][h]
        vN = resA["history"][-1]["val_head"][h]
        rel = (v1 - vN) / v1 if v1 != 0 else float("nan")
        flat[h] = {"val_epoch1": v1, "val_final": vN, "rel_drop": rel,
                  "flatlined": bool(rel < FLATLINE_REL_DROP)}
    overfit = {}
    for h in heads:
        t_final = resA["history"][-1]["train_head"][h]
        v_final = resA["history"][-1]["val_head"][h]
        overfit[h] = {"train_final": t_final, "val_final": v_final,
                     "val_over_train": v_final / t_final if t_final else float("nan")}
    print("\n[learning] PART 1 -- per-head verdicts (OLS arm, calibration split):")
    for h in heads:
        f = flat[h]
        print(f"  {h:>3}  epoch1={f['val_epoch1']:.5f} -> final={f['val_final']:.5f}  "
              f"drop={100*f['rel_drop']:.1f}%  {'FLATLINED' if f['flatlined'] else 'learning'}  "
              f"| val/train={overfit[h]['val_over_train']:.3f}")

    # ---- Part 4: near-constant inputs + zero-grad columns -----------------
    print("\n[learning] PART 4 -- constant/near-constant inputs on the training split")
    nci = near_constant_inputs(ep, X, m_tr, tr["cols"]["all"])
    zero_std_cols = [r["col"] for r in nci if r["constant_on_train"]]
    print(f"  exactly-constant-on-train columns: {zero_std_cols if zero_std_cols else 'NONE'}")
    print("  10 lowest-std columns on the training split:")
    for r in nci[:10]:
        print(f"    {r['col']:<22} std={r['train_std']:.6g}  mean={r['train_mean']:.6g}")
    print(f"  zero-gradient input columns confirmed by autograd (OLS arm, all epochs):")
    for k, cols in resA["zero_grad_input_cols"].items():
        print(f"    {k:<16} {cols if cols else '(none)'}")

    # ---- Part 3: capacity, on the OLS arm's best checkpoint ---------------
    print("\n[learning] PART 3 -- capacity used (OLS-init best-val checkpoint, calib split)")
    dev = torch.device("cpu")
    V = {k: torch.tensor(v, device=dev) for k, v in va.items() if k in DATA_KEYS}
    cap_model = Noctua(tr["Xa"].shape[1], tr["Xb"].shape[1], tr["Xs"].shape[1], a.hidden)
    cap_model.load_state_dict(resA["best_state"])
    cap = capacity_report(cap_model, V)
    for layer in ("a_layer1", "a_layer2", "b_layer1", "b_layer2"):
        c = cap[layer]
        redundant = c["eff_rank_95"] < EFF_RANK_REDUNDANT_FRAC * c["n_units"]
        print(f"  {layer:<10} dead={c['dead_frac']*100:5.1f}% ({len(c['dead_unit_idx'])}/{c['n_units']})  "
              f"eff_rank_95={c['eff_rank_95']:2d}/{c['n_units']}  "
              f"{'REDUNDANT' if redundant else 'in range'}  "
              f"|W| mean={c['weight_abs']['mean']:.4f} p90={c['weight_abs']['p90']:.4f}")

    # ---- Part 5: 150-epoch convergence -------------------------------------
    resC = None
    if not a.skip_long:
        print(f"\n[learning] PART 5 -- {a.long_epochs}-epoch run, OLS-init, "
              f"OneCycleLR scheduled for {a.long_epochs} epochs")
        resC = train_instrumented(tr, wtr, va, hidden=a.hidden, epochs=a.long_epochs,
                                  seed=a.seed, ols_beta=ols.beta, verbose=a.verbose)
        val40 = resC["history"][a.epochs - 1]["val_head"]["total"]
        valEnd = resC["history"][-1]["val_head"]["total"]
        print(f"  done in {resC['train_seconds']:.1f}s  best_val={resC['best_val']:.6f} "
              f"@epoch {resC['best_epoch']}")
        print(f"  val @ epoch {a.epochs}: {val40:.6f}   val @ epoch {a.long_epochs}: {valEnd:.6f}")
        if resC["best_epoch"] >= a.epochs:
            print(f"  -> STILL FALLING past epoch {a.epochs} within this 150-epoch schedule: "
                  f"best came at epoch {resC['best_epoch']}. Training longer (under a schedule "
                  f"shaped for it) finds a better checkpoint than the shipped 40-epoch budget.")
        else:
            print(f"  -> best checkpoint came at epoch {resC['best_epoch']}, BEFORE the shipped "
                  f"40-epoch budget: 150 epochs (this schedule) does not find anything the "
                  f"40-epoch schedule wouldn't, on this split/seed.")
    else:
        print("\n[learning] PART 5 skipped (--skip-long)")

    # ---- Fidelity check: real, unmodified train_model() --------------------
    fidelity = None
    if not a.skip_fidelity:
        print(f"\n[learning] FIDELITY CHECK -- real noctua.train.train_model(), "
              f"{a.epochs} epochs, same seed/split/hidden/ols_beta")
        t0 = time.time()
        _, real_best = train_model(tr, wtr, va, hidden=a.hidden, epochs=a.epochs, seed=a.seed,
                                   verbose=False, ols_beta=ols.beta)
        dt = time.time() - t0
        diff = resA["best_val"] - real_best
        rel = abs(diff) / real_best if real_best else float("nan")
        fidelity = {"real_train_model_best_val": float(real_best),
                    "instrumented_arm_best_val": resA["best_val"],
                    "abs_diff": float(diff), "rel_diff": float(rel), "seconds": dt}
        tag = "MATCH" if rel < 0.01 else "MISMATCH -- instrumented numbers below need scrutiny"
        print(f"  real train_model best_val={real_best:.6f}  instrumented best_val="
              f"{resA['best_val']:.6f}  rel_diff={100*rel:.3f}%  -> {tag}")
    else:
        print("\n[learning] fidelity check skipped (--skip-fidelity)")

    # ---- assemble output ---------------------------------------------------
    def strip_history(res):
        return [{k: v for k, v in rec.items()} for rec in res["history"]]

    out["part1_learning_curves"] = {
        "epochs": a.epochs, "history": strip_history(resA),
        "flatline_rule_rel_drop_lt": FLATLINE_REL_DROP,
        "per_head_verdict": flat, "per_head_overfit_ratio": overfit,
    }
    out["part2_ols_vs_random_init"] = {
        "epochs": a.epochs,
        "ols_arm": {"history": strip_history(resA), "best_val": resA["best_val"],
                    "best_epoch": resA["best_epoch"], "final_val_total": ols_final},
        "random_arm": {"history": strip_history(resB), "best_val": resB["best_val"],
                       "best_epoch": resB["best_epoch"], "final_val_total": rnd_final},
        "gap_pct_final_epoch": gap_pct,
        "decision_thresholds": {"warm_start_below_pct": INIT_GAP_WARM_START_PCT,
                                "init_dominated_above_pct": INIT_GAP_DOMINANT_PCT},
        "verdict": verdict2,
    }
    out["part3_capacity"] = {
        "source": "OLS-init arm's best-validation checkpoint "
                 f"(epoch {resA['best_epoch']} of {a.epochs})",
        "layers": cap,
        "redundant_rule": f"eff_rank_95 < {EFF_RANK_REDUNDANT_FRAC} * width",
    }
    out["part4_gradient_health"] = {
        "near_constant_inputs_train_split": nci,
        "constant_on_train": zero_std_cols,
        "zero_grad_input_cols_confirmed": resA["zero_grad_input_cols"],
        "grad_clip_threshold": GRAD_CLIP,
    }
    if resC is not None:
        out["part5_convergence_150ep"] = {
            "long_epochs": a.long_epochs, "history": strip_history(resC),
            "best_val": resC["best_val"], "best_epoch": resC["best_epoch"],
            "val_at_epoch_40": resC["history"][a.epochs - 1]["val_head"]["total"],
            "val_at_final_epoch": resC["history"][-1]["val_head"]["total"],
            "still_falling_past_epoch_40": bool(resC["best_epoch"] >= a.epochs),
        }
    if fidelity is not None:
        out["fidelity_check"] = fidelity

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\n[learning] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
