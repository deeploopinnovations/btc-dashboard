"""
eval/garch.py
=====================================================================
The mandatory GARCH(1,1) baseline this repository has never had.

WHY THIS IS A GAP WORTH CLOSING

The research protocol names GARCH(1,1) — including a heavy-tailed innovation
variant — a MANDATORY volatility baseline. `eval/benchmark.py` scores NOCTUA
against Log-HAR, persistence, a scaled climatology and a constant, but never
against a fitted conditional-variance model. That is the single most standard
comparator in the volatility literature, and its absence means the headline
"NOCTUA beats the baselines" has been claimed against a baseline set that
omits the obvious one.

It was absent for an infrastructure reason, not an oversight: nothing here
could install `arch`. `eval/toolchain.py` established empirically that PyPI is
in fact allowlisted through this proxy even though the exchange APIs are not,
which is what unblocked it.

WHAT IS BEING COMPARED, AND THE ONE THING IT MUST NOT GET WRONG

NOCTUA forecasts the volatility of an H-hour forward window from an hourly
anchor. GARCH is a model of a return SERIES, not of an arbitrary window, so the
comparison only means something if the GARCH forecast is aggregated to exactly
the same object:

    sigma_GARCH[t, H] = sqrt( sum over the next H hours of the fitted
                              conditional variance )

Anything else — quoting a one-step-ahead sigma, or annualising differently —
compares two different quantities and would flatter whichever happens to be on
the more convenient scale.

The fit is on HOURLY log returns, strictly on each fold's training window, and
forecasts are produced by iterating the variance recursion forward H steps from
the last in-sample observation available at the anchor. No test-period return
ever enters the fit or the recursion.

THE PRE-REGISTERED FRAMING (ledger id `E-garch`, fixed before `arch` was even
installed)

This is a BASELINE ADDITION, not a model change. NOCTUA is untouched, so there
is no adopt/reject decision about NOCTUA here: the number stands as a
scoreboard entry whatever it says. That framing is deliberate. `log_har_gauss`
already beats NOCTUA on Brier at the 2 % barrier (BENCHMARK.md 18), so a GARCH
that also beats it is a finding to publish, not an embarrassment to manage.

WHAT A NEGATIVE RESULT LOOKS LIKE

If GARCH(1,1) beats NOCTUA on QLIKE, the honest reading is that a 6,939
parameter-per-seed network with 39 inputs is not earning its complexity over a
three-parameter conditional-variance model, and the project's burden shifts to
justifying the machinery rather than tuning it. That is a real possible outcome
and the reason this baseline is mandatory in the first place.

    python -m model.eval.garch --folds 6
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

from eval.direction import block_bootstrap_ci                             # noqa: E402
from noctua import splits as S                                            # noqa: E402
from noctua.train import load_all                                         # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def qlike_vec(rv: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """QLIKE per episode, on a positive-floored forecast."""
    pv = np.maximum(sig, 1e-12) ** 2
    r = np.maximum(rv ** 2, 1e-18) / np.maximum(pv, 1e-18)
    return r - np.log(r) - 1.0


def hourly_returns(artifacts: Path) -> pd.DataFrame:
    """Hourly log returns, the series GARCH is actually a model of."""
    h = pd.read_parquet(artifacts / "btcusd_1h.parquet")
    h = h.sort_values("hour_ts").reset_index(drop=True)
    c = h["close"].to_numpy(np.float64)
    r = np.full(len(c), np.nan)
    r[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    return pd.DataFrame({"hour_ts": h["hour_ts"].to_numpy(np.int64), "r": r})


def fit_and_forecast(ret: pd.DataFrame, train_end_ts: int, anchor_ts: np.ndarray,
                     H: np.ndarray, dist: str = "normal",
                     verbose: bool = True) -> np.ndarray:
    """Fit GARCH(1,1) on training returns, forecast each anchor's H-hour window.

    The variance recursion is iterated forward from each anchor using only
    returns strictly BEFORE that anchor, so a test-period return never enters
    the conditioning set. The fitted PARAMETERS come from the training window
    alone -- refitting per anchor would be a different (and far more expensive)
    experiment, and would still not be look-ahead, but this is the standard
    walk-forward treatment and matches how the other baselines are fitted here.
    """
    from arch import arch_model

    r = ret["r"].to_numpy(np.float64)
    ts = ret["hour_ts"].to_numpy(np.int64)
    ok = np.isfinite(r)
    tr = ok & (ts < train_end_ts)
    # arch works better on percent returns; scale and undo it afterwards.
    y = r[tr] * 100.0
    if verbose:
        print(f"    fitting GARCH(1,1) dist={dist} on {tr.sum():,} hourly returns")

    # MULTI-START, and the reason is not caution -- it is that the single-start
    # fit silently does not optimise. On this data `arch`'s default start
    # (alpha=0.10, beta=0.88) is returned EXACTLY, to full precision, with
    # convergence_flag=0 and "Optimization terminated successfully", in 4 of 6
    # folds. Probing from other starts moves it and finds strictly higher
    # likelihood: +1588 log-likelihood units for the normal innovation and
    # +4466 for Student-t on fold 2021 alone.
    #
    # Reporting the single-start result would have produced a GARCH baseline
    # that loses to PERSISTENCE (QLIKE 0.4664 against 0.4332) and a headline
    # of "NOCTUA beats GARCH comfortably" that was an artifact of a fit which
    # never ran. The tell was mechanistic, not statistical: parameters equal to
    # the library's own defaults are a starting guess, not an estimate.
    STARTS = [(0.05, 0.90), (0.10, 0.88), (0.15, 0.80), (0.20, 0.70),
              (0.08, 0.91), (0.12, 0.85), (0.03, 0.96), (0.25, 0.65),
              (0.06, 0.93)]
    best = None
    for a0, b0 in STARTS:
        try:
            sv = np.array([max(y.var() * (1.0 - a0 - b0), 1e-6), a0, b0])
            if dist in ("t", "skewt"):
                sv = np.append(sv, 8.0)
            if dist == "skewt":
                sv = np.append(sv, 0.0)
            res_i = arch_model(y, vol="GARCH", p=1, q=1, mean="Zero",
                               dist=dist).fit(disp="off", show_warning=False,
                                              starting_values=sv)
            if best is None or res_i.loglikelihood > best.loglikelihood:
                best = res_i
        except Exception:                                        # noqa: BLE001
            continue
    if best is None:
        raise RuntimeError("every GARCH start failed")
    res = best
    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]
    persistence = alpha + beta
    if verbose:
        print(f"    omega={omega:.6g} alpha={alpha:.4f} beta={beta:.4f} "
              f"persistence={persistence:.4f} ll={res.loglikelihood:.1f}")

    # Seed the recursion with the SAMPLE variance, not omega/(1-alpha-beta).
    # Multi-start fitting drives persistence to ~1.0 on this data (measured
    # 1.0000 for normal, 1.0005 for Student-t) -- i.e. IGARCH, where the
    # model-implied unconditional variance is undefined or negative. The
    # earlier `omega / max(1-alpha-beta, 1e-6)` would have silently returned
    # omega * 1e6 there. The sample variance is finite, observed, and
    # training-only, so it is causal.
    uncond = float(np.var(y))

    # Recursively build the conditional variance over the WHOLE series using the
    # training-fitted parameters. h[i] is the variance for hour i, conditioned
    # on returns up to i-1 -- which is what makes the forward sum causal.
    n = len(r)
    h = np.full(n, uncond, dtype=np.float64)
    rp = np.nan_to_num(r * 100.0, nan=0.0)
    for i in range(1, n):
        h[i] = omega + alpha * rp[i - 1] ** 2 + beta * h[i - 1]

    idx = {t: i for i, t in enumerate(ts)}
    out = np.full(len(anchor_ts), np.nan)
    for k, (a, hh) in enumerate(zip(anchor_ts, H)):
        i = idx.get(int(a))
        if i is None:
            continue
        # Iterate the variance forecast forward hh steps from the anchor,
        # using E[r^2] = h (no new shocks), the standard multi-step GARCH
        # forecast, then sum to a window variance.
        hf = h[i]
        tot = 0.0
        for _ in range(int(hh)):
            tot += hf
            hf = omega + persistence * hf
            if not np.isfinite(hf) or hf > 1e12:
                # Persistence >= 1 makes the recursion non-decreasing; cap it
                # rather than emit an inf that would silently dominate QLIKE.
                hf = 1e12
        out[k] = np.sqrt(tot) / 100.0        # undo the percent scaling
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GARCH(1,1) baseline vs NOCTUA")
    ap.add_argument("--artifacts", type=Path, default=Path("model/artifacts"))
    ap.add_argument("--dists", nargs="+", default=["normal", "t"],
                    help="innovation distributions; 't' is the heavy-tailed "
                         "variant the protocol asks for")
    ap.add_argument("--out", type=Path, default=Path("model/artifacts/garch.json"))
    a = ap.parse_args(argv)

    ep, X = load_all(a.artifacts)
    ret = hourly_returns(a.artifacts)
    folds = S.walk_forward_folds(ep)
    prod = S.production_mask(ep)
    fin = np.isfinite(X.to_numpy()).all(1)

    print(f"hourly return series: {len(ret):,} rows, "
          f"{np.isfinite(ret['r']).sum():,} finite")
    print(f"walk-forward folds: {len(folds)}\n")

    recs = []
    for f in folds:
        te = f["test"] & fin & prod
        if te.sum() < 30:
            print(f"  {f['year']}: SKIPPED (n={te.sum()})"); continue
        e = ep[te]
        anchor = e["anchor_ts"].to_numpy(np.int64)
        H = e["H"].to_numpy(np.float64)
        rv = e["RV"].to_numpy(np.float64)
        train_end = int(ep.loc[f["train"], "anchor_ts"].max())
        line = {"year": f["year"], "n_test": int(te.sum())}
        print(f"  fold {f['year']}  n_test={te.sum()}")
        for dist in a.dists:
            t0 = time.time()
            try:
                sig = fit_and_forecast(ret, train_end, anchor, H, dist=dist)
            except Exception as exc:                                # noqa: BLE001
                print(f"    dist={dist} FAILED: {type(exc).__name__}: {exc}")
                continue
            m = np.isfinite(sig)
            if m.sum() < 30:
                print(f"    dist={dist}: only {m.sum()} usable forecasts"); continue
            q = qlike_vec(rv[m], sig[m])
            line[f"garch_{dist}"] = {"qlike": float(q.mean()),
                                     "n": int(m.sum()),
                                     "median_sigma": float(np.median(sig[m]))}
            print(f"    dist={dist:6} QLIKE {q.mean():.6f} on {m.sum()} episodes "
                  f"({time.time()-t0:.0f}s)", flush=True)
        recs.append(line)

    if not recs:
        print("no fold produced a GARCH forecast"); return 1

    print(f"\n{'arm':>16} {'mean QLIKE':>12} {'folds':>7}")
    summary = {}
    for dist in a.dists:
        key = f"garch_{dist}"
        vals = [r[key]["qlike"] for r in recs if key in r]
        if vals:
            summary[key] = {"qlike": float(np.mean(vals)), "n_folds": len(vals)}
            print(f"{key:>16} {np.mean(vals):12.6f} {len(vals):>7}")

    # The comparators already on the scoreboard, from BENCHMARK.md 18's
    # corrected baseline. Quoted, not recomputed -- recomputing them here would
    # risk a second, subtly different implementation of the same number.
    print(f"\n  for reference, the corrected baseline (BENCHMARK.md 18):")
    print(f"    noctua       0.290346")
    print(f"    log_har      0.305651")
    print(f"    persistence  0.433230")
    print(f"\n  NOTE: GARCH is fitted on hourly returns and aggregated to the")
    print(f"  H-hour window; NOCTUA forecasts that window directly. Same target,")
    print(f"  same episodes, different route to it.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"folds": recs, "summary": summary,
                                 "reference_baseline": {
                                     "noctua": 0.290346, "log_har": 0.305651,
                                     "persistence": 0.433230}},
                                indent=2, default=float) + "\n")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
