# BASELINE_MANIFEST — Phase 0

The reproducible baseline for every later phase. Produced under the protocol's
Rule 2: nothing is modified until the benchmark reproduces.

**This manifest documents the CORRECTED configuration** — the one that ships —
not the configuration the benchmark scored before §16. See "What changed and
why" below.

## 1. Provenance

| | |
|---|---|
| code SHA | `a7db8a5` (branch `claude/btc-volatility-model-1xepb9`) |
| benchmark command | `python -m model.eval.benchmark` |
| data cutoff | 2026-08-09 19:00:00 UTC |
| timezone | all timestamps UTC; `anchor_ts` is Unix seconds |
| wall clock | ~592 s for 6 folds × 3 seeds |
| peak RSS | ≈1.70 GB (sampled at 15 s intervals; `/usr/bin/time -v` unavailable) |

## 2. Environment — pinned, not floored

`model/requirements-research.txt`:

    numpy==2.4.6   pandas==3.0.5   torch==2.13.0+cu130
    scikit-learn==1.9.0   scipy==1.17.1   pyarrow==25.0.0

Python 3.11.15. **Determinism was verified with these exact versions**; changing
any invalidates §17's verification until re-run.

`model/serve/requirements.txt` deliberately omits torch and scikit-learn —
`serve/runtime.py` reimplements the forward pass in pure NumPy so the serving
image carries no deep-learning runtime. That is a design property, not an
omission, and must not be "fixed".

## 3. Data snapshot

| file | bytes | fingerprint |
|---|---|---|
| `episodes.parquet` | 24,724,722 | md5 `99245f3e6c91d3c0` |
| `features.parquet` | 60,883,811 | md5 `b6368e6b858a1df8` |
| `btcusd_1min.parquet` | 157,328,807 | **partial** `9e89946289374ca7` |
| `btcusd_1h.parquet` | 13,094,438 | **partial** `e2e8ed96e5f076e5` |
| `data/newdata/funding_btc.parquet` | 2,723,380 | **partial** `0c220941589b06af` |
| `data/newdata/dvol_btc.parquet` | 8,496 | **partial** `f6faf4f4e0027263` |

**partial** = size plus md5 of the first and last 1 MB. Full content hashing of
the 157 MB file exceeds the container's command timeout. Stated rather than
implied, because a partial hash detects a replaced snapshot but not an interior
edit.

## 4. Universe, horizons, splits

Single asset: BTC/USD. 510,496 episodes, 42 feature columns.
Horizons H ∈ {6, 12, 19, 24} hours. Production slice: `anchor_hour == 17`, H=19.

| split | n | span |
|---|---|---|
| train | 189,831 | 2017-08-01 → 2022-12-30 |
| calib | 52,359 | 2023-01-01 → 2024-06-29 |
| test | 73,867 | 2024-07-01 → 2026-08-09 |

`TRAIN_END` 2023-01-01, `CALIB_END` 2024-07-01, **embargo 24 h = max(H)**,
verified at +24.0 h on both boundaries in both split mechanisms (§16).
Walk-forward: 6 anchored folds, test years 2021–2026.

## 5. Targets

`RV` = √(Σ 5-minute realized variance over the H-hour forward window).
`M_up`/`M_dn` = running extremes over hourly high/low (themselves the max/min of
1-minute bar highs/lows), relative to `s_tau` = the close of the hour before the
anchor. Model target for Stage A is the log hourly vol rate; Stage B predicts
standardized excursion quantiles conditioned on σ.

## 6. Hyperparameters

hidden 32 · seeds 3 (0,1,2) · epochs 40 with validation-restored best
checkpoint · width-32 committee of 4 specialists (neural, gaussian, empirical,
EVT), equal-weighted · `blend_w` 0.25 · Stage B σ-reference
`causal_har_1d_clipped`, clip bounds refit per fold on that fold's training
episodes.

## 7. Aggregate metrics — THE BASELINE

Volatility QLIKE, 6-fold walk-forward:

| model | QLIKE |
|---|---|
| **noctua** | **0.290346** |
| log_har | 0.305651 |
| persistence | 0.433230 |

Barrier metrics (upside, 2 % barrier):

| model | pinball_up | crps_up | brier_up_2.0 | DSC/UNC |
|---|---|---|---|---|
| **noctua_v2** | **0.003303** | **0.004882** | 0.195381 | 0.08834 |
| log_har_gauss | 0.003416 | 0.004916 | **0.193062** | **0.09028** |
| persistence | 0.003592 | 0.005144 | 0.198783 | 0.07769 |
| scaled_clim | 0.003505 | 0.005070 | 0.196106 | 0.08871 |
| climatology | 0.003769 | 0.005463 | 0.216664 | 0.00000 |
| noctua_shuffled | 0.003810 | 0.005553 | 0.215179 | 0.01251 |

**Stated plainly: `log_har_gauss` beats `noctua_v2` on Brier and DSC/UNC at the
2 % up barrier.** NOCTUA wins pinball and CRPS. This is consistent with what
this repository has already reported — that NOCTUA does not clearly beat
`log_har_gauss` on barrier discrimination (4/6 folds, t = +0.46, noise) — and it
is repeated here rather than buried because a manifest that only records the
model's wins is not a baseline, it is advertising.

`climatology` at DSC/UNC 0.00000 is the by-construction check: a constant
forecaster has exactly zero discrimination. `noctua_shuffled` at 0.01251 is the
shuffled-input null.

## 8. Calibration — Christoffersen conditional coverage at α = 5 %

Per fold, upside: p_cc = 0.196, 0.411, **0.0018**, 0.413, **0.0002**, 0.104.
Downside: p_cc = 0.166, 0.112, 0.929, 0.148, 0.411, 0.331.

**Two folds fail conditional coverage on the upside** (2023 at hit-rate 9.3 %
against a nominal 5 %, and 2025 at 1.1 %). Failing in opposite directions —
over-breaching then under-breaching — is consistent with the regime shift
documented in §6i rather than with a fixed miscalibration. Recorded as an open
weakness of the baseline, not smoothed over.

## 9. Integrity

| check | result | reference |
|---|---|---|
| determinism, two runs | **351/351 metrics bit-identical**, worst Δ 0.000e+00 | §17 |
| feature causality, 42 cols | **42/42 CAUSAL**, positive control caught 12/12 | §16 |
| train/test window overlap | **0 hours** | §16 |
| scaler fit on train only | unmoved (0.000e+00) under test-row corruption | §16 |
| eval path = shipped config | **fixed in §16**; was training against realized RV | §16 |

**UNVERIFIED:** Deribit API semantics for the funding/DVOL timestamp columns
were inferred from data structure, not from fetched documentation — no network
call is possible from this container.

## 10. What changed and why

The previously committed `benchmark.json` was **stale** (§16): it predated the
freshness fix that regenerated `features.parquet`. Separately, the benchmark
was calling `run_fold` without `sigma_ref_fn`, so Stage B trained against
realized RV — a train/serve skew the code's own docstring documents — while the
shipped artifact uses the causal reference.

Correcting it moved QLIKE **+0.256 %** (worse) and downside DSC/UNC **+7.11 %**
(better). Adoption was pre-registered as unconditional, better or worse,
because it is what ships. `log_har` and `persistence` are **bit-identical
(0.000 %)** across the change — neither touches Stage B, so a correct fix must
leave them untouched, and it did.

*Educational research only. Not financial advice.*
