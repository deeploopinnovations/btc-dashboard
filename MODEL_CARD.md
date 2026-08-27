# Model Card — NOCTUA Volatility Forecaster

**Version:** a7db8a5 (branch `claude/btc-volatility-model-1xepb9`)  
**Data cutoff:** 2026-08-09 19:00:00 UTC  
**Status:** Research artifact. Not financial advice.

---

## 1. Model Purpose

The model predicts **Bitcoin/USD realized volatility** and **touch probabilities** at fixed barriers (0.5%, 1%, 2%, 3%, 5%) over a 19-hour forecast window. The intended use is **option premium assessment**: helping options sellers decide whether to sell premium or buy a protective straddle over a fixed window by forecasting the probability that price will breach a strike.

The model is specifically **not** designed for directional trading. Direction prediction has been closed by measurement (best skill 0.180% against 4.98% for barriers, a 27x gap).

---

## 2. Architecture

Two-stage ensemble:

**Stage A — Volatility:** Predicts log hourly volatility rate as a residual from Log-HAR (linear base with 5 terms: `har_1d`, `har_5d`, `har_22d`, `cal_H`, `cal_weekend_frac`) plus a gated residual MLP. Hidden width 32. Trained against log RV normalized by √H.

**Stage B — Barriers:** Committee of 4 specialists (neural network, Gaussian first-passage, empirical quantiles, extreme value theory), all width 32. Predicts standardized excursion quantiles conditional on a causal HAR-based volatility reference. Equal-weighted ensemble. 17 quantile levels: {0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995}.

Touch probabilities are produced by quadrature-mixing Stage B's conditional law over Stage A's predictive quantiles for volatility.

**Parameters:**
- Hidden width (Stage A & B): 32
- Ensemble committee size: 4
- Linear base terms: 5
- Quantile levels: 17
- Seeds: 3 (fixed 0, 1, 2)
- Epochs: 40 with validation-restored checkpoint
- Blend weight (Stage A contribution): 0.25

**Total model inputs:** 42 features across 6 families (realized variance measures, HAR lags, semi-variances, calendar features, regime flags, momentum/volatility-of-volatility).

---

## 3. Training Data and Cutoff

**Asset:** BTC/USD spot from Bitstamp 1-minute OHLCV.

**Training window:** 2017-08-01 to 2022-12-30 (189,831 episodes)  
**Calibration window:** 2023-01-01 to 2024-06-29 (52,359 episodes)  
**Test window:** 2024-07-01 to 2026-08-09 (73,867 episodes)

**Total episodes across all horizons H ∈ {6, 12, 19, 24}:** 510,496

**Data cutoff:** 2026-08-09 19:00:00 UTC (hard boundary; features are causal up to H-1 relative to the anchor hour).

**Embargo:** 24 hours at all split boundaries to prevent lookahead. Verified: train-to-calib and calib-to-test gaps are exactly 24 hours.

---

## 4. Baseline Metrics (Walk-Forward, 6 Folds, Production Slice)

Production slice: anchor_hour == 17 UTC, H = 19 hours.

**Volatility (QLIKE, lower is better):**

| Model | QLIKE |
|---|---|
| **noctua_v2** | **0.290346** |
| log_har_gauss | 0.305651 |
| persistence | 0.433230 |

**Barrier metrics (2% upside, lower Brier is better):**

| Model | Pinball | CRPS | Brier | DSC/UNC |
|---|---|---|---|---|
| **noctua_v2** | **0.003303** | **0.004882** | 0.195381 | 0.08834 |
| log_har_gauss | 0.003416 | 0.004916 | **0.193062** | **0.09028** |
| persistence | 0.003592 | 0.005144 | 0.198783 | 0.07769 |

**Key observation:** `log_har_gauss` beats `noctua_v2` on Brier (0.193062 vs 0.195381) and DSC/UNC (0.09028 vs 0.08834) at the 2% barrier. NOCTUA wins on pinball and CRPS — i.e., on the full predictive distribution. On binary barrier events alone, the models are approximately level.

**Calibration (Christoffersen conditional coverage, α = 5%):**

Per fold, upside p_cc: 0.196, 0.411, **0.0018**, 0.413, **0.0002**, 0.104.

Two folds fail conditional coverage on the upside (folds 3 and 5, corresponding to 2023 and 2025 test years). Fold 3 over-breaches (hit rate 9.3% vs nominal 5%); Fold 5 under-breaches (hit rate 1.1%). Over and under in opposite directions is consistent with regime shift rather than fixed miscalibration.

---

## 5. LIMITATIONS — Required Reading

### 5.1 Direction is Not Predictable

**Best skill: 0.180% against 4.98% for the same model on barriers (27× gap).**  
Nothing clears a shuffled permutation null at the deployed configuration.

Source: `model/eval/direction.py`, recorded in ledger as experiment `direction`. The literature supports this: order-flow imbalance carries ≈−0.1 to −0.4% R² one minute ahead and decays within minutes. This model forecasts 6–24 hours out.

**Consequence:** The `upside` output is **pinned to 50.0** in production because direction has no skill. The model forecasts whether a strike will be touched, not which direction it will touch first.

### 5.2 Volatility Within 10% of Realized is Not Attainable

Stated goal: predict volatility within ~10% of realized. This is not achievable.

**In-sample R² of realized volatility on the full causal feature set: 0.665**  
**Residual standard deviation in logs: 0.378**

This residual sd puts "within 10%" at roughly **40–45% of nights for an oracle with perfect parameter knowledge**.

Source: `model/ROADMAP.md` §4, citing the in-sample regression fit of RV on the base feature set.

### 5.3 Volatility Spikes Are Under-Forecast by 45%

On spike nights (volatility jumps >1σ), the model's median RV/sigma ratio is **1.453** vs **0.964** on normal nights — a **45% underforecast** of realized move relative to sigma.

Spike nights are 7.7% of episodes but carry **25.8% of total loss**.

Source: `model/ROADMAP.md` (Priority 1), citing the decomposition through NOCTUA's own Stage B mapping. The onset AUC is 0.733, indicating the information exists but modelling or parameterization is not capturing it.

### 5.4 Log-HAR+Gaussian Beats NOCTUA on Barrier Discrimination

At the 2% upside barrier:
- **NOCTUA Brier:** 0.195381
- **log_har_gauss Brier:** 0.193062 (better)

**NOCTUA DSC/UNC:** 0.08834  
**log_har_gauss DSC/UNC:** 0.09028 (better)

NOCTUA wins on pinball (0.003303 vs 0.003416) and CRPS, but on the discrimination metrics for binary barrier events, a much simpler linear baseline is competitive or superior.

Source: `BASELINE_MANIFEST.md` §7 and §8; also `model/BENCHMARK.md` §1–2.

### 5.5 Conditional Coverage Fails in Two Folds on the Upside

Christoffersen p-values for conditional coverage, upside, per fold: 0.196, 0.411, **0.0018**, 0.413, **0.0002**, 0.104.

Folds 3 and 5 fail (2023 and 2025 test years). The failures are in opposite directions: fold 3 over-breaches by 4.3 percentage points; fold 5 under-breaches by 3.9 pp. This pattern is consistent with the regime shift documented in §6i of the protocol (spot ETF launch) rather than with a fixed model miscalibration, but remains an open weakness of the baseline.

Source: `BASELINE_MANIFEST.md` §8.

### 5.6 Three of Seven Feature Families Are Empty

**Empty families:**
1. **Options / forward-looking:** No implied volatility, no options flow, no commitment of traders.
2. **Cross-asset:** No equity market, no gold, no crypto pairs, no macro indicators.
3. **Events / macro:** No central bank calendars, no regulatory announcements, no sentiment.

The model is trained exclusively on realized price microstructure (realized variance, realized semivariance, path efficiency, HAR lags, calendar structure) and one realized regime flag (post-ETF).

Source: `FEATURE_CATALOG.md` §19 (the ledger index), confirmed by reading `model/noctua/spec.py` and `model/noctua/features.py`.

### 5.7 Horizons of 1 Month and Beyond Are Blocked by Data

The current training span (2017-08-01 → 2026-08-09) gives approximately **35 non-overlapping windows** for a 3-month horizon with a 90-day embargo. That is insufficient for walk-forward validation.

Requested scope (Phase 2): short horizons (intraday → 1 week) and medium horizons (1 week → 3 months). Medium-horizon work is deferred pending data accumulation.

Source: `model/PHASES.md` §Phase 2.

---

## 6. Feature List

**42 total columns:**

- **Realized variance:** rv5, rv5_pos, rv5_neg, bpv5, rq5
- **Realized jump measures:** jump_share_1d, jump_share_5d, semi_signed_jump_1d, semi_signed_jump_5d
- **HAR lags (Group A, 31 cols):** har_1d, har_5d, har_22d (3 core); plus seasonal, rng, momentum, vov, regime flags
- **Calendar (Group C, 8 cols):** cal_hour_sin, cal_hour_cos, cal_dow_sin, cal_dow_cos, cal_H, cal_weekend_frac, cal_month_sin, cal_month_cos
- **Seasonal (Group B, 3 cols):** seas_1d, seas_5d, seas_22d

Feature causality verified: 42/42 causal. Positive control (future-looking feature) caught 12/12 times.

---

## 7. Code and Reproducibility

**SHA:** a7db8a5  
**Branch:** claude/btc-volatility-model-1xepb9  
**Determinism:** PASS. Two identical runs, seeds 0/1/2: all 351 metrics bit-identical (worst Δ 0.000e+00).

**Environment (pinned, not floored):**
```
numpy==2.4.6
pandas==3.0.5
torch==2.13.0+cu130
scikit-learn==1.9.0
scipy==1.17.1
pyarrow==25.0.0
Python 3.11.15
```

**Serving:** `serve/runtime.py` reimplements the forward pass in pure NumPy (no torch, no scikit-learn at runtime). This is a design property — serving runtime must not depend on deep-learning libraries.

---

## 8. What Changed Between Version Control and This Card

The previously committed `benchmark.json` was **stale**: it predated the freshness fix that regenerated `features.parquet` (mtime: features 2026-08-16 > episodes 2026-08-10).

The benchmark was also calling `run_fold` without `sigma_ref_fn`, so Stage B trained against realized RV (train/serve skew documented in the code) while the shipped artifact uses a causal reference.

Both issues were fixed before this card was written. The correction moved QLIKE +0.256% (worse, unconditionally adopted) and downside DSC/UNC +7.11% (better). `log_har` and `persistence` are bit-identical (0.000%) across the fix — a correct fix requires baseline-only models to be untouched, and they were.

Source: `BASELINE_MANIFEST.md` §10 and §16b.

---

*This model is an educational research artifact. It demonstrates the measurement and reporting of a volatility forecaster's strengths and weaknesses; it is not financial advice and carries no warranty.*
