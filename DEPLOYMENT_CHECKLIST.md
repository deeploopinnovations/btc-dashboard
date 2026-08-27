# Deployment Checklist — NOCTUA Model

**As of:** 2026-08-27  
**Current deployment:** noctua_v2.npz (99,820 bytes, frozen research artifact from 2026-08-10)  
**Awaiting activation:** noctua_v2_refreshed_2026-08-09.npz (110,450 bytes, production fit from 2026-08-21)

This checklist formalizes the gates that must be satisfied before the model can be deployed to production. Each gate statement its status (MET / NOT MET / PARTIAL), supporting evidence, and — where not met — what is required.

---

## Gate 1: Artifact Serialization and Versioning

**Status:** PARTIAL

**Description:** Model, preprocessing, calibration, and configuration are serialized as a single versioned artifact. The artifact includes metadata for reproducibility.

**Evidence:**
- `serve/noctua_v2.npz` exists (99,820 bytes, committed Aug 16 02:41)
- `serve/noctua_v2_refreshed_2026-08-09.npz` exists (110,450 bytes, committed Aug 21 02:14)
- Both are NumPy npz archives containing:
  - Neural network weights (committee of 4 × multiple seeds)
  - Preprocessing scalers (std_all, std_base, std_shape)
  - Hyperparameters (blend_w=0.25, quantile levels)
  - Calibration reference (clip bounds per fold)
  - Feature column lists (feat_cols, base_cols, shape_cols)
  - Log-HAR coefficients for the Stage A base

**Issues:**
- Metadata in both artifacts **lacks code SHA, fit date, training window span, and data cutoff**. The metadata dict keys seen: `version`, `n_params_total`, `feat_cols`, `base_cols`, `shape_cols`, `blend_w`, `cal_shrink` — but no `code_sha`, `fit_date`, `data_cutoff`, `training_window`. This means artifact lineage must be inferred from filename and surrounding documentation rather than from the artifact itself.
- The refreshed artifact (noctua_v2_refreshed_2026-08-09.npz) **has NOT been activated in production**. The loading logic in `serve/runtime.py:load_model()` checks `if (here / V2_NAME).exists()` where `V2_NAME = "noctua_v2.npz"`. The frozen research artifact is served.

**What is required to fully meet this gate:**
1. Re-export both artifacts with complete metadata: `code_sha`, `fit_date`, `training_window_start`, `training_window_end`, `data_cutoff`, `eval_folds`, `eval_metrics_path`.
2. Decide: keep serving noctua_v2.npz (frozen, research), or swap to noctua_v2_refreshed_2026-08-09.npz (production fit). The current state is not a deployment block — it is a deliberate holding pattern — but it must be explicit in version control (e.g., a `ACTIVE_MODEL` file or a runtime config file).

---

## Gate 2: Code SHA, Environment, Data Snapshot

**Status:** MET with caveats

**Description:** The artifact is tethered to specific code, dependencies, and data snapshots so that re-running the pipeline reproduces exact metrics.

**Evidence:**

**Code SHA:**
- Documented in `BASELINE_MANIFEST.md` §1: `a7db8a5` (branch `claude/btc-volatility-model-1xepb9`)
- Verified determinism: two runs, identical inputs, seeds 0/1/2 → all 351 metrics bit-identical (worst Δ 0.000e+00)
- Feature causality: 42/42 features pass causal audit; positive control caught 12/12 times

**Data Snapshot:**
- `BASELINE_MANIFEST.md` §3 documents all parquets:
  - `episodes.parquet`: 24,724,722 bytes, partial MD5 `99245f3e6c91d3c0`
  - `features.parquet`: 60,883,811 bytes, partial MD5 `b6368e6b858a1df8`
  - `btcusd_1min.parquet`: 157,328,807 bytes, partial MD5 `9e89946289374ca7`
  - `btcusd_1h.parquet`: 13,094,438 bytes, partial MD5 `e2e8ed96e5f076e5`
  - `funding_btc.parquet`: 2,723,380 bytes, partial MD5 `0c220941589b06af`
  - `dvol_btc.parquet`: 8,496 bytes, partial MD5 `f6faf4f4e0027263`
- Note: Hashes are **partial** (first + last 1 MB) due to container timeout; full content hashing of the 157 MB file exceeds limits.

**Environment (pinned, not floored):**
- `model/requirements-research.txt`: numpy==2.4.6, pandas==3.0.5, torch==2.13.0+cu130, scikit-learn==1.9.0, scipy==1.17.1, pyarrow==25.0.0
- Python 3.11.15
- `model/serve/requirements.txt` deliberately omits torch and scikit-learn — the serving runtime uses pure NumPy

**Issues:**
- Data snapshot pinning **requires manual** re-ingestion from the Bitstamp mirror. No workflow re-runs `noctua.ingest` → `noctua.episodes` → `noctua.features` end to end. Stages 0→1, 1→2, 2→3 are manual steps (`DATA_LINEAGE.md` §1: "Not run by any workflow").
- The Bitstamp mirror source (`ff137/bitstamp-btcusd-minute-data`) freshness relative to live market is **unverified this session** (`FEATURE_CATALOG.md` §0.4). The claim "daily-updated, verified live to 2026-08-10 01:57 UTC" is from a prior session.

**What is required:**
- Document the manual ingest process formally (steps, expected outputs, verification checks).
- Either (a) add a scheduled workflow to re-run Stages 0→3 on a fixed cadence, or (b) document why manual control is intentional.

---

## Gate 3: Feature Parity (Offline / Online)

**Status:** PARTIAL

**Description:** Features used in training are computed identically in the serving path. No train/serve skew.

**Evidence:**

**Code-level parity:**
- Training uses `model/noctua/features.py:build_features()`
- Serving uses the same `noctua/features.py` imported in `serve/predict.py:forecast()` (line 42)
- Both use the same 42 feature columns, same order, same lag groups (A/B/C documented in `FEATURE_CATALOG.md` §0.2)
- Causality audit: 42/42 features pass; no lookahead detected (zero violations over 306,261 episodes)

**Offline / Online paths:**
- Offline: `serve/history.py:load_bundle()` loads committed `data/noctua_history.parquet` (365 days of hourly data)
- Online: `serve/fetch.py:fetch_bars()` fetches 5-minute bars from Bitstamp, aggregates to hourly
- Both feed into the same `build_features()` call (line 86 of `serve/predict.py`)

**Issues:**
- **Online fetch lag is unverified**: `serve/fetch.py` fetches from Bitstamp; the actual latency of the public API relative to true market time is not documented. Network jitter and API rate limits could introduce multi-second delays.
- **Committed history freshness**: `data/noctua_history.parquet` is a static bundle. The last hour it contains is likely several commits old. If the online fetch fails, serving falls back to the committed bundle's last row, which could be hours stale.
- **Feature computation environment**: `serve/adaptive.py:volatility_correction()` applies a learned calibration factor to sigma. This is computed from settled episodes (line 95 of `serve/predict.py`). The calibration window is measured in real time but estimated from historical data — the guarantee is that it carries no lookahead (`serve/adaptive.py`'s own docstring). **This is verified by `model/tests/test_adaptive.py`** but the verification is not run in the live serving environment, only in CI.

**What is required:**
- Document the online fetch latency SLA (target freshness, expected jitter, timeout/fallback behavior).
- Add a runtime check: if online fetch is stale (>1 hour old), emit a warning and require explicit user acknowledgement before serving.
- Run `test_adaptive.py` as a pre-flight check before each forecast, or document why it is only a CI gate.

---

## Gate 4: Hyperparameters and Architecture

**Status:** MET

**Description:** The model's architecture and hyperparameters are documented and reproducible.

**Evidence:**
- `BASELINE_MANIFEST.md` §6: Hidden width 32, seeds 3 (0,1,2), epochs 40, committee of 4 specialists, blend_w 0.25
- `BASELINE_MANIFEST.md` §5: 42 feature columns, 17 quantile levels
- `TARGET_SPEC.md` §1: Stage A is log hourly volatility rate; Stage B predicts standardized excursions conditional on causal HAR reference
- Quantile levels hardcoded in `serve/predict.py:ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)` and barrier grid in `BARRIER_GRID_PCT = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)`
- Model metadata in artifacts: `blend_w`, `feat_cols`, `base_cols`, `shape_cols` are read from the artifact at runtime

**No issues identified.**

---

## Gate 5: Evaluation Report and Metrics

**Status:** MET

**Description:** Out-of-sample performance is documented with a reproducible rule-book (pre-registered before metrics were computed).

**Evidence:**
- Walk-forward validation: 6 expanding folds, test years 2021–2026 (production slice, anchor_hour=17, H=19)
- Baseline metrics documented in `BASELINE_MANIFEST.md` §7–8:
  - QLIKE: noctua 0.290346 (beats log_har 0.305651, persistence 0.433230)
  - Brier at 2% up: noctua 0.195381 (loses to log_har_gauss 0.193062)
  - Calibration (Christoffersen p_cc): 2 of 6 folds fail upside conditional coverage (folds 3 & 5)
- Proper scores documented in `model/BENCHMARK.md` §1–2
- Reproducibility verified in `BASELINE_MANIFEST.md` §9: determinism check passed (bit-identical)
- Protocol for future phases documented in `model/PHASES.md` with gates for each phase

**Issues:**
- The two artifacts in `serve/` (frozen and refreshed) have **different evaluation records**:
  - Frozen (noctua_v2.npz): 6 folds, trained 2017-08 → 2022-12, calibrated 2023-01 → 2024-06
  - Refreshed (noctua_v2_refreshed_2026-08-09.npz): 6 folds, trained 2017-08 → 2024-06, calibrated 2024-07 → 2025-12 (and rolled forward to 2026-07 on each fold)
- The refreshed artifact's evaluation metrics are **not in this repository's documentation** (BASELINE_MANIFEST.md, MODEL_CARD.md describe the frozen artifact). The refresh was adopted in experiment `6o` in the ledger, but the full evaluation report for the refreshed artifact does not exist as a committed markdown file.

**What is required:**
- Create and commit `REFRESHED_ARTIFACT_EVALUATION.md` documenting the 6-fold walk-forward metrics for noctua_v2_refreshed_2026-08-09.npz, following the same structure as BASELINE_MANIFEST.md.
- If the refreshed artifact is to be deployed, update BASELINE_MANIFEST.md (or rename it to BASELINE_MANIFEST_FROZEN.md and create a new one for the active artifact).

---

## Gate 6: Calibration Validation

**Status:** NOT MET

**Description:** The model's probability forecasts are well-calibrated under real conditions. Christoffersen conditional coverage and sharpness are validated.

**Evidence:**
- Conditional coverage p-values computed for all 6 folds (upside and downside)
- Upside p_cc: 0.196, 0.411, 0.0018, 0.413, 0.0002, 0.104
- **Two folds fail** (p < 0.05): fold 3 (2023) at p=0.0018, fold 5 (2025) at p=0.0002

**Why it is NOT MET:**
Failing Christoffersen in 2 of 6 folds means the model's confidence intervals are miscalibrated in those years. Fold 3 over-breaches (hit rate 9.3% vs 5% nominal); fold 5 under-breaches (hit rate 1.1%). The failures are in opposite directions, consistent with regime shift rather than a fixed miscalibration, but are not resolved. A deployed model must either:
1. Pass conditional coverage in all folds, OR
2. Document the miscalibration explicitly and implement a calib-year-aware correction, OR
3. Reduce the stated confidence level in those years

**What is required:**
1. Investigate whether the 2023 and 2025 miscalibrations are structural (market regime change documented in `model/ROADMAP.md` §3 — spot ETF launch 2024 moved from pre-ETF to post-ETF) or model-specific.
2. If structural: document it in the model card and implement a regime-aware calibration correction (e.g., different clip bounds per regime).
3. If model-specific: retrain with a different loss function (e.g., pinball loss with different quantiles, or a recalibration layer).
4. Re-validate on 2026 data once sufficient episodes are available (currently only 21,163 test episodes for 2026, truncated at data end).

---

## Gate 7: Serving Runtime and Latency

**Status:** MET

**Description:** The serving runtime is fast, dependency-light, and tested.

**Evidence:**
- Pure NumPy + SciPy serving runtime in `serve/runtime.py` (no torch, no sklearn)
- Weights file: 99,820 bytes (v2) or 110,450 bytes (refreshed)
- Latency target: ~6 ms per forecast (per `serve/README.md`)
- CI tests in `.github/workflows/model-ci.yml`:
  - `test_serving.py`: parity between PyTorch training runtime and NumPy serving runtime
  - `test_history.py`: history loading and feature alignment
  - `test_adaptive.py`: causality of the adaptive calibration
  - `test_selfimprove.py`: guards on online self-improvement (ACI coverage guarantee)
  - Smoke tests on the exported artifacts (line 47)

**No issues identified.**

---

## Gate 8: Canary / Shadow Deployment

**Status:** NOT MET

**Description:** The model is shadow-forecast against live data for a period before being made primary.

**Evidence / Issue:**
- No shadow deployment infrastructure is documented or committed to this repository.
- The serving space is published at a known URL (`serve/README.md` mentions it is on Hugging Face), but no A/B test or canary infrastructure is described.
- The live dashboard (`src/data.js`) consumes the `/api/noctua` endpoint; a canary would require dual-endpoint logic in the frontend, which is not documented.

**What is required:**
1. Deploy the new model to a shadow environment (separate Hugging Face Space, or a parallel endpoint) for 1–2 weeks.
2. Log forecasts from both the old and new models (or old and new artifacts) at the same times.
3. Compare forecasts on real-time data: do touch probabilities diverge significantly? Are there systematic shifts in calibration?
4. If shadow metrics are acceptable, swap to the new model as primary.
5. Keep the old model as a fallback for 30 days before decommissioning.

---

## Gate 9: Rollback and Fallback

**Status:** MET with documented limitations

**Description:** The serving system can roll back to a prior known-good model or to a simple baseline if the primary model fails.

**Evidence:**
- **Primary artifact:** `serve/noctua_v2.npz` (currently served)
- **Fallback to older version:** `serve/noctua_weights.npz` (v1, 203,046 bytes, dated Aug 10 14:44) is available as a fallback. The loading logic in `serve/runtime.py:load_model()` checks v2 first, then falls back to v1.
- **Simple baseline:** `model/noctua/baselines.py` implements `log_har_gauss` (Log-HAR + Gaussian first-passage). This requires PyTorch at serving time, so it is not suitable as an in-production fallback, but it is documented and reproducible offline.
- **Persistence baseline:** Trailing realized volatility is always available and requires no model. Could be used as an emergency fallback.

**Issues:**
- The fallback logic is implicit (if v2 not found, load v1). There is no explicit versioning file (e.g., `ACTIVE_ARTIFACT_VERSION.txt`) that can be edited to swap between v2 and refreshed-v2 without code changes.
- The v1 fallback (noctua_weights.npz) was exported before some fixes (e.g., the freshness fix in experiment `freshness`). Serving from v1 is a regression, not a stable fallback.

**What is required to fully meet:**
1. Create a runtime config file (e.g., `model/serve/ACTIVE_ARTIFACT.txt` or an env var `NOCTUA_ARTIFACT`) that specifies which artifact to load.
2. Document the rollback procedure: how to edit the config, how to verify the new artifact loads, how to monitor for regressions.
3. Re-export v1 (noctua_weights.npz) with the same freshness fix as v2, or remove it from the fallback chain.

---

## Gate 10: Monitoring and Alerting

**Status:** PARTIAL

**Description:** The live serving system detects when its forecasts diverge from reality and alerts.

**Evidence:**
- `serve/adaptive.py` implements online calibration: compares forecast to settled episodes and applies a multiplicative correction to sigma (lines 96–150).
- The correction is measured only on settled episodes (no lookahead): window_start >= fit_end + embargo.
- CI test in `.github/workflows/model-ci.yml` (lines 56–59) gates the causality of this correction: `test_adaptive.py` must pass.

**Issues:**
- The online calibration is **enabled by default** (line 97 of `serve/predict.py`), but there is **no threshold for disabling it**. If the correction factor diverges from 1.0 significantly, serving continues without escalation.
- No production alerting is documented (e.g., "if correction factor > 1.2 for 7 consecutive days, page the oncall").
- The correction is **epoch-level** (applied to every forecast), not hour-level. Drift that develops gradually may not be caught.

**What is required:**
1. Add thresholds for the correction factor (e.g., if |1 - factor| > 0.15 for >5 consecutive days, trigger an alert).
2. Document the alerting policy and who is notified.
3. Log all forecasts and their corrections to a central system for offline analysis.

---

## Gate 11: Deployment Documentation and Runbooks

**Status:** PARTIAL

**Description:** Clear runbooks exist for deployment, rollback, and troubleshooting.

**Evidence:**
- `serve/README.md` documents the model, endpoints, and data requirements.
- `serve/predict.py` has inline docstrings explaining the forecast pipeline.
- `.github/workflows/model-ci.yml` documents the PR gate.

**Issues:**
- No runbook exists for:
  - How to export and commit a new artifact
  - How to re-ingest data from the Bitstamp mirror
  - How to run the full walk-forward evaluation
  - How to swap to the refreshed artifact (currently both artifacts exist but no procedure is documented)
  - How to respond if conditional coverage fails
  - How to respond if the online calibration diverges

**What is required:**
- Create `DEPLOYMENT_RUNBOOK.md` with step-by-step procedures for:
  - Exporting a trained model to an npz artifact
  - Ingesting new data (full pipeline: ingest → episodes → features)
  - Running the evaluation (benchmark, falsification, regime tests)
  - Promoting a model from research to production
  - Rolling back to a prior version
  - Emergency fallback (serving from baseline if model fails)

---

## Gate 12: Versioning and Lineage

**Status:** PARTIAL

**Description:** Every artifact, code version, and data snapshot is tagged and traceable.

**Evidence:**
- Code SHA: `a7db8a5` documented in BASELINE_MANIFEST.md
- Data cutoff: 2026-08-09 documented in BASELINE_MANIFEST.md
- Artifact mtimes: noctua_v2.npz (2026-08-10), noctua_v2_refreshed_2026-08-09.npz (2026-08-21)
- Features parquet updated at 2026-08-16 (postdating the artifact freeze, consistent with manual re-run)

**Issues:**
- Metadata **inside the artifacts** lacks SHA, fit date, training window. The only tag is `version: "NOCTUA-v2"`, which is the same for both the frozen and refreshed artifacts.
- No git tag links the code SHA to a commit.
- The refreshed artifact is named by its fit date (`2026-08-09`) rather than by a version number or release tag.

**What is required:**
1. Tag the code commit `a7db8a5` with `v0.1-frozen` or similar.
2. Re-export the artifacts with complete metadata (code_sha, fit_date, training_window_end, evaluation_folds, evaluation_metrics_path).
3. If the refreshed artifact is promoted, create a git tag (e.g., `v0.2-refreshed-2026-08-21`) and rename the file to match (e.g., `noctua_v2_2026-08-21.npz`).
4. Create a `VERSIONS.md` file documenting the lineage of all deployed or near-deployment artifacts.

---

## Summary: Gates MET, PARTIAL, and NOT MET

| Gate | Status | Notes |
|---|---|---|
| 1. Artifact Serialization | PARTIAL | Both artifacts exist; metadata lacks SHA/dates; refresh not activated |
| 2. Code SHA, Environment, Data | MET | Code pinned; environment pinned; data partially hashed; ingest is manual |
| 3. Feature Parity (offline/online) | PARTIAL | Code-level parity verified; online latency unverified; adaptive correction only gated in CI |
| 4. Hyperparameters and Architecture | MET | Fully documented and reproducible |
| 5. Evaluation Report and Metrics | MET | Frozen artifact fully evaluated; refreshed artifact evaluation not committed |
| 6. Calibration Validation | NOT MET | Two folds fail Christoffersen conditional coverage (2023, 2025) |
| 7. Serving Runtime and Latency | MET | Pure NumPy, fast, well-tested |
| 8. Canary / Shadow Deployment | NOT MET | No shadow infra documented; no A/B test plan |
| 9. Rollback and Fallback | MET | v1 and v2 both available; log_har baseline available; no explicit version config |
| 10. Monitoring and Alerting | PARTIAL | Online calibration exists; no alert thresholds or escalation policy |
| 11. Deployment Runbooks | PARTIAL | High-level docs exist; step-by-step procedures not documented |
| 12. Versioning and Lineage | PARTIAL | Code and data dated; artifacts lack internal metadata; no version tags in git |

---

## Critical Path to Production

If the goal is to deploy **noctua_v2_refreshed_2026-08-09.npz** as the primary model:

1. **Immediate (blocking):**
   - [ ] Create `REFRESHED_ARTIFACT_EVALUATION.md` with full 6-fold metrics
   - [ ] Investigate Christoffersen failures in 2023 & 2025; document root cause (regime vs model)
   - [ ] If regime-related, document and accept the miscalibration; otherwise, retrain

2. **Short-term (before canary):**
   - [ ] Update artifact metadata with code SHA, fit date, training window, data cutoff
   - [ ] Re-export both artifacts with complete metadata
   - [ ] Create explicit runtime config to swap between frozen and refreshed
   - [ ] Deploy to shadow environment for 1–2 weeks

3. **Before production swap:**
   - [ ] Confirm shadow metrics are acceptable
   - [ ] Document canary results and any issues found
   - [ ] Update monitoring thresholds for the new model's calibration factor
   - [ ] Create rollback plan (keep frozen v2 available, test restore procedure)

4. **Post-deployment:**
   - [ ] Keep the old model as a fallback for 30 days
   - [ ] Monitor calibration drift weekly
   - [ ] Collect settlement data to re-validate Christoffersen quarterly

---

*Educational research artifact, not financial advice.*
