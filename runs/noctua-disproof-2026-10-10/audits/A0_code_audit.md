# A0 code audit: fresh_test.py (NOCTUA, 2024-07-01 to 2026-10-10)

**Verdict: DEFECTS FOUND** (2 blocking, 9 minor). No lookahead found in the scored inputs.

Scope: `fresh_test.py`, `noctua/features.py`, `serve/runtime.py`, `serve/adaptive.py`, `serve/predict.py` (correction and reporting path), `noctua/infer.py`, `policy/dataset.py`, `policy/backtest.py`, `policy/runtime.py`, `policy/README.md`, `policy/weights/noctua_trader_v1_metrics.json`. Numbers below come from `evals/fresh_test.json` and from read-only recomputations (not saved to the repo). Nothing in the repo was edited except this file.

---

## Blocking

### B1. The served correction is estimated on all 24 anchor hours but applied to the 17:00 product

- **Where:** `model/serve/adaptive.py:96` and `:184` (`_settled_anchors` with `STRIDE_HOURS`, every hour, not only 17:00). Replicated faithfully at `fresh_test.py:146-156`. Applied at `model/serve/predict.py:159-161`. The reported sigma is `sig_mean` from that corrected object (`predict.py:239-240`).
- **Failure:** On the 831 test anchors at 17:00, median RV/sigma_med is 0.853 raw and 0.861 after the shipped correction. Over all test anchors the same median is 0.989. The trailing factor averages over all hours, so it sits at about 0.98 (range 0.91 to 1.07) and barely moves the 17:00 forecast. Measured effects:
  - The documented "median RV/sigma 0.874 -> 0.990" in the `adaptive.py` header is not reproduced at the served hour.
  - Reported sigma (`sigma_mean x factor`, the product's reported number) has QLIKE 0.2996 against `har5_frozen` at 0.2780, a 7.8% loss with a CI entirely below zero (`fresh_test.json`, `paired_vs`). With a causal trailing factor estimated from settled 17:00 episodes only (diagnostic run, not in the repo), the same 831 anchors give QLIKE 0.2344.
  - Barrier breaches at 17:00 sit far below nominal: up 1%: 1/831 (0.12%); dn 1%: 5/831; up 5%: 16/831 (1.9%) (`fresh_test.json`, `tails`).
- **Direction:** this defect makes the shipped reported forecast look worse than its own causal design allows. The headline "served NOCTUA loses to HAR" therefore depends on an implementation choice, not only on the model. Blocking because any verdict on the served product is conditional on it.
- **Needed:** either estimate the factor on the served anchor hour, or pre-register the all-hour design and report the 17:00 result with this caveat.

### B2. The "fresh" test is not fresh for the trader block, and the NOCTUA design was chosen on the same era

- **Where:** `fresh_test.py:341-375` scores trader_v1 over 2024-07 to 2026-10. `model/policy/weights/noctua_trader_v1_metrics.json` already has a `test` block over that period, and `model/policy/README.md:39` says the test split is "scored **once**, after selection". The `adaptive.py` docstring reports "MEASURED EFFECT (test era, 2024-07 onward, out of sample)" and a window-robustness check "on the test era". `predict.py:68` (`REPORT_FUNCTIONAL = "mean"`) was adopted after functional comparisons on the production slice (`predict.py:200-215`).
- **Failure:** the fresh trader numbers do not reproduce the shipped test numbers over the same period. Fresh: Sharpe 0.442, CAGR 2.46%, 830 days (2024-07-01 17:00 to 2026-10-08 17:00). Shipped: Sharpe 0.482, CAGR 2.71%, 826 days. The cause is not investigated (data bundle or sample definition; **unverified**). The NOCTUA correction design and reporting functional were chosen while looking at the 2024-07+ data, so the NOCTUA numbers here are not out-of-sample for design decisions.
- **Needed:** label the trader block as a re-score, show shipped and fresh figures side by side, and state the design contamination.

---

## Minor

### M1. Direction block mislabels its statistics
- `fresh_test.py:332` computes `base` (first-half base rate), which is never used. `:335` `brier_half` uses a constant 0.5, not that base rate. `:337` `auc_like_corr` is a Pearson correlation, not an AUC.
- No effect on the conclusion: Brier 0.2495 against 0.25, so no measurable direction skill.

### M2. The trader block has no exposure-matched control
- Average trader position is 0.139 (`fresh_test.py:343-346`). The Sharpe comparisons against 1x buy-and-hold, 0.5x and vol-target rules (`:354-355`, target 0.25, cap 0.5) cannot separate timing from exposure. The README's own constant-0.14 control (Sharpe 0.41) is absent from this block.
- The vol-target defaults differ from `policy/backtest.py:82` (target 0.5, cap 1).
- Direction: the Sharpe edge cannot be attributed to timing from this block.

### M3. Trader execution and cost assumptions (magnitude unverified)
- Entry is `close[a-1]`, the 17:00 print that the decision also uses (`policy/dataset.py:133-135`, `:174`). No latency, no spread beyond 6 bps.
- Funding is the Deribit inverse perpetual `interest_1h` (`dataset.py:144`, `:172`; `data/newdata/funding_btc.parquet`, `source = deribit`), applied to a linear BTCUSDT simple return (`backtest.py:29-30`; the docstring at `dataset.py:4` says BTCUSDT).
- Direction: flatters the trader if real execution is worse than the close print.

### M4. Missing funding is treated as zero
- `dataset.py:86` (`nan_to_num`) and `:177` (`np.where(..., 0.0)`). Only 7 hourly funding values are missing, all in the final hours after 2026-10-08 07:00, so the scored sample is unaffected. Worth fixing for rebuilds.

### M5. Asymmetric scaling conventions across arms
- Baselines get an in-sample QLIKE rescale (`fresh_test.py:210`, `:224-225`, `:228`, `:233-234`). NOCTUA's arms do not (`:178-179`).
- The Gaussian barrier arm uses that mean-type scaled `har5_frozen` (`:309`, `:316`), while NOCTUA's median-based arms are unscaled.
- Direction: this handicaps NOCTUA's raw median arm in QLIKE, so it is conservative toward NOCTUA, but the paired table is not like-for-like. Should be stated.

### M6. The barrier path contradicts the code's own comments
- `predict.py:159-161` moves `sigma_atoms`, which are the barrier curves, using the median factor. `predict.py:170-180` says the trailing scalar must never reach the predictive object, and the `qlike_scale` docstring in `adaptive.py` says the same.
- `fresh_test.py:304` replicates the moved-level path, so the served barrier arm measures that path. The numerical effect is small (up 5%: 16 vs 17 breaches), but code and documentation disagree on what ships.

### M7. Data provenance at the splice (unverified)
- The comment at `fresh_test.py:42` says the asset history ends 2026-08-15. In fact `data/assets/btc_history.parquet` ends 2026-08-15 11:00, `data/noctua_history.parquet` ends 2026-10-05 17:00, and live Bitstamp/Coinbase bars cover the rest (`model/serve/fetch.py:181`).
- Mean rv5 falls from 1.91e-5 to 1.62e-5 across 2026-08-15. Whether this is regime or venue was not tested.
- The `fresh_2026-08-15+` subset is only 56 episodes.

### M8. Known calendar defect in the shipped artifact (not lookahead)
- `noctua/features.py:296` (`+ 4`) makes `fut_dow >= 5` count Friday and Saturday, not the weekend (documented at `features.py:289-293`). `cal_weekend_frac` is in the artifact's `base_cols`. The defect is pinned for train/serve parity, so NOCTUA's calendar effect is mismeasured.

### M9. Rolling-window summary is descriptive only
- `fresh_test.py:278-281` (`frac_windows_positive`) uses overlapping 180-day windows, so it is not an independent test.

---

## Checked and clean

- **Trailing sums and EWMA.** `fresh_test.py:86-92` and `noctua/features.py:55-73` are exclusive of the anchor row. The EWMA loop (`fresh_test.py:167-172`) uses rv5 up to i-1. Seasonal features (`features.py:250-264`) end at or before the anchor.
- **DVOL.** `dvol_prev` (`fresh_test.py:164`) uses the value stamped a-1. The horizon conversion `sqrt(19/8760)` (`:165`) is correct, and DVOL is not handicapped. Timestamps are integer seconds aligned to `hour_ts`, with zero NaN on 17:00 test anchors. The DVOL "volatility" column's definition (close vs candle average) is not in the repo, so the boundary timing is **unverified**, but the script's use is causal under the stated convention.
- **Adaptive replica.** The settled-episode rule (`fresh_test.py:150-156`) matches `adaptive.py:_settled_anchors` (`last = a - H`). Episode windows end at or before a-2.
- **Monthly refit and training embargo.** The refit mask `E.dt + 19h <= month start` (`fresh_test.py:218`) and the train embargo (`:187`) are correct.
- **Targets.** RV covers rows a..a+18 (`:123-126`). The direction target uses `close[a+18]` vs `close[a-1]` (`:330`).
- **QLIKE and scoring.** `qlike` (`:60-62`) is r - log r - 1 with r = RV^2/sigma^2. The QLIKE-optimal scale sqrt(mean(RV^2/sig^2)) (`:200-202`) is correct. The paired sign (`:263`) is positive when NOCTUA is better, and the JSON agrees (raw median vs HAR +12.6%). The circular block bootstrap (`:65-74`) and Wilson interval (`:77-84`) are correct.
- **Barrier logic.** Reference `close[a-1]`, window highs/lows over a..a+18 (`:288-290`). Breach is `M >= |u|` with M >= 0 (`:317-325`). Gaussian reflection `u = -sigma * Phi^-1(alpha/2)` gives P(touch) = alpha (`:316`). `NoctuaV2.safe_level` and `touch_prob` (`serve/runtime.py:406-441`) are consistent with this.
- **Units.** `sigma_med` and `sigma_mean` are in 19h-RV units (`serve/runtime.py:196-197`). The EWMA `sqrt(ew * H)` is correct.
- **Trader arithmetic.** Funding sign (`backtest.py:30`) charges longs when funding is positive. Fees apply to |delta pos| (`:30`). `tgt_fund` = sum of fund[a..a+23] (`dataset.py:175`). Features are causal (`dataset.py:133-160`). The NumPy trader was not re-checked against the torch model.
- **Artifact pinning.** `fresh_test.py:133` loads `serve/noctua_v2.npz`, not the refreshed artifact. The training cutoff is taken from the `noctua/splits.py` defaults (CALIB_END 2024-07-01); the fit log was not checked (**unverified**).
- **Not re-run:** `features.audit_lookahead` and the torch parity tests.
