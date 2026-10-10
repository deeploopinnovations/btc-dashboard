# A2 adversarial audit: F8 (served volatility) and F5 (barrier tails)

VERDICT F8: DISPROVED as worded ("the published volatility is worse than the raw model"). The narrower 17:00-only claims survive: served loses to raw median (-23%) and to frozen HAR5 (-7.8%, CI excludes 0) at the 17:00 anchor.
VERDICT F5: WEAKENED. "Barrier levels are conservative" holds at 17:00 for alpha 5%, 10%, 20% on both sides and for the 1% up side, but the quoted 5%/10% up-side rates do not reproduce from evals/fresh_test.json, the 1% down-side rate is not significantly below nominal, and the conservatism is hour-dependent.

## Replica check (attack a): PASSED, the replica is faithful
- Script: audits/scratch_A2/verify_replica.py. Real `serve.predict.forecast(model, hours, anchor_ts=..., raw={})` at 20 test-era 17:00 anchors (2024-07-01..2026-10-09), model = `serve.runtime.load_model('model/serve/noctua_v2.npz')`.
- Correction factor `raw['cal']['factor']` vs parquet `factor`: max abs diff 1.9e-12. All 240 settled episodes used.
- Published `sigma_window_pct` vs parquet `noctua_served_mean`: max rel diff 2.5e-4 (rounding to 3 dp in percent). `raw['pred']['sigma_mean']` already carries the factor; dividing it out matches parquet `noctua_raw_mean` to the same 2.5e-4.
- Off-hour spot check (hours 3, 11, 22; 12 anchors, audits/scratch_A2/verify_offhour.py): factor error <= 3.2e-12, served rel err <= 2.5e-4. The batch replica (hour_sweep.py) therefore holds at all hours.
- Caveat on the replica itself: it uses `build_features` without the `ok` finite-feature filter that `volatility_correction` applies. Nothing in the 20+12 checks shows that filter ever binding.

## Production schedule (attack b): the 17:00 anchor is NOT what the workflow publishes
- `.github/workflows/fetch-data.yml` runs `python model/serve/predict.py --out-dir data` every 30 min with no `--anchor`. `forecast()` then defaults to `anchor_ts = hour_ts[-1]`, the last closed hour, so the published hour cycles through all 24 hours.
- `_next_anchor()` in predict.py (the 17:00 helper) is defined but never called.
- The committed snapshot `data/noctua.json` has `anchor_utc = 2026-10-10 11:00:00+00:00`, not 17:00.
- The README describes a 17:00 product window, so the 17:00 analysis is a product-design view, not the live number.

## F8 numbers
Test era 2024-07-01..2026-10-08, 831 anchors per hour. QLIKE (lower is better):

| anchor | served mean (published) | raw median | raw mean | frozen HAR5 |
|---|---|---|---|---|
| 17:00 | 0.2996 | 0.2431 | 0.3103 | 0.2780 |
| all 24 hours pooled | 0.2281 | 0.2511 | 0.2324 | n/a |

- 17:00, served vs raw mean: served better by 3.45% (paired block bootstrap CI on QLIKE gain [0.0044, 0.0170], p(gain<=0)=0.0003).
- 17:00, served vs raw median: served worse by 23.2% (CI [-0.0765, -0.0354]).
- 17:00, served vs HAR5 frozen: served worse by 7.75% (CI [-0.0396, -0.0029], p=0.989). Vs HAR5 refit: worse by 9.3%.
- All 24 hours pooled: served beats raw median by 9.2% (CI [0.0065, 0.0406]) and raw mean by 1.8% (CI [0.0024, 0.0062]).
- Served QLIKE is below raw-mean QLIKE at all 24 anchor hours. Served is worse than raw median only at about 15-20 UTC; the worst ratio is at 17 UTC (0.2996 vs 0.2431, ratio 1.23).
- Sub-periods at 17:00, served / raw median / HAR5: 2024H2 0.267/0.239/0.255; 2025H1 0.303/0.269/0.317; 2025H2 0.324/0.244/0.264; 2026H1 0.275/0.227/0.274; 2026-07+ 0.353/0.231/0.282. Served loses to HAR5 in 2024H2, 2025H2 and 2026-07+, ties in 2026H1 (0.275 vs 0.274), and wins in 2025H1.

Mechanism, and why the finding's diagnostic is misleading:
- The finding's "median RV/forecast 0.85" is not evidence of bias for a mean-variance forecast. A QLIKE-optimal mean forecast has a median RV/sigma below 1 by construction.
- The right diagnostic is the QLIKE-optimal scalar sqrt(mean(RV^2/sigma^2)) at 17:00: raw median 0.998 (already calibrated), raw mean 0.803, served 0.815. The mean functional is about 20% high at 17:00, and the median-based factor (0.986) cannot see that. This part of the finding survives.
- The served correction is median-based but multiplies the mean. That mismatch is real, but it is a design issue, not an unfair comparison.

Fairness points:
- "Worse than the raw model" depends on which raw functional is the baseline. The product reports the mean (REPORT_FUNCTIONAL = "mean"), so raw mean is the apples-to-apples baseline, and the correction improves on it at every hour.
- The HAR5 comparator is calibrated on training data, so it is fair.

## F5 numbers
Independent recount (audits/scratch_A2/tails_check.py) using the `safe_levels` from a real `forecast()` call at each of 831 17:00 anchors (audits/scratch_A2/tails_payload.py). Excursions come from the hourly high/low over bars a..a+18 vs close of bar a-1, and the spot matches the payload exactly.

| alpha | up breaches (rate, Wilson 95%) | down breaches (rate, Wilson 95%) | Gaussian HAR5 up / down |
|---|---|---|---|
| 1% | 1 (0.12%, [0.02, 0.68]) | 5 (0.60%, [0.26, 1.40]) | 1.56% / 2.53% |
| 2% | 5 (0.60%) | 8 (0.96%) | 2.17% / 3.25% |
| 5% | 16 (1.93%, [1.19, 3.11]) | 26 (3.13%, [2.14, 4.55]) | 3.85% / 5.30% |
| 10% | 42 (5.05%, [3.76, 6.76]) | 62 (7.46%, [5.86, 9.45]) | 6.02% / 8.79% |
| 20% | 98 (11.8%) | 115 (13.8%) | 11.2% / 13.8% |

- Reproduction: fresh_test.json `tails` gives 1%: 0.12/0.60 (match), 5%: 1.925/3.129 (match), and 10%: 5.054/7.461. The finding's 5% up (2.05%) and 10% up (4.8%) do not match the file, and 10% down is 7.46 not 7.3. The finding's numbers look like 17 and 40 breaches, not the 16 and 42 in the file. This is a reproducibility error in the quoted numbers.
- The 1% down rate (0.60%, CI up to 1.40%) is consistent with nominal 1%.
- Half-year sub-periods at 10%: up 4.4-7.1%, down 6.5-9.8%. The down side is near nominal in 2024H2 (9.78%) and 2025H1 (9.39%), though each sub-period has about 180 days and a standard error near 2pp.
- Hour dependence (audits/scratch_A2/tails_by_hour.py, 30 test-era anchors per hour, batch replica): the 10% up side shows 0-3% breaches at 13-19 UTC but 10-13% at 0-9 UTC. Pooled over 24 hours: 5% up 3.2%, 5% down 4.5%, 10% up 6.8%, 10% down 8.1%. The conservatism is therefore much weaker, and absent in some hours, once hours are pooled. The per-hour sample is small (SE about 5pp).
- Scaling the 17:00 log-levels by 0.85 brings 10% breaches to 7.9% up and 10.2% down, i.e. close to nominal. The conservatism looks like the same mean-functional sigma over-forecast as F8, not an independent barrier-construction defect.
- Caveat on the measurement itself: hourly high/low understates the continuous-path maximum, which would bias breach rates downward. At 1-minute-derived bars the effect is small, and it does not change the sign of the finding.

## Files
- audits/scratch_A2/verify_replica.py, verify_offhour.py, replica_check.csv
- audits/scratch_A2/hour_sweep.py, hour_sweep_test.parquet, hour_sweep_summary.csv, boot_f8.py
- audits/scratch_A2/tails_payload.py, tails_payload_17utc.csv, tails_check.py, tails_by_hour.py, tails_by_hour_records.parquet
