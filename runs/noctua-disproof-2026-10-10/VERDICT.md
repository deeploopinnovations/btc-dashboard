# NOCTUA: fresh test, disproof audit and production readiness (2026-10-10)

**Readiness verdict: NOT READY for production decisions.** It is fine as a
research dashboard. Do not size real positions or sell real options on it yet.

Every number below survived, or was cut down by, an auditor whose only job was
to disprove it. Auditor verdicts are verbatim in `audits/`.

## 1. How good are the predictions?

Test: the pinned artifact `model/serve/noctua_v2.npz` (fit before 2024-07),
831 daily windows of 19 h opened at 17:00 UTC, 2024-07-01 to 2026-10-09.
Price history was extended to 2026-10-10 14:00 UTC and checked against Binance
(max hourly close gap 0.16 %, A3). QLIKE, lower is better.

| forecast | QLIKE | vs NOCTUA raw median |
|---|---|---|
| NOCTUA raw median (what the trader uses) | 0.243 | |
| HAR with weekday terms, fit 2023-01..2024-06 | **0.195** | HAR better by 20 % |
| HAR with weekday terms, fit 2018..2024-06 | 0.226 | HAR better by 7 % |
| HAR5 + DVOL, frozen | 0.252 | tie (CI crosses 0) |
| HAR5, monthly refit | 0.274 | NOCTUA better by 11 % |
| Log-HAR, frozen | 0.286 | NOCTUA better by 15 % |
| Deribit DVOL alone | 0.333 | NOCTUA better by 27 % |

- **Volatility.** NOCTUA beats the calendar-blind baselines the project has
  always compared against, but a 12-parameter HAR with weekday dummies beats it
  in every half-year (A1, reproduced independently). The "beats Log-HAR" headline
  is real but against the wrong baseline. **Status: disproved as a general claim.**
- **Does NOCTUA add anything?** Adding NOCTUA to the weekday HAR improves QLIKE
  by 0.01 to 0.02, and placebos give zero, so there is real information. But the
  CI crosses zero in 3 of 6 fit windows, it is driven by a few large days and it
  does not survive a multiple-testing correction (A4). **Status: weakened.**
- **Why it loses:** NOCTUA is miscalibrated by weekday (median RV/forecast 0.69
  on Fri and Sat anchors, 1.22 on Sun). One cause is a bug at
  `model/noctua/features.py:296`: `+ 4` should be `+ 3`, so `cal_weekend_frac`
  flags Friday and Saturday instead of Saturday and Sunday (A0, A1, verified).
- **Tails.** At 17:00 the barrier levels are conservative: at the 1 % level the
  up side broke 1 time in 831 and the down side 5 times. That is safe for a
  seller but means strikes are quoted too far out. Pooled over all hours the
  conservatism mostly disappears (A2). **Status: conservative, not calibrated.**
- **Direction.** No skill, again: correlation of prob_up with the outcome 0.02,
  Brier 0.2495 vs 0.25 for a coin flip. **Status: confirmed negative.**

## 2. The live forecast

- The dashboard workflow publishes every 30 minutes at the last closed hour,
  not at 17:00 (A2). Pooled over all hours, the published number (sigma_mean
  times the trailing correction) beats the raw model by 9 %.
- In the 17:00 window the README sells to option sellers, the published number
  is worse than a frozen HAR by 7.8 % (CI excludes 0) and worse than NOCTUA's own
  raw median by 23 %. Cause: the correction factor is a median over every hour
  (about 0.98) and cannot see that the mean forecast runs about 20 % high at
  17:00 (A0 B1, A2).
- The correction window and the choice to publish the mean were picked while
  looking at 2024-07+ data, so they are not out-of-sample for design (A0 B2).

## 3. The trader (NOCTUA-Trader v1)

Re-scored 2024-07-01..2026-10-08 (830 days) with the extended data:
Sharpe 0.44 vs buy-and-hold 0.37, CAGR 2.5 % vs 7.1 % (compounded),
average position 0.14. Since 2026-08-15: trader +4.1 %, buy-and-hold +30.4 %.

- A constant 14 % long has the same Sharpe as buy-and-hold, so the edge to
  explain is 0.07 Sharpe, CI [-0.19, +0.31]. It vanishes at 12 bp fees and flips
  sign if 2026 Q1 is left out. The -7.6 % vs -54 % drawdown is an exposure
  difference, not skill (A3). **Status: weakened; no evidence of timing skill.**

## 4. What would have to be true before production

1. Fix the weekday bug and add weekday terms, retrain on data before 2024-07
   only, and pre-register the comparison against the weekday HAR.
2. Compute the served correction on the served hour (or publish the median),
   and test it on the 17:00 window before switching.
3. Pass a **forward holdout**: the frozen model must beat the weekday HAR on
   days after 2026-10-10 that nobody has seen. The 2024-07+ era is used up;
   more analysis on it only adds multiple testing (supervisor stop, see below).
   The per-day QLIKE difference has a standard deviation of about 0.22, so
   detecting a 0.02 edge at two standard errors needs about 470 daily windows
   (1.3 years) and a 0.01 edge about 1,900 (5 years), before allowing for
   autocorrelation. Scoring every hourly anchor (with a block bootstrap) would
   shorten that.
4. For the trader: beat a matched-exposure constant on the forward log, net of
   realistic fees and slippage, before any real order. No exchange keys until then.

## Loop record

- Iteration 1: fresh test (plans/iter_01.md, evals/fresh_test.json); auditors
  A0 (code), A1 (vol edge), A2 (live forecast and tails), A3 (trader and data).
- Iteration 2: after A1's disproof, changed approach to an encompassing test
  (plans/iter_02.md, evals/iter2_encompassing.json); auditor A4.
- Iteration 3, supervisor: stopped the vol-edge path. Each extra baseline on the
  already-scored era added multiplicity without clean evidence; redirected to a
  forward holdout. Details in supervisor.json and trajectory.jsonl.
- Nothing was deployed, no orders were placed and no exchange keys were used.
