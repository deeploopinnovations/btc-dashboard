VERDICT: DISPROVED

The three named comparisons reproduce exactly and their CIs exclude zero against calendar-naive HAR baselines. The general claim does not survive: a HAR5 baseline that gets the weekday information NOCTUA itself uses, fitted only on pre-2024-07 data, beats the shipped raw median by about 22% of NOCTUA's mean QLIKE, with the CI on the other side of zero and the same sign in all five test half-years. The "edge" is a weekday miscalibration in NOCTUA's raw median, not a volatility-forecasting advantage.

## Setup
- Test: 831 anchors at 17:00 UTC, 2024-07-01 to 2026-10-09. The 2026-10-09 anchor is included; the claim states an end of 2026-10-08.
- Loss: QLIKE, claim convention. Edge = mean(QLIKE_base - QLIKE_NOCTUA) / QLIKE_base, so positive means NOCTUA wins. Block bootstrap, 20-day blocks, 4000 reps.
- Every fit uses anchors with dt < 2024-06-30 05:00 (the 19h window ends before the test starts). Recent windows are the two the brief named, 2022-07..2024-06 and 2023-01..2024-06. No test-era fitting.

## Reproduction (evals/fresh_test.json, test_all, via fresh_test.py)
| comparison | QLIKE base | NOCTUA raw_med QLIKE 0.2431 edge | 95% CI |
|---|---|---|---|
| frozen log-HAR 24/120/528h | 0.2859 | +14.95% | [0.0337, 0.0514] |
| frozen HAR5 (1h/6h terms) | 0.2780 | +12.56% | [0.0195, 0.0490] |
| expanding monthly-refit HAR5 | 0.2742 | +11.32% | [0.0158, 0.0446] |
| HAR5 + DVOL | 0.2524 | +3.68% | [-0.0053, 0.0258] |

All four match the claim. Code: /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/fresh_test.py, output evals/fresh_test.json.

## Attack 1: baseline handicap (refit, QLIKE fit, calendar terms)
Code: audits/scratch_A1/attacks.py (Attack 1), audits/scratch_A1/ablate.py. Negative edge = baseline beats NOCTUA raw.

| baseline | fit window | fit method | base QLIKE | edge | 95% CI |
|---|---|---|---|---|---|
| HAR5 | 2022H2-2024H1 | OLS + QLIKE scale | 0.2627 | +7.47% | [0.0050, 0.0341] |
| HAR5 | 2023H1-2024H1 | OLS + QLIKE scale | 0.2644 | +8.06% | [0.0068, 0.0353] |
| HAR5 | 2022H2-2024H1 | minimise QLIKE | 0.2604 | +6.63% | [0.0034, 0.0305] |
| HAR5 | 2023H1-2024H1 | minimise QLIKE | 0.2610 | +6.86% | [0.0048, 0.0306] |
| HAR5 | 2018-2024H1 | minimise QLIKE | 0.2953 | +17.66% | [0.0364, 0.0665] |
| HAR5 + weekend share | 2023H1-2024H1 | minimise QLIKE | 0.2413 | -0.76% | [-0.0157, 0.0131] |
| HAR5 + weekday dummies | 2018-2024H1 | minimise QLIKE | 0.2462 | +1.24% | [-0.0123, 0.0170] |
| HAR5 + weekday dummies | 2022H2-2024H1 | minimise QLIKE | 0.1894 | -28.35% | [-0.0714, -0.0363] |
| HAR5 + weekday dummies | 2023H1-2024H1 | minimise QLIKE | 0.1889 | -28.73% | [-0.0724, -0.0375] |

Reading:
- Refitting on a recent window roughly halves the gap (12.6% to about 7%), and it stays significant.
- Adding weekday dummies reverses it. At a fixed 17:00 anchor, the forward weekend share is a deterministic function of the anchor weekday, so the weekend-share term and the weekday dummies carry the same information. NOCTUA uses weekday information (cal_weekend_frac and cal_dow_sin/cos), so calendar-naive baselines are handicapped.
- The reversal depends on the fit window. The full-history calendar HAR ties (+1.2%, CI crosses zero). Both recent windows reverse.

Half-year stability, HAR5 + weekday dummies fitted on 2023H1-2024H1 (stab.py):
| period | n | NOCTUA QLIKE | baseline QLIKE | baseline lower by (% of NOCTUA) | CI of per-day (NOCTUA - base), QLIKE units |
|---|---|---|---|---|---|
| 2024H2 | 184 | 0.2389 | 0.1841 | 22.9% | [0.0102, 0.1089] |
| 2025H1 | 181 | 0.2692 | 0.2089 | 22.4% | [0.0163, 0.1057] |
| 2025H2 | 184 | 0.2444 | 0.2008 | 17.8% | [0.0072, 0.0768] |
| 2026H1 | 181 | 0.2267 | 0.1644 | 27.5% | [0.0354, 0.0893] |
| 2026Q3 | 101 | 0.2312 | 0.1836 | 20.6% | [0.0252, 0.0685] |

NOCTUA's weekday residual, median log(RV / sigma_NOCTUA), test era. In-sample 2023H1-2024H1 shows the same pattern (Sat -0.37, Fri -0.25, Sun +0.14), so the misfit predates the test:

| anchor weekday | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| NOCTUA | -0.19 | -0.19 | -0.13 | -0.11 | -0.37 | -0.38 | +0.20 |
| HAR5 + weekday | -0.21 | -0.14 | -0.25 | -0.17 | -0.28 | -0.18 | -0.07 |

Single-scalar check on NOCTUA's own output, scale fitted on pre-test data only:
- Constant: QLIKE 0.2478 (about tied with HAR5 frozen).
- Per-weekday scale, fitted 2023H1-2024H1: QLIKE 0.1921. This is on par with HAR5 + weekday (0.1889) and beats raw NOCTUA by 21%.
- Per-weekday scale, fitted 2018-2024H1: QLIKE 0.2269.

## Attack 2: concentration
Code: audits/scratch_A1/attacks.py (Attack 2). Claim convention.

| baseline | edge | drop top 1% abs-diff days | drop top 5% abs-diff days | drop top 5% NOCTUA-favouring days | median per-day diff | NOCTUA better on days |
|---|---|---|---|---|---|---|
| log-HAR frozen | +14.95% | +19.5% | +23.2% | +10.8% | +0.0313 | 71.4% |
| HAR5 frozen | +12.56% | +17.9% | +21.6% | +6.5% | +0.0395 | 70.4% |
| HAR5 refit | +11.32% | +16.4% | +20.3% | +5.0% | +0.0337 | 68.4% |
| HAR5 + weekday (2023H1-2024H1) | -28.73% | -24.5% | -15.8% | -47.9% | -0.0107 | 43.8% |

Reading: the gain over calendar-naive HAR is broad across days, but the top 5% of NOCTUA-favouring days carry about half of the frozen-HAR gain (12.6% to 6.5%). The reversal against the weekday HAR is mean-driven: its median day is only about 1% in its favour, and it wins 56% of days.

## Attack 3: other losses, Mincer-Zarnowitz, constant multipliers
Code: audits/scratch_A1/attacks.py (Attack 3).

| model | QLIKE | MSE of variance | MSE of vol | MAE of log vol |
|---|---|---|---|---|
| NOCTUA raw_med | 0.2431 | 1.992e-7 | 4.788e-5 | 0.2823 |
| NOCTUA raw_mean | 0.3103 | 2.519e-7 | 7.617e-5 | 0.3952 |
| HAR5 frozen | 0.2780 | 2.291e-7 | 6.433e-5 | 0.3387 |
| log-HAR frozen | 0.2859 | 2.278e-7 | 6.328e-5 | 0.3370 |
| HAR5 refit | 0.2742 | 2.267e-7 | 6.286e-5 | 0.3331 |
| HAR5 + weekday (2023H1-2024H1) | 0.1889 | 1.799e-7 | 4.113e-5 | 0.2574 |

- NOCTUA beats the calendar-naive HARs on all four losses. The weekday HAR beats NOCTUA on all four.
- Mincer-Zarnowitz on RV^2 (test, block-bootstrap CIs): NOCTUA raw_med slope 0.854 [0.693, 1.046], so slope 1 is inside the CI, R^2 0.274, log-vol slope 0.967. HAR5 frozen slope 0.698 [0.560, 0.843], so slope 1 is excluded (under-responsive). HAR5 + weekday slope 0.986 [0.814, 1.161], R^2 0.340, log-vol slope 1.088. NOCTUA's calibration is acceptable, but the weekday HAR is at least as well calibrated and more informative.
- Constant multipliers fitted on pre-2024-07 data, applied to HAR5 frozen: QLIKE-optimal c=1.000 gives 0.2780; MSE-of-variance-optimal c=0.907 gives 0.2598; MAE-log-optimal c=0.814 gives 0.2846. None reaches NOCTUA's 0.2431, so NOCTUA's advantage over calendar-naive HAR is not a plain level effect. The HAR5 refit column is NaN before the test, so its constant check was not run.

## Attack 4: lookahead
Code: audits/scratch_A1/lookahead.py, output lookahead_out.json.
- Target RV, recomputed as the sum of rv5[a..a+18]: max difference 0.0.
- lv1, lv6, lv24, lv120, lv528, recomputed from rv5[a-k..a-1]: max differences up to 7e-10 (floating point).
- ldvol, DVOL at hour a-1: difference 0.0 across 2026 rows; the NaN pattern matches.
- NOCTUA raw_med replay on 6 random test anchors: replay matches the stored value to 3e-18. Corrupting every bar at or after the anchor (OHLC, volume, rv5, bpv5, rq5) leaves the forecast unchanged (difference 0.0).
- Not checked: the full audit_lookahead() run and the served factor (not used by the raw claim).
- Not lookahead, but a real defect: features.py fut_dow uses +4 instead of +3, so the weekend share counts Friday and Saturday instead of Saturday and Sunday. Verified numerically: NOCTUA's weekend share is 0.63 for Thu anchors (true 0), 1.00 for Fri (true 0.63), 0.37 for Sat (true 1.00), and 0.00 for Sun (true 0.37). The defect is documented as P4-weekend-bug and is baked into the trained artifact. Also, cal_dow_sin/cos encodes seven weekdays in two numbers, which cannot represent a seven-level pattern.
- Splice at 2026-08-15 (splice.py): no level break in rv5 (mean 1.91e-5 before, 1.62e-5 after). Volume medians differ (36 vs 49 BTC/h), but there is no visible discontinuity.

## Limits
- The reversal depends on using 18 to 24 months of pre-test data. Full-history calendar HAR ties.
- Every number above is a diagnostic I computed; none is a new model choice made on test data. The two recent windows and the weekday specification are the ones the brief named or are the minimal calendar set.
- The shipped product is not changed by any of this. The per-weekday rescale of NOCTUA is a diagnostic only.

## Conclusion
The claim's three numbers are correct. They measure NOCTUA against baselines that omit the weekday information NOCTUA is supposed to use. Against a recent-window HAR with weekday dummies, fitted on pre-2024-07 data, NOCTUA's raw median has a mean QLIKE about 22% higher, with the CI excluding zero and the same sign in all five test half-years. NOCTUA's weekend and weekday handling is miscalibrated in-sample as well as out of sample. The defensible statement is that NOCTUA beats calendar-naive HAR and is miscalibrated by weekday, not that it has a volatility edge over HAR.

## Files
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/fresh_test.py
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/evals/fresh_test.json
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A1/attacks.py (attacks_out.json)
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A1/ablate.py (ablate_out.json)
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A1/stab.py (stab_out.json)
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A1/calendar_diag.py
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A1/lookahead.py (lookahead_out.json)
- /home/user/btc-dashboard/runs/noctua-disproof-2026-10-10/audits/scratch_A1/splice.py
- /home/user/btc-dashboard/model/noctua/features.py (cal_weekend_frac defect, lines using fut_dow)
