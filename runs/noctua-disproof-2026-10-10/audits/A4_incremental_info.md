VERDICT: WEAKENED

The claim reproduces exactly, and the NOCTUA increment stays positive with a CI above zero under every baseline I could fit using only pre-2024-07 data. It is not disproved. But the robustness the claim implies is overstated: with its own baseline the CI crosses zero in 3 of 6 fit windows, the lower bound of 0.001 does not survive a multiplicity correction, half-year CIs cross zero in 3 of 5 halves for the claimed spec, the 2023-window gain is almost entirely driven by the tails of the daily distribution, and the sign of the NOCTUA coefficient flips depending on the baseline. What survives is a small (about 0.01 to 0.02 QLIKE) increment that is specific to same-day NOCTUA (placebos give about zero). It is not a clean, stable, large increment.

Scope and conventions: 831 test anchors (17:00 UTC, 2024-07-01 to 2026-10-09). QLIKE per the claim (r - log r - 1 with r = RV^2/sigma^2). Gain = QLIKE(without NOCTUA) - QLIKE(with NOCTUA), so positive means NOCTUA helps. Block bootstrap, 20-day blocks, 4000 reps, seed 0, unless noted. OLS on log RV, QLIKE-optimal scale sqrt(mean(RV^2/sigma^2)) fit on the same pre-2024-06-30 window. Every fit uses dt < 2024-06-30. Code and outputs: audits/scratch_A4/ (common.py, replicate.py, check_align.py, attack12.py, attack1b.py, attack34.py, lookahead.py, power_dvol.py, placebo.py, and the *_out.json files).

Reproduction of the claim (replicate.py)
- Fit from 2018-01-01: 2124 train anchors. QLIKE base 0.2264, with NOCTUA 0.2081, gain +0.0183, CI [+0.0042, +0.0327], P(gain<=0)=0.0067. Matches the results file.
- Fit from 2023-01-01: 546 train anchors. QLIKE base 0.1947, with NOCTUA 0.1847, gain +0.0100, CI [+0.0011, +0.0196], P=0.0135. Matches.
- Log NOCTUA coefficient: +0.19 (2018 fit), +0.27 (2023 fit).

Attack 1: richer, still pre-2024-07-fitted baseline
(a) Naive union (all added features at once: semivariances, jump share, Parkinson 1d/5d, dow x lv24 interactions, DVOL where available). Not a fair test. Its test QLIKE without NOCTUA (0.2269, 2018 fit) is worse than the claim's own baseline (0.2264). Gain +0.0274, CI [+0.0175, +0.0369], but the NOCTUA coefficient turns negative (-0.21). This is a misspecified baseline being corrected, not evidence of added information.
(b) Selected baselines. Greedy forward selection over feature groups (semivariance, jump, Parkinson, dow x lv24, dow dummies, DVOL) using expanding yearly inner folds 2020 to 2024H1, all with dt < 2024-06-30. A group is added only if inner QLIKE improves by more than 0.0005. The test window was not used for selection, but I did look at the naive-union test result (a) before designing this, which is a mild form of peeking.

Fit from       chosen groups                           base QLIKE (claim base)  +NOCTUA  gain   CI95              coef lnoct
2018-01-01     dow x lv24, jump, dow, park             0.2183 (0.2264)          0.1987   +0.0196  [+0.0068,+0.0327]  -0.067
2021-01-01     dow x lv24, jump, dow, park             0.1959 (0.2019)          0.1849   +0.0110  [+0.0010,+0.0213]  -0.173
2022-01-01     dow x lv24, jump, dow                   0.1980 (0.1937)          0.1833   +0.0147  [+0.0040,+0.0255]  -0.146
2022-07-01     DVOL, dow x lv24, dow, park             0.1803 (0.1928)          0.1678   +0.0125  [+0.0016,+0.0248]  -0.145
2023-01-01     DVOL, dow x lv24, dow, park             0.1856 (0.1947)          0.1718   +0.0138  [+0.0047,+0.0243]  -0.243
2023-07-01     reused 2023-01 set (too few rows for inner folds)  0.1982 (0.1980)  0.1800   +0.0182  [+0.0087,+0.0291]  -0.495

- The increment survives in all six windows with CI above zero.
- The selected baseline is not uniformly stronger than the claim's baseline: it is worse in the 2022-01 window (0.1980 vs 0.1937) and tied in 2023-07.
- DVOL matters: a naive union with DVOL (fit from 2021-03-24, the first finite DVOL) gives base 0.1885, gain +0.0069, CI [-0.0038, +0.0184], which crosses zero.
- The NOCTUA coefficient is negative (-0.07 to -0.50) in every selected-baseline fit, versus positive (+0.19, +0.27) in the claim's baseline. The sign depends on the baseline, so NOCTUA is not reliably a positive second volatility signal.

Attack 2: fit-window sensitivity (claim baseline, lv HAR + 6 weekday dummies)
- 2018-01-01: gain +0.0183, CI [+0.0042, +0.0327]. Above zero.
- 2021-01-01: gain +0.0107, CI [-0.0013, +0.0227]. Crosses zero.
- 2022-01-01: gain +0.0084, CI [-0.0028, +0.0200]. Crosses zero.
- 2022-07-01: gain +0.0087, CI [-0.0024, +0.0205]. Crosses zero.
- 2023-01-01: gain +0.0100, CI [+0.0011, +0.0196]. Above zero.
- 2023-07-01: gain +0.0150, CI [+0.0054, +0.0252]. Above zero.
Result: CI above zero in 3 of 6 windows for the claimed baseline. With the selected baselines (Attack 1b), 6 of 6.

Attack 3: half-year stability (gain, CI95, 2000 reps)
Claimed spec, 2018 fit (base = lv HAR + dow):
- 2024H2 (n=184): +0.0028 [-0.0249, +0.0323]
- 2025H1 (n=181): +0.0170 [-0.0208, +0.0620]
- 2025H2 (n=184): +0.0144 [-0.0109, +0.0378]
- 2026H1 (n=181): +0.0410 [+0.0154, +0.0658]
- 2026 Jul to Oct 9 (n=101): +0.0153 [+0.0011, +0.0315]
Claimed spec, 2023 fit:
- 2024H2: +0.0027 [-0.0102, +0.0191]
- 2025H1: +0.0124 [-0.0068, +0.0344]
- 2025H2: +0.0083 [-0.0100, +0.0317]
- 2026H1: +0.0226 [+0.0020, +0.0450]
- 2026 Jul to Oct 9: -0.0005 [-0.0138, +0.0147]
Selected baseline, 2018 fit: all five half-year point estimates positive (+0.0078, +0.0274, +0.0181, +0.0304, +0.0104); CI above zero only in 2026H1 [+0.0094, +0.0523].
Selected baseline, 2023 fit: +0.0056, +0.0137, +0.0106, +0.0223, +0.0192; CI above zero in 2025H1 and 2026H1 only (the 2026H1 lower bound is -0.0010).
Tail dependence (mean of daily gains, untrimmed vs trimmed 5% each tail):
- Claimed spec, 2018 fit: 0.0183 vs 0.0112
- Claimed spec, 2023 fit: 0.0100 vs 0.0011
- Selected, 2018 fit: 0.0196 vs 0.0175
- Selected, 2023 fit: 0.0138 vs -0.0006
Share of days with positive gain: 0.53, 0.48, 0.61, 0.44 respectively. The median daily gain for the 2023 claimed spec is -0.001; the increment comes from a minority of large-error days.
Result: the sign is mostly consistent across halves, but the CIs cross zero in most halves, and the 2023-window increment is driven by the tails.

Attack 4: multiplicity
- Gain tests examined in this audit: 6 windows x 3 baselines (18) + 6 selected-baseline windows = K = 24. This undercounts the run, since A1 and the earlier results file tested other baselines too, and the tests are correlated, so Bonferroni is conservative. Treat the numbers as a bound, not a verdict.
- Claimed 2018 fit: gain +0.0183, 95% CI [+0.0043, +0.0325], one-sided P=0.0053, Bonferroni p=0.127, CI at 1-0.05/24 level [-0.0034, +0.0413].
- Claimed 2023 fit: gain +0.0100, 95% CI [+0.0014, +0.0194], one-sided P=0.0117, Bonferroni p=0.280, CI at adjusted level [-0.0032, +0.0246].
- Selected 2023 fit (the strongest of the selected): gain +0.0138, CI [+0.0047, +0.0243], Bonferroni p=0.013, CI at adjusted level [+0.0006, +0.0306].
- Block-length sensitivity, claimed 2023 fit (20k reps): block 10 CI [+0.0017, +0.0187]; block 20 [+0.0013, +0.0192]; block 40 [+0.0003, +0.0198]. The 0.001 lower bound depends on block length.
Result: the 0.001 lower bound is not meaningful. It is the least-robust lower bound of the windows tried, and it is not robust to the multiplicity of this run.

Attack 5: lookahead and placebos
- Feature invariance (lookahead.py): for six anchors (2019-03-04, 2022-06-01, 2024-07-01, 2025-05-13, 2026-03-02, 2026-09-30), I perturbed all OHLCV, rv5, rv5_pos/neg, bpv5 and rq5 bars at or after a+1 (and separately from a) with lognormal noise. NOCTUA's features changed by exactly 0 and sigma_med was identical to every printed digit in all 12 trials.
- Power check (power_dvol.py): changing the past bar a-1 (rv5 x50) moved sigma_med from 0.02044 to 0.03280. The null above is therefore not vacuous.
- Lag columns (check_align.py): lv1, lv6, lv24, lv120, lv528 match an independent recomputation with trailing sums exclusive of the anchor bar, with max abs difference 0.0 over all 2956 anchors. Including the anchor bar would give a 2.90 mismatch, so the check has power. Row-to-hour alignment is exact, and RV recomputes exactly.
- DVOL (power_dvol.py): ldvol equals log(DVOL stamped a-1) exactly over 400 anchors.
- No fit used dt >= 2024-06-30.
- Placebos (placebo.py), same pipeline, gain and CI95:
  Claimed spec, 2018 fit: NOCTUA +0.0183; N(0,1) noise +0.0005 [-0.0008, +0.0018]; NOCTUA lagged 7 days +0.0001 [-0.0001, +0.0002].
  Claimed spec, 2023 fit: NOCTUA +0.0100; noise +0.0002 [-0.0004, +0.0009]; lag 7 +0.0002 [-0.0006, +0.0008].
  Selected, 2018 fit: NOCTUA +0.0196; noise +0.0002 [-0.0008, +0.0012]; lag 7 +0.0002 [-0.0000, +0.0004].
  Selected, 2023 fit: NOCTUA +0.0138; noise +0.0001 [-0.0012, +0.0015]; lag 7 -0.0001 [-0.0007, +0.0005].
  The same-day NOCTUA increment is specific: a random regressor or a stale NOCTUA forecast adds about zero.
- In-sample caveat: NOCTUA's QLIKE with a scale fit before 2024-07 is 0.3049 in-sample (2018 to 2024H1) and 0.2589 out-of-sample. NOCTUA is not flattered by its training window on this measure. Its training cutoff is in the run notes, not in the artifact metadata, so I could not verify it directly.
- No lookahead found.

Other notes
- The premise that NOCTUA's 42 features include DVOL is not supported: `grep -i dvol model/noctua/features.py` finds nothing. DVOL is therefore a genuinely external baseline input, and the naive union with DVOL shows it absorbs much of the increment (gain +0.0069, CI crosses zero).
- DVOL starts 2021-03-24, so any DVOL specification effectively fits from that date regardless of the nominal start.

Answers to the attacks
1. The increment survives a richer, pre-2024-07-fitted baseline on the selected baselines (gain +0.011 to +0.020, all CI lower bounds above zero), but not on the naive union with DVOL (gain +0.007, CI crosses zero).
2. Not in all windows for the claimed baseline (3 of 6 cross zero). Yes in all six windows once the baseline is selected on pre-2024-07 data.
3. Partly. All half-year point estimates are positive for the claimed 2018 spec and the selected 2018 spec, but CIs cross zero in most halves, and the 2023-window increment is concentrated in the tails (trimmed mean about 0.001).
4. No. After a Bonferroni-style correction over 24 tests, the claimed lower bounds fall below zero (-0.0034 and -0.0032). Only the selected 2023 spec keeps a positive adjusted lower bound (+0.0006), and that bound is marginal.
5. No lookahead found.

Limits
- The selected baseline is itself a chosen model. Its selection used only pre-2024-07 data, but it was designed after I saw the naive-union test result.
- The NOCTUA coefficient sign flip is unexplained. I have not established whether NOCTUA contributes a mean-reversion correction to the baseline's level or shape, or a separate volatility signal.
- Bootstrap CIs use 20-day blocks on overlapping-in-time but non-overlapping-in-target daily episodes. Block length matters (see Attack 4).
- The test era has been scored before in this run, so these are not fresh out-of-sample results. No parameter was tuned on the test window in this audit beyond the design choices noted above.
