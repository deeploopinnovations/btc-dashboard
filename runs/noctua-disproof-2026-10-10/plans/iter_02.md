# Iteration 2 plan: after A1 disproved the headline vol edge

Execution feedback from iteration 1: A1 showed the HAR baselines in fresh_test.py
were calendar-naive. With 6 weekday dummies (fit before 2024-07 only) a HAR beats
the shipped raw NOCTUA median in every test half-year. A1 also found
model/noctua/features.py:296 uses `+ 4` where Monday=0 needs `+ 3`, so
cal_weekend_frac flags Fri+Sat instead of Sat+Sun.

Change of approach (not a retry of the same test):
1. Reproduce A1's reversal with my own code (independent of audits/scratch_A1).
2. Encompassing test: does NOCTUA add information ON TOP of the weekday HAR?
   Regress log RV on weekday HAR + log NOCTUA, fit before 2024-07, score 2024-07+.
3. Does a pre-test per-weekday recalibration of NOCTUA's own output close the gap?
4. Send an auditor to disprove whichever of these looks promising.
