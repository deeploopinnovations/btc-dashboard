# NOCTUA research rules

Short, enforceable, and each one earned. Every rule below exists because this
project broke it at least once and the break is on the record — the ledger id or
BENCHMARK section is cited so nobody has to take the rule on faith.

The rules are ordered by how expensive the mistake was, not by topic.

---

## The one that governs the rest

> **Be ambitious about the research, but be more ambitious about proving
> yourself wrong.**

Corollaries, all of them operational:

**A result you have not attacked is not a result.** Every candidate gets a
placebo, a positive control, and at least one attempt to explain it away as an
artifact. E2c survived four (`E2c-result`, `E2c-not-an-intercept`,
`E2c-sigma-sensitivity`, the test-shuffle control) and is *still* only a
candidate.

**Spend your best effort on the hypothesis you like most.** The IV result got
more adversarial scrutiny than any rejection did, because a rejection that is
wrong costs a missed idea and an adoption that is wrong costs the product.

---

## Rules about rules

**R1. A rule fixed after seeing the number is not a rule.**
Pre-register the endpoint, the guards, and the decision before the data is
scored. Amend only *before* results exist, and record the amendment rather than
editing silently (§22's amendment; `blend_ceiling`'s ceiling amendment).

**R2. A rule that cannot fail is not a rule.**
Before trusting a guard, prove it can return the other answer. Five guards in
this project could not fail: two documented but never implemented, one comparing
an expression to itself, one deciding on a NaN, one corruption that was a no-op
on 17.8% of its input. All five printed reassuring output.
*(pitfalls 7, 9, 11, 12; `iv-correction-audit`)*

**R3. A rule only binds if it binds when it is inconvenient.**
`iv_confirm.py` printed ADOPT. A stricter commitment recorded beforehand said an
inconclusive MCB is not a pass. The stricter one won, and the wide-slice run it
forced showed the headline effect was half what production said.
*(`E2-confirm-result` → `E2-confirm-wide`)*

**R4. The rule may have parts that are not executable. Honour those too.**
Guards inside a script and commitments in the ledger are both the rule. When
they disagree, the *stricter* one governs.

---

## Rules about evidence

**R5. State the minimum detectable effect before running.**
If the expected effect is below the MDE, the experiment is a non-measurement
*before* it runs — redesign it or don't run it. Production-slice MDEs are
~1.44% on spike QLIKE and ~8.44% on pooled. *(§26)*

**R6. An estimate smaller than its own standard error is a non-measurement, not
a null.**
It would look identical if the truth were +2% or −2%. Say "not resolvable",
never "no effect". *(`E-blend-1state`: mean 0.00049 against se 0.00876)*

**R7. Every fold-level verdict here is a statement about six years, not
thousands of episodes.**
24× more episodes bought 1.53× effective sample size — 6.4% survival. Between-
year heterogeneity dominates and no amount of extra scoring reduces it. *(§26)*

**R8. More data helps least where the loss is.**
Widening tightened deep-tail MCB 2.98× and spike QLIKE only 1.24× — and spike
carries 25.8% of total loss. Check *which* quantity gains before paying for
data. *(§26, §7a)*

**R9. Failed experiments stay in the multiple-testing family.**
Never drop a test from the correction because it failed. The family is every
hypothesis tried on overlapping data, not the surviving ones.

**R10. Report the worst fold, always.**
An average that hides a 2023 is not a summary. *(§23: four folds improved, one
gave it all back, and the worst-fold guard was decisive where the primary was
ambiguous.)*

---

## Rules about data

**R11. `feature_available_time <= prediction_time`, enforced by a decoy.**
Strict as-of joins. No backward fill, no future-aware normalisation, no scaler
fitted on test. A deliberate future-leak decoy must be caught, and the catch
rate reported.

**R12. Verify the corruption actually moved the rows it claims to corrupt.**
A pure multiplier is a no-op on zero. DVOL's minimum is 19.17 so it was safe
there; 17.8% of funding rates are exactly 0.0 so it was blind here. *(pitfalls
12; `funding-features`)*

**R13. Missing means NaN, never a neighbour.**
Dense grids, −1 out of range, no stale forward-fill. A gap must fail closed.

**R14. Split boundaries are data, not folklore.**
Read them from `noctua/splits.py`. I once typed plausible dates instead and a
subagent that read the file got it right. *(`iv-coverage-2`)*

**R15. Distinguish the volatility INDEX from historical REALIZED volatility.**
Different Deribit endpoints, different meanings. This confusion has cost this
project once already and must not return.

**R16. Coverage is measured per feature, per split, before the experiment is
designed.**
It is what ruled out a plain IV feature column (32.7% train vs 100% test) and
produced the residual design that worked. *(`iv-coverage-2`)*

**R34. A comment that says "(verified)" is not a verification.**
`direction_bench.py` carried the line *"Only `cal_H` varies with H at a fixed
anchor (verified)"*. Four columns do. The word had been written by someone who
believed it, which is exactly the state the word is supposed to rule out. If a
claim is load-bearing, the check that establishes it belongs in the code, runs
every time, and refuses. *(`D1-direction-bench`, §30)*

**R35. A feature whose definition contains the horizon must be computed at the
horizon.**
`seas_{d}d` is the realized volatility of `[a - 24d, a - 24d + H)` and
`cal_weekend_frac` is the weekend share of the forward window. Both are
functions of `(anchor, H)`, not of the anchor, and joining them per anchor
hands every horizon the values of whichever horizon sorted first. The tell that
this is structural rather than cosmetic: at `H = 168` two of those columns do
not exist at all, because the window would run past the anchor. *(§30)*

**R36. Build the derived table with the function that ships, not with a copy of
its arithmetic.**
The h4 feature matrix is produced by calling `noctua.features.build_features`,
and cross-checked against `features.parquet` at the one horizon the two tables
share: max |diff| 0.000e+00 over 127,080 episodes. A transcription would have
been correct on the day it was written and silently stale afterwards. This is
R18 pointed at data instead of at scoring. *(`vol-matrix`)*

---

## Rules about copying

**R17. Copying an audited implementation copies its assumptions too.**
`ivfeatures.py` was audited and correct *for DVOL*. The property that made it
correct — a strictly positive series — was never written down, so it did not
travel with the code. Write down what makes it correct. *(`funding-features`)*

**R18. Reimplementing the scoring elsewhere is how two "identical" comparisons
stop being identical.**
Route new slices and new arms through the *same* code path.
*(`benchmark.run_fold`'s `prod_override` and `post_shift_fn`)*

**R19. Prove the additive change is additive.**
`post_shift_fn=None` reproduced every control fold to four decimals across four
separate runs. Assert it; do not assume it.

---

## Rules about interpretation

**R20. Correcting a number in the humbler direction does not make the
correction right.**
The 1.5218 benchmark was a wrong correction of a wrong number. *(pitfalls 6)*

**R21. Relevance is not absolute size, and skill is not a sign.**
Check against the base rate and against the null, not against zero.
*(pitfalls 1–3)*

**R22. A ratio of two intervals measures precision only if both are on the same
scale.**
I wrote a scale-dependent primary endpoint into §22 while thinking about episode
counts. *(`E-power-amendment`)*

**R23. Name what each outcome will change, before the number exists.**
Otherwise any result gets narrated as confirming what you already believed.
§9's mechanism was explained confidently and backwards. *(`E-power-bands`)*

**R24. A correct answer is not evidence the machinery works.**
E-anchor's NaN CI produced the right verdict by luck. That is the worst case,
not the best. *(pitfalls 11)*

---

## Rules about controls and estimators

**R39. Ask whether the baseline is allowed to know what the candidate knows.**
Not "is the baseline simpler" — simpler is the point of a baseline — but "is it
denied an input the candidate has". NOCTUA takes `cal_H`; the scored `log_har`
and `har_short` were fitted once on a pooled multi-horizon sample and carried no
horizon term. Refitting them per horizon turned a **+0.14462 win into a −0.02362
loss** at H = 168, and a +0.03032 win into a −0.00705 loss at H = 24. The pooled
fit was costing the baseline a factor of **2.06** at the extreme horizon. Every
margin in the first matrix was partly a measurement of that asymmetry, and the
largest margin was almost entirely it. *(`vol-matrix-fair-result`)*

**R40. The straw-man check runs on your own headline too.**
`vol-matrix-fair-result` implicates `eval/benchmark.py`, which fits its
baseline on the same pooled multi-horizon sample. Registering
`E-prod-fairbaseline` to test the project's own headline is not optional once
the mechanism is known — a correction you apply only to the result you dislike
is not a correction. *(`E-prod-fairbaseline`)*

**R42. Enumerate the baseline family the repo already contains, before asking
whether the chosen baseline was fitted right.**
Twice in one session the strongest available baseline was already in
`noctua/baselines.py` and simply absent from the arm list — `log_har_cal` in the
four-horizon matrix, `har_short` on the production slice. `VOL_BASELINES` has
six entries; `eval/benchmark.py` scores one. Fixing the *fit* of `log_har_cal`
moved it 2.2 %; adding an arm that already existed moved the bar enough to erase
significance. **A missing arm is a larger error than a mis-fitted one, and it is
cheaper to find.** *(`E-prod-fairbaseline-result`)*

**R41. An artifact key is not a description of what it holds.**
I asserted that the headline was measured against a horizon-blind `log_har`,
because `benchmark.json` stores it under the key `"log_har"`. The line that
produces it reads `bl["log_har_cal"]` — the baseline already carries `cal_H`.
The claim was in a committed ledger entry before I read that line. **Read the
expression that computes the number, not the name it is filed under**, and do
it before writing the sentence that depends on it, not after. This is R14
("split boundaries are data, not folklore") pointed at derived values instead
of at constants. *(`vol-matrix-fair-result` correction note)*

**R37. Point the new guard at the code you already trust.**
`_verify_per_anchor` was written for `vol_matrix.py`, a file with no results
yet. Its first run refused — and what it refused was an assumption that had
already shipped a completed benchmark in a different file. A guard is cheapest
to write while building something new and most valuable when aimed at
something old. *(§30)*

**R38. A null produced with degraded inputs is not a null.**
The direction benchmark returned a clean NULL at all four horizons. It was
re-run from scratch anyway, because the arms had been fed mis-specified
features and a weakened arm failing is not evidence that a correct arm would
have failed. Rerunning cost twenty minutes; the alternative was a permanent
asterisk. *(`D1-direction-bench`)*

**R29. A fix to a control is not neutral — it changes the test.**
Correcting the placebo rotation from a mis-specified 91 days to the intended 365
moved the placebo's own score from +0.0022 to −0.0078 and cut the real-vs-placebo
margin by 30%. Had it moved the other way the guard would have failed, and "we
fixed a bug" would have been the reason a result vanished. **Bug fixes to
controls need the same pre-registration discipline as treatments**, or they
become a channel for outcome-shopping. *(`E2-placebo-recheck`)*

**R30. A resampling interval over same-signed observations is not a test.**
Every resample is a convex combination of them, so it excludes zero at every
alpha by construction — verified identical at α/4 and α/1000. Use the t-interval
and the exact sign-flip permutation instead, and check the 2⁻ⁿ permutation floor
*before* running: at n = 5 it is 0.03125, so the design can never clear a
Bonferroni threshold with family ≥ 2 however large the true effect.
*(pitfalls 13; `E2-audit-downgrade`)*

**R31. Noticing a defect and continuing to rely on it is worse than missing it.**
§25 wrote down that the bootstrap saturated — "an artifact of n, not
significance" — and left it as the primary endpoint anyway. Writing a caveat is
not the same as acting on one.

**R32. An unresolvable question can sometimes be made irrelevant.**
Deribit's candle-timestamp convention could not be established from here. Rather
than argue it, the lag was made a parameter and the result re-run at a setting
correct under *either* convention. Cost: one hour of feature freshness.
*(`E2-lag2-robust`)*

**R33. "The code does X" and "the artifact contains X" are different claims.**
§26 said `anchor_freshness` "now records" a paired estimator. The code did; the
artifacts did not, because one run predated the patch and the other came from
`--from-json`. Verify the artifact, not the source. *(`E2-paired-estimator`)*

## Rules about shipping

**R25. ADVANCE is not ADOPT.**
Clearing a rule earns candidate status. Adoption additionally requires: audit,
an untouched holdout, family-wide multiple-testing correction, regime stability,
seed stability, and — where honestly testable — economics after costs.
*(`ledger.py` VERDICTS)*

**R26. The untouched holdout gets exactly one evaluation.**
If it changes the model, it is no longer untouched, and renaming previously
examined data does not create a new one.

**R27. Quote the conservative estimate.**
E2c is −11.61% pooled on the production slice and −6.18% on the wide one. The
wide number is the one that goes in the model card.

**R28. Never manufacture a P&L.**
If tradable historical prices, spreads and fills are not available, say so and
run forward paper trading instead.

---

*Educational research only. Not financial advice.*
