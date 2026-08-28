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
