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

**R55. Uncommitted work in an ephemeral container is not state.**
This container restarted mid-audit and took with it a confirmed bug fix, a
rewritten self-test, a ledger entry and a rule — all verified, none committed.
The ledger went from 122 entries back to 121, `numpy` vanished from the
environment, and the scratchpad emptied. Nothing was recoverable; all of it had
to be rebuilt from the conversation transcript, which is the only reason it
survived at all. **Commit a verified result before starting the next one**, and
treat the transcript as the write-ahead log it accidentally is. *(this session)*

**R54. Rivals that all assume the same mechanism are not a control set.**
`P2-dst-shift` was framed as a sharp two-hypothesis test: lag 0 if the footprint
is UTC-anchored, lag +1 if it follows the US Eastern schedule — two rivals
predicting different integers, which felt like good design. **Both assumed the
answer was a clock.** The rival that mattered — *the warm half of the year
differs from the cold half for reasons unrelated to any schedule* — predicts the
**same** integer and was never tested. A plain Mar–Oct calendar split, touching
no timezone code, reproduces the result at corr@+1 = **3.6575** against the true
DST split's 3.6476 (the two splits agree on 96.19 % of episodes, so it is the
same partition reached differently — which is the point, not a defence). R50
asks for a control that *can return a different number*; this needed one that
could return the **same** number for a different reason.
*(`P2-dst-shift-audited`)*

**R53. A p-value from a library function is not a p-value until you know that
function's valid range at your n.**
An audit agent reported `scipy.stats.spearmanr` p = **0.000** for a rank
correlation on **three points** and built a "devastating, fatal flaw" on it. The
exact combinatorial answer is 1/3! = **0.167** for a specific monotone ordering,
0.333 either direction — an unremarkable coincidence. scipy returns its
asymptotic approximation without complaint at n = 3. The same report also wrote
"1/6 ≈ 0.0167", off by a factor of ten, and reasoned from it. This is R2 in a
new place: not a guard that cannot fail, but a **statistic that cannot be valid,
reported as though it were**. *(`P2-audit-seedvar`)*

**R52. A control horizon is clean only for the confound it was measured
against.**
H=24 is used across several experiments as the no-footprint control because
`P2-event-footprint` measured chi-square 3.0, p = 1.000 there. That establishes
the absence of an **hour-of-day** footprint. It does **not** establish that two
arbitrary 4-of-24-hour partitions score alike: `P2-event-window` found a
**0.14 %-of-base difference between two of them, reproducible across three
independent ensembles** — not seed noise (`P2-seed-variance`) and not
seasonality (`P2-artifact-locus`, corr with summer −0.00013). The two arms are
identical in winter and differ on the 10.9 % of episodes that are US summer, so
the whole artifact lives there. Name the confound a control rules out, and do
not borrow it for a different one. *(`P2-artifact-locus`)*

**R51. A paired bootstrap over episodes conditions on the fitted models, so it
omits training variance — and that omission is LARGEST where there is no signal.**
*(Measured, and narrower than the version first written here.)*
`P2-seed-variance` re-ran the `P2-event-window` contrast with three independent
3-seed ensembles. Across-seed-set sd against the bootstrap half-width:

| horizon | gaps | sd | half-width | ratio |
|---|---|---|---|---|
| H=1 (signal) | +0.00136 / +0.00128 / +0.00120 | 0.000079 | 0.00050 | **0.16** |
| H=24 (no signal) | −0.00037 / −0.00001 / −0.00043 | 0.000225 | 0.000375 | **0.60** |

The first draft of this rule claimed the omission "can be the whole effect".
**It cannot** — it is smaller than the bootstrap interval at both horizons.
What survives, and is the useful part: the seed component is **3.8× larger where
there is nothing to fit**, because two arms with real signal converge to similar
solutions that differ consistently while two arms with none differ by whatever
the optimiser landed on. So seed-set replication is most needed on **controls and
null cells**, which is exactly where it is least likely to be run — and is not
worth tripling the cost of a primary that clears widely.
**PROVISIONAL:** the chi-square CI on that sd from n=3 puts the ratio at
[0.082, 0.990] — it excludes 1.0 by one hundredth, so this is supported by a
marginal interval and should be re-measured with more seed sets before it gates
anything load-bearing. *(`P2-seed-variance-result`, `P2-audit-seedvar`)*

**R50. A control whose transformation is a symmetry of your statistic is not a
control.**
`P2-dst-alignment` pre-registered "no fixed hour offset may beat the Eastern
clock" as its guard. A fixed offset is a **relabelling** of 24 bins, so the bin
counts are a permutation of themselves and chi-square, every lift and the top-K
concentration are identical to UTC **by construction** — measured invariant to
1e-12 across all 23. The stated reasoning for the control ("a fixed offset moves
both halves of a doublet together and can never merge them") was correct, and is
precisely why it carries no information. The question to ask of a control is not
"would this catch the artifact I have in mind" but **"is there any input at all
for which this returns a different number"**. *(`P2-dst-alignment-result`)*

**R49. A control that reports only pass or fail throws away the measurement it
was making.**
`P2-intraday-basis`'s shuffled-hour arm was installed to catch capacity
masquerading as clock signal. It caught none — and because it was *scored at
every horizon* rather than reduced to a verdict, it showed that the cost of 23
extra columns is **1.86 % at H=1, 4.05 % at H=6, 0.02 % at H=24 and 1.33 % at
H=168**. That 200-fold variation, not the signal, explains the entire table:
the clock information is worth almost exactly the capacity it consumes, so it
reaches the bottom line only at the horizon where capacity happens to be free.
The primary could not have asked that question. Report a control's value, not
just its verdict. *(`P2-intraday-basis-result`)*

**R48. When two candidates and a randomised control all fail the same way,
the finding is about the pipeline, not about the candidates.**
`P2-mean-level` ran three level corrections through the model: one derived per
episode from its own quantiles, one constant, and one **shuffled** — the same
mean shift with the per-episode alignment destroyed. All three improved QLIKE
(+14.69 %, +10.72 %, +7.02 %) and **all three degraded all six barrier metrics
by essentially the same amount**, DSC −20 % in every case. `P2-scale-v2` had
already found this once with a fitted constant. Three arms failing identically
is not three failures; it is one property — *no post-hoc shift applied through
the predictive object can pass this battery* — and the randomised arm is what
turns "two corrections failed" into that sentence. Budget for the control.
*(`P2-mean-level-result`)*

**R47. A completeness mask is a sample-selection decision, and building one
over "all the columns" can delete a whole stratum without saying so.**
`pool_composition.py` masked on all 42 feature columns. **No H=168 episode has
a complete 42-column record** — `seas_1d` and `seas_5d` read a window that runs
past the anchor once H > 24d — so the mask silently removed the entire long
horizon, and the `{1,6,24,168}` panel became `{1,6,24}`, which was also one of
the treatment arms. The reference and a treatment were the same estimator under
two names, and the table would have read as "no effect". The corrected run
reproduces the defective run's number exactly as the `{1,6,24}` row, which is
what proves the diagnosis rather than merely asserting it. Mask on the column
policy the trusted code uses, and check the surviving row counts per stratum.
*(`P2-pool-composition-a1`, correction B)*

**R46. A component you rebuild to route through a different pipeline must be
validated against the original before anything is scored — and when they
disagree, the disagreement is the finding, not the obstacle.**
`arm_a_adopt.py` refit `har_short` because the OOF artifact is built on a
different episode table than the pipeline it had to run inside. Its
pre-registered validation refused the run: the refit scored **0.37951 against
the artifact's 0.41710**, 9 % apart, at correlation r = 0.9978. The refusal was
worth more than the run it blocked — it exposed that the artifact fits its OLS
teachers **per horizon** while the pipeline's own fit pools them, which is a
property of every Phase 1 and Phase 2 baseline number and had never been
stated. Without the validation the barrier battery would have run, produced
clean numbers, and reported them under Arm A's name for a different teacher.
*(`P2-armA-adopt-a1`; the finding it opened is `P2-pool-composition`)*

**R45. When a candidate improves the metric you optimise, the second metric —
the one that represents the actual product — is not a formality. It is the one
that knows whether you improved the thing or the proxy.**
The scale correction improved QLIKE by 9.7 % pooled and 33 % on spikes, fixed
the calibration ratio from 1.2662 to 0.9759, and passed three of five guards.
**Every one of the six barrier metrics degraded** — DSC −14.16 %, MCB +8.48 %,
Brier, CRPS, log score and pinball all worse. Four independent lines of evidence
(the scorecard, the overlay, the falsifier, the calibration ratio) pointed at
adopting it. The barrier clause, written into `E-scale` in Phase 0 and left
unmet for three phases, is what produced the correct answer and reversed all
four. *(`P2-scale-v2-result`)*

**R44. A cross-fitting guard that watches the producer does not watch the
consumer.**
`teacher_zoo.py` has three refusals proving no episode receives a prediction
from a teacher trained on it. All three fired correctly. They did not catch —
and structurally could not catch — `scale_falsifier.py` fitting **one constant
on calib pooled across all six folds** and applying it to every fold's test
slice, so that 2025 calibration data rescaled 2021 forecasts. The leak was not
in the teacher output; it was in a parameter fitted *over* the teacher output by
a later stage. **Every stage that fits anything on top of cross-fitted values
needs its own scope check**, and `FoldScopedFit` is it: the consumer declares
which fold it is fitting for and any read from another one refuses.
*(withdrawn scale-falsifier run; matters most for Arms B and C, which fit
stacking weights and a router over the same values)*

**R43. A sentinel value is a lie the pipeline tells itself. Ask what the
substituted value CLAIMS.**
`features.py` guards `log(0)` with `max(rv, EPS)`, `EPS = 1e-12`. That is a real
problem solved by inventing a number — and the number is not neutral. It asserts
*realized volatility was 1e-6*, the strongest possible statement the feature can
make, in the most dangerous direction. On 2022-07-13 an hour with **volume
exactly 0** became `har_1h = −13.8155`; `har_short` forecast σ = 0.000290; BTC
fell 3.1 % in the next hour. That one episode is **72.5 % of the entire fold's
mean QLIKE**. Prefer NaN and let the row fail closed — which is what R13 already
required. *(`P2-floor-defect`)*

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

**R56. A guard on a POOLED statistic cannot detect a PARTIAL violation, and
aggregation can invert its sign.**
`scorecard_rescaled` refused to run if the post-rescale calibration ratio was
exactly 1, the signature of a constant fitted on the slice it is scored against.
That check ran on the ratio POOLED across six folds. It fires correctly when
every fold leaked — the pooled ratio is then exactly 1 by construction — and is
blind when only some did, because the honest folds pull the average off 1. Worse
than blind: with two of six folds leaking the pooled ratio sat at 1.00217 while
the fully honest run sat at 1.00152, so the leaking run looked *more* honest than
the clean one and no tolerance on the pooled number could ever separate them. The
guard must be evaluated at the granularity at which the thing it guards is
FITTED — per fold, because `c` is fitted per fold. An adversarial agent reported
this slot as defective for a different and incorrect reason (an index-set
mismatch between calib and test, which `nanmean` makes impossible); the real
defect was found only by building the partial-leak case the agent never ran.
*(`P2-audit-rescaled-ranking`, `scorecard_rescaled.py --selftest`)*

**R57. An agent that searches the wrong surface reports absence, not absence.**
An audit concluded the served `sigma_window_pct` and `barrier_curves` were
"orphaned outputs... not consumed by any downstream system", having grepped
`src/*.js` and `scripts/*.js`. `model/serve/app.py:87` renders
`sigma_window_pct` to the reader; it is the published number. The true and
narrower finding — the dashboard front-end does not read NOCTUA's sigma or
barriers, only Kronos `upside` and `volAmp` — was available from the same
evidence and is worth knowing. Take the residue, not the headline.
*(`P2-audit-levelfix`)*

**R58. "Is the tail clustered?" has no answer until you name the tail depth —
and the answer can flip across it.**
I pre-registered band B for the day-concentration of extreme episodes and was
wrong twice over: the worst 5% of hours touch 180 of 401 days (fraction 0.449,
band C — labelling that is labelling the calendar), while the worst 1% touch 58
(0.145, band A — forty-one percent of them on ten days). Both crush both nulls at
p = 0.0000, so "statistically clustered" is true at either depth and decides
nothing; the fraction is what decides, and it moves by 3x between two depths of
the same distribution. A prediction stated without the depth was not a falsifiable
prediction. Name the depth, and read the MAGNITUDE rather than the p-value.
*(`P2-tail-clustering`)*

**R59. Prove whether the SOURCE moved before blaming it for a reproduction gap.**
The rebuilt corpus came back 3 episodes short of the published 476,362, and the
ready explanation was an upstream revision to a daily-updated dataset. It is
testable and it is false: the source's own provenance sidecar records 7,710,019
rows at 2026-08-29 04:19 against the original's 7,681,837 at 2026-08-09 14:37, a
difference of 28,182 rows against 28,182 elapsed minutes — **zero drift**. History
was appended to, never revised, so the minute grid is identical and the gap is a
bookkeeping error in my own published figure. Blaming the source would have closed
the question with the one explanation that the evidence excludes. Also: the gap is
0.00063% against contrasts of 0.006–0.062, so it is immaterial — which is a reason
to state it, not a reason to stop looking for it.
*(`corpus_manifest.json`, `P2-corpus-restored`)*

**R60. A detector that fires on nearly everything has the same value as one
that fires on nothing, and is more dangerous because it looks alive.**
The stagnation supervisor raised OSCILLATION on eight of nine topics — including
`features` (20 entries) and `phase2` (32) — so nothing it said could gate
anything, and its output read as a wall of confirmed stagnation. Two independent
defects, both invisible while the alerts looked plausible:
*the key* was `topic`, a chapter of the research rather than a question; and
*the test* was whether a bucket CONTAINED both an ADOPT and a REJECT anywhere,
which in 32 unrelated entries is close to certain. Oscillation is a claim about a
SEQUENCE on ONE question: same mechanism, date-ordered, consecutive decisive
verdicts reversing, and **excluding supersessions** — a successor overturning its
predecessor is this project's method working, not stagnation. Rekeyed to
topic/mechanism with that test, the real ledger yields exactly one alert, and it
is the right one: the level mechanism, `P2-armA-correction` REJECT →
`P2-level-report-adopt` ADOPT → `P2-mean-level-result` REJECT.
*(`supervisor.py --selftest`, 6/6)*

**R61. Put a DETERMINISTIC arm in every reproduction, so the data is controlled
before the model is blamed.**
The rebuilt corpus was 3 episodes short of the published count, which made every
reproduction discrepancy ambiguous between "the data moved" and "the model is not
reproducible". The scorecard settles it for free, because it scores seven teachers
at once: `har_short` came back 0.25697 against a published 0.25696, `garch_t`
0.41060 against 0.41073 — fitted on the *same* rebuilt corpus, reproducing to
1e-5. A reproduction containing only the stochastic arm could not have told a
data difference from a model difference at any cost, so carry the deterministic
arms.

**NARROWED, 2026-09-12, and the narrowing matters more than the rule.** I drew
from this that "the data is therefore not the explanation" for the neural arm's
gap. That inference is invalid. A closed-form or MLE estimator depends on sample
composition *smoothly* — drop three of 476,359 rows and the normal equations move
by order 1/n — while SGD depends on it *chaotically*, because three fewer rows
shift every minibatch boundary for the rest of training. So a deterministic arm
controls the data for other **smooth** estimators and not for a neural one. What
the deterministic arms actually licensed was the much weaker claim that the
corpus is not *grossly* different. Stating the stronger one sent me chasing an
environment difference that turned out to be nothing (R63).

**Now measured, so the narrowing is a number rather than an argument:** dropping
3 of 476,359 training episodes moves `noctua_v1` by up to **0.00209** and every
deterministic teacher by at most **0.00006** — a factor of **35**. That is the
sensitivity ratio between the arm you are testing and the arm you offered as its
control. *(`P2-sample-sensitivity`)*

**R62. "It reproduces" is a claim about the CONCLUSION and about the NUMBER, and
they can diverge.**
The rescaled ranking reproduced structurally and exactly: 3 of 4 horizons change
rank, NOCTUA best at H=1/6/24, `har_short` holding at H=168 — so ending teacher
mining was correctly founded. And the H=24 verdict still flipped, from a published
TIE (CI [-0.00183, +0.01231]) to a clear win (CI [+0.00260, +0.01582]), on the
same code, same corpus and same protocol. Report both halves: a structure that
survives re-running does not license the decimals quoted inside it, and a row
whose verdict moves between two honest runs of the same design must not be cited
as either outcome. Also: the measured seed sds (0.000079, 0.000225) could not be
used to bound this, because they are sds of PAIRED GAPS, where common training
noise cancels, and the quantity in question is a pooled LEVEL. Two different
statistics wearing the same word.
*(`P2-scorecard-rescaled-reproduced`)*

**R63. A pin you never check is a comment — but an unchecked pin is not thereby
the cause of anything. ~~Verify the versions~~ *Verify the versions, then test
whether they matter.*  [PARTLY FALSIFIED — see the correction below]**
`noctua_v1` missed its published QLIKE at all three horizons while every
deterministic teacher returned to 1e-5. I hypothesised seed nondeterminism; two
full runs came back **bit-identical**, 528 of 528 arrays, worst diff `0.000e+00`,
so that was falsified. The cause was one line of `requirements-research.txt`:
torch is pinned at **2.13.0+cu130** and the container had **2.14.0+cu130**, while
numpy, scipy, pandas and arch all matched their pins exactly. The correspondence
is the proof — the packages that drifted and the arms that moved are the same set:
OLS through numpy and GARCH MLE through arch converge to the same optimum whatever
the summation order, while SGD over thousands of steps amplifies a last-bit
difference into a pooled-loss shift of 0.002–0.004, enough to flip a verdict.
Nothing in CI compared the installed version to the pin.

**FALSIFIED THE SAME DAY, BY THE TEST I SET MYSELF.** The pin was satisfied
exactly — PyPI's `torch==2.13.0` is the `+cu130` build — and the rebuild came back
**bit-identical to the drifted one**, 528 of 528 arrays, worst |diff|
`0.000e+00`. The torch drift moves nothing. The "perfect correspondence" between
the packages that drifted and the arms that moved was a **coincidence**, and it
was mine: three packages had drifted, not one, and *which* of them touch a neural
arm is not evidence about *whether* their drift changed a number. What survives
is only the hygiene half — check pins, stamp the environment beside results — and
`env_check.py` earns its place on that alone. What does not survive is any claim
that this drift explained the reproduction gap. Two explanations offered, two
falsified by running them; the gap stays open rather than taking a third story.
*(`P2-repro-cause`, amended)*

**R64. A correspondence between which things changed and which results moved is
not a mechanism, and it is the most persuasive kind of wrong.**
Three packages had drifted from their pins; `torch` was one; `noctua_v1` was the
only arm that failed to reproduce and the only torch-dependent arm. The fit was
exact, the story was mechanical and correct in general (SGD does amplify kernel
arithmetic), and I wrote it into the ledger and a rule. Pinning torch exactly then
produced a **bit-identical** archive: the drift moves nothing. The failure was not
the hypothesis, it was treating a correspondence over three items as settled
before running the one-command test that could refute it — and the test was cheap.
Any explanation of this form gets run, not written up, however well it fits.
*(`P2-repro-cause`, `P2-sample-sensitivity`)*

**R65. Select by the mask that governs, never by a proxy for it — and an exact
zero in a stochastic arm is evidence about the harness.**
Three times I built a perturbation test and three times it perturbed nothing,
because the rows to drop were chosen by a stand-in for "is in the train slice":
the earliest rows in the episode file (removed as warm-up by the completeness
mask), then the earliest rows in the model table (2012, while
`splits.SAMPLE_START` is **2017-08-01**, so outside train, calib and test alike).
Selecting from the union of the folds' actual `train` masks worked on the first
attempt. The diagnostic was available from the first run and I used it only after
the second: a move of **exactly 0.00000** in an SGD-trained arm cannot happen if
the input really changed, so an exact zero there is a statement about your
harness, not about your model. Read a suspiciously clean null as a bug report.
*(`P2-sample-sensitivity`)*

**R66. The capacity effect of an extra column is not a constant — it changed
sign between two of this project's own experiments.**
`P2-event-window-result`'s shuffle arm measured **−0.34%** at H=1 and the figure
was quoted thereafter as "one extra column costs about 0.34%", including as a
hurdle in a decision document. `P3-exogenous-dvol`'s shuffle arm measures
**+1.46%** at H=1: same construct, same horizon, opposite sign, four times the
size, and it clears a Bonferroni-corrected interval. The sample differs (187,727
episodes against 476,359) and so does the column's type (continuous against a
sparse indicator); which drives the sign is open. What is settled is that no
design may assume a per-column cost from another design. Measure it inside your
own, with a control matched to the arm's **own width** — a one-column shuffle does
not control a three-column arm.

**NARROWED, and the narrowing vindicates the original figure.** Running the same
noise column on the FULL sample gives **−0.36%** at H=1, against
`P2-event-window-result`'s **−0.34%** — two independent experiments, different
columns, same sample, agreeing to 0.02 percentage points. So a per-column cost IS
reasonably stable *within* a sample and does not transfer *across* samples of
different size: −0.35% at 476,359 episodes against +1.46% at 187,727. The figure
the project had been quoting was right for the sample it came from, and the sign
flip was the 2.5×-smaller sample, not the construct. Which makes the real rule
narrower and more useful: **sample size, not the column, is what a capacity
measurement is conditional on.** *(`P3-colshuf-sample`)*
*(`P2-capacity-profile`, `P2-event-window-result`, `LABELLING_DECISION.md` amended)*

**R67. A pre-registered threshold can pass on its letter and fail on the property
it was written to detect — say so, and do not bank the pass.**
`D2` at H=168 was registered to need a gain "concentrated in the spike bucket",
operationalised as spike > calm. It came in at **+2.00% spike against +1.91%
calm** — passing by 0.09 percentage points while being, plainly, a uniform gain.
The threshold was a proxy for a qualitative claim and the proxy can be satisfied
without the claim. Report the margin, not just the verdict, and when the margin
makes the test vacuous treat the result as unestablished rather than as a win
defended by the letter of your own rule.
*(`P3-exogenous-dvol`)*

**R68. Decompose a multi-column win before crediting any column in it, and
subtract the width-matched capacity floor.**
`D2` cleared at H=168 while every one of its three columns failed alone:
`x_ivrv` +0.00165, `x_dvol_chg` +0.00110, `x_fund` +0.00062, none clearing, against
a one-column noise floor of +0.00051. Their sum, +0.00337, is D2's +0.00325 — the
effect is **additive with no interaction term to appeal to**, so the honest unit
of the finding is the block and not any feature in it. Two further things only the
decomposition shows: capacity scales with width (one noise column +0.00051, three
+0.00111), so the informative content is +0.00325 − 0.00111 = **+0.00214**,
independently equal to the direct paired contrast against `D2-shuf`; and `x_fund`
at 1.2× the floor is **nothing**, which a block-level result would have let pass
as a contributing feature. Per column the information is ~0.0007, far under what
one column can demonstrate at this n.
*(`P3-exogenous-decomposition`)*

**R69. A point-comparison guard with no tolerance fires on numerical noise —
and you may not fix it after it fires.**
The pre-registered barrier battery blocked `D2` on three metrics: Brier +0.0080%,
logs +0.0043%, DSC −1.67% (0.00120 → 0.00118). Every movement is in the fifth
decimal place, three metrics moved up by 1e-5 and three moved down by 1e-5, so the
substantive reading is *barriers unchanged*, not *barriers damaged* — nothing like
`P2-scale-v2`, the precedent the guard was written from, which degraded every
metric substantially. Both halves have to be reported: the guard blocked, and the
guard is over-sharp. **The guard still stands for this result.** It was registered
as a strict inequality, it failed, and rewriting it after seeing it fail is the
selective rigor this project exists to avoid. The fix belongs in the NEXT
pre-registration — a paired interval per barrier metric, not a bare inequality —
and the current arm stays unadopted in the meantime.
*(`P3-exogenous-barriers`, `P2-scale-v2-result`)*

**R70. Check a per-fold guard AT THE HORIZON, not across horizons pooled.**
Amendment 1 to `P3-exogenous-dvol` announced "5 test years, 2022–2026" from a
coverage table of 8,633 training episodes in fold 2022. That count is across all
four horizons; at H=168 alone it does not clear the ≥2000 gate, so the one
surviving result rests on **four** folds (2023–2026), not five. A guard that
applies per horizon must be evaluated per horizon, and an amendment written to
forestall exactly this kind of surprise got it wrong by pooling.
*(`P3-exogenous-barriers`)*

**R71. A published instrument may be calibrated for a different loss than yours —
check before importing it.**
`arXiv 2607.05291`'s Mincer–Zarnowitz recalibration is the standard way to separate
"better scaled" from "better informed". Applied here unchanged it **degraded QLIKE
for six of seven teachers**, by 31–38% at H=1, and left a post-correction
calibration ratio near **1.7** instead of 1. MZ minimises squared error; QLIKE is
minimised by the conditional **mean of variance** — this project's own `E-scale`.
Fitting a level by least squares and then scoring it with QLIKE charges a Jensen
gap that has nothing to do with whether the forecast is informative. The fix keeps
the part that answers the question (the slope β, which carries responsiveness) and
sets the level by the closed-form QLIKE optimum: ratios move to 0.87–1.15 and the
comparison becomes meaningful. Import the *question* a published method answers,
not its arithmetic.
*(`P3-mz-result`)*

**R72. Give every rival the same number of free parameters you give your candidate,
and a claim will often narrow.**
`P2-scorecard-rescaled` handed every teacher ONE free parameter (a level) and
concluded NOCTUA was best at H=1, H=6 and H=24. Handing every teacher **two**
(level and slope, fitted on calib, applied symmetrically) leaves the ranking intact
but moves H=6 from a win to **not established** — CI [−0.00391, +0.01498]. The
surviving claim, H=1 and H=24, is the first positive statement about NOCTUA in
Phase 2 to survive a strong symmetric control, and it is narrower than the one it
replaces. The corollary is the useful half: a rival that looks weak may simply be
the one you gave fewer parameters to.
*(`P3-mz-result`, `P2-scorecard-rescaled-reproduced`)*

**R74. Ask which FUNCTIONAL of your model's output the loss actually wants, before
asking anything else about the model.**
NOCTUA emits a predictive distribution. The teacher zoo read its **median** and
compared it against rivals whose single point forecast behaves like a **mean**,
under QLIKE — a loss minimised by the conditional **mean of variance**. Switching
to the model's own `sigma_mean`, from the same forward pass, fitting nothing,
improves raw QLIKE by **13.3%, 20.7%, 26.0% and 11.9%** at H=1/6/24/168. For
scale: the largest adopted gain in the programme before this was 6.5%, and the
whole exogenous-data effort produced one uncertain 1.9% at one horizon.
The value was available from the day the model first emitted a distribution, and
three things hid it: `sigma_mean` existed and was never scored; the fitted level
constant `c` was silently doing part of the median→mean conversion, so the gap
looked like a calibration problem rather than a functional one; and the serving
side found the same defect from the other end (`P2-level-report-adopt`) without
the research side connecting it. Before tuning an architecture, a feature set or a
loss, check that you are scoring the functional the loss is minimised by.
*(`P3-functional-parity`)*

**R75. A process that dies without a traceback was KILLED, not crashed — and
check the resource that binds, not the one you can see.**
`vol_matrix`, `direction_bench` and `econ_voltarget` were launched together
because CPU load was 0.58 on four cores. All three died. None left a traceback,
all three truncated mid-output, and memory came back fully reclaimed — the
signature of the OOM killer, not of three independent bugs. Each job holds the
full 476,359 × 42 episode table plus torch; load was never the constraint,
resident memory was. The distinction matters because a crash is a defect in your
code and a kill is a defect in your scheduling, and an hour spent debugging the
first when it was the second is an hour wasted. `scripts/regen_artifacts.sh` now
runs them in sequence, cheapest first, skipping any artifact that already exists
and printing free memory after each.
*(`scripts/regen_artifacts.sh`)*

**R76. A test suite cannot audit code it never reaches — run a static
undefined-name check, and run it first.**
`direction.block_bootstrap_ci` referenced a `block_len` that had been added to
the signature of the function BELOW it and not to its own. Every call raised a
bare `NameError`, across 14 call sites, for 106 commits. `anchor_freshness`
had a second one: `_verdict` read a `spike` mask off `main`'s local scope.
Neither was caught, and the reason is the same in both cases — neither line is
reachable by anything that runs. The first sits in modules nothing had rerun
since the bug landed; the second sits behind an `all("_q" in r ...)` guard that
the stored-fold path does not satisfy, so the diagnostic silently did not exist
rather than failing. Ten passing gates said nothing about either, because all
ten execute code. `ruff check --select F821` found both in under a second and is
now the first gate in `scripts/precommit.sh` and in CI.
Two corollaries, both earned here. First: when you add a parameter to one
function, check whether the edit landed in a neighbour — the commit that did
this said "default unchanged, so no existing number moves", which was true of
the function it meant to change and false of the one it hit. Second: the
blast radius of a latent crash is bounded by what has been RERUN, so establish
that before panicking — no artifact on this branch was produced by any of the
14 callers after the bug landed, so no published number ever passed through it.
*(`scripts/precommit.sh`, `.github/workflows/model-ci.yml`)*

**R77. A diagnostic whose job is to validate a threshold must run BEFORE the
threshold is used, not after.**
`P3-regularisation` pre-registered a bar — what a shuffled noise column achieved
by accident — and ran sixteen arms against it. The `col-shuf` arm existed in the
module for exactly one purpose: to reproduce that accident on the same sample the
arms use, because the bar had been measured on the DVOL-era subset. It was run
fourteen minutes AFTER the arms finished. On the full sample the accident at H=1
is **−0.363%**, not +1.46%: the shuffled column hurts, there was never anything
at that horizon to reproduce, and the entire H=1 family was scored against a
number that does not exist. Ordering is not a detail here — had the diagnostic
run first, the pre-registration would have been written differently or not at
all, and the correct finding (at H=6, where the accident is real at +0.363%,
input noise at 0.15 reaches it) would have been the whole experiment instead of
a footnote to a void one. The general form: whenever a rule contains a constant
that came from somewhere else, the run that re-derives that constant HERE is a
dependency of the pre-registration, not a companion to it.
*(`P3-regularisation-result`, and R66 for the import itself)*

**R78. Do not address a background job by a name pattern, and do not edit a
script while bash is reading it.**
Two ways the same queue broke in one session, neither of them about statistics.
First: `pkill -f "queue2.sh"` killed the shell that issued it, because the
harness puts the whole command string — pattern included — into the invoking
shell's own command line, so `pgrep -f`/`pkill -f` match the watcher as readily
as the watched. A waiter built on `while pgrep -f "seedtest.sh"` can therefore
wait on itself, or on a shell that merely mentioned the script. Wait on a **PID**
(`while kill -0 "$PID"`), and kill by PID.
Second: `seedtest.sh` was corrected while it was executing. Bash reads a script
lazily by byte offset, so an in-place edit that changes the file's length can
make the running shell resume in the middle of a line — the correction was right
and applying it that way was not. Edit a copy and relaunch, or make the edit
before the job starts.
Both were caught before they produced a wrong number, which is the only reason
they are a rule and not an incident.

**POSTSCRIPT, three hours later: the author of this rule broke it again.** A
`pkill -f "model.eval.prod_fairbaseline"` killed the shell that issued it, so a
file edit and a job launch chained behind it never ran, and the state had to be
reconstructed. Writing a rule down is not a mechanism. The replacement idiom is
concrete and has no judgement in it:

    PID=$(pgrep -f -- "<pattern>" | head -1)    # inspect only
    kill "$PID"                                  # act on the PID, never -f

and prefer not killing at all: launch into a scratch log, read the log, and let
the job finish. Anything that takes a pattern and acts on every match will
eventually match the hand holding it.
*(`scratchpad/chain.sh`)*

**R79. A regeneration script must reproduce the VARIANT the document reports,
not the tool's default.**
`scripts/regen_artifacts.sh` rebuilt `vol_matrix.json` with the bare invocation.
The document's volatility section is the **fair** run — OLS baselines refitted
per horizon, written to `vol_matrix_fair.json` — and `report.py` prefers that
file, falling through to `vol_matrix.json` only if it is absent. So the driver
was regenerating the horizon-blind matrix the ledger has already **rejected**,
and the report would have fallen through to it. Worse, it would have done so
quietly: the paragraph explaining which comparison this is, is gated on
`d.get("fair_baselines")`, so the rejected result would have appeared with the
sentence that identifies it simply missing. The report's own *Reproducing*
section carried the same bare command, so following the documented instructions
reproduced a different result from the one documented.
The general form: a default is a property of the tool, not of the result. When a
document reports a non-default run, the flag is part of the result's identity
and belongs in every place that claims to rebuild it — the driver, the
reproduction block, and the filename.
*(`scripts/regen_artifacts.sh`, `report.py` REPRO)*

**R80. When several designs oscillate on one question, question what they all
ASSUME, not what they disagree about.**
`phase2/level-scale` went REJECT → ADOPT → REJECT across three entries. The
three designs — a trailing QLIKE scalar, a fitted per-fold constant, and the
model's own per-episode mean/median delta — are three estimators of one number,
and they disagreed about the estimator while agreeing, silently, about the
intervention: **move the level**. The agreement was the part that was wrong.
There was no level error. The served scalar was the median of a predictive
distribution under a loss minimised by the mean, and the fix is to report the
other functional — which fits nothing and moves nothing downstream.
The evidence was already in hand and was read as a rejection instead of a
diagnosis. `P2-mean-level-result`'s **shuffled control** degraded every barrier
metric identically to the real arm (DSC 0.007254 against 0.007169). A control
that destroys the correction's content and reproduces the damage exactly proves
the damage belongs to the INTERVENTION CLASS, not the correction — and that
entry's own pre-registration had said in advance that this pattern "closes the
level question in the other direction". It closed. Nobody acted on it for
eleven days, because the arm had failed and a failed arm gets filed rather than
read.
So: a control that matches the treatment is not just a reason to reject the
treatment. It is a measurement of what the guard is actually responding to, and
it is often the most informative number in the run.
*(`P3-level-oscillation-closed`)*

**R81. A synthetic fixture tests LOGIC, never a CONVENTION — touch the real
data in the selftest or the fixture will agree with the bug.**
`frvp.py` filtered bars with `d["filled"] & ~d["bad_print"]`. In this corpus
`filled` marks a **forward-filled synthetic** bar, so that expression selects
exactly the rows to discard — of which the corpus has zero — and handed the
builder an empty frame. Twelve selftests passed, because the fixture set
`"filled": True` on its rows: it was written from the same wrong belief about
the column as the code, so it agreed with it. The tests were not weak; they
were testing the right logic against the wrong world.
No amount of synthetic coverage fixes this, because the fixture's author and
the code's author are the same and share the misunderstanding. The only check
that binds is one that reads the actual file: `frvp`'s selftest now builds from
a one-week slice of the real corpus and asserts it yields anchors, and it
FAILS loudly if the corpus is absent rather than passing vacuously.
Generalise: whenever a function encodes an assumption about what a column
MEANS — a flag's polarity, a unit, a timezone, a sign convention — that
assumption is unfalsifiable against data you generated yourself.
*(`P3-frvp-double-touch`, `eval/frvp.py`)*

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

**R73. Piping a gate's output through `tail` disarms the gate.**
A commit was pushed that failed CI on `ledger --validate` while the validator had
*already reported the problem locally, in the same command that did the push*:

    python -m model.research.ledger --validate 2>&1 | tail -1 && git commit ...

A pipeline returns the exit status of its **last** command, so `tail` returning 0
masked the validator returning 1 and the `&&` proceeded. The habit of piping a
long-running check through `tail` to keep the output short is exactly what
silently converts a gate into a print statement. Use `set -o pipefail`, or do not
pipe a gate at all. `scripts/precommit.sh` now runs the checks CI runs with
`pipefail` set, so the mistake cannot recur by hand.
*(`scripts/precommit.sh`)*

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
