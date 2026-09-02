# TEACHER_ZOO — the frozen protocol for Phase 2

**Frozen 2026-08-28, before any teacher prediction was generated.** Everything in
this file is fixed. It may be *amended only before results exist*, and an
amendment must be recorded in the file with its date, its reason, and a
statement of exactly what had already been seen when it was made (RULES R1).

Phase 1 is accepted and frozen. Its artifacts and conclusions are immutable
historical evidence and are not re-litigated here.

---

## 0. What changed, and why this protocol exists

Phase 1 measured NOCTUA against a properly specified baseline family and found
it does not win at any horizon. The response is not to keep treating stronger
models as opponents. It is to treat them as **teachers**: find out what
information or inductive bias lets each one win, transfer only the mechanisms
that survive falsification, and build a successor that must beat *the teachers*,
not NOCTUA.

**NOCTUA V2's minimum bar is the strongest properly specified teacher at that
horizon.** Beating NOCTUA V1 is not a result.

---

## 1. The common frame — identical for every model, no exceptions

Any model entering the zoo is scored through the same code path. Reimplementing
the scoring for a new arm is how two "identical" comparisons stop being
identical (R18).

| element | value | source of truth |
|---|---|---|
| episode table | `model/artifacts/episodes_h4.parquet` | H ∈ {1, 6, 24, 168} |
| feature table | built by `noctua.features.build_features` at each episode's own horizon | never joined per anchor (R35) |
| splits | `noctua.splits.walk_forward_folds` | test years 2021–2026 |
| embargo | `ep["H"].max()` hours, derived not typed | `splits.py` |
| target | `har_target(RV, H) = log(RV) − ½·log(H)` — the log hourly vol **rate** | `noctua/baselines.py` |
| scored quantity | σ = exp(ŷ)·√H | so every arm is compared on the same object |
| loss | QLIKE: `r − log r − 1` where `r = RV² / σ²` | identical expression in `garch.py`, `vol_matrix.py`, `benchmark.py` |
| primary estimator | paired **per-episode** moving-block bootstrap | STATS_PROTOCOL §1 |
| block length | `max(round(n^⅓), 2H)` | STATS_PROTOCOL §1b |
| fold-level | reported, **labelled underpowered** where `vol-matrix-power` says so | never the primary |
| small n | `small_n_inference`: t-interval governs, permutation floor 2⁻ⁿ stated | STATS_PROTOCOL §2–3 |
| spike / calm | RV in the top 5 % of the test slice's own distribution | reporting only, never a pass condition |

Barrier metrics (pinball, CRPS, Brier, CORP decomposition, Christoffersen) are
scored through `eval/benchmark.py` unchanged for any arm that emits a
distribution rather than a point σ.

---

## 2. The cross-fitting rule — the one that makes distillation legitimate

> **An episode may never receive a teacher prediction from a teacher that was
> trained on that episode, or on any episode whose forward window overlaps it,
> or on anything after it.**

This is not a formality. A stacker or a residual learner trained on in-sample
teacher forecasts is learning the teacher's *training* error, which is smaller
and differently shaped than its forecasting error, and the resulting model will
look excellent and fail in production.

Concretely, for every teacher **T**, every horizon **H** and every walk-forward
fold **f**:

1. `T` is fitted on `fold[f]["train"]` only.
2. Predictions are emitted for `fold[f]["calib"]` and `fold[f]["test"]`.
3. The **calib** predictions are the *only* teacher values any Phase 2 learner
   (residual model, stacker, router, student) may train or select on.
4. The **test** predictions are used *only* to score, never to fit anything —
   including teacher selection, weights, gates, hyperparameters or a decision
   to include an arm at all.
5. `train` predictions are **not emitted at all**. There is no legitimate use
   for them in this phase, so they are not produced and cannot be reached for.

### Amendment 1 — INNER cross-fitting, recorded 2026-08-29, before any Arm A result

The rule above says train-slice teacher predictions are not emitted at all.
That is right for the reason given, and it is also **insufficient**, which was
not noticed until Arm A was implemented and no Arm A number yet exists.

Arm A trains NOCTUA on `r = y − ŷ_T`. That target requires a teacher
prediction **for every training episode**. The rule as written forbids the only
values that would make the arm constructible, so one of the two has to give.

The resolution is not to relax the rule. It is that there are **two different
things** both called "a teacher prediction on the train slice":

- **In-sample** — the teacher fitted on all of `train` and asked about `train`.
  Forbidden, still, and for the original reason: its error is smaller than its
  forecasting error and small exactly where the teacher overfit, which is
  exactly where the student is asked to correct it.
- **Inner out-of-fold** — the train slice is itself split into an expanding
  forward sequence of inner blocks; the teacher is fitted on blocks `1..k` and
  predicts block `k+1`. No episode ever receives a prediction from a teacher
  that saw it.

Only the second is produced, into a separate artifact named
`teacher_oof_inner.npz` so it can never be confused with the outer one.

**The inner split is EXPANDING-FORWARD, never K-fold.** Ordinary K-fold fits
each held-out block partly on later blocks, which is look-ahead on a time
series. This repository has already discarded one estimator for exactly that —
`noctua/train.py`'s `sigma_ref` comment records a cross-fitted version that
"used ordinary K-fold, which fits each held-out block on later blocks too, and
was discarded". The same mistake is not made twice.

The first inner block has no predecessor and therefore gets no teacher
prediction; those episodes are dropped from the residual training set rather
than given a fallback value. A fallback would be R43 all over again — inventing
a number and letting the model read it as information.

**Teacher selection is itself cross-fitted.** Phase 1 already demonstrated why:
at H = 1 and H = 6 the best teacher *on test* is `garch_t`, while the
calibration slice chooses `persistence` and `har_short`. Choosing the teacher
by test performance would import that gap directly into every downstream arm.
Every "best teacher" statement in Phase 2 means **best by calibration QLIKE**,
and the test-side ranking is reported beside it as a separate, non-actionable
observation.

`teacher_zoo.py` refuses to write an OOF file whose fold masks intersect, and
refuses to emit a `train`-slice prediction at all. Both refusals are checked in
`pitfalls.py` and must be demonstrably capable of firing (R2).

---

## 3. The forward holdout — untouched, and here is what "untouched" forbids

    FREEZE DATE: 2026-08-28 (UTC). Holdout = anchors strictly after it.

Nothing in Phase 2 may touch it: not teacher selection, not hyperparameter
tuning, not architecture decisions, not feature selection, not stacking weights,
not router training, not distillation, not a decision about which arm to report.
`research/DATA_USE.md` documents why no *historical* holdout exists and is
unchanged by this phase.

---

## 4. Provenance — what is recorded for every candidate

Recorded before the model is run, in `research/teacher_ledger.json`:

version · source repository · licence for **code and weights separately** ·
commit or checkpoint hash · parameter count · peak inference RAM · context
length · target formulation · covariate support · probabilistic output
capability · **pretraining-corpus leakage risk** · whether evaluation is
zero-shot, fine-tuned or from scratch · compute cost per fold.

**Pretraining leakage is a first-class disqualifier, not a footnote.** A
foundation model whose pretraining corpus plausibly contains BTC price history
through our test period is not zero-shot on this problem — it has seen the
answer. Such a model may be recorded and reported, but it may not be a teacher
for any transferred mechanism, and its score carries the flag wherever it
appears.

If a model cannot be run because weights or infrastructure are unavailable, and
a reproducible external runner (CI, another permitted environment) also cannot
produce forecasts, it is recorded **BLOCKED BY INFRASTRUCTURE** — explicitly
*not* as a scientific finding about the model. Prediction artifacts plus hashes
and provenance are stored; large checkpoints are never moved into this
repository.

---

## 5. The scorecard — per horizon, and disagreement is preserved

There is **no universal winner** and no composite score. For each of
H = 1, 6, 24, 168 the scorecard reports, side by side:

pooled QLIKE · spike QLIKE · calm QLIKE · deep-tail behaviour · calibration ·
worst fold · compute cost.

Where these disagree — and Phase 1 showed they do, GARCH-t having the best spike
QLIKE and the worst calm QLIKE at H = 168 — **the disagreement is the finding**
and is reported as such. Collapsing it into one number would destroy the
information the mechanism work depends on.

### Amendment 2 — every QLIKE comparison reports raw AND rescaled, recorded 2026-09-02

Section 5 says the disagreement between metrics is the finding. It did not say
that one of the columns it prints — **calibration** — can silently determine
the ranking in the column beside it, and this phase has now demonstrated that
four separate times:

| finding | what it looked like | what it was |
|---|---|---|
| `garch_normal` spike wins at all four horizons | tail skill | level; rescaling sent H=168 spike 0.239 → 1.620 |
| NOCTUA's whole deficit against the zoo | missing information | one stable constant per horizon |
| `har_short`'s fitting panel, +8.11 % at H=6 | specification | level; +0.40 % CI [−0.00265, +0.00580] rescaled |
| the teacher ranking itself | which teacher is best | level; NOCTUA wins at H=1 and H=6 once equalised |

QLIKE is minimised by the conditional **mean** of variance. An arm whose
forecast happens to sit at the right level beats a better-informed arm that
sits low, and the calibration ratios in this zoo span **0.4160 to 2.4167**.
A ranking on raw QLIKE across arms that differ that much in level is a ranking
on level.

**So, from this amendment forward:** any comparison of two volatility forecasts
under QLIKE reports the raw contrast *and* the contrast with each arm carrying
its own scalar `c = sqrt(mean(RV²/σ²))`, fitted on that fold's **calib** slice
only and applied to that fold's test slice — per fold, through `FoldScopedFit`
(R44). The burden is on the raw number to explain itself.

**Two things this amendment does NOT do.** It does not make a rescaled forecast
adoptable: `P2-scale-v2` applied a fitted constant to the shipped model and
degraded all six barrier metrics, and the product is a touch-probability curve,
not a σ. And it does not change §9's adoption gate, which still runs on what
actually ships. The rescaled column answers *which arm knows more*; the raw
column answers *which arm forecasts better today*. Both are reported and
neither substitutes for the other.

*(`P2-scorecard-rescaled-result`, `P2-pool-composition-result`,
`P2-armA-correction`)*

---

## 6. Statistical discipline, carried over unchanged

- MDE stated **before** each comparison; below-MDE contrasts are labelled
  NOT POWERED before they run (R5).
- **Family size fixed before results.** Each Phase 2 arm family declares its
  size at pre-registration. Failed models stay in the family and count against
  the correction.
- Every guard must be demonstrably capable of failing (R2).
- Negative controls mandatory: a shuffled-teacher control and a
  time-rotated-teacher placebo for every transfer experiment.
- **R39 applies to teachers too**: before comparing, ask whether the arm is
  allowed to know what its competitor knows.

---

## 7. The stagnation supervisor for teacher mining

> **If three consecutive architecture changes based on the same teacher
> mechanism fail, that mechanism is closed.** Stop modifying it, return to the
> scorecard, and select a different source of information.

Enforced by `research/supervisor.py`, which counts consecutive REJECT/NULL
verdicts carrying the same `mechanism` tag in the teacher ledger and raises a
STOP when the count reaches three. The point is to prevent indefinite tuning of
a losing idea, which is the characteristic failure mode of distillation work.

---

## 8. What the shipped model may do during Phase 2

**Nothing.** The shipped NOCTUA artifact is not modified during teacher
discovery. No Phase 2 result may be adopted into it until it clears §9.

---

## 9. The adoption gate for NOCTUA V2

An architecture is adopted only when **all** hold:

1. It beats **that horizon's strongest calibration-selected teacher** under the
   locked paired-per-episode protocol — not NOCTUA V1.
2. No unacceptable tail or calibration degradation.
3. Directionally acceptable across folds; worst fold reported.
4. Each component has evidence for its own inclusion — an ensemble must beat
   **the best single teacher** and the **equal-weight** control, both mandatory.
5. For a distilled student: it is scored on **real forecasting accuracy**, kept
   strictly separate from **teacher imitation accuracy**. A student that
   imitates the teacher better but forecasts reality worse is **rejected**.

If the evidence shows a simple HAR/GARCH/teacher ensemble beats a neural NOCTUA
V2, that is the reported result and the neural complexity is not preserved.

---

## 10. Direction stays closed

`D1-direction-bench-corrected` is NULL at all four horizons with correct
features and n ≈ 49,000. **No attempt will be made to manufacture direction
skill from volatility teachers.** Direction reopens only on genuinely new
directional information, or a teacher with independently demonstrated
directional skill.

*Educational research only. Not financial advice.*
