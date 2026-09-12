# NOCTUA status and adversarial audit — external state

*Written 2026-09-05, at the moment five audit agents were dispatched. This file
is the EXTERNAL STATE of that audit: what is being attacked, by whom, what the
supervisor says about the trajectory, and what was believed BEFORE the agents
reported. Anything the agents overturn should be edited here with the
correction visible, not silently.*

---

## 1. What NOCTUA is, right now

**One change has ever shipped from Phase 2**, and it is a reporting fix rather
than a model change. The model artifact is untouched.

| | |
|---|---|
| shipped model | NOCTUA V1, 6,939 params/seed, 3-seed ensemble, unchanged |
| shipped change | `P2-level-report-adopt` — the published σ now carries a trailing QLIKE scalar; the predictive object does not |
| gate | `model/tests/test_level_report.py`, required in CI |
| ledger | 120 entries, 75 live: 20 ADOPT / 25 REJECT / 13 ADVANCE / 12 NULL / 5 OPEN |
| rules | 52, each earned by breaking it once |

### The production-slice numbers behind the shipped change (n = 2,046)

| reading | QLIKE | calibration ratio |
|---|---|---|
| `sigma_med` (was shipped) | 0.29688 | 1.2656 |
| `sigma_mean`, no fit | 0.25326 | **0.9120 — out of band** |
| `c · sigma_med`, 1 fit — **ADOPTED** | 0.26806 | 0.9761 |

The two functionals on the same episodes: `median(RV/σ) = 0.9664` against
`sqrt(mean(RV²/σ²)) = 1.1250`. **They point in opposite directions**, which is
why the previously-shipped median-based correction scored 0.31441 — worse than
not correcting at all (0.29519).

### The result that reframed the phase

| H | best raw | best RESCALED | NOCTUA calib ratio |
|---|---|---|---|
| 1 | `garch_t` | **`noctua_v1`** 0.54985 | 1.433 |
| 6 | `garch_t` | **`noctua_v1`** 0.34884 | 1.464 |
| 24 | `har_short` | `noctua_v1` 0.25087 (tie) | 1.457 |
| 168 | `har_short` | `har_short` | 1.166 |

Phase 1's *conclusion* stands — raw is what ships, and raw NOCTUA loses. Its
*diagnosis* does not: NOCTUA is not missing what the HAR family has. It has one
stable ~20 % level bias, and every post-hoc correction for it fails the barrier
battery.

---

## 2. The audit in flight

Five agents, mandate **DISPROVE — not verify, not defend**. Model tier by
stakes: Sonnet where a wrong answer costs shipped code or a research direction,
Haiku otherwise.

| # | target | why it is worth attacking | tier |
|---|---|---|---|
| 1 | `P2-level-report-adopt` | the only shipped change, live in `serve/predict.py` | Sonnet |
| 2 | `P2-scorecard-rescaled-result` | overturned Phase 1's diagnosis; killed teacher mining | Sonnet |
| 3 | `P2-dst-shift-result` | load-bearing for Phase 3; a harvester was built on it | Sonnet |
| 4 | `P2-seed-variance-result` + R51/R52 | rewrote two rules from n=3 | Haiku |
| 5 | `harvest_events.py` + `P3-attention-feature-a2` | never run against real data | Haiku |

Each agent was told to run INSPECT → PLAN → IMPLEMENT → EVALUATE, to execute
code rather than reason about it, to treat a test that could not have failed as
worthless (R2), and that **"no defect found" is a failure of imagination until
they have genuinely tried**.

### Specific leads handed to the agents

* #1 — whether `_settled_anchors`'s `- H` term actually excludes unsettled
  episodes; W-sensitivity of the 6.5 % claim across W ∈ {20,40,60,90,120}.
* #2 — whether the ranking survives a *different* equally-defensible level
  correction; whether NOCTUA's 3-seed ensembling confounds it against
  single-fit baselines.
* #3 — **the null distribution of the cross-correlation argmax.** If lag +1 is
  over-represented under no true shift, `p = 0.0005` is an artifact.
* #5 — `merge_archive`'s append-only guarantee under clock skew, and whether
  the self-test's "live-row-wins" check actually calls the function it names.

---

## 3. Supervisor: stagnation in the trajectory

`python -m model.research.supervisor` → 68 alerts (45 STALE, 14 REPETITION,
8 OSCILLATION, 1 UNCLOSED). Most STALE alerts are the append-only ledger
working as designed. Two findings are real, and one is about the supervisor
itself.

**FINDING 1 — the `forecast level` mechanism has been attacked four times.**
`P2-scale-v2` REJECT → `P2-mean-level` REJECT → `P2-level-report` ADOPT. Two
consecutive architecture failures, then a success that was *not* an architecture
change — it moved the correction out of the model and into the reporting layer.
**Redirection, already taken:** stop trying to move the level inside the
predictive object. Three arms including a randomised one degraded all six
barrier metrics identically, so the damage is from moving the level at all. The
one untested route is changing what the quantiles mean during *training*.

**FINDING 2 — the `clock / event timing` mechanism has failed three times as a
feature while succeeding once as a diagnostic.** `P2-intraday-basis` NULL,
`P2-event-window` NULL, `P2-dst-alignment` NULL — against `P2-dst-shift`
ADVANCE at p = 0.0005. Under TEACHER_ZOO §7 that is three strikes on the
feature channel. **Redirection, already taken:** the events are real and
located; what is exhausted is *timing as a feature*, because the clock already
carries it and it costs more capacity than it returns. Phase 3 therefore
targets **surprise magnitude**, not schedule — which is why the harvester
collects volume and tone rather than a calendar.

**FINDING 3 — the supervisor's own `OSCILLATION` detector is too blunt to
trust here.** It groups by broad topic, so it reports `phase2` — thirty
entries spanning a dozen distinct questions — as one oscillating question. A
phase is not a question. The detector is sound for narrow topics (`refresh`,
`baseline`) and produces noise for wide ones; it needs a finer grouping key
before its output can gate anything.

---

## 4. What was believed before the agents reported

Recorded so the audit can be scored honestly rather than rationalised:

1. The shipped reporting fix is correct and its CI gate can genuinely fail.
2. The rescaled ranking is fair — every arm gets the same 1-parameter correction.
3. `P2-dst-shift` is the strongest result of the phase and is not an estimator
   artifact.
4. R51's narrowing from n = 3 is under-powered and may not survive contact.
5. `merge_archive`'s live-row-wins guarantee is the weakest thing built today.

**Predictions 4 and 5 are where I expect defects.** If the agents come back
clean on all five, that is more likely a sign the audit was too easy than that
the work is sound.

---

## 5. Agent returns — scored against §4

### Agent 4 — `P2-seed-variance` / R51 / R52 (Haiku) — RETURNED

| its attack | its verdict | mine after re-testing |
|---|---|---|
| (a) n=3 sd is marginal | supported, borderline | **ACCEPTED** — ratio CI [0.082, 0.990], excludes 1.0 by 0.010 |
| (b) seed dependence = "FATAL FLAW" | devastating | **REFUTED on all three supports** |
| (c) "~3x" overstated | 3.81× | **ACCEPTED**, corrected in R51 |
| (d) `P2-artifact-locus` | all claims exact | **CONFIRMED** independently |
| (e) R52 justification weak | weak | **partially** — practical claim unaffected |

**Why (b) fails.** Tested rather than argued: |corr| between draws is 0.01110
for consecutive torch seeds vs 0.01099 for non-consecutive, against 0.01200
expected for independent draws — t = +0.040, p = 0.969. The agent asserted the
mechanism without testing it. Its `p = 0.000` came from `scipy.stats.spearmanr`
on **three points**, where the exact answer is 1/6 = 0.167; and it wrote
"1/6 ≈ 0.0167", off by ten, then reasoned from that. → **R53**.

**Score against §4:** prediction 4 ("R51's narrowing from n=3 is under-powered
and may not survive contact") is **CORRECT**. R51 is now marked PROVISIONAL.

**The instructive part:** the report's *headline* was wrong and its *footnote*
was right. Accepting it as written would have withdrawn a correct rule on a
false mechanism; dismissing it for containing a tenfold arithmetic error would
have lost the one interval that changed what I believe.

### Agents 5 and 6 — the shipped level fix, and the rescaled ranking (Haiku) — RETURNED

Both were dispatched before the artifact loss was known and were redirected
mid-flight with explicit instructions not to fabricate, estimate, or substitute a
synthetic dataset. Both complied: four of eight attacks came back NOT TESTABLE
and were reported as such rather than answered.

| agent | its verdict | mine after re-testing |
|---|---|---|
| 5 (a) look-ahead in `_settled_anchors` | no leak | **CONFIRMED** — 1,440 episodes, every one `row + H < anchor_row` |
| 5 (b) the 6.5% claim | NOT TESTABLE | **stands unaudited**, not confirmed |
| 5 (c) the CI gate can fail | passes, can fail | **CONFIRMED** — a leaked scalar moves 18 barrier fields |
| 5 (d) "sigma and barriers are orphaned outputs" | no consumers | **FALSE** — `app.py:87` renders it; it searched only `src/*.js` → **R57** |
| 6 (1,3,4) fold scope, family size, `optimal_c` inputs | clean | **CONFIRMED** by reading the source |
| 6 (2) "the leak guard is worthless" | critical defect | **REFUTED as stated** — but a REAL defect sits in the same slot → **R56** |

**Why agent 6's mechanism fails, and what was there instead.** It argued the
guard cannot fire because `c` is fitted over the CALIB valid-index set while the
ratio averages over the TEST one. Both sides use `nanmean`, which drops exactly
the masked entries, so a fully test-fitted run gives a pooled ratio of
1.000000000000 and the guard **does** fire — shown by building it with 3% holes in
test against 12% in calib, the agent's own stated condition. The real defect was
found by running the case the agent never built: the guard checked the **pooled**
ratio, so a leak in SOME folds passes silently, and with 2 of 6 folds leaking the
pooled ratio (1.00217) sits FURTHER from 1 than the honest run's (1.00152) — the
leaking run looks more honest. Now per-fold and naming the offending years.

**Score against §4:** prediction 1 **CORRECT**. Prediction 2 is now **UNTESTED** —
the rescaled ranking has never been independently recomputed, and
`P2-scorecard-rescaled-result` is demoted from ADVANCE to **OPEN** until it is.
Predictions 3 and 5 were already scored. Four of five beliefs survived; the one
that mattered most — that the ranking is fair — turns out to be unexamined rather
than examined and sound.

**The pattern across all three agents.** Three for three, the headline was wrong
and something smaller in the same report was right or pointed somewhere right.
Accepting any headline as written would have been an error; dismissing any report
wholesale would have lost R53, R56 and R57.

---

## 6. Corpus restored — what is now auditable that was not

`model/artifacts/` was lost with the container; the source is a public MIT
dataset, so the corpus was rebuilt behind a gate (`noctua/regenerate.py`, 5/5).

* **The holdout hazard was real.** The source updates daily; a clone taken
  2026-09-12 carries 21,697 minutes at or after 2026-08-28, inside the forward
  holdout. A naive re-ingest would have spent it silently. The corpus is pinned
  to a DATE, the row count cross-checks it, and a disagreement is refused.
* **Fidelity against statistics published BEFORE the loss**: filled minutes 0
  (published 0), zero-volume 17.0846% (17.08%), bad prints 0.1505% (0.15%). The
  zero-volume target is the discriminating one — on the untruncated corpus it
  reads 16.98%, so it separates the right truncation from a near-miss.
* **One honest residue**: 476,359 h4 episodes against a published 476,362. The
  easy explanation (upstream revision) is **false** — the source's own provenance
  sidecar differs from the original by exactly the elapsed minutes, zero drift →
  **R59**. Immaterial (0.00063% against contrasts of 0.006–0.062), recorded
  anyway.
* The pipeline's own H=6 cross-check between the two independently built feature
  tables: 127,074 episodes, max |diff| `0.000e+00`.

`teacher_oof.npz` is rebuilding. When it lands, the single load-bearing Phase 2
number becomes auditable for the first time.

---

## 7. Supervisor, rebuilt — it can now gate something

The defect recorded in §3 was worse than recorded: OSCILLATION fired on **eight of
nine topics**. Two causes, and the second was not the key at all — the test asked
whether a bucket CONTAINED both an ADOPT and a REJECT, never whether a verdict
flipped on one question. Rekeyed to topic/mechanism, date-ordered, consecutive
decisive reversals only, **supersessions excluded** (a successor overturning its
predecessor is the method working). The real ledger now yields **one** alert:

    phase2/level-scale
      P2-armA-correction REJECT -> P2-level-report-adopt ADOPT
      P2-level-report-adopt ADOPT -> P2-mean-level-result REJECT

which is the mechanism §3 identified by hand as attacked four times. REPETITION
now names it too: `phase2/level-scale`, 19 attempts — the most-attacked question
in the project. → **R60**, selftest 6/6.

---

## 8. The labelled-dataset decision — made

Delegated explicitly and now answered in `LABELLING_DECISION.md`, on a
measurement (`P2-tail-clustering`) rather than an opinion: **no** hand-built
event-label set, **yes** to one dense continuous exogenous channel.

The deciding number, and it flips across the tail depth (**R58**):

| tail | episodes | distinct days | of calendar | band |
|---|---|---|---|---|
| worst 5% | 480 | 180 of 401 | 0.449 | C — labelling the calendar |
| worst 1% | 96 | 58 of 401 | 0.145 | A — labellable |

Both beat both nulls at p = 0.0000, so significance decides nothing. The tail
carrying 52.8% of the loss is spread over 45% of days; the tail that fits on 58
days holds 15.9% of variance mass across 96 episodes and cannot clear a
0.34%-per-column cost at a 1.53× effective-n multiplier. My pre-registered band
(B) was wrong at both depths — the prediction had not named the depth.

---

*Educational research only. Not financial advice.*
