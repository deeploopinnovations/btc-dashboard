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

*Educational research only. Not financial advice.*
