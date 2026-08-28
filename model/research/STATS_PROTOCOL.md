# Statistical protocol — locked 2026-08-28

Locked after the adversarial statistical audit, not before. The audit found this
project's most promising result rested on an estimator that could not fail, so
the protocol below is written against that specific failure rather than in the
abstract.

Every rule here is enforceable by a check in `research/pitfalls.py`. A rule with
no executable check is a preference, not a protocol.

---

## 1. Which estimator is primary

**Paired per-episode, where episode-level pairing exists.** Arms trained
separately but scored on identical episode sets give paired deltas; use them.
This is the estimator with real power — n in the thousands rather than 5 or 6 —
and it is the one that can actually return a null.

**Fold-level is a stability check, never the primary**, unless the fold-level
design is shown *in advance* to be powered. It answers a different question —
whether an effect is stable across years — and E-power established that
between-year heterogeneity dominates it, so it is the question that matters for
deployment and the one this data usually cannot answer.

## 2. What is forbidden at small n

**A bootstrap over same-signed observations.** `check_bootstrap_can_fail`
(pitfalls 13) refuses it. At n = 5 with 5 same-signed deltas the interval
excludes zero at α/4 and at α/1000 identically — it is not responding to the
data.

**Quoting a confidence level the estimator cannot respond to.** If the interval
stops widening past α/3, saying "clears the Bonferroni-adjusted 98.75% level" is
describing a correction the statistic ignored.

## 3. What replaces it

`direction.small_n_inference` returns all three, and they are reported together
because they disagree usefully:

| tool | what it gives |
|---|---|
| t-interval | widens with α, proper for estimated variance at small n |
| exact sign-flip permutation | the test the paired fold design licenses |
| MDE at 80 % power | **noncentral-t**, not the z approximation `2.80 × se` |

The z approximation understated the MDE by 25–35 % at n = 5–6 and is not used.

**The permutation floor is checked before running.** One-sided p cannot go below
2⁻ⁿ: 0.03125 at n = 5, 0.015625 at n = 6. A design whose floor exceeds its
corrected threshold **cannot succeed** and must be redesigned or declared
underpowered before compute is spent.

## 4. Minimum detectable effect

Stated before every experiment, computed with noncentral-t from the fold-to-fold
sd the experiment will actually have — estimated from a comparable prior run, not
borrowed from an unrelated one. An expected effect below the MDE means the
experiment is a **non-measurement before it runs**.

Known MDEs on record:

| slice / target | MDE |
|---|---|
| production spike QLIKE | ~1.80 % (noncentral-t) |
| production pooled QLIKE | ~11.81 % |
| wide-slice pooled QLIKE | ~8.19 % |
| direction BSS, H = 1 | 0.0008 — the only powered direction horizon |
| direction BSS, H = 6 / 24 / 168 | 0.0042 / 0.0168 / 0.0517 — **not powered** |

## 5. Multiple testing

**The family is every hypothesis tested on overlapping folds, including
failures.** The ledger holds ~47 completed non-OPEN entries, the large majority
scored on the same walk-forward structure. A result quoted at "family = 4" when
the honest count is far larger is under-corrected.

**Romano-Wolf stepdown is the right tool and is NOT currently available.** It
needs a shared resampling scheme over the raw per-fold statistic of every family
member; the ledger stores only summary deltas and intervals for almost all of
them. Retrofitting requires re-running the family. **This is recorded as missing
infrastructure, not waved away** — until it exists, Bonferroni over the honest
family size is the fallback, and it is conservative.

## 6. Controls

Per **R29**, a change to a control is a change to the test. Any fix to a placebo,
shuffle, or decoy is pre-registered before it is applied, and the affected guard
is re-run rather than assumed to carry.

Placebos must be shown to be *hard*. A rotation that lands on the same calendar
position may retain seasonal information — that is conservative for a
"beats the placebo" guard, but it means the margin is a lower bound rather than
an unbiased estimate, and it is reported as such.

## 7. What a guard must satisfy

- it must be able to fail (**R2**), and there must be a test showing it can;
- it must be implemented, not merely documented — three guards in this project
  were documented and absent (**R2**, `iv-correction-audit`);
- it must not compare an expression to itself;
- it must not decide on a NaN (**pitfalls 11**);
- if it corrupts data, the corruption must be verified to have moved every row
  it claims to corrupt (**pitfalls 12**).

*Educational research only. Not financial advice.*
