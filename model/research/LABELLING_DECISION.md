# Should NOCTUA be trained on a labelled event dataset?

**Decision date:** 2026-09-12
**Status:** decided. One half is rejected, one half is adopted as the only
remaining open research channel.

The question put to me was: every day something happens in the world that moves
BTCUSDT, NOCTUA sees only that the number moved and never why, so should it be
trained on a labelled dataset instead of raw OHLCV — and if so, should that
dataset be built here?

## The decision

**NO to a hand-built event-label dataset.** Not on cost, and not on principle —
on a measurement. **YES to one dense, continuous exogenous channel**, harvested
hourly, which is already pre-registered as `P3-attention-feature` and already has
its collection workflow written. The two answers come from the same number.

## What the premise gets right

It is correct, and it is the most important true thing said about this model.
`P2-dataset-audit` confirmed it as fact, not impression:

* All 42 model inputs derive from OHLCV plus the clock.
* The entire extent of NOCTUA's world-knowledge is one binary, `reg_post_etf`,
  which flips once in January 2024.
* `data/news.json` holds 14 live headlines. It is not a corpus. `data/fg.json`
  holds a single current value.
* DVOL and funding exist in the repo, unwired.
* No macro calendar, no flows, no order book, no on-chain, no sentiment.

And the error is where the premise says it would be. The worst 5% of episodes
carry **52.8%** of NOCTUA's pooled QLIKE at H=1 and 52.0% at H=24. Half the loss
lives in one twentieth of the episodes. If anything is going to be fixed by
knowing what happened in the world, it is those episodes.

So the case for labelling is not naive. It is the best-supported research
direction the project has left. Which is why it deserved a measurement rather
than an opinion.

## The measurement that decides it

Concentration in the *loss* distribution does not imply concentration in
*calendar time*, and only the second one decides whether a label can target
anything. `P2-tail-clustering` measured it on the 9,600 surviving hourly
realized-variance observations (401 days):

| tail depth | episodes | distinct days | fraction of calendar | band |
|---|---|---|---|---|
| worst 5% | 480 | **180** of 401 | 0.449 | **C** — labelling the calendar |
| worst 1% | 96 | **58** of 401 | 0.145 | **A** — labellable |

Both depths beat both null models at p = 0.0000 (free permutation: 283.7 and
85.8 expected days; diurnal-preserving permutation: 286.3 and 85.9). So
"volatility is clustered" is statistically overwhelming and **decides nothing**.
The magnitude is what decides, and it splits:

* **The tail that carries the loss is not day-targetable.** The worst 5% — the
  52.8% of QLIKE — is spread over 45% of all days. "Label the big event days"
  describes labelling nearly every other day of the year. There is no small set
  of days to go and annotate.
* **The tail that is day-targetable does not carry enough.** The worst 1% fits
  on 58 days a year, 42% of it on ten days. That is genuinely annotatable. It
  holds 15.9% of realized-variance mass and 96 episodes in 400 days.

I pre-registered band B and was wrong at both depths. The prediction was not even
well-posed: I had not named the tail depth, and the answer moves by 3× across it
(logged as R58).

## Why 58 labellable days is not enough to fund a label

Three measured costs, all already in the ledger:

1. **`E-power`**: 24× more episodes bought **1.53×** effective sample size — on
   the order of six independent regime observations in six years. A label does
   not buy years of evidence; it buys one column.
2. **`P2-event-window-result`**: one extra column costs about **0.34%** at H=1.
   A feature must clear that to break even before it has helped anything.
3. **Three timing features have already failed** (`P2-dst-alignment-result`,
   `P2-event-window-result`, and the arm-A correction), and
   `P2-dst-shift-audited` removed the evidence that the driver is *scheduled*
   macro at all. The channel that survives is surprise **magnitude**, not
   surprise **timing**.

A sparse flag firing on 96 episodes, inside a sample with roughly six
independent regime observations, against a 0.34% entry cost, cannot be fitted.
It is not a close call and no amount of annotation quality changes it. The
limit is the effective sample size, not the labels.

And one more constraint the premise cannot be argued past: **direction is
measurably NULL at all four horizons.** No label can be justified by a promise
to fix direction, because nothing in this project has yet shown direction to be
predictable at all. A labelled dataset would be bought against volatility level,
which is the one thing already known to be a single stable constant.

## What is adopted instead, and why it is the same answer

The only shape that fits the measurement is a feature that is **dense** — a
value at every hour, not a flag on 58 days — and **continuous in magnitude**, so
that it informs the 5% tail spread across 180 days rather than only the ten
loudest days. That is exactly what `P3-attention-feature` pre-registers: hourly
GDELT attention volume as a surprise-magnitude channel, with its expected effect
pre-declared as concentrated in the worst 5%.

This is not a consolation prize for labelling. It is labelling, done at the only
granularity the sample size can pay for: instead of a human writing *"2026-03-14:
Trump spoke on tariffs"* against 58 days, every hour carries *how much the world
was talking*, which is the part of the label that generalises. A named event is a
category with one or two instances in six years; attention magnitude is the same
event expressed as a number that has 9,600 instances.

**The honest limitation, pre-registered before collection:** articles are written
*because* the price moved. This channel is a candidate lagging indicator and the
pre-registration says so. If it fails that test it will be recorded as a failure,
not rescued.

## What labelling can and cannot be expected to fix

* **Cannot fix direction.** Direction is NULL at all four horizons. There is no
  evidence a label would change that, and claiming otherwise would be the
  project's fourth timing story.
* **Cannot fix the level bias.** NOCTUA's deficit is one stable ~20% constant per
  horizon. That is an arithmetic defect, not missing world-knowledge, and every
  post-hoc correction of it has failed the barrier battery. The one shipped fix
  corrects the published functional instead.
* **Could plausibly fix part of the spike tail** — the 52.8% of loss in the worst
  5% — if, and only if, the exogenous signal arrives *before* the variance does.
  That is the open question and it is now the only one.

## Blocking dependency

The harvester (`.github/workflows/harvest-events.yml`, merge logic in
`model/eval/harvest_events.py`, 10/10 selftest) is written but **not registered**:
GitHub only registers `cron` and `workflow_dispatch` from the default branch, so
the live accumulation clock cannot start until PR #13 merges. Historical coverage
must come from the GDELT backfill path; live coverage accumulates only after the
merge. Nothing in this decision can be tested until then.

---

*Educational research only. Not financial advice.*
