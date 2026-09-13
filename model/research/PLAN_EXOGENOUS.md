# Plan: give NOCTUA information about the world, with the data that exists

**Written 2026-09-13, before any arm is run.** Supersedes the collection half of
`P3-attention-feature`, whose source is unreachable from here.

## Why the previous plan is dead, and why this one is better

`LABELLING_DECISION.md` concluded: no hand-built event-label set (the 5% tail
that carries 52.8% of the loss is spread over 45% of days, and the 1% tail that
fits on 58 days holds 96 episodes — not fundable against a 0.34%-per-column cost
at a 1.53× effective-n multiplier), and **yes** to one dense continuous exogenous
channel. That channel was hourly GDELT attention volume.

**GDELT is blocked by organization network policy** (`api.gdeltproject.org`,
CONNECT 403). No live feed, no backfill. The harvester stays written and tested;
it cannot be fed from here.

Two better series are already on disk and have never been wired in:

| series | rows | cadence | span | test-fold anchor coverage |
|---|---:|---|---|---|
| `data/newdata/dvol_btc.parquet` | 47,563 | hourly | 2021-03-24 → 2026-08-26 | 77.5% (2021), **100%** (2022–2026) |
| `data/newdata/funding_btc.parquet` | 64,206 | hourly | 2019-04-30 → 2026-08-26 | **100%** (2021–2026) |

DVOL is Deribit's 30-day BTC implied-volatility index. It is **strictly better
than the attention channel for this purpose**, on the exact objection that was
pre-registered against GDELT: news articles are written *because* the price moved,
so attention is a candidate lagging indicator. DVOL is the options market's
*forward* expectation — it is what people who are paid to price volatility think
the next month holds, and it moves when they learn something. It is the market's
own aggregate of every external factor, including whoever said what today,
published hourly, already covering the whole test period.

Funding is positioning: what leveraged longs are paying to stay long.

This is the user's question answered with the data that exists. NOCTUA cannot be
told "Trump spoke"; it can be told "the options market just repriced vol upward
relative to what was realized", which is the part of that event a volatility
model can use.

## The hypothesis, stated so it can fail

The shipped model's error is extremely concentrated — worst 5% of episodes carry
52.8% of pooled QLIKE at H=1. Those are episodes where realized volatility
arrived that the model did not expect from price history alone. If the options
market saw any of it coming, `IV − RV` measured at the anchor carries information
that OHLCV does not, and the gain must be **concentrated in the spike bucket**.

If the gain is uniform across the distribution, it is a level or capacity effect
and not information about events — which is the outcome three previous level
experiments already produced, and it would be reported as that.

## Features (deliberately few — each column costs ≈0.34% at H=1)

Built from hours **strictly before the anchor**, the project's existing
no-lookahead contract. `index_price` is excluded: it is a second price source and
adds nothing the model lacks.

1. `x_ivrv` = log(σ_dvol) − log(σ_realized trailing 24h), both as hourly vol
   rates. **The primary feature.** The variance risk premium: what the market
   expects *relative to* what just happened.
2. `x_dvol_chg` = log DVOL(a−1) − log DVOL(a−25). A 24-hour repricing. Large when
   the market has just learned something.
3. `x_fund` = funding `interest_1h` at a−1, z-scored on the train slice.

## Arms, and the controls that can sink them

| arm | columns added | purpose |
|---|---|---|
| `base` | — | the 40-column model, refit on the identical sample |
| `D1` | `x_ivrv` | the hypothesis, 1 column |
| `D2` | `x_ivrv`, `x_dvol_chg`, `x_fund` | the full channel, 3 columns |
| `D1-shuf` | `x_ivrv` permuted within fold | **the control that matters.** Same capacity, same marginal distribution, no information. If `D1-shuf` gains too, the gain is capacity or noise, not the world. |
| `D1-lag` | `x_ivrv` built from a−169 instead of a−1 | a week-stale version. If a week-old IV−RV does as well, the feature is a regime proxy, not news. |

## Sample: the same episodes for every arm, and why that is not optional

310,323 of 476,359 episodes predate DVOL. Requiring the feature would silently
make the comparison a different-sample comparison — the defect `P2-pool-composition`
already produced once, where a completeness mask over 42 columns deleted every
H=168 row and turned the widest panel into a treatment arm (R47).

So **every arm, `base` included, is fitted and scored on the DVOL-era sample
only** (anchors ≥ 2021-03-24 with both series present). The cost is stated up
front: training drops from 476,359 to roughly 196,000 episodes, and `base` here is
therefore **not** comparable to any previously published NOCTUA number. Only the
within-this-experiment contrasts are valid. `E-power` already measured that 24×
more episodes bought 1.53× effective sample size, so losing 60% of the rows costs
far less than it appears to.

## Amendment 1, written before any arm ran: the experiment is 5 folds, not 6

Measured after the plan was committed and before execution, because the plan
stated the sample but not what it implies per fold:

| fold | train episodes | with DVOL | with funding | calib with DVOL |
|---|---:|---:|---:|---:|
| 2021 | 101,369 | **0** | 40,173 | **0** |
| 2022 | 136,408 | 8,633 | 75,204 | 16,797 |
| 2023 | 171,447 | 43,672 | 110,243 | 16,796 |
| 2024 | 206,486 | 78,711 | 145,282 | 16,796 |
| 2025 | 241,621 | 113,846 | 180,417 | 16,795 |
| 2026 | 276,658 | 148,883 | 215,454 | 16,797 |

DVOL begins 2021-03-24 and fold 2021's train slice ends mid-2020, so **fold 2021
cannot be run at all** and drops out by the existing ≥2000-train guard. Fold 2022
has 8,633 train episodes across four horizons, ≈2,158 each — it passes the guard
by a margin of 8%, so it is the fold most likely to be noise.

This is a consequence of the registered sample, not a change to it, and it is
recorded here rather than discovered in the results. Consequences carried forward:

* The experiment is **5 test years (2022–2026)**, and H-by-fold counts are
  reported so the marginal fold is visible.
* Because the mask is identical across arms, every arm drops the same folds, so
  the pairing is unaffected. The run refuses if that ever stops being true.
* Funding alone would reach all 6 folds (fold 2021 has 40,173 train episodes).
  A funding-only arm is **not** added: it was not registered, and adding an arm
  after seeing the coverage table is how a family of 8 becomes a family of 10
  without the intervals noticing. It is recorded as a follow-up instead.

## Decision rule, fixed now

* Primary endpoint: **pooled QLIKE**, per horizon, H ∈ {1, 6, 24, 168}.
* Primary estimator: **paired per-episode moving-block bootstrap**, block
  `max(round(n^⅓), 2H)`, on the identical episodes. Fold-level intervals reported
  beside it and labelled, never as the rule.
* Family: 4 horizons × 2 live arms (`D1`, `D2`) = **8 comparisons**, Bonferroni →
  99.375% intervals. `D1-shuf` and `D1-lag` are controls, not members.
* **`D1` clears only if** its interval excludes zero favourably at the corrected
  level **and** `D1-shuf` does not clear at the same level. A gain that the
  shuffle reproduces is not adopted whatever its size.
* Secondary, pre-declared: the spike bucket (RV ≥ 95th pct) must show a **larger**
  relative gain than the calm bucket. If the gain is flat or calm-concentrated,
  the entry records the hypothesis as failed even if pooled QLIKE clears.
* Barrier battery (the six metrics `P2-scale-v2` degraded) reported for any arm
  that clears, because a QLIKE gain bought by wrecking the distribution is the
  failure mode this project has already met once.

## Prediction, recorded before running

`D1` clears at H=1 and H=6 and not at H=168; the gain is concentrated in the
spike bucket; `D1-shuf` does not clear; `D2` does not beat `D1` by enough to pay
for two extra columns. Confidence: moderate on H=1/H=6, low on the spike
concentration — three previous attempts to find structure in the tail have each
found a level effect instead.

**The honest prior:** every feature this project has added in Phase 2 has failed
(three timing features, two level corrections). The base rate for this arm
clearing is low, and the reason to run it anyway is that it is the first feature
built from information that is genuinely *outside* the price series.

## Known caveat carried in

`noctua_v1`'s pooled QLIKE is currently reproducible only to ~0.004 (see
`P2-scorecard-rescaled-reproduced`, `P2-repro-cause`). Every number here is a
**paired within-run contrast** on identical episodes with identical code, which is
exactly the comparison that irreproducibility does not touch — but no absolute
QLIKE from this experiment may be compared against a previously published one.

---

*Educational research only. Not financial advice.*
