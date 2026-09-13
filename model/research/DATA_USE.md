# Data-use ledger — which periods can still serve as an untouched holdout

**Answer: none of them.** Every calendar year in this repository has influenced
at least one of training, calibration, model selection, feature selection,
debugging, threshold choice, or a go/no-go decision. There is no historical
period left that qualifies as untouched, and renaming previously examined data
would not create one.

The rest of this document is the evidence for that claim, per year, so it can be
checked rather than believed.

---

## The splits, read from `noctua/splits.py` rather than remembered

    SAMPLE_START = 2017-08-01     TRAIN_END = 2023-01-01     CALIB_END = 2024-07-01
    walk-forward test folds: 2021, 2022, 2023, 2024, 2025, 2026
    episodes span 2012-01-01 -> 2026-08-09  (510,496 episodes)

## Per-year use

| year | episodes | train | calib | test | walk-forward test fold | verdict |
|---|---|---|---|---|---|---|
| 2012 | 33,903 | 0 | 0 | 0 | — | **examined** |
| 2013 | 35,033 | 0 | 0 | 0 | — | **examined** |
| 2014 | 35,040 | 0 | 0 | 0 | — | **examined** |
| 2015 | 34,669 | 0 | 0 | 0 | — | **examined** |
| 2016 | 35,136 | 0 | 0 | 0 | — | **examined** |
| 2017 | 35,040 | 14,688 | 0 | 0 | — | trained on |
| 2018 | 35,040 | 35,040 | 0 | 0 | — | trained on |
| 2019 | 35,040 | 35,040 | 0 | 0 | — | trained on |
| 2020 | 35,136 | 35,136 | 0 | 0 | — | trained on |
| 2021 | 35,040 | 35,040 | 0 | 0 | **YES** | trained on **and** scored |
| 2022 | 35,040 | 34,887 | 0 | 0 | **YES** | trained on **and** scored |
| 2023 | 35,040 | 0 | 35,040 | 0 | **YES** | calibrated on **and** scored |
| 2024 | 35,136 | 0 | 17,319 | 17,664 | **YES** | calibrated on **and** scored |
| 2025 | 35,040 | 0 | 0 | 35,040 | **YES** | scored repeatedly |
| 2026 | 21,163 | 0 | 0 | 21,163 | **YES** | scored repeatedly |

## Why 2012–2016 is not a holdout either

Those years sit before `SAMPLE_START` and were never trained on. They are still
disqualified, for two independent reasons:

1. **They influenced experiment design.** The coverage computations that ruled
   out a plain IV feature column and produced the residual-correction design
   (`iv-coverage-2`) were run across the full episode table, these years
   included. So did the funding-coverage measurement.
2. **The sample start is itself a modelling choice.** `SAMPLE_START = 2017-08-01`
   was selected to restrict to the modern-microstructure regime. Evaluating on
   the excluded era would be evaluating on data the pipeline was deliberately
   designed not to represent — a distribution-shift test, not a holdout.

## Why 2021–2026 is emphatically not a holdout

Every one of those years has been a scored walk-forward test fold in **at least
six** experiments on overlapping data: `E-anchor-verdict`, `E-blend`,
`E-blend-1state`, `E2-iv-correction`, `E2b-result`, `E2c-result`,
`E-power-result`, `E2-confirm-result`, `E2-confirm-wide`. Several of those
directly shaped later design decisions:

- 2023's volatility collapse produced the worst-fold guard that rejected the
  blend-weight rule (`E-blend-1state`).
- 2024's near-zero IV effect is visible in every per-fold table and informed how
  the shrinkage was read.
- The decision to drop `iv_level` (E2b → E2c) was made after seeing per-fold
  coefficient signs across 2022–2026.

A period used to choose a feature set cannot then test that feature set.

---

## What follows, per RULES.md R26

**A forward paper-trading holdout, frozen from the date below.** This is the only
honest option remaining.

    FREEZE DATE: 2026-08-28 (UTC)
    Holdout = observations with anchor timestamp strictly after the freeze.
    Last episode currently in episodes.parquet: 2026-08-09.

Conditions, fixed now:

1. **The candidate is frozen at the freeze date.** E2c's five features
   (`iv_chg_1h`, `iv_chg_6h`, `iv_chg_24h`, `iv_z_20d`, `ivrv_ratio`), no
   intercept, shrinkage `SIGMA_B = 0.10`, coefficients fitted on data up to the
   freeze only. Frozen means frozen: no refits, no feature changes, no
   re-tuning against holdout performance.
2. **One evaluation.** The holdout is scored once, when enough forward data has
   accumulated to clear the MDE (see below). Scoring it early and looking does
   not reset it.
3. **If it fails, it fails.** Per R26, a failed holdout is not re-run against a
   revised model and relabelled.
4. **The gap between 2026-08-09 and the freeze date is NOT holdout.** Those
   ~19 days exist only because `episodes.parquet` has not been rebuilt; they
   were available to be examined and so are disqualified by the same standard
   applied above.

### How much forward data is needed before it can be scored

Per R5, this must be stated before the fact rather than after. From
`E-power-result`, the binding constraint is the number of independent
year-scale regimes, not episodes: 24× more episodes bought 1.53× effective
sample size. The wide-slice effect is −6.18% pooled.

**This is the uncomfortable part, and it is stated plainly: at roughly one
independent regime-observation per year, a forward holdout capable of resolving
a 6% pooled effect on fold-level inference needs on the order of years, not
weeks.** A per-episode paired test on forward data is far better powered and is
the right primary for the holdout — but it measures sampling error, not the
between-regime variation that §26 showed dominates. Both will be reported, and
the fold-level one will be labelled underpowered until it is not.

*Educational research only. Not financial advice.*
