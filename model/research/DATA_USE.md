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

**Reporting note added 2026-09-28 (does not change the frozen candidate or its
primary).** `P4-iv-served-pe-result` found that, scored against a baseline
that carries serving's trailing level factor, E2c's walk-forward effect is not
distinguishable from zero on any metric: its headline was mostly a level
effect the served system already has. When this holdout is scored, the
comparison against the served baseline (serving's factor applied to both arms)
must be reported beside the primary, so a forward "win" is not a re-measurement
of that level effect.

## The freeze is CANDIDATE-SPECIFIC, and a second candidate does not inherit it

Added 2026-09-22, because the question came up while the dispersion correction
was being prepared for adoption and the answer is not the one a reader would
assume.

Condition 1 above names the frozen candidate: **E2c's five features, no
intercept, `SIGMA_B = 0.10`, coefficients fitted to the freeze**. "Frozen means
frozen: no refits, no feature changes." That is a statement about one specific
model, not a reservation of the forward window for whatever candidate happens
to be ready when enough data accumulates.

So `P3-dispersion-barriers-result`'s ADVANCE **cannot be confirmed on this
holdout**. Scoring a different candidate against it would either violate
condition 1 or silently redefine what was frozen, and relabelling after the
fact is what R26 exists to prevent. The same applies to any other candidate
that arrives later.

What a second candidate would need, stated so nobody has to invent it at the
moment of adoption:

1. **Its own freeze date**, which cannot be earlier than the day the candidate
   was fixed — for the dispersion λ that is no earlier than 2026-09-22.
2. **Its own accumulation period before scoring.** The arithmetic in the
   section above does not improve for being applied twice: at roughly one
   independent regime-observation per year, resolving an effect of this size on
   fold-level inference takes years. A per-episode paired test on forward data
   is far better powered and would be the primary, labelled for what it
   measures.
3. **No borrowing.** Two candidates frozen on two dates are two holdouts, and
   the second one's window starts later and is therefore shorter.

The consequence is uncomfortable and is the point of writing it down: **there is
no untouched data on which the dispersion correction can be confirmed today.**
Adoption, if it happens, would rest on walk-forward evidence plus the serving
contract in `tests/test_dispersion_report.py`, with that limitation stated
rather than papered over — and ADVANCE remains the honest verdict until a
forward window of its own exists and has been scored once.

## Second candidate: the clock-aware anchor, frozen 2026-09-27

Added 2026-09-27 under the three conditions above, on the day the candidate
was fixed (`P4-hour-anchor-result`, ADVANCE).

    FREEZE DATE: 2026-09-27 (UTC)
    Holdout = production nights (17:00 UTC anchor, H = 19) with anchor
    timestamp strictly after the freeze.

**What is frozen** -- the two arrays in `model/serve/noctua_v2.npz`, written
by `noctua/add_hour_anchor.py` and never refitted:

    season_profile   sha256(bytes) 836813fb13ec4769...   (24 values)
    har_beta_season  sha256(bytes) 27daa2be75a28cb6...   season coef +1.24151

together with the arithmetic in `noctua/season.py` and the rest of the
artifact unchanged. No refits, no profile updates, no coefficient changes.

**The comparison is paired and needs no second deployment.** The clock-blind
anchor is the same artifact with `HOUR_ANCHOR` off, which
`tests/test_hour_anchor.py` asserts is bit-identical to the pre-candidate
payload. So every forward night can be forecast both ways from one file,
whichever setting is live, and the two are scored on the same realised
outcome.

**Primary, stated before any forward night exists:** the per-episode paired
QLIKE difference (candidate minus clock-blind, both through serving's own
trailing factor) over holdout nights, block bootstrap, one evaluation. It
measures sampling error, not between-regime variation, and will be labelled
so. **Secondary:** the sign of the median log(RV / sigma_med) at 17:00 under
each, since the mechanism claim is that the candidate moves it toward zero.
The barrier battery is reported and labelled underpowered at any horizon
under a year.

**How and when it is scored:** `python -m model.eval.forward_hour_anchor`.
It forecasts each post-freeze 17:00 night both ways through
`serve/predict.forecast` itself and labels it with `episodes.build_episodes`.
**N_MIN = 300 nights**, fixed before any forward night exists: at the
walk-forward noise level a 95 % interval narrows below the walk-forward effect
at about 290. Below N_MIN the script prints the count and nothing else; at
N_MIN it scores once and writes `research/forward_hour_anchor_result.json`,
after which it refuses to rescore. That is roughly ten months of nights.

**Secondary metrics amended 2026-09-28, before any forward night exists**
(`P4-hour-anchor-exante-result`, R89): the all-night per-episode far-barrier
Brier, and per-episode Brier / log score / far-barrier Brier on nights flagged
EX ANTE as HOT (top decile of `har_6h - har_22d`) or HIGH (top decile of
`har_1d`), thresholds from the holdout nights themselves. No outcome-selected
subset is scored.

**Switch status, 2026-09-28: OFF.** The owner delegated the decision; a
pre-registered ship-blocker (`P4-hour-anchor-cond-result`) fired on
outcome-selected spike nights. Its design was flawed (R89) and the proper
checks favour the candidate, but a gate that fires is not relitigated. The
switch is reconsidered when this holdout is scored.

**What this freeze does NOT do:** it does not decide whether the flag is on.
Adoption, if it happens before the holdout is scored, rests on the
walk-forward evidence and the serving gate, with that limitation stated -- the
same position the dispersion correction is in.

## Third candidate: the true-weekend anchor, frozen 2026-09-29

Added 2026-09-29, the day the candidate was fixed (`P4-weekend-fix-result`,
ADVANCE; artifact increment committed in `c4f8974`).

    FREEZE DATE: 2026-09-29 (UTC)
    Holdout = production nights (17:00 UTC anchor, H = 19) with anchor
    timestamp strictly after the freeze.

**What is frozen** -- two arrays in `model/serve/noctua_v2.npz`, written by
`noctua/add_weekend_anchor.py` and never refitted, with the arithmetic in
`noctua/calendar.py` and the rest of the artifact unchanged:

    har_beta_weekend         sha256(bytes) 03ca1419345447ef...  [+0.03711, -0.13026]
    har_beta_weekend_season  sha256(bytes) 4b59300962e77815...  [+0.03713, -0.13026]

The coefficient is from the artifact's training split (through 2023) and is
smaller than the walk-forward's recent folds (-0.26 to -0.29): the holdout
tests the increment AS BUILT, not a refit on recent data.

**Paired, one file.** `WEEKEND_ANCHOR` off is asserted bit-identical to the
pre-candidate payload (`tests/test_weekend_anchor.py`), so every forward night
is forecast both ways from the same artifact, `HOUR_ANCHOR` off (the served
base at the freeze).

**Primary, stated before any forward night exists:** per-episode paired Brier
of the served barrier curves (shipped minus candidate), block bootstrap, one
evaluation. Walk-forward effect +0.000301 per night. **Reported:** per-episode
log score, far-barrier (+/-5%) Brier, QLIKE on sigma_mean (labelled
underpowered, ~1,400 nights needed), and Brier on EX-ANTE weekday classes
(Fri/Sat/Sun anchors; Mon-Thu). No outcome-selected subset.

**How and when:** `python -m model.eval.forward_weekend_anchor`. **N_MIN = 450
nights**, fixed now: at the walk-forward noise (95% half-width 0.000137 at
2,046 nights, block 38) the interval narrows below the effect at about 425.
Count only below N_MIN; one scoring at N_MIN into
`research/forward_weekend_anchor_result.json`; refuses to rescore. Roughly
fifteen months.

**Overlap with the clock-aware holdout.** Both holdouts use the same forward
nights. They answer different questions (each is scored against the shipped
anchor with the OTHER flag off), and neither's result may be used to amend the
other's design.

**Switch status, 2026-09-29: OFF.** Adoption before the holdout is scored would
rest on the walk-forward evidence and the audit, with that limitation stated.

## Fourth candidate: the day-of-week anchor, frozen 2026-09-29

Added 2026-09-29 (`P4-dow-anchor-result`, ADVANCE -- it replaces the weekend
increment as the calendar candidate; the third candidate's freeze stands and
is scored on its own terms).

    FREEZE DATE: 2026-09-29 (UTC); production nights strictly after it.

**What is frozen** -- two arrays in `model/serve/noctua_v2.npz`, written by
`noctua/add_dow_anchor.py` and never refitted, with `noctua/calendar.py`:

    har_beta_dow         sha256(bytes) a10cfe4816204035...  [intercept, Mon..Sat]
    har_beta_dow_season  sha256(bytes) e1053f4c9aff7f38...

Fitted on the artifact's split (through 2023). Frozen coefficients kept 0.82
of a yearly refit's Brier gain over 2023-2026, 0.62 in the latest years
(`eval/dow_staleness.py`): the holdout tests the increment AS BUILT.

**Paired, one file**, `DOW_ANCHOR` off vs on, every other anchor flag off
(`eval/forward_dow_anchor.py`, sharing `forward_weekend_anchor`'s scorer).
**Primary:** per-episode Brier of the served barrier curves. Walk-forward
effect +0.000444 per night. **Reported:** log score, far-barrier Brier, QLIKE,
Brier by ex-ante weekday class. **N_MIN = 450**: the walk-forward noise alone
needs ~200 nights, but at 0.67 of the effect (staleness) ~450.

**Caveat, stated at the freeze:** the candidate was designed after
`P4-weekend-calendar-perm` on the same walk-forward years. This holdout is
the first evidence that did not shape it.

**Switch status, 2026-09-29: OFF.** `DOW_ANCHOR` and `WEEKEND_ANCHOR` are
alternatives (both on raises).

*Educational research only. Not financial advice.*
