# NOCTUA continuation — phased execution plan

One phase at a time, each with a gate that must pass before the next opens.
Phases are sized so a failure is attributable: a phase changes one class of
thing, never several at once.

## Why phases, and why one change at a time — evidence from this repository

The question "does fixing several things at once create new bugs?" has a
measured answer here, not a stylistic one:

- `runtime.has_mx()` tested for a key that could not exist, returning `False`
  for every artifact. It was **harmless only because nothing consumed `q_mx`**.
  Had the head and the artifact been "fixed" in the same pass, an unvalidated
  head would have shipped silently. It was caught because exporting an artifact
  was a separate, isolated step (§6o).
- `reg_post_etf` looks like pure defect — an untrained random offset live in
  production. Ripping it out on sight would have been churn: measured, it moves
  `sigma_med` by 0.169 % and P(touch 2 %) by 0.17–0.29 pp (§11b).
- The 32 σ-atom integration looked removable and improved things on 769
  episodes. On 18,463 it **reversed sign** and the proposed mechanism was
  backwards (§10). It is load-bearing.

That is Chesterton's fence three times over. The discipline that catches it is
not cleverness, it is **measure the thing before removing it, and change one
thing per experiment**.

## Phase gates

### Phase 0 — Inspect and reproduce (IN PROGRESS)
Nothing may be modified until the committed benchmark reproduces.

**Gate v1 — VOID, and the reason matters.** It read: re-run
`python -m model.eval.benchmark --folds all` at SHA `ab170b4` and reproduce
`model/artifacts/benchmark.json` within 0.5 % relative on every headline metric.
Executing it exposed three defects, all in the rule:

1. **`--folds all` does not exist.** `argparse` exits 2. The command as written
   never ran.
2. **`model/artifacts/` is entirely `.gitignore`d.** The "committed"
   `benchmark.json` is not in git at any SHA. The gate asked to reproduce a file
   that was never committed, whose only provenance is an mtime.
3. **Relative tolerance is meaningless near zero.** `DSC_up_0.5` reported a
   1421 % deviation on an absolute difference of 0.00059.

This is NOT the same situation as §10, §11c or §13, where rules that were
well-formed produced results I did not want and were left alone. **A rule
naming a nonexistent flag and a nonexistent file was never testable.** It was
not failed; it could not be run. Replacing it is legitimate, and the
replacement below was written while the determinism run was still executing and
its outcome genuinely unknown — which is what keeps it a pre-registration.

**What gate v1 did establish**, and it stands: `model/artifacts/benchmark.json`
was **stale**, predating the freshness fix that regenerated `features.parquet`
(reference mtime 2026-08-14, features mtime 2026-08-16). The signature is
decisive — `climatology`, which reads only `episodes.parquet`, is bit-identical
at 0.000 %, while `persistence`, a baseline with no neural training at all,
moved 3.16 %. A pure baseline moves only if the data moved. The reproduced
values (QLIKE 0.2896 / 0.3057 / 0.4332) match §6d's post-fix figures; the stale
reference matches its pre-fix figures.

**Gate v2 — pre-registered, replacing v1.** Reproducibility means *the pipeline
returns the same answer from the same inputs*, which is the only property that
gates later phases. So:

    python -m model.eval.benchmark        # run A
    python -m model.eval.benchmark        # run B, identical inputs

PASS requires, for every metric in every model: |A − B| ≤ **1e-9 absolute**, OR
relative difference ≤ **0.1 %** where |A| > 1e-4. The absolute floor exists
because v1's relative-only form produced a meaningless 1421 % on a 1e-5
quantity. Seeds are fixed at 0,1,2 and no input changes between runs, so
anything above these thresholds is genuine nondeterminism and **does** block
every later phase.

Reproducing the *stale* artifact is explicitly NOT a condition, because that
file describes a feature matrix that no longer exists and matching it would be
a defect, not a success.

Deliverables: `BASELINE_MANIFEST.md` (written against the FRESH numbers, not
the stale reference), leakage re-audit, and pinned dependency versions — the
audit found `torch` and `scikit-learn` absent from `requirements.txt`
entirely, so "reproducible from a clean environment" is currently unachievable
by construction.

### Phase 1 — Lineage and specification artifacts
`DATA_LINEAGE.md`, `FEATURE_CATALOG.md`, `TARGET_SPEC.md`,
`SPLIT_MANIFEST.json`, and an `EXPERIMENT_REGISTRY` exported from the existing
`research/ledger.json` rather than duplicated beside it.
Gate: every one of the 42 feature columns has a documented `event_time`,
`publication_time`, `feature_time` and source, or is marked UNVERIFIED.

### Phase 2 — Horizon and target extension
Current scope is H ∈ {6, 12, 19, 24} hours. The requested scope is intraday→1
week (short) and 1 week→3 months (medium), plus a three-class direction target
with an ex-ante neutral band.
**This is the largest scope change in the plan and it is honest about cost:**
a 3-month horizon at a 2017 start yields ~35 non-overlapping windows. The
embargo alone becomes 90 days. Gate: `TARGET_SPEC.md` must state, per horizon,
the number of non-overlapping episodes available, before any model is trained
on it. If a horizon cannot support inference, it is declared BLOCKED BY DATA
rather than modelled badly.

### Phase 3 — E1, volatility target estimator
Sampled RV vs bipower/continuous variation vs jump decomposition, noise-robust
candidate where 1-minute data permits. One estimator change, everything else
identical.

### Phase 4 — E2, implied-volatility family (DESIGN CONSTRAINED BY COVERAGE)
**UNBLOCKED (§15).** The paged `get_volatility_index_data` route now returns
47,563 hourly rows spanning 2021-03-24 → 2026-08-26, with **15,552 rows inside
the training window**. Implied volatility is trainable. This is the highest
expected-information-gain experiment in the queue, because §12 showed the onset
ceiling exists precisely for want of a forward-looking input — but it does not
open until Phase 0's gate passes.

**A plain feature column is inadmissible, and this is measured, not argued.**
Ledger `iv-coverage-2`, re-derived through `noctua.splits.time_splits`: DVOL
covers **32.7 %** of training episodes (62,051 of 189,831) and **100 %** of
calibration and test (52,359 and 73,867). DVOL began trading 2021-03-24 and the
training window opens 2017-08-01. A fill value for the missing two thirds would
be learnable in training and constant at test.

`eval/iv_correction.py` implements the admissible design instead: NOCTUA is left
untouched and an IV-conditioned **residual correction** is fitted on top of its
cached out-of-sample forecasts, so the failure mode is a correction of zero. The
QLIKE fit is globally convex and its convergence is asserted rather than hoped
for — §20 records a GARCH baseline whose fit silently returned library defaults
in four of six folds. The placebo arm rotates the IV series circularly inside
the covered era, which preserves the exact set of scored episodes, so the margin
between arms cannot be a difference in slice.

`iv_term_slope` is **BLOCKED**: DVOL is a single 30-day constant-maturity index,
so no second tenor exists to difference. Not built, and no synthetic proxy
substituted. Funding rate is verified usable (hourly, 2019-04-30 →, 50.1 % inside
the training window) and is testable now.

### Phase 5 — E15 redo, spike weighting on a slice-primary endpoint
Already pre-registered as `levers2`: spike-conditional QLIKE with a
moving-block bootstrap CI, spike RV/σ toward 1.0, calm QLIKE guard at 1 %,
pooled reported but **not** a condition. This exists because §13's rule made
pooled QLIKE primary for a 6 %-of-episodes treatment — the exact error the
protocol names.

### Phase 5.5 — E-power, INSERTED AHEAD OF EVERYTHING (BENCHMARK.md §22, §23)

This phase was not in the original plan and now blocks the rest of it.

`benchmark.run_fold` scores `fold["test"] & finite & production_mask`, and
`production_mask` is `(H == 19) & (anchor_hour == 17)`. **Every fold is decided
on ~365 episodes, ~20 of them spike-flagged** — six folds is ~2,190 test and
~119 spike episodes, against a 510,496-episode population.

Two independent, pre-registered tests of the ensemble weight (§23) both came
back with intervals too wide to decide anything, and on the second
`research/pitfalls.check_not_a_coin_flip` said so outright: the estimate was one
eighteenth of its own standard error. That is a **non-measurement**, not a null.
It would have looked identical if the true effect were +2 % or −2 %.

Running more experiments at that resolution produces more non-measurements. So
E-power comes first: re-decide §21 — the one experiment here with a large,
unanimous, already-resolved effect — on all 24 anchor hours at H = 19, through
the same code path (`anchor_freshness --all-hours`), and measure how much the
interval actually tightens against the √24 = 4.90× that independence would give.

The ratio is a reusable constant. It tells every experiment after this one how
much power a wider slice really buys, instead of leaving it assumed.

**Gate:** if §21's verdict flips on the wider slice, every prior verdict resting
on a marginal CI is re-run before any new experiment starts. If it holds and
only the interval narrows, the production slice was adequate for effects of that
size, and the multiplier fixes the size below which it is not — which then sets
the minimum detectable effect each later phase must be powered for.

**Horizons are deliberately not mixed.** H = 6/12/19/24 at one anchor are nested
inside one another; that overlap is not the serial dependence a moving-block
bootstrap models. Holding H = 19 and varying the anchor hour keeps the
dependence in the one form the estimator is built for.

### Phase 6+ — remaining queue, ordered by expected information gain
E3 microstructure, E4 global/panel, E5 HAR+nonlinear residual, E6 multi-task
heads, E7 nonlinear challengers, E10/E11 window and cadence, E12 loss ablation,
E13 calibration ablation, E14 drop-one-family ablations, E16 foundation models
as challengers only. Reordered after each phase by what the evidence table says
is most uncertain, not by this list's order.

## Standing rules carried from the existing harness

- No experiment is admissible without a rule fixed before the data is scored
  (`research/ledger.add()` refuses it).
- `research/pitfalls.py` runs against an experiment's own output **before** the
  result is written down.
- `research/supervisor.py` is consulted before proposing any experiment on a
  topic already attempted, and its redirection is acted on.
- No agent result enters the record without being re-derived from the data.
  That check has rejected four agent verdicts to date.

*Educational research only. Not financial advice.*
