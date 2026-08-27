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

**Pre-registered gate, fixed before the run:** re-running
`python -m model.eval.benchmark --folds all` at SHA `ab170b4` must reproduce
`model/artifacts/benchmark.json` with every headline metric within **0.5 %
relative** (QLIKE, pinball, CRPS, and the Brier/DSC/MCB family at every
barrier). Seeds are fixed at 0,1,2, so a larger deviation means the pipeline is
not deterministic — which is itself a finding and blocks every later phase,
because no experiment's delta is interpretable on a non-reproducible base.

Deliverables: `BASELINE_MANIFEST.md`, leakage re-audit.

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

### Phase 4 — E2, implied-volatility family
**UNBLOCKED (§15).** The paged `get_volatility_index_data` route now returns
47,563 hourly rows spanning 2021-03-24 → 2026-08-26, with **15,552 rows inside
the training window**. Implied volatility is trainable. This is the highest
expected-information-gain experiment in the queue, because §12 showed the onset
ceiling exists precisely for want of a forward-looking input — but it does not
open until Phase 0's gate passes. Funding rate is verified usable (hourly, 2019-04-30 →, 50.1 % inside
the training window) and is testable now.

### Phase 5 — E15 redo, spike weighting on a slice-primary endpoint
Already pre-registered as `levers2`: spike-conditional QLIKE with a
moving-block bootstrap CI, spike RV/σ toward 1.0, calm QLIKE guard at 1 %,
pooled reported but **not** a condition. This exists because §13's rule made
pooled QLIKE primary for a 6 %-of-episodes treatment — the exact error the
protocol names.

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
