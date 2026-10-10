# NOCTUA fixes after the disproof audit (2026-10-10)

Follow-up to `runs/noctua-disproof-2026-10-10/` (PR #18). Every number here is
on 2024-07+ data, which is **not out of sample**: that era was used to find
these bugs. New evidence has to come from days after 2026-10-10.

## What was fixed

| finding | fix | where |
|---|---|---|
| weekday bug: `cal_weekend_frac` counts Fri+Sat (`+4`, should be `+3`) | new column `cal_weekend_frac_ss` (true Sat+Sun) read by every newly trained model, baseline and gate; legacy column kept so `noctua_v2.npz` stays bit-identical (audit A5 1c) | `noctua/features.py`, `spec.py`, `baselines.py`, `committee.py`, `train_v2.py` |
| live correction estimated over all hours, so the 17:00 window ran ~15-20 % high | factor estimated on the served hour (24 h stride, matched by timestamp so a gap cannot shift the hour) | `serve/adaptive.py` |
| live history too short for that window (34 of 60 same-hour episodes) | bundle 400 -> 460 days, back-filled from the training history (identical on the 9,600 overlapping hours for close) | `serve/history.py`, `data/noctua_history.parquet` |
| test script: direction base rate used test data, no weekday HAR, no matched-exposure control, served replica ignored the feature filter | fixed; reproduces every earlier number exactly | `runs/noctua-disproof-2026-10-10/fresh_test.py` (`--artifact`, `--tag`, `--served`) |
| pre-registered P4-weekend-fix script: arms would now read the new column; Stage B trained on the non-shipping target | arms pinned to the column they were registered on; shipped Stage B reference passed to every run | `eval/weekend_fix.py` |
| research scripts reading the legacy column | `eval/lag.py`, `eval/hour_anchor_cond.py`, `eval/vol_matrix.py` | |
| docs sold the calendar-blind win; trader numbers not reconciled | caveats and re-score added | `model/README.md`, `model/policy/README.md`, `model/AUDIT.md` |

## Scores, 17:00 UTC, H=19, 831 nights 2024-07..2026-10 (QLIKE, lower is better)

| forecast | v2 (shipped) | v3 (retrained, true weekend) |
|---|---|---|
| raw median | 0.243 | 0.236 |
| published (mean x same-hour factor) | 0.234 | 0.227 |
| published, old all-hour factor | 0.300 (VERDICT) | |
| weekday HAR, 2018-24 fit | 0.226 | 0.226 |
| weekday HAR, 2023-24 fit | 0.195 | 0.195 |
| Log-HAR (calendar-blind) | 0.286 | 0.286 |

- The live-correction fix is the big one: at 17:00 the published number goes
  from 23 % worse than the raw model to 4 % better.
- v3 improves on v2 by about 3 %, and ties the 2018-24 weekday HAR
  (-0.3 %, CI crosses 0). It still loses to the 2023-24 weekday HAR by 17 %.
- The weekday miscalibration is **not** fixed by the column: median RV/forecast
  on Fri/Sat/Sun anchors is 0.70/0.74/1.21 for v3 vs 0.69/0.69/1.22 for v2. The
  weekend share of the forward window cannot tell a Saturday night from a
  Sunday night that runs into the Monday open; the anchor's weekday can.
- Direction: still no skill (AUC 0.50).

**Decision:** serving stays on `noctua_v2.npz`; the trader and forward log stay
pinned to it. `noctua_v3.npz` is committed as a candidate for the forward
holdout, not switched on, because its gain is small and measured on the used-up
era.

## Audit of the fixes

`audits/A5_code_fixes.md` (Haiku auditor, told to break the fixes). It found a
gap that shifted the served hour, the short live history, research scripts on
the legacy column and the weekend_fix Stage B target; all four were fixed and
re-tested. Still open: the MIN_EPISODES gate is a step (factor 1.0 at 19
episodes, full correction at 20), now rarely reached with 60 episodes; and the
same-hour factor is worse at 00-13 UTC (up to 0.02 QLIKE) while better at
14-23 UTC.

Outputs: `evals/adaptive_same_hour.json` here, and
`runs/noctua-disproof-2026-10-10/evals/fresh_test_{v2,v3}_samehour.json`.
