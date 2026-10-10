# Loop memory: NOCTUA fresh test and disproof audit (2026-10-10)

Read this first when resuming. trajectory.jsonl has every command, plans/ every
plan, evals/ every number, audits/ every auditor verdict verbatim, VERDICT.md
the conclusions. supervisor.json holds the stagnation state.

## Ground rules (do not relearn)
- Artifact under test: model/serve/noctua_v2.npz (pinned, fit before 2024-07-01).
  Never score 2024-07+ with noctua_v2_refreshed_2026-08-09.npz: it is in-sample.
- The 2024-07+ era is USED UP: scored by PR #15, by this run, and by 5 auditors.
  More baselines on it only add multiplicity (A4). New evidence = forward days only.
- Production slice for the product story: anchor 17:00 UTC, H = 19 h. But the
  LIVE workflow (fetch-data.yml) publishes every 30 min at the last closed hour.
- Served = raw committee sigma_mean x trailing-60d median(RV/sigma_med) over
  ALL hours (serve/adaptive.py). The trader uses the RAW median.
- Any vol baseline must include weekday terms. Calendar-blind HARs are straw men.
- Quote compounded returns; compare a trader to a matched-exposure constant,
  not just buy-and-hold (Sharpe is scale-invariant).

## Execution feedback (lessons)
- L1 install pyarrow and scipy==1.17.1 first (policy/runtime imports scipy).
- L2 api.binance.com is geo-blocked; data-api.binance.vision works; serve/fetch.py
  (Bitstamp/Coinbase) works for the hourly tail.
- L3 fresh_test.py takes ~11 min (per-row served safe_level): run in background.
- L4 My first baselines were calendar-blind, which made NOCTUA look 11-15 %
  better than it is (A1). Always include the weekday in the baseline.
- L5 When quoting breach counts, say which arm (raw or served); A2 flagged a
  mismatch that was only arm labelling.
- L6 evals/hours_through_now.parquet is git-ignored (7 MB); fresh_test.py
  rebuilds it exactly from btc_history + evals/hours_tail_live.parquet.

## Claims (final status)
| id | claim | status | by |
|---|---|---|---|
| C1/C2 | vol beats HAR baselines | DISPROVED vs weekday HAR (wins only vs calendar-blind) | A1, repro |
| C3 | beats the options market | yes vs DVOL alone; ties HAR5+DVOL | fresh_test |
| C4 | edge stable | raw yes vs blind HAR; loses to weekday HAR every half-year | A1 |
| C5 | tails calibrated | conservative at 17:00, near nominal pooled over hours | A2 |
| C6 | trader beats BH / vol-target | WEAKENED: 0.07 Sharpe, CI [-0.19,0.31], gone at 12 bp | A3 |
| C7 | direction no skill | confirmed | fresh_test |
| C8 | served 17:00 number worse than HAR | survives at 17:00; pooled over hours served beats raw | A0, A2 |
| C9 | NOCTUA adds info over weekday HAR | WEAKENED: +0.01-0.02, not robust to multiplicity | A4 |
| C10 | features.py:296 weekday bug (+4 should be +3) | confirmed | A0, A1 |

## Next (not started)
Fix C10, add weekday terms, retrain pre-2024-07, pre-register vs weekday HAR,
then score on days after 2026-10-10 only.
