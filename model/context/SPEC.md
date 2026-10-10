# Point-in-time context dataset: record contract

Every file in `data/context/` (except `explain_only/`) is a long-format table
with exactly these columns. `model/context/pit.py` validates them.

| column | type | meaning |
|---|---|---|
| `source` | str | short source id, e.g. `fred`, `fomc`, `binance_metrics` |
| `series` | str | series id within the source, e.g. `DGS10`, `fomc_decision_upper` |
| `event_ts` | int64 | Unix seconds UTC: the moment the value *refers to* (end of the bar, the meeting, the print) |
| `available_ts` | int64 | Unix seconds UTC: the **earliest moment a trader could have read this value**. Conservative: when unsure, later. |
| `value` | float64 | the number (NaN allowed only when `text` carries the content) |
| `text` | str | optional free text (headline, label); empty string otherwise |
| `source_url` | str | URL the row (or its batch) was fetched from, or the archived page it was read from |
| `retrieved_at` | int64 | Unix seconds UTC when we fetched it |
| `pit_quality` | str | one of `exact` (source publishes the release time), `lagged` (we added a conservative publication lag), `scheduled` (a calendar entry known in advance; `available_ts` = when the schedule was public), `revisable` (source may have revised history; today's value may differ from what was seen then) |

## Rules

1. `available_ts >= event_ts` for every row except `pit_quality == "scheduled"`
   (a calendar entry is known before it happens; there `available_ts` is when
   the schedule was published, conservatively Jan 1 of the event's year or
   later).
2. A model may use a row at decision time `t` only if `available_ts < t`.
   `pit.asof()` is the only join that should be used.
3. Daily aggregates for calendar day D (UTC) have `event_ts` = D+1 00:00 UTC
   (the end of the day) and `available_ts` >= that plus the source's
   publication lag.
4. US-market daily closes (FRED) for day D: `event_ts` = D 21:00 UTC,
   `available_ts` = D+1 12:00 UTC unless the source documents earlier
   publication.
5. Revised series are tagged `revisable`; if a first-release vintage exists
   (ALFRED), prefer it and tag `exact`/`lagged`.
6. Hindsight explanations ("BTC fell because ...") go only in
   `data/context/explain_only/`. They are never features. `pit.load_all()`
   refuses to read that directory.
