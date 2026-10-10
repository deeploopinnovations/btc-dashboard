# Lookahead audit: data/context/attention.parquet

Auditor: adversarial review, 2026-10-10. Scope: `model/context/sources/attention.py` (read only, not run),
`model/context/pit.py`, `model/context/features.py::attention()`, `model/context/SPEC.md`, and the
parquet. No data file, fetch script or git state was modified. The audit read the copy at
`/tmp/claude-0/-home-user-btc-dashboard/a653e416-9fde-5630-b5ae-17dc7a498e81/scratchpad/attention_nogdelt.parquet`
(11,408 rows: `fear_greed` 3,170; `wiki_views_bitcoin` 4,119; `wiki_views_cryptocurrency` 4,119). GDELT is ignored.
Scratch scripts: `scratchpad/audit_att1.py`, `audit_att2.py`, `audit_att3.py`.

## Verdict: FAIL

- Wikipedia rows (`wiki_views_*`): `available_ts = D+1 06:00 UTC` is not supported by any source and is
  contradicted by the Wikimedia Cloud FAQ ("generally takes a full 24 hours to populate, sometimes longer").
  This is a SUSPECTED lookahead defect in the rows as stamped.
- `attention()` silently never uses the Wikipedia series (CONFIRMED). Current feature values are therefore
  not affected by the Wikipedia timing, but the defect becomes live as soon as the feature code is fixed.
- Fear & Greed (`fear_greed`) path: PASS WITH NOTES. Publication is a scheduled 00:00 UTC release (CONFIRMED),
  and no lookahead in `attention()` (CONFIRMED by perturbation test). The +1 h lag is thin but harmless
  for the 17:00 UTC decision.

## Summary

| # | Area | Status | Severity |
|---|---|---|---|
| F1 | `attention()` silently drops `att_wiki_rel` (series-name bug) | CONFIRMED | medium (latent leak once fixed) |
| F2 | Wikipedia `available_ts` = D+1 06:00 is earlier than documented latency | SUSPECTED (strong docs support) | high if the Wikipedia feature is enabled |
| F3 | No lookahead in `attention()` for the F&G features (perturbation test) | CONFIRMED | none |
| F4 | F&G is published at 00:00 UTC; `available_ts` = +1 h is valid but thin | CONFIRMED schedule; exact job lag SUSPECTED | low |
| F5 | F&G history not revised in 5 archived data points | CONFIRMED for 5 points; earlier methodology changes SUSPECTED | low |
| F6 | Dataset holds one vintage per series (no as-known history) | CONFIRMED | medium (for honest backtests) |
| F7 | F&G rows missing for 2018-04-14..16 and 2024-10-26 | CONFIRMED | low (no lookahead) |
| F8 | `att_fng` `max_age = 3D` is looser than a daily series needs | CONFIRMED | low |
| F9 | Bitcoin 2026-10-09 count 12,625 is 5.7x the prior days | CONFIRMED value; cause unknown | low (data quality) |
| F10 | F&G `event_ts` = D 00:00 (end of the window it describes) vs SPEC rule 3 (D+1 00:00) | CONFIRMED labelling difference | none (labelling only) |

## F1. CONFIRMED: Wikipedia feature is silently dropped

`features.py::attention()` gates the Wikipedia feature on `ctx.series.str.contains("pageviews")`. The series
names are `wiki_views_bitcoin` and `wiki_views_cryptocurrency`, which never contain "pageviews".

Command (re-runnable, read only):
```
/usr/bin/python3 -I -c "import sys; sys.path.insert(0,'/home/user/btc-dashboard/model'); \
import pandas as pd, numpy as np; from context import features; \
ctx=pd.read_parquet('<scratch>/attention_nogdelt.parquet'); \
print('pageviews substring matches:', ctx.series.str.contains('pageviews',na=False).sum()); \
print(list(features.attention(ctx, np.array([1791662400])).columns))"
```
Output:
```
pageviews substring matches: 0
['att_fng', 'att_fng_chg7']
```
So `att_wiki_rel` is never produced, and the attention group carries no Wikipedia signal, with no error.

## F2. SUSPECTED: Wikipedia `available_ts` = D+1 06:00 is too early

Fetch script: `WIKI_LAG = 6 * 3600`; `event = day + 86400` (D+1 00:00); `avail = event + WIKI_LAG`.
The parquet confirms it: every Wikipedia row has `available_ts - event_ts = 6 h` (checked by `audit_att1.py`).

Evidence on when a day's count is published:
- Wikimedia Cloud, Pageviews Analysis FAQ, https://pageviews.wmcloud.org/siteviews/faq (fetched 2026-10-10):
  "The Wikimedia pageviews API generally takes a full 24 hours to populate, sometimes longer. In some situations
  you may see data missing for yesterday's date as well, which will be left blank rather than showing a count of zero views."
- en.wikipedia, "Wikipedia:Pageview statistics", https://en.wikipedia.org/wiki/Wikipedia:Pageview_statistics
  (fetched 2026-10-10): "Pageview stats for the previous day will usually be available by 5 a.m. UTC, although this
  varies considerably depending on how long it takes to clean the data." This is a community page about the tool, not an API spec.
- Wikitech, Data Platform/Data Lake/Traffic/Pageviews: hourly dumps; no per-day latency stated.
- The AQS docs (doc.wikimedia.org analytics-api) state no latency either.
- The API returns only completed days: the request for daily counts 20261008..20261010 returned items through
  2026100900 and nothing for 2026101000 (run at 12:3x UTC). That shows the partial current day is withheld,
  not when a completed day is published.

Impact if the 24 h figure holds: the count for day D is published about D+2 00:00 UTC. The stamp puts it at
D+1 06:00. A 17:00 UTC decision on day T would use the count for T-1 (stamped T 06:00), which in reality
appears at T+1 00:00, about 7 h after the decision. Under the 24 h figure, `att_wiki_rel` would leak roughly
18 h of availability time.

Not measured: the actual publication time. The API response carries no timestamps, and no archive of the
pageviews API is reachable from here (web.archive.org CDX is unreachable). A forward test is needed (see Fix C).

Stored values are current: the Bitcoin and Cryptocurrency daily counts returned by the API at 12:33 UTC match the
stored rows (Bitcoin 2026-10-08 2206, 2026-10-09 12625; Cryptocurrency 2026-10-08 1482, 2026-10-09 1426).
This shows the values are stable a few hours after the fetch. It does not show they were complete at D+1 06:00.

## F3. CONFIRMED: no lookahead in the F&G path of `attention()`

Perturbation test: for 2,840 decision times at 17:00 UTC (2019-01-01 .. 2026-10-10), the features computed
from the full parquet were compared with the features computed after dropping every row with
`available_ts >= t`. Result: 0 mismatches.

Hand check at 2026-10-10 17:00 UTC: `att_fng` = 0.64 (stamp 2026-10-10 00:00 value 64, available 01:00, 16 h before
the decision). `att_fng_chg7` = -0.03 = (64 - 67)/100, with 67 the stamp 2026-10-03 00:00 value.

Across all 17:00 decision times, the hours from the decision to the newest usable F&G row are min 16, median 16,
max 40 (max = a missing-day gap). The `pit.asof` `side="left"` cut is correct.

Caveat: the perturbation test covers the cut inside `attention()` only. It cannot test upstream publication timing (F4).

## F4. CONFIRMED schedule; exact job lag SUSPECTED. F&G `available_ts` = +1 h is valid but thin

The F&G API and page both say the index updates once per day, and the countdown lands exactly on 00:00 UTC.

- Live API, `https://api.alternative.me/fng/?limit=3&format=json`, 2026-10-10 12:28 UTC: entry stamped
  1791590400 (= 2026-10-10 00:00 UTC) has `time_until_update` 41494. Two polls 20 s apart (12:30:31 and 12:30:51):
  `time_until_update` 41369 and 41349. Both project to 2026-10-11 00:00:00 UTC.
- Archived API, 2023-10-24 21:56:39 UTC (`web.archive.org/web/20231024215639/https://api.alternative.me/fng/`):
  stamp 1698105600 (= 2023-10-24 00:00 UTC), value 66, `time_until_update` 7401. Projects to 2023-10-25 00:00:00 UTC.
- Archived API, 2025-06-02 16:06:32 UTC (`web.archive.org/web/20250602160632/...`): stamp 1748822400 (= 2025-06-02
  00:00 UTC), value 64, `time_until_update` 28409. Projects to 2025-06-03 00:00:01 UTC.
- Archived page, `alternative.me/crypto/fear-and-greed-index/` (2025-06-02 08:12 UTC) states "The next update will happen
  in ..." and "Each day, we analyze...".

Conclusion: the stamp is the scheduled publication time (00:00 UTC), and `available_ts = stamp + 1 h` is valid only if the
job runs within 1 h of midnight. That is not documented, so the job lag is SUSPECTED, not measured.

Decision impact: a 17:00 UTC decision sees the stamp-D row 16 h after it is published. Raising the lag to +12 h
(noon UTC) changes no feature value at the 17:00 decision and tolerates a 12 h publication slip. That is the
recommended fix (Fix A).

Content window: the stamp describes "the previous 24 h" per the code comment. The page does not say whether the
00:00 computation uses data up to 00:00 or later. SUSPECTED, not testable from here.

## F5. CONFIRMED for 5 archived data points; earlier methodology changes SUSPECTED. No revision seen

Archived page, 2025-06-02 08:12 UTC: "Now" 64, "Yesterday" 56, "Last week" 73, "Last month" 65.
Stored values: 2025-06-02 = 64, 2025-06-01 = 56, 2025-05-26 = 73, 2025-05-03 = 65 (from `audit_att2.py`). Match.
Archived API, 2023-10-24 stamp value 66; stored 2023-10-24 = 66. Match.

The page says "Surveys (15%) currently paused" and that Twitter sentiment is live, so the composition has changed
over time. Whether the history before a method change is what was published at the time is SUSPECTED and
untested here. The current pit_quality `lagged` does not say that. SPEC rule 5 says to tag such series `revisable`.

## F6. CONFIRMED: no as-known vintage in the dataset

Every row in each series has the same `retrieved_at` (2026-10-10 12:01 UTC for the scratch copy). Each run
(`fetch_json` then `df.to_parquet(OUT)`) rewrites the whole history with the values the API returns that day.
So the file records no value as it was seen at its own `available_ts`; history is the latest vintage, and the
`pit.validate` check (`available_ts <= retrieved_at + 1 day`) cannot catch a revision.
The docstring claim "No lookahead: a row is written only when available_ts <= the fetch time" is necessary but not
sufficient: it proves the row was fetched after `available_ts`, not that it existed then.

## F7. CONFIRMED: F&G gaps (no lookahead)

Missing F&G days in the parquet (`audit_att2.py`): 2018-04-14, 2018-04-15, 2018-04-16, 2024-10-26. The feature code
falls back to older rows; `max_age` then governs. Not a lookahead issue. Note only.

## F8. CONFIRMED (minor): `att_fng` `max_age = 3D` is too loose

The series is daily. A 3-day tolerance accepts two missed publications silently. Two days (48 h) covers one missed
publication with margin.

## F9. CONFIRMED value, cause unknown (data quality)

Bitcoin daily count for 2026-10-09 is 12,625 against 2,199 to 2,267 for the previous five days (about 5.7x). The
Wikimedia FAQ notes that spikes can be automated traffic and that anomaly investigations can take time, which is
relevant to revisions (F6). The `user` agent filter does not fully remove this. Not lookahead.

## F10. CONFIRMED labelling difference (no lookahead)

SPEC rule 3 puts daily aggregates at `event_ts = D+1 00:00`. The F&G series puts the stamp at `D 00:00` (the end of
the window it describes). The Wikipedia rows follow SPEC. Only `available_ts` decides leakage, so this is labelling.
It should be written down in SPEC or the source docstring.

## Fixes (not applied; proposed)

Fix A, `model/context/sources/attention.py`:
```python
WIKI_LAG = 36 * 3600        # was 6 h. Wikimedia FAQ: "generally takes a full 24 hours to populate, sometimes longer"
FNG_LAG = 12 * 3600         # was 1 h. Index stamped 00:00 UTC; 17:00 decisions unchanged
...
avail = ts + FNG_LAG        # in fng_rows()
```
Cost, stated plainly: with 36 h, a 17:00 UTC decision on day T uses the Wikipedia count for T-2, not T-1. That
is one day staler. The F&G change costs nothing at 17:00. Also tag both as `revisable` (SPEC rule 5) until a
vintage is captured.

Fix B, `model/context/features.py::attention()`:
```python
WIKI = "wiki_views_bitcoin"
if (ctx.series == WIKI).any():
    F["att_wiki_rel"] = _logchg(ctx, WIKI, t, 7 * D, max_age=4 * D)
F["att_fng"] = _a(ctx, "fear_greed", t, max_age=2 * D) / 100.0
F["att_fng_chg7"] = _diff(ctx, "fear_greed", t, 7 * D, max_age=2 * D) / 100.0
```
Add a regression test that `att_wiki_rel` is present when the series exists, and rerun the perturbation test
(`scratchpad/audit_att3.py`) after the lag change.

Fix C, forward test for Wikipedia publication time (not run; needs a few days of calls): on each of days D+1, query
the daily count for day D at 06:00, 12:00, D+2 00:00 and D+2 12:00 UTC, and record when it first matches the final
value. Use the Wikimedia rate-limit guidance (the API returned 429 on back-to-back requests here).

Fix D, longer term: append-only vintages. Store each run's raw responses with `retrieved_at`. Let `asof` choose
the vintage with `retrieved_at < t` for revisable series. Until then, the history is not safe for an honest backtest.

## Not done / limits

- The fetch script was not run (it rewrites the parquet and writes /tmp/gdelt_cache). The F&G rows were checked by
  reading the parquet only.
- web.archive.org CDX and most snapshot endpoints are unreachable from here. Archive evidence is limited to three
  F&G API snapshots (2023-10-24, 2025-06-02), one F&G page snapshot (2025-06-02), and the live API.
- The actual Wikimedia publication time for day D was not measured (F2 is SUSPECTED).
- The F&G job's exact run time and content window are not documented (F4 is SUSPECTED).
