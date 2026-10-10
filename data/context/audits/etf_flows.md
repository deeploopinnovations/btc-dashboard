# Audit: data/context/etf_flows.parquet (Farside US spot BTC ETF net flows)

Audited 2026-10-10. Repo: /home/user/btc-dashboard. No data file, fetch script, or git state was modified.

## Verdict: PASS WITH NOTES

No lookahead found. The D+1 12:00 UTC convention is safe against every publication time I could find (first-release posts land about 03:47 UTC on D+1). Values match the live Farside table cell for cell and match independent reports on the dates checked. Two minor defects and one provenance gap are listed below. None of them leaks future information.

## Summary of findings

| # | Question | Result | Status |
|---|---|---|---|
| 1 | Is the D total complete by D+1 12:00 UTC? | Yes on every sampled day. First-release batches (total and IBIT together) appeared at 03:47-04:48 UTC on D+1. Sample is small (4 timestamps). | CONFIRMED for sample; SUSPECTED for the full history |
| 2 | Does Farside revise past days? | No revision found in the 4 sampled days. Revisions over the full history cannot be ruled out because no vintage archive exists. | SUSPECTED (unprovable here) |
| 3a | 5 dates vs independent reports | 5 of 5 agree (see table). | CONFIRMED |
| 3b | Weekends and holidays have no rows | Weekends: none. Holidays: 12 holidays absent, but 16 holidays present with total 0.0. | CONFIRMED defect (D2) |
| 3c | "-" cells not turned into 0 | 893 raw "-" cells equal 893 missing cells. No "-" became 0. | CONFIRMED |
| 4 | Does etf_flows() use anything after t? | No. 952 decision times tested, 0 changes under future-row deletion or poisoning. | CONFIRMED |
| 4b | Feature naming | etf_net1_bn uses a 2-day window, not 1 day. | CONFIRMED defect (D1) |

## 1. When is the D total complete?

Method: Farside's X account (@FarsideUK) posts each day's flows with timestamps. The status-ID timestamp (Twitter Snowflake: `(id >> 22) + 1288834974657` ms) was decoded and checked against the visible "Posted" time for the 2026-10-10 posts, and they agree.

| Flow day D | Evidence | Publication time (UTC) | Matches parquet? |
|---|---|---|---|
| 2025-11-11 | X status 1988468803613720778: TOTAL 524, IBIT 224.2, FBTC 165.9, ARKB 102.5 | 2025-11-12 04:48:01 (= 23:48 EST D) | Yes: 524.0, 224.2, 165.9, 102.5 |
| 2025-10-06 | CoinDesk "U.S. BTC ETFs See $1B Inflows" (Oct 7 2025): $1.2bn total, IBIT $970m, "according to Farside data" | Article published 2025-10-07 09:06 UTC | Yes: total 1205.2, IBIT 970.0 |
| 2026-09-17 | X post "Posted: Fri, 18 Sep 2026 03:47:03 GMT", TOTAL 159.5, IBIT 183.7, FBTC -16.6 | 2026-09-18 03:47 UTC (= 23:47 EDT D) | Yes: 159.5, 183.7, -16.6 |
| 2026-10-09 | X posts 2108766307256602725 to 2108766396767211658 (ETH, BTC, IBIT, TOTAL 21.1, IBIT 22.4) | 2026-10-10 03:47:21 to 03:47:42 GMT | Yes: 21.1, 22.4 |
| 2026-10-05 to 10-09 | X weekly summary, status 2108830082588406015: TOTAL -678.9, IBIT 1.1 | 2026-10-10 08:00:46 GMT | Yes: sums of parquet rows = -678.9 and 1.1 |

Reading: Farside posts one batch per day, with IBIT and the total included, at about 03:47 UTC on D+1 (23:47 ET on D). That is about 8 hours before the convention's available_ts (D+1 12:00 UTC), and about 13 hours before the 17:00 UTC decision. The "IBIT reports next morning" premise is not supported for these days.

Limits: four timestamps is a small sample. The X page scrape shows only the latest posts. No historical posting log exists. I found no report of a late IBIT figure for any day, so "late IBIT" stays SUSPECTED, not confirmed.

Aggregator note (SUSPECTED, not a Farside defect): satsintel.io (search snippet, snapshot time unknown) lists Oct 8 2026 total -238.6, while Farside's total is -244.1. The difference is exactly IBIT's -5.5, and the same page still shows IBIT's latest day as Oct 7 (-207.7). This suggests a secondary snapshot taken before IBIT's Oct 8 figure was loaded. That supports the idea that IBIT can arrive late, but the timing is unknown, so it is not proof. Satsintel's totals also differ from Farside in other rows, because they omit the BTC column (Oct 1: 88.1 = 102.7 - 14.6; Oct 6: 129.8 = 118.8 + 11.0; Sep 22: 709.7 = 714.7 - 5.0). Bitbo.io was not usable as a check: its per-fund values differ from Farside (Oct 7 IBIT -100.7 vs -207.7).

Conclusion for Q1: the convention is safe. The ~8 h margin (about 7 h against the latest observed post, 04:48 UTC) covers every timestamp found. The residual risk is an unobserved late IBIT update after 17:00 UTC on D+1, which I could not find evidence for. See fix F4 for an optional extra margin.

## 2. Revisions

- Live Farside table (fetched 2026-10-10 via Firecrawl scrape, 705 dated rows, 13 columns) parsed with the repo's own `etf_flows.parse()`. Compared with the parquet: 0 missing rows on either side, 0 value differences across all 8,272 rows and values (tolerance 0.5 US$). The parquet was built at 11:58:48 UTC on 2026-10-10 and still matches the live table.
- Post-release values match the table for the three sampled days (2025-11-11, 2026-09-17, 2026-10-09) and for the 2026-10-05 to 10-09 weekly sum. No revision was seen.
- Limit: the table is a single current snapshot. Nothing in the repo records earlier vintages, and no archive or changelog was found. A revision of an old day cannot be excluded for the full history, so tagging the series as `lagged` (which implies no revision) is an assumption, not a verified fact.
- Farside's page says only: "The above table is generated automatically, in real time. Farside Investors is not liable for any errors or inaccuracies in the data." It documents no revision policy.

## 3. Data checks

### 3a. Five dates vs independent reports

| Date (D) | Parquet total (US$m) | Parquet IBIT (US$m) | Independent report | Agrees |
|---|---|---|---|---|
| 2024-01-11 | 655.3 | 111.7 | WSJ: "Bitcoin ETFs Drew $655 Million in Net Inflows on Their First Day" | Yes (total) |
| 2024-11-07 | 1373.8 | 1119.9 | Cointelegraph via TradingView: "$1.37 billion recorded on Nov. 7" | Yes (total, rounded); IBIT not checked |
| 2025-02-25 | -1113.7 | -164.4 | CryptoSlate: "The Feb. 25 outflow of $1.1 billion ... According to Farside Investors data" | Yes (total, rounded) |
| 2025-10-06 | 1205.2 | 970.0 | CoinDesk, published 2025-10-07 09:06 UTC: "$1.2 billion ... led by IBIT with $970 million ... according to Farside" | Yes (total and IBIT) |
| 2026-10-07 | -484.9 | -207.7 | CryptoSocial (2026): "$484.9M on October 7, 2026, per Farside"; SatoshiNakamoto.TV: "IBIT -$207.7M" | Yes (total and IBIT) |

The 2026-10-07 check is the most recent. Its two independent sources were published after the flow day. IBIT was checked only where a source quotes it (2025-10-06, 2026-10-07).

### 3b. Weekends and holidays

- Weekday counts of total rows: Mon 138, Tue 143, Wed 143, Thu 141, Fri 140. Weekend rows: none.
- Business days with no row (12): 2025-07-04, 2025-09-01, 2025-11-27, 2025-12-25, 2026-01-01, 2026-01-19, 2026-02-16, 2026-04-03, 2026-05-25, 2026-06-19, 2026-07-03, 2026-09-07. All are NYSE holidays.
- Holidays WITH a row (16): 2024-01-15, 2024-02-19, 2024-03-29, 2024-05-27, 2024-06-19, 2024-07-04, 2024-09-02, 2024-11-28, 2024-12-25, 2025-01-01, 2025-01-09 (national day of mourning, NYSE closed), 2025-01-20, 2025-02-17, 2025-04-18, 2025-05-26, 2025-06-19. Each has total = 0.0 and no fund values. Farside shows them as zero-total rows. The loader's docstring says only "-" totals are skipped, so these rows are kept. Defect D2.
- Trading days lost: none. Every trading day has a row.

### 3c. "-" cells were not turned into 0

- Raw Farside table: 3,655 literal "0" cells and 893 literal "-" cells (`parse_num` returns None for "-").
- Parquet: 3,655 zero-valued fund cells (matches the raw "0" count exactly). Missing cells = 705 days x 12 funds - 7,567 numeric = 893, equal to the raw "-" count.
- Totals equal the sum of the parsed funds on all 689 days that have fund values (max absolute difference 1.2e-7 US$).

## 4. Feature code (model/context/features.py)

- `etf_flows(ctx, t)` (lines 142-144) calls `_window_sum(ctx, "etf_netflow_usd_total", t, k)` with k = 5D and 2D. `_window_sum` (lines 39-52) sums rows with `available_ts` in [t-k, t), using `searchsorted(side="left")`. Both bounds are strict or inclusive exactly as in `pit.asof`. Nothing at or after t is read.
- Boundary: the 2026-10-09 row (available 2026-10-10 12:00:00 UTC) enters at 12:00:01, not at 12:00:00. Verified.
- Leakage test (CONFIRMED): decision times at 17:00 UTC every day from 2024-03-01 to 2026-10-08 (952 times). For each t:
  - (a) I removed all rows with available_ts >= t, then recomputed features. Result: 0 changes.
  - (b) I multiplied every value with available_ts >= t by 1000, then recomputed. Result: 0 changes.
- Hand check: at 2026-10-09 17:00 UTC, etf_net5_bn = -0.700 = Oct 5 + Oct 6 + Oct 7 + Oct 8 rows (-89.8 + 118.8 - 484.9 - 244.1). etf_net1_bn = -0.729 = Oct 7 + Oct 8 rows only.
- Defect D1 (naming, not leakage): `etf_net1_bn` uses `2 * D`, so it is a two-day window that usually holds two rows. It is not one day.
- Minor note: `build()` sets `_present` from `notna`. A window with no rows gives 0, not NaN, so `present` = 1 even when nothing was published in the window. This did not occur in the data (no gap longer than 5 days).
- Consumer check: the only consumer is features.py. The evaluation file `data/context/evaluation/etf_flows.json` reports `train` coverage 0.0 and verdict UNTESTABLE. That is a modelling limit, not a data defect.

## Defects and proposed fixes

Only fixes are proposed here. None were applied, since the audit scope excluded modifying data or the fetch script.

**D1 (CONFIRMED, minor, naming): etf_net1_bn is a 2-day window.**
Evidence: features.py line 144, `_window_sum(..., 2 * D)`; hand check above.
Fix: rename the column to `etf_net2_bn` (keep the 2D window), or change the window to `D` if a one-day sum is wanted. Update the feature list in `data/context/evaluation/etf_flows.json` to match.

**D2 (CONFIRMED, minor, data): 16 holiday rows with total 0.0.**
Evidence: section 3b. These rows have zero fund values and are not trading days.
Fix: in `sources/etf_flows.py`, `parse()`, drop a day when it has no numeric fund cell, or drop NYSE holidays explicitly. Then rebuild the parquet and re-run `pit.validate`. Value-level impact on sums is zero, so no feature changes.

**D3 (CONFIRMED, provenance): pit_quality "lagged" implies the values were never revised, which is not verified.**
Evidence: section 2. Only a single current snapshot exists.
Fix: either tag the series `revisable` in `to_frame()` (SPEC rule 5), or start an archive: save each raw Farside HTML under a dated path with its `retrieved_at`. Future vintages can then be checked and tagged `exact`. Historical rows cannot be fixed this way.

**F4 (optional hardening, not a defect): extra margin before the decision time.**
Current: available_ts = D+1 12:00 UTC. Observed first release: 03:47-04:48 UTC on D+1.
Option: set available_ts = D+1 17:00 UTC. With the strict `<` rule, a row then first informs the decision at D+2 17:00 UTC, which costs one day of freshness. Use this only if the late-IBIT risk in section 1 is judged material.

## Process notes

- Live curl of farside.co.uk returned HTTP 403. I used a Firecrawl scrape instead. The scrape result was saved to a tool-results file, and I copied the markdown into the session scratchpad.
- An attempt to read /root/.ccr/README.md (proxy docs) was refused by the auto-mode permission classifier. I did not work around it.
- A stray file, /home/user/scratch_farside.html (a 403 page from my curl test), was written outside the repo. It was removed in a follow-up step.
- Not done, per instructions: no git commit or push, no package installs, no hearthbot tool calls, no edits to data or the fetch script.
