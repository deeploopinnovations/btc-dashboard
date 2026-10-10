# Audit: explain_only causes (150 largest BTC daily moves)

Scope: `causes_part_aa.csv`, `causes_part_ab.csv`, `causes_part_ac.csv` (50 rows each, 150 total), checked against `top_moves_input.csv`. Dates and ret_pct match the input row-for-row (0 mismatches, 0 duplicate dates).

Row numbers below are 0-based positions in the concatenated aa, ab, ac order.

## Method

- **Sample:** `random.seed(7)`, `random.sample(range(150), 30)`, sorted. Indices: 5, 6, 9, 12, 14, 15, 17, 18, 22, 23, 24, 28, 31, 38, 50, 54, 57, 61, 71, 73, 74, 82, 93, 101, 107, 108, 111, 129, 130, 137.
- **Verdict rules:**
  - SUPPORTED: cited page read (directly or via search excerpt), dated within 2 days of the move, and supports the core cause. Minor unsupported wording is noted but does not change the verdict.
  - WEAK: cited page exists but is dated more than 2 days from the move, gives a different day or magnitude, or only partly supports the cause. Also used where the cited page returned HTTP 403 and only corroborating coverage was found (rows 5, 6, 9, 12, 17, 22, 31, 61, 73, 93). Those verdicts are provisional and could move to SUPPORTED with direct access.
  - WRONG: cited page is readable and does not support the stated cause.
  - UNVERIFIABLE: no source cited, or nothing found.

## 1. Sample table (30 rows)

| Row | Date | ret_pct | Cited source (published) | Verdict | Finding |
|---|---|---|---|---|---|
| 5 | 2018-02-05 | -16.08 | business-standard / Reuters (2018-02-06) | WEAK | Page 403. Reuters and AFP coverage ties the regulatory fears to the Tuesday 6 Feb plunge, so the day may be off by one. |
| 6 | 2018-04-12 | +14.16 | cnbc.com (2018-04-12) | WEAK | Page 403. Same-day Reuters coverage says no cause could be identified. Short-squeeze framing comes from a syndicated copy. |
| 9 | 2018-07-17 | +8.77 | fxstreet (2018-07-19) | WEAK | Page 403, read via search excerpt. Published 2 days after. Credits BlackRock, Mastercard and IBM news plus a short squeeze, which conflicts with "no clear news". |
| 12 | 2018-10-11 | -5.60 | sharecast (undated) | WEAK | Page 403, undated. Bloomberg same-day coverage supports the global equity rout and margin-call selling. |
| 14 | 2018-11-19 | -14.82 | siliconangle (2018-11-19) | SUPPORTED | Attributes the drop to the Bitcoin Cash fork fight. The phrase "hash war" is not in the article. |
| 15 | 2018-11-20 | -8.10 | fortune.com (2018-11-26) | WRONG | Published 6 days after the move. Names no day and no cause, and does not mention Bitcoin Cash. |
| 17 | 2018-11-28 | +11.91 | business-standard / Reuters (2018-11-28) | WEAK | Page 403. Date matches. The "no specific trigger" claim could not be checked. |
| 18 | 2018-12-20 | +10.57 | siliconangle (2018-12-20) | WEAK | Article cites a Tether reserves report and investor sentiment. It does not mention short covering, which is the summary's main cause. |
| 22 | 2019-05-13 | +11.98 | business-standard (2019-05-14) | WEAK | Page 403. Reuters (republished) says BTC passed $8,000 on 13 May with no clear cause. This is the same URL cited for row 21, a different day. |
| 23 | 2019-05-19 | +12.95 | coindesk.com (2019-05-26, updated 2021) | WEAK | The $6,600 low is dated 17 May in the article, not 19 May. Published 7 days after. The no-cause claim is consistent. |
| 24 | 2019-06-26 | +9.95 | businesslive, redirects to businessday (2019-06-26) | SUPPORTED | Reuters content links the rally to Facebook's Libra and safe-haven demand. It calls Libra a "token", not a stablecoin. |
| 28 | 2019-10-23 | -6.92 | theblock.co (2019-10-23) | SUPPORTED | Five-month low near $7,450 and about $205M liquidated on BitMEX. The article does not say the liquidated positions were longs. |
| 31 | 2020-01-14 | +8.74 | bloomberg.com (2020-01-14) | WEAK | Page 403. Syndicated Bloomberg and Reuters copies tie the rise to a breakout and the CME options launch. |
| 38 | 2020-11-05 | +10.20 | none | UNVERIFIABLE | No source cited. |
| 50 | 2021-05-12 | -13.00 | ktvz.com (2021-05-19) | WEAK | Published 7 days after. The article covers Tesla suspending bitcoin payments (not purchases) and a 12% fall. It does not describe a week-long sell-off. The 12 May date appears only in the URL. |
| 54 | 2021-09-07 | -11.08 | substack roundup (2021-09-14, not in CSV) | WEAK | The page supports the Huobi large-order and Lanzhou mining-inspection claims, but it was published 7 days after. CSV source_published is blank. |
| 57 | 2021-11-26 | -8.82 | euronews.com (2021-11-26) | SUPPORTED | Ties the roughly 8% fall to Omicron fears. The liquidation claim is not in the article. |
| 61 | 2022-02-17 | -7.62 | bloomberg.com (2022-02-17) | WEAK | Page 403, read via search excerpt. Body ties the drop to Russia-Ukraine invasion fears. |
| 71 | 2022-11-08 | -9.91 | fortune.com (2022-11-10) | WEAK | Article says Binance withdrew its FTX deal. It does not mention a CoinDesk Alameda report, a Binance FTT sale, or an ~80% FTT fall. The large drop is dated 9 Nov. |
| 73 | 2022-11-10 | +10.54 | bloomberg.com (2022-11-10) | WEAK | Page 403. Syndicated copies tie the rebound to US CPI cooling more than forecast. |
| 74 | 2023-01-12 | +5.15 | theblock.co (2023-01-12) | SUPPORTED | Links the rise to the December CPI report. The "cooler than expected" framing is not in the article. |
| 82 | 2023-03-13 | +9.13 | tickmill.com (2023-03-13) | SUPPORTED | Monday rebound after the Fed backstop (BTFP). The "stealth QE" wording is not in the article. |
| 93 | 2023-10-23 | +10.24 | bloomberg.com (2023-10-23) | WEAK | Page 403. Syndicated copies support ETF optimism, the DTCC listing and the Grayscale ruling. |
| 101 | 2024-03-04 | +8.26 | forklog.com (2024-03-04) | SUPPORTED | Same-day article: rally above $65k, $246M in 24h liquidations, no named trigger. The "two-year high" claim is not in the article. |
| 107 | 2024-05-20 | +7.73 | none | UNVERIFIABLE | No source cited. |
| 108 | 2024-07-04 | -5.15 | theblock.co (2024-07-04) | SUPPORTED | Mt. Gox moved about 47,229 BTC; creditor-sale fears reported. |
| 111 | 2024-08-05 | -7.10 | coindesk.com (2024-08-05) | SUPPORTED | Bank of Japan hike and yen carry unwind; over $1B liquidated. |
| 129 | 2025-09-25 | -3.81 | none | UNVERIFIABLE | No source cited. |
| 130 | 2025-10-01 | +4.04 | none | UNVERIFIABLE | No source cited. |
| 137 | 2026-01-20 | -4.59 | cointelegraph.com (2026-01-22) | WEAK | Ties the drop to the Greenland tariff threat. The article's chart dates the reaction to 21 Jan and does not tie it to 20 Jan. |

**Sample counts:** SUPPORTED 9, WEAK 16, WRONG 1, UNVERIFIABLE 4 (total 30).

Note: 10 of the 30 sampled cited pages (rows 5, 6, 9, 12, 17, 22, 31, 61, 73, 93) returned HTTP 403, so their WEAK verdicts are provisional.

## 2. Rows flagged for date problems

**(a) Source published more than 1 day before the move:**
- Row 0 (2017-10-12, +12.92): source is dc.fortune.com dated 2017-10-09, 3 days early. It is pre-move coverage that does not explain the move.

**(b) Summary places the move on a different day, or contradicts the row:**
- Row 103 (2024-03-16, -6.14): summary says the sell-off was on 15 March and the source dates it one day earlier.
- Row 105 (2024-03-20, +9.65): summary notes that reports describe a 20 March drop, which conflicts with this +9.65% rally row.
- Row 116 (2025-02-24, -4.89): cited report is dated the following day (2025-02-25).
- Row 23 (2019-05-19, +12.95): source puts the $6,600 low on 17 May.
- Row 137 (2026-01-20, -4.59): source ties the reaction to 21 Jan.
- Row 148 (2026-09-18, +5.92): summary says sources conflict on the day's move and cite a gain of about 1.75%.
- Row 71 (2022-11-08, -9.91): source places the large drop on 9 Nov.

**(c) Sources published well after the move (more than 2 days; not a step-2 trigger but affects reliability):**
- Row 15 (6 days), row 21 (3 days), row 23 (8 days), row 50 (7 days), row 68 (10 days, published 2022-08-29 for a 2022-08-19 move), row 92 (7 days), row 98 (6 days), row 144 (20 days, published 2026-09-08 for a 2026-08-19 move).
- Row 54 (7 days, fetched; CSV date blank).

**Magnitude conflicts noted in summaries:**
- Row 117 (-5.09): source reports a 3.6% move.
- Row 143 (-4.34): summary cites an intraday slide near 6%.
- Row 148 (+5.92): source cites about 1.75%.

## 3. Empty source_url

- **34 of 150 rows (22.7%)** have an empty source_url: rows 1, 11, 19, 25, 26, 34, 35, 38, 39, 45, 49, 52, 53, 59, 76, 90, 91, 95, 100, 105, 107, 112, 120, 125, 126, 128, 129, 130, 132, 133, 135, 136, 139, 141.
- Of these, 6 rows carry a causal category other than no_clear_catalyst, while citing nothing: row 38 (other), row 120 (equity_market_risk_off), rows 128, 133, 139 and 141 (leverage_liquidation). Their cause fields are unsourced.
- Separately, 21 rows have a source_url but a blank source_published (rows 7, 10, 12, 42, 54, 69, 79, 86, 94, 102, 110, 115, 117, 119, 121, 123, 134, 143, 145, 146, 148), so their timing cannot be checked from the CSV.

## 4. Overall reliability statement

Reliability is low. Of the 30 sampled rows, only 9 (30%) are fully supported with a same-window source. The main failure modes were:
- Causes that are directionally plausible but contain unsupported specifics (wrong day, unsourced numbers, or qualifiers such as "stealth QE", "two-year high" or "cooler than expected").
- Sources published days or weeks after the move.
- Blank or unreachable citations.

One sampled row (15) is WRONG. Its cited page does not support the stated cause.

The 34 no-source rows and the 21 undated-source rows should not be read as established causes. Treat the cause_category field as a hypothesis, not a fact. Before use, the 7 flagged date conflicts should be corrected, and the 6 unsourced non-"no clear catalyst" rows should be sourced or relabelled.

Limits: 10 sampled pages were blocked (403), so their WEAK verdicts rest on search excerpts. Sample size is 30 of 150, so the rates carry roughly +/-17 points of uncertainty at 95% confidence. No CSV was edited.
