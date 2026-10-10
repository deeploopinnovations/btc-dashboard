# Audit: data/context/macro_daily.parquet

Auditor: adversarial lookahead review. Scope: `model/context/sources/macro_daily.py`, `model/context/pit.py`, `model/context/features.py` (`macro_daily` group), `data/context/macro_daily.parquet` (41,074 rows, fetched 2026-10-10 11:57 UTC). No data file, fetch script, or git state was modified. Scratch scripts and raw outputs live in the session scratchpad (`alfred_probe.py`, `probe_out.txt`, `holiday_probe.py`, `repro.py`, `feat_check.py`).

## Verdict: FAIL

One used feature (`mac_oil_chg5`, built from DCOILWTICO) reads prices up to 7 days before they were published. Two more publication-timing defects and one SPEC rule 5 violation are confirmed. Everything else checked is clean or conservative.

## Confirmed defects

### C1. DCOILWTICO is available 1 to 7 days before publication (CONFIRMED, used by features)

- Code: `avail = utc_ts(d + 1 day, 12:00)`, i.e. D+1 12:00 UTC, `pit_quality="lagged"`.
- FRED series page (fetched today): Release "Spot Prices" (EIA), next release Oct 15, 2026. The publication lag itself is what ALFRED confirms below; the weekly batch cadence is SUSPECTED, not verified.
- ALFRED first-vintage probe on 10 random observations (2015 to 2025), `probe_out.txt`: first vintage containing the observation is 2 to 8 days after D. Examples: obs 2025-06-17 first in vintage 2025-06-25 (lag 8); obs 2016-06-27 first in vintage 2016-06-29 (lag 2); obs 2020-03-25 first in vintage 2020-04-01 (lag 7). All 10 samples leak.
- Reproduce:
  ```
  python3 repro.py   # DCOILWTICO obs 2025-06-17
  #   vintage 2025-06-18 absent | 2025-06-24 absent | 2025-06-25 75.62
  ```
- Feature impact: `features.macro_daily` uses `_logchg(ctx,"DCOILWTICO",t,5D)`. A model at 2025-06-19 17:00 UTC sees the 06-17 price, which EIA had not published until 06-25.
- Fix: derive `available_ts` per observation from ALFRED first-release vintages (fetch `alfredgraph.csv?id=DCOILWTICO&vintage_date=V` for V = D+1..D+10 and take the first V containing D; store `available_ts = V 12:00 UTC`, tag `exact`). Interim conservative rule until that is built: `available_ts = D + 9 days 00:00 UTC` (covers the 8-day maximum seen). Re-run and re-validate.

### C2. DTWEXBGS holiday weeks are available one day early (CONFIRMED; next occurrence is Oct 12, 2026)

- Code: `next_monday_after(D)` at 21:30 UTC. When that Monday is a US federal holiday, H.10 is published the following Tuesday.
- ALFRED, `holiday_probe.py` output (obs Friday, check Monday/Tuesday vintages):
  ```
  Labor Day 2025      obs 2025-08-29 | 2025-09-01:absent 2025-09-02:120.6028
  Columbus Day 2025   obs 2025-10-10 | 2025-10-13:absent 2025-10-14:121.5217
  MLK Day 2026        obs 2026-01-16 | 2026-01-19:absent 2026-01-20:120.4478
  Presidents Day 2026 obs 2026-02-13 | 2026-02-16:absent 2026-02-17:117.5258
  Memorial Day 2026   obs 2026-05-22 | 2026-05-25:absent 2026-05-26:119.2868
  control (no holiday) obs 2026-06-12 | 2026-06-15:119.5073
  ```
- Forward-looking: FRED DTWEXBGS page (fetched today) reads "Next Release Date: Oct 13, 2026". The code would mark the Oct 9, 2026 observation available Oct 12 21:30 UTC, which is 24 h early. The Oct 12 (Columbus Day) holiday is the same case.
- Fix: available = first business day (Mon to Fri, excluding US federal holidays, e.g. `pandas.tseries.holiday.USFederalHolidayCalendar`) on or after the Monday following D, at 21:30 UTC. Equivalently, use the ALFRED first-vintage probe from C1 for every row.

### C3. DTWEXBGS uses the current revised vintage, not first release (CONFIRMED; SPEC rule 5)

- Source URL is `fredgraph.csv`, which returns the current vintage only. SPEC rule 5 says prefer the ALFRED first-release vintage where one exists.
- Sample of 9 valid observations (of 10; ALFRED graph endpoint returns empty for vintages before about 2019 for this series), `probe_out.txt`:

  | obs | first release | current | abs diff |
  |---|---|---|---|
  | 2019-08-12 | 116.8645 | 116.7548 | 0.110 |
  | 2019-10-17 | 116.4023 | 116.2615 | 0.141 |
  | 2020-02-14 | 116.4479 | 116.4223 | 0.026 |
  | 2020-09-24 | 117.9406 | 117.5473 | 0.393 |
  | 2020-10-02 | 117.1309 | 116.7321 | 0.399 |
  | 2020-10-07 | 116.7486 | 116.3447 | 0.404 |
  | 2021-04-15 | 112.9025 | 112.6181 | 0.284 |
  | 2024-07-05 | 123.8192 | 123.4762 | 0.343 |
  | 2025-07-22 | 120.1036 | 119.6810 | **0.423** |

- Max absolute difference 0.423 index points (0.35%). Mean 0.28. Every first release is higher than the current value, so the revision is systematic. A 7-observation median |log change| of DTWEXBGS is 0.53% (`feat_check.py`), so this revision bias is about half a typical `mac_dxy_chg7` move.
- Fix: replace the fetch with per-observation first-release values from ALFRED (`alfredgraph.csv?vintage_date=` probe as in C1), tag `exact` or `lagged`. Keep `revisable` only for the pre-2019 gap.

### C4. Feature defect, not lookahead: negative WTI price produces a 30-sigma feature (CONFIRMED)

- DCOILWTICO 2020-04-20 = -36.98 (only non-positive value in the file). `features._logchg` clips to 1e-12, so `mac_oil_chg5` = -30.625 on 2020-04-21 (`feat_check.py` output). It returns -0.8 on 04-22 and -0.29 on 04-23.
- Fix: in `_logchg`, return NaN where `now <= 0` or `then <= 0` (and in `macro_daily`, use a level or a difference for oil instead of a log change).

### C5. DFF is available one or more days early (CONFIRMED, not used by features)

- Code: D+1 12:00 UTC for every calendar day, including Saturdays and Sundays (1,541 of 5,395 DFF rows are weekend days).
- ALFRED: obs Friday 2019-06-14 first appears in vintage Monday 2019-06-17 (`repro.py`: vintage 06-15 absent, 06-16 absent, 06-17 2.36). Holiday example obs 2023-01-22 first appears 2023-01-24 (MLK Monday 01-23 shifted).
- Fix: `available_ts` = next business day (skip weekends and federal holidays) at 14:00 UTC, or the ALFRED first-vintage probe. Not used by `features.py` today, so no model impact yet.

## Suspected (not reproduced)

- S1. DFF intraday time: 12:00 UTC on the next business day is about 08:00 EDT. The NY Fed publishes the EFFR around 9 am ET. ALFRED only has day granularity, so this could not be resolved. Fix with C5 (14:00 UTC).
- S2. DefiLlama backfill: the stored snapshot matches a fresh fetch exactly (`llama_now.json`, 3,237 overlapping days, max relative diff 0.0), so there is no evidence of change yet. The risk is that DefiLlama recomputes past totals when coins are added or corrected, and a full overwrite destroys the first-seen values. Fix: append-only daily snapshots with `retrieved_at`, never re-fetch history into the same file, and use first-seen values.
- S3. DefiLlama label off by one day: a snapshot with timestamp D 00:00 UTC is the boundary state at the start of D, but is labelled as end of day D (`event = D+1 00:00`). This is conservative (no leak); it wastes a day. Fix: label `event = D 00:00`, keep `available = event + 1 day + 6h`.
- S4. SP500 first-release timing cannot be checked: ALFRED returns no vintages for SP500 (empty response for 2024-06-03, 2026-10-01). FRED restricts S&P series to 10 years of history (the file starts 2016-10-10). `lagged` (D+1 12:00 UTC) is consistent with the FRED close convention and is conservative for a 17:00 UTC decision.
- S5. VIXCLS: ALFRED shows first vintage on day D itself for several samples (e.g. 2019-09-04, 2020-07-17), so the close was public on D. Our D+1 12:00 UTC is conservative. No defect.

## Checked and clean

- **Publication timing, other FRED daily series** (ALFRED first vintage, 5 samples each): DGS10, DGS2, T10YIE, NASDAQCOM, VIXCLS first appear on day D or D+1 to D+3 (weekend/holiday). Where the first vintage is D+1 itself (VIXCLS 2015-06-22, DGS10 2024-09-04, DGS2 2024-09-04), publication was on D+1 at an unknown hour, so the 12:00 UTC cutoff cannot be verified at day granularity. No change proposed. Every other sample's first vintage is D+1 or later, so no leak is shown there.
- **Revisions, other series**: DFF (10 samples), DCOILWTICO (10), BAMLH0A0HYM2 (10): first release equals current in every sampled case (max abs diff 0.0). NASDAQCOM max diff 7.8e-5 (float rounding). VIXCLS and DGS10/DGS2 zero diffs.
- **BAMLH0A0HYM2**: first vintage equals the observation day, so `available = D+2 12:00` is conservative (wastes about a day). Not used by features. FRED notes the series is restricted to 3 years of observations from April 2026, so a full re-run truncates history.
- **Timezone check**: VIXCLS 2020-03-16 = 82.69, `event_ts` = 2020-03-16 21:00 UTC, `available_ts` = 2020-03-17 12:00 UTC (pass; row is in the file as expected). Event times are fixed UTC; no DST logic. 21:00 UTC is one hour late during EDT (close is 20:00 UTC) which is harmless because availability is the next day.
- **Schema**: no duplicate (series, event_ts); `pit.validate` passes; 4 rows have `available_ts` up to 3 minutes after `retrieved_at` (within the 1-day tolerance; harmless).
- **Feature code, lookahead**: `pit.asof` uses `searchsorted(side="left") - 1` on `available_ts`, which is the strict `available_ts < t` join. `_logchg`, `_diff`, `_a` all go through `asof`. `_window_sum` is not used by `macro_daily`. Perturbation test (`feat_check.py`): for 120 random 17:00 UTC anchors, multiplying every row with `available_ts >= t` by 3 and adding 1000 changed zero features. The features are correct given the `available_ts` they receive; the leaks are in the timestamps themselves (C1, C2, C5).
- **Unused series**: DFF, T10YIE, BAMLH0A0HYM2 and `stablecoin_mcap_usdt` are not read by `features.macro_daily`, so C5 and S2 do not reach a model yet.

## Proposed fix order

1. C1 (DCOILWTICO): ALFRED-based `available_ts` or the D+9 days interim rule. Affects a live feature.
2. C2 (DTWEXBGS holiday shift) before the Oct 12, 2026 holiday, because the next observation (Oct 9) is affected.
3. C3 (first-release DTWEXBGS values, `exact`/`lagged`).
4. C4 (non-positive price guard in `_logchg`).
5. C5 and S2/S3 (DFF business-day rule; append-only DefiLlama snapshots).

After fixing, re-run the ALFRED probes in this report and the perturbation test, and record the new `available_ts` rule in SPEC.md.
