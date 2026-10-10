# Audit: data/context/macro_calendar.parquet

Auditor role: adversarial. Goal was to disprove "no lookahead".
Target: `data/context/macro_calendar.parquet` (817 rows), built by `model/context/sources/macro_calendar.py`.
Feature code: `model/context/features.py::macro_calendar()`, `model/context/pit.py` (`asof`, `next_scheduled`).
No data file or fetch script was modified. Probe scripts live in the session scratchpad
(`probe_features.py`, `impact.py`), not in the repo.

## Verdict: FAIL (narrow)

The feature code does not read values from after t. The defects are in the schedule rows:
release dates that were moved after Jan 1 are stored at their post-hoc date with `available_ts` = Jan 1,
and a cancelled FOMC meeting is absent, so its absence reveals the cancellation early.
Both change `cal_*_hours_to` / `cal_*_in_hold` at specific decision times. Scope is about 6 CPI/NFP releases
and 1 FOMC meeting out of about 800 rows. The fix is small.

## Findings

| ID | Status | Severity | Summary |
|---|---|---|---|
| F1 | CONFIRMED | Medium | Moved CPI/NFP releases (6 events, 2025-26 lapses) are stored at the actual date, with `available_ts` = Jan 1 of the event year. The public schedule before Jan 1 was different. |
| F2 | CONFIRMED | Low | The 2020-03-17/18 FOMC meeting (cancelled) has no row. Its absence reveals the cancellation before it was public. |
| F3 | CONFIRMED (coverage, not lookahead) | Low | Oct-2025 CPI and NFP were cancelled and have no rows. Oct-2025 NFP first print is dropped by the joint-release rule. |
| F4 | PASS | - | Release times and DST handling (08:30 ET, 14:00 ET, 2020 emergency times). |
| F5 | PASS (5 of 8 verified) | - | First-print values and vintage dates for the random sample. |
| F6 | PASS | - | `macro_calendar()` value features use nothing at or after t. |
| F7 | SUSPECTED | - | Whether the Fed/BLS published next year's calendar by Jan 1 each year (not verified against a dated publication). |

### F1 (CONFIRMED): moved releases carry post-hoc dates

Evidence, BLS lapse notice (fetched with Firecrawl, https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm, last modified 2026-02-12):

| Release | Original date | Revised date | Stored in dataset |
|---|---|---|---|
| CPI Sep-2025 | Wed 2025-10-15 | Fri 2025-10-24 | cpi_release 2025-10-24 |
| NFP Sep-2025 | Fri 2025-10-03 | Thu 2025-11-20 | nfp_release 2025-11-20 |
| CPI Oct-2025 | Thu 2025-11-13 | canceled | no row |
| NFP Oct-2025 | Fri 2025-11-07 | canceled | no row |
| NFP Nov-2025 | Fri 2025-12-05 | Tue 2025-12-16 | nfp_release 2025-12-16 |
| CPI Nov-2025 | Wed 2025-12-10 | Thu 2025-12-18 | cpi_release 2025-12-18 |
| NFP Jan-2026 | Fri 2026-02-06 | Wed 2026-02-11 | nfp_release 2026-02-11 |
| CPI Jan-2026 | Wed 2026-02-11 | Fri 2026-02-13 | cpi_release 2026-02-13 |

The Sep-2025 CPI reschedule notice (https://www.bls.gov/bls/092025-cpi-reschedule-notice.htm) is dated 2025-10-10.
BLS FAQ: "Publication of September data was delayed by more than a week (from October 15 to October 24)."

Every `cpi_release` / `nfp_release` row has `pit_quality = scheduled` and `available_ts` = Jan 1 of the event year.
SPEC rule 1 says that is when the schedule was public. For the rows above it was not: the new date was published later.

Probe output (`scratchpad/audit/impact.py`, UTC 17:00 decision times):

```
2025-10-05 17:00 cpi dataset hours_to 451.5 | original-schedule hours_to 235.5
2025-09-20 17:00 nfp dataset hours_to 1460.5 | original-schedule hours_to 307.5
2025-11-01 17:00 cpi dataset hours_to 1124.5 (next = Dec 18)
```

At 2025-10-05 the public schedule said CPI on 2025-10-15, and the feature said 2025-10-24. The 2025-10-24 date
was not published until 2025-10-10. The NFP case at 2025-09-20 has the same problem.
The ALFRED vintage dates for these releases are correct (see F5), so the problem is only the scheduled rows.

Also SUSPECTED: decision times in the shutdown windows where the trader did not yet know about a move
(for example 2025-10-02 17:00 UTC, when the Oct 3 NFP was still on the schedule) get in_hold = 0 in the dataset.
The notice dates for the other five moves are not verified in this audit.

Fix (F1):
1. Build `cpi_release` / `nfp_release` "scheduled" rows from the calendar as published by Jan 1 (original dates,
   available_ts = Jan 1 or the earlier publication date). Do not use the realized date.
2. Add a `schedule_change` row per move: event_ts = the new date, available_ts = the date the notice was published,
   and a `cancelled` flag for the Oct-2025 CPI and NFP. Source each notice date from its BLS page.
3. In `pit.next_scheduled`, skip any scheduled event that a change or cancellation with available_ts < t has superseded,
   and use the superseding date instead. Keep the `*_release` rows for the actual realized dates for the value features.

### F2 (CONFIRMED): cancelled FOMC meeting is absent

Source: https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm lists "March 17-18 (cancelled) Meeting - 2020".
The 2020 scheduled statements in the dataset are 7 (Jan, Apr, Jun, Jul, Sep, Nov, Dec). Every other year has 8.
`build_fomc` skips headings containing "cancel", so no row is written. Only scheduled-then-cancelled meetings are affected.

Probe: `features.macro_calendar` run at 2020-03-02 17:00 UTC gives `cal_fomc_hours_to` = 1393 h (next = Apr 29).
With the Jan-1 calendar, the next meeting is Mar 18 14:00 ET = 18:00 UTC, which is 385 h away.
The absence encodes a cancellation that the Fed announced around the Mar 15 actions. The Board's closed-meeting
page (https://www.federalreserve.gov/aboutthefed/boardmeetings/20200317closed.htm) says the Mar 17 Board meeting was cancelled
after the Mar 15 meeting. The FOMC-level notice date was not pinned down, so the exact date is SUSPECTED.

Fix (F2): write the `fomc_statement` scheduled row for 2020-03-18 (available_ts = 2020-01-01 UTC), add a `cancelled`
row with available_ts = the cancellation notice date, and handle it as in F1 step 3.

### F3 (CONFIRMED, coverage): cancelled releases and joint release

- No row exists for the cancelled Oct-2025 CPI or NFP. This is correct for the realized series, but it removes the
  original dates from the schedule (see F1).
- The Dec-16-2025 NFP release carried Oct and Nov 2025 together. The builder keeps only the latest reference month
  (`joint release, kept 2025-11, dropped ['2025-10']`), so the Oct first print is not in the dataset. This is a coverage gap.

### F4 (PASS): release times and DST

- `et_ts()` uses `zoneinfo.America/New_York`, so DST is applied per date. Examples from the parquet:
  `2014-03-18 08:30 EDT`, `2014-11-20 08:30 EST`, `2014-03-19 14:00 EDT`, `2020-11-05 14:00 EST`.
- Unscheduled moves, verified against the Fed press-release headers (fetched from federalreserve.gov):
  - `monetary20200303a.htm`: "March 03, 2020 ... For release at 10:00 a.m. EST". Dataset: 2020-03-03 10:00 EST. OK.
  - `monetary20200315a.htm`: "March 15, 2020 ... For release at 5:00 p.m. EDT". Dataset: 2020-03-15 17:00 EDT. OK.
- Scheduled FOMC statements are at 14:00 ET, the documented time.

### F5 (PASS, 5 of 8 verified): first prints and vintage dates

Eight months were drawn with `random.seed(20261010)` from the `cpi_mom_first` and `nfp_first` rows.

| Series, month | ALFRED vintage = release | Dataset value | Source check |
|---|---|---|---|
| CPI 2018-11 | 2018-12-12 | +0.0194% (rounds to 0.0) | BLS release of 2018-12-12: "unchanged in November on a seasonally adjusted basis" (verified, `bls.gov/news.release/archives/cpi_12122018.htm`) |
| NFP 2020-11 | 2020-12-04 | +245k | BLS `empsit_12042020.pdf`: "rose by 245,000 in November" (verified) |
| NFP 2020-12 | 2021-01-08 | -140k | CNBC 2021-01-08: "fell by 140,000 in December" (verified) |
| NFP 2021-07 | 2021-08-06 | +943k | CNBC 2021-08-06: "increased by 943,000" (verified) |
| NFP 2022-06 | 2022-07-08 | +372k | BLS `empsit_07082022.pdf`: "rose by 372,000 in June" (verified) |
| CPI 2014-10 | 2014-11-20 | +0.0038% | not verified against a source |
| CPI 2019-05 | 2019-06-12 | +0.0773% | not verified (the BLS page fetched was a different month) |
| CPI 2021-05 | 2021-06-10 | +0.6442% | not verified |

Vintage dates also match the actual release dates in the BLS lapse notice: CPI 2025-10-24, 2025-12-18, 2026-01-13,
2026-02-13; NFP 2025-11-20, 2025-12-16, 2026-02-11. The ALFRED CSV for `CPIAUCSL` vintage 2025-10-24 contains
data through 2025-09. No ALFRED vintage was found to precede its release.

The 2026 NFP check: CNBC 2026-03-06, "fell by 92,000 in February", matches the dataset value of -92 for 2026-03-06.

Computation: CPI MoM = `(lv[m]/lv[m-1]-1)*100` within one vintage, and NFP = `lv[m]-lv[m-1]` within one vintage. Both match
the BLS definition, since the prior month is taken from the same release.

### F6 (PASS): feature code

`scratchpad/audit/probe_features.py`:
- Test 1: drop all rows with `available_ts >= t`, recompute `macro_calendar` at 419 anchors (daily 17:00 UTC plus each
  event +/- 1 s and 1 h). 0 differ.
- Test 2: multiply the values of all rows with `available_ts >= t` by 1000 and add 7. 0 of 17 anchors differ.
- Boundary: `pit.asof` at `available_ts == t` excludes the row for `fomc_statement`, `cpi_mom_first` and `nfp_first`.
- `_sched` / `next_scheduled` use only events with `event_ts > t` and `available_ts < t`.
- `cal_fomc_target_chg_bp` and `cal_*_recent` use `asof` and `asof(t-k)` only.

The only way the feature sees post-t information is through the scheduled-row dates in F1 and F2.

### F7 (SUSPECTED): calendar publication

The Fed publishes the next year's calendar in advance (`fomccalendars.htm` lists the 2026 meetings; the calendar page
was fetched in this audit). BLS publishes its release schedule in advance as well. Jan 1 of the event year is therefore a
conservative date for undisrupted events, and this audit did not check a dated publication. Affected only by F1 and F2.

## Not checked

- The 3 unverified CPI months in F5, and all CPI/NFP values before 2014 and after 2026-09.
- The notice date of each of the five later BLS moves (F1 fix step 2).
- Whether ALFRED ever records a vintage before the BLS release for an undisrupted month. Only the sampled months were checked.
- Upper-target parsing: 32 sampled FOMC meetings matched expected rates (all mismatches were errors in my expected list, not the data).

## Re-run

```
python3 -I /tmp/.../scratchpad/audit/probe_features.py   # F6
python3 -I /tmp/.../scratchpad/audit/impact.py           # F1 examples
curl -sSLf --http1.1 "https://alfred.stlouisfed.org/series/downloaddata?seid=CPIAUCSL"   # vintage list (669 dates)
```
