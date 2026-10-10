# Lookahead audit: data/context/derivatives.parquet

Auditor: adversarial review, 2026-10-10. Scope: `model/context/sources/derivatives.py`, the
parquet it wrote, `model/context/pit.py`, `model/context/features.py::derivatives()` and
`build()`. No data file, fetch script or git state was modified. Scratch scripts live in the
session scratchpad (`align.py`, `clean.py`, `xcorr.py`, `trunc.py`, `deliv.py`).

## Verdict: PASS WITH NOTES

No lookahead found. Binance timestamps are UTC and mark the END of the 5-minute window, and
`available_ts = stamp + 300 s` is conservative. The feature code passes a truncation test with
zero mismatches. Four defects are real but are about missing data, not future data: a silent
zero-fill, an always-true presence flag, stale live coverage, and an unverified revision risk.

| # | Area | Status | Severity |
|---|---|---|---|
| F1 | Binance metrics stamp semantics (end of window, UTC) | CONFIRMED correct | none |
| F2 | +300 s lag sufficiency | CONFIRMED sufficient | none |
| F3 | Timezone (UTC) and crash alignment | CONFIRMED | none |
| F4 | Daily file re-publication / revisions | SUSPECTED (strong indication) | medium |
| F5 | Funding timing (event = available = T) | CONFIRMED safe | none |
| F6 | Expiry calendar (10+ dates vs Deribit) | CONFIRMED correct | none |
| F7 | Feature lookahead (truncation test) | CONFIRMED no leak | none |
| F8 | `build()` zero-fills missing Binance features; `derivatives_present` is always 1 | CONFIRMED defect | high (silent) |
| F9 | Live tail: Binance features NaN after the archive ends | CONFIRMED gap | medium |
| F10 | Long coverage gaps in top-trader and taker fields (2021-22) | CONFIRMED, handled only by F8 | medium |
| F11 | Options settlement feed has no rows for 2026-08-28 and 2026-09-25 | CONFIRMED, calendar unaffected | low |
| F12 | 2027 expiry rows withheld until 2027-01-01 | CONFIRMED by design | low |

---

## F1. Meaning of `create_time`: end of window, not start (CONFIRMED correct)

Question: is the stamp the snapshot moment, or the start of a window whose values are known five
minutes later?

Test. For every 5-minute bar, compare the absolute change in `sum_open_interest` stamped T
(`OI(T) - OI(T-5m)`) with the absolute log return of two candidate price bars:
- bar ENDING at T, `[T-5m, T)`: the hypothesis "stamp = end of window / snapshot at T"
- bar STARTING at T, `[T, T+5m)`: the hypothesis "stamp = window start"

Price bars come from `data.binance.vision/.../futures/um/daily/klines/BTCUSDT/5m`, which uses
the same UTC clock. Taker ratio was tested the same way against the signed bar return.

Results (`clean.py`, `xcorr.py`):

| Sample | corr(\|dOI\|, bar ENDING at T) | corr(\|dOI\|, bar STARTING at T) |
|---|---|---|
| 2022-11-08 (FTX) | 0.588 | 0.238 |
| 2022-11-09 | 0.211 | 0.131 |
| 2021-05-19 (crash) | 0.462 | 0.382 |
| 2021-05-18 | 0.280 | 0.088 |
| 80 random days 2021-2025, mean | 0.368 | 0.237 |

Taker buy/sell ratio vs signed return, 80 random days: ENDING 0.392, STARTING 0.228.

Lag profile over 80 days (`xcorr.py`), correlation of |dOI| at T with |return| of the bar
`[T+L*5m, T+(L+1)*5m)`: L=-1 (bar ending at T) 0.369 is the peak. L=0 (starting at T) is 0.237.
The peak is at the bar that ends at the stamp.

Documentation. Binance REST `topLongShortAccountRatio` / `globalLongShortAccountRatio`
documents `timestamp` as "End time of the period, in milliseconds" (source:
developers.binance.com Market Data page, fetched via Firecrawl). The basis endpoint documents
`timestamp` as "Start time of the period". The extracted page does not state the semantics for
`openInterestHist`, so the empirical test above is the decisive evidence.

Conclusion: `create_time` is the end of the 5-minute window (values reflect trades up to T).
The `event_ts` in the parquet is correct as the "moment the value refers to".

## F2. Is +300 s enough? (CONFIRMED sufficient)

Stamp T describes the market up to T, so it is public by T plus processing time. `available_ts =
T + 300` is conservative. Even under the window-start reading (public at T+300), +300 would be
exactly sufficient, because `pit.asof` uses strict `<`.

Hourly row for hour h has event h:55 and available h+1:00. A 17:00 decision sees the 15:55 row
(available 16:00), not the 16:55 row (available 17:00). This is one extra conservative step, not
a leak. The 17:00 boundary was checked directly in F7.

## F3. Timezone and crash alignment (CONFIRMED UTC)

- Day files cover 00:00 to 23:55 UTC: 2,230 cached files, all stamps on the 5-minute grid and
  all within their own date (no cross-day stamps, `clean` check).
- The lag test in F1 peaks at the bar that ends at the stamp, on the same UTC clock as the
  kline open times. An hour of timezone offset would move the peak by 12 bars, and no such
  shift appears.
- `btc_history.parquet` `hour_ts` is UTC hour start. Its open/close match Binance USD-M
  5m klines within about 0.1%, consistent with a different venue (spot vs perpetual basis).
- Hourly OI from `derivatives.parquet` vs `btc_history.parquet` close:

2022-11-08 (FTX):

| UTC hour | BTC close | OI (last snap) | OI chg % | Price chg % |
|---|---|---|---|---|
| 15:00 | 19520 | 132921 | | |
| 16:00 | 20371 | 123599 | -7.01 | +4.36 |
| 17:00 | 19340 | 120394 | -2.59 | -5.06 |
| 19:00 | 18280 | 116961 | -3.59 | -1.78 |
| 21:00 | 18701 | 123727 | +3.80 | +2.97 |

2021-05-19 (crash):

| UTC hour | BTC close | OI (last snap) | OI chg % | Price chg % |
|---|---|---|---|---|
| 11:00 | 38680 | 39906 | -13.47 | -1.63 |
| 12:00 | 33886 | 38936 | -2.43 | -12.40 |
| 13:00 | 35957 | 33028 | -15.17 | +6.11 |
| 15:00 | 37434 | 26152 | -13.72 | -0.64 |
| 16:00 | 39802 | 25164 | -3.78 | +6.32 |

OI collapses coincide with violent hourly moves on the same clock (46k to 24.8k open interest
over the day in 2021).

## F4. Re-publication and revisions of the daily files (SUSPECTED, strong indication)

Evidence (`heads_all.txt`, HEAD requests, all 2,230 cached metrics files):
- 1,411 of 2,230 files have `Last-Modified` in 2026, months or years after their day (456 in
  Mar 2026, 437 in Jul 2026, 439 in Aug 2026, and others).
- Example: `BTCUSDT-metrics-2024-05-10.zip` Last-Modified 2026-07-15. `2025-06-01` Last-Modified
  2026-07-31. `2020-09-01` Last-Modified 2026-03-18.
- Missingness changed across re-uploads. 2021 days re-uploaded in 2026 (n=334) have 0.0% NaN
  top-trader counts. 2021 days last modified in 2023 (n=31) have 4.5% NaN.
- Our cache matches the current remote bytes for all 2,230 files (ETag equals local MD5 for
  every file, and no remote Last-Modified is newer than our download). So the parquet reflects
  the current archive. The contents of the original publications are not available here, so
  whether earlier vintages differed cannot be confirmed.

Impact: a backtest built from today's archive may use values that were not what a trader saw at
the time. The SPEC asks for `revisable` in that case (rule 5). The source tags it `lagged`.
The cache is in `/tmp/binance_cache`, which is not persistent, so a rebuild elsewhere would
silently pick up whatever is current.

## F5. Funding timing (CONFIRMED safe)

- 7,395 rows, from 2020-01-01 to 2026-09-30 16:00 UTC. Hours of day are 0, 8 and 16 only. All
  gaps are exactly 8 h. No minute offsets. `event_ts == available_ts` for every row.
- The final rate is fixed at `calc_time`. The predicted rate that Binance shows earlier is not
  used, which is correct. A 16:00 rate is first visible at 16:00, so a 17:00 decision may use
  it. `pit.asof` strict `<` excludes a decision exactly at 16:00, which is conservative.
- Staleness: `max_age=9H` on an 8-hour cadence tolerates one hour of slack. Correct.

## F6. Expiry calendar (CONFIRMED correct)

Source: `expiry_calendar()`. Rows: 128 (96 monthly, 32 quarterly). Every event is the last Friday of
its month, at 08:00 UTC. Quarterly months are only 3, 6, 9 and 12.

Verified against Deribit's public `get_last_settlements_by_currency?currency=BTC&type=delivery`
history (89,000 rows, 2019-01 to 2026-10, saved to scratchpad `deliveries.json`):

| Date (UTC) | Type | Deribit option settlements | Settlement time |
|---|---|---|---|
| 2019-07-26 Fri | M | 46 | 08:00 |
| 2020-12-25 Fri (Christmas) | Q | 80 | 08:00 |
| 2021-07-30 Fri | M | 58 | 08:00 |
| 2022-03-25 Fri | Q | 71 | 08:00 |
| 2023-08-25 Fri | M | 87 | 08:00 |
| 2024-03-29 Fri (Good Friday) | Q | 134 | 08:00 |
| 2024-06-28 Fri | Q | 139 | 08:00 |
| 2025-01-31 Fri | M | 159 | 08:00 |
| 2025-10-31 Fri | M | 102 | 08:00 |
| 2026-06-26 Fri | Q | 129 | 08:00 |
| 2026-07-31 Fri | M | 121 | 08:00 |
| 2026-08-28 Fri | M | 0 (see F11) | futures row at 08:00 |
| 2026-09-25 Fri | Q | 0 (see F11) | futures row at 08:00 |

Across all 2,463 option expiry dates, 2,461 settle at 08:00 UTC. The other two first settle at
08:39 and 09:18 UTC (not investigated further; neither is a monthly or quarterly date). Holidays are not shifted: Christmas 2020 and Good Friday 2024 expired on the
Friday. Every last-Friday date from 2019 to 2026 that appears in the calendar also appears in
the Deribit futures or options feeds, with the exceptions in F11.

Availability: `max(event - 90 d, Jan 1 of event year)`. This satisfies SPEC rule 1 for
`scheduled` rows. The expiry schedule is deterministic, so no outcome information is involved.

## F7. Feature code: lookahead test (CONFIRMED no leak)

Method (`trunc.py`): for 44 random anchor times, plus 2021-05-19 13:30, 2022-11-08 20:00,
2022-11-09 12:00 and 2024-03-29 17:00, compare `derivatives(ctx_truncated, t)` where
`ctx_truncated = ctx[available_ts < t]`, against `derivatives(ctx_full, t)`.

Result: 44 anchors, 0 feature mismatches. Every `derivatives()` feature uses only rows with
`available_ts < t`.

Code review of `derivatives()`:
- `_logchg` / `_a` go through `pit.asof` (strict `<`). The `then` side uses `t - k` with the same
  rule.
- The funding, top-trader, global, taker and OI features are all `_a` calls with `max_age`
  values.
- The `liq_*` features are skipped because no liquidation series exists.
- `_sched` uses `pit.next_scheduled`, which requires `available_ts < t`. Future expiry dates are
  visible only after their 90-day-before availability date.

`max_age` values: OI/LS/taker 3 h, funding 9 h. Median gap is 1 h for hourly series and 8 h for
funding. So 3 h and 9 h are sensible. Stale values are dropped, not carried forward. Gaps in
the series (F10) are handled by these rules.

## F8. Silent zero-fill and always-true presence flag (CONFIRMED defect, high)

`features.build()` does `F.fillna(0.0)` after computing `present = F.notna().any(axis=1)`.
The derivatives group contains the Deribit calendar features, which are non-NaN at every
decision time. So `derivatives_present` is 1 always, and all Binance features become 0 when
missing.

Reproduced (`build('derivatives', 2026-10-10 17:00 UTC)`):

```
der_oi_chg24        0.000000   <- Binance missing (see F9), filled to 0
der_oi_chg7d        0.000000
der_top_ls          0.000000
der_glob_ls         0.000000
der_taker_ls        0.000000
der_bn_funding_bp   0.000000
der_exp_m_hours_to  1.000000   <- calendar, valid
derivatives_present 1.000000   <- says "present", but no Binance value is present
```

Why it matters: a log ratio of 0 means "ratio 1.0", and an OI change of 0 means "flat". Both are
real market values. The model cannot tell "missing" from "neutral", so missing-data regimes
(2021-22, live tail) are learned as neutral.

Fix (`model/context/features.py`):
```python
def build(group, t, ctx=None):
    ...
    F = GROUPS[group](ctx, t).replace([np.inf, -np.inf], np.nan)
    binance = [c for c in F.columns if c.startswith(("der_oi", "der_top", "der_glob",
                                                    "der_taker", "der_bn"))]
    F[f"{group}_present"] = F[binance].notna().any(axis=1).astype(float) if binance \
        else F.notna().any(axis=1).astype(float)
    F[binance] = F[binance].fillna(0.0)  # or better: keep NaN and impute in training
    ...
```
Better still, add one presence flag per source (`der_bn_present`, `der_cal_present`) and leave
calendar features unfilled. Imputation (median from training data, plus indicator) is preferable
to zero for ratio features.

## F9. Live tail: Binance data ends before decision time (CONFIRMED gap)

- Metrics: the last archive row is `2026-10-09 23:55 UTC`, published about 06:41 UTC on
  2026-10-10 (Last-Modified). A 17:00 decision on 2026-10-10 is therefore 17 h past the last row.
  That exceeds `max_age=3H`, so `oi`, `ls` and `taker` features are NaN and zero-filled (F8).
- Funding: the last row is `2026-09-30 16:00`. The October monthly file appears after month-end,
  so from 2026-10-01 onward `der_bn_funding_bp` is NaN and zero-filled.
- Not a leak. It is an operational gap, and it means the live pipeline does not yet have
  today's derivatives data.

Fix: for live decisions, call the REST endpoints `openInterestHist`, `topLongShortAccountRatio`,
`globalLongShortAccountRatio`, `takerlongshortRatio` (period 5m) and `fundingRate`, and set
`available_ts = stamp + 300` as the archive path does. Or document that features are valid only
for backtests and block live use until the feed is added.

## F10. Long coverage gaps in the top-trader and taker fields (CONFIRMED)

Largest gaps in the hourly series (`ctx.available_ts` diffs, `trunc.py`):
- `global_ls_count`: 480 h ending 2021-12-30 14:00, plus an 11.4 h gap at 2024-02-16 13:35.
- `toptrader_ls_sum`: 2,456 h ending 2022-09-03 08:00 and 2,183 h ending 2022-01-31.
- `taker_ls_vol`: 2,358 h ending 2022-01-31 and 731 h ending 2021-12-30.

Stamp-level missingness: 2022 archive days (last modified 2023) are 87% NaN for top-trader
counts. The blanks are in the archive files themselves; the parser does not create them. Before F8 is fixed,
these periods are zero-filled and look like real data.

Fix: as F8. Also mark the 2021-12 to 2022-09 span as `present=0` in training and exclude it or
treat it separately.

## F11. Options settlement feed lacks 2026-08-28 and 2026-09-25 (CONFIRMED, calendar unaffected)

`get_last_settlements_by_currency` shows options for every Friday to 2026-07-31 (for example
121 settlements on 31JUL26), but zero option rows for 28AUG26 and 25SEP26. The futures rows for
both dates exist at 08:00 UTC. The calendar's dates are correct (the monthly and quarterly
futures settle on them). This is a gap in the public feed used for the check, not in the
calendar. No action on the parquet is needed.

## F12. 2027 expiry rows are withheld until 2027-01-01 (CONFIRMED, by design)

`EXPIRY_YEARS` goes to 2027, but `available_ts >= 2027-01-01` is later than `NOW + 1 day`, so
2027 rows are dropped. Until the script is re-run after 2027-01-01, `next_scheduled` returns NaN
for expiries after 2026-12-25. `_sched` then uses the cap (1.0, meaning no expiry within the
window). That is correct for the Jan 29 2027 expiry, which is more than 10 days away, but the
rule should be written down. Re-run the source script after 2027-01-01.

---

## Other notes

- Revision risk (F4) also applies to the cache: `/tmp/binance_cache` is not persistent. Store
  raw archives with a manifest (URL, ETag, MD5, fetch time) under `data/context/raw/` and set
  `pit_quality = "revisable"` for metrics until first-seen vintages exist.
- `build()` is the only path that zero-fills. `pit.asof` correctly returns NaN.
- Performance: `next_scheduled` loops per anchor. This is not a correctness issue.

## Fix list (confirmed defects only)

1. F8: stop zero-filling Binance features and compute presence per source, excluding calendar
   columns (`model/context/features.py::build`).
2. F9: add a live REST path or restrict live use to the archive horizon, and document the
   funding gap after 2026-09-30.
3. F10: mark 2021-12 to 2022-09 (top-trader and taker fields) as absent in training and exclude
   it from the derivatives feature set, or impute with a missingness indicator.
4. F11, F12: documentation only. Re-run `derivatives.py` after 2027-01-01 to add 2027 expiries.

No change to `data/context/derivatives.parquet` is required for lookahead. Items F4 (revision
risk, SUSPECTED) and F8 (zero-fill) should be addressed before the derivatives group is used in
a model.

## Reproduction

Scratch scripts in `/tmp/claude-0/-home-user-btc-dashboard/a653e416-9fde-5630-b5ae-17dc7a498e81/scratchpad/`:
- `clean.py`: per-day timing test (F1, F3)
- `xcorr.py`: lag profile over 80 days (F1)
- `trunc.py`: truncation lookahead test and staleness (F7, F10)
- `deliv.py`: Deribit settlement paging (F6, F11), output `deliveries.json`
- `heads_all.txt`: ETag and Last-Modified for all 2,230 metrics files (F4)
