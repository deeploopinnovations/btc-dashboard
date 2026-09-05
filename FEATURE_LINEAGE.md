# Feature lineage

Every input, where it comes from, when a trader could actually have had it, and
how much of each split it covers. Numbers are re-derived from the artifacts, not
recalled. Where something could not be established, it says so rather than
guessing — an unresolved field is more useful than a confident wrong one.

Requirement enforced throughout: **`feature_available_time <= prediction_time`**,
with strict as-of joins, no backward fill from future values, no future-aware
normalisation, and no scaler fitted on test.

---

## 1. BTC price and volume — the internal cascade

| field | value |
|---|---|
| source | 1-minute OHLCV bars, `model/artifacts/btcusd_1min.parquet` (157 MB, committed pipeline input) |
| aggregation | `episodes.build_hourly` → exact gap-free hourly table, 128,031 rows, 2012-01-01 → 2026-08-10 |
| raw timestamp | `hour_ts`, epoch seconds, **UTC**, marking the hour's start |
| available to a trader | at the hour's **close**, i.e. `hour_ts + 3600` |
| publication delay | none beyond bar close — this is own-exchange data, not a published index |
| prediction timestamp | `s_tau` is the close of the **previous** hour, so every feature is a function of bars at or before `anchor − 1h` |
| coverage | **100 % / 100 % / 100 %** across train / calib / test, all 11 families |

**Derived families** (39 columns reaching Stage A; `cal_H` is the only one that
varies with horizon at a fixed anchor):

| family | cols | what it is |
|---|---|---|
| `har_*` | 5 | realized-vol cascade, 1h / 6h / 1d / 5d / 22d |
| `semi_*` | 6 | signed semivariance and jump decomposition |
| `jump_*` | 2 | jump share of variance |
| `rq_*` | 1 | realized quarticity, an estimator-noise proxy |
| `rng_*` | 4 | high–low range measures |
| `eff_*` | 3 | path efficiency — **`NON_MODEL_COLS`, not model inputs** |
| `mom_*` | 5 | trailing returns |
| `vov_*` | 2 | vol-of-vol |
| `reg_*` | 3 | regime indicators |
| `seas_*` | 3 | seasonality |
| `cal_*` | 8 | calendar |

All trailing windows are **exclusive of their own row** (`_trailing_sum`), and
`mom_ret` shifts explicitly (`cur[1:] = logc[:-1]`). Verified by the leakage
suite, not assumed.

## 2. Deribit DVOL — implied volatility

| field | value |
|---|---|
| source | `public/get_volatility_index_data` — the **implied-volatility INDEX**, confirmed distinct from `public/get_historical_volatility` (realized) |
| file | `data/newdata/dvol_btc.parquet`, 47,563 rows |
| span | 2021-03-24 → 2026-08-26 UTC, every gap exactly 3600 s, 0 duplicates |
| raw timestamp | `ts_ms` / `ts`, **UTC** |
| **available to a trader** | **UNRESOLVED — see below** |
| transformation | endpoint returns candles `[ts_ms, open, high, low, close]`; the harvester keeps **close** only |
| prediction timestamp | `anchor − lag_hours × 3600`, `lag_hours` configurable, **default 1, robustness setting 2** |
| coverage | **32.7 % / 100 % / 100 %** train / calib / test |

**The unresolved field, stated plainly.** The endpoint returns *candles*, and it
could not be established whether `ts_ms` marks the candle's **open** or its
**close**. If it marks the open — the prevailing convention across exchanges,
including Deribit's own TradingView-format chart endpoint — then the close
stamped at hour `h` is not determined until `h+1`, and a 1-hour lag reads a value
knowable only *at* the anchor. The endpoint is unreachable from this container
and the fetched documentation does not state the convention.

**No leakage test in this repository can detect this**, because they all verify
the code's indexing against the `ts` column, never whether `ts` means what the
code assumes. Handled by making the result independent of the answer: at
`--lag-hours 2`, correct under either convention, the effect *strengthens*
(−0.033551 against −0.031000, 108.2 % retention).

*Residual caveat:* `harvest_dvol()` falls back to the realized-vol endpoint if the
index fetch fails, unioning rows into the same column with **no per-row
provenance flag**. No evidence it ever fired; also no way to prove it never did.
A `source` column would settle it and does not exist.

## 3. Deribit funding rate

| field | value |
|---|---|
| source | `public/get_funding_rate_history` |
| file | `data/newdata/funding_btc.parquet`, 64,206 rows |
| span | 2019-04-30 10:00 → 2026-08-26 17:00 UTC |
| gaps | hourly at 64,204 of 64,205; **one 10,800 s gap at 2020-08-27 05:00 UTC** (two missing stamps) |
| raw timestamp | `ts`, **UTC**, hour-aligned |
| units | `interest_1h` is the **additive per-hour** rate — established from the data, not asserted: `interest_8h` tracks the trailing 8-hour **sum** at corr 0.9990 and 0.37 % median relative error, against 87.49 % for the mean and 87.88 % for `interest_1h × 8` |
| prediction timestamp | most recent stamp strictly before the anchor |
| coverage | **67.8 % / 100 % / 100 %** (`fund_rate`); trailing-window features lower in train only — `fund_z_20d` 65.7 %, `fund_cum_7d` 67.1 % |

**17.8 % of `interest_1h` values are exactly 0.0**, which broke the inherited
multiplicative leakage corruption (`0 × anything == 0`). Fixed with an affine
corruption; decoy catch rate went 9/10 → 10/10.

## 4. What is enforced, and by what

| requirement | enforced by |
|---|---|
| `available_time <= prediction_time` | `leakage.py` corrupt-and-diff + a future-reading decoy that **must** be caught |
| corruption actually moves the rows | `pitfalls.check_corruption_bites` (check 12) |
| missing → NaN, never a neighbour | dense hourly grid, `pos_of` returns −1 out of range |
| no scaler fitted on test | `standardise()` moments from history folds only; verified by audit |
| positional alignment | hard `np.array_equal` assert on `anchor_ts`, `SystemExit` on mismatch |
| horizon-appropriate embargo | `time_splits` derives `embargo_hours` from `max(H)` |

## 5. Known gaps

1. **DVOL candle open-vs-close** — unresolved; neutralised by the 2-hour lag
   rather than settled. Settling it needs either a live probe from the harvest
   runner or a re-harvest retaining OHLC so `open[t]` can be chained to
   `close[t-1]`.
2. **No provenance column on `dvol_btc.parquet`** — the realized-vol fallback
   cannot be ruled out retrospectively.
3. **`iv_term_slope` BLOCKED** — DVOL is a single 30-day constant-maturity index;
   a slope needs two tenors at one instant. Not built, no proxy substituted.
4. **Maturity mismatch, unaddressed.** DVOL is a **30-day** implied vol. It is
   used as an input to 6h–24h forecasts. That mismatch is real and is *not*
   corrected for anywhere; it is a reason the correction is modest and a caution
   against reading `iv_level` as a horizon-matched forecast.

*Educational research only. Not financial advice.*
