# Dataset audit — what the model is actually fed, and what it cannot possibly know

Run 2026-09-01. Every number below was measured from the artifacts, not recalled.

---

## 1. The answer to "is the dataset labelled with events?"

**No. There is not a single event label anywhere in the model's inputs.**

All **42** feature columns derive from exactly two things: OHLCV bars, and the
clock.

| family | n | what it is |
|---|---|---|
| `har_*` | 5 | trailing realized vol at 1h/6h/1d/5d/22d |
| `semi_*` | 6 | realized semivariance, signed jump variation |
| `rng_*` | 4 | Parkinson / Garman-Klass range estimators |
| `mom_*` | 5 | trailing returns, drawdown, distance to MA |
| `vov_*`, `jump_*`, `rq_*` | 5 | vol-of-vol, jump share, realized quarticity |
| `seas_*` | 3 | same clock window on prior days |
| `eff_*` | 3 | path efficiency |
| `reg_*` | 3 | RV percentile, vol trend, post-ETF flag |
| `cal_*` | 8 | hour, day-of-week, month, weekend fraction, horizon |

`reg_post_etf` is a single binary that flips once, in January 2024. That is the
**entire** extent of "the model knows something happened in the world".

What exists in the repository but is **not** in the model:

- `data/news.json` — **14 headlines**, refreshed live for the dashboard. Not a
  historical corpus. It cannot be joined to 2021 episodes because it has never
  contained 2021.
- `data/fg.json` — a **single current** Fear & Greed value. No history.
- `iv_features.parquet` — Deribit DVOL, 36.9 % coverage, 62,204 rows inside the
  training window. Repeatedly tested (`E2`, `E2c`, `E2-confirm`), never adopted.
- `funding_features.parquet` — perpetual funding, 50.0 % coverage, 128,780 rows
  in training. Built, verified, never wired in.

And absent entirely: macro calendar (FOMC, CPI, NFP), exchange outages, ETF
flows, on-chain flows, liquidations, order book, options flow, social sentiment,
regulatory announcements, exchange listings/delistings.

---

## 2. Why this matters more for volatility than for direction

The user's intuition is right, and it is right **asymmetrically**. These two
failures have different causes and the same fix does not address both.

**Volatility — event data would plausibly help, and here is the headroom.**
Forecast error is extraordinarily concentrated. Share of *total* pooled QLIKE
carried by the worst episodes:

| horizon | model | worst 0.1 % | worst 1 % | worst 5 % |
|---|---|---:|---:|---:|
| 1h | `har_short` | **48.0 %** | 61.4 % | 76.0 % |
| 1h | `noctua_v1` | 11.6 % | 31.3 % | **52.8 %** |
| 24h | `noctua_v1` | 8.7 % | 27.4 % | **52.0 %** |
| 168h | `noctua_v1` | 4.8 % | 22.4 % | 48.5 % |

**Roughly half of all forecast error lives in 5 % of episodes.** Those are
shocks. A model fed only trailing price statistics learns that volatility
clusters — it cannot learn that CPI prints at 13:30 UTC on a known date.

The literature is unambiguous that *scheduled* macro announcements are a
forecastable volatility driver, and that is precisely the subset a calendar
supplies for free, with no look-ahead risk, because the *schedule* is known in
advance even though the *outcome* is not.

Note `garch_t` has the **least** concentrated error (37.7 % in the worst 5 % at
H=1 against `har_short`'s 76.0 %). That is the mechanism behind its spike
advantage: it over-reacts to the last shock, so it is less often caught flat.

**Direction — event data would mostly not help, and could hurt.**
`D1-direction-bench-corrected` measured 8 of 8 model arms failing at
n ≈ 49,000 per horizon, with calibration slopes between −0.03 and +0.06 against
a required [0.8, 1.2]. By the time news is public it is priced; that is the
efficient-market null, and it is what the data shows. Adding a news feed to a
direction model mainly adds **look-ahead risk**, because published timestamps
are unreliable and a headline dated 14:00 was often known at 13:47.

---

## 3. Data integrity: what I checked and what I found

**Clean, and I checked because I suspected otherwise.** The 1-minute series has
a `filled` column. It is **0 for every one of 7,681,837 rows** — no synthetic
bars. `bad_print` flags 11,566 minutes (0.15 %), and only 52 modern-era hours
are affected.

**17.08 % of minutes have zero volume, and the rate is wildly non-stationary** —
94.95 % in 2012 falling to 0.48 % in 2026, and varying 8× *across the test
folds alone* (0.48 % in 2026 to 3.94 % in 2023).

I hypothesised this biases the target downward, since a dead minute contributes
nothing to realized variance. **The hypothesis is falsified.** Against
Parkinson's range estimator — which uses only high/low and barely cares how many
minutes traded — the ratio `rv5/Parkinson` *rises* with dead-minute density
(1.15 → 1.90, Spearman +0.119) rather than falling. Quiet markets genuinely have
low volatility rather than mismeasured volatility.

**But that test exposed something else.** Two standard estimators of the *same*
target disagree by **15 % in the cleanest hours and 90 % in the deadest**.
"Realized volatility" is not one number; the estimator is a modelling choice
with a first-order effect, and the project has only ever used one.

**The zero-variance floor**, recorded separately as `P2-floor-defect`: an hour
with no trades becomes `har_1h = −13.8155`, the confident assertion that
volatility was 1e-6. One such episode carried **72.5 %** of a fold's entire
QLIKE. Rare (11 modern-era bars) and catastrophic. Still live in the shipped
feature path.

---

## 4. The honest ranking of why this model underperforms

1. **No exogenous information at all.** 42 features, one binary regime flag,
   zero events. Half the error is in 5 % of episodes.
2. **The target is estimator-dependent** and only one estimator was ever tried.
3. **~6 independent regime-observations.** `E-power` measured that 24× more
   episodes bought 1.53× effective sample size. This is the binding statistical
   constraint and no amount of architecture fixes it.
4. **A sentinel value that lies** (`P2-floor-defect`), unfixed in production.
5. **Median-vs-mean mismatch**, now measured: the fix improves QLIKE 9.7 % and
   degrades every barrier metric (`P2-scale-v2-result`).
6. **Direction is genuinely near-unpredictable** from public price history, and
   this is a property of the market rather than of the model.

---

## 5. What I would do about it, in priority order

1. **A scheduled-macro calendar.** FOMC, CPI, NFP, options expiry, halving. Free
   of look-ahead by construction — the schedule is public in advance. Directly
   targets the 5 % of episodes carrying half the error. Highest value per unit
   of risk in the whole list.
2. **Wire in the exogenous series already harvested.** DVOL and funding sit in
   the repository unused, with 62k and 129k training-window rows.
3. **Fix the floor**, in the shipped path, on its own merits.
4. **Add a second RV estimator** and treat the disagreement as information.
5. **Do not add a news feed to the direction model.** It would add look-ahead
   risk against a measured near-zero signal.

*Educational research only. Not financial advice.*
