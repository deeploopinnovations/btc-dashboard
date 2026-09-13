# FEATURE_CATALOG — Phase 1

Documents every one of the 42 columns in `model/artifacts/features.parquet`
against the Phase 1 gate in `model/PHASES.md`: `event_time`, `publication_time`,
`feature_time`, and `source`, for each column, or an explicit UNVERIFIED with
reason. Produced by reading `model/noctua/features.py`, `model/noctua/episodes.py`,
`model/noctua/spec.py`, and `model/artifacts/leakage.json` (the leakage audit
output). No column here was classified from its name alone — every "what it
measures" line was checked against the code that computes it (see §0.3 for the
one name that does not mean what it appears to).

Nothing in `model/serve/`, `model/noctua/`, or `model/eval/` was modified to
produce this document. This is documentation, not a fix.

---

## 0. Method and conventions

### 0.1 Definitions used throughout

- **event_time** — the real-world span of market data (or calendar structure)
  the feature's *value* describes.
- **publication_time** — when that value became knowable in the world. For
  Bitstamp trade prints this is effectively the same second as `event_time`
  (a real exchange trade is public immediately); see §0.4 for the one caveat
  this session could not close.
- **feature_time** — the specific array index this repository's code actually
  reads when building the row the model sees for an episode anchored at hour
  `a`, stated relative to `a`.
- **source** — the artifact file and the code path that produced it.

### 0.2 Two lag groups exist inside `build_features`, not one

`features.py`'s docstring and BENCHMARK.md §6d describe a single
`extra_lag_hours` parameter (default **0**, shipped to both training and
serving — see `noctua/features.py:117-121`) that determines how many hours
of real information are discarded beyond the no-lookahead contract's minimum.
Reading the code line by line shows this parameter does **not** apply
uniformly to all 42 columns:

- **Group A — 31 columns** (all `har_*`, `semi_*`, `jump_*`, `rq_noise_1d`,
  `rng_*`, `mom_*`, `vov_*`, `reg_rv_vs_year`, `reg_vol_trend`,
  `reg_post_etf`): built into the `F` dict, then indexed at
  `prev_rows = rows - extra_lag_hours` (`features.py:242-248`). Effective lag
  = contract minimum + `extra_lag_hours`. At the shipped default (0), the
  freshest hour used is `a-1`. At the old default (1 — every number in
  BENCHMARK.md before §6d), it was `a-2`.
- **Group B — 3 columns** (`seas_1d`, `seas_5d`, `seas_22d`): computed
  *after* the `prev_rows` indexing, directly from `rows` (the anchor's own
  row) with their own boundary test `end <= rows` (`features.py:250-264`).
  This is **never parameterized by `extra_lag_hours`** — the seasonal window
  always ends at `a-1` regardless of the setting. Not previously called out
  in BENCHMARK.md.
- **Group C — 8 columns** (`cal_hour_sin/cos`, `cal_dow_sin/cos`, `cal_H`,
  `cal_weekend_frac`, `cal_month_sin/cos`): computed from `anchor_ts`, `H`,
  `dow` directly (`features.py:277-300`) — not lagged at all, and not
  supposed to be: this is the documented deliberate exception, calendar
  structure of the *forward* window, known years in advance.

31 + 3 + 8 = 42. This partition is corroborated by BENCHMARK.md §6d itself:
"31 of 42 features move" when `extra_lag_hours` went 1→0 — the 11 that don't
move are exactly Group B (3) + Group C (8).

### 0.3 One name checked against its code, as required

`jump_share_1d`/`jump_share_5d` do compute what the name says:
`max(sum(rv5,k) - sum(bpv5,k), 0) / (sum(rv5,k) + eps)` — realized variance
minus jump-robust bipower variation, floored at zero, as a fraction of total
RV. This is the standard Barndorff-Nielsen/Shephard jump-share decomposition.
`semi_signed_jump_1d`/`_5d`, despite also containing "jump" in the name, do
**not** touch bipower variation at all — they compute `(RV_pos - RV_neg) /
(RV_pos + RV_neg)`, the Patton–Sheppard **realized signed jump (RSJ)**
statistic, which is a semivariance-asymmetry measure, not a bipower-based
jump estimate. Both names are accurate to their respective literatures; they
just do not compute the same *kind* of "jump" as each other, and reading the
code was necessary to see that.

### 0.4 What is UNVERIFIED, and why

Every `event_time`/`feature_time` claim below is established from the code
itself (`build_features`, corroborated by `leakage.json`'s per-column
causality audit). One thing is **not** established this session:

> **UNVERIFIED: the true `publication_time` of the underlying Bitstamp trade
> relative to when it reaches this repo's training corpus.** `btcusd_1min.parquet`
> is built (`noctua/ingest.py`) from `ff137/bitstamp-btcusd-minute-data`, a
> GitHub mirror described in `model/RESEARCH_PLAN.md` as "daily-updated" and
> "verified live to 2026-08-10 01:57 UTC at time of writing" — a claim from a
> prior session, not re-verified in this one, and no network call was made
> this session to confirm the mirror's actual update lag against Bitstamp's
> own trades. This does not create a *leakage* risk (the leakage audit tests
> the code's array indexing against the committed parquet, and that passed
> 42/42); it means the offline training corpus's freshness relative to the
> live market is a separate, unmeasured question from whether the code reads
> the corpus correctly. Flagged rather than assumed.

Every other field below (`event_time`, `feature_time`, `source`, lag,
CAUSAL/structural-only) is code-established and none of the 42 columns
required an UNVERIFIED mark on those axes.

### 0.5 Family assignment is this document's judgment, not the code's

The 7 families (returns/trend, volatility/HAR, microstructure,
options/forward-looking, cross-asset, events/macro, calendar/regime) are the
protocol's, not `features.py`'s own. The module's own docstring groups
columns by *prefix* (`har_`, `seas_`, `semi_`, `jump_`, `rq_`, `rng_`, `mom_`,
`vov_`, `cal_`, `reg_`), which is a finer partition than the 7 families ask
for. The mapping below is a considered assignment onto the coarser scheme,
stated so the reader can disagree with a specific placement without having
to re-derive the whole table. The 42 columns **do** partition cleanly across
the 7 families with this mapping — no column needed to be split or forced —
and three of the seven families end up with **zero** columns, entirely
independent of where the boundary judgment calls landed.

---

## 1. Family: volatility / HAR — 11 columns

Trailing realized-volatility level and its range/vol-of-vol variants, all
derived from the hourly bar table. `NON_MODEL_COLS`/`SHAPE_COLS` membership
noted from `spec.py` where relevant.

| column | measures | window | feature_time (rel. anchor `a`) | effective lag (shipped) | leakage verdict |
|---|---|---|---|---|---|
| `har_1h` | log realized-vol rate, trailing 1h | 1h | ends `a-1` | contract-exact (0 extra) | CAUSAL |
| `har_6h` | log realized-vol rate, trailing 6h | 6h | ends `a-1` | contract-exact | CAUSAL |
| `har_1d` | log realized-vol rate, trailing 24h | 24h | ends `a-1` | contract-exact | CAUSAL |
| `har_5d` | log realized-vol rate, trailing 120h | 120h | ends `a-1` | contract-exact | CAUSAL |
| `har_22d` | log realized-vol rate, trailing 528h | 528h | ends `a-1` | contract-exact | CAUSAL |
| `rng_park_1d` | Parkinson range estimator, trailing mean, log scale | 24h | ends `a-1` | contract-exact | CAUSAL |
| `rng_gk_1d` | Garman-Klass range estimator, trailing mean, log scale | 24h | ends `a-1` | contract-exact | CAUSAL |
| `rng_park_5d` | Parkinson range estimator, trailing mean | 120h | ends `a-1` | contract-exact | CAUSAL |
| `rng_gk_5d` | Garman-Klass range estimator, trailing mean | 120h | ends `a-1` | contract-exact | CAUSAL |
| `vov_5d` | std dev of trailing log-RV series ("vol of vol") | 120h | ends `a-1` | contract-exact | CAUSAL |
| `vov_22d` | std dev of trailing log-RV series | 528h | ends `a-1` | contract-exact | CAUSAL |

- **event_time** (all rows): the hourly bars `[a-window, a-1]`, strictly
  before the anchor.
- **publication_time**: contemporaneous with each hour's close (Bitstamp
  trades are public in real time); §0.4's caveat applies to the offline
  corpus's own freshness, not to these values individually.
- **source**: `model/artifacts/btcusd_1h.parquet`, columns `rv5`, `high`,
  `low`, `close`, `open`, built by `build_hourly()` in `noctua/episodes.py`
  from `btcusd_1min.parquet` (Bitstamp, via `noctua/ingest.py`). Aggregated
  by `build_features()` in `noctua/features.py`.
- **Group A** lag rules apply (§0.2): shown lag is at the shipped default
  `extra_lag_hours=0`; at the old default (1) every window here ends one
  hour earlier (`a-2`).
- `har_1d`, `har_5d`, `har_22d` are also in `spec.py`'s `BASE_COLS` (Stage
  A's linear input). None of the range/vov columns are in `BASE_COLS`;
  `vov_5d`/`vov_22d` are in `SHAPE_COLS` (Stage B input); the `rng_*`
  columns are in **neither** `BASE_COLS` nor `SHAPE_COLS` nor
  `NON_MODEL_COLS` — they are computed and stored in `features.parquet` but
  not read by either stage's column list in `spec.py`. Worth flagging: this
  is a fourth category `spec.py` does not name (built, stored, but consumed
  by no model input list read this session), distinct from the deliberate
  `NON_MODEL_COLS` exclusion the `eff_*` columns get (§2).

---

## 2. Family: microstructure — 12 columns

Realized-measure decompositions built from the 5-minute return grid within
each hour: semivariance/signed-jump asymmetry, bipower-variation jump share,
quarticity-based noise attenuation, and the path-efficiency ratio.

| column | measures | window | feature_time | effective lag | leakage verdict |
|---|---|---|---|---|---|
| `semi_neg_share_1d` | share of trailing RV from downside 5-min returns | 24h | ends `a-1` | contract-exact | CAUSAL |
| `semi_signed_jump_1d` | Patton-Sheppard signed jump (RV⁺−RV⁻)/(RV⁺+RV⁻) | 24h | ends `a-1` | contract-exact | CAUSAL |
| `semi_neg_1d` | log realized downside-semivariance rate | 24h | ends `a-1` | contract-exact | CAUSAL |
| `semi_neg_share_5d` | share of trailing RV from downside returns | 120h | ends `a-1` | contract-exact | CAUSAL |
| `semi_signed_jump_5d` | signed jump (RV⁺−RV⁻)/(RV⁺+RV⁻) | 120h | ends `a-1` | contract-exact | CAUSAL |
| `semi_neg_5d` | log realized downside-semivariance rate | 120h | ends `a-1` | contract-exact | CAUSAL |
| `jump_share_1d` | (RV − bipower-variation)⁺ / RV — fraction of trailing RV from jumps | 24h | ends `a-1` | contract-exact | CAUSAL |
| `jump_share_5d` | (RV − bipower-variation)⁺ / RV | 120h | ends `a-1` | contract-exact | CAUSAL |
| `rq_noise_1d` | √(realized quarticity)/RV — HARQ noise-attenuation term | 24h | ends `a-1` | contract-exact | CAUSAL |
| `eff_1d` | log(high-low span)/√(trailing RV) — path travel per unit vol | 24h | ends `a-1` | contract-exact | CAUSAL |
| `eff_3d` | same ratio | 72h | ends `a-1` | contract-exact | CAUSAL |
| `eff_7d` | same ratio | 168h | ends `a-1` | contract-exact | CAUSAL |

- **event_time / publication_time / source**: same as §1 — hourly bars
  `[a-window, a-1]` from `btcusd_1h.parquet`, ultimately from Bitstamp
  1-minute bars via `noctua/episodes.py`'s 5-minute return grid
  (`rv5`, `rv5_pos`, `rv5_neg`, `bpv5`, `rq5` columns, built in
  `build_hourly()`).
- **Group A** lag rules apply (§0.2).
- **`eff_1d`/`eff_3d`/`eff_7d` are computed but not fed to the model.**
  `spec.py`'s `NON_MODEL_COLS = EFFICIENCY_COLS = ("eff_1d","eff_3d","eff_7d")`
  is an explicit exclusion list, with a comment explaining why: "adding a
  column to features.py silently widens the wide block Xa" otherwise. The
  ablation that tested them (`eval/efficiency.py`, per `spec.py`'s comment)
  found no gain; they are retained in `features.parquet` so the negative
  result stays reproducible, not because they are in the pipeline to stay.
  This is worth stating plainly: **3 of the 42 columns in features.parquet
  are dead to the model by design.**
- The other 9 columns in this family (`semi_*`, `jump_share_*`,
  `rq_noise_1d`) are in `SHAPE_COLS` (Stage B input) per `spec.py`.

---

## 3. Family: returns / trend — 5 columns

Momentum, distance-to-moving-average, and drawdown, all derived from the
hourly close series.

| column | measures | window | feature_time | effective lag | leakage verdict |
|---|---|---|---|---|---|
| `mom_ret_1d` | log return, close(`a-1-24h`) → close(`a-1`) | 24h | ends `a-1` | contract-exact | CAUSAL |
| `mom_ret_5d` | log return over 120h | 120h | ends `a-1` | contract-exact | CAUSAL |
| `mom_ret_22d` | log return over 528h | 528h | ends `a-1` | contract-exact | CAUSAL |
| `mom_dist_ma100` | log-distance, close(`a-1`) to trailing 2400h (~100d) mean log-close | 2400h | ends `a-1` | contract-exact | CAUSAL |
| `mom_drawdown_90d` | log-distance, close(`a-1`) to trailing 2160h (~90d) running max log-close | 2160h | ends `a-1` | contract-exact | CAUSAL |

- **event_time / publication_time / source**: hourly close series
  `[a-window, a-1]` from `btcusd_1h.parquet` (Bitstamp).
- **Group A** lag rules apply. All five are in `spec.py`'s `SHAPE_COLS`.
- These carry the two longest lookback windows in the whole feature set
  (2400h, 2160h) — the reason `serve/history.py`'s committed hourly bundle
  exists at all is that a naive 30-minute cron cannot re-fetch 100 days of
  5-minute bars per forecast (see DATA_LINEAGE.md §4).

---

## 4. Family: calendar / regime — 14 columns

Two genuinely different kinds of feature share this family by the protocol's
naming: backward-looking regime indicators (RV-vs-history, volume trend,
post-ETF flag, same-clock-window seasonal RV) and forward-looking clock/
calendar encodings of the anchor and its forecast window.

### 4a. Regime and seasonal (backward-looking) — 6 columns

| column | measures | window | feature_time | effective lag | leakage verdict | structural-only |
|---|---|---|---|---|---|---|
| `reg_rv_vs_year` | log(trailing 1d mean RV rate) − log(trailing 365d mean RV rate) | 24h vs 8760h | ends `a-1` | contract-exact | CAUSAL | no |
| `reg_vol_trend` | log(trailing 5d mean volume) − log(trailing 22d mean volume) | 120h vs 528h | ends `a-1` | contract-exact | CAUSAL | no |
| `reg_post_etf` | binary: `hour_ts >= 2024-01-11 00:00 UTC` (spot-ETF-launch flag) | n/a (constant threshold) | reads `hour_ts` at `a - extra_lag_hours` | contract-exact | CAUSAL | **yes** |
| `seas_1d` | log RV over the same H-hour clock window, 1 day back | H hours, offset 24h | ends `a-1` (NOT parameterized by `extra_lag_hours`, see §0.2 Group B) | always contract-exact regardless of setting | CAUSAL | no |
| `seas_5d` | same, 5 days back | H hours, offset 120h | ends `a-1` (Group B) | always contract-exact | CAUSAL | no |
| `seas_22d` | same, 22 days back | H hours, offset 528h | ends `a-1` (Group B) | always contract-exact | CAUSAL | no |

- **event_time**: hourly bars strictly before `a`, per window above. For
  `seas_*` the window is `H` hours long (the same length as the forecast
  horizon) starting `d*24` hours before `a` — i.e. what realized vol looked
  like the last time this exact horizon-length window occupied this same
  clock position, `d` days ago.
- **source**: `btcusd_1h.parquet` for the RV/volume/timestamp inputs.
- **`reg_post_etf` is `structural-only`** per `leakage.json`: it is flagged
  in `structural_only_columns` because it keys only off `hour_ts` compared
  to a hardcoded constant, so the audit's corruption test (which perturbs
  `rv5`/`close`/`high`/`low`/`volume`/etc.) cannot stress it — a CAUSAL
  verdict here is guaranteed by construction (the column literally cannot
  read those fields), not demonstrated by the empirical test. Necessary but
  not sufficient, exactly as `leakage.py`'s own docstring states.
- **`seas_*`'s lag exemption is this document's own finding** (§0.2), not
  called out in BENCHMARK.md or `leakage.json` — it changes nothing about
  the CAUSAL verdict (both lag groups pass the contract), but it means a
  future change to `extra_lag_hours` would move 31 columns, not 34, and
  three of the "regime" family's members would not move with it.

### 4b. Clock and forward-window calendar (deliberately forward-looking) — 8 columns

| column | measures | feature_time | leakage verdict | structural-only |
|---|---|---|---|---|
| `cal_hour_sin` | sin encoding of anchor hour-of-day | at anchor, from `anchor_ts` | CAUSAL | **yes** |
| `cal_hour_cos` | cos encoding of anchor hour-of-day | at anchor | CAUSAL | **yes** |
| `cal_dow_sin` | sin encoding of anchor day-of-week | at anchor | CAUSAL | **yes** |
| `cal_dow_cos` | cos encoding of anchor day-of-week | at anchor | CAUSAL | **yes** |
| `cal_H` | horizon length, `H/24` (days) | at anchor (a model input, not data) | CAUSAL | **yes** |
| `cal_weekend_frac` | fraction of the FORWARD `[a, a+H)` window landing on Sat/Sun | at anchor, computed over the forecast window | CAUSAL | **yes** |
| `cal_month_sin` | sin encoding of anchor's calendar month | at anchor | CAUSAL | **yes** |
| `cal_month_cos` | cos encoding of anchor's calendar month | at anchor | CAUSAL | **yes** |

- **event_time**: for `cal_hour_*`/`cal_dow_*`/`cal_month_*`/`cal_H`, the
  anchor instant itself — these describe the anchor, not a trailing window.
  For `cal_weekend_frac`, the event_time is explicitly the **forecast
  window** `[a, a+H)`, i.e. it describes time *after* the anchor.
- **publication_time**: the calendar is known deterministically for all
  time — there is no publication delay to speak of. `features.py`'s own
  docstring states this is the single deliberate exception to the
  no-lookahead contract and defends it explicitly: "The calendar is known
  years in advance, so using it is not lookahead."
- **source**: `episodes.parquet` columns `anchor_ts`, `H`, `dow` (built by
  `build_episodes()` in `noctua/episodes.py` from `anchor_ts` arithmetic —
  no market data at all), consumed directly in `build_features()`
  (`features.py:277-300`).
- **All 8 are `structural-only`** per `leakage.json`: none of them read the
  hourly OHLCV/RV columns the audit corrupts, so — like `reg_post_etf` — a
  CAUSAL verdict here is guaranteed by not touching the tested data, not
  demonstrated against it. This is the entire `structural_only_columns`
  list from the audit (9 total: these 8 + `reg_post_etf`).
- `cal_hour_sin/cos`, `cal_dow_sin/cos`, `cal_H`, `cal_weekend_frac` are in
  `spec.py`'s `SHAPE_COLS` and `BASE_COLS` (`cal_H`, `cal_weekend_frac`
  only, for `BASE_COLS`). `cal_month_sin/cos` are in **neither** list —
  computed and stored, consumed by no model input list read this session
  (same fourth category noted in §1 for the `rng_*` columns).

---

## 5. Families with ZERO columns — the model's blind spots

Three of the protocol's seven families have **no representation at all**
among the 42 columns, and this is a Phase 1 finding in its own right, not an
omission of this document:

- **options / forward-looking — 0 columns.** No implied-volatility or
  options-market input exists anywhere in `features.py`. `dvol_btc.parquet`
  (Deribit DVOL, the implied-vol index) has been harvested
  (`data/newdata/dvol_btc.parquet`, 47,563 rows, 2021-03-24 → 2026-08-26 per
  `leakage.json`) but is grep-confirmed absent from both `features.py` and
  `spec.py` — nothing joins it in. BENCHMARK.md §12/§15 name this directly:
  the model's onset-prediction ceiling (AUC 0.733) exists "because every
  current input derives from BTC's own past bars," and implied volatility is
  identified as the one candidate that is forward-looking by construction.
  Phase 4 (E2, implied-volatility family) targets this gap and remains
  closed pending Phase 0's gate.
- **cross-asset — 0 columns.** No ETH/SOL/XRP/LTC/other-instrument column
  exists in this 42-column set. `data/assets/*.parquet` (harvested via
  `.github/workflows/harvest-assets.yml`) is used only by separate ablation
  scripts (BENCHMARK.md §6c's pooled-training experiment), not by
  `features.py` — the 42 columns that ship are BTC-only.
  `funding_btc.parquet` and `dvol_btc.parquet` are themselves BTC-market
  data (Deribit BTC-PERPETUAL funding, BTC DVOL), not other-asset data, so
  they would not fill this family even once wired in.
- **events / macro — 0 columns.** No news, macro-economic, or scheduled-event
  input exists in `features.py`. `fetch-data.yml` does fetch news
  (CryptoPanic/GDELT/Exa) and Fear & Greed into `data/news.json`/`data/fg.json`
  for the live dashboard's own display, but grep-confirms neither is
  consumed by `noctua/features.py` or referenced in `noctua/spec.py` — that
  data feeds the website, not the model.

**Non-empty families, for contrast:** volatility/HAR (11), microstructure
(12), returns/trend (5), calendar/regime (14) — summing to 42.

---

## 6. Summary

| family | columns | empty? |
|---|---|---|
| volatility / HAR | 11 | no |
| microstructure | 12 | no |
| returns / trend | 5 | no |
| calendar / regime | 14 | no |
| options / forward-looking | 0 | **yes** |
| cross-asset | 0 | **yes** |
| events / macro | 0 | **yes** |
| **total** | **42** | — |

**Documentation status against the Phase 1 gate:** all 42 columns have a
code-established `event_time`, `feature_time`, and `source`. All 42 carry a
CAUSAL verdict from `leakage.json`, and 9 of those 42 (`reg_post_etf` + all 8
`cal_*`) are flagged `structural-only` — CAUSAL by construction because the
audit's corruption mechanism cannot reach the fields they actually read, not
because the empirical test stressed them, per §4. One caveat applies
uniformly and is stated once rather than 42 times: `publication_time` for
the offline training corpus's ingestion cadence relative to true Bitstamp
trade time is **UNVERIFIED** (§0.4) — no network call was made this session
to confirm the GitHub mirror's own update lag. This does not weaken the
causality verdicts, which are about the code's array indexing against the
committed parquet, not about the parquet's freshness relative to the market.
No column in this catalog required a silent omission; every row above is
either fully documented or explicitly marked with the reason it is not.

*Educational research only. Not financial advice.*
