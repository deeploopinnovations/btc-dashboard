# DATA_LINEAGE — Phase 1

Traces every dataset the model touches from raw source to model input, per
`model/PHASES.md`'s Phase 1 deliverable. Read from `model/noctua/ingest.py`,
`model/noctua/episodes.py`, `model/noctua/features.py`, `model/serve/history.py`,
`model/serve/predict.py`, `.github/workflows/*.yml`, `model/eval/leakage.json`,
`model/eval/datasources.py`, `model/eval/harvest_newdata.py`, and
`BASELINE_MANIFEST.md`. Nothing in `model/serve/`, `model/noctua/`, or
`model/eval/` was modified to produce this document.

---

## 1. The core pipeline: 1-minute bars → features

Four stages, three artifacts, one code path per arrow. All four stages read
their input from a file and write their output to a file — there is no
in-memory-only step.

```
btcusd_1min.parquet  --[noctua/episodes.py: build_hourly()]-->  btcusd_1h.parquet
btcusd_1h.parquet    --[noctua/episodes.py: build_episodes()]--> episodes.parquet
btcusd_1h.parquet +
episodes.parquet     --[noctua/features.py: build_features()]--> features.parquet
```

### Stage 0 → 1: raw CSV → `btcusd_1min.parquet`

- **Source**: `ff137/bitstamp-btcusd-minute-data` (MIT-licensed GitHub repo),
  described in `model/RESEARCH_PLAN.md` §3.1 as BTC/USD 1-minute OHLCV on
  Bitstamp, `2012-01-01` → present, "daily-updated," "~7.68 M 1-minute bars."
  That span/row-count claim is from a prior session's citation of the mirror
  and was not re-fetched or re-verified this session (no network call was
  made to the mirror this session; see §6).
- **Code**: `noctua/ingest.py` — `ingest(repo, out_dir)`. Reads
  `data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz` (bulk) and
  `data/updates/btcusd_bitstamp_1min_latest.csv` (daily append) from a local
  clone of the mirror, concatenates, deduplicates on `timestamp`, reindexes
  onto a strict 1-minute grid (missing minutes forward-filled and flagged in
  a `filled` column), repairs OHLC internal-consistency violations, and flags
  isolated "bad print" wicks (§1's docstring: a bar whose high/low excursion
  is large, extreme relative to trailing MAD, and leaves no trace in
  neighbouring closes — carried as a `bad_print` flag, not deleted).
- **Invocation**: `python -m noctua.ingest --repo <clone> --out <dir>`.
  **Not run by any GitHub Actions workflow** — grep of `.github/workflows/`
  finds zero references to `ingest.py` or `noctua.ingest`. This stage is run
  ad hoc/manually against a local clone of the mirror, not on a schedule.
- **Output**: `model/artifacts/btcusd_1min.parquet` — 157,328,807 bytes per
  `BASELINE_MANIFEST.md` §3 (partial-hashed: size + md5 of first/last 1 MB
  only, full hash exceeds the container's command timeout).

### Stage 1 → 2: `btcusd_1min.parquet` → `btcusd_1h.parquet` + `episodes.parquet`

- **Code**: `noctua/episodes.py`. `build_hourly(df)` aggregates 1-minute
  bars to an hourly table: OHLCV, a "clean" high/low with bad-print wicks
  neutralised, and five realized-measure columns computed on a 5-minute
  return grid (`rv5`, `rv5_pos`, `rv5_neg`, `bpv5`, `rq5`) plus `rv1` (1-minute
  RV, kept only for sensitivity — explicitly never fed downstream because it
  correlates 0.953 with the forward target, per the code comment at
  `episodes.py:213-215`). `build_episodes(hours, horizons)` then builds one
  row per (anchor hour, horizon) with `S_tau` = close of hour `a-1`, forward
  labels `R`/`M_up`/`M_dn`/`RV`/`RV1` over `[a, a+H)`, and a `row` column
  pointing back into the hourly table for feature joining.
- **No-lookahead contract** (`episodes.py:31-35`, verbatim): `S_tau` = close
  of hour `a-1`; labels = hours `[a, a+H)`; features = hours `<= a-1`.
- **Invocation**: `python -m noctua.episodes --parquet btcusd_1min.parquet
  --out <dir>`. Default horizons `H ∈ {6, 12, 19, 24}`. Like Stage 0→1,
  **not run by any workflow** — this is a manual/research-session step.
- **Output**: `btcusd_1h.parquet` (13,094,438 bytes) and `episodes.parquet`
  (24,724,722 bytes, 510,496 rows across all horizons — `BASELINE_MANIFEST.md`
  §3-4).

### Stage 2 → 3: hourly bars + episodes → `features.parquet`

- **Code**: `noctua/features.py`. `build_features(hours, episodes,
  extra_lag_hours=0)` builds the 42 columns cataloged in `FEATURE_CATALOG.md`
  — see that document for per-column detail. `audit_lookahead()` in the same
  file numerically re-verifies the contract by corrupting future bars and
  confirming no probed feature moves (0.000e+00 over 306,261 episodes per
  `leakage.json`'s `baseline_audit_lookahead`).
- **Invocation**: `python -m noctua.features --artifacts <dir>
  [--extra-lag-hours N] [--audit]`. Not run by any workflow — manual, same
  as Stages 0→1 and 1→2. `BASELINE_MANIFEST.md` records `features.parquet`'s
  mtime (2026-08-16) postdating `episodes.parquet`'s (2026-08-10), i.e. the
  feature matrix was last regenerated separately from — and after — the
  hourly/episode tables, consistent with a manual re-run rather than a
  scheduled one.
- **Output**: `features.parquet` (60,883,811 bytes, 510,496 × 42 —
  `BASELINE_MANIFEST.md` §3-4).

**Consequence of "not run by any workflow"**: reproducing this pipeline from
scratch is a manual, multi-command process (clone the mirror → `ingest` →
`episodes` → `features`), not something CI exercises end to end. `model-ci.yml`
(the PR gate) explicitly skips it — its own header comment states it
"exercises everything that does not need market data or the ~160 MB training
parquet." No workflow in this repository regenerates `btcusd_1min.parquet`,
`btcusd_1h.parquet`, `episodes.parquet`, or `features.parquet`.

---

## 2. The parallel serving path — same feature code, different raw data

Training and serving share `build_features()` and `build_hourly()` by
design (`features.py`'s docstring: "Serving calls this function with no
argument, so the default is what training and production BOTH get") but they
do **not** share the same raw-data ingestion path:

- **Training/backtest path** (§1 above): static parquet files built from the
  Bitstamp GitHub-mirror CSVs, regenerated manually.
- **Serving path**: `serve/history.py` + `serve/fetch.py` + `serve/predict.py`.
  `data/noctua_history.parquet` (~400 days of hourly aggregates, "built by
  the same `build_hourly` code as training" per `history.py`'s docstring) is
  committed to the repo and topped up by a **live 5-minute feed** each run;
  `serve/predict.py` calls `build_features(hours, ep)` directly on the
  merged bundle. `.github/workflows/fetch-data.yml` runs this every 30
  minutes (`cron: '*/30 * * * *'`), committing `data/noctua.json` /
  `data/kronos.json` and topping up `data/noctua_history.parquet`; the
  bundle file itself is rewritten roughly weekly per `history.py`'s
  `PERSIST_AFTER_HOURS` comment.
  `history.py`'s own incident note: an early cron run failed with
  `fetch_bitstamp: only 83.2h of history, need >= 528h` because
  `LOOKBACK_HOURS` had been set assuming the 22-day HAR term was the longest
  window in the feature set — it is not (`mom_dist_ma100` needs 2400h,
  `reg_rv_vs_year` needs 8760h). The committed bundle exists specifically to
  avoid re-fetching ~105,000 5-minute candles per forecast.

**Both paths produce hourly tables with the same schema, via the same
`build_hourly()` function**, but they are two different data-freshness
regimes: the training corpus's Stage 0→1 freshness relative to true market
time is UNVERIFIED this session (§6); the serving path's freshness is
bounded by the 30-minute cron and the live-feed's own completeness rule
("an hour is only accepted from the live feed if the fetch contains all 12
of its 5-minute bars," per `history.py`).

---

## 3. The new data: `funding_btc.parquet` and `dvol_btc.parquet`

**Both files are harvested and committed but NOT used by the model.** Grep
of `noctua/features.py` and `noctua/spec.py` for `fund`/`dvol` (case
insensitive) returns zero matches — nothing in the feature-building or
model-input-column code reads either file. This is stated explicitly because
the file names and the model's known weakness (§5 of `FEATURE_CATALOG.md`:
the options/forward-looking family is empty) could otherwise suggest
otherwise.

### 3.1 `data/newdata/funding_btc.parquet`

- **Source**: Deribit's `get_funding_rate_history` public endpoint
  (`https://www.deribit.com/api/v2/public/get_funding_rate_history`), with
  Binance's `fapi.binance.com/fapi/v1/fundingRate` as a fallback if Deribit's
  endpoint fails outright (`model/eval/harvest_newdata.py`). The harvester
  walks backward in 30-day chunks until three consecutive empty responses,
  rather than assuming a fixed depth.
- **Span measured this session** (from `model/eval/leakage.json`,
  `new_data.funding`, produced by `model/eval/leakage.py`'s
  `audit_new_data()` re-reading the committed parquet — not re-fetched):
  **64,206 rows, 2019-04-30 10:00:00 UTC → 2026-08-26 17:00:00 UTC**,
  monotonic increasing, 0 duplicated timestamps.
- **Cadence**: hourly. `spacing_seconds_value_counts` = `{3600: 64204,
  10800: 1}` — 64,204 of 64,205 gaps are exactly one hour; one is 3 hours.
- **Known gap**: `gap_locations` records exactly one: **2020-08-27 05:00:00
  UTC → 2020-08-27 08:00:00 UTC, a 3-hour gap.**
- **Timestamp semantics — what was established from the data's own shape,
  and what was not** (`leakage.json` `new_data.funding` and
  `audit_new_data()`'s docstring):
  - `prev_index_price` in row `i` equals `index_price` in row `i-1` for
    **100.0%** of rows (`fraction_matching: 1.0`). Read as evidence the
    stamped hour's `interest_1h` is computed from a price change *ending at*
    that timestamp — a trailing, already-known-by-`t` quantity, the same
    convention this repo's own hourly `close` uses — rather than a
    forward-settlement stamp (which would not need to reference the
    immediately preceding row's own realized price change to define
    itself).
  - `interest_8h` changes on **98.39%** of hourly rows
    (`interest_8h_changes_every_hour_fraction`), consistent with a rolling
    trailing-8h reference updated continuously, not a value that is only
    meaningful at the 00:00/08:00/16:00 UTC settlement instants.
  - **UNVERIFIED**: Deribit's own documented definition of the funding-rate
    `timestamp` field was not fetched or consulted this session (no network
    call was made) — the "known as of t" reading above is inferred from the
    data's internal structure, not confirmed against an API reference.
    `index_price`'s own residual publication delay relative to its
    timestamp (it is an average across constituent exchanges) was also not
    established.
- **Required lag if ever used as a feature** (`leakage.json`,
  `new_data.required_lag_if_used_as_a_feature`, established from this
  repo's own stated contract, `episodes.py:35`, "features = hours <= a-1"):
  a funding row stamped at hour `h` must not be joined to an anchor `a`
  until `a >= h+1` — the identical lag convention `build_features` already
  applies to the hourly OHLC table (`rows - extra_lag_hours`, never `rows`
  itself). Joining on `ts == anchor_ts` would manufacture exactly the kind
  of leak the audit exists to catch.
- **Training-window overlap**: BENCHMARK.md §14 reports 32,196 rows (50.1%,
  1,341 days) fall inside the training window as previously measured; this
  session's `leakage.json` re-measurement of row count/span (64,206 rows,
  2019-04-30 → 2026-08-26) is consistent with that citation but the 50.1%
  overlap figure itself was not independently recomputed this session — it
  is carried from BENCHMARK.md §14, not re-derived from `leakage.json`.

### 3.2 `data/newdata/dvol_btc.parquet`

- **Source**: Deribit's `get_volatility_index_data` endpoint (paged on
  `start_timestamp`/`end_timestamp`/`resolution`) — **not**
  `get_historical_volatility`, which BENCHMARK.md §14 documents as a
  different endpoint (Deribit's own *realized*-vol series, no time-range
  parameters, "there is nothing to page") that an earlier harvest mistakenly
  used, yielding only 383 rows / 15 days with zero training-window overlap.
  §15 documents the fix to the index endpoint.
- **Span measured this session** (`leakage.json`, `new_data.dvol`):
  **47,563 rows, 2021-03-24 00:00:00 UTC → 2026-08-26 18:00:00 UTC**,
  monotonic increasing, 0 duplicated timestamps, 0 non-hourly gaps
  (`spacing_seconds_value_counts: {3600: 47562}` — every single gap is
  exactly one hour).
- **Cadence**: hourly, fully regular (no gaps at all, unlike funding).
- **Timestamp semantics — UNVERIFIED**: DVOL is a single "volatility" column
  with no OHLC breakdown, so whether the hourly value is a point-in-time
  index level sampled *at* the hour, or an intra-hour aggregate (e.g. a
  candle close), **cannot be distinguished from the data alone**
  (`leakage.json`, `new_data.unverified`). The audit's stated conservative
  recommendation — treat it as known-as-of-its-timestamp, same lag as
  everything else — holds regardless of which is true, but which is true
  was not established.
- **Training-window overlap**: BENCHMARK.md §15 reports 15,552 rows inside
  the training window after the endpoint fix (up from 0 before). Not
  independently recomputed against `leakage.json` this session; carried from
  §15's citation.

Both files' `required_lag_if_used_as_a_feature` note applies identically to
DVOL (same contract, same reasoning).

---

## 4. GitHub Actions workflows

| workflow | schedule | writes | purpose |
|---|---|---|---|
| `fetch-data.yml` | every 30 min (`*/30 * * * *`) + manual | `data/kronos.json`, `data/noctua.json`, `data/news.json`, `data/fg.json`, tops up `data/noctua_history.parquet` | Live serving snapshot: runs the NOCTUA forecast on the live 5-minute feed, fetches news (CryptoPanic + GDELT, + Exa if a key is set) and Fear & Greed, smoke-tests, commits only on change. |
| `harvest-newdata.yml` | daily, `11 1 * * *` (shortly after the 17:00 UTC production anchor closes) + manual (`workflow_dispatch`, default `days=3500`) + on push to the harvester/workflow file | `data/newdata/funding_btc.parquet`, `data/newdata/dvol_btc.parquet` | Funding-rate and DVOL harvest from Deribit (with Binance funding fallback). Scheduled because both are live series with "real forward value every day they're not collected" (workflow header comment). |
| `harvest-assets.yml` | **not scheduled** — manual `workflow_dispatch` (default symbols `btc,eth,sol,xrp,ltc`, default `days=3300`) + on push to the harvester/workflow file | `data/assets/*_history.parquet` | One-off cross-asset hourly bundles for zero-shot testing (BENCHMARK.md §6c's pooled-training experiment). Not used by the 42-column production `features.py`. |
| `model-ci.yml` | on PR touching `model/**` etc., on push to `main`, + manual | none (test-only) | PR gate: imports, exported-weights parity, legacy JSON contract, existing snapshot smoke gate. Explicitly skips anything needing "market data or the ~160 MB training parquet" — does not exercise the ingest→episodes→features pipeline. |
| `kronos-eval.yml`, `recover-kronos-artifact.yml` | not inspected in depth this session | — | Named but not read for this document; out of scope for the core lineage question (Kronos is a separate/legacy forecast path per `serve/predict.py`'s docstring, distinct from NOCTUA). |

None of `fetch-data.yml`, `harvest-newdata.yml`, or `harvest-assets.yml`
invoke `noctua.ingest`, `noctua.episodes`, or `noctua.features` (verified by
grep, §1). The training pipeline's four artifacts are not workflow-managed.

---

## 5. The network architecture — load-bearing, not incidental

**This container cannot reach any exchange.** `model/eval/datasources.py`
probes 18 endpoints across the major venues (Binance, Kraken, OKX, Bybit, and
others per the script) and every one returns 403 through the egress proxy —
documented as "all 18 endpoints probed return 403" in BENCHMARK.md §14 and
independently corroborated by this session's read of `datasources.py`
(`"Policy denial (403 Forbidden)"` / `"Tunnel rejected (proxy 403)"` are the
two failure branches the script itself distinguishes). The container's
proxy allows only `raw.githubusercontent.com` and `gist.github.com`
(`harvest_newdata.py`'s docstring, citing `datasources.py`).

**The workaround, and why it is the architecture rather than a workaround**:
harvesting happens on **GitHub Actions runners**, which have ordinary
internet access, and the runner **commits the result back to the repository**
where the development container reads it as a plain file. This pattern is
identical across `harvest-assets.yml` and `harvest-newdata.yml`, and both
workflow files' own comments describe it as deliberate: "this repo's
development container reaches only raw.githubusercontent.com and
gist.github.com ... so the fetch happens here, on a runner that has ordinary
internet access, and the result is committed for offline use"
(`harvest-newdata.yml:12-15`). BENCHMARK.md §14 confirms the route "works end
to end": `harvest-newdata.yml` ran on a runner and committed
`data/newdata/{funding_btc,dvol_btc}.parquet` at commit `f27f818`, authored
by `github-actions[bot]`.

**Consequence for this document's own limits**: everything in §3 above that
is marked UNVERIFIED (Deribit's documented field semantics, DVOL's intra-hour
meaning, `index_price`'s own publication delay) is UNVERIFIED specifically
*because* no network call to an exchange is possible from this session's
container — the same boundary that makes the runner-harvest architecture
necessary in the first place also means this document cannot independently
confirm what the harvested files' source API promises, only what their
committed contents structurally imply.

`harvest_newdata.py`'s docstring adds one more layer worth recording
verbatim: Deribit itself was never one of the 18 probed endpoints in
`datasources.py`, so its blockedness from *this* container is "asserted by
extension, not itself re-probed." The harvest workflow instead orders its
fetch to try Deribit first specifically because `src/data.js` (the live
dashboard) already calls Deribit's public book-summary endpoint directly
from a **browser**, with no proxy and no rate complaint — reasoning that a
server-to-server call from a GitHub runner faces fewer obstacles than a
browser call, not a re-verified fact about the runner itself.

---

## 6. What this document could not verify

- **Bitstamp mirror ingestion lag** (§1, Stage 0→1): the "daily-updated"
  claim and the specific "verified live to 2026-08-10 01:57 UTC" freshness
  citation both come from `model/RESEARCH_PLAN.md`, a prior session's
  citation not re-fetched this session — no network call to
  `github.com/ff137/bitstamp-btcusd-minute-data` was made.
- **Deribit funding-rate `timestamp` field semantics** (§3.1): inferred from
  data shape (`prev_index_price` chaining, `interest_8h` update frequency),
  not confirmed against Deribit's API documentation.
- **DVOL's intra-hour meaning** (§3.2): point-in-time index level vs.
  intra-hour aggregate cannot be distinguished from the data alone.
- **`index_price`'s own publication delay** relative to its timestamp
  (§3.1): not established either way.
- **funding's 50.1%/dvol's 15,552-row training-window-overlap figures**:
  carried from BENCHMARK.md §14/§15 citations, not independently
  recomputed against the freshly re-measured span in `leakage.json` this
  session (the span and gap-count figures in §3 above *were* independently
  measured this session, via `leakage.json`; the overlap-percentage figures
  were not re-derived).
- **`kronos-eval.yml` / `recover-kronos-artifact.yml`**: not read this
  session; out of scope for the core btcusd→features lineage question this
  document answers.

*Educational research only. Not financial advice.*
