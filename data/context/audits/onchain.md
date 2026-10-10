# Lookahead audit: data/context/onchain.parquet

Auditor run: 2026-10-10 (retrieved_at in the file: 2026-10-10 11:58 UTC).
Scope: model/context/SPEC.md, pit.py, features.py (onchain group), sources/onchain.py.
Nothing in data/ or the fetch script was modified. Scratch scripts live in the session scratchpad.

## Verdict: FAIL (scoped)

- Mechanical point-in-time handling in features.py: PASS.
- Daily and intraday timestamp conventions: PASS (one unverified availability assumption, noted).
- Past halving rows: PASS.
- Exchange-flow features (`onc_exch_net7_pct`, `onc_exch_sply_chg30`) used in backtests: FAIL. They are built from revisable, never-reviewed Coin Metrics data stored as a single 2026 vintage.
- Next-halving estimate row: FAIL as stored (leaks 2026 chain data back to 2009). Latent, because no feature reads it today.

## Confirmed defects

### D1. Exchange flows and supply are revised history, stored as one 2026 vintage (CONFIRMED structure; revision size NOT measured)

Evidence:
- Single vintage. All 442,350 rows share `retrieved_at` = 2026-10-10 11:58. No prior snapshots exist, so any backtest at t reads values as they stand in October 2026.
- Coin Metrics marks FlowInExNtv, FlowOutExNtv and SplyExNtv as `flash` (status text stored in `text`) for their entire history: 5,648 rows each, back to 2011-04-24. None has reached `reviewed`.
- Those old rows were rewritten long after the fact. `FlowInExNtv-status-time` from the live API:
  - 2011-05-01: 2026-04-02
  - 2013-01-01: 2026-04-02
  - 2017-12-01: 2026-04-09
  - 2019-06-01: 2026-04-07
  - 2021-06-01: 2026-04-15
  - 2023-01-01: 2026-04-15
  - 2025-01-01: 2026-04-13
  - 2025-09-01: 2026-04-16
  - 2026-03-01: 2026-04-18
  - 2026-09-01: 2026-09-02 (ordinary first write)
  Pre-April-2026 values were last written in April 2026, months to years after the day they describe.
- Coin Metrics docs (service-and-support/faqs): "Certain data types can be subject to look-ahead bias because observations are collected, revised, and finalized over time... A dataset queried today reflects everything Coin Metrics now knows, not what it knew at any earlier instant." Exchange flows are estimated by the common-input-ownership heuristic and depend on a "predetermined universe" of seed exchanges, so the tagging set can change.
- Short-horizon check: 900 stored CM flow/supply/active/MVRV rows from 2026-04-14 on match a fresh pull exactly, and four blockchain.com series (n-transactions, difficulty, hash-rate, fees) match in full. So changes are not visible within hours. The exposure is the multi-month rewrite above.
- features.py `onchain()` reads these rows through `pit.asof` and `_window_sum` with no gate on `pit_quality == "revisable"` or on status. SPEC rule 5 asks for a first-release vintage when one exists. The community API does not offer one.

Impact: a 2018 to 2025 backtest of the two exchange-flow features sees values rewritten in April 2026. Magnitude is unknown, because the API cannot return past vintages.

Proposed fix:
1. Immediately: exclude `onc_exch_net7_pct` and `onc_exch_sply_chg30` from any backtest or model selection on history. Permit them only for dates after the start of forward snapshot capture.
2. Start an append-only snapshot job: fetch the flow, supply and status fields daily and write each fetch's `retrieved_at`. Set each row's `available_ts` to the first `retrieved_at` at which that value was seen, not `event_ts + 6h`. Keep the vintages. Do not overwrite existing files.
3. Store the `-status` and `-status-time` fields as columns. Use `status-time` as a lower bound on availability for the value.
4. In features.py, refuse `pit_quality == "revisable"` series unless the caller passes an explicit opt-in flag. Add a test that fails if a revisable series enters a backtest feature group.

### D2. Next-halving estimate leaks the 2026 chain state into 2009-available data (CONFIRMED; latent)

Evidence:
- The row is `event_ts` 2028-04-12 19:08 UTC, `available_ts` 2009-01-03 (`pit_quality` = `scheduled`), value 1050000.
- Source code: `est = RETRIEVED + (1050000 - tip) * 600`, with tip 970757 measured on 2026-10-10. The date therefore encodes the 2026 chain height and clock, plus an assumed 600 s block time.
- `pit.next_scheduled(df, "halving", 2020-06-01 17:00 UTC)` returns 68,930 h, implying 2028-04-12 19:08. The same method applied in 2020 (block 630000 time plus 420000 x 600 s) gives 2028-05-06 11:23. The stored date is 23.7 days earlier, which is 2026 information in a 2020 feature. The realized block interval since block 840000 is 597 s, so the stored date reflects realized 2024 to 2026 pace.
- SPEC defines `scheduled` as a calendar entry known in advance. A wall-clock projection from the current tip is not that.
- Latent: `grep` shows no feature in model/context/features.py or evaluate.py reads the `halving` source. `pit.load_all()` does include it, so any future `next_scheduled('halving', ...)` leaks.

Proposed fix:
1. Keep only the protocol fact (height 1050000) as a `scheduled` row with an `available_ts` of 2009-01-03, if desired.
2. Drop the wall-clock date from the dataset. If an ETA is needed, compute it in features.py from the chain height known at t and the realized block interval up to t.
3. If the date must be stored, tag it `revisable` with `available_ts` = the retrieval time (2026-10-10 11:58). It then cannot be seen in any backtest before 2026, which is conservative.

## Checks that passed

### P1. Daily timestamp convention: stamp D covers calendar day D (CONFIRMED)

Method: blockstream.info blocks from height 970757 (tip) back to 2026-10-04, tallied by UTC header date, compared with Coin Metrics `BlkCnt` and blockchain.com `n-transactions`.

| day | blockstream blocks | CM BlkCnt stamped D | blockstream tx sum | blockchain.com n-tx stamped D |
|---|---|---|---|---|
| Oct 4 | 143 | 143 | 714,491 | 714,491 (parquet) |
| Oct 5 | 177 | 177 | 904,518 | 899,894 |
| Oct 6 | 155 | 155 | 749,817 | 754,441 |
| Oct 7 | 154 | 154 | 749,255 | 749,255 |
| Oct 8 | 139 | 139 | 630,220 | 630,220 |
| Oct 9 | 146 | 146 | 657,407 | 657,407 |

If stamp D covered D+1 (end-stamped), the counts would be shifted by one day. They are not. Stamp D covers D. The loader's `event_ts = D+1 00:00` (SPEC rule 3) is correct.

The Oct 5 and Oct 6 n-transactions differ from the header-time tally by 4,624 (block 970101, header time 2026-10-05 23:59:35, counted by blockchain.com on Oct 6). This reflects a different day-assignment rule, not lookahead, since the block is still within the stamped day's neighborhood. Minor.

Difficulty: retarget block 969696 at 2026-10-03 07:16:36 UTC. The stamp for Oct 3 is 1.327310e14, a time-weighted blend of the old and new values. The stamp for Oct 4 is 1.327160e14, the new difficulty. Again stamp D covers D.

Availability lag: Coin Metrics `-status-time` for the last six daily rows (Oct 4 to Oct 9) is 01:13 to 02:43 UTC the next day. The assumed 6 h lag is therefore conservative for recent data. Coin Metrics historical rows have later rewrites (D1).

### P2. Mempool intraday rows (CONFIRMED no lookahead; event semantics partly SUSPECTED)

Raw API point x = 2026-10-10 10:30 (value 41,467,396) is stored with `event_ts` 10:45, `available_ts` 11:45. The raw point at 10:45 is excluded because its `available_ts` is 12:00, after retrieval at 11:58. `event_ts` is the end of the 15-minute interval, not the snapshot instant. That is conservative. The API reports period `minute` and does not say whether values are snapshots or windows. Either way the value is available at least 75 min after its stamp, so there is no leakage. Zero rows have `available_ts > event_ts + 1h`.

### P3. Past halvings (CONFIRMED)

Heights 210000, 420000, 630000 and 840000 are stored with block timestamps from blockstream: 2012-11-28 15:24, 2016-07-09 16:46, 2020-05-11 19:23, 2024-04-20 00:09 UTC. `available_ts` = block time + 2 h, `pit_quality` = `lagged`. Consistent with SPEC. The fallback path (unverified date plus 2 days) was not used.

### P4. Feature code is point-in-time (CONFIRMED by perturbation)

Test: for 40 random 17:00 UTC anchors between 2014 and 2026, every row with `available_ts >= t` had its value replaced with `value * 1.7 + 12345`. Then `features.onchain(ctx, [t])` was recomputed. The maximum change in any of the nine onchain features was 0.0 for every anchor. Windows are `[t-k, t)` (`_window_sum`). At 17:00 UTC on day T, the newest daily row used is D = T-1 (available T 06:00).

## Other notes (not defects)

- `_window_sum` has no `max_age`. A stalled upstream series would produce stale sums silently. Not lookahead.
- `fetch_coinmetrics` keeps a partial history if pagination fails and only logs the failure.
- The loader discards `-status-time`, which is the data needed for D1 fix 3.
- Blockchain.com revision size: no status field and no vintage, so not measured. Covered by D1 reasoning for the dataset as a whole, but the blockchain.com daily series are not confirmed as revised.
- The blockchain.com 6 h availability for daily rows is an assumption. Its first publication time was not observed (SUSPECTED).

## Reproduction

- Stamp convention: block tally from `https://blockstream.info/api/blocks/{tip}` paged backward, compared with `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc&metrics=BlkCnt,...&frequency=1d`.
- Status times: `...asset-metrics?assets=btc&metrics=FlowInExNtv&frequency=1d&start_time=<date>T00:00:00&paging_from=start&page_size=1`. Note that `paging_from=start` is required, since the default returns the newest rows.
- PIT perturbation: `python3 -I` with `sys.path` set to `model/`, calling `context.features.onchain` on `pit.load(onchain.parquet)` after corrupting rows with `available_ts >= t`.
