# Point-in-time context dataset for BTC

Real-world context for NOCTUA-Trader, recorded the way a trader could have
seen it. Every row carries the moment it became public (`available_ts`) and
the URL it came from. The record contract is `model/context/SPEC.md`; the
loader and as-of join are `model/context/pit.py`.

Collected 2026-10-10 by Haiku 5.5 workers (one per source), then attacked by
separate Haiku auditors whose reports are in `audits/`. Every defect an
auditor confirmed was fixed before the tests below were run.

## What is here

| file | what | from | how `available_ts` is set | caveat |
|---|---|---|---|---|
| `macro_calendar.parquet` | FOMC statement schedule, target rate, emergency moves; CPI and NFP release schedule and first prints | 2014 | schedules: Jan 1 of the year; prints: the 08:30 ET release (ALFRED first vintage) | 2025-26 shutdown moves recorded with cancel rows; the cancelled March 2020 FOMC meeting has no row |
| `macro_daily.parquet` | 10y/2y yields, dollar index, VIX, S&P 500, Nasdaq, fed funds, breakevens, HY spread, WTI; stablecoin market cap | 2012 (S&P 2016, HY 2023, stablecoins 2017-11) | US close D: D+1 12:00 UTC; WTI D+9; dollar index the Tuesday after; fed funds next business day | dollar index and stablecoins are today's revised values |
| `derivatives.parquet` | Binance BTCUSDT open interest, top-trader and global long/short ratios, taker ratio (hourly); Binance funding; Deribit expiry calendar | OI 2020-09, funding 2020-01 | snapshot + 5 min; funding at its time | Binance re-uploaded many archive files in 2026, so tagged revisable; no liquidation history exists in the free archive |
| `onchain.parquet` | blockchain.com hash rate, difficulty, transactions, mempool, fees, addresses; Coin Metrics active addresses, MVRV, exchange flows; halvings | 2009 | day D: D+1 06:00 UTC | all revisable; **exchange flows are not used as features** because Coin Metrics rewrites them (audit D1) |
| `attention.parquet` | Crypto Fear & Greed; Wikipedia pageviews for "Bitcoin" and "Cryptocurrency" | F&G 2018-02, Wikipedia 2015-07 | F&G +12 h; Wikipedia D+2 12:00 UTC | GDELT news volume is not included: the API rate-limited this container into uselessness |
| `etf_flows.parquet` | US spot ETF daily net flows, total and per fund | 2024-01-11 | D+1 12:00 UTC | starts after the training period, so it cannot be tested yet |
| `explain_only/causes.csv` | hindsight causes for the 150 largest daily moves (2017-2026) | | | **never a feature**: `pit.load` refuses this folder. The fact-check (`audits/explain_only.md`) found 9 of 30 sampled rows fully supported, so treat it as a reading list, not ground truth |

## How a source is tested

`model/context/evaluate.py`, on the trader's own dataset (`model/policy/dataset.py`)
and its v1 configuration. Fit on train and validation, scored once on the
test split (2024-07-01 onward, 826 daily decisions at 17:00 UTC), which is
out of sample for NOCTUA itself.

- **Placebo:** the same features rotated in time by a random offset of at
  least 90 days within each split. Values, distribution and rhythm are kept;
  only the alignment with the market is destroyed. 99 placebos per source.
  (A plain day shuffle was also run; see `evaluation/*_shuffle.json`.)
- **Counts only if** the real features beat the base trader AND beat at
  least 95% of their own placebos.
- Two forecast checks run alongside: ridge regressions for next-24 h
  volatility and direction.

## Results

| source | trader test Sharpe (base 0.48) | placebo median | placebo p | vol forecast vs base | direction vs base | verdict |
|---|---|---|---|---|---|---|
| derivatives positioning | **0.81** | 0.39 | **0.01** | -3.5% (worse) | -11.8% (worse) | passes the trader rule |
| cross-market macro | **0.75** | 0.41 | 0.02 | -0.6% | -0.2% | passes the trader rule |
| macro calendar | 0.37 | 0.43 | 0.91 | +0.3% (ns) | -2.1% | does not count |
| on-chain | 0.41 | 0.42 | 0.71 | -1.2% | -1.7% | does not count |
| attention | 0.43 | 0.45 | 0.56 | -4.2% | -6.5% | does not count |
| ETF flows | | | | | | untestable (no training-period data) |

CAGR is compounded. With derivatives the trader made 4.1% a year at a
-4.9% worst drawdown, against 2.7% and -7.6% for the base trader.

**How much to believe it.** Two of five sources clear the rule, but:

- five sources were tested, so a p of 0.02 is not convincing on its own;
  0.01 is the smallest p 99 placebos can give.
- neither source improves the volatility or direction forecasts, so the gain
  is not a cleaner signal the trader found but something about how it sizes
  risk with these inputs. That is unexplained.
- the paired bootstrap of Sharpe(real) minus Sharpe(one placebo) has a 95%
  interval that includes zero for both.
- the derivatives and macro sources were first scored before their audit
  fixes, so the test split has been looked at more than once.

The right next step is the forward holdout: freeze the two configurations
today and score them only on days after 2026-10-10.

## Rebuilding

    python3 model/context/sources/<source>.py         # each fetch script
    python3 model/tests/test_context_pit.py           # validity + lookahead gate
    cd model && python3 -m context.evaluate --groups all --n-placebo-trader 99

`etf_flows.py` needs `--snapshot FILE` here, because farside.co.uk returns 403
to plain requests from this container.
