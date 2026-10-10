VERDICT C6: WEAKENED
VERDICT DATA: SURVIVED

# A3 audit: NOCTUA-Trader v1 (C6) and the newest price bars

Scope: read-only. Scratch code in `runs/noctua-disproof-2026-10-10/audits/scratch_A3/`
(`trader_positions.py`, `analysis.py`, `analysis2.py`, `analysis3.py`, `compare_data.py`,
`rv5_check.py`, `fetch_binance.py`; outputs `binance_1h.json`, `daily16_compare.csv`,
`trader_positions.parquet`). Nothing in the repo was edited.

## 1. Reproduction of the headline numbers (survives)

I recomputed the 830 daily positions with the same code path as `fresh_test.py`
(`policy.dataset.features_at` + `policy.runtime.NumpyTrader` on `evals/hours_through_now.parquet`,
17:00 UTC anchors from 2024-07-01) and the same P&L (`policy.backtest.pnl`, 6 bps fee on turnover,
funding subtracted as `pos * fund`). Everything matches `evals/fresh_test.json` "trader":

| | Sharpe | CAGR | MDD | total | avg pos |
|---|---|---|---|---|---|
| NOCTUA-Trader v1 | 0.4416 | 2.46% | -7.60% | +5.68% | 0.1393 |
| Buy-and-hold | 0.3728 | 7.07% | -54.06% | +16.80% | 1.00 |

- Position range 0.067 to 0.281 (5th to 95th percentile 0.094 to 0.203). It is never flat and never near the 0.5 cap.
- Sharpe difference vs B&H: +0.069, block-bootstrap CI [-0.195, +0.312], P(<=0) = 0.33. The "not significant" statement is correct.
- Fresh window (55 anchors, 2026-08-15 17:00 to 2026-10-08 17:00): trader +4.11% net (avg position 0.173), B&H +30.36% net. Reproduced.

Note: the README in `model/policy/README.md` reports 0.48 vs 0.41 Sharpe, CAGR 2.7% vs 8.6%, over
826 days. The JSON gives 0.44 vs 0.37 over 830 days. The two documents disagree for the same protocol. The
gap comes from the extra days, but the README should be reconciled with the JSON.

## 2. Is the Sharpe edge more than holding a small long? (WEAKENED)

The 0.44 vs 0.37 gap is almost entirely a timing effect on a 14% exposure that is statistically indistinguishable from noise.

- **Constant exposure has the same Sharpe as B&H.** A constant 0.1393 position gives Sharpe 0.3728, identical to B&H, because scaling does not change Sharpe. Its MDD is -9.48% and its CAGR is 2.06%. The trader's edge over the matched constant is +0.069, CI [-0.195, +0.312], P(<=0) = 0.33.
- **The -54% vs -7.6% drawdown comparison is an exposure artefact.** The matched constant has -9.5%. The trader's -7.6% is only slightly better, which is a small, real vol-responsiveness effect (corr(position, trailing 30d vol) = -0.21).
- **Lagged 60-day trailing average of the trader's own position (causal, days 60 to 829).** Trader Sharpe 0.5425, proxy 0.4951, constant 0.4767. The slow component explains about 0.018 of the roughly 0.066 edge. The remaining day-to-day variation gives +0.047 vs the proxy, CI [-0.23, +0.33].
- **Correlation of daily position with next-day return.** Pearson +0.034 (p = 0.32), Spearman +0.034 (p = 0.33), block-bootstrap CI [-0.023, +0.089] with P(<=0) = 0.13. Position is strongly autocorrelated (lag-1 0.66).
- **Regression of trader P&L on B&H daily return.** Beta 0.134 (mean position 0.139), alpha +0.27 bp/day, t = 1.16.
- **Matched-exposure vol-scaling.** Scaling to the same mean exposure (k / trailing-30d vol, cap 0.5) gives Sharpe 0.354, below the constant. A position fitted only to log trailing vol (R^2 0.039) gives Sharpe 0.369. So the trader's edge is not vol-scaling.
- **Placebos.** A full permutation placebo (2,000 draws) is not a fair null, because permuting raises turnover and fees: its mean Sharpe is 0.23 (sd 0.16). The circular-shift placebo (829 shifts, which keeps turnover and autocorrelation) has mean Sharpe edge -0.085 vs the matched constant, and the observed +0.069 gives one-sided p = 0.163. Neither supports a timing skill claim.

## 3. Fragility of the edge (strongest finding against C6)

- **Fees.** Gross of fees, the trader's Sharpe is 0.520 vs B&H 0.373 (edge +0.146, CI [-0.115, +0.387]). At 3 bps, 0.481; at 6 bps (the model's assumption), 0.442; at 12 bps, 0.364, below B&H's 0.372. Fee drag is 1.05% of total log wealth over 830 days. Its Sharpe effect (about 0.08) is larger than the net edge (0.07). The README's "edge vanishes at 12 bps" is confirmed.
- **Quarter concentration.** Mean daily edge vs the matched constant, by sub-period (bp/day): 2024H2 -0.34, 2025H1 -0.24, 2025H2 +0.20, 2026H1 +0.79, 2026Q3 +0.12. Leave-one-quarter-out Sharpe edge: dropping 2026Q1 flips the sign to -0.056. Dropping 2025Q4 leaves +0.017. Every other single-quarter drop leaves +0.056 to +0.108. The edge rests on one quarter of a falling market.
- **Fresh-window reversal.** On the 55 most recent anchors the trader's average position (0.173) was above its overall mean, yet it earned +4.1% against B&H's +30.4%. This is 55 days of noise, but it runs against the headline.
- **"Fresh" is not a holdout.** The 55 fresh anchors are part of the 830-day test scored in `fresh_test.json`, and that test has already been reported in the README. The label overstates what was held out.

## 4. Things that flatter or mislead the trader (summary)

- Sharpe is quoted against B&H, which is a scale-invariant comparison. The meaningful benchmark is constant exposure, which ties B&H.
- The drawdown headline (-7.6% vs -54%) compares exposures, not skill.
- Cash is assumed to earn 0%. This understates the trader. At a 4.5% cash yield its CAGR would be 6.42% (vs B&H 7.07%) and its excess-return Sharpe edge stays about +0.067. So the cash assumption works against the trader, not for it.
- The headline omits that CAGR is lower than B&H (2.5% vs 7.1%).

## 5. Pnl formula, fees and funding (confirmed correct)

- `backtest.pnl`: `pos * expm1(ret) - 6e-4 * |pos_t - pos_{t-1}| - pos * fund`. Fees are charged on position changes. Turnover is 0.021/day, so the fee drag is about 0.46% per year.
- Funding sign: a long pays when Deribit `interest_1h` is positive. The code subtracts `pos * fund`, so the sign is correct. `interest_1h` is positive on 81% of days, with mean 1.4 bp/day.
- Timing: `features_at` reads rows up to a-1 (the 16:00 bar). `targets` uses entry `close[a-1]` and exit `close[a+23]`, so there is no lookahead in entry or exit.
- DVOL and funding files cover through 2026-10-10. `dvol_present` and `fund_present` are 1.0 on all 830 test anchors, so no fallback values are used.

## 6. Data integrity (SURVIVED)

Source: Binance `data-api.binance.vision` BTCUSDT klines, fetched directly (`binance_1h.json`, 1,864 hourly bars from 2026-07-25 to 2026-10-10 15:00 UTC).

- **Hourly closes, `evals/hours_through_now.parquet` vs Binance.** Over 1,863 overlapping hours (2026-07-25 to 2026-10-10 14:00), mean difference -0.05%, max absolute difference 0.16%. After 2026-08-15 the mean absolute difference is 0.04% and the max is 0.14%.
- **Hourly closes, `data/assets/btc_history.parquet` vs Binance.** Over 516 overlapping hours, mean difference -0.10%, max 0.16%. This is a stable basis (BTC/USD vs USDT), not an error.
- **16:00 UTC entry bars, 2026-08-01 to 2026-10-09 (70 days).** Max absolute difference 0.12% (parquet vs Binance). Zero days exceed 0.5%. The 2026-10-10 16:00 bar was not yet printed on Binance at fetch time, and the fresh window ends at the 10-08 anchor, so it is not needed.
- **Coverage.** `btc_history.parquet` actually ends 2026-08-15 11:00 UTC, not 00:00. The 16:00 entry bar for 2026-08-15 comes from the live extension. It is 0.11% below Binance.
- **Fresh B&H return, price only.** Binance 16:00 close on 2026-08-15 is 63,096.63. Binance 16:00 close on 2026-10-09 is 82,746.00. Price-only return is +31.14%. Funding costs -0.63% and the entry fee about -0.06%, giving +30.36%. The claimed +30.4% is real.
- **Realised variance.** Hourly `rv5` vs realised variance from Binance 5-minute bars (1,695 complete hours): correlation 0.9988, median ratio 0.974. The ratio is 0.970 before 2026-08-15 and 0.975 after, so the live tail is not biased. No day falls outside 0.76 to 1.04.

Caveat: the live bars come from Bitstamp/Coinbase (`fetch.py`), not Binance. They agree with Binance within 0.16%. This does not change the conclusions.

## Bottom line

- C6 numbers reproduce exactly, and the "not significant" statement is correct. But the trader's Sharpe edge is a 0.07 timing difference on a 14% position, with CI [-0.19, +0.31]. It disappears at 12 bps, it is concentrated in one quarter, and it is equal to a plain constant position's Sharpe, which is B&H's. Calling it an edge is WEAKENED, not DISPROVED.
- The newest bars are real. The +30.4% B&H move since 2026-08-15 is confirmed against Binance, and the trader's +4.1% is the net result of its own position.
