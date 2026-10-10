# NOCTUA-Trader v1

An open-weight BTC position policy built on NOCTUA. Once a day at **17:00 UTC**
it chooses a position for the next 24 hours. The weights are in
`weights/noctua_trader_v1.npz` (56 KB, 5-seed ensemble, 2,561 parameters per
seed) and run locally in NumPy + SciPy, with no PyTorch.

**Backtest and paper trading only.** Nothing here places an order or needs
exchange keys.

## Bottom line

On data it never saw, **it does not show timing skill.** It earns a slightly
better Sharpe than buy-and-hold by holding a small long and shrinking it when
volatility is high. That difference is within noise. Treat it as a risk-sized
long, not a market-timing model.

## What it sees and decides

- **Inputs (62, all known at the anchor):**
  - NOCTUA's 42 engineered features
  - NOCTUA's own 19-hour forecast: volatility level, and the location, skew
    and width of its return quantiles
  - trailing returns over 1 hour to 30 days, and distance from the 20-day and
    50-day means
  - Deribit DVOL and perpetual funding, each with a "present" flag
  - the clock
- **Output:** a position in [0, 0.5], long-only (configuration `long_only=true`,
  `max_leverage=0.5`).
- **Training:** end to end on **net** P&L, with 6 bps fees on turnover and perp
  funding inside the loss. The objective is mean minus 5 × variance.

## Protocol

| split | dates | use |
|---|---|---|
| train | 2018-09 to 2022-12 | fit (all 24 anchor hours as augmentation) |
| val | 2023-01 to 2024-06 | early stopping and choosing the configuration |
| test | 2024-07 to 2026-10, 826 days | scored **once**, after selection |

- Each split boundary has a 24-hour embargo.
- NOCTUA itself was fit on the same train/calib split, so its forecasts are
  out of sample only in the test period.
- `dataset.py` pins `serve/noctua_v2.npz` explicitly. The refreshed artifact in
  `serve/` was fit through 2026-08 and would leak into the test period.

## Results (test, 17:00 UTC decisions, compounded)

| | Sharpe | CAGR | max drawdown | avg position |
|---|---|---|---|---|
| **NOCTUA-Trader v1** | **0.48** | 2.7% | -7.6% | 0.14 |
| buy-and-hold (1×, with funding) | 0.41 | 8.6% | -54% | 1.00 |
| constant position 0.14 | 0.41 | 2.3% | n/a | 0.14 |
| NOCTUA vol-target long | 0.30 | 4.1% | -52% | 0.96 |
| 20-day momentum, vol-targeted | 0.42 | 9.4% | -36% | long/short |

The table is the original scoring window: 826 days, 2024-07 to 2026-10 (see
Protocol). Re-scored on the extended price data over 830 days, 2024-07-01 to
2026-10-08, the trader gets Sharpe 0.44 vs 0.37 for buy-and-hold, CAGR 2.5% vs
7.1% compounded, and an average position of 0.14
([VERDICT §3](../../runs/noctua-disproof-2026-10-10/VERDICT.md)). The re-score
adds 4 days and rebuilt price data; why it lands lower was not investigated
(data bundle or sample definition, A0 B2).

- **Sharpe vs buy-and-hold:** +0.08, 95% block-bootstrap CI [-0.19, +0.33].
  Not significant.
- **Exposure control.** A constant 14% long has the same Sharpe as
  buy-and-hold, because scaling does not change Sharpe. So the edge to explain
  is 0.07 Sharpe, CI [-0.19, +0.31] on the re-score. It vanishes at 12 bp fees
  and flips sign if 2026 Q1 is left out (VERDICT §3; A3).
  **Status: no evidence of timing skill.**

## How the swarm was used

**The sweep.** Five Haiku 5.5 workers trained 58 configurations in parallel
(`weights/sweep_validation.json`):

- objectives
- feature sets
- model size and regularisation
- trading constraints
- optimiser settings

None beat buy-and-hold on validation with a CI that excludes zero. Removing
NOCTUA's forecast columns barely changed validation Sharpe (1.34 vs 1.37). A
price-only policy scored about the same as the full one.

**The disproof audit.** Three Haiku workers then tried to break the result:

- **Lookahead.** For 202 anchors, every hour at or after the anchor was
  corrupted, along with DVOL and funding. The maximum feature change was
  2.8e-13, which is BLAS rounding. A positive control on the hour before the
  anchor moves 48 features. Verdict: **no leak**.
- **Backtest arithmetic.** P&L, CAGR, Sharpe, drawdown, the funding sign, the
  24-hour spacing, and torch/NumPy loss parity were all recomputed
  independently and reproduce exactly. Verdict: **no defect**.
- **Timing skill.**
  - Shuffle placebo: one-sided p = 0.13.
  - Correlation of position with the next day's return: 0.04, p = 0.30.
  - Splitting the test period by date flips the sign of the edge over constant
    exposure.
  - The edge vanishes at 12 bps fees.
  - Position correlates -0.14 with trailing 30-day volatility.

  Verdict: **no evidence of timing skill**. The test is underpowered (Sharpe SE
  ≈ 0.7 over 2.3 years), so this is absence of evidence, not proof of absence.
  The auditor's claim that the buy-and-hold baseline does not reproduce was
  checked and is its own error: it treated log returns as simple returns and
  omitted funding.

## Use it

```bash
pip install -r model/serve/requirements-ci.txt
python model/policy/paper.py --fetch     # today's paper decision + settle old ones
python model/tests/test_policy_runtime.py
```

```bash
# Retrain (needs torch). The dataset rebuilds in about a minute.
python model/policy/dataset.py
python model/policy/train.py --out model/policy/runs/x --set long_only=true max_leverage=0.5 objective='"utility"' gamma=10
```

## Known limits

- **Price source.** Prices are the repo's BTC/USD hourly bundle, used as a
  proxy for BTCUSDT.
- **Funding.** Funding is Deribit's BTC perpetual rate, not Binance's.
- **Paper P&L.** `paper.py` P&L leaves out funding, because the committed bars
  do not carry it.
- **Live inputs.** The paper trader needs fresh DVOL and funding files, and it
  warns when they are stale.
- **Next step.** Whether new, point-in-time data adds direction skill is being
  tested in a separate thread. Any new source has to beat its own shuffled
  placebo in this harness.

*Educational research only. Not financial advice.*
