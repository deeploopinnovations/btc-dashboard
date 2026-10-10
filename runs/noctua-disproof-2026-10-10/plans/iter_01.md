# Iteration 1 plan: fresh test of the shipped NOCTUA

Written 2026-10-10 before any code ran.

## What "fresh" means here
- The artifact under test is the PINNED `model/serve/noctua_v2.npz` (fit on data
  before 2024-07). Nothing is refit on or after 2024-07-01.
- Price history is extended to 2026-10-10 14:00 UTC with `paper.fetch_tail`
  (Bitstamp/Coinbase 5-minute bars), so the latest ~8 weeks were never part of
  any earlier report (asset history ended 2026-08-15; the bundle 2026-10-05).
- The scoring code is new and independent of `eval/benchmark.py`, so a bug
  shared with the earlier benchmark cannot carry over silently.

## Claims to test (each one an auditor will try to disprove)
- C1 NOCTUA's 19 h volatility forecast beats a frozen Log-HAR on QLIKE, 2024-07 onward.
- C2 It also beats stronger baselines: frozen HAR with 1 h/6 h terms, an
  expanding-window refit HAR (uses information NOCTUA never had), EWMA.
- C3 It beats the options market: Deribit DVOL scaled to 19 h.
- C4 The edge holds in every half-year and in the freshest window (2026-08-15+).
- C5 Deep-tail barriers (alpha 1 %, 5 %) are calibrated on test: breach rate
  inside the binomial 95 % band.
- C6 NOCTUA-Trader v1 beats buy-and-hold AND a no-neural-net vol-target rule.
- C7 (negative) Direction has no skill. Confirm rather than assume.

## Protocol
- Production slice only: anchor 17:00 UTC, 19 h window, one episode a day.
- Baselines get every advantage NOCTUA got: fit on the same pre-2024-07 data,
  QLIKE-optimal scale fitted on that data only.
- Paired QLIKE differences, circular block bootstrap (block 20 days, 4000 reps).
- Report "raw" (as the trader uses it) and "served" (with serve/adaptive.py's
  causal correction, as the dashboard uses it).
- Returns compounded. Nothing tuned on 2024-07+.

## Done when
Each claim has a number, a CI and a status of supported / not supported, and
evals/fresh_test.json is written.
