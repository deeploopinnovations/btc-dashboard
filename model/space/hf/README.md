---
title: NOCTUA Paper Agent
emoji: 🦉
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# NOCTUA paper-trading agent

**Paper trading only. No exchange account, no keys, no orders.** Educational
research, not financial advice.

Every 6 hours (00/06/12/18 UTC) the agent reads NOCTUA's volatility forecast
for the next window, chooses a simulated BTC exposure in {-1, -0.5, 0, +0.5,
+1}, and, when the window closes, learns from what every exposure would have
earned. Each step is logged with what FLAT, HALF, HOLD and a volatility-target
rule earned on the same move, so the comparison builds up on its own.

The code lives in [deeploopinnovations/btc-dashboard](https://github.com/deeploopinnovations/btc-dashboard)
(`model/rl/`, `model/space/`); this Space clones it at `GIT_REF` on every start,
so a restart runs the latest code. State and the log persist in the dataset
named by `RL_DATASET`, so a restart or sleep resumes where it stopped.
The design and its pre-registered test are `P5-rl-paper` in
`model/research/ledger.json`.
