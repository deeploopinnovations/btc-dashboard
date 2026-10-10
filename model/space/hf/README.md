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

> Runs today on GitHub Actions (`.github/workflows/paper-agent.yml`), because
> Gradio Spaces on the free tier now require HF PRO. This Space bootstrap is
> ready for when PRO is enabled: `python -m model.space.deploy --ref main --warm-start`.

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

## What the registered replay found (P5-rl-paper-result)

Same code path as the live agent, every 6-hour step on the period the shipped
NOCTUA model never saw (2024-07-01 to 2026-08-09, 3,078 steps), compounded:

| arm | per year | total | Sharpe | max drawdown |
|---|---|---|---|---|
| this agent (MV) | −10.1 % | −20.2 % | −0.61 | 34 % |
| hold BTC | +1.6 % | +3.4 % | +0.27 | 54 % |
| hold half | +3.6 % | +7.6 % | +0.27 | 30 % |
| volatility-target rule | −8.4 % | −16.9 % | −0.01 | 54 % |
| classic RL bandit (not deployed) | −65.6 % | −89.4 % | −3.40 | 90 % |

**No learner beat simply holding** (the agent is not statistically separable
from holding, from the volatility-target rule, or from a copy fed shuffled
inputs), and classic RL was significantly worse. NOCTUA's forecast carries no
exploitable 6-hour trading edge. The agent runs anyway, on paper, to collect
forward evidence and a loss log — not because it makes money.
