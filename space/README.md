# NOCTUA forward holdout (Hugging Face Space)

The frozen NOCTUA-Trader v1 policy, scored only on 17:00 UTC anchors from
2026-10-10 on. Paper trading only: no orders, no exchange keys.

| file | role |
|---|---|
| `forward.py` | one pass: refresh public data, decide new anchors, settle finished ones |
| `../.github/workflows/noctua-forward.yml` | runs `forward.py` daily and commits `data/forward/forward_log.json` |
| `static/index.html` | the Space page; reads that log from GitHub |
| `app.py` | Gradio version that runs the model inside the Space (needs HF PRO) |
| `deploy.py` | creates the private Space and uploads it |

```bash
HF_TOKEN=... python space/deploy.py                 # free static Space
HF_TOKEN=... python space/deploy.py --mode gradio   # CPU Space, HF PRO only
NOCTUA_STATE_DIR=/tmp/fw python space/forward.py --no-push   # local dry run
```

**What is frozen.** The policy weights (`model/policy/weights/noctua_trader_v1.npz`)
and the pinned forecaster (`model/serve/noctua_v2.npz`). Every log row records
both files' SHA-256 prefixes, so a silent swap shows up in the log.

**Late decisions.** Every input reads only bars before the anchor, so a decision
computed after the fact equals the on-time one. It is still flagged `late`
when it was made more than an hour after the anchor.

**Settlement.** Same arithmetic as `policy/backtest.py`: 6 bps fees on turnover
and Deribit perp funding over the 24 h hold. Benchmarks are buy-and-hold 1x and
a constant 0.14 position (the backtest's average).
