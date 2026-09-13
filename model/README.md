# NOCTUA

A **6,445-parameter** model (×3 seeds = 19,335 stored weights) for one
decision: the BTC option seller's **19-hour overnight window**, 17:00 UTC
(22:30 IST) → 12:00 UTC (17:30 IST), which is the Delta Exchange daily-option
holding period.

It answers *"which levels are strong enough that they will not break?"* as a
calibrated **barrier survival curve**, not a point forecast.

| | |
|---|---|
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | problem formalization, literature, architecture, protocol |
| [`RESULTS.md`](RESULTS.md) | out-of-sample results **including the negative ones** |
| `noctua/` | pipeline: ingest → episodes → features → train → evaluate → export |
| `serve/` | NumPy-only runtime + Hugging Face Space app |
| [`BENCHMARK.md`](BENCHMARK.md) | the adversarial benchmark and every correction made to it — **the current source of truth for numbers** |
| [`AUDIT.md`](AUDIT.md) | end-to-end audit: what is established, what is not, and what is next |
| `eval/kronos_baseline.py` | head-to-head vs Kronos — run, 120 episodes (see `BENCHMARK.md` §5) |

## Headline

Out-of-sample walk-forward, 2,046 non-overlapping production episodes:

- **Volatility:** the shipped committee beats a well-specified Log-HAR by
  **4.04 % QLIKE**, p = 0.0002.
- **Deep-tail barriers:** calibration error **1.09 pp vs 3.33 pp** for the
  Gaussian first-passage baseline at α = 1 %. The textbook model understates
  deep-tail touch risk by 2–4×.
- **Body barriers (α ≥ 10 %):** the Gaussian is better. Reported, not hidden.
- **Direction: no skill, and now measured properly.** `eval/direction.py`
  scores the model's own `prob_up` walk-forward against a base-rate
  climatology, and separately asks whether ANY predictor built from these 42
  features can beat it. See `AUDIT.md` §2 — the served `upside` field is
  pinned to 50.0 and the raw value is deliberately wired to nothing.
- **Kronos:** benchmarked, 120 episodes. NOCTUA is better calibrated and
  ~190,000× faster; Kronos's barrier discrimination does **not** clear a
  shuffled control. See `BENCHMARK.md` §5.

> **Numbers here are summaries.** Where this file and `BENCHMARK.md` disagree,
> `BENCHMARK.md` is right — it is regenerated from `model/artifacts/*.json`,
> this file is written by hand and has been stale before.

## Why it replaced the Kronos scrape

`data/kronos.json` had been serving a fossil: the public demo's last genuine
timestamp was `2026-07-04 11:00:26 UTC`, ~880 hours stale, and
`scripts/parse-kronos.js` re-committed it every 30 minutes. That script is now
retired; `model/serve/predict.py` produces `data/kronos.json` (legacy shape,
drop-in) and `data/noctua.json` (barrier curves + safe strikes).

## Quick start

```bash
pip install -r serve/requirements-ci.txt
python serve/predict.py --out-dir data      # live forecast
```

Full reproduction from raw data is in [`RESULTS.md` §8](RESULTS.md#8-reproducing).

---

*Educational research only. Not financial advice.*
