# Kronos Local — run the 12B-token foundation model on your own PC

This folder gives you a **local UI + API** for Kronos (shiyu-coder/Kronos), so the
dashboard no longer depends on scraping the public demo page.

## What you get

| Endpoint | Purpose |
|---|---|
| `http://127.0.0.1:8899/` | Dark status page (upside %, vol-amp %, model, age) |
| `http://127.0.0.1:8899/api/kronos` | JSON in the exact shape `fetchKronos()` expects |
| `../data/kronos_local.json` | File mirror — commit it and the GH-Pages site can read it |

Metrics are derived from **Monte-Carlo sampling**: `KRONOS_SAMPLES` (default 24)
independent forecast paths of the next 24 hourly candles from live Binance data.

- **Upside Probability** = % of paths whose 24h-ahead close > current close
- **Volatility Amplification** = % of paths whose predicted realized vol > trailing 24h realized vol

## How to run

Double-click **`run.bat`** — it activates `C:\Users\DELL\kronos-env`, points
`KRONOS_REPO` at `C:\Users\DELL\Kronos`, and starts the server. First run
downloads weights from Hugging Face (~100 MB, cached afterwards).

Your install layout is **correct as-is**: `C:\Users\DELL\Kronos` is the code repo,
`C:\Users\DELL\kronos-env` is the Python venv. They work together; neither needs
reinstalling.

## RAM trouble? (we hit a MemoryError on your PC)

The default model is **Kronos-small (24.7M params)**. If it still OOMs, edit
`run.bat` and uncomment:

```
set KRONOS_MODEL=NeoQuasar/Kronos-mini
```

Also close Chrome tabs before first model load; refresh happens hourly in a
background thread and is much lighter than the initial load.

## Tunables (env vars)

| Var | Default | Notes |
|---|---|---|
| `KRONOS_MODEL` | `NeoQuasar/Kronos-small` | `Kronos-mini` for low RAM |
| `KRONOS_TOKENIZER` | `NeoQuasar/Kronos-Tokenizer-base` | mini needs `Kronos-Tokenizer-2k` |
| `KRONOS_SAMPLES` | `24` | fewer = faster, noisier probabilities |
| `KRONOS_DEVICE` | `cpu` | set `cuda:0` if you add a GPU |
