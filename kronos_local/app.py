"""
kronos_local/app.py  (v1)
=====================================================================
Local Kronos prediction server — replaces scraping the public demo page.

What it does
------------
1. Pulls the last 400 BTC/USDT 1-hour candles from Binance (free, no key).
2. Runs the Kronos foundation model (NeoQuasar/Kronos-small by default —
   24.7M params, chosen deliberately because the base model triggered a
   MemoryError on this machine) with N Monte-Carlo samples.
3. Computes the two metrics the dashboard already consumes:
      • upside  — % of MC sample paths whose 24h-ahead close > current close
      • volAmp  — % of MC paths whose predicted 24h realised vol exceeds
                  the trailing 24h realised vol (amplification probability)
4. Serves them at  http://127.0.0.1:8899/api/kronos  in EXACTLY the JSON
   shape src/data.js fetchKronos() produces, plus a minimal HTML status page
   at /.  Point the dashboard at it or just read it manually.

Run (Windows)
-------------
    C:\\Users\\DELL\\kronos-env\\Scripts\\activate
    cd C:\\Users\\DELL\\BTC_Dashboard\\kronos_local
    set KRONOS_REPO=C:\\Users\\DELL\\Kronos
    python app.py
    → open http://127.0.0.1:8899/

First run downloads ~100MB of weights from Hugging Face into the HF cache.
Inference is CPU-friendly: ~20-60s for 24 samples on a typical laptop.
"""
import os
import sys
import json
import time
import threading
import traceback
from datetime import datetime, timezone

# ── locate the Kronos repo (model/ package) ──────────────────────────────
KRONOS_REPO = os.environ.get("KRONOS_REPO", r"C:\Users\DELL\Kronos")
if KRONOS_REPO not in sys.path:
    sys.path.insert(0, KRONOS_REPO)

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify

# Tunables ────────────────────────────────────────────────────────────────
MODEL_NAME     = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_NAME = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
DEVICE         = os.environ.get("KRONOS_DEVICE", "cpu")
CONTEXT_LEN    = 400        # input candles (max_context 512 for small)
PRED_LEN       = 24         # 24 × 1h = next 24h
N_SAMPLES      = int(os.environ.get("KRONOS_SAMPLES", "24"))
REFRESH_SEC    = 3600       # recompute hourly
PORT           = 8899

app = Flask(__name__)
state = {"result": None, "status": "booting", "error": None, "started": time.time()}

_predictor = None


def load_predictor():
    """Lazy-load model once. Kept out of module import so Flask starts fast."""
    global _predictor
    if _predictor is not None:
        return _predictor
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402 (repo package)
    print(f"[kronos-local] loading {TOKENIZER_NAME} + {MODEL_NAME} on {DEVICE} …")
    tok = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
    mdl = Kronos.from_pretrained(MODEL_NAME)
    _predictor = KronosPredictor(mdl, tok, device=DEVICE, max_context=512)
    print("[kronos-local] model ready")
    return _predictor


def fetch_candles(limit=CONTEXT_LEN):
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "1h", "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "tb_base", "tb_quote", "ignore",
    ])
    for c in ("open", "high", "low", "close", "volume", "quote_vol"):
        df[c] = df[c].astype(float)
    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.rename(columns={"quote_vol": "amount"})
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


def realised_vol(closes):
    lr = np.diff(np.log(np.asarray(closes, dtype=float)))
    return float(np.std(lr, ddof=1)) if len(lr) > 2 else 0.0


def run_inference():
    pred = load_predictor()
    df = fetch_candles()
    last_close = float(df["close"].iloc[-1])
    last_ts = df["timestamps"].iloc[-1]

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = pd.Series(df["timestamps"].dt.tz_localize(None))
    y_ts = pd.Series(pd.date_range(
        last_ts.tz_localize(None) + pd.Timedelta(hours=1), periods=PRED_LEN, freq="h"))

    trailing_vol = realised_vol(df["close"].iloc[-25:])

    ups, amps = 0, 0
    t0 = time.time()
    for i in range(N_SAMPLES):
        # sample_count=1 + T=1.0 → one stochastic MC path per call
        p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                         pred_len=PRED_LEN, T=1.0, top_p=0.9,
                         sample_count=1, verbose=False)
        path_close = p["close"].astype(float)
        if float(path_close.iloc[-1]) > last_close:
            ups += 1
        if realised_vol(pd.concat([df["close"].iloc[-1:], path_close])) > trailing_vol:
            amps += 1
        print(f"[kronos-local] sample {i+1}/{N_SAMPLES} "
              f"end={float(path_close.iloc[-1]):,.0f} ({time.time()-t0:.0f}s)")

    now = datetime.now(timezone.utc)
    return {
        "upside":   round(100.0 * ups / N_SAMPLES, 1),
        "volAmp":   round(100.0 * amps / N_SAMPLES, 1),
        "sourceTs": now.strftime("%Y-%m-%d %H:%M:%S"),
        "sourceMs": int(now.timestamp() * 1000),
        "tz": "UTC",
        "ageHrs": 0.0,
        "freshness": "fresh",
        "fetchedAt": int(time.time() * 1000),
        "proxy": "local-inference",
        "_updatedMs": int(time.time() * 1000),
        "model": MODEL_NAME,
        "samples": N_SAMPLES,
        "lastClose": last_close,
        "trailingVol24hPct": round(trailing_vol * (24 ** 0.5) * 100, 2),
    }


def worker():
    while True:
        try:
            state["status"] = "running-inference"
            state["result"] = run_inference()
            state["status"] = "idle"
            state["error"] = None
            # Optionally mirror to the dashboard's snapshot dir for offline use
            out = os.path.join(os.path.dirname(__file__), "..", "data", "kronos_local.json")
            with open(out, "w") as f:
                json.dump(state["result"], f, indent=2)
            print(f"[kronos-local] upside={state['result']['upside']}% "
                  f"volAmp={state['result']['volAmp']}% → sleeping {REFRESH_SEC}s")
        except MemoryError:
            state["status"] = "error"
            state["error"] = ("MemoryError — close other apps or set "
                              "KRONOS_MODEL=NeoQuasar/Kronos-mini and restart.")
            traceback.print_exc()
        except Exception as e:  # noqa: BLE001 — surface everything to the UI
            state["status"] = "error"
            state["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        time.sleep(REFRESH_SEC)


@app.get("/api/kronos")
def api_kronos():
    if state["result"] is None:
        return jsonify({"status": state["status"], "error": state["error"]}), 503
    return jsonify(state["result"])


@app.get("/")
def home():
    r = state["result"]
    body = (
        f"<h2>Kronos local · {MODEL_NAME}</h2>"
        f"<p>status: <b>{state['status']}</b>"
        + (f" · <span style='color:#f87171'>{state['error']}</span>" if state["error"] else "")
        + "</p>"
    )
    if r:
        body += (
            f"<p style='font-size:28px'>Upside (24h): <b>{r['upside']}%</b> · "
            f"Vol-Amp (24h): <b>{r['volAmp']}%</b></p>"
            f"<p>last close ${r['lastClose']:,.0f} · {r['samples']} MC samples · "
            f"computed {r['sourceTs']} UTC</p>"
            f"<p><a href='/api/kronos'>/api/kronos</a> (JSON, dashboard-compatible)</p>"
        )
    else:
        body += "<p>First inference in progress — refresh in a minute…</p>"
    return ("<html><body style='font-family:system-ui;background:#0a0a0f;"
            f"color:#e7e7f0;padding:32px'>{body}</body></html>")


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"[kronos-local] http://127.0.0.1:{PORT}/")
    app.run(host="127.0.0.1", port=PORT)
