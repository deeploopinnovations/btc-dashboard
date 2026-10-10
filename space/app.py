"""
space/app.py -- the Hugging Face Space UI for NOCTUA-Trader's forward holdout.

A background thread runs forward.run_once() every 30 minutes while the Space
is awake. The page shows the latest decision, NOCTUA's volatility forecast,
and the forward record against buy-and-hold and a constant 0.14 position.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone

import gradio as gr
import numpy as np
import pandas as pd

import forward as FW

EVERY_S = 30 * 60
_lock = threading.Lock()
_last = {"result": None, "error": None, "ran_utc": None}


def refresh() -> None:
    with _lock:
        try:
            _last["result"] = FW.run_once()
            _last["error"] = None
        except Exception:  # noqa: BLE001 - shown on the page, retried next tick
            _last["error"] = traceback.format_exc(limit=3)
        _last["ran_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def loop() -> None:
    while True:
        refresh()
        time.sleep(EVERY_S)


def _log() -> list[dict]:
    p = FW.STATE / "forward_log.json"
    return json.loads(p.read_text()) if p.exists() else []


def _pct(x):
    return "n/a" if x is None else f"{100 * x:+.2f}%"


def render():
    r, log = _last["result"], _log()
    head = [f"**Last run:** {_last['ran_utc'] or 'starting...'}"]
    if _last["error"]:
        head.append(f"\n**Last run failed** (retries in 30 min):\n```\n{_last['error']}\n```")
    if r:
        head.append(f" · market data through {r['data_last_utc'][:16].replace('T', ' ')} UTC")
        stale = [k.replace("_error", "") for k in r if k.endswith("_error")]
        if stale:
            head.append(f" · fetch failed for: {', '.join(stale)}")
    status = "".join(head)

    if log:
        e = log[-1]
        dv = f"{e['dvol']:.1f}" if e.get("dvol") is not None else "missing"
        latest = (
            f"### Decision for {e['anchor_utc'][:16].replace('T', ' ')} UTC\n"
            f"- **Position:** {e['position']:.3f} BTC per 1 of capital, held 24 h (long-only, cap 0.5)\n"
            f"- **NOCTUA volatility forecast:** {100 * e['noctua_ann_vol']:.1f}% annualised "
            f"({100 * e['noctua_sigma_19h']:.2f}% over the next 19 h)\n"
            f"- **Deribit DVOL:** {dv}\n"
            f"- **Entry price:** ${e['entry_price']:,.0f}"
            + (" · decided late (Space was asleep)" if e.get("late") else "")
            + (f" · **stale inputs:** {', '.join(e['stale_inputs'])}" if e.get("stale_inputs") else "")
        )
    else:
        latest = (f"### No forward decisions yet\nThe first one is made at "
                  f"{FW.FROZEN['forward_start_utc'][:16].replace('T', ' ')} UTC.")

    s = FW.summary(log)
    rows = [{"strategy": name,
             "total return (compounded)": _pct(s.get(k, {}).get("total_return")),
             "Sharpe (from 20 days)": s.get(k, {}).get("sharpe") or "n/a",
             "max drawdown": _pct(s.get(k, {}).get("max_drawdown"))}
            for name, k in (("NOCTUA-Trader v1", "trader"), ("buy and hold 1x", "buy_hold"),
                            ("constant 0.14", "const_0.14"))]
    record = pd.DataFrame(rows)

    done = [e for e in log if "pnl" in e]
    if done:
        eq = pd.DataFrame({
            "date": pd.to_datetime([e["anchor_ts"] + 86400 for e in done], unit="s"),
            "NOCTUA-Trader v1": np.cumprod([1 + e["pnl"] for e in done]),
            "buy and hold 1x": np.cumprod([1 + e["pnl_buy_hold"] for e in done]),
            "constant 0.14": np.cumprod([1 + e["pnl_const"] for e in done]),
        }).melt("date", var_name="strategy", value_name="equity")
    else:
        eq = pd.DataFrame({"date": [], "strategy": [], "equity": []})

    table = pd.DataFrame([{
        "anchor (UTC)": e["anchor_utc"][:16].replace("T", " "),
        "position": e["position"],
        "NOCTUA ann. vol": f"{100 * e['noctua_ann_vol']:.1f}%",
        "BTC 24h": _pct(e.get("btc_return")),
        "trader P&L": _pct(e.get("pnl")),
        "late": e.get("late", False),
    } for e in reversed(log)])
    frozen = (f"{s['decisions']} decisions, {s['settled']} settled, {s['late_decisions']} decided late. "
              f"Frozen weights: policy `{s['policy_sha256']}`, forecaster `{s['forecaster_sha256']}`.")
    return status, latest, record, eq, table, frozen


def run_now():
    refresh()
    return render()


with gr.Blocks(title="NOCTUA forward holdout") as demo:
    gr.Markdown(
        "# NOCTUA-Trader: forward holdout\n"
        "The frozen NOCTUA-Trader v1 policy, scored only on days after 2026-10-10. "
        "It decides a BTC position at 17:00 UTC and holds it 24 hours. "
        "**Paper trading only:** no orders, no exchange keys. In backtests it showed "
        "no timing skill; it mostly holds a small long and trims it when volatility is high.")
    status = gr.Markdown()
    latest = gr.Markdown()
    gr.Markdown("### Forward record")
    record = gr.Dataframe(interactive=False)
    plot = gr.LinePlot(x="date", y="equity", color="strategy", height=320)
    frozen = gr.Markdown()
    gr.Markdown("### Every decision")
    table = gr.Dataframe(interactive=False)
    btn = gr.Button("Run now")
    outs = [status, latest, record, plot, table, frozen]
    btn.click(run_now, outputs=outs)
    demo.load(render, outputs=outs)

threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
