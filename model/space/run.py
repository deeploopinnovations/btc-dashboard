"""
space/run.py
=====================================================================
The NOCTUA paper-trading agent, running in a Hugging Face Space.

PAPER TRADING ONLY. No exchange account, no keys, no orders: every position
is simulated on the same Bitstamp BTC/USD bars NOCTUA serves on.

The Space's own app.py (model/space/hf/app.py) clones this repository at
GIT_REF and calls `main()`. A background thread wakes every POLL_S seconds;
around each 6-hour slot (00/06/12/18 UTC) it fetches bars, settles the window
that just closed (the agent learns from it), and decides the next position.
State and the per-step log are mirrored to the HF dataset RL_DATASET after
every change, so a restart -- or the free tier putting the Space to sleep --
resumes exactly where it stopped. Slots slept through hold the position, and
their P&L still counts when they settle.

  env        RL_DATASET  <user>/<dataset> for state + log (else local only)
             HF_TOKEN    a token that can write that dataset (Space secret)
             NOCTUA_DEBUG=1 for the stage-by-stage forecast trace in the logs
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl import env as E                                            # noqa: E402
from rl.loop import PaperTrader                                    # noqa: E402
from rl.store import HFStore, LocalStore                           # noqa: E402

LIVE_KINDS = ("MV",)      # TS trades noise (tests/test_rl.py) and is barred from live use
POLL_S = 120
BUSY_S = 55 * 60          # after a slot boundary, poll the feed for this long
STATUS: dict = {"started": int(time.time()), "error": None, "ticks": 0}


def make_store():
    repo, tok = os.environ.get("RL_DATASET"), os.environ.get("HF_TOKEN")
    root = Path(os.environ.get("RL_DIR", "/tmp/rl_store"))
    if repo and tok:
        st = HFStore(root, repo, token=tok)
        STATUS["store"] = f"{repo}: {st.restore()}"
        return st
    STATUS["store"] = "local only (no RL_DATASET / HF_TOKEN): lost on restart"
    return LocalStore(root)


def due(now: int, trader: PaperTrader) -> bool:
    """Fetch only when something can happen: just after a slot, or a window overdue."""
    if now - E.slot_floor(now) <= BUSY_S:
        return True
    return any(now >= d["E"] + E.H * E.HOUR for d in trader.pending)


def loop(trader: PaperTrader, stop: threading.Event) -> None:
    from serve.fetch import fetch_bars
    from serve.history import get_hours
    while not stop.is_set():
        now = int(time.time())
        try:
            if due(now, trader):
                hours, info = get_hours(fetch_bars, persist=False, verbose=False, now_ts=now)
                ev = trader.tick(now, hours)
                STATUS.update(last_tick=now, feed=info.get("source"), ticks=STATUS["ticks"] + 1,
                              last_events=[{k: e.get(k) for k in ("type", "E", "a", "R", "u")} for e in ev],
                              flush=trader.last_flush, error=None)
        except Exception as e:                                  # noqa: BLE001 -- keep the loop alive
            STATUS.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(), error_at=now)
        stop.wait(POLL_S)


def ts(t) -> str:
    return "—" if t is None else pd.Timestamp(int(t), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def summary(trader: PaperTrader) -> tuple:
    log = trader.store.read_log()
    nxt = E.slot_floor(int(time.time())) + E.H * E.HOUR
    head = [
        "**PAPER TRADING ONLY** — no exchange, no keys, no orders. Educational research, not advice.",
        f"Agent **{trader.agent.kind}** · position **{trader.w:+.1f}** BTC-equity · "
        f"learned from **{trader.agent.n_updates}** settled windows · {len(trader.pending)} pending",
        f"Last slot decided: {ts(trader.last_slot)} · next slot: {ts(nxt)} · last tick: {ts(STATUS.get('last_tick'))}",
        f"Store: {STATUS.get('store')} · last sync: {STATUS.get('flush')}",
    ]
    if STATUS.get("error"):
        head.append(f"⚠️ Last error ({ts(STATUS.get('error_at'))}): `{STATUS['error']}`")
    live = sorted((r for r in log if r.get("type") == "settled"), key=lambda r: r["E"])
    if live:
        tot = {"agent": sum(r["u"] for r in live)}
        for k in ("FLAT", "HALF", "HOLD", "VT"):
            tot[k] = sum(r["u_baseline"][k] for r in live)
        head.append(f"Since live start, {len(live)} settled steps, total utility: " +
                    " · ".join(f"{k} {v:+.4f}" for k, v in tot.items()) +
                    ". No claim of skill until the forward log clears VT at 99.5% (P5-rl-paper).")
    rows = [{"slot": ts(r["E"]), "action": r["a"], "R %": round(100 * r["R"], 3),
             "u": round(r["u"], 6), "u HOLD": round(r["u_baseline"]["HOLD"], 6),
             "u VT": round(r["u_baseline"]["VT"], 6), "regret": round(r["regret"], 6),
             "missed": r["missed"]} for r in live[-40:][::-1]]
    return "\n\n".join(head), pd.DataFrame(rows)


def build_ui(trader: PaperTrader):
    import gradio as gr
    with gr.Blocks(title="NOCTUA paper agent") as demo:
        gr.Markdown("# NOCTUA paper-trading agent (BTC, 6-hour steps)")
        md = gr.Markdown()
        table = gr.Dataframe(label="Settled steps (newest first)", interactive=False)
        btn = gr.Button("Refresh")
        btn.click(lambda: summary(trader), outputs=[md, table])
        demo.load(lambda: summary(trader), outputs=[md, table])
    return demo


def main(kind: str = "MV") -> None:
    if kind not in LIVE_KINDS:
        raise SystemExit(f"{kind} is barred from live use (tests/test_rl.py); allowed: {LIVE_KINDS}")
    trader = PaperTrader(make_store(), agent_kind=kind)
    if trader.agent.kind not in LIVE_KINDS:
        raise SystemExit(f"stored agent is {trader.agent.kind}, barred from live use")
    stop = threading.Event()
    threading.Thread(target=loop, args=(trader, stop), daemon=True).start()
    build_ui(trader).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
