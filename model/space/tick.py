"""
space/tick.py
=====================================================================
One step of the NOCTUA paper-trading agent, for a scheduled job (the
`paper-agent` GitHub workflow). PAPER TRADING ONLY: no exchange, no orders.

Restores state + log from the private HF dataset, fetches bars, runs one
idempotent PaperTrader.tick (settle what closed, decide the current slot if it
is on time), and uploads state + log if anything changed. The workflow fires
several times around each 00/06/12/18 UTC slot; extra runs do nothing, and a
slot no run reaches in time is logged as missed (the position is held, never
filled at a stale price).

  HF_PAPER_TOKEN  token of the project's own HF account (refused otherwise)
  RL_DATASET      default <that account>/noctua-rl-paper

    python -m model.space.tick             # live: reads and writes the dataset
    python -m model.space.tick --dry-run   # restores, ticks, uploads nothing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.loop import PaperTrader                                    # noqa: E402
from rl.store import HFStore, LocalStore                           # noqa: E402

POINTER = Path(__file__).resolve().parents[1] / "research" / "hf_store.json"
LIVE_KINDS = ("MV",)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one paper-agent step")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_PAPER_TOKEN")
    if not token:
        raise SystemExit("HF_PAPER_TOKEN is not set (GitHub secret of the project's HF account)")
    from huggingface_hub import HfApi
    owner = json.loads(POINTER.read_text())["repo"].split("/")[0]
    user = HfApi(token=token).whoami()["name"]
    if user != owner:
        raise SystemExit(f"REFUSING: HF_PAPER_TOKEN belongs to {user!r}, the project's account is {owner!r}")
    repo = os.environ.get("RL_DATASET") or f"{owner}/noctua-rl-paper"

    root = Path(tempfile.mkdtemp(prefix="paper_agent_"))
    store = HFStore(root, repo, token=token)
    print(f"[tick] {repo}: {store.restore()}")
    if a.dry_run:
        dry = LocalStore(root / "dry")
        for f in ("state.json", "log.jsonl"):
            if (root / f).exists():
                shutil.copy(root / f, dry.root / f)
        store = dry
    trader = PaperTrader(store, agent_kind="MV")
    if trader.agent.kind not in LIVE_KINDS:
        raise SystemExit(f"stored agent is {trader.agent.kind}, barred from live use")

    from serve.fetch import fetch_bars
    from serve.history import get_hours
    now = int(time.time())
    hours, info = get_hours(fetch_bars, persist=False, verbose=False, now_ts=now)
    events = trader.tick(now, hours)
    for e in events:
        print("[tick]", json.dumps({k: e.get(k) for k in ("type", "E", "a", "w_prev", "R", "u", "regret", "sigma")}))
    print(f"[tick] feed {info.get('source')}, last bar {int(hours.hour_ts.iloc[-1])}; "
          f"position {trader.w:+.1f}; learned from {trader.agent.n_updates} windows; "
          f"{len(trader.pending)} pending; sync: {trader.last_flush or 'nothing changed'}")
    if trader.last_flush and trader.last_flush.startswith("upload failed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
