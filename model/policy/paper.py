"""
policy/paper.py
=====================================================================
Local PAPER trader for NOCTUA-Trader. It places no orders and needs no
exchange keys: it computes today's target position at the 17:00 UTC
anchor, appends it to a JSON log, and settles earlier entries once their
24 h hold has elapsed in the price history.

    python policy/paper.py                 # committed history only (offline)
    python policy/paper.py --fetch         # also fetch the latest hourly tail

Log: data/noctua_trader_paper.json (override with --log).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import backtest as BT              # noqa: E402
from policy import dataset as DS               # noqa: E402
from policy.runtime import NumpyTrader         # noqa: E402

DEFAULT_LOG = DS.ROOT / "data" / "noctua_trader_paper.json"


def fetch_tail(h: pd.DataFrame) -> pd.DataFrame:
    """Committed bars plus a live tail long enough to close the gap since the
    bundle ends. A fixed 72 h tail on an older bundle leaves a hole, and every
    trailing feature counts rows, not hours, so the hole must fail loudly."""
    import time
    from serve.fetch import fetch_bars
    from serve.history import check_continuity, hours_from_bars, merge
    behind_h = int((time.time() - int(h.hour_ts.iloc[-1])) // 3600) + 6
    h = merge(h, hours_from_bars(fetch_bars(tail_hours=max(72, behind_h))))
    c = check_continuity(h, max_gap_hours=1)
    if not c["contiguous"]:
        raise RuntimeError(f"hourly history has a gap after the live merge: {c}")
    return h


def latest_anchor_row(h: pd.DataFrame, hour: int = DS.PROD_HOUR) -> int:
    """Row index of the most recent anchor at `hour` UTC whose prior hour is
    closed (the anchor row itself need not exist yet)."""
    ts = h.hour_ts.to_numpy(np.int64)
    nxt = int(ts[-1]) + 3600                     # the anchor right after the last bar
    a = nxt - ((nxt // 3600 - hour) % 24) * 3600
    return int(np.searchsorted(ts, a)) if a <= ts[-1] else len(ts)


def decide(h: pd.DataFrame, model: NumpyTrader) -> dict:
    row = latest_anchor_row(h)
    hh = h
    if row >= len(h):                           # anchor bar not printed yet: pad one
        pad = h.iloc[[-1]].copy()
        pad["hour_ts"] = int(h.hour_ts.iloc[-1]) + 3600
        hh = pd.concat([h, pad], ignore_index=True)
    F = DS.features_at(hh, np.array([row]))
    stale = []
    if F.dvol_present.iloc[0] == 0:
        stale.append("dvol")
    if F.fund_present.iloc[0] == 0:
        stale.append("funding")
    pos = float(model.position(F)[0])
    return {
        "anchor_utc": datetime.fromtimestamp(int(F.anchor_ts.iloc[0]), timezone.utc).isoformat(),
        "anchor_ts": int(F.anchor_ts.iloc[0]),
        "entry_price": float(hh.close.iloc[row - 1]),
        "position": round(pos, 4),
        "stale_inputs": stale,
        "model": model.meta["name"] + f"-v{model.meta['version']}",
    }


def settle(log: list[dict], h: pd.DataFrame, fee_bps: float = BT.DEFAULT_FEE_BPS) -> None:
    ts = h.hour_ts.to_numpy(np.int64)
    close = h.close.to_numpy(np.float64)
    prev = 0.0
    for e in sorted(log, key=lambda e: e["anchor_ts"]):
        exit_ts = e["anchor_ts"] + (DS.HOLD_H - 1) * 3600
        if "pnl" not in e and exit_ts <= ts[-1]:
            px = float(close[int(np.searchsorted(ts, exit_ts))])
            r = np.log(px / e["entry_price"])
            # funding is not in the committed bars, so paper P&L excludes it
            e["exit_price"] = px
            e["pnl"] = float(BT.pnl(np.array([e["position"]]), np.array([r]),
                                    np.array([0.0]), fee_bps, prev0=prev)[0])
        prev = e["position"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="fetch the latest hourly tail")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--weights", type=Path, default=None)
    a = ap.parse_args(argv)

    h = DS.load_hours()
    if a.fetch:
        h = fetch_tail(h)
    model = NumpyTrader(a.weights) if a.weights else NumpyTrader()

    log = json.loads(a.log.read_text()) if a.log.exists() else []
    d = decide(h, model)
    if not any(e["anchor_ts"] == d["anchor_ts"] for e in log):
        log.append(d)
    settle(log, h)
    a.log.write_text(json.dumps(log, indent=1))

    done = [e["pnl"] for e in log if "pnl" in e]
    print(json.dumps(d, indent=1))
    if done:
        print("settled paper days:", len(done), "| total:",
              round(float(np.prod(1 + np.array(done)) - 1), 4))
    if d["stale_inputs"]:
        print("WARNING: no fresh", ", ".join(d["stale_inputs"]),
              "at this anchor; the policy saw them as missing.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
