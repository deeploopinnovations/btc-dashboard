"""
space/forward.py
=====================================================================
The forward holdout for NOCTUA-Trader v1, run on a Hugging Face Space.

Frozen setup: the v1 policy weights (model/policy/weights/noctua_trader_v1.npz)
on top of the pinned NOCTUA forecaster (model/serve/noctua_v2.npz). Nothing is
retrained here. Every 17:00 UTC anchor from FORWARD_START on is decided once,
logged with the wall-clock time it was decided, and settled 24 h later.

Paper trading only: no orders, no exchange keys. Data comes from public
endpoints (Bitstamp 5-minute bars, Coinbase as fallback; Deribit DVOL and
perpetual funding).

A free Space sleeps when nobody visits it. When it wakes, missed anchors are
decided from data that existed at the anchor (every input reads rows <= a-1),
so a late decision equals the on-time one; it is still flagged `late` so the
record shows what was decided in real time.

State (the log plus the fetched data tails) persists in a private Hugging Face
dataset repo, LOG_REPO, written with the HF_TOKEN Space secret. Without either,
the run still works and keeps its state on local disk only.

    python space/forward.py            # one pass: refresh, decide, settle, push
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from policy import backtest as BT              # noqa: E402
from policy import dataset as DS               # noqa: E402
from policy.runtime import DEFAULT_WEIGHTS, NumpyTrader  # noqa: E402

FORWARD_START = int(pd.Timestamp("2026-10-10 17:00", tz="UTC").timestamp())
DAY = 86400
CONST_POS = 0.14            # the backtest's average position: the "just hold a bit" baseline
LATE_AFTER_S = 3600         # decided more than an hour after the anchor
NEWDATA = ROOT / "data" / "newdata"
STATE = Path(os.environ.get("NOCTUA_STATE_DIR", ROOT / "state"))
LOG_REPO = os.environ.get("LOG_REPO", "")
STATE_FILES = ("forward_log.json", "hours_tail.parquet", "dvol_btc.parquet", "funding_btc.parquet")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


FROZEN = {
    "policy": "NOCTUA-Trader-v1",
    "policy_sha256": sha256(DEFAULT_WEIGHTS),
    "forecaster_sha256": sha256(DS.NOCTUA_ARTIFACT),
    "forward_start_utc": datetime.fromtimestamp(FORWARD_START, timezone.utc).isoformat(),
}


# ---------------------------------------------------------------- state I/O
def _api():
    if not (LOG_REPO and os.environ.get("HF_TOKEN")):
        return None
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def pull_state() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    api = _api()
    if api is None:
        return
    from huggingface_hub import hf_hub_download
    for f in STATE_FILES:
        try:
            p = hf_hub_download(LOG_REPO, f, repo_type="dataset", token=api.token,
                                local_dir=STATE, force_download=True)
        except Exception:  # noqa: BLE001 - first run: nothing there yet
            continue
        if Path(p) != STATE / f:
            (STATE / f).write_bytes(Path(p).read_bytes())


def push_state(message: str) -> bool:
    api = _api()
    if api is None:
        return False
    api.upload_folder(repo_id=LOG_REPO, repo_type="dataset", folder_path=str(STATE),
                      allow_patterns=list(STATE_FILES), commit_message=message)
    return True


# ------------------------------------------------------------ data refresh
def _union(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    if new.empty:
        return old
    both = pd.concat([old, new[[c for c in new.columns if c in old.columns]]], ignore_index=True)
    return both.drop_duplicates("ts", keep="last").sort_values("ts", ignore_index=True)


def refresh_exogenous() -> dict:
    """Bring DVOL and funding up to now. The policy reads them from
    data/newdata/, so the merged series is written back there."""
    from eval import harvest_newdata as HN
    now_ms = int(time.time() * 1000)
    out = {}
    for name, fetch, cols in (
        ("dvol_btc", HN.fetch_deribit_dvol_index, ["ts_ms", "volatility", "ts"]),
        ("funding_btc", HN.fetch_deribit_funding, None),
    ):
        path = NEWDATA / f"{name}.parquet"
        cur = pd.read_parquet(path)
        cached = STATE / f"{name}.parquet"
        if cached.exists():
            cur = _union(cur, pd.read_parquet(cached))
        since_ms = (int(cur.ts.max()) - 2 * DAY) * 1000
        try:
            new = fetch(since_ms, now_ms, verbose=False)
        except Exception as e:  # noqa: BLE001 - stale inputs are flagged per decision
            new = pd.DataFrame()
            out[name + "_error"] = f"{type(e).__name__}: {e}"
        if cols and not new.empty:
            new = new[cols]
        merged = _union(cur, new)
        merged.to_parquet(path)
        tail = merged[merged.ts >= int(time.time()) - 60 * DAY]
        tail.to_parquet(cached)
        out[name + "_last_utc"] = datetime.fromtimestamp(int(merged.ts.max()), timezone.utc).isoformat()
    return out


def refresh_hours() -> pd.DataFrame:
    from policy.paper import fetch_tail
    from serve.history import merge
    h = DS.load_hours()
    committed_end = int(h.hour_ts.iloc[-1])
    cached = STATE / "hours_tail.parquet"
    if cached.exists():
        h = merge(h, pd.read_parquet(cached))
    h = fetch_tail(h)                              # raises on any gap
    h[h.hour_ts > committed_end].reset_index(drop=True).to_parquet(cached)
    return h


# --------------------------------------------------------- decide / settle
def decide(h: pd.DataFrame, model: NumpyTrader, anchor_ts: int) -> dict:
    ts = h.hour_ts.to_numpy(np.int64)
    row = int(np.searchsorted(ts, anchor_ts))
    hh = h
    if row >= len(h):                              # anchor bar not printed yet
        pad = h.iloc[[-1]].copy()
        pad["hour_ts"] = int(ts[-1]) + 3600
        hh = pd.concat([h, pad], ignore_index=True)
    assert int(hh.hour_ts.iloc[row]) == anchor_ts and int(hh.hour_ts.iloc[row - 1]) == anchor_ts - 3600
    F = DS.features_at(hh, np.array([row]))
    pos = float(model.position(F)[0])
    sig19 = float(np.exp(F.nx_logsig_med.iloc[0]))
    now = int(time.time())
    return {
        "anchor_utc": datetime.fromtimestamp(anchor_ts, timezone.utc).isoformat(),
        "anchor_ts": anchor_ts,
        "decided_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "late": bool(now - anchor_ts > LATE_AFTER_S),
        "entry_price": float(hh.close.iloc[row - 1]),
        "position": round(pos, 4),
        "noctua_sigma_19h": round(sig19, 5),
        "noctua_ann_vol": round(sig19 * np.sqrt(24 * 365 / DS.NOCTUA_H), 4),
        "dvol": round(float(np.exp(F.dvol_log.iloc[0]) * 100), 2) if F.dvol_present.iloc[0] else None,
        "stale_inputs": [k for k, c in (("dvol", "dvol_present"), ("funding", "fund_present"))
                         if F[c].iloc[0] == 0],
        "policy_sha256": FROZEN["policy_sha256"],
        "forecaster_sha256": FROZEN["forecaster_sha256"],
    }


def settle(log: list[dict], h: pd.DataFrame) -> None:
    """24 h hold: entry close[a-1], exit close[a+23], funding over a..a+23.
    Same arithmetic as the backtest (policy/backtest.pnl), fees included."""
    ts = h.hour_ts.to_numpy(np.int64)
    close = h.close.to_numpy(np.float64)
    fund = DS._hourly_series(NEWDATA / "funding_btc.parquet", "interest_1h", "ts", ts)
    prev = {"trader": 0.0, "const": 0.0}
    for e in sorted(log, key=lambda e: e["anchor_ts"]):
        a = e["anchor_ts"]
        exit_ts = a + (DS.HOLD_H - 1) * 3600
        if "pnl" not in e and exit_ts <= ts[-1]:
            i0, i1 = int(np.searchsorted(ts, a)), int(np.searchsorted(ts, exit_ts))
            px = float(close[i1])
            r = float(np.log(px / e["entry_price"]))
            f = fund[i0:i1 + 1]
            e["exit_price"] = px
            e["btc_return"] = float(np.exp(r) - 1)
            e["funding"] = float(np.nansum(f))
            e["funding_complete"] = bool(np.isfinite(f).all())
            e["pnl"] = float(BT.pnl(np.array([e["position"]]), np.array([r]), np.array([e["funding"]]),
                                    prev0=prev["trader"])[0])
            e["pnl_buy_hold"] = float(np.exp(r) - 1 - e["funding"])
            e["pnl_const"] = float(BT.pnl(np.array([CONST_POS]), np.array([r]), np.array([e["funding"]]),
                                          prev0=prev["const"])[0])
            e["settled_utc"] = datetime.now(timezone.utc).isoformat()
        prev = {"trader": e["position"], "const": CONST_POS}


def summary(log: list[dict]) -> dict:
    done = sorted((e for e in log if "pnl" in e), key=lambda e: e["anchor_ts"])
    out = {"decisions": len(log), "settled": len(done),
           "late_decisions": sum(e.get("late", False) for e in log), **FROZEN}
    for k, col in (("trader", "pnl"), ("buy_hold", "pnl_buy_hold"), ("const_0.14", "pnl_const")):
        p = np.array([e[col] for e in done])
        if len(p):
            m = BT.metrics(p)
            out[k] = {"total_return": round(m["total_return"], 4),
                      "sharpe": round(m["sharpe"], 2) if len(p) >= 20 else None,
                      "max_drawdown": round(m["max_drawdown"], 4)}
    return out


def run_once(push: bool = True) -> dict:
    pull_state()
    info = refresh_exogenous()
    h = refresh_hours()
    path = STATE / "forward_log.json"
    log = json.loads(path.read_text()) if path.exists() else []
    seen = {e["anchor_ts"] for e in log}
    model = NumpyTrader()
    last_closed = int(h.hour_ts.iloc[-1])          # anchor a needs bar a-1 closed
    new = 0
    for a in range(FORWARD_START, last_closed + 3600 + 1, DAY):
        if a not in seen:
            log.append(decide(h, model, a))
            new += 1
    before = sum("pnl" in e for e in log)
    settle(log, h)
    settled = sum("pnl" in e for e in log) - before
    log.sort(key=lambda e: e["anchor_ts"])
    path.write_text(json.dumps(log, indent=1))
    pushed = False
    if push and (new or settled):
        pushed = push_state(f"forward: +{new} decisions, +{settled} settled")
    return {"new_decisions": new, "newly_settled": settled, "pushed": pushed,
            "data_last_utc": datetime.fromtimestamp(last_closed, timezone.utc).isoformat(),
            **info, "summary": summary(log), "latest": log[-1] if log else None}


if __name__ == "__main__":
    print(json.dumps(run_once(push="--no-push" not in sys.argv), indent=1, default=str))
