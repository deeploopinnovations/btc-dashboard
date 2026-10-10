"""
runs/noctua-disproof-2026-10-10/loop.py
=====================================================================
External state for the NOCTUA disproof loop. Everything a later session
needs to resume lives in this folder; nothing lives only in a chat.

    trajectory.jsonl   one line per command/tool call: iter, phase, cmd, outcome
    MEMORY.md          the loop's persistent memory: established / refuted /
                       lessons (execution feedback) / open questions
    plans/iter_NN.md   the plan written BEFORE each implement step
    evals/*.json       every evaluation's raw numbers
    audits/*.md        every auditor's verdict, verbatim
    supervisor.json    stagnation state, written by `supervise`

Loop: inspect -> plan -> implement -> evaluate -> audit -> supervise.

Supervisor rule: an iteration's "evidence signature" is the set of claim
ids whose status changed plus the failure signature (first line of any
error). If two consecutive iterations add no new claim status AND repeat
the same failure signature (or both have none), the path is declared
stagnant: it is stopped, the reason is written to MEMORY.md, and the next
plan must change approach.

    python loop.py log  --iter 1 --phase implement --cmd "..." --outcome "..."
    python loop.py iter --iter 1 --claims C1:survived,C2:refuted --failure ""
    python loop.py supervise
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
TRAJ = HERE / "trajectory.jsonl"
SUP = HERE / "supervisor.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(it: int, phase: str, cmd: str, outcome: str) -> None:
    with TRAJ.open("a") as f:
        f.write(json.dumps({"ts": _now(), "iter": it, "phase": phase,
                            "cmd": cmd, "outcome": outcome}) + "\n")


def _state() -> dict:
    return json.loads(SUP.read_text()) if SUP.exists() else {"iterations": [], "stopped_paths": []}


def record_iter(it: int, claims: dict, failure: str, path: str) -> None:
    s = _state()
    s["iterations"] = [r for r in s["iterations"] if r["iter"] != it]
    s["iterations"].append({"iter": it, "path": path, "claims": claims,
                            "failure": failure.strip().splitlines()[0] if failure.strip() else "",
                            "ts": _now()})
    s["iterations"].sort(key=lambda r: r["iter"])
    SUP.write_text(json.dumps(s, indent=2) + "\n")


def supervise() -> dict:
    """Return {'stagnant': bool, 'reason': str} for the latest path."""
    s = _state()
    its = s["iterations"]
    verdict = {"stagnant": False, "reason": "fewer than two iterations on this path"}
    if len(its) >= 2:
        a, b = its[-2], its[-1]
        seen = {}
        for r in its[:-1]:
            seen.update(r["claims"])
        new = {k: v for k, v in b["claims"].items() if seen.get(k) != v}
        same_fail = a["failure"] == b["failure"]
        if a["path"] == b["path"] and not new and same_fail:
            verdict = {"stagnant": True,
                       "reason": f"iters {a['iter']} and {b['iter']} on path '{b['path']}' "
                                 f"added no new claim status and repeated failure "
                                 f"'{b['failure'] or 'none'}'"}
            if b["path"] not in s["stopped_paths"]:
                s["stopped_paths"].append(b["path"])
        else:
            verdict = {"stagnant": False,
                       "reason": f"new claim status: {sorted(new)}" if new else "failure changed"}
    s["last_verdict"] = verdict | {"ts": _now()}
    SUP.write_text(json.dumps(s, indent=2) + "\n")
    return verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    l = sub.add_parser("log")
    l.add_argument("--iter", type=int, required=True)
    l.add_argument("--phase", required=True)
    l.add_argument("--cmd", required=True)
    l.add_argument("--outcome", required=True)
    i = sub.add_parser("iter")
    i.add_argument("--iter", type=int, required=True)
    i.add_argument("--claims", default="")
    i.add_argument("--failure", default="")
    i.add_argument("--path", default="main")
    sub.add_parser("supervise")
    a = ap.parse_args()
    if a.op == "log":
        log(a.iter, a.phase, a.cmd, a.outcome)
    elif a.op == "iter":
        cl = dict(kv.split(":", 1) for kv in a.claims.split(",") if kv)
        record_iter(a.iter, cl, a.failure, a.path)
    else:
        print(json.dumps(supervise(), indent=2))
