"""
research/ledger.py
=====================================================================
Persistent memory: every experiment, its PRE-REGISTERED rule, and its verdict.

WHY

This project's discipline is pre-registration -- fix the decision rule before
the data is scored -- and its record of that lives in prose across ~1,500 lines
of BENCHMARK.md. Prose cannot be queried, so three things kept happening:

  * an experiment was re-run without noticing it repeated an earlier one
    (6l -> 6m -> 6n were three designs of one question, and the fact that all
    three disagreed only became visible when someone tabulated them by hand);
  * a superseded number stayed in circulation (the 1.5218 benchmark, and the
    "92% is shape" figure, both quoted for weeks after being wrong);
  * a rule's ambiguity was discovered only at the moment of applying it
    ("moves by >= 1pp" -- moves, or improves?).

The ledger is the machine-readable half. It does not replace BENCHMARK.md,
which carries the reasoning; it carries the STATE, so a supervisor can ask
questions like "how many times has this statistic been decided by a win count"
without reading anything.

APPEND-ONLY. A wrong entry is superseded, never edited: `supersedes` and
`superseded_by` make the correction history part of the record, because in this
project the corrections are as informative as the results.

    python -m model.research.ledger --list
    python -m model.research.ledger --open        # unresolved questions
    python -m model.research.ledger --corrections # what turned out to be wrong
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEDGER = Path(__file__).with_name("ledger.json")

# ADVANCE is not a synonym for ADOPT and the distinction is load-bearing.
#
#   ADOPT     the change is in the shipped artifact.
#   ADVANCE   the change cleared every condition of its own pre-registered rule
#             and is STILL NOT ADOPTED, because the rule itself said it could
#             not adopt alone -- typically because the experiment measured a
#             cheaper proxy than the product. E2c re-weighted a recorded median
#             and never rebuilt the barrier curves, and NOCTUA's product is a
#             touch-probability curve, not a sigma.
#   NULL      ran, decided nothing.
#   REJECT    ran, failed its rule.
#
# The vocabulary refused ADVANCE the first time it was used, which was the
# ledger working: a state with no name is a state that gets quietly folded into
# a neighbouring one, and folding ADVANCE into ADOPT is how an unconfirmed
# result reaches an artifact.
VERDICTS = ("ADOPT", "ADVANCE", "REJECT", "NULL", "WITHDRAWN", "OPEN")


def load(path: Path = LEDGER) -> dict:
    if not path.exists():
        return {"experiments": [], "schema": 1}
    return json.loads(path.read_text())


def save(d: dict, path: Path = LEDGER) -> None:
    path.write_text(json.dumps(d, indent=2) + "\n")


def add(entry: dict, path: Path = LEDGER) -> dict:
    """Append one experiment. Refuses an entry missing its pre-registered rule."""
    required = ("id", "question", "rule", "verdict")
    missing = [k for k in required if not entry.get(k)]
    if missing:
        raise ValueError(f"ledger entry missing {missing}; a result without a "
                         f"pre-registered rule is not admissible here")
    if entry["verdict"] not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    d = load(path)
    if any(e["id"] == entry["id"] for e in d["experiments"]):
        raise ValueError(f"{entry['id']} exists; supersede it, do not edit it")
    d["experiments"].append(entry)
    for prev in entry.get("supersedes", []):
        for e in d["experiments"]:
            if e["id"] == prev:
                e.setdefault("superseded_by", []).append(entry["id"])
    save(d, path)
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Research ledger")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--open", action="store_true", help="unresolved questions")
    ap.add_argument("--corrections", action="store_true",
                    help="entries that were superseded, and by what")
    ap.add_argument("--path", type=Path, default=LEDGER)
    a = ap.parse_args(argv)
    d = load(a.path)
    es = d["experiments"]

    if a.corrections:
        n = 0
        for e in es:
            if e.get("superseded_by") or e["verdict"] == "WITHDRAWN":
                n += 1
                print(f"{e['id']:16} {e['verdict']:10} {e['question'][:64]}")
                for s in e.get("superseded_by", []):
                    print(f"{'':16} -> superseded by {s}")
        print(f"\n{n} of {len(es)} entries were corrected or withdrawn.")
        return 0

    if a.open:
        # A SUPERSEDED entry is not an open question, whatever its verdict says.
        # Pre-registrations are appended with verdict OPEN and later superseded
        # by the entry carrying the result -- the ledger is append-only, so the
        # original keeps the word "OPEN" forever. Listing those here made
        # settled work look unstarted, which is exactly the mistake this view
        # exists to prevent: `supervisor.py` reads it before proposing an
        # experiment, so a stale row here means re-running a finished one.
        shown = 0
        for e in es:
            if e["verdict"] == "OPEN" and not e.get("superseded_by"):
                print(f"{e['id']:16} {e['question']}")
                print(f"{'':16} rule: {e['rule']}")
                shown += 1
        stale = sum(1 for e in es
                    if e["verdict"] == "OPEN" and e.get("superseded_by"))
        print(f"\n{shown} open, {stale} pre-registration(s) already answered "
              f"by a superseding entry (hidden)")
        return 0

    print(f"{'id':16} {'verdict':10} {'question'}")
    for e in es:
        flag = " *" if e.get("superseded_by") else ""
        print(f"{e['id']:16} {e['verdict']:10} {e['question'][:70]}{flag}")
    print(f"\n{len(es)} experiments. '*' = later superseded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
