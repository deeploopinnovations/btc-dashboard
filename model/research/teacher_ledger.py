"""
research/teacher_ledger.py
=====================================================================
The Phase 2 record: candidates, mechanisms, and what was transferred.

WHY THIS IS SEPARATE FROM research/ledger.py

`ledger.json` records EXPERIMENTS -- a question, a pre-registered rule, a
verdict. Phase 2 also has to record two other kinds of thing that do not fit
that shape:

  CANDIDATES   a model with provenance: licence, checkpoint hash, parameter
               count, RAM, context length, pretraining-corpus leakage risk.
               A candidate is not an experiment; it is an object that
               experiments are run on, and it can be BLOCKED_BY_INFRASTRUCTURE
               or DISQUALIFIED_LEAKAGE without any experiment existing.

  MECHANISMS   a falsifiable claim about WHY a teacher wins, with the test
               that would kill it and a `mechanism` tag. The tag is what the
               stagnation supervisor counts: three consecutive failures
               carrying the same tag closes that mechanism (TEACHER_ZOO 7).

Squeezing those into the experiment schema would mean either losing the
provenance fields or filling them with nulls on two thirds of the rows.

Append-only, and for the same reason as `ledger.py`: a wrong entry is
superseded, never edited, so the record shows what was believed at the time.

    python -m model.research.teacher_ledger --validate
    python -m model.research.teacher_ledger --scorecard
    python -m model.research.teacher_ledger --stagnation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LEDGER = Path(__file__).with_name("teacher_ledger.json")

VERDICTS = ("OPEN", "ADOPT", "ADVANCE", "REJECT", "NULL", "WITHDRAWN",
            "BLOCKED_BY_INFRASTRUCTURE", "DISQUALIFIED_LEAKAGE")

# BLOCKED_BY_INFRASTRUCTURE is deliberately NOT a synonym for REJECT.
# TEACHER_ZOO section 4: a model we could not run is not a model that failed,
# and recording it as a failure would quietly convert an environment limit into
# a scientific claim about the model.

CANDIDATE_KEYS = ("id", "name", "verdict", "source", "licence_code",
                  "licence_weights", "leakage_risk", "eval_mode")
MECHANISM_KEYS = ("id", "teacher", "mechanism", "claim", "falsifier", "verdict")
LEAKAGE = ("HIGH", "MEDIUM", "LOW", "UNKNOWN", "NOT_APPLICABLE")


def load(path: Path = LEDGER) -> dict:
    if not path.exists():
        return {"schema": 1, "candidates": [], "mechanisms": []}
    return json.loads(path.read_text())


def save(d: dict, path: Path = LEDGER) -> None:
    path.write_text(json.dumps(d, indent=1) + "\n")


def _add(section: str, keys: tuple, entry: dict, path: Path) -> dict:
    missing = [k for k in keys if not entry.get(k)]
    if missing:
        raise ValueError(f"{section} entry missing {missing}")
    if entry["verdict"] not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    d = load(path)
    if any(e["id"] == entry["id"] for e in d[section]):
        raise ValueError(f"{entry['id']} exists; supersede it, do not edit it")
    d[section].append(entry)
    for prev in entry.get("supersedes", []):
        for e in d[section]:
            if e["id"] == prev:
                e.setdefault("superseded_by", []).append(entry["id"])
    save(d, path)
    return entry


def add_candidate(entry: dict, path: Path = LEDGER) -> dict:
    if entry.get("leakage_risk") not in LEAKAGE:
        raise ValueError(f"leakage_risk must be one of {LEAKAGE}")
    return _add("candidates", CANDIDATE_KEYS, entry, path)


def add_mechanism(entry: dict, path: Path = LEDGER) -> dict:
    return _add("mechanisms", MECHANISM_KEYS, entry, path)


def validate(d: dict) -> list[str]:
    """Checked against the FILE, not the write path -- research/ledger.py was
    corrupted twice by scripts appending directly, and this file will be
    written by scripts too."""
    errs, seen = [], set()
    for section, keys in (("candidates", CANDIDATE_KEYS),
                          ("mechanisms", MECHANISM_KEYS)):
        for i, e in enumerate(d.get(section, [])):
            who = e.get("id", f"<{section}[{i}]>")
            for k in keys:
                if not e.get(k):
                    errs.append(f"{who}: missing or empty `{k}`")
            if e.get("verdict") and e["verdict"] not in VERDICTS:
                errs.append(f"{who}: verdict {e['verdict']!r} not in {VERDICTS}")
            if e.get("id") in seen:
                errs.append(f"{who}: duplicate id")
            seen.add(e.get("id"))
        ids = {e.get("id") for e in d.get(section, [])}
        for e in d.get(section, []):
            for k in ("supersedes", "superseded_by"):
                for ref in e.get(k, []):
                    if ref not in ids:
                        errs.append(f"{e.get('id')}: {k} -> unknown id {ref!r}")
    for e in d.get("candidates", []):
        if e.get("leakage_risk") not in LEAKAGE:
            errs.append(f"{e.get('id')}: leakage_risk {e.get('leakage_risk')!r} "
                        f"not in {LEAKAGE}")
        # A model that could not be run must not carry a scientific verdict.
        if e.get("verdict") == "BLOCKED_BY_INFRASTRUCTURE" and e.get("scored"):
            errs.append(f"{e.get('id')}: BLOCKED_BY_INFRASTRUCTURE but marked scored")
    return errs


def stagnation(d: dict, limit: int = 3) -> list[str]:
    """TEACHER_ZOO section 7. Three consecutive failures carrying the same
    `mechanism` tag closes that mechanism.

    Counts CONSECUTIVE failures in insertion order within each tag, not the
    total: a mechanism that failed twice, then produced a real result, then
    failed once, is not stagnant. Getting that wrong would close a mechanism
    that had just been shown to work."""
    runs: dict[str, int] = {}
    stopped = []
    for e in d.get("mechanisms", []):
        tag = e.get("mechanism")
        if not tag:
            continue
        if e.get("verdict") in ("REJECT", "NULL"):
            runs[tag] = runs.get(tag, 0) + 1
            if runs[tag] >= limit and tag not in stopped:
                stopped.append(tag)
        elif e.get("verdict") in ("ADOPT", "ADVANCE"):
            runs[tag] = 0
    return stopped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 teacher ledger")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stagnation", action="store_true",
                    help="mechanisms closed by three consecutive failures")
    ap.add_argument("--path", type=Path, default=LEDGER)
    a = ap.parse_args(argv)
    d = load(a.path)

    if a.validate:
        errs = validate(d)
        for e in errs:
            print(f"  {e}")
        print(f"{len(d.get('candidates', []))} candidate(s), "
              f"{len(d.get('mechanisms', []))} mechanism(s), {len(errs)} problem(s)")
        return 1 if errs else 0

    if a.stagnation:
        st = stagnation(d)
        for t in st:
            print(f"  STOP: mechanism {t!r} has 3 consecutive failures — "
                  f"return to the scorecard and pick a different information source")
        print(f"{len(st)} mechanism(s) closed by the stagnation rule")
        return 0

    for e in d.get("candidates", []):
        flag = f"  [leak {e['leakage_risk']}]" if e.get("leakage_risk") not in (
            "LOW", "NOT_APPLICABLE") else ""
        print(f"{e['id']:<22} {e['verdict']:<26} {e['name']}{flag}")
    for e in d.get("mechanisms", []):
        print(f"{e['id']:<22} {e['verdict']:<26} [{e['mechanism']}] {e['claim'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
