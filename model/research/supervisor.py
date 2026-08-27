"""
research/supervisor.py
=====================================================================
Stagnation detection: notice when the RULE is failing, not the model.

THE FAILURE MODE THIS EXISTS FOR

Three experiments -- 6l, 6m, 6n -- asked one question ("does refreshing the
training window help?") and produced 2/6, 5/6 and 3/6 on the same deep-tail
statistic, because a six-sample win count on an effect of ~0.003 against
per-window scatter of ~0.013 is a coin flip. The mean favoured the refresh in
all three. Nobody noticed the pattern until the third design, and noticing it
required tabulating them by hand.

That is a trajectory a supervisor should have interrupted after the second
design, with a specific redirection: stop re-running the experiment, change the
statistic. Re-running a decision procedure that cannot resolve the effect is
not evidence-gathering, it is sampling noise repeatedly and reporting whichever
draw arrives.

WHAT IT DETECTS

  1. REPETITION      -- one question attacked N times with different designs
  2. OSCILLATION     -- verdicts flipping across those attempts
  3. UNRESOLVABILITY -- the deciding statistic is smaller than its own
                        standard error at the sample size being used
  4. STALE NUMBERS   -- a withdrawn or superseded result still being cited
  5. UNCLOSED LOOPS  -- questions left OPEN with no follow-up recorded

Each detection carries a REDIRECTION -- what to do instead -- because a
supervisor that only says "you are stuck" adds nothing a frustrated researcher
did not already know.

    python -m model.research.supervisor
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.research.ledger import load                                    # noqa: E402

# questions that are "the same question" wear different ids; group by topic
TOPIC = [
    ("refresh",  r"refresh|training window|frozen"),
    ("shape",    r"shape|travel|first-passage|barrier error"),
    ("direction", r"direction|sign of"),
    ("head",     r"head\b|q_mx|lam_r"),
    ("regime",   r"regime|post.etf|flag"),
]


def topic_of(entry) -> str:
    """Explicit `topic` first, regex only as a fallback.

    The regex-only version MISSED the 6l/6m/6n/6o sequence -- the exact case
    this supervisor was built for -- because 6m, 6n and 6o phrase their
    question as "Same, with ...". A stagnation detector that cannot see the
    stagnation it was written for is worse than none, since it reports "no
    alerts" and is believed. Found by running it, which is the point of
    running it.
    """
    if isinstance(entry, dict) and entry.get("topic"):
        return entry["topic"]
    q = entry["question"] if isinstance(entry, dict) else str(entry)
    for name, pat in TOPIC:
        if re.search(pat, q, re.I):
            return name
    return "other"


def detect(d: dict) -> list:
    es = d["experiments"]
    alerts = []

    by_topic = defaultdict(list)
    for e in es:
        by_topic[topic_of(e)].append(e)

    for topic, group in sorted(by_topic.items()):
        if topic == "other" or len(group) < 3:
            continue
        verdicts = [e["verdict"] for e in group]
        ids = ", ".join(e["id"] for e in group)
        alerts.append(("REPETITION", topic,
                       f"{len(group)} attempts on one topic ({ids})",
                       "before attempting again, state what the NEXT design "
                       "measures that the previous ones could not"))
        distinct = {v for v in verdicts if v in ("ADOPT", "REJECT")}
        if len(distinct) > 1:
            alerts.append(("OSCILLATION", topic,
                           f"verdicts flip across attempts: {verdicts}",
                           "a verdict that changes with the design is a "
                           "property of the design; decide on the effect size "
                           "with an interval, not on a count"))

    for e in es:
        if e.get("superseded_by") or e["verdict"] == "WITHDRAWN":
            alerts.append(("STALE", e["id"],
                           f"withdrawn/superseded: {e['result'][:70]}",
                           "grep the repo for this number before citing it; "
                           f"successor: {', '.join(e.get('superseded_by', [])) or 'none'}"))
        # An OPEN entry is only an unclosed promise if it is BOTH unsuperseded
        # and carries no substantive result.
        #
        # The ledger is append-only, so a pre-registration written with verdict
        # OPEN keeps that word forever, even after the entry carrying its
        # result supersedes it. The first version of this check flagged E2b and
        # E2c as unclosed while both were finished and superseded -- telling a
        # supervisor to re-run completed experiments, which is precisely the
        # waste it exists to prevent. Same defect, and same fix, as
        # `ledger.py --open`.
        #
        # The second half matters too: a structural MEASUREMENT (coverage
        # counts) or an AMENDMENT is recorded with verdict OPEN because it is
        # not a hypothesis test, but it has a result and is not a promise. The
        # marker for a genuine promise is the placeholder text a
        # pre-registration is written with.
        if e["verdict"] == "OPEN" and not e.get("superseded_by"):
            # The marker is the OPENING of the result field, not its length.
            # This project writes a promise as "pre-registered before the run,
            # ..." and everything else -- a measurement, a prediction, an
            # amendment -- opens with its own content. A length cutoff was the
            # first attempt and it silently exempted the two longest promises,
            # E-scale and E2-confirm, because their pre-registrations carry
            # motivation text. A filter that lets the most elaborate promises
            # through is worse than no filter.
            if (e.get("result") or "").strip().lower().startswith("pre-registered"):
                alerts.append(("UNCLOSED", e["id"], e["question"][:70],
                               "an OPEN question with no result is a promise; "
                               "either run it or record why it was dropped"))
    return alerts


def unresolvable(deltas, name: str = "statistic") -> tuple:
    """Is the deciding statistic even resolvable at this sample size?"""
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("inf")
    ok = abs(d.mean()) > se
    msg = (f"{name}: n={n}, |mean| {abs(d.mean()):.5f}, se {se:.5f} -> "
           + ("resolvable" if ok else "NOT resolvable"))
    redirect = ("" if ok else
                f"do not re-run this design. Either raise n to about "
                f"{int(np.ceil((d.std(ddof=1)/max(abs(d.mean()),1e-12))**2))} "
                f"units, or decide on the mean with a bootstrap CI instead of "
                f"a win count.")
    return ok, msg, redirect


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect stagnation in the record")
    ap.add_argument("--path", type=Path, default=None)
    a = ap.parse_args(argv)
    d = load(a.path) if a.path else load()

    alerts = detect(d)
    order = {"OSCILLATION": 0, "REPETITION": 1, "STALE": 2, "UNCLOSED": 3}
    alerts.sort(key=lambda x: order.get(x[0], 9))
    for kind, subject, detail, redirect in alerts:
        print(f"[{kind:11}] {subject}")
        print(f"              {detail}")
        print(f"        ---> {redirect}\n")

    print("worked example -- the deep-tail statistic that flipped three times:")
    ok, msg, red = unresolvable(
        [0.0078, 0.0019, -0.0047, 0.0007, 0.0029, -0.0108], "6l tail MCB")
    print(f"  {msg}")
    print(f"  ---> {red}")
    ok2, msg2, _ = unresolvable(
        [-0.00273, -0.00064, 0.00586, -0.01304, -0.00582, -0.02062, 0.00420,
         0.00141, -0.00584, 0.01082, -0.03343, -0.01270, -0.00936, -0.01193,
         -0.00843, 0.00714], "6o tail MCB (16 monthly windows)")
    print(f"  {msg2}")
    print(f"\n{len(alerts)} alerts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
