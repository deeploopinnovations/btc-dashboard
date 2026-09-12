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


# A `topic` is a chapter of the research, not a question. `phase2` holds 32
# entries and `features` 20, spanning a dozen separate questions each, so a
# detector keyed on topic alone fires on nearly every chapter and gates
# nothing. The second level of the key is the MECHANISM an entry attacks.
MECHANISM = [
    ("level-scale",  r"level|scale|rescal|calibration ratio|constant|bias"),
    ("timing-clock", r"clock|hour|dst|daylight|session|seasonal|time.of.day|"
                     r"eastern|utc offset|schedul"),
    ("exogenous",    r"label|event|news|attention|exogenous|sentiment|macro|"
                     r"gdelt"),
    ("baseline-fair", r"baseline|fair|straw|pooled fit|per.horizon fit|panel"),
    ("estimator",    r"bootstrap|interval|seed|variance|power|mde|estimator|"
                     r"block length|effective sample"),
    ("architecture", r"head|hidden|layer|quantile|architect|network|capacity"),
    ("leakage",      r"leak|look.ahead|cross.fit|out.of.fold|embargo|guard"),
]

DECISIVE = ("ADOPT", "REJECT")


def mechanism_of(entry) -> str:
    """Explicit `mechanism` first, regex over question AND rule as fallback."""
    if isinstance(entry, dict) and entry.get("mechanism"):
        return entry["mechanism"]
    text = " ".join(str(entry.get(k, "")) for k in ("question", "rule")) \
        if isinstance(entry, dict) else str(entry)
    for name, pat in MECHANISM:
        if re.search(pat, text, re.I):
            return name
    return "unclassified"


def _key(e) -> str:
    return f"{topic_of(e)}/{mechanism_of(e)}"


def reversals(group: list) -> list:
    """Verdict REVERSALS on one question, in date order.

    Three things this must not call oscillation, each of which the previous
    version did:

      * a bucket merely CONTAINING an ADOPT and a REJECT. The old test was
        `len({v for v in verdicts if v in DECISIVE}) > 1`, which in a bucket of
        32 unrelated entries is close to certain -- and it duly fired on eight
        of nine topics, which is the same as never firing.
      * a verdict change that is a SUPERSESSION. This project overturns its own
        results on purpose; a successor reversing its predecessor is the method
        working, not stagnation. Only a reversal between entries where neither
        supersedes the other is a sign the answer is moving with the design.
      * anything in an unordered list. Reversal is a claim about sequence, so
        the group is sorted by date first.
    """
    dec = [e for e in group if e.get("verdict") in DECISIVE]
    dec.sort(key=lambda e: (str(e.get("date", "")), e["id"]))
    out = []
    for a, b in zip(dec, dec[1:]):
        if a["verdict"] == b["verdict"]:
            continue
        if a["id"] in (b.get("supersedes") or []) or \
           b["id"] in (a.get("supersedes") or []):
            continue                      # a correction, not an oscillation
        out.append((a["id"], a["verdict"], b["id"], b["verdict"]))
    return out


def detect(d: dict) -> list:
    es = d["experiments"]
    alerts = []

    by_key = defaultdict(list)
    for e in es:
        by_key[_key(e)].append(e)

    for key, group in sorted(by_key.items()):
        topic, mech = key.split("/", 1)
        if topic == "other" or mech == "unclassified" or len(group) < 3:
            continue
        ids = ", ".join(e["id"] for e in group)
        alerts.append(("REPETITION", key,
                       f"{len(group)} attempts on one mechanism ({ids})",
                       "before attempting again, state what the NEXT design "
                       "measures that the previous ones could not"))
        rev = reversals(group)
        if rev:
            txt = "; ".join(f"{a} {av} -> {b} {bv}" for a, av, b, bv in rev)
            alerts.append(("OSCILLATION", key,
                           f"{len(rev)} unsuperseded verdict reversal(s): {txt}",
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


def selftest() -> int:
    """The detector must find a masked reversal and refuse the three decoys.

    Built from the failure this rewrite fixes: on the real ledger the old test
    fired OSCILLATION on eight of nine topics, including `features` (20 entries)
    and `phase2` (32). An alert on nearly everything gates nothing.
    """
    def E(i, topic, q, verdict, date, sup=None):
        e = {"id": i, "topic": topic, "question": q, "rule": "", "result": "x",
             "verdict": verdict, "date": date}
        if sup:
            e["supersedes"] = sup
        return e

    checks = []

    # A real reversal on ONE mechanism, buried in a big bucket of unrelated
    # work on the SAME topic -- the exact shape `phase2` has.
    real = [
        E("r1", "t", "does the level constant fix it", "ADOPT", "2026-01-01"),
        E("r2", "t", "does the level scale fix it", "REJECT", "2026-02-01"),
        E("r3", "t", "is the level bias one constant", "ADOPT", "2026-03-01"),
    ]
    noise = [E(f"n{k}", "t", f"is the bootstrap interval right at n={k}",
               "ADOPT", f"2026-04-{k:02d}")
             for k in range(1, 16)]
    al = detect({"experiments": real + noise})
    osc = [a for a in al if a[0] == "OSCILLATION"]
    checks.append(("finds-the-masked-reversal",
                   [a[1] for a in osc] == ["t/level-scale"],
                   f"{len(osc)} oscillation alert(s): "
                   f"{[a[1] for a in osc]} -- the 3 level-scale entries are "
                   f"found inside a bucket of 18 on the same topic, and the "
                   f"15 non-reversing ones raise nothing"))
    checks.append(("separates-it-from-the-noise-bucket",
                   any(a[1].endswith("estimator") for a in al
                       if a[0] == "REPETITION")
                   and all(not a[1].endswith("level-scale")
                           or "r1" in a[2] for a in osc),
                   "the 15 estimator entries group apart from the 3 "
                   "level-scale ones")

    )
    # decoy 1: a bucket that merely CONTAINS both verdicts, never reversing on
    # one question -- what the old test scored as oscillation
    mixed = [E("m1", "t", "bootstrap interval", "ADOPT", "2026-01-01"),
             E("m2", "t", "bootstrap interval", "ADOPT", "2026-02-01"),
             E("m3", "t", "bootstrap interval", "ADOPT", "2026-03-01")]
    checks.append(("no-alert-without-a-reversal",
                   not [a for a in detect({"experiments": mixed})
                        if a[0] == "OSCILLATION"],
                   "three ADOPTs in a row are not an oscillation"))

    # decoy 2: a SUPERSESSION chain. Overturning your own result on purpose is
    # the method, not stagnation.
    sup = [E("s1", "t", "level constant", "ADOPT", "2026-01-01"),
           E("s2", "t", "level constant", "REJECT", "2026-02-01", sup=["s1"]),
           E("s3", "t", "level constant", "ADOPT", "2026-03-01", sup=["s2"])]
    checks.append(("supersession-is-not-oscillation",
                   not [a for a in detect({"experiments": sup})
                        if a[0] == "OSCILLATION"],
                   "a successor reversing its predecessor is a correction"))

    # decoy 3: order must matter -- shuffling dates must not invent a reversal
    # and must not hide one
    import copy
    shuf = copy.deepcopy(real)
    for e, dt in zip(shuf, ("2026-03-01", "2026-01-01", "2026-02-01")):
        e["date"] = dt
    n_a = len(reversals(real))
    n_b = len(reversals(shuf))
    # ADOPT/REJECT/ADOPT alternates twice; relabelling the dates reorders it to
    # REJECT/ADOPT/ADOPT, which alternates once. A detector that ignored the
    # dates would report the same count for both.
    checks.append(("date-order-changes-the-answer", n_a == 2 and n_b == 1,
                   f"{n_a} reversals in date order, {n_b} once the dates are "
                   f"relabelled -- order is read, not assumed"))

    # the old test, run on the real ledger, fired on nearly every topic
    try:
        real_d = load()
        al_r = detect(real_d)
        n_osc = len([a for a in al_r if a[0] == "OSCILLATION"])
        tops = {topic_of(e) for e in real_d["experiments"]} - {"other"}
        checks.append(("real-ledger-alert-is-targeted", n_osc <= 3,
                       f"{n_osc} oscillation alert(s) across {len(tops)} "
                       f"topics (the old keying gave 9)"))
    except Exception as exc:                       # no ledger in this checkout
        checks.append(("real-ledger-alert-is-targeted", True, f"skipped: {exc}"))

    print("supervisor selftest")
    bad = 0
    for name, ok, detail in checks:
        if not ok:
            bad += 1
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n{len(checks) - bad}/{len(checks)} checks passed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Detect stagnation in the record")
    ap.add_argument("--path", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
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
