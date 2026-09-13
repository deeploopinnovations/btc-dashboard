# The research loop

An agent harness for NOCTUA's **development**, not a change to the model.

## The scoping claim, stated plainly

The five components below — persistent memory, an inspect→plan→implement→evaluate
loop, execution feedback, a stagnation supervisor, external statefulness — are
an architecture for **agents**. NOCTUA is a 6,939-parameter-per-seed MLP with
quantile heads; it has no inference-time loop to put them in, and attaching
them to it would be cargo cult.

What they are good for is the process that *builds* NOCTUA, and this project is
unusually strong evidence that the process needed them. In one working session
the following all happened, every one caught late and by hand:

| what went wrong | how it was caught |
|---|---|
| a Brownian benchmark corrected from 1.5958 to 1.5218 on a false premise, hiding a third of the gap | re-reading which column the extremes came from |
| `has_mx()` testing for a key that cannot exist — a guard that returned False for every artifact | exporting an artifact that had the head and watching the guard deny it |
| a relevance test implemented as `mean |Δ|`, which a random number generator passes | comparing it against the mean *signed* Δ |
| "92% of barrier error is shape" computed through an instrument that loses to the base rate by 6.55× | scoring the base rate, which nobody had done |
| a veto driven entirely by giving one arm 3× the calibration data | matching the arms |
| one question, three designs, verdicts 2/6 → 5/6 → 3/6 | tabulating them by hand |

Every one is a *process* failure, not a modelling failure. That is what this
harness is for.

## 1. Persistent memory — `ledger.json`, `ledger.py`

Append-only record of every experiment: question, **pre-registered rule**,
result, verdict, and what it supersedes. `ledger.add()` refuses an entry with no
rule, because a result whose rule was chosen afterwards is not admissible here.

Corrections are first-class: `--corrections` lists what turned out to be wrong
and what replaced it. In this project 3 of 18 entries are withdrawn or
superseded, and those three were quoted in prose for weeks after being wrong.

```
python -m model.research.ledger --list | --open | --corrections
```

## 2. The loop — inspect → plan → implement → evaluate

**Inspect.** Read the ledger before proposing anything. If the question already
has an entry, you are not proposing an experiment, you are proposing a *design
change* — and you must state what it measures that the previous design could
not. (§6n exists because §6m's arm was not deployable; that is a legitimate
answer. "Run it again and hope" is not.)

**Plan.** Write the decision rule **and commit it** before the data is scored.
The rule must name: the statistic, the threshold, the sample size, and what a
negative result looks like. If the rule contains a word like "moves", decide now
whether it means *changes* or *improves* — that exact ambiguity cost this
project a wrong verdict.

**Implement.** Build the measurement. Arms differ in the treatment and nothing
else; run `pitfalls.check_arms_matched` on the sizes.

**Evaluate.** Run `pitfalls` against your own outputs **before** writing the
result down. Then report the verdict the rule dictates — including when it is
not the one you want. §6o adopted on the tail branch while its QLIKE count fell
one short of the bar, and said so.

## 3. Execution feedback — `pitfalls.py`

Eight runnable checks, each derived from a specific error committed here, each
carrying the provenance of the experiment it would have saved. `--self-test`
asserts every check still fires on its historical case, so the checks cannot
rot into decoration.

```
python -m model.research.pitfalls --self-test
```

Append-only, like the ledger: a mistake that stops being checked comes back.

## 4. Supervisor — `supervisor.py`

Detects **oscillation** (verdicts flipping across designs of one question),
**repetition** (a topic attacked three or more times), **stale numbers**
(a withdrawn result still in circulation), and **unclosed loops**. Every alert
carries a redirection, because "you are stuck" is not information.

`unresolvable()` answers the question that would have stopped the 6l→6n loop
after the second attempt: *is the deciding statistic larger than its own
standard error at this n?* For §6l's tail deltas it is not — |mean| 0.00037
against se 0.00265 — and it prescribes the fix: raise n to ~315 units, or decide
on the mean with a CI. §6o did the latter and resolved it at n=16 (|mean|
0.00594, se 0.00282).

The supervisor's own first run found a defect in the supervisor: its regex topic
grouping missed the 6l/6m/6n/6o sequence, the exact case it was built for,
because those entries phrase their question as "Same, with…". A stagnation
detector that cannot see the stagnation it was written for is worse than none,
because it reports "no alerts" and is believed. Fixed with an explicit `topic`
field; the regex remains as a fallback.

## 5. External statefulness

Already true and worth naming: `model/artifacts/*.json` holds every
experiment's raw output, `BENCHMARK.md` holds the reasoning, `ledger.json` holds
the queryable state, and the git history holds the trajectory — including the
commits that were later corrected, which are deliberately not rewritten.

The one thing NOT externalised is the agent transcript. Sub-agent transcripts
live in the runner's scratch space and vanish with the container. That is a real
gap: three findings this session came back from agents whose reasoning is no
longer inspectable, and two of those findings had to be rejected on re-derivation.
The mitigation in force is that **no agent result enters BENCHMARK.md without
being re-derived from the data first** — which is why both rejections happened
before publication rather than after.

## The rule that outranks the rest

An agent's result is a *hypothesis* until the number is re-derived. In this
session that check rejected two of four agent verdicts — the shape agent's
"predictable and useful", and the anatomy agent's framing of a QLIKE artifact as
a directional bias. Neither was carelessness; both were plausible readings that
did not survive contact with the data.
