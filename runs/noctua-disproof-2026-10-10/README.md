# NOCTUA fresh test and disproof audit, 2026-10-10

Start with [VERDICT.md](VERDICT.md). To resume the loop, read [MEMORY.md](MEMORY.md).

| path | what |
|---|---|
| `VERDICT.md` | answers: how good, fresh test, production readiness |
| `MEMORY.md` | loop memory: rules, lessons from mistakes, claim status |
| `trajectory.jsonl` | every command and its outcome, in order |
| `plans/` | the plan written before each iteration |
| `evals/` | raw numbers (`fresh_test.json`, `iter2_encompassing.json`) |
| `audits/` | each auditor's verdict verbatim, plus their scratch code |
| `supervisor.json` | stagnation detector state |
| `loop.py` | trajectory logger and supervisor |
| `fresh_test.py` | the test (`python runs/noctua-disproof-2026-10-10/fresh_test.py`, ~11 min) |
