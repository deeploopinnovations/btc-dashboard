# Agent trajectory — external state

*Append-only. The ledger (`ledger.json`) records RESULTS; this file records the
TRAJECTORY that produced them: each inspect → plan → implement → evaluate cycle,
the execution feedback it generated, and what the next cycle did differently
because of it. A future session should be able to read this and know not only
what was concluded but which moves were tried, which failed, and why.*

*Conventions: each cycle names its ledger ids; "feedback" is what execution
returned that the plan did not anticipate; "lesson" is the change it caused,
with the rule or pitfall that now encodes it.*

---

## 2026-09-27 → 2026-09-28: from "no breakthrough" to the clock-aware anchor

**Starting state.** Owner feedback: long work, no breakthrough for NOCTUA.
Ledger ~190 entries; phase-3 lines on dispersion, slope and level repeatedly
ADVANCE-not-ADOPT. Supervisor at start: REPETITION on the calibration mechanism
(30 attempts), i.e. the trajectory was re-auditing, not building.

| # | cycle | inspect → plan | implement | evaluate (ledger) | feedback → lesson |
|---|---|---|---|---|---|
| 1 | zoo stack | teacher zoo only ever used as a scoreboard → combine it | convex QLIKE stack, shrink to NOCTUA | `P4-zoo-stack-v2-result` ADVANCE: +6.6% H=1 | first real forecast gain; but on sigma, not the product |
| 2 | stacked anchor into the product | swap the served anchor for the stack | `stack_barriers.py` | `P4-stack-anchor-result` REJECT: worse on 5/6, **mirror better on 4/6** | a gain at H=1 does not reach H=19; the mirror's win was the clue |
| 3 | online stack | refit weekly | `online_stack.py` with a rolling-level control | `P4-online-stack-result` REJECT | the control caught it: the gain was a rolling level |
| 4 | why did the mirror win? | calib-only diagnostic by anchor hour | `hod_diag`, `hod_lhc` | anchor over-forecasts 17:00 by ~11%; serving factor never reads 17:00 | **the breakthrough came from reading a failure, not from a new model** |
| 5 | clock-aware anchor | give the anchor the window's seasonal variance | `season.py`, `hour_anchor.py` (placebo, mirror, served factor) | `P4-hour-anchor-result` ADVANCE; reproduced bit for bit | first product test scored against the SERVED object on level |
| 6 | swarm audit #1 | one haiku agent, 5 attacks | — | `P4-hour-anchor-audit(-result)` | agent invented a CI transformation (rejected); I found the real weakness it missed: fold bootstrap cannot fail at 6/6 → **R88** |
| 7 | serve it (off) | artifact increment, runtime, gate | `add_hour_anchor.py`, `test_hour_anchor.py` | gate 12/12 | har_beta did not reproduce (features moved since training) → fit an INCREMENT, keep the shipped anchor byte-identical |
| 8 | CI failure | gate imported eval/ | AST check instead | CI green | **pitfall `serving-gate-imports`**, now in precommit |
| 9 | dispersion on the served base | re-test the old ADVANCE where it would ship | `served_dispersion.py` | `P4-served-dispersion-result` REJECT | the dispersion "win" was a width change compensating the 17:00 level defect |
| 10 | weekend residual | where is the other half of the bias? | calib-only profiles | `P4-weekend-residual` DIAGNOSTIC | year-specific, not a static shape; stopped instead of tuning on the same folds |
| 11 | network vs no network | does NOCTUA earn its place? | `product_score.py`, curves from `run_fold` | `P4-simple-vs-noctua-result`: earns it on shape (logs, pinball), not on ranking (DSC) | per-episode estimator with power now exists |
| 12 | per-episode confirmation | R88 asks for it | `hour_anchor_pe.py` | `P4-hour-anchor-pe-result`: all 4 barrier losses better, mirror worse on all 4 | — |
| 13 | forward holdout, executable | freeze 2026-09-27, N_MIN 300 | `forward_hour_anchor.py` | selftest 6/6 | count-only below N_MIN (no peeking by construction) |
| 14 | owner delegates the switch | swarm of 4 + a ship-blocker, rule fixed in AUDIT_STATE first | `hour_anchor_cond.py`, agents 1–4 | agents 1–3 no defect; agent 4 circular; **ship-blocker fired** | `P4-hour-anchor-cond-result` REJECT → switch OFF |
| 15 | was the gate fair? | forecaster's dilemma | `hour_anchor_exante.py` (no retraining, saved arrays) | `P4-hour-anchor-exante-result`: no ex-ante tail damage | block stands anyway; **R89** + pitfall `subset-not-outcome-selected` |
| 16 | my own bug | G_C absorbed by the median factor | fixed before reading the comparison | `P4-level-alternatives-result`: clock beats simpler level fixes on the tails | pitfall `arm-not-degenerate` |
| 17 | E2c, per episode, served base | the project's largest effect was never tested with an estimator that can fail | `iv_served_pe.py` (inputs rebuilt from committed DVOL) | `P4-iv-served-pe` — running | **operational failure:** my waiter polled `pgrep -f "model.eval.blend_ceiling"`, which matches the waiter's OWN command line; the loop never exited and the test sat unstarted for ~85 min. Fixed by waiting on the PID with `kill -0`. Lesson: never wait on a process by a pattern that appears in the waiter's own command |

## What the trajectory itself shows

- **Building beat auditing when the build was aimed at a measured defect.**
  Cycles 1–3 built models and found nothing that reached the product; cycle 4
  inspected WHY a failure failed and found the defect cycle 5 fixed.
- **Every agent verdict needed checking.** Of five agent audits in this period,
  two contained a fabricated or circular claim (cycles 6, 14/agent 4). The swarm
  is useful for coverage; its conclusions are inputs, not verdicts.
- **The most consequential error was mine and was in a gate, not in evidence**
  (cycle 14). Next time the ship-blocker gets the adversarial review BEFORE it
  runs.
- **Supervisor state at the end:** no OSCILLATION or UNRESOLVABLE in phase 4;
  REPETITION flags on stacking (closed), dispersion (closed) and the hour anchor
  (confirmations, now handed to the forward holdout — no further walk-forward
  mining of it).
