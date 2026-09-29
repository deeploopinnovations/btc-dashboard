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
| 17 | E2c, per episode, served base | the project's largest effect was never tested with an estimator that can fail | `iv_served_pe.py` (inputs rebuilt from committed DVOL) | `P4-iv-served-pe-result` REJECT: nothing separates on the served base (QLIKE +0.98% ns); large same-signed gains on HOT/HIGH nights, not significant | **operational failure:** my waiter polled `pgrep -f "model.eval.blend_ceiling"`, which matches the waiter's OWN command line; the loop never exited and the test sat unstarted for ~85 min. Fixed by waiting on the PID with `kill -0`. Lesson: never wait on a process by a pattern that appears in the waiter's own command |

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
- **The factor-free benchmark overstated a level-type effect three times in
  two days** (dispersion, the stacked anchor's mirror, E2c). The single most
  consequential methodological change of the period is that product tests now
  score against the served base -- serving's trailing factor on every arm.

## 2026-09-29: persistence, then the ranking gap

| # | cycle | inspect → plan | implement | evaluate (ledger) | feedback → lesson |
|---|---|---|---|---|---|
| 18 | persist state | a restart once wiped `model/artifacts/` | whitelist every result JSON + the per-episode arrays in `.gitignore`, commit | 36 JSONs + 3 npz tracked (`ce6f304`) | the 150 MB raw 1-min source cannot go to GitHub (100 MB limit) -- still a single point of failure, flagged to the owner |
| 19 | the ranking gap | audits: NOCTUA does not rank nights better than a simple model; har_short's reactivity was clock-biased at 17:00 | `short_anchor.py`: deseasonalised 1h/6h vol in the clock-aware anchor, raw control, no-network screen | `P4-short-anchor-result` REJECT (screen): pinball only; **DSC 6/6 folds, t misses by 0.00005** | the pre-registered consequence of a failed screen was "the line closes" -- honoured; the DSC pattern kept as a hypothesis for a future freeze, not claimed |
| 20 | reactive-anchor audit | 3 agents to disprove the DSC pattern | agents A-C | `P4-short-anchor-audit`: A clean, B not disconfirmed, C selection/dependence accepted | **not frozen** -- likely real, but no per-episode score could resolve it forward; a frozen candidate that can never be scored is a promise, not a test |
| 21 | large-file storage | owner: store on HF public | public dataset repo created; HF Job to rebuild + upload server-side | **Job refused: 402 Payment Required** (Jobs need Pro/credits) | fallback built: `noctua/hf_store.py` verifies the committed hash (local corpus MATCHES) and uploads from the container once a write token is in the environment as `HF_TOKEN`; the connector cannot carry a 150 MB binary inline |
| 21b | storage, second blocker | re-check before uploading | `hf_store.reachable()` | the egress proxy answers **403 to huggingface.co** (organisation network policy); `HF_TOKEN` still absent | not retried (policy denials are reported, not retried); the store now refuses with the remedy instead of a stack trace. Two owner actions needed: allow huggingface.co, add `HF_TOKEN` |
| 22 | weekend factor → weekend BUG | build a trailing weekend-split factor for `P4-weekend-residual`'s gap; while choosing its weekend definition, checked the shipped column against pandas' calendar | train-only diagnostic, `test_weekend_column.py`, `weekend_fix.py` | `P4-weekend-bug` DIAGNOSTIC: `cal_weekend_frac` counts **Fri+Sat** (epoch offset +4, should be +3); true Sat+Sun coefficient -0.09..-0.29 on train beside it. `P4-weekend-fix` pre-registered (increment + Tue/Wed placebo + full-retrain secondary) | **the defect sat in the anchor since phase 1 and three analyses built on it** (`P4-weekend-residual`, the cond subsets, `vol_matrix`'s self-check). Lesson: verify a calendar feature against an INDEPENDENT calendar, never against a copy of its own formula -- now pinned in CI. The trailing-factor design was dropped: it would have patched a symptom of a mislabelled input |
