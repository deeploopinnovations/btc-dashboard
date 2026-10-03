# NOCTUA — the next development plan (2026-09-30)

## Where NOCTUA stands, from the ledger rather than from memory

* **What ships:** NOCTUA-v2 blended with a Log-HAR anchor (`blend_w` 0.25),
  a trailing 60-day median factor (now actually 60 days —
  `P4-factor-window-result`), the conditional mean as the reported sigma, and
  every anchor increment OFF.
* **Is the network worth its complexity?** Yes, for a stated reason:
  `P4-simple-vs-noctua-result` — better log score and pinball than a
  clock-aware Log-HAR with a Gaussian first-passage law at 99.5 %, a TIE on
  Brier and on which nights are riskier (DSC). NOCTUA earns its place on the
  **shape** of the barrier distribution, not on ranking nights. If that ever
  stops holding forward, the simple model ships.
* **What could improve it, and is waiting on forward data:** six frozen
  holdouts (clock-aware, weekend, day-of-week, clock + day-of-week, next-day
  implied vol; the dispersion candidate before them), scored once each at
  N_MIN by `forward-holdouts.yml`. Earliest decision: the implied-vol holdout
  at 200 nights (~mid-April 2027); the calendar ones at 300–450 (mid-2027 to
  end-2027). **The binding constraint on the research is calendar time.**
* **What the supervisor says:** phase-4 calibration, clock and implied-vol
  lines are at 8–14 attempts. More walk-forward work on them is repetition;
  they move to their holdouts. New effort goes to things those attempts could
  not measure.

## The plan, in phases — one change per phase, each gated before the next

| phase | what | why this, now | done when |
|---|---|---|---|
| 0 ✅ | Observability and serving parity: `NOCTUA_DEBUG`, the pipeline trace, the 60-day window restored | a wrong forecast does not crash; the trace found a research/serving mismatch on its first live run | gates 20/20, trace byte-neutral, `P4-factor-window-result` |
| 1 | **The dashboard shows one model by design** (finding F4) | the user-facing panel is NOCTUA ~15 % of the time and a third-party directional demo the rest | owner picks: wider window for this snapshot / drop the demo fallback / reliable cron trigger (queued as a task) |
| 2 | **Freeze the configuration that would actually ship**: day-of-week calendar + implied vol with the IV increment refitted on the calendar anchor's residual | `P4-iv-on-calendar` found the two complementary (+0.71 % Brier beyond calendar + intercept), serving refuses to combine them, and no holdout scores the combination. What it measures that nothing else does: the forward value of the joint served configuration, with the placebo contrast (R90) | built OFF, gate (off bit-identical, on = algebra, combination explicit), forward scorer with N_MIN derived from the saved arrays, frozen in DATA_USE.md, in the daily workflow |
| 3 | **Keep the holdouts honest while they accrue** | a holdout nobody audits can rot (audit L found one) | monthly: frozen hashes, count-only output, workflow green; lock files only by the workflow |
| 4 | **New information, aimed at shape**: the next-day option smile's WINGS (butterfly / risk-neutral tail mass) against the tails of the night's excursions | 92 % of barrier error is path and shape, not volatility (`BENCHMARK.md` §6f); NOCTUA's edge is shape; the harvested trades carry strikes; skew → asymmetry was NULL, wings → tail thickness is untested | registered before any wings-vs-outcome look; screen on the Gaussian law with a shuffled-wings placebo; R90 |

**Considered and not started: reinforcement learning.** The forecast is a
supervised problem with full next-morning labels and proper scoring rules; RL
would learn from less information than it already has. RL (or a contextual
bandit over strike choice) belongs to the TRADING decision layer, and needs a
reward built from historical option quotes, spreads and fills, which this
project does not have; the standing rule forbids manufacturing that P&L. It
becomes phase 5 if a quote/order-book archive (e.g. Deribit history via a data
vendor) is obtained: registered first, against a fixed-threshold selling rule
as the baseline.

**Phase 5 started 2026-10-03, on spot BTC, PAPER ONLY:** `P5-rl-paper-result`
— no learner beat holding out of sample; the agent runs anyway on GitHub
Actions (`paper-agent.yml`) to collect forward evidence and a loss log in the
private dataset `msdgaming2222/noctua-rl-paper`. A learner earns a claim of
skill only by clearing the volatility-target rule at 99.5 % in that forward log.

## How each phase is run (the discipline, written down so it survives the session)

1. **Brainstorm before touching code.** Write two or three ways to do it and
   pick by evidence; say what each would measure. (Phase 2 of this cycle chose
   "restore 60" over "adopt the served window" only after measuring both.)
2. **Know why the code is there before changing it.** `git log -S`, the
   ledger entry that introduced it, the R-rules, the pitfalls. The 400-day
   bundle was right for what it was built for; it was the later consumer that
   was never checked.
3. **One change per phase.** Findings outside the phase are recorded or
   queued, not fixed on the way (F4 was queued; F7/F8 were left).
4. **Test first, and prove the test can fail.** Red for the right reason,
   then green; then break the implementation on purpose and see the gate name
   the break.
5. **Verify on the path it will run on.** The trace passed offline and
   crashed live; only the live run found it.
6. **Debug with the trace, not with guesses.** `NOCTUA_DEBUG=1`, find the
   first stage whose output is wrong, form one hypothesis, test it minimally,
   then fix. Unset the variable to go back to normal.
7. **Audit to disprove.** Agents attack each claim; every agent claim is
   re-run before it counts — right verdicts with invalid evidence have
   happened, and so have wrong "no defect" reports hiding a real defect.
