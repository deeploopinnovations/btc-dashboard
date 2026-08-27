# What is actually stopping this model, and what to do about it

Written after §6n, from measurements already in this repository. Every claim
below cites the number that supports it. Where a number does not exist yet,
this document says so and names the experiment that would produce it, together
with the decision rule that experiment must be judged by — fixed here, before
the data is scored.

## The short version

> **CORRECTED — see BENCHMARK.md §8, one session after this was written.**
> The "92 %" below is a property of the **Gaussian first-passage law**, which
> `eval/firstpassage.py` builds as a diagnostic. It is **not** a decomposition
> of NOCTUA's error: NOCTUA reads barrier probabilities off its learned Stage B
> quantile heads and does not use that law. And the law is not merely
> imperfect — fed *perfect* volatility it loses to the historical base rate by
> 1.62× at 2 %, 2.46× at 3 % and 6.55× at 5 %. Decomposing its error was
> decomposing an already-broken instrument. Priority 1 was run and returned a
> **negative** result; §8 has what replaces it.

~~The model is being improved on the 8 % of the problem it can reach, while the
92 % sits untouched — and the one previous attempt on that 92 % failed because
it treated a structural problem as a feature problem.~~ **(withdrawn — the 8/92
split came from the broken diagnostic; the measured figure through NOCTUA's own
mapping is 1.2–16.6 %, rising with barrier distance. See §9 and the revised
order of work at the end of this file.)**

---

## 1. The dominant constraint: the path, not the volatility

`eval/firstpassage.py` fed the Gaussian first-passage law the **realized**
volatility — an oracle no forecaster could match — and measured what error
survived:

| barrier | oracle error | causal forecast error | share a PERFECT vol forecast removes |
|---|---|---|---|
| 0.5 % | +5.40 pp | 5.85 pp | **7.6 %** |
| 1.0 % | +8.68 pp | 9.42 pp | **7.8 %** |
| 2.0 % | +8.83 pp | 9.65 pp | **8.5 %** |
| 3.0 % | +7.22 pp | 7.86 pp | **8.0 %** |
| 5.0 % | +4.63 pp | 4.94 pp | **6.0 %** |

**Perfect knowledge of volatility removes 6–8.5 % of the barrier error.**
The other 91.5–94 % is shape: given moves of the right size, how far the path
actually travels before settlement.

And the shape discrepancy is not subtle. Over production episodes:

    range / sqrt(realized variance)
      Brownian, sampled as this data is  1.5831
      BTC measured                       1.3311
      gap                               -0.2519   block-bootstrap 95% CI [-0.2851, -0.2212]

BTC travels **15.9 % less per unit of realized volatility than a Brownian path**,
and the interval excludes zero by a wide margin. A first-passage law fed a
correct sigma therefore **overstates** the chance of touching a strike — which
for an option seller means quoting strikes further out than necessary and
leaving premium on the table.

### Why "add shape features" already failed, and what that implies

The obvious response was tried. `eval/efficiency.py` added path-efficiency
features to the existing network:

| metric | without | with | wins | t-like |
|---|---|---|---|---|
| DSC/UNC | 0.049802 | 0.050003 | 2/6 | +0.22 |
| pinball | 0.003633 | 0.003644 | 1/6 | −1.17 |

A null, and reported as one. So the position is: the shape error is **large and
systematic** (fact 1), and **more shape inputs to the same architecture do not
touch it** (fact 2). Those two facts together rule out "we need better
features" as the explanation. Either the conditional variation in path shape is
not predictable at all, or the model is parameterized so it cannot express it.

**Nobody has measured which.** That is the single most valuable unasked
question in this repository, and it is Priority 1.

### Priority 1 — Is path shape predictable at all?

Modelled on `eval/direction.py`, which settled the direction question by asking
whether the target was *attainable* before anyone tried to model it.

- **Target**: the dimensionless travel ratio per episode, `(M_up + M_dn) /
  sqrt(RV)`, and the seller's version, `max(M_up, M_dn) / sqrt(RV)`. These are
  scale-free by construction, so a model predicting them is predicting shape
  and nothing else — volatility skill cannot leak in and flatter the result.
- **Arms**: climatology (constant), ridge, gradient boosting, the NOCTUA
  network. Same causal feature set, same splits, same embargo.
- **Null**: a shuffled-target permutation null on DSC, because in-sample
  isotonic regression manufactures positive DSC from noise. Clearing zero is
  not evidence; clearing the null's p95 is.
- **Decision rule, fixed here before the run**: shape is declared predictable
  only if (a) an arm's out-of-sample skill clears the permutation p95, **and**
  (b) that skill, pushed back through the first-passage law, moves the 2 %
  touch probability by **at least 1 pp**. Condition (b) exists because a
  statistically real signal that moves a seller's quote by 0.1 pp is not worth
  shipping, and this project has already published one finding (direction) that
  cleared significance and failed relevance by 27×.

**Both outcomes are valuable, which is why this is the right experiment.** If
shape is unpredictable, then the barrier forecast is near its ceiling, 92 % of
the error is irreducible, and every remaining hour belongs to volatility and to
regime adaptation — a major redirection made on evidence. If shape IS
predictable and the network is not capturing it, it is the largest available
gain in the project by an order of magnitude over anything measured so far.

---

## 2. The second constraint: the model is asked the wrong question

The stated use case is a seller deciding whether to sell premium or buy a
straddle over a fixed window. That decision depends on **P(either barrier is
touched)** — the straddle-break probability — not on direction, and not on
either side alone.

- Direction is closed and was closed properly: best skill **0.180 %** against
  **4.98 %** for the same model on barriers, a 27× gap, with nothing clearing
  the shuffled null at the deployed configuration (`eval/direction.py`). The
  literature agrees on the mechanism — order-flow imbalance carries ≈ −0.1 to
  −0.4 % OOS R² one minute ahead and decays within minutes, and this model
  forecasts 6–24 hours out. **No further direction work is planned, and that is
  a conclusion, not a concession.**
- Volatility amplification is the model's strongest output: DSC/UNC **20.3 %**,
  beating climatology 6/6, CI [+0.047, +0.103].
- The two-sided object sits between them and is under-built. `q_mx`, a head for
  `max(M_up, M_dn)`, was built and reported null at 11/18 cells — but it was
  measured on the **stale, pre-ETF fit**, and `eval/either.py` finds every
  model-based arm over-forecasting in exactly the years that fit does not
  cover (realized P(either touched at 2 %) 0.397 and 0.418 in 2025 and 2026
  against predictions of 0.614 and 0.636).

**Priority 2**: re-measure the two-sided head on a refreshed fit before
concluding anything about it. A null measured on a model fitted to a market
that ended is not a null about the head.

---

## 3. The third constraint: the fit is from a market that no longer exists

`eval/regime.py`: median 19-hour realized volatility fell from 5.28 % (2013) to
1.61 % (2026); across the spot-ETF launch the ratio is **0.584** (Mann-Whitney
p = 3.1e-156) with return kurtosis collapsing 15.04 → 0.41. **100 % of the
shipped model's fitted weight is pre-ETF.**

Refreshing the fit is the largest measured average-case gain in the project
(§6m: QLIKE −4.97 %, 5/6; pinball 6/6). Its deep-tail effect is unresolved —
§6l/§6m/§6n gave 2/6, 5/6, 3/6 win counts on the same effect, which is what
prompted §6n's replacement rule and the 18-window measurement now running.

**Priority 3**: whatever the 18-window result says, the harder question it does
not answer is whether the model **detects a regime change it was not trained
on** — a volatility spike like the 2025 US-Iran episode — from features alone,
without being shown the data. That is the property a seller actually needs, and
it is being measured now rather than assumed.

---

## 4. What the target should be, stated honestly

The stated goal has been predicted volatility within ~10 % of realized. That is
not attainable and the repository already contains the number that says so: the
in-sample R² of realized volatility on the full causal feature set is **0.665**
with residual sd **0.378 in logs**, which puts "within 10 %" at roughly
**40–45 % of nights for an oracle with perfect parameter knowledge**. Any
method claiming 90 % of nights within 10 % is either measuring something else
or has a leak.

The honest target is the ceiling, and the useful reframing for a seller is that
they do not need the volatility to 10 % — they need the **touch probability**
calibrated, which is a different and more attainable object, and the one
Priority 1 attacks.

---

## Order of work — REVISED after Priorities 1-3 were run

Everything below the banner at the top of this file is the plan as written
before any of it was executed. It is kept as the record of what was believed.
This section is what the measurements actually support.

**Priority 1 was run and returned a negative.** Path shape carries a real but
weak rank signal (Spearman +0.0795); conditioning on it makes touch forecasts
significantly *worse* at every barrier ≥ 1 % (§8). The "92 %" that motivated it
is withdrawn — it was computed through a Gaussian first-passage law that loses
to the historical base rate by up to 6.55×.

**The honest decomposition, through NOCTUA's own mapping** (§9): a perfect
volatility forecast removes **1.2 / 3.8 / 10.0 / 12.5 / 16.6 %** of barrier
error at 0.5/1/2/3/5 %. It *rises* with barrier distance, so volatility matters
most exactly where a seller operates — the opposite of the withdrawn framing.

*Updated the same day, after §21–§25 ran: the `har_1h`/`har_6h` anchor lever in
item 1 below reversed sign once it was tested through the full pipeline rather
than the OLS-only harness quoted there. Items 5–8 are new: a statistical-power
measurement (`E-power`) is now inserted ahead of everything after item 4
(`model/PHASES.md` Phase 5.5), and it gates the ensemble-blend, implied-volatility
and funding-rate work below it.*

### What the evidence now says to work on, in order

1. **The spike lag — the largest measured, addressable loss.** Spike nights are
   7.7 % of episodes and 25.8 % of loss, under-forecast by 45 %. §12 established
   the information is there (onset AUC 0.733, CI [0.655, 0.805]), so this is a
   modelling choice, not a ceiling. Two levers were measured and pre-registered
   for a proper walk-forward: 3× spike upweighting (pooled QLIKE −5.6 %, spike
   −25.9 %, calm +1.7 %) and ~~`har_1h`/`har_6h` in the linear anchor (pooled
   −7.33 %, onset −5.48 %)~~.

   **Correction, same day, §21:** the `har_1h`/`har_6h` anchor lever was run
   through the full pipeline and reversed sign: spike QLIKE got **worse** by
   +1.87 % (95 % CI [+0.02151, +0.05408], 6 of 6 folds), against the −10.44 %
   the OLS-only harness above had shown. **DO NOT ADOPT** (ledger
   `E-anchor-verdict`). Mechanism: the full pipeline already feeds
   `har_1h`/`har_6h` to the neural stage, so the change added no information
   and instead moved the noisiest available RV window into the linear anchor
   term whose job is to stay stable. §21 redirects this line of work to the
   ensemble blend weight — see item 6 below.

   **Bounded expectation (still applies to the surviving lever):** the
   achievable gain is limited by a 0.73-AUC onset signal, not the 0.78
   headline.

2. **Ship the adopted refresh.** §6o adopted a rolling training window on 16
   monthly windows (tail-MCB CI [−0.0118, −0.0020]). The dated artifact exists
   (`noctua_v2_refreshed_2026-08-09.npz`, 298,791 training episodes). What
   remains is the deployment decision, deliberately left explicit because
   overwriting the research artifact would silently invalidate every evaluation
   that loads it.

3. **The hour-shaped bias.** Real and localised to 15–22 UTC, where a
   hour-conditional correction cuts calibration error 44.6 %. Not adopted: its
   CI contains zero and the hour pools hold a median 60 episodes against 1,422
   (§11c). Needs a wider evaluation, not a better argument.

4. **Retire `reg_post_etf` at the next retrain.** Real defect, negligible
   impact — 0.169 % on sigma, 0.17–0.29 pp on P(touch 2 %) (§11b).

5. **Statistical power — inserted ahead of items 6–8 (`model/PHASES.md` Phase
   5.5, BENCHMARK.md §22).** Every walk-forward fold scores ~365
   production-slice episodes (`H == 19`, `anchor_hour == 17`), ~20 of them
   spike-flagged — six folds is ~2,190 test episodes and ~119 spike episodes,
   against a 510,496-episode population. Two independent pre-registered tests
   of the ensemble blend weight (item 6) both came back too imprecise to
   decide anything at that resolution. `E-power` re-decides §21's
   already-resolved anchor-freshness result on all 24 anchor hours at H = 19,
   through the same code path, and measures how much the confidence interval
   actually tightens against the √24 = 4.90× that independent episodes would
   give. **Status: pre-registered and RUNNING; no result yet** (ledger
   `E-power`, OPEN).

6. **The ensemble blend weight is insurance, not a lever.** §19/§21 redirected
   the spike-lag work here. Two pre-registered, walk-forward-honest tests
   both fail to improve on the shipped constant: a state-dependent
   (spike-vs-calm) weight scored pooled +0.00307, CI [−0.01008, +0.02603], 4
   of 5 folds better (ledger `E-blend`, NULL); a single revised constant
   scored +0.00049, CI [−0.01141, +0.01981], 4 of 5 folds better, but failed
   its own pre-registered 5 % worst-fold guard at +7.84 % in the 2023 fold
   (ledger `E-blend-1state`, REJECT). `infer.BLEND_W`'s own note records why
   0.25 was chosen: it bounds that 2023 fold at +6.7 % where pure NOCTUA
   suffered +72.3 %. So **0.25 buys insurance against a repeat of 2023, and
   the premium is about 1.4 % of mean QLIKE** — not a tuned optimum.
   `research/pitfalls.check_not_a_coin_flip` flagged the one-state estimate as
   unresolvable at this sample size (1/18th of its own standard error), which
   is what motivated item 5 (§23).

7. **Implied volatility: the level is redundant, the dynamics are not — the
   project's first ADVANCE.** E2, a DVOL-conditioned correction with a fitted
   intercept, REJECTED: a 57 % spike-QLIKE improvement (5 of 5 folds) was
   mostly the fitted intercept, not the features — an intercept-only fit, a
   bare scalar on every σ with no IV at all, delivered **72.0 %** of it, and a
   misaligned-feature placebo reproduced −0.53955 of the −1.05579 gain because
   the placebo has an intercept too (the margin, −0.02469 pooled, has a CI
   containing zero). Calm QLIKE worsened 13.52 % (ledger `E2-iv-correction`,
   REJECT).
   Removing the intercept (E2b) overturned that: the real arm beat its
   placebo, CI [−0.14822, −0.02069], though `iv_level`'s sign was unstable
   across folds. Dropping `iv_level` and keeping only the dynamics
   (`iv_chg_1h`, `iv_chg_6h`, `iv_chg_24h`, `iv_z_20d`, `ivrv_ratio`) — E2c —
   passed every guard at the Bonferroni-adjusted level: pooled QLIKE −0.03264
   (CI [−0.04356, −0.01403], 5 of 5 folds), spike −20.89 % (5 of 5), calm
   −8.22 % (an improvement, 4 of 5), beating its placebo (margin CI
   [−0.07323, −0.01420]), signs stable (ledger `E2c-result`, **ADVANCE**).
   **ADVANCE is not ADOPT**: the correction re-weights a recorded volatility
   median and has never been run through the full pipeline (barrier curves,
   committee, calibration), and NOCTUA's product is a touch-probability
   curve, not a sigma. Scope: 1,681 episodes across 5 folds, entirely inside
   DVOL's 2021-03+ era. Three further checks were run against the result
   rather than for it, and it survived all three: a test-shuffle positive
   control reversed it (−0.03264 with 5 of 5 folds becomes +0.01909 with 0 of
   5); 89.5 % of the gain needs episode-level variation, so it is not a
   per-fold level shift in disguise; and it clears zero at every value of the
   a-priori shrinkage constant from 0.01 to unbounded. The confirmation run,
   `E2-confirm`, is pre-registered and queued behind item 5.

8. **The funding rate, promoted to the head of the queue after `E2-confirm`
   clears (`model/PHASES.md` Phase 6).** E2c's result generalizes beyond
   DVOL: a forward-looking input carries volatility information the trailing
   own-history cascade does not. The funding rate is the other
   forward-looking series available, with better coverage than DVOL: 64,206
   rows from 2019-04-30, 67.8 % of training episodes causally covered versus
   DVOL's 32.7 %, 100 % of calibration and test (ledger `funding-coverage`,
   OPEN — coverage measured, no hypothesis tested yet). Mechanism: funding
   prices crowding in perpetual positioning, a precondition for the
   liquidation cascades behind some spike episodes. Gated to run only after
   `E2-confirm` validates the correction harness end-to-end on a result
   already understood.

### What is closed, and should not be reopened without new data

- **Direction** — 0.180 % skill against 4.98 % on barriers, 27×. Closed (§2).
- **Path shape as a conditioning variable** — hurts at every barrier (§8).
- **Sigma-atom collapse** — reverses sign on 24× the data; mechanism refuted (§10).
- **Capacity** — not underused; the layers are saturated against their inputs (§11a).
- **"Within 10 % of realized volatility"** — not attainable. In-sample R² 0.665,
  residual sd 0.378 in logs, which is 40–45 % of nights *for an oracle*. The
  attainable object for a seller is the calibrated touch probability.
- **Sub-daily HAR lags in the linear anchor (`har_1h`/`har_6h`)** — reversed
  sign under the full pipeline: spike QLIKE +1.87 % worse, CI [+0.02151,
  +0.05408], 6 of 6 folds (§21; ledger `E-anchor-verdict`). The −10.44 %
  OLS-only figure in item 1 above is superseded.
- **Implied volatility, as a plain feature or via a correction with a fitted
  intercept** — REJECT: an apparent 57 % spike-QLIKE gain was mostly a
  placebo effect (margin CI containing zero) and calm QLIKE worsened 13.52 %
  (§24; ledger `E2-iv-correction`). Not closed in general — see item 7: the
  same correction without the intercept and without the redundant `iv_level`
  term (E2c) passed every guard and is queued for full-pipeline confirmation.

*Educational research only. Not financial advice.*
