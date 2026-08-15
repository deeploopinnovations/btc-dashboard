# NOCTUA-BENCH — results

Walk-forward, 6 expanding folds (2021–2026), embargoed splits, production
slice only (19 h window opened 17:00 UTC, one non-overlapping episode per day).
Every competitor is refit inside each fold. Reproduce with:

```
python -m model.eval.benchmark      # proper scores + CORP + Christoffersen
python -m model.eval.falsify        # is it reasoning or pattern-matching?
python -m model.eval.synthetic      # instruments with a known right answer
python -m model.eval.regimes        # conditional calibration, real BTC, OOS
```

---

## 0. The benchmark caught its own author first

The metric this project had been quoting — **mean |coverage error|** — is
cheatable, and here is the proof, from the benchmark's own table:

| model | mean coverage error, pp |
|---|---|
| **scaled_clim** (unconditional shape × trailing vol) | **1.360** ← *wins* |
| noctua_v2 | 1.629 |
| climatology (constant, input-blind) | 2.493 |
| noctua_shuffled (features scrambled) | 2.969 |
| log_har_gauss | 3.332 |
| persistence | 3.855 |

A near-trivial baseline **beats the shipped model**, and a constant that
ignores every input beats both real econometric competitors. Marginal coverage
is what an unconditional distribution is *fitted to*, so scoring it rewards
having fitted an intercept. The 1.629 pp figure quoted throughout `RESULTS_V2.md`
should be read as a calibration diagnostic and never as evidence of skill.

Everything below is a strictly proper score or a falsification test.

---

## 1. Strictly proper scores

Lower is better. A proper score charges jointly for calibration **and**
sharpness, so calibration alone cannot win one.

| model | pinball | CRPS | Brier | log score |
|---|---|---|---|---|
| **noctua_v2** | **0.003633** | **0.005307** | 0.180584 | **0.529089** |
| log_har_gauss | 0.003785 | 0.005344 | **0.180476** | 0.533638 |
| scaled_clim | 0.003803 | 0.005498 | 0.182808 | 0.538220 |
| persistence | 0.003945 | 0.005570 | 0.183798 | 0.547539 |
| climatology | 0.004109 | 0.005910 | 0.189167 | 0.555597 |
| noctua_shuffled | 0.004269 | 0.006099 | 0.191451 | 0.563156 |

NOCTUA wins pinball, CRPS and log score — the three that grade the **whole
predictive distribution**. On Brier over the fixed binary barrier events it is
level with `log_har_gauss` (0.180584 vs 0.180476, a 0.06 % gap).

Volatility QLIKE: **noctua 0.2998**, log_har 0.3145, persistence 0.4474.

---

## 2. CORP decomposition — the part that cannot be faked

`S = MCB − DSC + UNC` via isotonic recalibration
(Dimitriadis, Gneiting & Jordan, *PNAS* 2021). **DSC is 0 by construction for
any constant forecaster**, so it cannot be reached without genuine conditional
information.

| model | DSC (higher better) | MCB (lower better) |
|---|---|---|
| log_har_gauss | **0.008707** | 0.024737 |
| scaled_clim | 0.008362 | 0.026725 |
| **noctua_v2** | 0.008178 | **0.024317** |
| persistence | 0.007307 | 0.026659 |
| noctua_shuffled | 0.000913 | 0.027919 |
| **climatology** | **0.000000** | 0.024721 |

Two things to read here, and the second is not flattering.

**The benchmark is sound.** Climatology scores DSC = 0.000000 exactly, and
`noctua_shuffled` — the real model reading a *random other episode's* features,
identical marginals, destroyed alignment — collapses to 0.000913. Skill shows
up in this column or it does not exist.

**NOCTUA does not win it.** Its discrimination is marginally *below*
`log_har_gauss` (0.008178 vs 0.008707, ~6 % of the DSC level). NOCTUA instead
posts the **lowest miscalibration of any competitor**. So on binary barrier
events the committee is buying calibration at a small cost in sharpness, and
against a plain Log-HAR + Gaussian first-passage baseline it is **effectively
tied, not ahead**. With 6 folds I did not establish that the 6 % gap is
distinguishable from noise in either direction, and it should not be quoted as
if it were.

The honest summary: NOCTUA's advantage is real on the **full distribution**
(pinball, CRPS, log score) and on **volatility** (QLIKE), and is **absent on
binary barrier discrimination**.

---

## 3. Conditional coverage (Christoffersen 1998), α = 5 %

| side | hit rate | p(uncond.) | p(independence) | p(joint) |
|---|---|---|---|---|
| up | 4.90 % | 0.256 | 0.477 | 0.311 |
| down | 7.04 % | 0.360 | 0.721 | 0.332 |

No rejection anywhere. Independence matters most: breaches do **not cluster**,
so the model is genuinely adapting to state rather than carrying a fixed level
through calm and violent weeks alike. (Contrast the synthetic control in
`eval/benchmark.py`, where clustered breaches sail through unconditional
coverage at p = 0.72 and independence rejects at p = 0.0000.)

---

## 4. Is it reasoning, or pattern-matching?

All checks in `eval/falsify.py` pass.

- **Monotone response.** σ is *strictly* increasing in trailing volatility
  across a 13-point sweep, elasticity **d log σ / d log vol = 0.889** — near
  proportional with mild shrinkage toward the mean, which is the economically
  correct behaviour. A model that had learned a correlation rather than the
  mechanism would not be forced to get this right.
- **Sharpness.** The α = 5 % level has CV **0.197** and a **2.24×** spread
  across episodes. A lookup table sits at 0 and 1.00×.
- **Feature use.** 37 of 39 features move the forecast; only `reg_post_etf`
  and `cal_H` are inert. `har_1d` dominates (22.1 % σ shift), with
  `cal_weekend_frac` (4.3 %) and `cal_dow_sin` (3.7 %) next — calendar
  structure outranks the longer HAR components.
- **Off-distribution.** At 20×, 400× and 1/400× training volatility the output
  stays finite, monotone, positive and in range. It does not produce confident
  nonsense where it has no data.

## 5. Instruments with a known right answer

`eval/synthetic.py` feeds processes whose true barrier law is exact (driftless
GBM: `P(max ≥ u) = 2Φ(−u/σ)`) or Monte-Carlo-precise (GARCH(1,1), Merton jump
diffusion). NOCTUA has only ever seen Bitcoin.

| instrument | true σ | model σ | ratio |
|---|---|---|---|
| GBM 1 % | 1.000 % | 1.049 % | 1.049 |
| GBM 3 % | 3.000 % | 2.867 % | 0.956 |
| GBM 9 % | 9.000 % | 8.109 % | 0.901 |
| GARCH(1,1) | 3.000 % | 2.837 % | 0.946 |
| jump diffusion | 2.317 % | 2.171 % | 0.937 |

**Across a 9× volatility range it has never seen, σ is recovered to a 1.16×
spread.** It is measuring volatility from the series, not recalling Bitcoin's.

Touch probabilities run *conservative* — model/true ratios 1.00–1.24 at
σ ≥ 3 % — which is the safe direction. The exception is genuine: at σ = 1 %
(below anything in Bitcoin's history) it puts 12.8 % on a 2 % barrier where
Gaussian truth is 4.6 %. It imposes a crypto-like fat tail on a thin-tailed
instrument. Correct for its actual deployment, wrong in general, and worth
knowing before pointing it at anything but BTC.

---

## 6. What this found, and the fix

Marginal calibration was concealing a real defect. Out of sample from
2024-07, realized vol lands **below** the forecast **66.4 %** of the time,
median ratio **0.874**. Every barrier scales with σ, so strikes were quoted too
far out — safe, but forfeiting premium every night, with breach rates near half
of nominal.

It is **not a fixed constant**:

| split | median RV/σ | fraction below |
|---|---|---|
| train | 0.970 | 55.0 % |
| calib | 1.009 | 48.6 % ← essentially unbiased |
| test | 0.874 | 66.4 % ← biased |

A correction fitted on calib would be 1.01 and do nothing. One fitted on test
would be look-ahead — fitting the evaluation data, precisely the cheat this
benchmark exists to catch. The bias is a **regime property**, so the correction
must move with the regime.

`serve/adaptive.py` estimates it from episodes that have already **settled**
strictly before the anchor — causal by construction, median rather than mean
because the ratio is right-skewed, clipped to [0.70, 1.40] so a data fault
cannot masquerade as a regime shift.

| | before | after |
|---|---|---|
| median RV/σ (test era) | 0.874 | **0.990** |
| fraction below | 66.4 % | **51.5 %** |
| barrier calibration | 2.073 pp | **1.373 pp** |

On the calibration split the factor comes out at **1.011** — it correctly does
nothing where the model was already unbiased. That self-cancelling behaviour is
the point: this is insurance against regime change, not a tuning knob. Windows
of 30/60/90/180 days all land within 0.98–0.99, so the horizon is not fitted to
the answer. Lag-1 autocorrelation of the episode-level ratio is −0.021 —
individual nights are unpredictable and no attempt is made to predict them;
only the drifting *level* is tracked.

---

## 7. Where this leaves the model

**Established.** It carries genuine conditional information (DSC ≫ 0, and
shuffling the features destroys it). It beats every baseline on the full
predictive distribution and on volatility. It responds correctly and
monotonically to its causal driver. It does not break off-distribution. It
recovers σ on processes it has never seen. Breaches do not cluster.

**Not established.** That it beats a competent classical baseline on *binary
barrier discrimination* — there it is tied with Log-HAR + Gaussian, and by DSC
marginally behind. The committee's contribution is calibration and distribution
shape, not sharper event prediction.

**Measured, but only as compute.** The Kronos head-to-head has not produced
accuracy numbers, but one hard figure did come out of trying. Running
Kronos-small on a CPU runner over this project's own production episodes --
512-hour context, 19-hour horizon, 32 sampled paths -- costs

    117.4 seconds per episode

and it is remarkably steady: 117.36 s at episode 1, 117.37 s at episode 161,
across five and a half hours. NOCTUA answers the same question in 0.6 ms.
That is a factor of roughly **196,000**, against a parameter ratio of 1,300
(24.7 M vs 19,134), because Kronos must autoregressively generate 32 paths
while NOCTUA evaluates four matmuls and a cumulative sum.

None of that says which model is more ACCURATE, and it must not be read that
way. It does bound what is deployable on a free 2-vCPU Space: at 117 s per
forecast with a 30-minute refresh, Kronos-small would spend 6.5% of its life
computing one number.

**Measured at last.** Kronos-small, 120 production episodes, 32 independent
sampled paths each, same anchors, same barriers, graded by one implementation
of one set of rules.

| model | Brier | log score | DSC ↑ | MCB ↓ |
|---|---|---|---|---|
| **noctua_v2** | **0.146371** | **0.443084** | **0.018334** | **0.023980** |
| kronos-small | 0.214208 | 0.661172 | 0.006580 | 0.080062 |
| climatology | *0.140725* | *0.431316* | 0.000000 | 0.000000 |

NOCTUA wins Brier, log score and discrimination. Paired episode-level bootstrap
(2,000 resamples over the 120 episodes, which preserves the within-episode
dependence between barriers and sides):

    Brier advantage to NOCTUA   +0.0504 .. +0.0868    P(better) = 1.000
    DSC   advantage to NOCTUA   +0.0055 .. +0.0202    P(better) = 1.000

Both intervals exclude zero. Against a model **1,291× larger** (24.7 M
parameters against 19,134), on the task this repo was built for.

Four things must be said alongside that, because none of them flatters us.

**Climatology's Brier and log score are the lowest in the table and are
IN-SAMPLE.** Its base rate is estimated from the evaluation set itself, so those
two columns are not a fair comparison and neither model "loses" to it. Only
`DSC = 0.000000` is a valid control — and it is the one that matters, since it
confirms the harness cannot manufacture discrimination from nothing.

**Kronos is genuinely better calibrated on the volatility LEVEL.** Median
predicted/realized: Kronos **1.172**, NOCTUA **1.232**. Both over-forecast; the
24.7 M-parameter general-purpose model is closer to unbiased than the
purpose-built one. That is a real result and it is not softened here.

**Kronos carries real conditional information.** DSC 0.006580 is not zero and
not close to it — it is 36% of NOCTUA's, from a model that has never been told
what a barrier is. At the 1% barrier it is nearly level (0.018317 vs 0.020565).
Its weakness is calibration (MCB 0.080062, 3.3× NOCTUA's), which is what you
would expect from sampled paths that were never fitted to this question.

**Three earlier runs were invalid, and two of those were our fault.** Run 1
used `top_p=0.9`, whose nucleus truncation removes exactly the tail where large
moves live. Runs 1 and 2 both then hit a worse defect:
`KronosPredictor.predict(sample_count=32)` **averages** its draws
(`kronos.py:467`) and returns one smoothed path, so `n_paths` was 1, every
"probability" was exactly 0.0 or 1.0, and the sampled volatility ratio read
0.259 and 0.373. That reads as "Kronos is 2.7× too calm" and it is not — with
genuine sampling the ratio is **1.172**. Publishing the earlier number would
have been a false claim about someone else's model, produced by our harness and
flattering to ours. `eval/kronos_ci.py` now draws via `predict_batch` and
`assert_ensemble()` aborts on episode 1 rather than 120.

A fourth run was lost to plumbing rather than science: the compute succeeded and
the commit step failed, and the predictions were recovered from the Actions
artifact by `recover-kronos-artifact.yml` rather than paying 3h49m again.

**Compute.** 114.2 s/episode for Kronos against 0.6 ms for NOCTUA — a factor of
roughly 190,000, against a parameter ratio of 1,291.

**Scope.** 120 daily non-overlapping episodes from one asset over one four-month
window, against Kronos-**small**. It is not a claim about Kronos-base, about
other horizons, or about other instruments.

Getting Kronos to run at all took moving it into CI: `huggingface.co` weight
downloads are blocked from the development environment (HTTP 403 via the proxy);
only metadata is reachable, which confirms Kronos-small at 24.7 M parameters and
Kronos-base at 102.3 M against NOCTUA's 19,134 — roughly 1,300× and 5,300×
larger — but says nothing about accuracy.

Run 1 was invalid for an honest reason: `top_p=0.9` truncates the token
distribution exactly where large moves live, and the sampled paths came out at
0.377 % volatility against 1.467 % realized.

Run 2 was invalid because of **a bug in this repo**. `KronosPredictor.predict(
sample_count=32)` does not return 32 paths — it averages them internally
(`kronos.py:467`; the `predict_batch` docstring says "automatically averaged
internally") and returns one smoothed path. So:

| | run 1 | run 2 | what it actually was |
|---|---|---|---|
| sampled σ / realized RV | 0.259 | 0.373 | the average of 32 paths, not Kronos's σ |
| paths per episode | 1 | 1 | asked for 32 |
| touch probabilities | {0, 1} | {0, 1} | an indicator on the mean path |

The tempting read of run 2 — *"Kronos under-states volatility by 2.7×"* — is
false, and it is the kind of claim that is easy to publish because it flatters
the model doing the measuring. Scoring those 0/1 indicators against NOCTUA's
calibrated curve would have compared a probability forecast to a point forecast
and called the result a win.

`eval/kronos_ci.py` now draws paths via `predict_batch` with the context
repeated and `sample_count=1`, which returns each rollout intact at unchanged
compute, and `assert_ensemble()` aborts in the first episode — not the last —
if the ensemble is ever collapsed again. Until that run lands, every claim in
this repo is against Log-HAR and the baselines above, which is the harder
comparison, and **none of them is a measurement of Kronos**.

*Educational research only. Not financial advice.*
