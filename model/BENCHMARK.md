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

## 5b. "But Kronos shows an upside probability" — what that number is

The natural objection to §6f's conclusion that direction is not predictable is
empirical and fair: the Kronos dashboard displays **Upside Probability (Next
24h): 66.7 %**, so evidently something can do it.

It cannot. 66.7 % is 2/3.

`kronos_local/app.py:133` computes

```python
"upside": round(100.0 * ups / N_SAMPLES, 1)
```

— the fraction of Monte-Carlo rollouts that happened to finish higher. Every
one of the 120 stored `p_up` values in `data/kronos_predictions.json` is an
exact multiple of 1/32. It is a **count of coin flips, not a calibrated
probability**, and the count carries sampling error nobody displays:

| | |
|---|---|
| Monte-Carlo standard error at p = 0.5, n = 32 | **8.8 pp** |
| displayed precision | 0.1 pp — **88× more precision than it has** |
| episodes asserting 0 % or 100 % certainty | **9 of 120** |

A displayed "66.7 %" has a 95 % interval of roughly [50 %, 83 %] from
Monte-Carlo noise alone, before asking whether the model knows anything. And
on nine occasions it claimed *absolute certainty* about the direction of
Bitcoin 19 hours ahead, on the strength of 32 rollouts landing the same way.

That is an argument about sampling error. The stored episodes carry realized
outcomes, so `eval/kronos_direction.py` simply scores it:

| forecaster | log loss | DSC | shuffled null p95 | clears? | vs base rate | 95 % CI |
|---|---|---|---|---|---|---|
| base rate (constant) | **0.68967** | 0 | — | — | 0 | — |
| constant 0.5 | 0.69315 | 0 | 0 | — | −0.00348 | [−0.016, +0.009] |
| **Kronos `p_up`** | **1.20547** | 0.011105 | 0.016231 | **no** | **−0.51580** | **[−0.991, −0.183]** |

**Kronos's direction call scores 75 % worse than a constant**, and its
discrimination does not clear the shuffled null — so there is no conditional
information in it either. The 0 %/100 % episodes are what destroy it: when a
forecast asserts certainty and is wrong, log loss is unbounded.

For scale, NOCTUA's own `prob_up` — which is *not published*, precisely
because it was known to be unreliable — loses 0.071 nats to a constant.
Kronos loses 0.516, **seven times worse**.

On the barrier task, from the paired 120-episode run in §5: NOCTUA DSC
0.01815 (clears its shuffled p95 of 0.008546), Kronos 0.00568 (does **not**
clear its 0.006238). Kronos's Brier of 0.2111 is worse than climatology's
0.1396.

**So the answer to "how is Kronos doing it" is that it is not.** It is
displaying a 32-sample proportion to one decimal place. Producing a number is
not the same as having skill, and the only way to tell the difference is to
score it against what happened.

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

## 6b. The training defect, and the one change that fixed it

Sections 1–2 say NOCTUA wins the full predictive distribution and is *behind*
`log_har_gauss` on binary barrier discrimination. That gap had a cause in the
training setup, not in the features.

**Stage B was trained on an easier problem than it faces.** It learns quantiles
of `M_up/sigma` conditioned on `log sigma`, and in training `sigma` was `RV` —
the *realized* window volatility. At serving it is the model's own forecast.
Dividing by the realized value removes the volatility-forecast error from the
training problem but not from the deployed one:

| | sd |
|---|---|
| `M_up / RV_true` — what training fitted | 0.5490 |
| `M_up / sigma_served` — what serving faces | 0.9312 |

Stage B was fitted to a distribution 40 % narrower than the one its output is
applied to. Worse, `RV` sat in the target's *denominator* and in the
*conditioner*, so noise in `RV` alone manufactures dependence: permuting `RV`,
which destroys every economic relationship it has with `M_up`, still leaves
Spearman −0.4331 against −0.0556 actually observed. The arithmetic artifact was
larger than the real relationship (Pearson's spurious correlation of ratios,
1897).

**The fix** retargets stage B onto a volatility reference that is causal by
construction: `exp(har_1d) * sqrt(H)`, where `har_1d` is a feature built from
bars strictly before the anchor. Nothing is fitted, so nothing can leak; clip
bounds come from each fold's training episodes only. It lands at sd 0.9272
against the 0.9312 serving faces — and it is a *weaker* forecast than the
deployed one, which makes the arm conservative.

Six walk-forward folds, three seeds, identical in every other respect:

| arm | DSC/UNC | vs base | folds won | t-like | pinball | CRPS | QLIKE |
|---|---|---|---|---|---|---|---|
| baseline | 0.04980 | — | — | — | 0.003633 | 0.005307 | 0.2998 |
| **serve_consistent** | **0.05382** | **+8.1 %** | **6/6** | **+3.78** | **0.003594** | **0.005277** | 0.2996 |
| uniqueness | 0.04981 | +0.0 % | 3/6 | +0.80 | 0.003633 | 0.005307 | 0.2998 |
| nonoverlap | 0.04493 | −9.8 % | 0/6 | −3.90 | 0.003734 | 0.005477 | 0.2953 |

Discrimination improves in **every fold**, and pinball (5/6) and CRPS (5/6)
improve with it, so this is not discrimination bought by wrecking the
distribution. QLIKE is unchanged, as expected — the change is to stage B.

**What this does NOT establish.** It does not put NOCTUA ahead of
`log_har_gauss`. The +8.1 % is against our own baseline; against log_har the
correct paired statistic is **4/6 folds, mean +0.00083, t-like +0.46** — noise,
with one fold at −13.4 %. The honest reading is that the gap *closed*: baseline
won 1 of 6 folds against log_har at mean −0.00318, serve_consistent wins 4 of 6
at mean +0.00083. Behind → level. Not ahead.

**Two arms that did not work, both informative.**

*uniqueness* is **provably vacuous here**, not an experimental null. Our
episodes sit on a complete regular grid — every hour, every horizon — so
concurrency is a constant of the grid (61), average uniqueness is exactly 1/61
on the training split, and the normalised multiplier is 1.0000 to machine
precision. Measured, it moves DSC/UNC by ~1e-5 against the ~4e-3 that
`serve_consistent` moves — about 100× smaller and not in a consistent direction
(3 folds up, 3 down), i.e. effectively equivalent rather than exactly equal.
López de Prado's uniqueness weighting is built for *event-driven* labels with
data-dependent holding periods; on a regular grid there is nothing for it to
grip. The 60.9× redundancy (8,380 effective observations from 510,496 episodes)
is real and must be attacked by removing samples or shrinking capacity, not by
reweighting.

*nonoverlap* — training only on the 4,965 non-overlapping episodes — is worse
in **0/6 folds won**, −9.8 % DSC/UNC, and worse on pinball and CRPS. The 100×
overlapping augmentation is genuinely buying barrier discrimination and
distribution shape. It does slightly improve QLIKE (0.2953), so the redundancy
does cost something on the volatility level, but not enough to justify the
trade.

**Checked at the deployment configuration before shipping.** Walk-forward is
six refits; the deployed model is one fit applied to 2024-07 onward, so the two
can disagree. On that single split the raw model gains less (DSC/UNC +1.2 %)
and its raw coverage error gets *worse* (2.073 → 2.593 pp). That regression is
entirely a level effect, and `serve/adaptive.py` — which always runs in
production — absorbs it:

| | raw coverage | after the adaptive correction |
|---|---|---|
| old (trained on RV) | 2.073 pp | 1.360 pp |
| **new (causal reference)** | 2.593 pp | **1.284 pp** |

So as actually deployed the new artifact is better on discrimination *and* on
coverage (−5.6 %), with QLIKE unchanged. Shipped: `train_v2.py` now trains
stage B against the causal reference and the artifact records
`stage_b_sigma_ref: causal_har_1d_clipped`.

**A packaging bug this surfaced.** `train_v2` recorded
`"feat_cols": list(X.columns)` — correct only while every column in
features.parquet was a model input. Adding the `eff_*` research columns made
the artifact declare 42 inputs while its weights expected 39, and serving died
with a matmul dimension error. `prepare()` now returns the columns it actually
consumed and the artifact records those. The research columns were added
several commits earlier and the mismatch was invisible until the artifact was
next rebuilt.

**An earlier version of this result was discarded.** The first cross-fitted
reference used ordinary K-fold, fitting each held-out block on every *other*
block including later ones — not causal for a time series, so a 2021 training
episode's sigma was computed partly from 2025 data. That run was 5/6 positive
and is not reported. Caught in review; the replacement fits nothing at all.

---

## 6c. Training on other crypto: volatility transfers, barrier shape does not

The question the 3300-day harvest existed to answer. BTC, ETH, LTC and XRP all
reach back to 2017-08-02, so altcoin episodes pass through the *same*
`splits.time_splits` boundaries as BTC's and contribute genuine pre-`TRAIN_END`
training data. SOL is **excluded from training**: it lists later (899 days, all
of it inside BTC's test era) and pooling it would reintroduce exactly the
contemporaneous cross-sectional leak the long harvest was run to avoid.

Test and calib slices are forced BTC-only in both arms — adding assets must be
judged on the instrument that ships, not on a pool average a strong altcoin
could carry.

| metric | BTC-only | pooled | folds won | t-like |
|---|---|---|---|---|
| DSC/UNC | 0.049802 | 0.048905 | 2/6 | −0.94 |
| pinball | 0.003633 | 0.003627 | 5/6 | +0.77 |
| CRPS | 0.005307 | 0.005312 | 3/6 | −0.51 |
| **QLIKE** | **0.299840** | **0.294241** | **5/6** | **+1.63** |

**Volatility improves in 5 of 6 folds, 1.87 %.** Barrier discrimination does
not move and if anything drifts negative: 2/6 and a t-like of −0.94, with fold
swings (−6.6 % to +5.3 %) dwarfing the −1.8 %. CRPS flat.

A note on what "1.87 %" is, because the two available averages differ and the
larger one is the one quoted. Every percentage in this section is a ratio of
the columns above, i.e. of fold-AVERAGED scores. Averaging the six per-fold
percentages instead gives **1.45 %** for QLIKE — smaller, because a ratio of
means weights each fold by its own QLIKE level, and the levels line up with
the effect: the fold pooling helps most is the highest-QLIKE one (2023, 0.435,
−4.9 %) and the one fold it HURTS is the lowest (2026, 0.204, +1.3 %). For DSC/UNC
the two agree (−1.80 % vs −2.00 %). Neither convention changes the direction
of anything here, and both are reported so the reader can pick.

**These numbers are WEAKER than the ones first published here, and the reason
matters more than the numbers.** The first version of this section reported
QLIKE 6/6, 2.69 %, t-like +2.74. It was computed before the ETH gap filter
existed, i.e. with 2,924 mislabelled episodes in the pooled arm. Review caught
the gap; refiltering and re-running gives 5/6, 1.87 %, t-like +1.63. Both
figures are ratios of fold-averaged QLIKE, so the comparison is like for like;
under the per-fold average the same pair reads 2.42 % → 1.45 %.

I predicted the opposite. The commit that added the filter argued that
stretched windows inflate RV, so the defect must have biased *against* pooling
and "the re-run confirms rather than rescues the result". It did not: it
weakened it, on five of six folds, and fold 2026 flipped sign entirely
(−1.7 % → +1.3 %). The mechanism reasoning was plausible and untested, which is
the same habit that produced the venue-cap error earlier in this file.

At t-like +1.63 on n = 6 this is a modest effect, not a strong one, and it
should be read that way.

| fold | n_train BTC → pooled | DSC/UNC | QLIKE |
|---|---|---|---|
| 2021 | 102,087 → 301,821 | −5.2 % | −0.9 % |
| 2022 | 137,127 → 441,421 | +1.5 % | −2.6 % |
| 2023 | 172,167 → 581,520 | −4.2 % | −4.9 % |
| 2024 | 207,207 → 721,477 | −2.7 % | −0.6 % |
| 2025 | 242,343 → 861,895 | +5.3 % | −1.0 % |
| 2026 | 277,383 → 1,001,925 | −6.6 % | **+1.3 %** |

The split is mechanically sensible rather than surprising. Crypto volatility is
strongly correlated across assets, so 845,000 extra episodes sharpen the
volatility cascade. Excursion *shape* — how far a path travels per unit of
realized vol — reflects each asset's own microstructure, liquidity and tick
size, so pooling it adds variance without adding signal. That is the same
boundary `eval/cross_asset.py` found from the other direction: σ transfers
zero-shot to unseen altcoins, and the excursion shape is where transfer gets
shakier.

### Does the gain depend on the expiry? Yes — and it is gone by H = 19

Scored through `run_fold` with `prod_override`, one horizon at a time, so every
figure comes from the same committee, both barriers and the same grid as the
aggregate above. One train/test split, not six folds.

| H | n | QLIKE BTC | QLIKE pooled | Δ | DSC/UNC BTC | DSC/UNC pooled | Δ |
|---|---|---|---|---|---|---|---|
| 6 | 770 | 0.3673 | 0.3553 | **−3.3 %** | 0.08341 | 0.08146 | −2.3 % |
| 12 | 769 | 0.2884 | 0.2858 | −0.9 % | 0.04565 | 0.04535 | −0.6 % |
| **19** | 769 | 0.2478 | 0.2509 | **+1.2 %** | 0.03957 | 0.03830 | −3.2 % |
| 24 | 769 | 0.2361 | 0.2373 | +0.5 % | 0.03830 | 0.03698 | −3.5 % |

**The volatility gain is a short-horizon effect.** −3.3 % at H = 6, decaying
through −0.9 % at H = 12, and *negative* by the deployed H = 19. Discrimination
is negative at every horizon. Both survived the ETH gap fix and the switch to
committee scoring, so neither is an artifact of the two defects this section
has already had.

That is a sharper reason not to ship pooling than the aggregate gave. The
aggregate said "helps a metric the product does not optimise"; this says the
help is not even present at the expiry the product is sold at.

**The H = 19 sign flip persists, and now means something narrower.** This table
says +1.2 % worse at H = 19; the six-fold walk-forward, which scores H = 19 and
nothing else, says 1.87 % better in 5 of 6 folds. Fixing the estimator removed
the *DSC* half of that discrepancy — the old table's DSC column was the neural
head alone on upside barriers, and its H = 6 baseline read 0.12641 against
0.08341 here, a 34 % difference — but the QLIKE half is unchanged. So the
original attribution stands for QLIKE: one split of ~770 episodes cannot
resolve a 1–2 % difference that six refits can. Six folds beat one, and the
walk-forward remains the measurement.

**Not shipped.** The deployed product is a barrier forecast, and pooling does
not improve barrier discrimination — and the QLIKE gain that survives the gap
fix is concentrated at horizons shorter than the one that ships. A 1.9 % gain
at H = 19 would be worth arguing about for a volatility product; +1.2 % in the
wrong direction there does not justify tripling the training set and adding
three assets' data-quality risk to a model whose output is a touch
probability. Recorded as a modest, replicated effect on the wrong metric, at
the wrong expiry.

---

## 6d. The hour the feature builder was throwing away

The largest single gain measured in this project, and it came from an audit of
the pipeline rather than from any modelling idea.

`features.py` states its no-lookahead contract as "hourly rows with index
≤ a−1". It was delivering ≤ a−2. `_trailing_sum(x,k)[i]` is already
`sum(x[i-k:i])`, exclusive of `i`, and the result was **then** indexed at
`a−1` as well — a second shift, applied on top of one that already satisfied
the contract. The last complete hour before the anchor was discarded on every
episode, in training and in serving alike.

The consequence is sharpest for `har_1h`, which *is* a one-hour window: at the
old setting it was not the last hour's realized volatility but the hour before
it. Mean absolute shift when corrected: **0.613 in log-vol units**. 31 of 42
features move.

The double shift was defensive, not accidental — the contract note says it
means "an off-by-one cannot silently leak the anchor hour itself" — and it is
safe in the only direction that matters. But the safety was redundant.
`audit_lookahead()` verifies the contract **numerically**, by corrupting every
bar at or after each probed anchor and confirming no feature moves, and it
passes at the tighter setting with max feature change **0.000e+00 over 306,261
probed episodes**.

Same six walk-forward folds, same seeds, same committee, same causal stage-B
reference rebuilt from each arm's own features. One variable changed.

| metric | lag 1 (old) | lag 0 (contract) | relative | folds won | t-like |
|---|---|---|---|---|---|
| **QLIKE** | 0.299623 | **0.289636** | **−3.33 %** | **6/6** | **+4.06** |
| **pinball** | 0.003594 | **0.003574** | −0.54 % | **6/6** | **+4.99** |
| **CRPS** | 0.005277 | **0.005250** | −0.51 % | **6/6** | **+3.63** |
| DSC/UNC | 0.053817 | 0.055256 | +2.67 % | 4/6 | +1.68 |

The decision rule was written before the run — *ship only on ≥ 5 of 6 folds on
DSC/UNC or QLIKE* — and QLIKE cleared it at 6/6 with the strongest t-like this
project has produced. **Barrier discrimination did not clear it: 4/6 at +1.68
is not a result and is recorded as the null it is.**

`extra_lag_hours` is now a parameter with default 0. The old behaviour remains
exactly reproducible at 1, which matters because **every number in this file
above this section was measured at 1.** They are not wrong; they describe a
model fitted on features that were an hour staler than they needed to be.

**Why this counts as information recovery rather than a loosened constraint.**
Nothing was relaxed. The contract is unchanged, the numerical audit is
unchanged and still passes, and the aggregates still end strictly before the
anchor. What changed is that the implementation now meets the contract exactly
instead of with an hour to spare.

---

## 6e. Every number above describes an anchor the product almost never uses

`splits.production_mask` is `H == 19 AND anchor_hour == 17`, and every headline
in this file is measured on it. `serve/predict.py:74` anchors at the last
**closed** hour instead, and the publishing cron never overrides it —
`_next_anchor()`, the function that would pin serving to 17:00 UTC, is defined
and called from nowhere.

Measured over the 102 published forecasts in this repository's git history:

| | |
|---|---|
| anchored at 17:00 UTC | **5 of 102 (4.9 %)** |
| anchor hours actually used | all 24, roughly uniform |
| lag from window start to publication | median **1.33 h**, max 8.87 h |

So ~95 % of what has ever been shown to a user came from a configuration this
file had never scored. That is not a leak — it is a claim-versus-product
mismatch, and it was the most serious thing the end-to-end audit turned up.

`eval/anchors.py` settles it. Six folds, three arms, identical models per fold
(`run_fold` is deterministic given its seeds, so only the scored slice moves);
the wide arms subsampled to 6,000 episodes drawn once with a fixed seed so
per-fold sample sizes match the 17:00 arm and the comparison is paired on the
anchor rather than on power.

| arm | DSC/UNC | QLIKE | **vs Log-HAR** |
|---|---|---|---|
| 17:00 — the benchmarked slice | 0.05526 | 0.2896 | **−6.14 %** |
| all served anchors | 0.06196 | 0.2715 | **−6.06 %** |
| every anchor except 17:00 | 0.06465 | 0.2668 | **−7.13 %** |

**The claim holds.** The advantage over a calibrated Log-HAR is 6.14 % at the
benchmarked anchor and 6.06 % across all served anchors — a gap of 0.08 pp.
Barrier discrimination is *higher* away from 17:00, not lower.

The absolute QLIKE gaps between arms are a level effect: 17:00 UTC sits just
after the US equity open, near the intraday volatility peak, so those episodes
are harder in absolute terms. The ratio against Log-HAR, which faces the same
episodes, is what transfers — and it is flat.

`BLEND_W`, the committee's equal weighting and the adaptive correction were all
tuned on the 17:00 slice alone. The reasonable worry was that constants fitted
at the seasonal peak would not hold at the trough. They do.

**Scope, restated honestly.** The headlines in this file may be read as
applying across anchor hours, not only at 17:00. What remains unfixed is the
publication lag: anchoring at the last closed hour means the window has been
running a median 1.33 hours before the forecast is published. That is a
product defect, not a model one.

---

## 6f. The ceiling on volatility work — 92 % of the barrier error is not volatility

The most consequential measurement in this file, because it redirects the
whole research programme.

A barrier forecast composes two independent claims:

    P(touch u)  =  f( sigma_hat , shape )

the **volatility** claim (how big will the moves be) and the **shape** claim
(given moves of that size, how far does the path actually travel before
settlement). Reported together, the blame is unassignable — and they call for
completely different work.

`eval/firstpassage.py` separates them the only clean way available: give the
Gaussian first-passage law the **realized** volatility — a perfect forecast no
one could have made — and see what error survives. Whatever remains is pure
shape error, and it is the part no volatility model can ever remove.

### BTC paths chop; they do not travel like Brownian motion

Under driftless Brownian motion, `E[range] / sqrt(realized variance)` is the
constant `sqrt(8/π) = 1.5958` **in continuous time**.

> **Correction.** This section first compared BTC against that constant and
> reported a gap of −0.2646. That was wrong, and in the flattering direction.
> A running maximum observed at finitely many points is always below the true
> continuous maximum, while realized variance built from the same increments
> is unbiased — so the ratio is biased **down by discretization alone**,
> before any question about Bitcoin. Episodes measure RV from 5-minute bars
> over 19 hours, i.e. 228 increments, so the benchmark must be simulated at
> that resolution. `brownian_control()` does it: **1.5218**, not 1.5958.
> **About 28 % of the originally reported gap was my own sampling artifact.**

| | |
|---|---|
| Brownian, continuous time | 1.5958 |
| **Brownian, sampled as the data is** (19 h × 12 five-minute steps) | **1.5218** |
| **BTC measured** | mean **1.3311**, median **1.2942** |
| IQR | [1.0157, 1.6120] |
| 5–95 pct | [0.5899, 2.1835] |
| **mean − sampled benchmark** | **−0.1907**, block-bootstrap 95 % CI **[−0.2239, −0.1600]** |

BTC still burns realized variance without travelling, and the interval is
still nowhere near zero — the conclusion survives, at about 72 % of the
originally claimed size. (`LITERATURE.md` §5 located the theory — Feller's
1951 range distribution — but no published empirical measurement of this ratio
on real financial data. Reported as a measurement, not a novelty claim. It
reproduces the figure in `features.py`'s `eff_*` docstring to four decimals.)

### What that does to the textbook barrier formula

One-sided (where the reflection principle is exact — a two-sided version needs
the full Feller series, and approximating it as twice the one-sided
probability would manufacture the very overstatement being tested for):

| barrier | realized touch rate | Gaussian **fed realized vol** | ratio | error |
|---|---|---|---|---|
| 0.5 % | 0.7762 | 0.8302 | 1.070 | **+5.40 pp** |
| 1.0 % | 0.5932 | 0.6800 | 1.146 | **+8.68 pp** |
| 2.0 % | 0.3643 | 0.4526 | 1.242 | **+8.83 pp** |
| 3.0 % | 0.2338 | 0.3060 | 1.309 | **+7.22 pp** |
| 5.0 % | 0.1065 | 0.1528 | 1.435 | **+4.63 pp** |

**Given a perfect volatility forecast, the textbook formula still overstates
touch risk at every barrier**, and the error grows in relative terms as the
strike moves out — 1.07× at 0.5 %, 1.44× at 5 %.

### The number that redirects the programme

Comparing the oracle against the same formula fed a causal forecast
(`exp(har_1d)·sqrt(H)`), on the 5,324 episodes where both exist:

| barrier | oracle error | causal-forecast error | **share removable by perfect volatility** |
|---|---|---|---|
| 0.5 % | 5.41 pp | 5.85 pp | **7.6 %** |
| 1.0 % | 8.68 pp | 9.42 pp | **7.8 %** |
| 2.0 % | 8.84 pp | 9.65 pp | **8.5 %** |
| 3.0 % | 7.23 pp | 7.86 pp | **8.0 %** |
| 5.0 % | 4.64 pp | 4.94 pp | **6.0 %** |

**A perfect volatility forecast — omniscience — would remove only 6–8.5 % of
the barrier error. The other 91.5–94 % is shape.**

Three consequences, stated plainly:

1. **Further volatility work has a hard ceiling on the deployed task.** The
   `serve_consistent` fix and the freshness fix were both worth having, and
   between them they moved QLIKE by ~3 %. But QLIKE is the volatility metric;
   on the barrier question that the product is actually sold on, the entire
   volatility channel is worth at most 8.5 %.
2. **This is precisely what NOCTUA already does differently, and why it
   wins.** It does not assume a Brownian first-passage law — Stage B learns
   the excursion distribution from data, conditioned on shape features and
   mixed over 32 volatility atoms. That is why its deep-tail calibration is
   1.09 pp against the Gaussian's 3.33 pp. The model's advantage was always in
   the 92 %, and now that is measured rather than asserted.
3. **The next research effort belongs in the shape, not the level.** Concretely:
   the excursion head, the coupling penalty, the mixing integral and the
   conditioning features that describe *path geometry* rather than path size.
   `eff_*` — range per unit of realized vol — was ablated and found null for
   discrimination, but that ablation now looks under-powered rather than
   decisive given how much of the error it was aimed at.

---

## 6g. The sell-versus-straddle call — the model's strongest validated output

The dashboard publishes `p_vol_amplify`: the probability that realized
volatility over the forward window will exceed the trailing realized
volatility over a window of the same length. In the seller's language, *will
it get wilder than it has been?* — the decision between collecting premium on
a short strangle and paying for a long straddle.

**It had never been scored.** The volatility level (QLIKE) and the barrier
probabilities (CORP, Christoffersen) were measured exhaustively; this binary
call was derived in `serve/predict.py:155`, published on every refresh, and
appeared in no evaluation anywhere. Same asymmetry of attention that left
direction unmeasured.

A good level forecast does not imply a good exceedance probability. The
amplification call reads the predictive *spread*, not the centre — it is
`1 − F(trailing)` on Stage A's own quantile curve — so a model with a perfect
median and a 20 % too-tight distribution would systematically understate the
chance of a wild night, which is exactly the error that breaks a seller.

Six walk-forward folds, production slice, against a causal climatology:

| forecaster | log loss | MCB | **DSC/UNC** | clears null? | gain vs climatology | 95 % CI | folds |
|---|---|---|---|---|---|---|---|
| **NOCTUA** | **0.61005** | 0.01227 | **20.303 %** | **yes** | **+0.07485** | [+0.047, +0.103] | **6/6** |
| HAR + lognormal | 0.61265 | **0.00915** | 16.943 % | yes | +0.07226 | [+0.060, +0.084] | 6/6 |
| persistence (0.5) | 0.69315 | 0.00509 | 0 % | no | −0.00824 | [−0.014, −0.002] | 3/6 |
| climatology | 0.68491 | 0.00097 | 0 % | no | 0 | — | 0/6 |

**This is the model's strongest measured skill by a wide margin.** A Brier
skill score of 20.3 % against 4.98 % on barrier discrimination and 0.18 % on
direction — the amplification question is **113× more answerable than the
direction question** and **4× more than the barrier question**, on the same
episodes, from the same model.

Two honest qualifications:

- **NOCTUA leads on discrimination, not on everything.** Its DSC/UNC is 3.4 pp
  above the HAR-plus-lognormal competitor, but its *calibration is worse*
  (MCB 0.01227 against 0.00915), and the log-loss gap is only 0.0026 nats.
  The neural model discriminates better and is calibrated slightly worse. A
  recalibration layer on this specific output is an obvious, cheap next step
  and is not yet done.
- **The naive route already gets most of it.** HAR with a lognormal spread
  fitted on training residuals reaches 16.9 % skill. The sophisticated model
  adds a fifth on top of that, not an order of magnitude.

The base rate is 0.4286 — **43 % of production nights are wilder than the day
before them**, and both real forecasters move that number around substantially
and reproducibly.

---

## 6h. Stage B: modelling the max excursion directly — a capability, not an accuracy win

§6f established that ~92 % of the barrier error is excursion shape, so Stage B
is where the work belongs. This is the first attempt, and it did not deliver
what I predicted.

### The defect is real and was measured first

A seller short both wings is exposed to **either** barrier breaking. The model
produced no such number — anyone needing it had to build `1 − (1−p_up)(1−p_dn)`
from the two marginals, which assumes independence. The sides are strongly
negatively dependent: a path with a fixed realized-variance budget cannot spend
it travelling both ways.

| | Spearman(M_up/RV, M_dn/RV) |
|---|---|
| independence (what the construction assumes) | 0 |
| **BTC measured**, 5,324 production episodes | **−0.687** |
| **driftless Brownian**, continuous limit | **−0.806** |

The Brownian figure is my own simulation, converged over 228 → 16,000 steps
(−0.8069, −0.8058, −0.8047, −0.8047, −0.8070) and agreeing with the analytic
value attributed to Jaworski & Dąbrowski (2024), *Mathematics* 12(17) 2707.
*Caveat: the series as relayed to me did not reproduce that value when I
evaluated it — it summed to 0.9516 — so I am relying on the simulation and
citing the paper for the analytic result without having reproduced its
algebra.*

The useful reading is that **BTC sits between the two extremes and much nearer
Brownian than independence.** Assuming independence is therefore a *larger*
error than assuming the textbook diffusion would have been.

Its cost runs against a seller: independence understates P(either breaks) —
realized 0.8922 against 0.8368 at 1 %, 0.6362 against 0.5995 at 2 %.

### The fix, and the result

`StageB` gained `q_mx`, a fourth monotone-quantile head predicting the
standardized max(M_up, M_dn) directly. Six folds; adopt rule ≥ 5/6 at each
barrier, fixed before the run:

| barrier | model | log loss | DSC/UNC | vs independence | 95 % CI | folds |
|---|---|---|---|---|---|---|
| **1.0 %** | **q_mx head** | **0.27819** | **20.47 %** | **+0.02533** | **[+0.013, +0.042]** | 4/6 |
| 1.0 % | independence | 0.30352 | 18.72 % | — | — | — |
| 1.0 % | two-sided Gaussian | 0.79083 | 17.61 % | −0.48731 | [−0.608, −0.359] | 1/6 |
| 2.0 % | q_mx head | 0.62897 | 13.34 % | −0.00215 | [−0.013, +0.010] | 4/6 |
| 2.0 % | independence | **0.62682** | 13.18 % | — | — | — |
| 2.0 % | two-sided Gaussian | 0.74216 | **23.60 %** | −0.11534 | [−0.205, −0.039] | 2/6 |
| 3.0 % | q_mx head | 0.63089 | 12.77 % | −0.00345 | [−0.010, +0.004] | 3/6 |
| 3.0 % | independence | 0.62744 | 12.70 % | — | — | — |
| 3.0 % | **two-sided Gaussian** | **0.60385** | **22.59 %** | +0.02359 | [−0.015, +0.061] | 5/6 |

**The rule is not met at any barrier — 11 of 18 (barrier, fold) cells. This is
a null result on the accuracy claim and nothing is claimed as an improvement.**

Three things are worth keeping from it:

1. **At the 1 % barrier the head is genuinely better** — +0.025 nats with a CI
   that excludes zero, and 20.47 % against 18.72 % discrimination. But 4/6
   folds is not consistency, and by the pre-registered rule that is not enough.
2. **At 2–3 % the head and independence are indistinguishable**, and the
   plain **two-sided Gaussian beats both at 3 %** (log loss 0.60385, 5/6
   folds) while discriminating far better at 2–3 % (23.6 % / 22.6 % against
   ~13 %). That is an uncomfortable result and it is reported rather than
   buried: the textbook formula, on the model's own sigma, is the best
   available answer for the wider strikes a seller actually writes.
3. **Folds 2025 and 2026 show every model-based arm overstating badly**
   (realized 0.397 / 0.418 against head 0.614 / 0.636). The recent-era
   marginals are too wide. That is a calibration problem shared by both
   constructions and is the more promising thread than the head was.

### What ships

`q_mx` stays in the code, because the model previously **could not answer the
question at all** and now can, at no measured accuracy cost. That is a
capability addition, not an accuracy improvement, and the distinction is the
whole point. The shipped artifact is unchanged — no retrain was performed on
the strength of a null.

**Second failed prediction in a row.** The `lam_r` ablation (§AUDIT 5b) and
this one were both well-motivated by independent measurements, both looked
good on the first fold, and neither survived six. The pre-registered rules are
what makes that visible instead of quietly rewriting the hypothesis around the
result.

---

## 6i. The market the model was fitted on no longer exists

Prompted by a domain observation that turned out to be both correct and larger
than expected: BTC volatility has fallen year on year, with a step at the
spot-ETF launch, and the model's training era is dominated by a regime that no
longer occurs.

Median 19-hour realized volatility, production anchors:

| year | median RV | p95 RV | P(move > 2 %) | return kurtosis |
|---|---|---|---|---|
| 2013 | 5.280 % | 21.283 % | 78.4 % | **15.04** |
| 2017 | 4.012 % | 10.265 % | 85.5 % | 3.30 |
| 2021 | 3.315 % | 7.952 % | 89.6 % | 3.96 |
| 2023 | 1.710 % | 3.498 % | 46.0 % | 5.26 |
| 2024 | 2.049 % | 4.154 % | 61.2 % | 5.41 |
| 2025 | 1.520 % | 3.480 % | 39.7 % | 2.94 |
| 2026 | 1.606 % | 3.258 % | 41.8 % | **0.41** |

**Volatility down ~60–70 %, and the fat tails went with it** — kurtosis 15.04
→ 0.41, p95 realized vol 21.3 % → 3.3 %. Split at the ETF launch
(2024-01-11):

| | n | median RV | p95 RV |
|---|---|---|---|
| pre | 4,384 | 2.975 % | 9.188 % |
| post | 941 | **1.736 %** | **3.827 %** |

Ratio **0.584**, Mann-Whitney **p = 3.1e-156**, KS **0.4344**. Not drift — a
different market.

This is the mechanism behind two things already measured independently and
attributed only vaguely before: `eval/either.py` finding every model arm
over-forecasting in 2025–26 (realized P(either at 2 %) 0.397 / 0.418 against
0.614 / 0.636), and `serve/adaptive.py` quietly applying a 0.93–0.96 shrink
every night. The model is being asked about a market 42 % quieter than its
training median, and it says so.

### The defect this exposed: an untrained weight switched on in production

`reg_post_etf` is the flag `hour_ts >= 2024-01-11`. In the **shipped** split
(train ends 2023-01-01):

| | rows | value | standardizes to |
|---|---|---|---|
| train | 189,831 | identically **0** | exactly 0.0 |
| test | 73,867 | identically **1** | exactly 1.0 |

A constant input receives zero gradient, so its first-layer weight never
leaves random initialization. Measured on the shipped artifact, that weight is
of **completely ordinary magnitude** — mean |contribution| 0.088–0.114 per
seed against a mean |W| over all inputs of 0.096–0.116.

So at serve time **every hidden unit in both stages receives a full-strength,
never-trained random offset that was identically absent during training.** It
is not a weak feature; it is noise injected only in production. A flag that
never varies in training carries no information by construction, so there is
nothing to lose by removing it.

**Why the walk-forward mostly hides it.** Fold 2021 trains to 2020-07 and
tests 2021 — the flag is 0 on both sides and nothing flips. Only folds 2024
and 2025 straddle the launch at all. The shipped split exposes it fully, which
is why `eval/regime.py` reports the training standard deviation of the flag
per fold alongside the scores: the folds where it is 0.000 are the ones where
the defect is live.

*(A cosmetic wrinkle in that script's output: it prints the test-set standard
deviation of the flag as 0.000000 too, which is correct — the test set is
identically 1 — but reads as though nothing differs. The means, 0.0 and 1.0,
are the informative numbers.)*

### Removing the flag: measured, and it is a wash

Six folds, with and against:

| metric | with flag | without | delta | folds better |
|---|---|---|---|---|
| DSC/UNC | 0.055477 | 0.055577 | +0.000101 | 3/6 |
| pinball | 0.003577 | 0.003578 | +0.000001 | 2/6 |
| CRPS | 0.005251 | 0.005243 | −0.000008 | 2/6 |
| QLIKE | 0.290346 | 0.291492 | **+0.001146 (worse)** | 2/6 |

**No detectable effect, and QLIKE is marginally worse without it. Third failed
prediction in a row.**

The distinction from the previous two nulls is worth drawing, because it
changes what to do. There the *mechanism* was speculative. Here the mechanism
is proven — a constant input receives zero gradient, its weight demonstrably
sits at initialization, and I measured that weight to be of ordinary magnitude
— but the *effect* is below detection: one input's worth of random offset among
39, apparently absorbed by the rest of the network.

So the flag is **not removed on the strength of a null**, and no improvement is
claimed from touching it. It is documented as a design defect to fix at the
next retrain, when the split moves forward anyway.

### The real finding is the split, not the flag

The flag is a symptom. `eval/regime.py`'s weight audit gives the disease:

| year of training data | share of the shipped model's fitted weight |
|---|---|
| 2017 | 3.49 % |
| 2018 | 10.18 % |
| 2019 | 13.49 % |
| 2020 | 17.93 % |
| 2021 | 23.69 % |
| 2022 | 31.22 % |
| **post-ETF (2024-01-11 onward)** | **0.00 %** |

`TRAIN_END` is 2023-01-01 and the last training anchor is 2022-12-30 — over a
year before the launch. **100 % of the shipped model's fitted weight comes from
a regime that has since ended.** The 900-day half-life redistributes recency
*within* the old regime; it cannot reach the new one. Its reference clock is
the training set's own end date, now ~3.6 years stale against a 2026 serve
date.

And the shift reaches every volatility-scale input, not just the flag —
train-vs-test KS on the shipped split:

| feature | KS | sd train → test |
|---|---|---|
| `har_22d` | **0.614** | 0.417 → 0.262 |
| `rng_gk_5d` | 0.523 | 0.481 → 0.332 |
| `har_5d` | 0.490 | 0.488 → 0.320 |
| `vov_22d` | 0.427 | 0.097 → 0.068 |
| `har_1d` | 0.403 | 0.551 → 0.444 |
| `cal_hour_sin` (control) | **0.0001** | 0.707 → 0.707 |

The calendar features are unmoved, which is the control that says this is a
volatility-regime effect and not a data artifact. Every RV-scale feature is
served on a distribution 25–40 % narrower than the one its weights were fitted
against.

**That is the honest answer to "what is stopping the model".** Not capacity,
not architecture, not the loss. The deployed artifact is a well-built estimator
of a market that stopped existing in January 2024, held together in production
by `serve/adaptive.py`'s nightly 0.93–0.96 shrink — a patch doing structural
work. The fix is to move the training window forward and re-fit, which is a
larger change than anything attempted in this session and should be measured
the same way: pre-registered, walk-forward, and reported as a null if it is one.

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
| **noctua_v2** | **0.145176** | **0.440193** | **0.018148** | **0.023733** |
| kronos-small | 0.211059 | 0.651811 | 0.005683 | 0.077150 |
| climatology | *0.139591* | *0.428622* | 0.000000 | 0.000000 |

**It replicated, by accident.** Two Kronos runs finished independently — one on
this branch, one on `main` — 119 of their 120 anchors shared, neither aware of
the other. Agreement across two separate Monte-Carlo samplings:

| | σ/RV | noctua Brier | kronos Brier | noctua DSC | kronos DSC |
|---|---|---|---|---|---|
| run A | 1.172 | 0.146371 | 0.214208 | 0.018334 | 0.006580 |
| run B | 1.184 | 0.145176 | 0.211059 | 0.018148 | 0.005683 |

Same ordering, same margins to within a few percent. The table above reports
run B, which is the file on `main`.

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
two columns are not a fair comparison and neither model "loses" to it.

**And `DSC = 0` from climatology is a WEAKER control than it looks.** A constant
forecaster is pinned to zero by construction, so it cannot detect
discrimination manufactured by the scorer. CORP fits its isotonic regression on
the same 120 outcomes it then scores, so ANY forecaster with variation collects
some DSC from that in-sample fit — including one whose ordering is pure noise.
The control that actually measures this keeps each model's marginal forecast
distribution and destroys only its alignment with outcomes:

| model | real DSC | shuffled mean | shuffled p95 | clears the floor? |
|---|---|---|---|---|
| noctua_v2 | 0.018148 | 0.005096 | 0.008546 | **yes** |
| kronos-small | 0.005683 | 0.003722 | 0.006238 | **no** |

The isotonic manufactures DSC of roughly 0.004–0.006 out of noise alone.

**Kronos is genuinely better calibrated on the volatility LEVEL.** Median
predicted/realized: Kronos **1.172**, NOCTUA **1.232**. Both over-forecast; the
24.7 M-parameter general-purpose model is closer to unbiased than the
purpose-built one. That is a real result and it is not softened here.

**RETRACTED: "Kronos carries real conditional information."** An earlier
version of this section said exactly that, on the strength of DSC 0.0057–0.0066
being "not zero and not close to it". It does not survive the shuffled control
above: Kronos's DSC falls INSIDE the noise band the in-sample isotonic produces
from randomly-ordered predictions (p95 = 0.006238). On this evidence Kronos's
barrier discrimination is **not distinguishable from zero**, and the honest
statement is that 120 episodes cannot resolve it either way — not that it is
absent. NOCTUA's DSC clears the same floor by a wide margin.

Kronos's miscalibration is separately large and is measurable: MCB 0.077150,
3.3× NOCTUA's, which is what you would expect from sampled paths never fitted
to this question.

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

**Compute.** 114.8 s/episode for Kronos against 0.6 ms for NOCTUA — a factor of
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

---

## 6j. How accurate can a volatility forecast be? An answer to "within 10 %"

A reasonable target — *get predicted volatility within ~10 % of realized* — is
not attainable by any model, and that is worth establishing with a number
rather than an argument.

Realized volatility over a 19-hour window is not a hidden constant waiting to
be estimated. It is one draw from a distribution. A forecaster holding the
exactly correct conditional distribution is still wrong on any individual
night, because the night itself is random.

Regressing log realized volatility on **all 22 usable features, in-sample** —
deliberately a generous upper bound, since an in-sample fit cannot be beaten
out of sample:

| | |
|---|---|
| in-sample R2 of the full feature set (upper bound) | **0.665** |
| R2 of the causal `har_1d` reference alone | 0.453 |
| residual standard deviation, log units | **0.378** |

**A third of log-volatility variance is not explainable from trailing
information at all**, and that is the optimistic reading. What the residual
means for a forecast with perfect coefficients:

| coverage | realized / forecast lands within |
|---|---|
| 50 % of episodes | [0.79x, 1.21x] |
| 68 % of episodes | [0.73x, 1.38x] |
| 90 % of episodes | [0.60x, 1.98x] |

So "within 10 %" is achieved on roughly **40-45 % of nights by an oracle**, and
no amount of architecture, data or compute moves that materially — the limit is
the randomness of the outcome, not the quality of the estimator.

**What is attainable, and is what a seller actually needs:**

1. **Calibration** — when the model says a 3 % barrier has a 5 % chance of
   breaking, it breaks about 5 % of the time. Currently 1.09 pp error at
   alpha = 1 % against the Gaussian's 3.33 pp.
2. **Median unbiasedness** — realized/forecast centred on 1.0 rather than
   running high. Exactly what the regime shift broke and what
   `serve/adaptive.py` patches nightly.
3. **Discrimination** — ranking wild nights above calm ones. The amplification
   call does this at 20.3 % Brier skill, the most useful number the model
   produces.

A forecaster that is well calibrated and discriminating, but "only" within
21 % on half its nights, is a good volatility model. One that hits within 10 %
more often while being miscalibrated in the tail will bankrupt a seller.

---

## 6k. Training through the post-ETF era: better on average, vetoed on the tail

The fix §6i implies, tested. Both arms are scored on **exactly the same
held-out window** — production anchors from 2025-01-01 onward, unseen by
either — so the comparison isolates the training data rather than the
difficulty of the test years.

| arm | train | of which post-ETF | calib | common test |
|---|---|---|---|---|
| shipped | <= 2023-01-01, 189,831 | **0** | 52,359 | 585 production episodes |
| forward | <= 2024-07-01, 242,343 | **16,359** | 17,511 | the same 585 |

Identical test masks, asserted in code rather than assumed.

| metric | shipped | forward | change |
|---|---|---|---|
| **QLIKE** | 0.240829 | **0.238357** | **-1.03 %** better |
| **DSC/UNC** | 0.045829 | **0.047199** | **+2.99 %** better |
| pinball | 0.002555 | 0.002550 | better |
| CRPS | 0.003805 | 0.003811 | marginally worse |

**But the pre-registered rule had a veto**, and it fires. Aggregate deep-tail
miscalibration (MCB, alpha <= 2 %, up and down summed):

| barrier | shipped | forward | |
|---|---|---|---|
| 0.5 % | 0.101023 | 0.101023 | identical |
| 1.0 % | 0.008405 | 0.008342 | better |
| **2.0 %** | **0.029823** | **0.033246** | **worse** |
| **total** | **0.139251** | **0.142610** | **+2.41 % worse** |

The rule was *adopt only if QLIKE improves AND deep-tail calibration does not
worsen*, precisely because an average-case win bought with tail degradation is
the wrong trade for someone short options. **So this is not adopted.**

Two honest qualifications, in both directions:

- **This is the most promising result of the recent attempts.** The three
  preceding experiments (`lam_r`, `q_mx`, dropping `reg_post_etf`) were flat.
  This one moves QLIKE and discrimination together, in the direction theory
  predicts, and the mechanism is understood.
- **It is one window of 585 episodes.** That is far too little to resolve a
  2.4 % change in a calibration statistic, and the veto may be noise as easily
  as the gain may be. The identical 0.5 % figure across two independently
  trained models is itself a warning: ~98 % of episodes touch a 0.5 % barrier,
  so the isotonic fit there is nearly degenerate and MCB carries little
  information at that level.

**The correct response is more power, not a verdict.** A rolling version with
several independent test windows is the next step; declaring either a win or a
failure from one split would repeat the error this benchmark exists to catch.

---

## 6l. Refreshing the fit: the largest gain yet, and a veto I caused myself

§6k's single window was under-powered, so this walks six quarterly test
windows through 2025-2026, comparing a **frozen** fit (trained once to
2023-01-01, what actually ships) against a **refreshed** one (retrained to each
window's own embargoed boundary). Same episodes per window in both arms.

The first pass scored the 17:00 production slice: ~90 episodes per window. That
cannot resolve these effects — a six-window sign test is p = 0.109 even at 5/6
and p = 0.344 at 4/6. So the second pass scores **every H = 19 anchor hour**
inside each window, capped at 1,200 episodes drawn once with a fixed seed.
That is legitimate because §6e already established the model's edge is flat
across anchors (−6.14 % at 17:00 against −6.06 % across all), and it buys ~13×
the test data from the same calendar time.

| metric | frozen | refreshed | change | windows won |
|---|---|---|---|---|
| **QLIKE** | 0.238593 | **0.226621** | **−5.02 %** | **5/6** |
| pinball | 0.002592 | **0.002534** | −2.24 % | **5/6** |
| CRPS | 0.003849 | **0.003785** | −1.68 % | **5/6** |
| DSC/UNC | 0.072259 | 0.077377 | +7.08 % | 3/6 |
| deep-tail MCB | 0.248019 | 0.247655 | −0.15 % | 2/6 |

**This is the largest and most consistent gain measured anywhere in this
project** — bigger than the feature-freshness fix (−3.33 % QLIKE) that was
shipped, and it moves pinball and CRPS with it at 5/6 apiece.

**And the pre-registered rule still says DO NOT ADOPT.** The rule was *QLIKE
wins in ≥ 5 of 6 windows AND deep-tail MCB does not worsen in more than 2 of
6*. QLIKE clears at 5/6. Tail MCB is worse in 4 of 6, so the veto fires.

That verdict stands. But two things about the rule are worth recording, and
neither changes it:

**1. The veto is a win-count on a statistic whose average improved.** Per-window
tail-MCB deltas: `+0.0078, +0.0019, −0.0047, +0.0007, +0.0029, −0.0108`. Three
near-zero positive deltas outvote two of the largest magnitudes in the set, and
the mean moves *favourably* (−0.15 %). A count-based veto on an effect this
small counts noise.

**2. The tail comparison is confounded by my own experiment design.** The
refreshed arm was given a much smaller calibration slice — 17,511 episodes
against the frozen arm's 52,359 in §6k, and two quarters against eighteen
months here. The calibration slice is precisely the machinery that fixes tail
calibration. So the arm expected to have better tails was handed a third of the
data with which to fit them.

**What that means.** The refresh is not adopted, because the rule that was
fixed in advance says no and bending a rule after seeing the result is the
failure this benchmark exists to prevent. But the veto is very likely an
artifact, and the clean resolution is a specific, cheap follow-up: **match the
calibration-slice sizes between arms and re-run.** If the tail condition still
fails on matched calibration, the refresh genuinely costs tail accuracy and
should stay rejected. If it passes, the refresh is the biggest available
improvement to the model and should ship.

That experiment is the single highest-value next step in this repository, and
it is specified here rather than run because the session budget ended, not
because the question is open.

*Educational research only. Not financial advice.*
