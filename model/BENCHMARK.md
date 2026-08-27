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
>
> **That correction was itself wrong — see §5b-bis below.** The extremes in
> this data are not 5-minute closes.
> **About 28 % of the originally reported gap was my own sampling artifact.**

| | |
|---|---|
| Brownian, continuous time | 1.5958 |
| ~~Brownian, sampled as the data is~~ (19 h × 12 five-minute steps) | ~~1.5218~~ — superseded, see §5b-bis |
| **BTC measured** | mean **1.3311**, median **1.2942** |
| IQR | [1.0157, 1.6120] |
| 5–95 pct | [0.5899, 2.1835] |
| ~~mean − sampled benchmark~~ | ~~−0.1907 [−0.2239, −0.1600]~~ — superseded by **−0.2519 [−0.2851, −0.2212]**, §5b-bis |

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

## 6m. Matched calibration: the veto was mine, not the market's

§6l vetoed the refresh and named the follow-up that would settle it — *match
the calibration-slice sizes between arms and re-run* — together with the
verdict each outcome would carry. That is a pre-registration, written before
this run, and it is quoted here so the rule cannot be read as chosen after the
fact:

> If the tail condition still fails on matched calibration, the refresh
> genuinely costs tail accuracy and should stay rejected. If it passes, the
> refresh is the biggest available improvement to the model and should ship.

**It passes.**

`equalize_calib()` truncates both arms to the same number of calibration
episodes — the smaller of the two, kept most-recent-first, so each arm still
calibrates on the data nearest its own window and stays strictly causal. Same
six quarterly windows, same 1,200 test episodes per window drawn with the same
fixed seed, same three seeds, same everything else.

| metric | frozen | refreshed | change | windows won |
|---|---|---|---|---|
| **QLIKE** | 0.238468 | **0.226621** | **−4.97 %** | **5/6** |
| pinball | 0.002606 | **0.002534** | −2.76 % | **6/6** |
| CRPS | 0.003869 | **0.003785** | −2.17 % | **5/6** |
| **deep-tail MCB** | 0.253032 | **0.247655** | **−2.13 %** | **5/6** |
| DSC/UNC | 0.075862 | 0.077377 | +2.00 % | 3/6 |

Pre-registered rule: QLIKE wins ≥ 5/6 **and** deep-tail MCB worse in ≤ 2/6.
QLIKE clears at 5/6; tail MCB is worse in **1** of 6. **ADOPT.**

### The whole veto was one experimental-design error, and it can be measured exactly

The refreshed arm was never truncated — it was always the smaller slice — so
its results are **bit-identical** between the two runs. Only the frozen arm
changed, and only in how much data it calibrated on:

| arm | tail MCB, §6l run | tail MCB, matched | Δ |
|---|---|---|---|
| refreshed | 0.247655 | 0.247655 | **0.000000** |
| frozen | 0.248019 | 0.253032 | **+0.005013** |

Identical network, identical training episodes, identical test episodes,
identical seeds. The only difference is 52,359 calibration episodes against
17,511. That one change is the entire margin the veto rested on. Per window,
the tail delta flips sign in four of six:

| window | Δ tail (§6l) | Δ tail (matched) | Δ QLIKE |
|---|---|---|---|
| 2025-01 | +0.00776 | **−0.00085** | −4.38 % |
| 2025-04 | +0.00191 | **−0.00827** | −4.82 % |
| 2025-07 | −0.00470 | **−0.01302** | −11.51 % |
| 2025-10 | +0.00072 | **−0.00159** | +6.87 % |
| 2026-01 | +0.00293 | +0.00510 | −6.93 % |
| 2026-04 | −0.01081 | **−0.01363** | −14.88 % |

§6l guessed the veto was "very likely an artifact." It was, and the artifact
was mine: I built the arms, gave one of them three times the calibration data,
and then scored them on a miscalibration statistic.

### The finding hiding inside the correction

**Calibration slice size is worth more tail accuracy than the refresh itself.**
Cutting the frozen arm from 52k to 17k calibration episodes cost +0.005013 of
deep-tail MCB. The refresh's own tail gain is −0.005377. These are the same
order of magnitude, and nothing in this project had measured the first one.

That has a direct consequence for what ships. Neither arm above is a
deployable configuration: the frozen arm is stale, and the refreshed arm is
fresh but thinly calibrated. A deployment wants both, and §6n measures that
arm rather than assuming the two effects add.

### What "adopt" does and does not license

**Read §6n before acting on this verdict.** The arm this section adopted is
not deployable, the deployable one was measured next, and it fails the same
rule at 3/6. What survives both is the average-case gain, not the tail claim.

It licenses a **maintenance policy** — retrain on a rolling window instead of
serving a fit frozen at 2023-01-01 — not a change to the model. And it must
not be implemented by moving `splits.TRAIN_END` forward, because every number
in this document is scored on `ts >= CALIB_END`; advancing that boundary would
silently consume the held-out set that makes these tables mean anything. The
research split stays frozen for scoring. The production fit is a separate,
explicitly-dated artifact.

*Educational research only. Not financial advice.*

## 6n. The deployable arm fails the same rule, and that settles a different question

§6m adopted a refreshed fit under matched calibration. That arm cannot ship:
it calibrates on ~17,500 episodes, and §6m itself measured that shrinking the
calibration slice to that size costs +0.005013 of deep-tail MCB. §6m therefore
named this follow-up before running it — *"§6n measures that arm rather than
assuming the two effects add"* — because a deployment wants a fresh training
window **and** a large calibration slice, and nothing had established that the
two gains combine.

`--big-calib` builds that arm: the refreshed fit trains to 18 months before its
window and calibrates over those 18 months, so it is 6–21 months fresher than
the frozen fit while calibrating on 52,359–52,647 episodes against the frozen
arm's 52,359. The frozen arm is left exactly as it ships. Its numbers reproduce
§6l to every printed digit, which is the determinism check for this comparison.

| metric | frozen | refreshed | change | windows won |
|---|---|---|---|---|
| **QLIKE** | 0.238593 | **0.232179** | **−2.69 %** | **5/6** |
| CRPS | 0.003849 | **0.003811** | −1.00 % | 5/6 |
| DSC/UNC | 0.072259 | **0.077620** | **+7.42 %** | 4/6 |
| pinball | 0.002592 | **0.002559** | −1.27 % | 4/6 |
| deep-tail MCB | 0.248019 | **0.245390** | −1.06 % | 3/6 |

QLIKE clears at 5/6. Deep-tail MCB is worse in **3** of 6, one over the
allowance. **DO NOT ADOPT** — the same rule, applied the same way, on the arm
that could actually be deployed.

**I predicted the opposite.** The `--big-calib` docstring, written before the
run, calls this "the arm that could actually ship" on the reasoning that it is
simultaneously fresh and properly calibrated. It is the weakest of the three on
tail calibration and it more than halves the QLIKE gain. Recorded as the failed
prediction it is, alongside `lam_r`, `q_mx`, dropping `reg_post_etf` and the
forward split.

### Three designs, and what actually replicates

| | refreshed trains to | refreshed n_calib | ΔQLIKE | QLIKE wins | tail wins | mean Δtail | verdict |
|---|---|---|---|---|---|---|---|
| §6l | window − 6 mo | ~17.5 k | −5.02 % | 5/6 | 2/6 | −0.00036 | reject |
| §6m | window − 6 mo | ~17.5 k (both) | −4.97 % | 5/6 | **5/6** | −0.00538 | adopt |
| §6n | window − 18 mo | ~52.4 k (both) | −2.69 % | 5/6 | 3/6 | −0.00263 | reject |

**What replicates: the average-case gain.** QLIKE favours the refresh in 5 of 6
windows in every design — 15 of 18 window-comparisons — and the effect is
dose-dependent: the arm trained six months from its window gains ~5 %, the arm
trained eighteen months from it gains ~2.7 %, and the ordering holds
window-by-window (2026-04: −14.88 % against −7.40 %; 2025-01: −4.38 % against
−0.05 %). A monotone dose-response across an independent axis is much harder to
get from noise than a win count is.

**What does not replicate: the tail claim, in either direction.** Per-window
tail deltas flip sign between designs on the same window and the same test
episodes — 2026-04 is −0.01363 under §6m and +0.00832 under §6n; 2025-10 is
−0.00159 and +0.00371. The only stable feature is that the **mean** favours the
refresh in all three designs (−0.00036, −0.00538, −0.00263). The win count,
which is what the rule reads, lands on 2, 5 and 3 out of 6 depending on a
nuisance choice that is not the treatment.

### The rule is what failed here, and I do not get to replace it on this data

The veto was written to catch one thing: *"average-case wins bought with tail
degradation stay rejected."* Across three designs the tail mean never degrades.
The condition the rule exists to catch does not occur — and the rule fires
anyway in two designs out of three, because a 6-sample win count on an effect
of ~0.003 against per-window scatter of ~0.013 is a coin flip with extra steps.

The disciplined consequence is not to re-decide the tail on a better statistic
now. Choosing a rule after seeing which one passes is precisely the failure
this benchmark was built to prevent, and it would be a worse error than the
confound §6m corrected. So:

- **the average-case improvement is established** and is the part of §6m's
  ADOPT that survives — QLIKE, CRPS and pinball, replicated three times, dose-
  dependent;
- **the deep-tail question is open**, not resolved either way, and six windows
  cannot resolve it;
- **no fourth design gets run to break the tie.** Three is already at the edge
  of what can be reported without multiplicity becoming the story.

**Fixed now, for the next measurement, before it is run:** the tail is to be
decided on the *mean* deep-tail MCB delta with a moving-block bootstrap CI
(block length n^(1/3), as `eval/direction.py` uses), on **monthly** rather than
quarterly windows through the unseen era — 18 windows instead of 6 — with the
refreshed arm at `window − 6 mo` and calibration matched. Adopt if the CI
excludes zero on the favourable side, or if it contains zero while QLIKE clears
5/6, since a tail effect indistinguishable from zero is not degradation. That
rule is recorded here before that data is scored.

Until then the shipped configuration stays frozen, and `train_v2.py --train-end
/--calib-end` exists so that refreshing production is an explicit, dated act
rather than an edit to `splits.TRAIN_END` that would consume the held-out set
every table above is scored on.

*Educational research only. Not financial advice.*

## 5b-bis. The discretization correction in §5b was itself wrong

§5b corrected an overstated shape gap and, in doing so, understated it. Both
attempts are recorded because the second one was published for several
commits and the reasoning that produced it looked sound.

**Attempt 1** compared BTC's `range / sqrt(realized variance)` against the
continuous-time constant `sqrt(8/pi) = 1.5958` and reported **−0.2646**.

**Attempt 2** objected — correctly in principle — that a running maximum
observed at finitely many points is always below the true continuous maximum,
so comparing a discretely-sampled measurement against a continuous constant
overstates the gap. It simulated 5-minute closes, obtained **1.5218**, and
revised the gap to **−0.1907**, claiming ~28 % of the original was
discretization.

**The premise was false.** The measured excursions do not come from 5-minute
closes. `episodes.build_hourly` sets `hour_high = max of the 1-minute bar
HIGHS`, and `M_up` is a running max over those hourly highs — so the numerator
inherits intra-minute tick extremes and sits very close to the *continuous*
running maximum. The 5-minute object is the **denominator** alone,
`RV = sqrt(sum of rv5)`. Attempt 2 sampled the numerator four resolutions too
coarse, which biases the benchmark down and the gap toward zero — it made BTC
look **more** Brownian than it is.

Measured, n = 60,000 simulated 19-hour paths, RV always at 5 minutes:

| extremes sampled at | benchmark ratio |
|---|---|
| 5-minute closes | 1.5190 | 
| 1-minute closes | 1.5605 |
| **10-second closes** | **1.5860** |
| continuous limit `sqrt(8/pi)` | 1.5958 |

`brownian_control()` now samples extremes at 10 seconds and RV at 5 minutes,
giving **1.5831** at the production horizon. That default is deliberately
**conservative**: real 1-minute bar highs are built from ticks, so the true
benchmark sits nearer 1.5958, and 1.5831 understates the gap rather than
flattering it. The residual ambiguity is ~0.01 against a gap of ~0.25.

### The corrected measurement

| quantity | value |
|---|---|
| Brownian benchmark, sampled as the data is | **1.5831** |
| BTC measured mean | 1.3311 (median 1.2942, IQR [1.0157, 1.6120]) |
| **gap** | **−0.2519**, block-bootstrap 95 % CI **[−0.2851, −0.2212]** |
| against the continuous constant | −0.2646 |

The published interval [−0.2239, −0.1600] and the correct one [−0.2851,
−0.2212] barely overlap, so this is a material change, not a rounding
adjustment: **the shape gap is ~32 % larger than this document has been
claiming.** BTC travels **15.9 % less per unit of realized volatility than a
Brownian path**, not 12.5 %.

Nothing downstream reverses. Every conclusion built on §5b — that BTC chops,
that a Gaussian first-passage law fed a correct sigma **overstates** touch
risk, that a seller using one quotes strikes further out than necessary — is
strengthened, not weakened. The oracle experiment is unaffected because it
never used the benchmark: a perfect volatility forecast still removes only
**6.0–8.5 %** of the barrier error, and the Gaussian law fed realized
volatility still overstates the touch probability at every barrier
(+5.40, +8.68, +8.83, +7.22, +4.63 pp at 0.5/1/2/3/5 %).

The methodological point is the one worth keeping: a correction is not
self-verifying because it moves a number in the humbler direction. Attempt 2
was more careful than attempt 1 and still wrong, because it never checked
which column the data's extremes actually came from.

*Educational research only. Not financial advice.*

## 7. Where the error lives, and whether the network learns

Two audits run in parallel, both re-derived against the data before being
written down. Where an agent's framing was wrong, that is recorded too.

### 7a. The model reacts to volatility; it does not anticipate it

`eval/anatomy.py` scores the **shipped** artifact (`serve.runtime.load_model()`,
no retraining) over 18,463 served test episodes, cross-checked on the 769
non-overlapping 17:00 episodes.

| population | share of episodes | share of QLIKE loss | **median RV / σ** |
|---|---|---|---|
| normal | 92.3 % | 74.2 % | **0.964** |
| **volatility spike** | **7.7 %** | **25.8 %** | **1.453** |

Spike episodes are flagged causally — a trailing 180-day 95th percentile of
production RV using strictly prior days. **On the 7.7 % of nights that carry a
quarter of the loss, the model under-forecasts volatility by 45 %.** For an
option seller that is the expensive direction: the strike breaks.

`RV/σ` is a ratio, so this does not depend on the scoring rule. That matters,
because a second reported finding does:

> **A framing correction.** The audit reported "the worst 100 episodes are
> 100 % under-forecasts, against a 48.4 % base rate" as evidence of directional
> bias. It is mostly an artifact of QLIKE. `QLIKE = log σ² + RV²/σ²` diverges as
> σ → 0 but grows only logarithmically as σ → ∞, so the extreme tail of any
> QLIKE ranking is dominated by under-forecasts even for a symmetric model.
> Measured directly: at a factor-2 error the penalty ratio is **1.60×**, at
> factor-4 it is **4.67×**. The claim is kept, demoted to corroboration. The
> load-bearing number is the scale-free `RV/σ = 1.453`.

The mechanism is a **one-day lag**, visible on every stress cluster in the test
era:

| date | realized 19 h RV | predicted σ | vs the model's own median |
|---|---|---|---|
| 2024-08-04 | 9.40 % | 2.31 % | 1.29 × |
| 2024-08-05 | 10.58 % | 5.32 % | 2.96 × |
| 2025-01-19 | 7.50 % | 2.18 % | 1.21 × |
| 2025-01-20 | 7.02 % | 3.98 % | 2.22 × |
| 2026-02-05 | 8.35 % | 3.92 % | 2.18 × |
| 2026-02-06 | 6.55 % | 4.72 % | 2.63 × |
| **2025-04-06** | **5.78 %** | **1.56 %** | **0.87 ×** |

On the first day of each cluster the model sits near its unconditional median;
by the second it has widened 2–3×. On 2025-04-06 — a top-10 volatility day — it
was **more confident than usual**. Lead/lag correlation of predicted σ against
realized RV: **0.920** dated one day *after* the realization, **0.507** one day
*before*.

> **WITHDRAWN — see §12.** The paragraph below asserts a mechanism that was
> never tested. It is wrong: a spike-tomorrow classifier clears its nulls
> (AUC 0.777), and on onset days specifically reaches 0.733 with a CI of
> [0.6547, 0.8052]. The information is there; the model was not using it.
> The original text is kept because this document is a record.

This is structural rather than a bug: every input the model has is a trailing
statistic, so a trailing forecast is what the feature set can express. It is
also the single most decision-relevant limitation for a seller, and it is now
measured rather than suspected.

### 7b. The US-Iran hypothesis is not supported by the data

The project has been working from the premise that the June 2025 US-Iran
escalation re-created pre-ETF fat-tail volatility, and that a model failing to
widen for it would be failing the real test. Measured on production anchors:

| | 19 h realized volatility |
|---|---|
| June 2025 window, peak | **2.54 %** (2025-06-12) |
| June 2025 window, median | 1.17 % |
| test era median | 1.66 % |
| test era p90 / p95 / p99 | 2.89 % / 3.62 % / 5.02 % |
| test era max | 7.84 % (2026-02-05) |

**The June-2025 peak sits at the 84.4th percentile of the test era — below even
the 90th.** For BTC at a 19-hour horizon it was not a stress event, and its
in-window median (1.17 %) is *below* the era's median. The real stress events in
this period are 2024-08-05, 2025-01-19/20, 2026-02-05 and the 2025-10-10 flash
crash. The premise was reasonable and it is simply not what the data shows;
§7a's lag finding stands on those events instead, which is a stronger test
because they are larger.

### 7c. The network does learn — and the OLS seed is not just a warm start

`eval/learning.py`, 1 fold, 1 seed, measured at the **restored best checkpoint**
that actually ships. (The first pass measured a 40-epoch model; `train.py`
lines 235-246 keep `best_state` and call `load_state_dict(best_state)`, so the
deployed weights are the epoch-6 checkpoint and diagnostics on a run-out model
describe nothing that exists.)

Per head, epoch 0 (the Log-HAR/OLS seed) versus the shipped checkpoint:

| head | epoch 0 | best (epoch 6) | change |
|---|---|---|---|
| `mx` max excursion | 0.13105 | 0.08381 | **−36.0 %** |
| `r` terminal return | 0.22791 | 0.15748 | −30.9 % |
| `dn` downside | 0.13214 | 0.09916 | −25.0 % |
| `up` upside | 0.13290 | 0.10702 | −19.5 % |
| `a` volatility | 0.07709 | 0.06502 | −15.7 % |

Every head beats its linear initialization at the checkpoint that ships. The
network is doing real work, not decorating an OLS fit.

**Early stopping is load-bearing, and it works.** Validation bottoms at epoch 6
(0.512490) and degrades monotonically to 0.565905 by epoch 39 — the `a` head
ends 17 % *worse than its own initialization* if left to run. Because the
trainer restores the best state, none of that ships. This also settles whether
more epochs would help: they would not, and the 150-epoch arm was cancelled
rather than spending hours confirming a direction the data already fixed.

**Initialization, both arms early-stopped** — the only fair comparison, since
comparing two overfit models answers a different question:

| arm | best val | at epoch |
|---|---|---|
| OLS-seeded | **0.512490** | 6 |
| random init | 0.540947 | 12 |

A **5.55 %** gap. Random-init finds a real minimum of its own, so the network is
not merely inheriting OLS — but it does not close the gap in the shipped budget
either. The seed buys a better basin, not just a faster start.

**Capacity is not fully used.** Effective rank of the hidden activations at 95 %
of variance, out of a nominal width of 32:

| layer | dead units | effective rank |
|---|---|---|
| stage A, layer 1 | 0 % | 17 / 32 |
| stage A, layer 2 | 0 % | 12 / 32 |
| stage B, layer 1 | 0 % | 13 / 32 |
| stage B, layer 2 | **21.9 %** | **9 / 32** |

> **WITHDRAWN — see §11a.** The ranks below are compared against a nominal
> width of 32. Stage B's first layer is `Linear(22 -> 32)`, so 32 is an
> unreachable denominator, and the real ceiling is the data: Stage A's input
> spans 19 dimensions and Stage B's spans 13, against achieved ranks of 17
> and 13. That is saturation, not underuse.

Three of four layers behave like a much narrower network, and Stage B's second
layer has ~22 % literally dead units. Weight magnitudes are unremarkable
(0.097–0.113), so the redundancy is in rank, not in weights collapsing to zero.
This is consistent with §3's capacity study, which found more width does not
help — the model is not capacity-starved, which is another way of saying more
parameters are not the missing ingredient.

**`reg_post_etf` confirmed, and it is the only one.** Autograd on every
minibatch of all 40 epochs: its weight column receives *exactly* zero gradient
in both stages, reproducing §6i at full scale. All 39 inputs were scanned and
**no second constant feature exists** (next-lowest std 0.041). The defect is
isolated, not systemic.

*Educational research only. Not financial advice.*

## 6o. Eighteen windows, the replacement rule, and an ADOPT that holds

§6n fixed a rule before this data was scored, because three six-window designs
had produced 2/6, 5/6 and 3/6 on the same deep-tail effect and a win count on
~0.003 against per-window scatter of ~0.013 is a coin flip. The replacement:
monthly windows, the refreshed arm at `window − 6 months`, calibration matched,
and the tail decided on the **mean** delta with a moving-block bootstrap CI
rather than on how many windows fell each way.

Eighteen windows were attempted; **16 produced both arms**. February 2025 and
February 2026 were skipped identically for both arms (train/calib too small),
so the comparison stays paired and the exclusion is not selective.

| metric | frozen | refreshed | change | windows won |
|---|---|---|---|---|
| **QLIKE** | 0.239715 | **0.228761** | **−4.57 %** | 13/16 |
| pinball | 0.002524 | **0.002432** | −3.65 % | 13/16 |
| CRPS | 0.003758 | **0.003662** | −2.54 % | 14/16 |
| DSC/UNC | 0.078198 | **0.085689** | **+9.58 %** | 11/16 |
| **deep-tail MCB** | 0.223847 | **0.217903** | **−2.66 %** | 11/16 |

**Mean deep-tail MCB delta −0.005943, moving-block 95 % CI [−0.011755,
−0.002035]** — excludes zero on the favourable side. Mean QLIKE delta
−0.010954, one-sided sign test p = 0.01064.

**Verdict: ADOPT**, on the rule's first branch — the tail CI excludes zero
favourably.

### Being exact about what passed and what did not

The QLIKE win count is **13/16 against a bar of 14** (the same 83.3 % rate as
§6n's "5/6", read as a rate). Under the rule's *second* branch this would not
adopt. Adoption rests entirely on the tail interval, which is how the rule was
written — the tail was the veto condition, and a tail that is not merely absent
but significantly favourable is the strongest outcome available. It would be
misleading to report this as both conditions passing, so it is not.

The count wobbling again while the mean is decisive (−4.57 %, p = 0.01064) is
the same pathology §6n diagnosed, showing up one more time. That is a point in
favour of having replaced the statistic rather than the threshold.

### What was adopted, and what deliberately was not

`train_v2.py --train-end 2026-02-09 --calib-end 2026-08-09` produces a dated
production fit: **298,791 training episodes against the frozen split's 189,831**
(+57 %), 17,223 calibration episodes — the same slice size the experiment
validated — and 6,939 parameters per seed. It is written to
`serve/noctua_v2_refreshed_2026-08-09.npz`.

**`serve/noctua_v2.npz` was NOT overwritten, and that is a decision, not an
oversight.** `serve.runtime.load_model()` is what the evaluation code loads:
`eval/anatomy.py` scored the shipped artifact over test episodes. An artifact
retrained through 2026-08-09 has those episodes *in sample*, so overwriting the
default path would silently invalidate every evaluation that reads it, with no
error and no visible symptom. The frozen artifact stays as the research
instrument; the refreshed one is dated in its filename and in its own metadata
(`train_end`, `calib_end`, `n_train`, `n_calib`, `frozen_research_split:
false`), so the two can never be confused for one another.

The architecture question raised in `AUDIT.md` §7 resolves itself here: the
refresh was measured through `run_fold` → `train_model`, i.e. on the **current
four-head source model**, so shipping the refresh ships the architecture the
experiment actually validated rather than a different one.

### A dead guard, found by exporting the artifact

The refreshed artifact exports 6,939 params/seed — it carries `q_mx` — yet
`runtime.has_mx()` still reported `False`. Cause: it tested for
`b.q_mx.weight`, **a key that cannot exist for any artifact**. `q_mx` is a
`MonotoneQuantileHead`, so its `state_dict` keys are `b.q_mx.median.weight`,
`b.q_mx.up.weight`, `b.q_mx.dn.weight`; the head has no bare `.weight`. The
guard returned `False` unconditionally — a constant wearing a guard's name —
and would have silently dropped the head from serving on any artifact carrying
it.

It was harmless only by luck: the deployed artifact genuinely lacks the head,
and nothing in `serve/predict.py` consumes `q_mx`. Fixed, and `test_serving.py`
now has four regression checks including one asserting the impossible bare-key
form is *not* treated as present. This is the third defect of this exact
shape in the repo's history (metadata disagreeing with weights; `feat_cols`
42-vs-39), and the first caught by a test rather than by an outage.

*Educational research only. Not financial advice.*

## 8. Priority 1 returns a negative — and the instrument that posed it is broken

`ROADMAP.md` Priority 1 asked whether the conditional variation in path shape is
predictable, with a two-part rule fixed before the run: clear a shuffled
permutation null **and** move the 2 % touch probability by ≥ 1 pp. The answer is
no, and finding that out invalidated the framing that produced the question.

### The agent's verdict did not survive checking

`eval/shape.py` reported "predictable and useful" on all four arms. Two of its
three criteria fail:

**Clearing the null was read backwards.** The gradient-boosted arms have
out-of-sample **R² = −0.016** (range) and **−0.021** (max) — *worse than the
constant they are scored against*. They "clear" only because the permutation
null sits at −0.047 / −0.040. Beating a null centred well below zero means
"less bad than fitting shuffled targets", not "predictive".

**No arm has an R² interval excluding zero favourably**: ridge range
[−0.0038, +0.0168], ridge max [−0.0066, +0.0116], gbdt range [−0.0364,
+0.0051] all contain zero; gbdt max [−0.0397, −0.0015] excludes zero on the
*wrong* side.

**The relevance test was an absolute value, so noise passes it.** Reported mean
|Δ| in P(touch 2 %) is 1.05–2.53 pp while the mean **signed** Δ is 0.01–0.40 pp.
A random number generator scores well on `mean |Δ|`. ROADMAP asked whether shape
"moves the 2 % touch probability by ≥ 1 pp" and meant *improves*; the ambiguity
was in the rule, the correct reading is not ambiguous.

What is real is a weak rank association: Spearman **+0.0795** (p = 0.028), R²
**+0.019** on 769 held-out production episodes.

### The test that matters, and it is a clean negative

`eval/shape_relevance.py`. Both arms fed the **realized** volatility, so shape
is the only difference; Brier on the realized touch indicator; moving-block
bootstrap CI on the paired difference.

| barrier | realized | const shape | predicted shape | Δ | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| 0.5 % | 0.9844 | 0.01503 | 0.01502 | −0.00001 | [−0.00007, +0.00005] | no effect |
| 1.0 % | 0.8518 | 0.12672 | 0.12756 | +0.00084 | [+0.00044, +0.00114] | **HURTS** |
| 2.0 % | 0.4512 | 0.40217 | 0.40866 | +0.00649 | [+0.00504, +0.00794] | **HURTS** |
| 3.0 % | 0.2562 | 0.46876 | 0.47959 | +0.01083 | [+0.00867, +0.01300] | **HURTS** |
| 5.0 % | 0.0702 | 0.42747 | 0.44426 | +0.01679 | [+0.01396, +0.01954] | **HURTS** |

Conditioning on predicted shape makes the touch forecast **significantly worse
at every barrier a seller cares about**. A Spearman of 0.08 is real, and it is
far too weak to survive being multiplied into a variance: the noise it injects
costs more than the ordering it recovers.

**Verdict against the pre-registered rule: shape is NOT predictable-and-useful.**
Condition (a) is arguable on rank; condition (b) fails decisively once read as
intended. Recorded as the fifth failed prediction in this project, alongside
`lam_r`, `q_mx`, dropping `reg_post_etf` and §6n's big-calib arm.

### The larger finding: the diagnostic loses to a constant

The comparison needs a floor, so the base rate was scored too — a forecaster
ignoring every input:

| barrier | Gaussian + oracle vol + fitted shape | base rate | ratio |
|---|---|---|---|
| 0.5 % | 0.01503 | 0.01536 | 0.98 × |
| 1.0 % | 0.12672 | 0.12627 | 1.00 × |
| **2.0 %** | 0.40217 | 0.24762 | **1.62 ×** |
| **3.0 %** | 0.46876 | 0.19055 | **2.46 ×** |
| **5.0 %** | 0.42747 | 0.06529 | **6.55 ×** |

**Given perfect volatility, the Gaussian first-passage law is up to 6.55×
worse than quoting the historical frequency.** By this benchmark's own Rule 1
that is a forecaster with negative skill.

This reaches back into §5b. `eval/firstpassage.py` computes "a perfect
volatility forecast removes only 6.0–8.5 % of the barrier error" by comparing
that law fed realized volatility against the same law fed a causal forecast.
Both arms are the broken instrument, so **the 92 % figure is a property of the
Gaussian first-passage forecaster and cannot be attributed to NOCTUA.**
`ROADMAP.md` attributed it to "the model". That was an overreach, made by me,
and it is corrected there.

**NOCTUA never used that law.** `infer.touch_prob` reads barrier probabilities
off the learned Stage B quantile heads through `survival_from_quantiles`; the
only `norm_cdf` in serving is a fallback. And the learned heads do beat the
constant — at the 2 % up barrier, Brier = MCB − DSC + UNC = 0.01230 − 0.01747
+ 0.20124 = **0.19607** against climatology's 0.01542 − 0 + 0.20124 =
**0.21666**, a 9.5 % improvement. The architecture was already right on the
point this whole line of work was set up to investigate.

### What actually survives

- The direct measurement stands: BTC's range / √RV is **1.3311** against a
  Brownian **1.5831** (§5b-bis). It is a measurement, not a model output, and
  does not depend on the law.
- "Stop investing in volatility because 92 % is shape" **does not follow** and
  is withdrawn.
- The honest replacement question is the same oracle experiment run through
  **NOCTUA's own mapping** — feed Stage B `log(realized RV)` in place of the
  forecast σ and measure how much barrier error that removes. That is the
  decomposition ROADMAP should have specified, it is cheap, and it is
  specified here rather than run because this session's budget ended.

*Educational research only. Not financial advice.*

## 9. The oracle decomposition, redone through NOCTUA's own mapping

§8 withdrew the "92 % of barrier error is shape" figure because it was computed
through the Gaussian first-passage law, which loses to the base rate. This is
the same question asked through the mapping NOCTUA actually uses:
`infer.touch_prob` over the learned Stage B quantile heads.

### The design, and the confound it has to avoid

Stage B is conditioned on `log_sigma` and integrated over 32 sigma ATOMS from
Stage A. Swapping those atoms for a point mass at the realized RV changes two
things at once — the LOCATION of sigma (accuracy) and the SPREAD over atoms
(the model's uncertainty about sigma). Reporting their sum as "what better
volatility forecasting would buy" overstates it, because no forecaster gets the
second for free. So three arms, both oracle comparisons single-atom:

| arm | sigma |
|---|---|
| `committee_32atom` | 32 atoms from Stage A — what ships |
| `point_forecast` | ONE atom at Stage A's median |
| `point_oracle` | ONE atom at the REALIZED RV |

769 held-out production episodes, Brier on the realized touch indicator
(averaged over the up and down barriers), with the historical base rate as a
floor.

| barrier | realized | 32-atom | point-forecast | **point-oracle** | base rate | addressable | 95 % CI |
|---|---|---|---|---|---|---|---|
| 0.5 % | 0.7464 | 0.18955 | 0.18973 | 0.18744 | 0.19506 | 1.2 % | [−0.00639, +0.00183] |
| 1.0 % | 0.5150 | 0.23682 | 0.23589 | 0.22697 | 0.24989 | 3.8 % | [−0.01275, −0.00402] |
| 2.0 % | 0.2224 | 0.17279 | 0.17046 | **0.15343** | 0.17980 | **10.0 %** | [−0.01981, −0.01387] |
| 3.0 % | 0.1144 | 0.10834 | 0.10650 | **0.09321** | 0.11242 | **12.5 %** | [−0.01593, −0.01058] |
| 5.0 % | 0.0221 | 0.03432 | 0.03290 | **0.02744** | 0.03371 | **16.6 %** | [−0.00757, −0.00324] |

**Perfect volatility knowledge removes 10–17 % of the barrier error at the
strikes a seller actually sells**, and the CI excludes zero at every barrier
from 1 % out. The share **rises with barrier distance** — 1.2 % at 0.5 % out to
16.6 % at 5 % — where the discredited Gaussian decomposition reported a flat
6–8.5 %. Deep barriers depend on the volatility level; near ones are decided by
path detail.

This does not resurrect "volatility is the whole game": 83–90 % of the error at
those barriers still is not the volatility level. But "a perfect volatility
forecast buys almost nothing" was wrong, and the correction matters most
exactly where the money is.

### Two things this turned up that were not the question

**1. At the 5 % barrier NOCTUA is slightly worse than the base rate.** Brier
0.03432 against 0.03371. A small margin on a rare event (realized 2.21 %), but
by this benchmark's Rule 1 a forecaster losing to a constant is a red flag, and
it is recorded rather than left in a table for someone else to notice.

**2. Collapsing the 32 sigma atoms to a single point at the median IMPROVES the
barrier forecast** at 4 of 5 barriers, and roughly halves calibration error:

| barrier | calibration error, 32-atom | point-forecast |
|---|---|---|
| 1.0 % | 3.02 pp | **1.61 pp** |
| 2.0 % | 5.81 pp | **3.09 pp** |
| 3.0 % | 4.59 pp | **1.47 pp** |
| 5.0 % | 4.00 pp | **1.48 pp** |

The atom integration is supposed to propagate Stage A's uncertainty into
Stage B. On this evidence it is instead inflating touch probabilities, which is
what Jensen's inequality predicts: `P(touch)` is convex in sigma over the
relevant range, so averaging over a spread of sigmas exceeds the probability at
the mean sigma. That is a candidate mechanism for the over-forecast bias
documented in §7a and papered over by `serve/adaptive.py`'s nightly shrink.

**This is one split of 769 episodes and is NOT adopted on that basis.**

**DECISION RULE, fixed here before the test is run:** collapse the atoms only
if, on the wide slice (every H = 19 anchor hour, ~24× the episodes) split by
year, the single-atom arm (a) improves mean Brier at the **2 %** barrier with a
block-bootstrap CI excluding zero, (b) does not worsen mean Brier at **any** of
0.5/1/3/5 %, and (c) wins the 2 % barrier in at least 5 of 6 years. Condition
(b) is deliberately strict: the atom spread exists to represent genuine
uncertainty, and an average-case win bought by removing a safety margin at one
barrier is the wrong trade for a seller.

*Educational research only. Not financial advice.*

## 10. The sigma-atom finding was a small-sample fluke, and my mechanism was backwards

§9 observed that collapsing Stage B's 32 sigma atoms to a point at the median
improved the barrier forecast on 769 production episodes, proposed Jensen's
inequality as the mechanism, and fixed a rule before testing it. The rule says
**DO NOT ADOPT**, on every condition. Both of §9's claims are wrong.

### The empirical finding reverses on 24× the data

Wide slice: every H = 19 anchor hour in the test split, 18,463 episodes against
769. Brier on the realized touch indicator, moving-block bootstrap CI, and the
narrow-slice numbers replicated bit-for-bit first to confirm the pipelines are
comparable.

| barrier | 32-atom | single-atom | single − 32-atom | 95 % CI | winner |
|---|---|---|---|---|---|
| 0.5 % | 0.18419 | 0.18443 | +0.00023 | [+0.00007, +0.00040] | **32-atom** |
| 1.0 % | 0.23404 | 0.23492 | +0.00089 | [+0.00050, +0.00129] | **32-atom** |
| **2.0 %** | **0.18791** | 0.18913 | **+0.00122** | **[+0.00059, +0.00189]** | **32-atom** |
| 3.0 % | 0.11907 | 0.11941 | +0.00035 | [−0.00028, +0.00101] | tie |
| 5.0 % | 0.03877 | 0.03830 | −0.00047 | [−0.00085, −0.00007] | single |

**At the 2 % barrier the sign flips**: single-atom was better on 769 episodes
and is significantly worse on 18,463, with a CI excluding zero. Per year at
2 %, single-atom wins **0 of 3**. The 32-atom integration is doing real work
that the narrow slice was too small to show.

### The mechanism was backwards

§9 asserted P(touch) is convex in sigma "over the relevant range", so averaging
over atoms would exceed the value at the mean and inflate touch probabilities.
Measured directly — average of the 32 atoms' own touch probabilities against
the probability at a single atom placed at the mean atom sigma:

| barrier | avg-of-32 | at-mean | gap | 95 % CI | curvature |
|---|---|---|---|---|---|
| 0.5 % | 0.7404 | 0.7675 | **−0.0271** | [−0.0280, −0.0262] | **concave** |
| 1.0 % | 0.5137 | 0.5464 | **−0.0328** | [−0.0334, −0.0321] | **concave** |
| 2.0 % | 0.2694 | 0.2861 | **−0.0168** | [−0.0176, −0.0160] | **concave** |
| 3.0 % | 0.1568 | 0.1601 | −0.0033 | [−0.0041, −0.0026] | concave |
| 5.0 % | 0.0650 | 0.0575 | +0.0075 | [+0.0072, +0.0077] | convex |

Concave at four of five barriers, convex only at 5 %. `P(touch)` is an S-curve
in sigma — it saturates toward 1 as sigma grows, so it is **concave** wherever
the touch probability is already substantial, and convex only deep in the tail
where the probability is near zero. At 0.5–3 % the model sits in the concave
region, so the atom integration **deflates** touch probability relative to the
value at the mean. That is the opposite of the over-forecast mechanism §9
proposed, and it means the integration is if anything a partial *counterweight*
to the over-forecasting in §7a, not its cause.

### The rule had an unsatisfiable condition, and I wrote it

Condition (c) required winning "at least 5 of 6 years". The test split begins
2024-07-01 and the data ends 2026-08-09, so **only three calendar years exist**
— 5 of 6 cannot be reached without scoring the model on data it trained on. The
condition could only ever fail.

That is a distinct failure mode from the ones already catalogued: at application
time an unsatisfiable condition is indistinguishable from a failed one, and gets
reported as evidence against the hypothesis. It is now
`research/pitfalls.check_rule_satisfiable`, to be run when a rule is **written**
rather than when it is applied. It was found by the agent executing the rule,
not by its author — who had, at that point, already been burned twice by rule
design in the same session.

**Verdict: DO NOT ADOPT.** (a) fails — 2 % is significantly worse. (b) fails at
0.5, 1, 2 and 3 %. (c) is unsatisfiable. The shipped 32-atom integration stays.

The value here is entirely negative and entirely real: a promising result on 769
episodes, a plausible mechanism, and a clean rule, all three of which dissolved
on contact with 24× the data.

*Educational research only. Not financial advice.*

## 11. Three defects: one false alarm, one real-but-negligible, one open

### 11a. "Capacity is underused" — WITHDRAWN, it was a denominator error

§7c reported hidden-layer effective ranks of 17/12/13/9 against "a nominal
width of 32" and concluded that three of four layers "behave like a much
narrower network". That comparison is wrong twice over.

**The denominator is unreachable.** Stage B's first layer is
`Linear(22 -> 32)` — 21 shape columns plus `log_sigma`. A linear map from 22
dimensions cannot have rank above 22, so reporting "13 / 32" measures an
achieved rank against a ceiling the architecture forbids.

**And the real ceiling is the data, not the width.** Effective rank at 95 % of
variance, measured on the standardized inputs before any weight is applied:

| | activation rank (§7c) | input spans | architectural ceiling |
|---|---|---|---|
| Stage A layer 1 | 17 | **19** | 32 |
| Stage B layer 1 | 13 | **13** | 22 |

**Stage B carries exactly as many directions as its input contains, and Stage A
is within two of its input's rank.** A layer cannot represent more directions
than its input has. This is saturation, not underuse.

The finding is withdrawn. It is also consistent with §3's capacity study, which
found more width does not help — for the reason now measured: the inputs span
~13–19 dimensions, so width beyond that has nothing to carry. The 21.9 % dead
units in Stage B's *second* layer stand as measured, but in a layer whose input
already spans 13 dimensions they are not evidence of a training pathology.

### 11b. `reg_post_etf` — a real defect that moves almost nothing

The flag is identically 0 across 189,831 training rows and identically 1 in
test, so its first-layer weight never leaves random initialisation, and it
enters production at ordinary magnitude. §6i established that. The question
nobody had asked is **how much the live forecast actually moves**, which is
different from whether removing it improves a fold average (measured at 2–3 of
6 and reported as a null).

Measured on the shipped artifact over 769 held-out production episodes, zeroing
the flag:

| quantity | median abs change | mean abs change | 5–95 % |
|---|---|---|---|
| `sigma_med` | **0.169 %** relative | 0.201 % | [−0.489 %, +0.153 %] |
| P(touch 2 %, up) | **0.167 pp** | 0.205 pp | — |
| P(touch 2 %, dn) | **0.286 pp** | 0.292 pp | — |

**A real bug with negligible practical impact.** An untrained random offset is
moving live forecasts by about a fifth of a percent of sigma and under a third
of a percentage point of touch probability. It should be removed at the next
retrain — a feature that is constant in training carries no information by
construction, so there is nothing to lose — but it is not an emergency, and the
earlier "null" verdict was not hiding a live problem.

### 11c. The adaptive correction: the bias is hour-shaped, but the fix is not established

`serve/adaptive.py` applies one global shrink nightly. §7a found the
over-forecast bias is localised — median RV/sigma 0.85–0.94 at hours 15–22 UTC
against 0.99–1.06 at hours 0–13. Walk-forward, causal, same [0.70, 1.40] clip
and `MIN_EPISODES = 20` as production:

| | peak bucket (15–22 UTC) cal. error | overall |
|---|---|---|
| no correction | 0.01124 | 0.00663 |
| global (what ships) | 0.01087 | — |
| **hour-conditional** | **0.00602** | 0.00683 |

−44.6 % on the peak bucket, +3.1 % overall, and median RV/sigma bias nearly
eliminated (peak 0.897 → 0.998; the global correction barely moves it, 0.897 →
0.906). The agent's pre-registered rule fires **ADOPT (targeted)**.

**I am not adopting it, and the rule is not what decides that.** Two things the
rule did not ask about:

1. **Its own CI contains zero**: the bootstrap interval on the peak-bucket gain
   is [−0.0071, +0.0014] at n = 4,000. The point estimate is large and the
   mechanism is clean, and neither of those is significance.
2. **The hour pools are thin.** Median pool size is **60 episodes per hour
   bucket against 1,422 for the global correction** — a 24× reduction in the
   data estimating a multiplier that scales every published forecast.

This project rejected an almost identical pattern earlier today: §10's
sigma-atom collapse had a clean mechanism and a good point estimate on a small
slice, and **reversed sign** on 24× the data. A 60-episode pool is exactly that
exposure. The finding is recorded as promising and the experiment as
under-powered; the rule that would settle it is a wider evaluation, not a
tighter argument.

### 11d. An unplanned finding: the shipped correction makes QLIKE worse

Not what the experiment was looking for, and it concerns live behaviour:

| arm | mean QLIKE | vs no correction |
|---|---|---|
| no correction | 0.24581 | — |
| **global (ships today)** | **0.25194** | **+0.00613**, CI [0.0030, 0.0094] |
| hour-conditional | 0.25289 | +0.00708 |

**The correction that has been running in production significantly worsens mean
QLIKE**, with a CI excluding zero unfavourably. This is expected once
decomposed — QLIKE punishes under-forecasts asymmetrically (measured in §7a at
1.60× for a factor-2 error, 4.67× at factor-4), so a median-targeting shrink is
not QLIKE-optimal — and the correction was never justified on QLIKE; it was
justified on barrier calibration, which it does improve. But it is a real
trade-off that was being made silently, and it is now on the record: the
nightly shrink buys calibration and pays for it in sharpness.

*Educational research only. Not financial advice.*

## 12. The one-day lag is NOT an information ceiling — my §7a claim was wrong

§7a measured that NOCTUA under-forecasts spike nights by 45 % (median RV/sigma
1.453) while those nights carry 25.8 % of total loss, and explained it as
**structural**: "every input the model has is a trailing statistic, so a
trailing forecast is what the feature set can express." That explanation was
asserted, not tested. Tested, it is wrong.

### The information is there, and it clears its nulls

A spike-tomorrow classifier on the causal features, walk-forward, scored
against a shuffled-target permutation null as `eval/direction.py` does. Every
non-constant arm clears both an AUC null and a DSC null on both the production
(n ≈ 2,245) and wide (n ≈ 48,840) populations, and beats the causal constant
with a bootstrap CI excluding zero:

| slice | model | AUC | AUC null p95 | DSC | DSC null p95 |
|---|---|---|---|---|---|
| production | logistic | 0.7771 | 0.5437 | 0.004718 | 0.000497 |
| production | gbdt | 0.7480 | 0.5502 | 0.005302 | 0.000547 |
| wide | logistic | 0.7775 | 0.5073 | 0.004263 | 0.000031 |
| wide | gbdt | 0.7590 | 0.5091 | 0.005001 | 0.000028 |

The constant does not clear (AUC 0.4841), behaving as the null it should be.
The signs are unambiguous, so this is not the "clears a null centred below
zero" trap that §8 caught.

### But the decomposition that matters was not done, and it changes the size of the prize

Volatility clusters. A spike-tomorrow classifier can score well purely on
persistence — "today is wild, so tomorrow probably is" — while still being
caught flat on **day one of a cluster**, which is precisely the failure mode
§7a documented. So the test that decides this is not overall AUC; it is AUC on
**onset** days (spike today, calm yesterday) against **continuation** days.

Re-derived independently, production episodes, causal 180-day trailing
95th-percentile flag, 303 spike days of which 177 onset and 126 continuation:

| the classifier is asked | AUC | n positives |
|---|---|---|
| all spike days | 0.8055 | 76 |
| **continuation** (already underway) | **0.9293** | 28 |
| **onset** (first day of a cluster) | **0.7332** | 48 |

Onset: bootstrap 95 % CI **[0.6547, 0.8052]**, excluding 0.5, and clearing a
shuffled-label null whose p95 is 0.5642.

**So the anticipatory signal is real, and it is much weaker than the headline
suggests.** Persistence does most of the work (0.929); genuine anticipation of
a cluster's first day sits at 0.733. Both facts matter: §7a was wrong that the
lag is a ceiling, and a plan built on the 0.78 headline would over-promise,
because the days that cost the most are the 0.73 days.

### Two levers, measured, neither adopted yet

**Asymmetric upweighting of high-volatility training episodes** (single seed,
served population n = 18,463; the 1.0× control reproduces §7a's shipped numbers
closely, 0.2498 against 0.2458 pooled QLIKE, which is what licenses the deltas):

| weight | pooled QLIKE | spike QLIKE | normal QLIKE | spike RV/sigma |
|---|---|---|---|---|
| 1.0× | 0.2498 | 0.8567 | 0.1989 | 1.4643 |
| 3.0× | **0.2358** (−5.6 %) | **0.6344** (−25.9 %) | 0.2024 (+1.7 %) | 1.3484 |
| 8.0× | **0.2340** (−6.3 %) | **0.4195** (−51.0 %) | 0.2185 (+9.8 %) | 1.1943 |

Pooled QLIKE improves at both weights because spike loss dominates the total,
and calm nights pay for it — +1.7 % at 3×, +9.8 % at 8×. The trade-off is real
and is stated rather than buried.

**Freshness in the linear anchor**: adding the already-computed but unused
`har_1h` / `har_6h` to the OLS baseline gives pooled QLIKE −7.33 %, spike
−10.44 %, onset-day −5.48 % on the served population. `har_1h` alone is much
worse pooled (0.906), so freshness complements the HAR cascade rather than
replacing it.

**Neither is adopted here.** Both are single-seed, one is OLS-only, and this
session has already watched a clean single-slice result reverse sign on more
data (§10). **DECISION RULE, fixed now before either is run properly:** adopt a
lever only if, on the standard 6-fold walk-forward at 3 seeds, it improves
pooled QLIKE in ≥ 5 of 6 folds **and** improves spike-episode RV/sigma toward
1.0 **and** worsens calm-episode QLIKE by no more than 3 % — the last condition
because an option seller who stops trusting calm nights has not been helped.
Sample size is adequate by construction: the effect sizes here (−5.6 %) are an
order of magnitude above the ~0.003 effects that made §6l's win counts a coin
flip.

### The correction to §7a

The sentence "this is structural, not a bug" is withdrawn. It was a plausible
mechanism stated as a finding, and the check that would have caught it —
*is the information actually absent, or did we merely fail to use it?* — is the
same check `eval/direction.py` was built to perform, and was not run. The lag
is a modelling choice, and the ceiling on fixing it is a 0.73-AUC onset signal
rather than the 0.78 headline.

*Educational research only. Not financial advice.*

## 13. Spike upweighting: DO NOT ADOPT — and this time the rule was right

§12 fixed the rule before this ran: 6-fold walk-forward at 3 seeds, adopt only
if pooled QLIKE improves in ≥ 5 of 6 folds **and** spike RV/sigma moves toward
1.0 **and** calm-episode QLIKE worsens by no more than 3 %. Run through
`run_fold`'s `extra_w` hook, so network, committee, calibration, embargo and
scoring are byte-identical across arms and the only difference is one
multiplier. Spike flag as §7a defines it (causal trailing 180-day 95th
percentile, strictly prior days): 31,019 of 510,496 episodes, 6.08 %.

| arm | pooled | spike | calm | RV/sigma (spike) | pooled wins | calm cost |
|---|---|---|---|---|---|---|
| 1.0× control | 0.2903 | 1.8670 | 0.1989 | 1.6718 | — | — |
| **3×** | 0.2813 | **1.7017** | 0.1988 | **1.6209** | **4/6** | **−0.02 %** |

**Verdict: DO NOT ADOPT.** Pooled QLIKE wins 4 of 6 against a bar of 5. The
other two conditions pass — RV/sigma moves toward 1.0, and calm nights cost
−0.02 %, i.e. nothing.

### Why this is not §6l again

§6l failed a win-count rule on an effect whose mean had improved, and the rule
turned out to be the problem. The obvious reading here is the same story, so it
was checked rather than assumed. Moving-block bootstrap on the paired fold
deltas:

| quantity | mean delta | 95 % CI | excludes zero |
|---|---|---|---|
| pooled QLIKE | −0.00901 | [−0.01647, **+0.00018**] | **no** |
| **spike QLIKE** | −0.16526 | [−0.22739, −0.10217] | **yes** |
| calm QLIKE | −0.00005 | [−0.00663, +0.00656] | no |
| **\|RV/sigma − 1\| on spikes** | −0.05095 | [−0.06080, −0.04255] | **yes** |

**The rule and the interval agree.** The pooled CI contains zero — barely, at
+0.00018, but it contains it — so even deciding on the mean would not establish
a pooled gain. Unlike §6l, there is no case that the rule vetoed something the
data supports.

`research/pitfalls.check_not_a_coin_flip` passed on this experiment
(|mean| 0.00901 against se 0.00392, resolvable at n = 6), which is what makes
the "the rule was too crude" defence unavailable: the design had enough power
to see this effect, and what it saw was a pooled gain that does not clear zero.

### What the lever actually does, and why the pooled metric cannot see it

Spike QLIKE improves by **−0.165 with a CI excluding zero**, and spike
calibration moves toward 1.0 by **−0.051, also excluding zero**, at a calm cost
indistinguishable from nothing. The lever does precisely what it was designed
to do. It simply does not move the pooled number reliably, because spikes are
6.08 % of episodes — a large improvement on a small slice, diluted.

That is a statement about the metric, not a defence of the lever. And it points
at a rule-design mistake that is mine: **§12 chose pooled QLIKE as the primary
condition for a lever aimed at 6 % of episodes.** A treatment targeted at a
minority slice should be judged on that slice, with the majority slice as the
guard — which is the opposite of how §12 wrote it.

**That correction applies to the NEXT measurement, not this one.** Rewriting a
rule after seeing which arrangement passes is the failure this benchmark exists
to prevent, and it has been enforced against three other results this session
(§6l, §10, §11c). The verdict stands: DO NOT ADOPT.

**DECISION RULE for the successor, fixed here before it runs:** judge a
spike-targeted lever on **spike-conditional QLIKE** with a moving-block
bootstrap CI excluding zero, **and** spike RV/sigma moving toward 1.0, **and**
calm-episode QLIKE not worsening by more than 1 % with its own CI — the guard
tightened from 3 % to 1 % precisely because calm cost turned out not to be the
binding constraint and a loose guard was never tested. Pooled QLIKE is reported
but is not a condition, because a 6 %-of-episodes treatment cannot be expected
to move it and requiring so guarantees a null.

### A failed prediction, recorded

`levers.py`'s docstring, written before the run, said the likelier trap was
that the lever "buys spike accuracy by making the other 92 % of nights worse".
It did not: calm cost was **−0.02 %**, and calm QLIKE was *better* in four of
six folds. The binding constraint was the pooled win count, which the docstring
did not flag at all. Sixth failed prediction in this project.

*Educational research only. Not financial advice.*

## 14. The data route works — and the first harvest verified two claims wrong in opposite directions

`.github/workflows/harvest-newdata.yml` ran on a GitHub Actions runner and
committed `data/newdata/{funding_btc,dvol_btc}.parquet` (commit `f27f818`,
authored by `github-actions[bot]`). **The route around the container's network
boundary works end to end**: nothing here can reach an exchange — all 18
endpoints probed return 403 through the proxy — but a runner can, and commits
the result back where the container reads it.

That also means the history depths, previously **UNVERIFIED-FROM-CONTAINER**,
are now measurable. Both were wrong, in opposite directions.

### Funding rate: materially better than reported

| | reported from repo citation | **measured on the harvested file** |
|---|---|---|
| cadence | 8-hourly settlements | **hourly** (`interest_1h`, `interest_8h`) |
| rows | 7,402 | **64,206** |
| span | 2019-09 → 2026-06 | **2019-04-30 → 2026-08-26** (2,675 days) |
| **inside the training window** | not stated | **32,196 rows — 50.1 %, 1,341 days** |

8.7× more records, four months earlier, at 8× finer resolution, and — the
number that decides usability — **half the series sits inside the training
window**. This is a trainable feature, not a post-hoc overlay.

### DVOL: unusable as harvested, and for a reason worth stating

| | reported | **measured** |
|---|---|---|
| span | ~2.7 years (2023-09 →) | **15 days** (2026-08-10 → 2026-08-26) |
| rows | 1,000 daily closes | 383 |
| **inside the training window** | — | **zero** |

The harvester's own docstring flagged that `get_historical_volatility` takes no
time-range parameters, so "there is nothing to page". Correct — and the
consequence was not carried through to the recommendation, which cited the
repo's 2.7-year figure from a *different* prior fetch.

**The endpoints measure different things.** `get_historical_volatility` is
Deribit's own *realized*-volatility estimate over a recent window.
**DVOL is the implied-volatility index**, served by
`public/get_volatility_index_data`, which takes `start_timestamp` /
`end_timestamp` / `resolution` and pages exactly like the funding endpoint
already does in the same file.

That distinction is the whole point of the feature. §12 established the onset
ceiling (AUC 0.733) exists **because every current input derives from BTC's own
past bars**. Implied volatility is the one candidate that is forward-looking by
construction. Harvesting a realized-volatility series by mistake would have
added another backward-looking feature and tested nothing — while appearing to
test the hypothesis.

`fetch_deribit_dvol_index()` now pages the index endpoint, with the old call
kept as a fallback that announces which route produced the data, because a
15-day series and a 3-year series arriving under the same filename is precisely
how a useless feature gets trained on. **UNVERIFIED FROM THIS CONTAINER** —
every exchange endpoint is 403 here; the next scheduled run on a runner will
confirm or refute it.

### What this changes

Funding rate is promoted from "best-corroborated candidate" to **verified and
trainable**: hourly, 7.3 years, 50 % training-window overlap. DVOL is demoted
from rank 2 to **blocked pending a working harvest**, and its earlier ranking
rested on a citation rather than on the endpoint actually being called.

The pre-registered rule in `eval/newdata.py` for whether either earns a place
in the model is unchanged and was fixed before any of this data existed.

*Educational research only. Not financial advice.*


## 15. The DVOL fix worked — implied volatility is now trainable

§14 diagnosed the 15-day DVOL harvest as the wrong endpoint:
`get_historical_volatility` is Deribit's *realized*-volatility series with no
time range, while DVOL is the implied-volatility **index**, served by
`get_volatility_index_data`, which pages on `start_timestamp`/`end_timestamp`.
The fix was written here and marked UNVERIFIED-FROM-CONTAINER, because every
exchange endpoint is 403 through this proxy. The scheduled workflow then ran it
on a GitHub Actions runner and committed the result.

| | before the fix | **after** |
|---|---|---|
| rows | 383 | **47,563** |
| span | 2026-08-10 → 2026-08-26 (15 days) | **2021-03-24 → 2026-08-26 (1,981 days)** |
| resolution | hourly | hourly |
| **rows inside the training window** | **0** | **15,552** |
| level | median 26.8 | median 55.57, range [19.17, 166.39] |

**Implied volatility is now a trainable feature**: 5.4 years of hourly history
with substantial training-window overlap.

Note this also corrects the repo's *original* citation, not just the harvest.
`OPTION_BUYER_ALPHA.md` records DVOL as "1,000 daily closes 2023-09→2026-06".
The index endpoint returns hourly data from 2021-03-24 — roughly 2.5 years
earlier and 24× finer. That earlier figure came from whatever route that
analysis used; it is not the ceiling.

### Why this matters, and why Phase 4 still does not open yet

§12 located the model's binding constraint precisely: onset prediction caps at
AUC 0.733 **because every current input derives from BTC's own past bars**, and
onset is by definition not yet in the price. Implied volatility is the one
candidate that is forward-looking by construction — it is the market's own
forecast, which is exactly the information class the feature set lacks.

**It is still not tested, and Phase 4 remains closed until Phase 0's gate
passes.** The temptation to jump straight to the exciting feature is the
failure mode this plan exists to prevent: an experiment's delta is
uninterpretable on a base that has not been shown to reproduce. The ordering is
not ceremony — §10 and §13 both turned on effects smaller than the tolerance a
non-reproducible pipeline would carry.

The pre-registered rule for whether an implied-volatility feature earns its
place is already fixed in `eval/newdata.py`, written before this data existed.

*Educational research only. Not financial advice.*

## 16. Phase 0's leakage audit: no leak — and the benchmark does not train the shipped model

### The audit, and what it actually attacked

`eval/leakage.py` extends `test_features.py`'s causality check rather than
duplicating it, and its most important design choice is a **positive control**:
a decoy feature that deliberately reads `rv5[row]` — the anchor hour itself
instead of `row-1`. Without one, "no violations found" is indistinguishable
from "the harness cannot detect violations."

The decoy also exposed a flaw in the probing scheme: a naive random probe would
almost never land on the single row where the decoy is detectable. Forcing the
probe set to include every episode anchored exactly at the cut caught it
**12/12**. Only then does a clean result mean anything.

| item | result |
|---|---|
| feature causality, 42 columns × 6 cut points × 2 corruption styles | **42/42 CAUSAL**, 0 violated |
| max window overlap, training vs earliest test episode | **0 hours** (−4,440 h margin) |
| embargo at both boundaries, both split mechanisms | **+24.0 h**, binding with zero slack |
| standardizer refit after corrupting test rows | moved by **0.000e+00** |
| `EmpiricalSpecialist` quantiles, same attack | moved by **0.000e+00** |
| per-fold sigma clip bounds | genuinely fold-dependent: [0.0052, 0.1449] → [0.0039, 0.1197] |

**Verdict: NO LEAKAGE FOUND**, on an audit that demonstrated its own detection
power first.

One caveat recorded rather than buried: `corp_decomposition`'s isotonic fit is
in-sample on the test batch. That is the CORP method's own design (Dimitriadis,
Gneiting & Jordan 2021), touches no trained parameter, and applies identically
to all six competitors including climatology — so it advantages nothing. But
**DSC and MCB are not out-of-sample skill estimates the way pinball, CRPS and
Brier are**, and this document has not always been careful to say so.

### The finding that matters more than the audit

`eval/benchmark.py:514` calls:

    r = run_fold(ep, X, f, a.hidden, a.seeds)

with **no `sigma_ref_fn`**. So `sigma_ref_all` stays `None`, and `prepare()`
falls through to its default — whose own docstring says:

> *Default is RV, the REALIZED window volatility, and that is a train/serve
> skew … RV appears in the denominator of the target AND in the conditioner, so
> noise in RV alone produces Spearman −0.4331.*

Meanwhile `train_v2.py:121-126` builds the causal reference and stamps
`stage_b_sigma_ref: "causal_har_1d_clipped"` into the shipped artifact, and
§6b records that retargeting as an **adopted fix** (DSC/UNC 0.04980 → 0.05382,
6/6 folds).

**So the headline benchmark trains Stage B in the exact way this repository
documents as a defect, while the deployed model does not.** Every number in the
main table describes a model that is not the shipped model.

This is the **fourth** defect of the measured-≠-shipped class here, after the
artifact metadata disagreeing with its weights, the `feat_cols` 42-vs-39 crash,
and `has_mx()` returning False for every artifact.
`research/pitfalls.check_measured_is_shipped` exists for exactly this — and did
not catch it, because it compares **artifact metadata**, not the **evaluation
path**. A check that only inspects the artifact cannot see a harness that
trains something else.

### Consequence for the phase plan

Not a leak, so it does not block on integrity grounds. But it means the Phase 0
baseline, once determinism is settled, is a baseline **of the wrong
configuration**. Correcting it changes every headline number, so it is a
change, not a fix-in-passing, and it gets its own gated step rather than being
folded into Phase 0 silently — which is precisely the "don't fix several things
at once" discipline this plan opens with.

**Pre-registered, before the corrected benchmark is run:** re-run with
`sigma_ref_fn` supplied. The corrected configuration is adopted as the new
baseline **unconditionally** — not because it scores better, but because it is
what ships. If it scores *worse*, that is the honest baseline and every past
comparison in this document was flattered by a defect. Recording that
expectation now removes the temptation to keep whichever number looks better.

The new pitfall this earns: **a measured-vs-shipped check must compare the
evaluation path's configuration against the production trainer's, not just the
artifact's metadata.**

*Educational research only. Not financial advice.*

## 17. Phase 0 determinism: PASS, bit-identical

Gate v2 asked the only question that gates later phases — does the pipeline
return the same answer from the same inputs?

| | result |
|---|---|
| metrics compared | **351** |
| worst absolute difference | **0.000e+00** |
| worst relative difference | **0.000000 %** |
| metrics failing the gate | **0** |

**Bit-identical across two independent runs.** Seeds 0/1/2, no input changed.

This closes the diagnosis opened in §16: gate v1's failure was **entirely the
stale artifact and not one bit of nondeterminism**. Had determinism been the
cause, every experiment in this document measured at deltas of 1–5 % would have
been uninterpretable. It was not, and they are not.

Combined with §16's leakage verdict, Phase 0's integrity conditions are met:
the pipeline is deterministic and free of the leaks an adversarial audit with a
working positive control could find.

### What is fixed, and what is deliberately not deleted

`benchmark.py:main()` now supplies the causal sigma reference, matching what
`train_v2.py` builds for the shipped artifact. Two choices worth stating:

**The callable form, not a precomputed array.** Clip bounds refit on each
fold's own training episodes, so nothing downstream of a fold boundary can
influence the reference. A single global clip would be a subtle leak — exactly
the class §16's audit was probing for.

**`--no-causal-sigma` is retained.** Deleting the old path would make every
prior comparison in this document unverifiable. "It was wrong" is not a reason
to destroy the ability to confirm what it was, and this file's value depends on
its superseded numbers staying reproducible.

*Educational research only. Not financial advice.*

## 18. The corrected baseline — adopted unconditionally, and it is slightly worse

§16 pre-registered this: the configuration that ships becomes the baseline
**whether it scores better or worse**, because a baseline measuring something
that is not deployed is not a baseline. Result:

| metric | realized-RV (what the benchmark scored) | causal σ-ref (**what ships**) | change |
|---|---|---|---|
| **QLIKE noctua** | 0.289605 | **0.290346** | **+0.256 % — worse** |
| QLIKE log_har | 0.305651 | 0.305651 | **0.000 %** |
| QLIKE persistence | 0.433230 | 0.433230 | **0.000 %** |
| pinball_up | 0.003305 | 0.003303 | −0.053 % |
| CRPS_up | 0.004890 | 0.004882 | −0.163 % |
| Brier up 2 % | 0.195717 | 0.195381 | −0.172 % |
| MCB up 2 % | 0.012241 | 0.011918 | **−2.637 %** |
| DSC/UNC dn 2 % | 0.05989 | 0.06415 | **+7.11 %** |

**ADOPTED.** Volatility QLIKE is marginally worse; barrier calibration and
downside discrimination are meaningfully better. That is the same trade §6b
found when it first introduced the causal reference, now confirmed on the
headline benchmark rather than on an ablation.

### The control that makes this credible

`log_har` and `persistence` are **bit-identical at 0.000 %**. Neither touches
Stage B, so a correct fix must leave them exactly unchanged — and it did. A
change that had perturbed the harness more broadly would have moved them.
Without that control, "the numbers moved a bit" would be indistinguishable from
"I broke something adjacent".

### What the corrected baseline says about the model, unflatteringly

| model | pinball_up | CRPS_up | Brier up 2 % | DSC/UNC |
|---|---|---|---|---|
| **noctua_v2** | **0.003303** | **0.004882** | 0.195381 | 0.08834 |
| log_har_gauss | 0.003416 | 0.004916 | **0.193062** | **0.09028** |
| climatology | 0.003769 | 0.005463 | 0.216664 | 0.00000 |
| noctua_shuffled | 0.003810 | 0.005553 | 0.215179 | 0.01251 |

**`log_har_gauss` beats `noctua_v2` on Brier and discrimination at the 2 %
barrier.** NOCTUA wins the distributional scores — pinball and CRPS — which are
what the quantile heads are actually trained on. This is consistent with what
this document already said (4/6 folds, t = +0.46, noise), and it is restated at
the top of the new baseline rather than left for a reader to find, because the
baseline is where a model's weaknesses should be hardest to miss.

`climatology` at exactly 0.00000 DSC/UNC is the by-construction check working:
a constant forecaster has zero discrimination. `noctua_shuffled` at 0.01251
bounds what shuffled inputs buy.

### Phase 0 is CLOSED

| condition | result |
|---|---|
| determinism | PASS — 351/351 bit-identical (§17) |
| leakage | NO LEAK — positive control verified (§16) |
| baseline reproduces | PASS on gate v2; gate v1 void (§16) |
| eval path = shipped config | FIXED (§16, §18) |
| environment pinned | `model/requirements-research.txt` |
| `BASELINE_MANIFEST.md` | written against corrected numbers |

**Phase 1 opens.**

*Educational research only. Not financial advice.*

## 19. Phase 1 finds the lag mechanism: the freshest features are absent from the dominant term

### First, a correction to the Phase 1 catalog

The feature-catalog agent reported that `rng_*` and `cal_month_*` sit in "a
fourth, unnamed category" belonging to neither the model inputs nor
`NON_MODEL_COLS`, implying they are computed but unused. **That is wrong, and
the count was wrong too.** Verified against the shipped artifact:

| | count |
|---|---|
| columns in `features.parquet` | 42 |
| `NON_MODEL_COLS` (`eff_1d/3d/7d`) | 3 |
| **`feat_cols` — Stage A's actual input** | **39** |
| `BASE_COLS` ∪ `SHAPE_COLS` | 24 |
| in `feat_cols` but not in BASE ∪ SHAPE | **15**, not 6 |

Stage A's first-layer weight is `(32, 39)`, confirming all 39 reach the network.
The 15 are **not unused** — they feed Stage A. They are absent from Stage B's
shape set and from the OLS base. That is a different, and much more
interesting, fact.

### The finding

    BASE_COLS = ['har_1d', 'har_5d', 'har_22d', 'cal_H', 'cal_weekend_frac']
    BLEND_W   = 0.25        # Stage A median = 0.25 * neural + 0.75 * Log-HAR

`har_1h` and `har_6h` are computed, stored, and fed to the neural stage — and
are **absent from `BASE_COLS`**, the input set of the Log-HAR anchor that
carries **75 % of the blended weight**.

**So the dominant term's fastest input is `har_1d`.** A component whose finest
resolution is one day cannot respond to anything faster than a day, and it
decides three-quarters of the forecast.

### Why this matters: it is the lag mechanism, and it closes a chain

Three sections converge here.

- §7a measured the lag — spike nights under-forecast 45 %, 25.8 % of total loss,
  predicted σ correlating 0.920 with realized RV dated one day *after* and only
  0.507 one day *before* — and explained it as **structural**: "every input the
  model has is a trailing statistic."
- §12 **refuted** that: a spike-onset classifier clears its nulls at AUC 0.733
  (CI [0.6547, 0.8052]). The information is present.
- §19 now supplies the mechanism neither had: the information is present, the
  *neural* stage receives it, and the *75 %-weight anchor does not*.

§12's Arm 4 measured exactly the predicted consequence without knowing the
cause — adding `har_1h`/`har_6h` to the OLS baseline moved pooled QLIKE
**−7.33 %**, spike **−10.44 %**, onset-day **−5.48 %**. Those gains are large
precisely because the features were never in the term that dominates.

This is the difference between "the model lags" and "the model lags *because*
the component holding 75 % of the weight cannot see below daily resolution."
The second is actionable.

### Pre-registered experiment, fixed before it runs

**Hypothesis:** adding `har_1h` and `har_6h` to `BASE_COLS` reduces the
volatility lag, with the largest effect on spike-onset episodes.

**Single change:** `BASE_COLS` gains two entries. Nothing else — same
architecture, seeds, folds, embargo, blend weight, calibration.

**Primary endpoint — on the slice the treatment targets**, per the §13 lesson
that made pooled QLIKE primary for a 6 %-of-episodes treatment and produced an
uninterpretable verdict: **spike-episode QLIKE**, with a moving-block bootstrap
CI that must exclude zero on the favourable side.

**Guardrails, all of which must hold:**
- calm-episode QLIKE must not worsen by more than **1 %** (own CI);
- pooled QLIKE reported but **not** a condition;
- deep-tail barrier MCB must not worsen with a CI excluding zero unfavourably;
- the Log-HAR baseline itself will change (it gains inputs), so
  `log_har_gauss` is re-scored and reported rather than treated as fixed.

**Expected direction:** spike QLIKE improves. **Rejection:** if spike QLIKE's CI
contains zero, the §12 Arm 4 gain was an artifact of the OLS-only harness and
does not survive the full pipeline.

**Compute:** 6 folds × 3 seeds × 2 arms, ~20 minutes.

**A caution recorded in advance:** `BASE_COLS` also defines the OLS anchor that
`train_v2` fits and the artifact stores as `har_beta`. Changing it changes the
shipped artifact's shape, so this is a Phase-gated change and not a
fix-in-passing — the same discipline that kept §16's sigma_ref correction in
its own step.

*Educational research only. Not financial advice.*

## 20. The mandatory GARCH baseline, and the fit that silently never ran

The research protocol names GARCH(1,1) — with a heavy-tailed variant — a
**mandatory** volatility baseline. This repository never had one, and had been
claiming "NOCTUA beats the baselines" against a set that omitted the single
most standard comparator in the volatility literature.

It was missing for an infrastructure reason, not an oversight: nothing here
could install `arch`. `eval/toolchain.py` established empirically that **PyPI is
allowlisted through this proxy** even though every exchange API is not, which
unblocked it in minutes.

### The result

| model | QLIKE (6 walk-forward folds) | vs NOCTUA |
|---|---|---|
| **noctua** | **0.290346** | — |
| log_har | 0.305651 | +5.3 % |
| **garch_t** (Student-t innovations) | **0.325937** | **+12.2 %** |
| garch_normal | 0.340448 | +17.3 % |
| persistence | 0.433230 | +49.2 % |

**NOCTUA clears the mandatory baseline**, and so does Log-HAR. The heavy-tailed
variant beats the normal one, as the protocol anticipated it would — measured
`nu` around 4.2, which is very fat-tailed and consistent with hourly BTC returns
ranging −34.7 % to +29.9 %.

The comparison is like-for-like by construction: GARCH is a model of a return
*series*, so its forecast is aggregated to exactly NOCTUA's object —
`sigma[t,H] = sqrt(sum of the next H hours' conditional variance)` — on the same
episodes. Quoting a one-step-ahead sigma instead would have compared two
different quantities.

### The fit silently never ran, and the failure flattered NOCTUA 3×

The first run reported **garch_normal 0.4664 — worse than persistence**, which
would have been a headline of "NOCTUA beats GARCH by 38 %".

That number was an artifact. On this data `arch`'s optimiser returns its own
default starting values (`alpha=0.10, beta=0.88`) **exactly, to full
precision**, in 4 of 6 folds — while reporting `convergence_flag = 0` and
*"Optimization terminated successfully"* in 0.1 s on 74,455 observations.

**The tell was mechanistic, not statistical.** Parameters equal to a library's
documented defaults are a starting guess, not an estimate. Probing from other
starts moved it and found strictly higher likelihood every time:

| | single-start ll | multi-start ll | gain |
|---|---|---|---|
| normal | −85,061.9 | **−83,473.8** | **+1,588** |
| Student-t | −71,546.0 | **−67,080.0** | **+4,466** |

Tighter tolerances (`ftol` 1e-12, 1e-14; `maxiter` 5,000) changed nothing — the
optimiser was not iterating at all, so there was nothing to tighten.

Fixing it moved GARCH from 0.4664 to 0.3259. **The broken fit inflated NOCTUA's
margin from 12.2 % to 38 % — a factor of three, in NOCTUA's favour.** A baseline
that quietly under-performs is the most comfortable kind of bug to have, and the
only reason it was caught is that a GARCH losing to persistence is not credible.

### A second defect the fix exposed

With multi-start, persistence estimates land at **1.0000–1.0024** — IGARCH,
where the model-implied unconditional variance `omega/(1-alpha-beta)` is
undefined or negative. The original recursion seeded from exactly that
expression with a `max(..., 1e-6)` floor, which would have silently returned
`omega * 1e6`. Now seeded from the **sample variance** of training returns —
finite, observed, training-only, therefore causal — with an explicit cap on the
forward recursion instead of letting an infinity dominate QLIKE.

### What this does and does not settle

It settles that NOCTUA earns its complexity over a three-parameter conditional
variance model on QLIKE, by 12.2 %. It does **not** settle the barrier question:
§18 records `log_har_gauss` beating NOCTUA on Brier at the 2 % barrier, and
GARCH has not been given a barrier forecast here at all, because turning a
variance path into a touch probability requires the first-passage assumption
§8 showed to be badly misspecified for BTC. Scoring GARCH on barriers through a
law that loses to the base rate would measure the law, not GARCH.

*Educational research only. Not financial advice.*

## 21. The fresh anchor is not neutral — it is reliably worse, on exactly its target

§19 pre-registered this experiment and named its own rejection clause in
advance. The clause fired, but not in the way it was written, and the
difference is the interesting part.

### The rule, and what it said

    PRIMARY, on the slice the treatment targets: spike-episode QLIKE, with a
    moving-block bootstrap CI that must exclude zero on the favourable side.
    ...
    REJECTION: if spike QLIKE's CI contains zero, §12's Arm 4 gain was an
    artifact of the OLS-only harness and does not survive the full pipeline.

Single change, six walk-forward folds, three seeds, both arms:

    control  BASE_COLS (5)  har_1d,                   har_5d, har_22d, cal_H, cal_weekend_frac
    treated  BASE_COLS (7)  har_1d, har_1h, har_6h,   har_5d, har_22d, cal_H, cal_weekend_frac

### The result

| fold | n spike | spike QLIKE control | treated | delta | rel | pooled control | pooled treated |
|---|---|---|---|---|---|---|---|
| 2021 | 24 | 3.2380 | 3.2775 | +0.0395 | +1.22 % | 0.3369 | 0.3417 |
| 2022 | 23 | 1.3542 | 1.3739 | +0.0197 | +1.45 % | 0.2745 | 0.2766 |
| 2023 | 13 | 3.6483 | 3.6635 | +0.0152 | +0.42 % | 0.4340 | 0.4302 |
| 2024 | 26 | 1.3418 | 1.4012 | +0.0594 | +4.42 % | 0.2485 | 0.2531 |
| 2025 | 22 | 0.9571 | 1.0223 | +0.0652 | +6.81 % | 0.2523 | 0.2553 |
| 2026 | 11 | 0.6626 | 0.6729 | +0.0103 | +1.55 % | 0.1959 | 0.1936 |

| quantity | control | treated | delta | 95 % CI (block) | 95 % CI (iid) | folds worse |
|---|---|---|---|---|---|---|
| **spike QLIKE** | 1.86701 | 1.90188 | **+0.03487** | **[+0.02151, +0.05408]** | [+0.01837, +0.05233] | **6 of 6** |
| calm QLIKE | 0.19887 | 0.19807 | −0.00079 | [−0.00202, +0.00098] | [−0.00277, +0.00098] | 3 of 6 |
| pooled QLIKE | 0.29035 | 0.29175 | +0.00140 | [−0.00048, +0.00367] | [−0.00130, +0.00391] | 4 of 6 |
| deep-tail MCB | 0.22531 | 0.22501 | −0.00030 | [−0.00186, +0.00115] | [−0.00147, +0.00086] | 3 of 6 |
| `log_har_gauss` | 0.30565 | 0.30565 | ±0 | — | — | — |

**DO NOT ADOPT.** The primary condition required the spike CI to exclude zero
*favourably*. It excludes zero **unfavourably**, on both bootstrap variants and
in all six folds independently: +1.87 % on the slice the change was designed to
help.

That is a stronger statement than §19's rejection clause anticipated. The
clause was written for "contains zero" — no effect, the OLS-only gain washing
out. What happened is a **sign reversal**: §12's Arm 4 measured −10.44 % on
spike episodes in an OLS-only harness, and the same two columns in the full
pipeline give +1.87 %. Recording that the pre-registered rule did not name this
outcome matters more than the verdict it produced, because a rule that has to
be reinterpreted after the fact is doing less work than it appears to.

### Why the sign flips, and why the flip is the useful finding

The two harnesses are not the same experiment.

- **§12's Arm 4** was OLS alone. Adding `har_1h`/`har_6h` gave that regression
  information it did not otherwise have, and it used it.
- **The full pipeline** already routes `har_1h`/`har_6h` into the neural stage,
  which holds 25 % of the blend. The information was never missing from the
  *model*; it was missing from one *term*.

So the treatment does not add information. It moves a noisy statistic into the
term whose job is to be stable. A 1-hour realized-volatility estimate is built
from 60 one-minute returns and is correspondingly noisy — the entire reason
HAR averages over 1-day, 5-day and 22-day windows is to trade resolution for
estimator variance. Putting the noisiest available window into the 75 %-weight
anchor buys duplicated signal at the cost of variance in the component that was
bounding the model's worst folds.

That reframes §19's finding. §19 said "the dominant term cannot see below daily
resolution" and treated that as a defect. §21 says the anchor's blindness below
a day is not a bug to be fixed by giving it sharper eyes — **it is what the
anchor is for**. The lag is real (§7a), the information is present (§12), and
the place it is being lost is not the anchor's input set.

### What this redirects to, which §19 also named in advance

§19's "what a negative result looks like" paragraph said, before any of this ran:

> That would redirect the work away from feature placement and toward the blend
> weight itself — which is the next thing to test, not a dead end.

That is now the live hypothesis, and §21 sharpens it. If the anchor is the
robust term and the neural stage is the fast one, then a **constant** 0.25 is a
single compromise being asked to serve two regimes that want opposite things.
`eval/blend_ceiling.py` tests it, with the ceiling measured before the lever is
built.

### A defect found in the machinery, not the model

The first run of `anchor_freshness.py` printed this verdict with every
confidence interval reading `[nan, nan]`. `direction.block_bootstrap_ci`
returns `(nan, nan)` below n = 20 — a correct guard for its intended argument,
a per-episode loss difference, where fewer than 20 points means the caller
erred. The unit here is a **fold**, and there are six. The rule then asked

    primary = sp_hi < 0.0

and `nan < 0.0` is `False`, so the script printed `PRIMARY ... : False` and
`DO NOT ADOPT` **without any statistic having consulted the data**.

The verdict was right by luck — all six folds moved the wrong way — which is
the worst version of this failure, because a correct answer is not evidence
that the machinery works. NaN is uniquely dangerous in a decision rule: it
silently satisfies every "did not clear the bar" branch, so a rule returns the
same answer whether the effect is large, zero, or simply unmeasured.

Three things changed as a result:

- `direction.mean_ci` — a CI defined at every usable n, returning the
  pre-registered moving-block interval *and* the iid interval *and* the sign
  count, so the estimator choice at n = 6 is visible rather than picked;
- `direction.ci_excludes_zero` — **raises** on an undefined interval instead of
  answering `False`;
- `research/pitfalls.check_ci_is_defined` — check 11, with this run as its
  historical case, so the self-test now expects 11 failures.

The scores above were not recomputed: `anchor_freshness.py --from-json`
re-derives the verdict from the stored per-fold records through the same code
path, so a broken *statistic* was fixed without a single model being retrained
and without any opportunity to change a *score*.

*Educational research only. Not financial advice.*

## 22. Pre-registration: how much statistical power does this project actually have?

A number surfaced while wiring `eval/blend_ceiling.py` that reframes every
verdict in this document, including the ones already recorded.

    benchmark.run_fold:  m_te = fold["test"] & finite & production_mask
    splits.production_mask:  (H == 19) & (anchor_hour == 17)

**Every walk-forward fold is scored on ~365 episodes** — one 19-hour window
opened at 17:00 UTC per day — of which roughly 20 carry the causal spike flag.
Six folds is ~2,190 test episodes and ~119 spike episodes *in total*. The
episode population is 510,496.

That is a defensible choice for a deployment decision: 17:00/19h is the actual
trade, and scoring it is scoring the product. It is a much less obvious choice
for a *research* decision about a model's internals, where the question is
whether an effect exists at all, and where the cost of a slice is power.

§7a established that spike nights carry 25.8 % of total loss. Every experiment
aimed at them — §13's upweighting, §19/§21's fresh anchor, and the blend weight
now under test — has been decided on about twenty episodes per fold.

### The question, which is measurable rather than rhetorical

Widening the slice from one anchor hour to all 24, at the same H = 19, gives
24× the episodes. It does **not** give 24× the information: episodes anchored
at 16:00 and 17:00 share 18 of their 19 hours. The honest quantity is the
**effective sample size multiplier** — how much a CI actually tightens when the
nominal n rises 24-fold.

    expected under independence:   sqrt(24) = 4.90x tighter
    observed:                      to be measured

The ratio of those two is a reusable constant for this project. It tells every
future experiment how much power a wider slice really buys, instead of leaving
it to be assumed in either direction.

Mixing horizons is deliberately excluded. Episodes at the same anchor with
H = 6, 12, 19 and 24 are nested inside one another and their overlap is not
serial dependence a moving-block bootstrap can model. Holding H = 19 and
varying only the anchor hour keeps the dependence in the one form the estimator
is built for.

### Pre-registered, fixed before it runs

**Design.** Re-decide §21 — the one experiment in this document with a large,
unanimous, already-resolved effect — on the 24-anchor-hour slice at H = 19, via
`run_fold`'s existing `prod_override`. Same arms, same seeds, same folds, same
rule. §21 is the right subject precisely *because* its verdict is not in doubt:
the point is to measure the estimator, and that requires a case where a
disagreement would be informative rather than ambiguous.

**Primary endpoint: the CI width ratio**, block-bootstrap spike-QLIKE CI on the
production slice divided by the same CI on the 24-hour slice. This is a
measurement, not a hypothesis test, and it is reported with no threshold to
clear.

**The one thing that would be a finding, stated in advance:** if the §21 verdict
*flips* — if spike QLIKE's CI crosses zero, or excludes it favourably, on the
wider slice — then the production slice has been deciding research questions it
does not have the power to decide, and every prior verdict in this document
that rested on a marginal CI needs re-running. If the verdict holds and only
the interval narrows, the production slice was adequate for effects of that
size, and the multiplier tells us the size below which it is not.

**What this cannot do.** A wider slice cannot rescue an effect that is absent,
and a tighter CI around a null is still a null. This measures resolution, not
truth.

### Amendment, recorded while the run was still going and before any CI width existed

**The primary quantity as written above is not scale-free, and that is a defect
in this pre-registration.** The ratio is of *absolute* interval widths, and the
two slices do not share a scale. From the folds visible mid-run, control
spike-QLIKE levels are 3.2380 / 1.3542 / 3.6483 on the production slice against
1.1360 / 0.5984 / 1.1163 on the 24-hour slice for the same fold years. An
interval around a quantity three times larger is wider for reasons that have
nothing to do with precision.

Both are therefore reported: the **absolute** ratio because it was
pre-registered, and changing a rule quietly is worse than stating a flawed one;
and the **scale-normalised** ratio — each interval width divided by that slice's
own control baseline — as the interpretable one, since relative resolution is
what determines the size of effect an experiment can detect.

**A separate observation, which is not part of the power question and must not
be reported as one.** The relative treatment effect also looks smaller on the
wide slice: 1.87 % on the production slice against 0.91 % / 1.57 % / 1.26 % on
the first three wide-slice folds. That is a change in the *effect*, not in its
*precision*. Two readings will be available and this data cannot separate them —
either the 17:00 anchor genuinely carries a larger treatment effect than other
hours, or the production-slice estimate was inflated and the wide slice is
regression to the mean. Both will be stated, and neither will be picked.

**Compute.** 6 folds × 3 seeds × 2 arms on a 24× wider scoring slice; training
cost is unchanged, only scoring widens.

*Educational research only. Not financial advice.*

## 23. The ensemble weight is not a mean-improvement lever. It is tail insurance.

§19 named the blend weight as what a negative anchor result would redirect to,
and §21 delivered that negative result. Two pre-registered tests followed, in
increasing order of simplicity, and both reject — but what they reject, and
what falls out of rejecting it, is worth more than a win would have been.

### The constant-w surface, which is not flat

`eval/blend_ceiling.py` recovers the raw neural median by inverting the affine
blend, so the whole weight surface comes out of one benchmark run instead of one
6-fold retrain per weight. Mean over the six folds:

| w | 0.00 | 0.15 | **0.25** | 0.35 | **0.45** | 0.55 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|
| pooled | 0.3057 | 0.2952 | **0.2904** | 0.2875 | **0.2863** | 0.2870 | 0.2940 | 0.3138 |
| spike | 1.9211 | 1.8823 | 1.8678 | 1.8622 | 1.8627 | 1.8775 | 1.9282 | 2.0430 |
| calm | 0.2103 | 0.2024 | 0.1989 | 0.1968 | 0.1963 | 0.1969 | 0.2031 | 0.2200 |

The minimum is at **w = 0.45**, 1.41 % below the shipped 0.25. **That number is
not a result and must not be quoted as one** — it is chosen after seeing every
fold's test score, which is the definition of tuning on test. It is a reason to
run an honest version.

### Two honest versions, both rejected

**Two-state** (`blend_ceiling.py`): separate weights for spike-risk and calm
episodes, fitted on prior folds only.

| arm | pooled Δ | 95 % CI | folds better |
|---|---|---|---|
| raw argmin | +0.01355 | [−0.00686, +0.05027] | 3 of 5 |
| **shrunk** (primary) | **+0.00307** | **[−0.01008, +0.02603]** | 4 of 5 |

**One-state** (`blend_one_state.py`): a single constant, one estimated parameter
instead of two, at Bonferroni-adjusted 97.5 % intervals because it is the second
look at these forecasts.

| arm | pooled Δ | 97.5 % CI | folds better |
|---|---|---|---|
| raw argmin | +0.01167 | [−0.00872, +0.04383] | 2 of 4 |
| **shrunk** (primary) | **+0.00049** | **[−0.01141, +0.01981]** | 4 of 5 |

The combination-forecasting literature predicted the shape of this exactly. The
raw argmin was the worse estimator in both tests, and the shrinkage cut its
damage by 77 % (two-state) and 96 % (one-state) without ever turning a loss into
a win — which is what Smith & Wallis (OBES 2009) and Claeskens et al. (IJF 2016)
say estimated combination weights do, and what Clements & Vasnev (J. Forecasting
2024) found for the OLS weights inside the HAR model that is NOCTUA's own anchor.

### The finding: one fold decides everything, and it is the one 0.25 was chosen for

Per-fold, the one-state shrunk rule:

| fold | w shipped → shrunk | Δ pooled |
|---|---|---|
| 2022 | 0.25 → 0.55 | **−0.56 %** |
| 2023 | 0.25 → 0.50 | **+7.84 %** |
| 2024 | 0.25 → 0.35 | **−2.20 %** |
| 2025 | 0.25 → 0.40 | **−3.09 %** |
| 2026 | 0.25 → 0.45 | **−8.59 %** |

**Four of five folds improve, several substantially, and 2023 destroys it.**

2023 is the volatility collapse. `infer.BLEND_W`'s own note records that 0.25
was chosen *because* it bounded that fold at +6.7 % where pure NOCTUA suffered
+72.3 %. The worst-fold guard — pre-registered at 5 % precisely because of that
history, before any of these numbers existed — is what rejects the rule, at
+7.84 %.

So the correct statement about `BLEND_W = 0.25` is not "it is optimal". It is:

> **0.25 buys insurance against a 2023, and the premium is about 1.4 % of mean
> QLIKE.** Raising it collects the premium back in four years out of five and
> hands it all over in the fifth.

That is a stated, quantified trade rather than a tuned constant, and whether the
premium is worth paying is a risk-appetite question, not a QLIKE question.

### Why no rule can find the weight: the parameter is not stable

The two-state in-sample oracle — the best a two-state rule could do *with*
foresight — clears **7.07 %**, well above the 2 % floor pre-registered as "worth
building". The headroom is real. What is missing is any way to reach it:

| fold | oracle w_calm | oracle w_spike |
|---|---|---|
| 2021 | 0.65 | 1.00 |
| 2022 | 0.35 | 0.70 |
| 2023 | **0.00** | **0.00** |
| 2024 | 0.70 | 0.55 |
| 2025 | 0.45 | 1.00 |
| 2026 | 0.80 | 1.00 |

Fold-to-fold standard deviation of 0.28–0.51 on a parameter bounded in [0, 1].
2023 wants *nothing* from the neural stage and 2026 wants almost everything. A
causal spike flag cannot separate those, because the thing that separates them
is not spike-vs-calm — it is whether volatility is about to collapse.

(The per-episode oracle reads 31.33 %. It is reported only to be dismissed: it
fits one free parameter per observation. The pre-registration was amended to the
two-state ceiling *before any fold ran*, on the grounds that a floor 2a would
clear regardless of the data is a rule that cannot fail.)

### And then the machinery said the quiet part

`research/pitfalls.check_not_a_coin_flip` fired on the one-state result:

    n=5, |mean| 0.00049 vs sd 0.01958 (se 0.00876); NOT resolvable at this n

The estimate is one eighteenth of its own standard error. This is not a null —
it is a non-measurement, and it would have looked identical if the true effect
were +2 % or −2 %.

Which is exactly §22's thesis, arrived at from a different direction: every fold
here is **~365 episodes**, ~20 of them spike-flagged. Two independent tests of
the ensemble weight have now returned intervals too wide to decide anything, and
the pre-registered measurement of how much power the production slice gives up
is no longer a methodological curiosity. It is the blocking question.

*Educational research only. Not financial advice.*

## 24. Implied volatility: REJECT — and the placebo is why we know

E2 asked whether Deribit's DVOL index carries information NOCTUA's own-past-bars
cascade does not. It is the first candidate feature in this project that is not
a statistic of BTC's history: it is what option sellers charge to insure against
BTC's *future*, which is the information class §12's onset problem needs.

The coverage forced the design. Re-derived through `noctua.splits.time_splits`:

| split | episodes | with a causal IV observation at anchor − 1h |
|---|---|---|
| train | 189,831 | 62,051 — **32.7 %** |
| calib | 52,359 | 52,359 — 100 % |
| test | 73,867 | 73,867 — 100 % |

DVOL began trading 2021-03-24; training opens 2017-08-01. A fill value for the
missing two thirds would be learnable in training and constant at test. So
NOCTUA was left untouched and an IV-conditioned **multiplicative correction** was
fitted on top of its cached out-of-sample forecasts —
`log σ_corrected = log σ̂ + z·β` — by minimising QLIKE. The objective is globally
convex (Hessian `4·E[zz'·r̂·e^{−2z·β}]`, PSD everywhere), verified here against
finite differences to 1.4e-10 on the gradient and 6.8e-10 on the Hessian.

### The headline, and why it is not one

| arm | spike QLIKE Δ | 95 % CI | folds better |
|---|---|---|---|
| shrunk, real features | **−1.05579** | [−1.95499, −0.55027] | **5 of 5** |

A 57 % improvement on spike episodes, unanimous across folds, interval nowhere
near zero. Reported on its own, that is a headline result.

### The placebo

The third arm fits the identical machinery on IV features **circularly rotated
one year inside the covered era** — same marginals, same autocorrelation, the
exact same scored episodes, every bit of the correction's flexibility, and no
possible alignment with the episode being forecast.

| arm | spike QLIKE Δ | folds better |
|---|---|---|
| shrunk, real features | −1.05579 | 5 of 5 |
| **placebo, misaligned features** | **−0.53955** | **5 of 5** |

**A series deliberately misaligned by a year reproduces more than half the
"gain", also 5 of 5.** The margin between them — the part that could be IV — is
−0.02469 pooled, CI [−0.04184, +0.00395], containing zero.

### Where the gain actually comes from: the intercept

A decomposition of the rejected result. Fit an **intercept only** — no IV
features at all, just a scalar multiplying every σ̂:

| fold | base spike | + intercept | + full IV | intercept gain | IV's extra | fitted a | e^a |
|---|---|---|---|---|---|---|---|
| 2022 | 1.3547 | 0.7005 | 0.4481 | −0.6543 | −0.2524 | +0.1840 | 1.202 |
| 2023 | 3.6506 | 2.3008 | 1.4744 | −1.3498 | −0.8264 | +0.1597 | 1.173 |
| 2024 | 1.3424 | 0.7077 | 0.7485 | −0.6348 | **+0.0408** | +0.1732 | 1.189 |
| 2025 | 0.9573 | 0.5320 | 0.4446 | −0.4253 | −0.0874 | +0.1579 | 1.171 |
| 2026 | 0.6630 | 0.3616 | 0.1809 | −0.3014 | −0.1807 | +0.1299 | 1.139 |

**72.0 % of the spike gain is a bare scalar**, and the scalar is stable:
e^a between 1.139 and 1.202 across five folds fitted independently.

That is not implied volatility. It is the point-forecast functional. `infer.py`'s
own note records that QLIKE is minimised by the conditional *mean* of variance
while NOCTUA reports the *median*, and that the mean over-forecasts at ratio
1.205 and scores worse. The fit has discovered a value **between** them, ~1.17,
and the placebo collects it too because the placebo also has an intercept.

### The verdict, rule applied verbatim

    PRIMARY shrunk pooled QLIKE CI excludes zero favourably : False
    GUARD   beats the placebo, CI excluding zero            : False
    GUARD   spike QLIKE not worse (CI True, point −66.25%)  : True
    GUARD   calm QLIKE within 1%  (+13.52% of calm base)    : False
    GUARD   RAW coefficient signs stable across folds       : False  — iv_level
    -> REJECT

Three independent failures. Calm QLIKE **worsens 13.52 %** with a CI excluding
zero — the scalar buys spike accuracy by over-forecasting everything, and calm
episodes pay for it. And `iv_level`'s raw coefficient flips sign across folds,
caught by the sign-stability guard that an audit found *documented but never
implemented* hours before this ran.

This is the `(a)` branch of the failure taxonomy written into
`eval/ivfeatures.py` before any of it ran: **implied volatility is persistence in
a different costume**, its level already carried by `har_1d`.

### Two things this earns

**The placebo arm is now standard.** Without it, this result reads as a 57 %
spike improvement. The cost of running it was one extra fit per fold.

**A separate, real finding fell out**, and it is not about IV: a stable ~1.17×
scaling of σ improves spike QLIKE by ~36 % and worsens calm QLIKE by ~13 %. That
is the QLIKE asymmetry made concrete — under-forecasting is penalised 1.60× at a
factor-2 error — and it is a live question about the point-forecast functional,
not a feature question. **Pre-registered as E-scale, and queued behind E-power**,
because a lever that trades one slice against another must be judged with known
resolution, and §22 says we do not have that yet.

### Pre-registered follow-up, fixed before it runs

**E2b: the same test with the intercept removed.** With `β₀` held at zero the
correction can only *reshape* σ̂, never rescale it, so the 72 % confound is gone
and the IV features are tested on incremental content alone. The placebo arm
loses its intercept too. Same rule, same guards, same shrinkage constants.

**Expected:** near-nothing, since the margin over the placebo already contained
zero. **What would change the conclusion:** if the intercept-free real arm beats
the intercept-free placebo with a CI excluding zero, then IV *does* carry shape
information that the intercept was masking, and `ivfeatures.py`'s branch `(a)` is
wrong.

*Educational research only. Not financial advice.*

## 25. E2c: DVOL's *dynamics* clear every guard — the first ADVANCE in this sequence

§24 rejected implied volatility and explained why: a fitted intercept was
collecting a ~1.17× level correction that had nothing to do with IV, the placebo
collected it too, and the margin between them contained zero.

E2b removed the intercept from **both** arms. Its pre-registration named its own
overturning condition in advance, and the condition fired: the real features beat
the placebo, CI [−0.14822, −0.02069]. Every guard passed except sign stability,
which failed on **`iv_level` alone** — the one feature `ivfeatures.py`'s failure
taxonomy had predicted, before any IV experiment ran, would be redundant with
`har_1d`.

E2c is that prediction acted on: drop `iv_level`, keep the five dynamics
features, at Bonferroni-adjusted intervals for the third test on these forecasts.

### Result

| quantity | Δ | 98.33 % CI | folds better |
|---|---|---|---|
| **pooled QLIKE** | **−0.03264** | **[−0.04356, −0.01403]** | **5 of 5** |
| spike QLIKE | −0.33295 (−20.89 %) | [−0.58819, −0.09022] | 5 of 5 |
| calm QLIKE | −0.01744 (−8.22 %) | [−0.02955, −0.00766] | 4 of 5 |
| **placebo margin** | **−0.04533** | **[−0.07323, −0.01420]** | — |

    PRIMARY shrunk pooled QLIKE CI excludes zero favourably : True
    GUARD   beats the placebo, CI excluding zero            : True
    GUARD   spike QLIKE not worse (point −20.89%)           : True
    GUARD   calm QLIKE within 1%  (−8.22% of calm base)     : True
    GUARD   RAW coefficient signs stable across folds       : True
    -> ADVANCE   (NOT an adoption: the barrier curves were not rebuilt)

Unusually for this document, spike **and** calm both improve. Every prior lever
traded one against the other.

### The coefficients tell a coherent story

Shrunk β on standardised features, five independently fitted folds:

| fold | iv_chg_1h | iv_chg_6h | iv_chg_24h | iv_z_20d | ivrv_ratio |
|---|---|---|---|---|---|
| 2022 | −0.040 | +0.040 | +0.008 | +0.026 | +0.024 |
| 2023 | −0.064 | +0.037 | +0.042 | +0.055 | +0.046 |
| 2024 | −0.036 | +0.032 | +0.063 | +0.050 | +0.051 |
| 2025 | −0.025 | +0.038 | +0.057 | +0.035 | +0.045 |
| 2026 | −0.013 | +0.032 | +0.059 | +0.044 | +0.060 |

Every sign holds in every fold, and λ ≥ 0.85 after the first — the shrinkage had
little to do, because the estimates were already stable.

Read economically: **IV rising over 6–24 hours predicts higher realized
volatility** (`iv_chg_6h`, `iv_chg_24h` positive), as does IV standing high
against its own 20-day history (`iv_z_20d`) and a wide variance risk premium
(`ivrv_ratio`). The one-hour change enters *negatively*, which is what you expect
if the shortest window is mostly quote noise the longer ones have already priced.
Magnitude is modest and sensible: a 1σ move in `iv_chg_24h` scales σ by
e^0.06 ≈ 1.06.

This is the first input in the project that is not a function of BTC's own past
bars, and it is the first to improve both slices at once.

### Scrutiny, because an ADVANCE deserves more of it than a rejection

**Positive control.** Fit on history exactly as E2c does, then shuffle the
*scored* fold's feature rows before applying the correction:

| | pooled Δ | folds better |
|---|---|---|
| aligned | −0.03264 | 5 of 5 |
| test-shuffled | **+0.01909** | **0 of 5** |

A complete reversal. The gain requires the test-fold features to line up with the
episodes they are forecasting, which is what out-of-sample signal means.

**The bootstrap saturates, and that is not extra evidence.** The pooled CI clears
zero at α/1, α/3, α/6, α/12 and α/24 — but the intervals at α/6 and beyond are
*identical* to α/3, because with five fold-level observations the resampling has
only five distinct values and cannot produce a wider interval. Surviving α/24 is
an artifact of n, not a strong significance claim. What carries the result is
5 of 5 folds and the placebo margin, not the tail of the interval.

**Is it a disguised intercept?** Dropping β₀ forces the correction to average
~0 on the *history* folds, not within the fold being scored — so a correction
with little within-fold spread would still be acting as a per-fold level shift,
the §24 confound in subtler form. Measured:

| fold | mean shift | sd of shift | sd ÷ \|mean\| | e^shift, p5–p95 | Δ from its fold-mean alone | Δ full |
|---|---|---|---|---|---|---|
| 2022 | +0.0113 | 0.0769 | 6.79 | 0.919–1.117 | −0.00707 | −0.02942 |
| 2023 | +0.0159 | 0.1040 | 6.53 | 0.878–1.219 | −0.01450 | −0.04777 |
| 2024 | −0.0024 | 0.1043 | 43.12 | 0.844–1.175 | +0.00125 | −0.00098 |
| 2025 | −0.0136 | 0.0809 | 5.95 | 0.868–1.134 | +0.00088 | −0.03360 |
| 2026 | −0.0274 | 0.1136 | 4.14 | 0.828–1.178 | +0.00232 | −0.05142 |

The within-fold spread is **4–43× the fold mean**, the fold means are tiny and
change sign, and replacing the varying shift with its own fold mean captures
−0.00342 of the −0.03264. **89.5 % of the gain requires episode-level
variation.** The correction is doing what it claims — moving σ by roughly ±20 %
at the tails, episode by episode — and is not a level shift wearing a
regression's clothes.

**Does it hinge on `SIGMA_B`?** The shrinkage scale was fixed a priori at 0.10,
and an a-priori constant is only defensible if the conclusion does not depend on
it. Swept across eight orders of magnitude, from near-total shrinkage to none at
all:

| SIGMA_B | pooled Δ | 98.33 % CI | folds better |
|---|---|---|---|
| 0.01 | −0.02325 | [−0.03742, −0.00818] | 5/5 |
| 0.02 | −0.03129 | [−0.04340, −0.01408] | 5/5 |
| 0.05 | −0.03374 | [−0.04379, −0.01532] | 5/5 |
| **0.10** | **−0.03264** | **[−0.04356, −0.01403]** | **5/5** |
| 0.20 | −0.03205 | [−0.04326, −0.01341] | 4/5 |
| 0.50 | −0.03184 | [−0.04315, −0.01320] | 4/5 |
| 1.00 | −0.03181 | [−0.04313, −0.01317] | 4/5 |
| ∞ (none) | −0.03180 | [−0.04313, −0.01316] | 4/5 |

Every row clears zero. The effect is flat between −0.032 and −0.034 across the
whole range, moving only under the most aggressive shrinkage.

That flatness is itself informative: **the shrinkage is nearly inert here**,
because the coefficients were already stable (λ ≥ 0.85 after the first fold).
The same machinery was decisive for the blend weight in §23, where it cut the
raw estimator's damage by 77 %. The difference is not the estimator — it is that
the IV coefficients are a stable parameter and the blend weight is not.

**The selection risk is real and is recorded, not glossed.** E2c drops the
feature that just failed. What separates that from fishing is that
`ivfeatures.py` predicted this exact split in writing before any IV experiment
ran, and E2b's coefficients matched the written prediction. The reader is
entitled to weigh that themselves, so both the prediction and the sequence are in
the ledger.

**Scope.** 1,681 episodes across 5 folds, all inside DVOL's era. The correction
cannot be validated before 2021-03 because the instrument did not exist. For
*serving* that is not a limitation — every live episode has IV — but it does mean
this is a modern-era result and is not claimed beyond it.

### What ADVANCE means, precisely

It means the next step, not the artifact. The pre-registration says this cannot
adopt alone, and the reason is specific: E2c re-weights a recorded median. It
never rebuilt the barrier curves, the committee, or the calibration, and NOCTUA's
product is a touch-probability curve, not a σ. Scaling σ moves every barrier
probability, and nothing here has checked what that does to CORP calibration or
to Christoffersen coverage.

So the confirmation run is a **fresh test**: apply the correction inside the full
pipeline and re-score the barriers, with deep-tail MCB and conditional coverage
as guards. It is queued behind E-power, for the reason §22 gives — a result
whose resolution is unmeasured is a result waiting to be misread.

*Educational research only. Not financial advice.*
