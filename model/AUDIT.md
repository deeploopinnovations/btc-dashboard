# NOCTUA — end-to-end audit

*Educational research only. Not financial advice.*

Where the model stands against its two stated goals, what is blocking the
second one, every defect found reading the pipeline line by line, and what the
next phase of work should be.

Method: four agents were dispatched to read in parallel — a leakage audit of
the feature pipeline, a train/serve skew audit, an inventory of every numeric
claim in the documentation cross-checked against the artifact JSONs, and a
test-coverage map. **Nothing in this file is reported on an agent's say-so.**
Every finding below was re-derived against the code or the data before it was
written down; where an agent's framing turned out to be wrong, the correction
is recorded rather than quietly dropped. Section 6 lists what they got wrong.

---

## 1. The two goals, scored

| goal | status | evidence |
|---|---|---|
| **Volatility** | **achieved** | 4.04 % QLIKE better than a calibrated Log-HAR, p = 0.0002; beats persistence, climatology, scaled climatology and a shuffled control on the full adversarial benchmark |
| **Barrier / excursion** | **achieved against the naive baselines; level with the strong one** | deep-tail calibration error 1.09 pp vs 3.33 pp for Gaussian first-passage at α = 1 %; leads persistence, climatology, scaled climatology and a shuffled control on pinball. The Gaussian is better in the body (α ≥ 10 %) |
| **Volatility amplification** — *will it get wilder?* | **achieved, and the strongest skill in the model** | DSC/UNC **20.3 %**, beats climatology 6/6 folds, CI [+0.047, +0.103] nats. See `BENCHMARK.md` §6g |
| **Direction** | **not achieved — and now measured properly** | §2 |

Two honest qualifications on the middle row, both established earlier and not
softened here. On barrier *discrimination* NOCTUA is **level with, not ahead
of, `log_har_gauss`** — paired across folds it is 4/6 with mean +0.00083 and
a t-like of +0.46, which is noise. The `serve_consistent` change closed a gap
from behind rather than opening one. And the barrier advantage that is real
lives in the deep tail; in the body the textbook model wins.

The first two rows are documented in `BENCHMARK.md`, which is the source of
truth for numbers. This file does not restate them; it exists for the third.

---

## 2. Direction: the finding, and why it is not a failure of effort

### 2.1 What was there before

Direction had been measured **once** — a two-row table in `noctua/evaluate.py`,
on a single split, on the v1 model:

```
NOCTUA        log_loss 0.694344   brier 0.250581
constant_50   log_loss 0.693147   brier 0.250000
```

NOCTUA lost to a coin flip, and nothing followed up. That asymmetry of
attention is itself an audit finding: the half of the mandate that moved
easily absorbed six folds, five baselines, adversarial controls and a
per-expiry decomposition, and the half that matters commercially got a
footnote on one split of a model that no longer ships.

### 2.2 Why re-running that table would have settled nothing

Two explanations produce that same number and call for opposite work.

**(a) Model failure.** The signal is in the features and this architecture
throws it away.

**(b) Efficient-market null.** There is no exploitable sign information at
6–24 h in trailing price/volatility features, for anyone, with any model.

A second score for NOCTUA cannot separate them. So `eval/direction.py` asks
the *attainability* question instead: fit the strongest sign predictors the
feature set admits — L2 logistic on all 39 model features, a regularized
gradient-boosting classifier, and a momentum-only control — walk-forward
across the same six folds, and see whether **anything** clears the bar.

One correction to the obvious framing, found while writing the experiment and
recorded because it changes the interpretation: the convenient story is that
`q_r` is an unsupervised by-product of a model trained on absolute
excursions. **That is false.** `train.py:179` puts `pinball_loss(qr, r)` — a
strictly proper scoring rule on the *signed* standardized return — directly
into the objective, next to the two excursion heads. The model is explicitly
trained to locate the return distribution. If it has no directional skill, the
explanation cannot be that nobody asked it to.

### 2.3 The statistics, and the version of them that lies

`corp_decomposition`'s DSC is computed with an **in-sample** isotonic fit, so
it is mechanically positive on pure noise — PAV finds a monotone staircase in
any finite sample. Comparing DSC to zero is meaningless. The null is generated
by permuting predictions against outcomes, leaving both marginals and the
isotonic machinery intact and destroying only the pairing; the observed DSC
must clear that distribution's upper tail. This is the same control that
retracted the "Kronos carries conditional information" claim in
`BENCHMARK.md` §5.

Episodes at consecutive anchors share most of their window, so IID resampling
would give intervals several times too narrow. All confidence intervals use a
moving-block bootstrap with block length n^(1/3).

Two slices are reported, deliberately: `production` (H = 19, anchor 17:00 —
what ships, ~365 episodes/fold, honest but nearly powerless for a sign test)
and `wide_H19` (every anchor hour, ~8,700/fold — overlapping, hence the block
bootstrap, but where the power is).

### 2.4 Results

From `model/artifacts/direction.json`. `DSC/UNC` is the Brier skill score —
the fraction of the achievable improvement over climatology that the forecast
actually captures. `clears` is whether DSC exceeds the shuffled null's 95th
percentile. `gain` is mean log-loss improvement over the causal base rate, in
nats, with a moving-block bootstrap 95 % CI.

**Production slice — the deployed configuration (H = 19, 17:00, n = 2,046):**

| model | log loss | DSC/UNC | clears null? | gain vs base rate | 95 % CI |
|---|---|---|---|---|---|
| base rate (constant) | 0.69337 | 0 | — | 0 | — |
| momentum | **0.69228** | 0.552 % | **no** | +0.00110 | [−0.00050, +0.00263] |
| logistic (39 features) | 0.69798 | 0.225 % | no | −0.00461 | [−0.00929, +0.00005] |
| gradient boosting | 0.70101 | 0.443 % | no | −0.00764 | [−0.01392, −0.00108] |
| **NOCTUA `prob_up`** | **0.76648** | 0.412 % | no | **−0.07310** | [−0.09586, −0.05245] |

**Nothing clears the null at the configuration that ships.** Not the model,
not a 39-feature logistic, not gradient boosting, not momentum.

**Wide slice — every H = 19 anchor hour, 24× the power (n = 49,111):**

| model | log loss | DSC/UNC | clears null? | gain vs base rate | 95 % CI |
|---|---|---|---|---|---|
| base rate (constant) | 0.69363 | 0 | — | 0 | — |
| momentum | **0.69288** | 0.180 % | **yes** | +0.00075 | [−0.00025, +0.00176] |
| logistic | 0.69673 | 0.120 % | yes | −0.00310 | [−0.00573, −0.00054] |
| gradient boosting | 0.70150 | 0.113 % | yes | −0.00787 | [−0.01222, −0.00368] |
| **NOCTUA `prob_up`** | 0.76473 | 0.184 % | **yes** | **−0.07110** | [−0.08466, −0.05794] |

Three things, and the second is the one that matters.

**(1) A trace of real conditional information exists.** With 49,111 episodes,
four arms clear the shuffled null. This is not nothing: the sign of the BTC
return at 19 hours is not perfectly unpredictable.

**(2) It is worthless, and worse than worthless when served.** The largest
skill score is 0.180 %. For scale, *the same model on the same episodes scores
DSC/UNC ≈ 4.98 % on barrier discrimination* — **27× more skill on the
excursion question than on the sign question**. And the trace does not convert
into a better forecast: momentum, the only arm that improves on the base rate
at all, has a CI that includes zero. Every other arm is **significantly
worse** than a constant.

**(3) NOCTUA's direction call fails through miscalibration, not ignorance.**
Its DSC clears the null — there is signal in it — but:

| | DSC | MCB | ratio |
|---|---|---|---|
| momentum | 0.000450 | 0.000400 | 0.9 |
| logistic | 0.000300 | 0.002140 | 7.1 |
| gradient boosting | 0.000283 | 0.004330 | 15.3 |
| **NOCTUA** | 0.000460 | **0.026800** | **58.3** |

Its mean forecast is nearly right (0.5177 against a base rate of 0.5093); the
damage is in the *spread*. `prob_up` swings confidently around 0.5 and the
swings are noise, so it loses 0.071 nats to a constant — a **10 % worse log
loss than saying nothing at all**, with a CI nowhere near zero.

This is a direct, quantitative vindication of a decision already in the code:
`serve/predict.py` pins the published `upside` to 50.0 and leaves the raw
`p_up` inert precisely because `src/data.js` would otherwise skew strike
selection with it. That call was made on a single split of the v1 model. It
was right, and it is now right for a measured reason.

**Scale check.** Applying Zhang (2026)'s identity `R² = κ(2·DA − 1)²` with
κ ∈ [0.48, 0.64]: even a directional accuracy of 0.52 — far above anything
measured here — buys an R²_OOS of 0.08–0.10 %. A DA of 0.55, the range the
equity literature calls "good", buys 0.48–0.64 %. There is no configuration of
this feature set in which direction becomes a product.

### 2.5 What the literature says about this result

The measured outcome replicates the current published position rather than
contradicting it. Verified sources:

- **Zhang (2026), "A Quadratic Link between Out-of-Sample R² and Directional
  Accuracy"**, arXiv:2602.07841 (econ.EM, v2 14 Feb 2026). Read in full and
  verified. Under a random-walk baseline with sign correctness independent of
  magnitude, `E[R²_OOS] = κ·(2p − 1)²` for MSE-optimal point forecasts, with
  κ estimated at **0.55** (S&P 500) and **0.48** (DJIA) against a Gaussian
  baseline κ ≈ 0.64. Two consequences matter here. A correctly specified
  volatility model with no independent directional information is *provably*
  at DA = 0.5 and R²_OOS = 0 — the signature of doing volatility right, not
  of doing direction wrong. And the map is quadratic, so even a DA of 0.55
  implies R²_OOS of roughly 0.5–1.5 %, with negative realized values expected
  from finite-sample noise. *Caveat, since this is load-bearing: a short
  (~2,200 word) single-author note, not a canonical reference. Its derivation
  is elementary and checks out; cite it as a clarifying identity, not as
  authority.*
- **Christoffersen & Diebold (2006)**, *Management Science* 52(9), 1273–1287 —
  the canonical statement that sign predictability is a *derivative* of
  volatility predictability rather than an independent phenomenon. Not
  fetched directly (egress-blocked); corroborated through Brou & Luger (2026),
  *Journal of Banking and Finance*, arXiv:2606.04153, which cites and builds
  on it and finds sign-conditioned-on-magnitude gains that are real but modest
  and at **monthly** horizons.
- **Bysik & Ślepaczuk (2026)**, arXiv:2606.00060 — hourly BTC/USDT, ~70,000
  observations, XGBoost/LSTM/iTransformer under rigorous 27-fold rolling
  walk-forward. Directional trading collapses under 10 bp costs (XGBoost
  long-only ARC +73.5 % gross → **−64.0 % net**).
- **Young (2026)**, arXiv:2607.26245 — 43 microstructure features, walk-forward
  calibrated, 355,814 rows: model AUC 0.8377 vs the naive market-implied prior
  0.8405. The model **underperformed** the trivial baseline out of sample after
  showing a small in-sample edge.
- **Cont, Cucuringu & Zhang**, arXiv:2112.13213 — order-flow imbalance explains
  83.8 % of *contemporaneous* one-minute return variation, but forward-looking
  OOS R² at one minute is ≈ **−0.10 to −0.37 %**, decaying to nothing "beyond
  several minutes".

The last one is the load-bearing result for what to do next: the best-studied
short-horizon directional signal in the most liquid market on earth is worth a
fraction of a percent of R² one minute ahead and is gone within minutes. This
model forecasts 6–24 **hours** ahead.

---

## 3. Defects found by reading the pipeline

Ordered by consequence. Every one verified directly.

### 3.1 The model is served at anchors it was never scored at — **raised as highest-consequence; measured, and it holds**

`splits.production_mask` is `H == 19 AND anchor_hour == 17`. Every headline in
`BENCHMARK.md` — QLIKE, DSC/UNC, barrier calibration, the `BLEND_W` sweep, the
committee weighting, the training-method ablation — is measured on that slice
and no other.

`serve/predict.py:74` picks its anchor differently:

```python
if anchor_ts is None:
    anchor_ts = int(hour_ts[-1])          # the last CLOSED hour
```

and the publishing cron never passes `--anchor`. `_next_anchor()`, the
function that would pin serving to 17:00 UTC, is defined at `predict.py:56`
and **called from nowhere in the repository** (verified by grep).

Measured over the 102 published forecasts in this repo's own git history:

| | |
|---|---|
| anchored at 17:00 UTC | **5 of 102 (4.9 %)** |
| anchor hours actually used | all 24, roughly uniform |
| lag from window start to publication | median **1.33 h**, mean 1.46 h, max 8.87 h |
| fraction of the 19 h window already elapsed at publication | mean **7.7 %**, max 46.7 % |

Two distinct problems sit here:

1. **~95 % of everything ever shown to a user was produced at a configuration
   that has never been scored.** Not a leak and not a modelling error — a
   claim-versus-product mismatch. **Now measured, and the claim survives:**
   NOCTUA beats Log-HAR by 6.14 % at the benchmarked anchor and 6.06 % across
   all served anchors, with *higher* barrier discrimination away from 17:00.
   See §5.1. This was the audit's most serious finding and it resolved in the
   model's favour — which is worth stating as plainly as it would have been
   had it gone the other way.
2. **The forecast is anchored in the past.** Because the anchor is the last
   *closed* hour, the 19-hour window has already been running for a median
   1.33 hours by the time the forecast is published, and the quoted `spot` is
   the close of the hour before that. The window the user is shown is not the
   window they can still act on.

### 3.2 Trailing features are an hour staler than the contract states

`features.py`'s header promises "hourly rows with index ≤ a−1". It delivers
≤ a−2: `_trailing_sum(x,k)[i]` is already `sum(x[i-k:i])`, exclusive of `i`,
and the result was *then* indexed at `a−1` as well. Verified numerically.

Consequence: the last complete hour before the anchor is discarded on every
episode, in training and serving alike, and `har_1h` — a one-hour window — is
not the last hour's realized volatility but the hour before it.

The double shift was defensive and safe in the only direction that matters,
but it is redundant: `audit_lookahead()` verifies the contract *numerically*
by corrupting the future, and passes at both settings with max feature change
0.000e+00 over 200,000 probed episodes. `build_features(...,
extra_lag_hours=)` now exposes the choice; the default reproduces the
committed `features.parquet` bit-for-bit. `eval/freshness.py` measures whether
recovering the hour is worth anything.

### 3.3 Two live bugs in the v1 training path, dormant only by luck

`train.py` is a second file that can produce a servable artifact, and
`runtime.load_model()` promotes it automatically if `noctua_v2.npz` is ever
missing. It still contains both defects that were found and fixed in
`train_v2.py` and never back-ported:

- **`train.py:271` writes `"feat_cols": list(X.columns)`** rather than the
  columns actually consumed. This is exactly the bug that made the artifact
  declare 42 inputs while its weights expected 39 and killed serving with a
  matmul error. It is dormant only because the on-disk v1 checkpoint predates
  the `eff_*` research columns. Regenerating it today reproduces the crash.
- **`train.py`'s `main()` never passes `sigma_ref`** (verified: lines 236,
  237, 255), so Stage B falls back to `sigma = RV` — fitting against realized
  volatility while serving against forecast volatility, the train/serve skew
  the whole `serve_consistent` work existed to remove. This one would *not*
  crash. It would silently ship too-tight quantiles.

### 3.4 Stage B is queried outside its trained σ range about 2 % of the time

The `sigma_ref` clip bounds are fitted on the training split
(`[0.004916, 0.130733]`) and never reapplied to `sigma_atoms` at serve time.
Measured:

| split | below floor | above ceiling |
|---|---|---|
| train | 0.50 % | 0.50 % |
| calib | 3.72 % | 0.00 % |
| test | **2.01 %** | 0.00 % |

Real but minor, and entirely at the calm end — the model is asked about
quieter conditions than Stage B's training range covered, never wilder ones.
Worth a bound, not an alarm.

### 3.5 Documentation that had drifted from the artifacts

Verified against `model/artifacts/*.json` and the shipped `.npz`:

| claim | where | actual | disposition |
|---|---|---|---|
| "49,866-parameter model" | `README.md` | 6,445/seed, 19,335 across 3 seeds (49,866 is v1) | corrected |
| barrier error "0.94 pp" | `README.md` | shipped committee is 1.09 pp at α = 1 % | corrected |
| "Kronos: not benchmarked — blocked" | `README.md` | benchmarked, 120 episodes | corrected |
| "114.2 s/episode" | `BENCHMARK.md` | 114.784 | corrected |
| Kronos "20–60 s" | `RESULTS.md` | an estimate; later measured at 114.8 s | annotated |
| whole file reads as current | `RESULTS.md` | describes v1 | scope banner added |

### 3.6 What was checked and found clean

Recording these because confirming correctness is worth as much as finding a
defect, and because an audit that only lists problems gives no sense of scale.

- **No look-ahead leak in any feature.** Every trailing window verified to end
  strictly before the anchor; the numeric audit passes with max change exactly
  0.0 over 200,000 probed episodes. The one forward-looking construct,
  `cal_weekend_frac`, is pure calendar arithmetic that touches no price series.
- **The purge/embargo is correct**: `max(H) = 24` hours, applied symmetrically
  at both the train/calib and calib/test boundaries, in `time_splits` and in
  `walk_forward_folds`.
- **Standardization is fitted on train only**, persisted, and applied at serve
  time by an identical formula that never recomputes from serving data.
- **The v2 artifact's declared columns match its weight shapes** (39 = 39, no
  `eff_*`), and training and serving call the *same* `build_features`, so
  feature computation cannot skew between them.
- **The adaptive volatility correction is strictly causal** — estimated only
  from episodes that have already settled, median rather than mean.
- **`upside` is pinned to 50.0 on purpose**, with the raw `p_up` exposed as an
  inert field and an explicit warning in the payload, because `src/data.js`
  would otherwise pipe it into strike selection. This was the right call made
  on weaker evidence than §2 now provides.

---

## 4. What is stopping the model, in one paragraph each

**Direction is not blocked by the model.** It is blocked by the information
content of hourly OHLCV at a 6–24 hour horizon. §2 measures the ceiling rather
than asserting it, and the literature in §2.5 says the ceiling is where it is
for everyone. The work that would move it is a change of *data*, not of
architecture.

**Volatility is not blocked; it is bounded by two ceilings, and the second one
is the real answer to "what is stopping this model".**

*Statistical ceiling:* six folds on ~365 production episodes each cannot
resolve 1–2 % effects, which is why the pooling result moved when a defect was
fixed and why `t`-like statistics near 2 have repeatedly failed to replicate.
The binding constraint there is independent test episodes, not capacity — the
capacity sweep already showed more parameters do not help.

*Structural ceiling, and the more important one:* on the barrier question the
product is actually sold on, **a perfect volatility forecast is worth only
6–8.5 %** (§5c). Ninety-two per cent of the error is the shape of the
excursion — how far a path travels per unit of volatility — and BTC does not
travel like Brownian motion. So "make the volatility model as accurate as
possible" has a measured, hard limit that further volatility work cannot pass.
The leverage is in Stage B: the excursion heads, the mixing integral, and
features describing path *geometry* rather than path *size*.

**The product claim is bounded by §3.1.** Until the benchmark is measured at
the anchors actually served, the honest scope of every headline is "at 17:00
UTC", and that is 5 % of what ships.

---

## 5. Next research path

Ordered by expected value per unit of work, with the decision rule that ends
each one written before it starts.

### 5.1 Close the served-configuration gap — **DONE, and the claim generalizes**

The decision rule, written before the run: *if QLIKE-vs-HAR at served anchors
is within 1 pp of the 17:00 figure, restate the claim; otherwise fix the
serving anchor.*

Six walk-forward folds, three arms, identical models per fold (`run_fold` is
deterministic given its seeds, so only the scored slice changes). The wide arms
are subsampled to 6,000 episodes drawn once with a fixed seed, which matches
their per-fold sample size to the 17:00 arm's and makes the comparison paired
on the anchor rather than on power.

| arm | DSC/UNC | QLIKE | **vs Log-HAR** | ΔDSC vs 17:00 | ΔQLIKE vs 17:00 |
|---|---|---|---|---|---|
| 17:00 — the benchmarked slice | 0.05526 | 0.2896 | **−6.14 %** | — | — |
| all served anchors | 0.06196 | 0.2715 | **−6.06 %** | +0.0067 | −6.25 % |
| every anchor except 17:00 | 0.06465 | 0.2668 | **−7.13 %** | +0.0094 | −7.90 % |

**The product claim holds at the anchors that ship.** NOCTUA's advantage over
a calibrated Log-HAR is 6.14 % at the benchmarked anchor and 6.06 % across all
served anchors — a gap of **0.08 pp**, comfortably inside the 1 pp rule.
Barrier discrimination is not merely preserved but *higher* away from 17:00
(0.06465 against 0.05526).

The absolute QLIKE differences between arms are a level effect, not a skill
effect: 17:00 UTC sits just after the US equity open, near the intraday
volatility peak, so its episodes are simply harder in absolute terms. The
ratio against Log-HAR — which faces the same episodes — is the comparison that
transfers, and it is flat across anchors.

**So the concern raised in §3.1 is resolved for model validity and remains
open for product honesty.** The benchmark did not need the anchor it was
measured at; the headline generalizes and should now be stated as "across
anchor hours" rather than silently as "at 17:00 UTC". `BLEND_W`, the committee
weights and the adaptive correction were tuned at the peak-volatility anchor
and evidently did not overfit to it.

What is *not* resolved is the second half of §3.1: the forecast is anchored at
the last **closed** hour, so the 19-hour window has been running a median 1.33
hours by the time it is published. That is a product defect independent of
model quality — anchoring at the next hour boundary instead would publish a
window the reader can still act on — and it is the one piece of §3.1 still
worth fixing.

### 5.2 Recover the discarded hour — **DONE, and it is the largest measured gain in the project**

The decision rule, written before the run: *ship `extra_lag_hours=0` only if
it wins on DSC/UNC or QLIKE in ≥ 5 of 6 folds; a 4/6 is noise and gets
reported as null.* It cleared that bar on QLIKE at 6/6.

Same six walk-forward folds, same seeds, same committee, same causal stage-B
reference rebuilt from each arm's own features. One variable changed.

| metric | lag 1 (shipped) | lag 0 (fresh) | relative | folds won | t-like |
|---|---|---|---|---|---|
| **QLIKE** | 0.299623 | **0.289636** | **−3.33 %** | **6/6** | **+4.06** |
| **pinball** | 0.003594 | **0.003574** | −0.54 % | **6/6** | **+4.99** |
| **CRPS** | 0.005277 | **0.005250** | −0.51 % | **6/6** | **+3.63** |
| DSC/UNC | 0.053817 | 0.055256 | +2.67 % | 4/6 | +1.68 |

Barrier discrimination moves in the right direction but 4/6 with t-like +1.68
is not a result, and is reported as the null it is. The volatility and
distributional scores are unambiguous: **6/6 on all three, with the strongest
t-like statistics this project has produced.**

For scale against everything else measured here:

| change | best metric | folds | t-like | shipped? |
|---|---|---|---|---|
| **feature freshness** | **QLIKE −3.33 %** | **6/6** | **+4.06** | **yes** |
| `serve_consistent` stage-B target | DSC/UNC +8.1 % | 6/6 | +3.78 | yes |
| multi-asset pooling | QLIKE −1.87 % | 5/6 | +1.63 | no |
| path-efficiency features | — | 2/6 | — | no |
| more capacity | — | — | — | no |

The mechanism is not mysterious and does not need one: `har_1h` is a
one-hour window, so at the old setting it was not the last hour's realized
volatility but the hour before it — mean absolute shift 0.613 in log-vol
units when corrected. 31 of 42 features move. The causality audit passes at
the new setting with max feature change 0.000e+00 over 306,261 episodes, so
this is information recovery, not a loosened constraint.

### 5.3 Back-port the two v1 fixes, or delete the v1 path

§3.3. The cheapest correct action is to delete `train.py`'s `main()` and its
`noctua_weights.npz` artifact so there is exactly one path that can produce a
servable model. If v1 is kept as a fallback, port both fixes and add a CI
check that fails if the v1 artifact is regenerated from the unpatched path.

### 5.4 Direction: buy the data or drop the goal *(the real decision)*

The literature is consistent that short-horizon directional information lives
in order flow, not in bars. If direction is to be pursued seriously, the
minimum viable input set is:

- **order-book imbalance and signed trade flow** at 1 s–1 min, aggregated to
  the anchor — the only source with measured, replicated directional content
  in crypto;
- **perpetual funding rate and basis** — a genuine institutional signal,
  though the literature is clear it is a carry/risk-premium signal rather than
  a directional one;
- **options skew** from Deribit — no verified evidence found that it carries
  *directional* information at this horizon, so it should enter as a
  hypothesis to falsify, not as an assumed feature.

**Decision rule, set now to prevent a fishing expedition:** the honest test is
whether the ceiling arm in `eval/direction.py`, refit with order-flow
features, clears the shuffled DSC null *and* beats the base rate on log loss
with a block-bootstrap CI excluding zero, on the production slice. If it does
not, direction is closed as a goal and the model is described accurately as a
volatility-and-excursion forecaster. Given Cont–Cucuringu–Zhang's decay
result, the prior on success at 6–24 h should be **low**, and the cost of the
data should be weighed against that prior before it is bought.

### 5.5 Raise evaluation power, since it now binds

Every remaining volatility question is limited by ~2,046 production episodes.
Options, in order of preference: score the wide H = 19 slice with the block
bootstrap already written for §2 (24× the episodes at the cost of overlap);
extend the harvest further back than 2017-08; or adopt the multi-horizon grid
as the primary evaluation with expiry as a reported dimension rather than a
filter. This is the enabling work for everything else — a 1 % effect cannot be
confirmed at the current sample size no matter how good the model is.

### 5.6 Do not, on present evidence

- **Multi-asset pooling.** Measured, corrected, and gone by the deployed
  expiry (`BENCHMARK.md` §6c).
- **More capacity.** The sweep says it does not help.
- **Path-efficiency features.** Ablated, null.
- **ACI self-improvement.** The guard vetoed its own candidate at
  e = 9.75e-186; the mechanism works, the candidate was worthless.

---

## 5b. Inside the network — is it learning correctly, or scoring well by accident?

An instrumented training run, reported with its caveats attached because they
change what may be concluded. **Configuration: hidden 64, a 50,000-row
training subsample, 18 epochs.** The shipped model is hidden 32 on 189,831
rows for 40 epochs. Relative sizes of loss terms and importance *rankings*
should survive that reduction; anything about overfitting or absolute loss
levels should not, and is not relied on below.

**HEALTHY — the neural residual is not decorative.** The obvious failure mode
for a model with a linear Log-HAR base plus a learned residual is that the
linear term does all the work. Measured: residual/base variance ratio
**0.322**, with a base–residual correlation of **0.043**. The network
contributes about a third of the base's variance and does so almost
orthogonally to it — it is adding information rather than re-fitting the
regression it was initialised from.

**HEALTHY — seeds agree.** Best validation loss across three seeds 0.42583 /
0.42626 / 0.42858, spread **0.00275**. Permutation-importance rank
correlation between seeds **0.78–0.83**. Early stopping fires at epoch 4–5 in
this configuration and the tracked best state is restored.

**A FINDING I REJECTED, and the reason matters more than the finding.** The
permutation ranking puts `cal_dow_sin` top of the Stage A inputs at **+16.7 %
of loss** — three times `har_6h`, six times `har_1d`. Read naively that says
the day of the week matters more to this volatility model than the HAR
cascade does, which would be alarming.

It is an artifact of the method. Permutation importance splits credit among
correlated predictors, and the HAR cascade is strongly intercorrelated —
measured on the real data:

| | har_1h | har_6h | har_1d | har_5d | har_22d |
|---|---|---|---|---|---|
| har_1d | 0.302 | 0.654 | 1.000 | **0.851** | 0.753 |
| har_22d | 0.151 | 0.462 | 0.753 | **0.880** | 1.000 |

Permuting `har_1d` leaves `har_5d` (r = 0.851) and `har_22d` (r = 0.753) to
carry the same signal, so the measured loss increase is small. `cal_dow_sin`
is essentially orthogonal to the whole block (|r| ≤ 0.094), so permuting it
destroys information nothing else holds and it collects full credit. The
ranking measures *redundancy*, not importance, and the two are not the same.
**Nothing was changed on the strength of it.**

**SUSPICIOUS — tested, and the result is NULL.** `pinball_loss(q_r, r)` — the
terminal-return head — is the **largest single term in the objective**,
measured at the validation minimum: `a = 0.0654, r = 0.1560, up = 0.1059,
dn = 0.0985`. Roughly 2.4× the stage-A volatility term. Combined with §2
(the sign is not predictable) and §5c (92 % of the barrier error is excursion
shape), the model appeared to be spending its largest share of gradient budget
on the one quantity it cannot forecast.

`eval/losshead.py` down-weights it, six folds, adopt rule ≥ 5/6 on DSC/UNC
fixed before the run:

| `lam_r` | DSC/UNC | pinball | CRPS | QLIKE | folds better than shipped | t-like |
|---|---|---|---|---|---|---|
| **1.0 (shipped)** | 0.05526 | 0.003574 | 0.005250 | 0.2896 | — | — |
| 0.25 | 0.05554 | 0.003574 | 0.005249 | 0.2897 | **4/6** | +1.46 |
| 0.0 | 0.05568 | 0.003577 | 0.005249 | 0.2894 | **4/6** | +0.78 |

**4 of 6 does not clear the rule, so this is reported as the null it is and
nothing ships.** The direction is consistently favourable — every arm's mean
DSC/UNC is above the shipped one, and the first fold looked convincing at
0.04155 → 0.04295 — but +0.5 % relative on 4/6 folds with t-like +1.46 is
exactly the size of effect this project has already watched fail to replicate
twice (the pooling result, the path-efficiency features).

Worth stating plainly because it is the third time: a hypothesis motivated by
three independent measurements, with a favourable first fold, still did not
survive six. The pre-registered rule is what stopped it from being written up
as a win, and that is the rule's whole purpose.

The cost of the return head is therefore **not** the capacity it consumes —
that turns out to be nearly free. It stays at 1.0.

**INERT.** The coupling penalty falls to exactly 0.00000 from epoch 3 onward
(from 0.00332 at initialisation). The path identities it enforces are
satisfied almost immediately and it contributes nothing to the gradient
thereafter. Not harmful — it is a constraint, and a satisfied constraint
costing nothing is the desired state — but it is not the regulariser it might
be assumed to be, and that matters for the `lam_r` ablation, where deleting
`q_r` outright would remove it.

**Candidates for pruning, not yet tested.** Seven Stage A inputs and six
Stage B inputs have *negative* permutation importance — permuting them
improves held-out loss. Stage A: `vov_5d` (−0.44 %), `vov_22d` (−0.25 %),
`cal_month_cos`, `semi_signed_jump_5d`, `reg_post_etf`, `semi_neg_share_1d`,
`mom_dist_ma100`. The magnitudes are small but well outside the reported
per-feature noise (std ~1e-5). Worth an ablation; not acted on yet.

**One literature warning that does NOT apply here.** Brini (2026) reports
realized-quarticity (HARQ) terms being badly unstable out of sample — QLIKE
ratios up to 5.13×. This model's `rq_noise_1d` has *positive* importance
(+1.87 % of Stage A loss). The warning is real in general and does not
transfer to this implementation, which is worth recording so it is not
"fixed" on the strength of a citation.

---

## 5c. Where the barrier error actually lives

Moved to `BENCHMARK.md` §6f in full. The headline, because it governs §7:
feeding the Gaussian first-passage law the **realized** volatility — perfect
foresight — still leaves 91.5–94 % of the causal forecaster's barrier error
in place. Volatility work has a hard ceiling of **6–8.5 %** on the task the
product is sold on. BTC's range per unit of realized volatility is **1.331**
against the Brownian **1.5958** (CI on the difference [−0.298, −0.234]): the
price chops rather than travels, and the textbook barrier formula overstates
touch risk at every strike from 0.5 % to 5 % even when handed the right sigma.

---

## 6. Where the agent reports were wrong

Kept because an audit that hides its own error rate is not an audit.

- **"`q_r` is a by-product; nothing in the loss rewards the sign."** I wrote
  this into `eval/direction.py`'s docstring before checking. It is false —
  `train.py:179` scores `q_r` with pinball loss on the signed return.
  Corrected before the file was committed. It matters: it moves the
  explanation for the null result from "architectural neglect" to "the signal
  is not there".
- **The doc-inventory agent reported a "10× latency discrepancy"** between
  RESULTS.md (~6 ms) and BENCHMARK.md (0.6 ms). Both are plausibly correct for
  their respective models — v1 is 7.7× larger than a v2 seed — so this is a
  scope ambiguity, not a contradiction. Fixed with a scope banner rather than
  a number change.
- **The doc-inventory agent's "0.94 pp" mismatch was real but its explanation
  was not** — the number does not correspond to any single α in `eval.json`
  (nearest is α = 2 % at 0.9103, mean across α is 1.9007). It is stale and
  untraceable, which is worse than merely stale, and is why the README now
  points at `BENCHMARK.md` as the source of truth.
- **The leakage agent's "dormant landmine"** in `ingest.py` — a forward-looking
  `next_step` in the bad-print detector — is correctly diagnosed as inert:
  it feeds only `M_up_clean`/`M_dn_clean`, which are used nowhere. Confirmed
  by grep. Recorded here so that wiring those columns into training later is
  known to require fixing the detector first.
