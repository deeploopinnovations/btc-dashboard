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

### 3.1 The model is served at anchors it was never scored at — **highest-consequence finding**

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
   claim-versus-product mismatch. `eval/anchors.py` measures whether the
   benchmark survives at the served anchors.
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

**Volatility is not blocked; it is bounded by evaluation power.** Six folds on
~365 production episodes each is a small sample for distinguishing 1–2 %
effects, which is why the pooling result moved when a defect was fixed and why
`t`-like statistics near 2 have repeatedly failed to replicate. The binding
constraint on further volatility work is the number of independent test
episodes, not model capacity — the capacity sweep already showed more
parameters do not help.

**The product claim is bounded by §3.1.** Until the benchmark is measured at
the anchors actually served, the honest scope of every headline is "at 17:00
UTC", and that is 5 % of what ships.

---

## 5. Next research path

Ordered by expected value per unit of work, with the decision rule that ends
each one written before it starts.

### 5.1 Close the served-configuration gap *(cheap, highest value)*

Run `eval/anchors.py`. If the benchmark holds across anchor hours, the
headline generalizes and the claim can be restated honestly. If it does not,
either pin serving to the benchmarked anchor via the already-written
`_next_anchor()`, or re-tune `BLEND_W`, the committee weights and the adaptive
correction per anchor bucket. **Decision rule:** if QLIKE-vs-HAR at served
anchors is within 1 pp of the 17:00 figure, restate the claim; otherwise fix
the serving anchor.

Separately, anchor the forecast at the *next* hour boundary rather than the
last closed one, so the published window is one the reader can still act on.

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
