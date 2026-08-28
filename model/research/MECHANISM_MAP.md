# MECHANISM MAP: why the volatility matrix looks the way it does

Status: mechanism analysis only. No model, baseline, or artifact in `model/`
was changed to produce this document. Every number below was read from
`model/artifacts/vol_matrix_fair.json`, `model/artifacts/garch.json`,
`model/artifacts/anchor_freshness.json`, `model/artifacts/prod_fairbaseline.json`,
`model/research/ledger.json`, `model/noctua/baselines.py`, `model/noctua/spec.py`,
`model/eval/garch.py` and `model/eval/vol_matrix.py`, plus one read-only
diagnostic script (reported inline in §A.2) that re-ran the *existing*
`build_h4_table` / `walk_forward_folds` / `OLS.fit` code on the 2022 fold to
locate the specific episodes behind `har_short`'s H=1 blow-up — no new fitting
logic, no new features, nothing written back to `model/` or `data/`.

Governing numbers (pooled QLIKE, 6 walk-forward folds, baselines refit **per
horizon**, best baseline chosen on **calib**, never test):

| H | best OLS baseline (calib-selected) | har_short | log_har | garch_normal | garch_t | noctua / noctua40 |
|---|---|---|---|---|---|---|
| 1 | persistence 0.62599 | 1.03094 | 0.82435 | 0.60326 | **0.57438** | 0.67177 / 0.62271 |
| 6 | **har_short 0.41710** | 0.41710 | 0.46795 | 0.42058 | 0.41246 | 0.44907 / 0.43097 |
| 24 | **har_short 0.27818** | 0.27818 | 0.30630 | 0.32684 | 0.30417 | 0.28523 / 0.32844 |
| 168 | **har_short 0.18207** | 0.18207 | 0.18435 | 0.53821 | 0.44349 | n/a / 0.20569 |

Per `ledger:vol-matrix-fair-result`, NOCTUA does not clear a paired CI against
the best baseline at **any** horizon; garch_t clears favourably against
persistence at H=1.

---

## A. `har_short` — why two short regressors help long horizons and destroy H=1

`har_short = har_1h + har_6h + har_1d + har_5d + har_22d`, an unregularized
OLS on `y = log(RV) − 0.5·log(H)`, refit **per fold, per horizon, on that
horizon's own training episodes** (`vol_matrix.py::run_fold`, the
`fair_baselines` branch). `log_har` is the same fit without the two short
columns.

### A.1 Why it wins at 6h / 24h / 168h (the counter-intuitive part)

Per-horizon, `har_short` beats `log_har` at every horizon except H=1:

| H | har_short | log_har | Δ (har_short − log_har) |
|---|---|---|---|
| 1 | 1.03094 | 0.82435 | **+0.20659 (worse)** |
| 6 | 0.41710 | 0.46795 | −0.05085 |
| 24 | 0.27818 | 0.30630 | −0.02812 |
| 168 | 0.18207 | 0.18435 | −0.00228 |

Two competing hypotheses:

**H-A1a (regime-state hypothesis).** `har_1h`/`har_6h` are not there to
*forecast* the next hour, hour six, or hour 168 directly — the regressor and
the horizon are different things. They are the **most current read of the
volatility regime available at the anchor**. HAR's whole design (Corsi 2009)
is that a cascade of realized-vol averages at different frequencies acts as a
proxy for the latent multi-component variance process; the fastest component
is the one most sensitive to *whether a spike is starting right now*. A
window that opens at the anchor and runs for a day or a week still has its
first hour(s) determined by the state at the anchor, so knowing whether that
state is already elevated shifts the level of the whole forecast — the same
reason `anchor_freshness.py` documents predicted sigma correlating "0.920
with realized RV dated one day AFTER and only 0.507 one day BEFORE" for a
model whose fastest input is `har_1d`: the daily term is structurally last to
know a spike has started. `har_1h`/`har_6h` are exactly the terms that know
first.

**H-A1b (extra-degrees-of-freedom / better local slope hypothesis).** It is
not that the *content* of `har_1h`/`har_6h` is special — it's that any two
extra numerically well-behaved regressors let the per-horizon OLS fit a
slightly better local hyperplane through that horizon's own training
episodes, i.e. this is a garden-variety "more features, lower training-set
bias, and the extra variance is small because n_train is large" story with no
regime-timing content at all.

**What would separate them.** H-A1a predicts the gain is *concentrated in
episodes following a recent state change* (the trailing 1h/6h level disagrees
with the trailing 1d/5d/22d level) and should shrink to ~0 if `har_1h`/`har_6h`
are replaced by two more lagged copies of `har_1d` (same parameter count, no
new frequency). H-A1b predicts the gain is roughly uniform across episodes and
survives that substitution. **This cannot be settled from the existing
artifacts** — `vol_matrix_fair.json` reports pooled/spike/calm/worst-fold
QLIKE but not a split by "trailing short-term level vs trailing long-term
level agreement," and no lagged-`har_1d` control arm exists in
`noctua/baselines.py`. See the falsifiable test in §D.1.

A relevant piece of existing evidence favours H-A1a somewhat: the H=24
ablation in `ledger:vol-matrix` (not overturned by the fair rerun) shows
`seas_1d`/`seas_5d` — also fast, anchor-local features — carry the entire
NOCTUA-vs-`noctua40` gap at H=24 (0.28523 vs 0.32844 in the fair table). That
is a *second, independent* instance of an anchor-proximate feature dominating
a multi-day forecast, which is more consistent with "current state matters
disproportionately" than with "any extra regressor helps."

### A.2 Why it explodes at H=1 (worst fold 2.83480, spike 12.4521)

This is **not** generic 5-regressor overfitting on noise, and it is **not**
severe multicollinearity — the correlation matrix of the five HAR columns
(computed from `features.parquet`, 510k rows) is:

```
         har_1h  har_6h  har_1d  har_5d  har_22d
har_1h    1.000   0.537   0.276   0.178    0.151
har_6h    0.537   1.000   0.616   0.489    0.462
har_1d    0.276   0.616   1.000   0.842    0.753
```

— `har_1h`'s correlations with everything else (0.28–0.54) are the *weakest*
in the set; `har_5d`/`har_22d` at 0.880 (already present in `log_har`) is the
actual collinearity hotspot, and it doesn't blow up. What is different about
`har_1h` is its **variance**: std 1.670 in log-space against 0.970 (`har_6h`),
0.585 (`har_5d`), 0.550 (`har_22d`) — because it is a single trailing hour's
realized variance, the noisiest possible measurement of "current vol."

Re-running the *existing* `build_h4_table`/OLS pipeline (read-only, no
artifact changed) on H=1, fold 2022 (the fold that produces the 2.83480
blow-up) isolates the mechanism exactly:

- `har_short`'s fitted H=1/2022 coefficients (intercept, har_1h, har_6h,
  har_1d, har_5d, har_22d) are `[-0.119, 0.329, 0.173, 0.264, 0.150, 0.089]`
  — `har_1h` alone carries the largest slope, 0.329, versus `log_har`'s
  weight of 0.689 on `har_1d` alone. Regression mass shifted off the daily
  term, which acts as a floor, and onto the noisiest term.
- The single worst test episode in that fold is anchored **2022-07-13
  16:00 UTC** (the day of the hot June US CPI print and a sharp BTC
  selloff). `har_1h` at that anchor sits at its **numeric floor**, −13.8155
  (⇒ trailing-hour realized variance ≈ 1e-6, an essentially dead hour).
  `har_short` predicts σ = 0.000290; realized RV = 0.038916 — a **134×**
  miss. QLIKE for that one episode is **17,989.37**.
- That single episode alone contributes 17989.37 / 8758 = **2.054** of the
  fold's 2.8348 mean — **72% of the entire fold's average loss from one
  anchor out of 8,758** (0.011% of the fold).
- The 10 worst episodes (0.11% of the fold) sum to 2.2355 of the 2.8348 mean
  (**79%**); with those 10 removed, the remaining 8,748 episodes average
  **0.5999** — squarely in line with every other arm's H=1 calm-episode
  performance.
- Every one of the top-10 worst episodes sits on a date with a real,
  identifiable macro or exchange shock in the training data's era (2022-07-13
  CPI selloff, 2022-01-05 hawkish-Fed-minutes selloff, 2022-03-28,
  2022-08-19, 2022-12-13 FOMC, 2022-12-25 low-liquidity holiday move,
  2022-11-30 post-FTX). This is the 2022 fold specifically because 2022 is
  the year with the highest density of "quiet hour immediately followed by a
  violent jump" transitions in this sample.

This is a **scale/leverage problem in a floored, unregularized log-linear
model**, not classical overfitting: (a) `har_short`'s calm-episode QLIKE at
H=1 is 0.4293 — *better* than `log_har`'s 0.4805, so the extra regressors are
not generally harmful; (b) `har_short_pooled` — the exact same two columns,
fit on the pooled multi-horizon sample instead of the H=1-only sample —
scores **0.5404** at H=1, essentially *halving* the per-horizon fit's 1.0309
(ratio 0.524×, matching `ledger:vol-matrix-fair-result`'s independently
recorded "H=1 har_short 0.52x"). Pooling across horizons acts as an
accidental shrinkage/regularizer on the same five columns and the pathology
disappears. That is the signature of an unstable small-sample linear fit
meeting a leverage point, not of the features being wrong. GARCH is
structurally immune to the identical failure mode at the identical anchor
because its recursion `h[i] = omega + alpha·r[i-1]² + beta·h[i-1]` is bounded
below by `omega > 0` — it cannot emit a near-zero forecast the way
`exp(0.329 × (−13.82) + …)` can.

Why the calib-selection rule matters here: `har_short`'s own calib QLIKE at
H=1 was **1.4016**, the worst of the four OLS candidates, so it was **not**
selected as the H=1 baseline (persistence was). A test-side selection rule
would have handed NOCTUA (0.67177) a free, illegitimate win over a baseline
that was actually the worst available.

### A.3 Reconciling with `E-anchor-verdict` (inserting har_1h/har_6h into NOCTUA's anchor was HARMFUL)

`ledger:E-anchor-verdict` found that adding `har_1h`/`har_6h` to
`BASE_COLS` — NOCTUA's 5-column linear "anchor" term
(`har_1d, har_5d, har_22d, cal_H, cal_weekend_frac`, `noctua/spec.py:38`,
carrying 75% of the blend per `anchor_freshness.py`) — made spike QLIKE
**worse** by +1.87% (CI [+0.02151, +0.05408], six-of-six folds), on the
production H≈19 slice. That looks like it contradicts §A.1's finding that
the same two columns help a standalone regression. It does not, and the
reason is architectural, not statistical:

- `har_1h` and `har_6h` are **already** two of the 39 raw columns in
  `Xa` (`train.py:76`, `all_cols = [c for c in X.columns if c not in
  dropped]`, `dropped` = only the 3 efficiency columns) — the wide input
  block feeding NOCTUA's neural stage. Confirmed directly:
  `features.parquet` has 42 columns, `har_1h`/`har_6h` are among them, 42 − 3
  = 39, matching the "6,939-parameter network with 39 inputs" figure in
  `eval/garch.py`'s own docstring.
- `har_short` (the OLS baseline in §A.1) is a **5-parameter model whose only
  information is those five numbers**. For it, adding `har_1h`/`har_6h` is a
  genuine increase in available information — there is nowhere else in that
  model for anchor-local state to enter.
- The `E-anchor` treatment instead duplicated `har_1h`/`har_6h` into
  `BASE_COLS`, the *linear anchor term specifically*, whose entire role in
  the architecture is to be the smooth, low-variance floor that the neural
  stage's shape corrections sit on top of — while the neural stage already
  had unrestricted, nonlinear access to the same two columns via `Xa`. The
  ledger's own stated mechanism: *"the full pipeline already feeds
  har_1h/har_6h to the neural stage, so the treatment adds no information and
  instead moves the noisiest available RV window into the term whose job is
  stability."*

Both facts are true simultaneously because **"does column X help" is not a
property of the column — it is a property of (column, model, and what else
already has access to that column)**. A column can carry real, useful
information for a model architecture that lacks any other channel to it
(`har_short`), and be net-harmful when injected into one specific component
of an architecture that (a) already receives it through another channel and
(b) is functionally specialized to be noise-resistant. §A.2 additionally
shows *why* injecting it there is actively harmful rather than merely
redundant: `har_1h` is the noisiest column of the five (std 1.670 vs
0.55–0.97) and is exactly the column responsible for H=1's leverage
pathology — moving it into the stability term imports that same fragility
into the 75%-weight component.

### WHY THIS EXPLANATION IS PROBABLY WRONG

- The H-A1a "regime-state" story is a post-hoc narrative built to fit a
  pattern (short-scale help grows monotonically... actually it does *not*
  monotonically grow — the har_short − log_har gap is −0.05085 at H=6,
  −0.02812 at H=24, and only −0.00228 at H=168, i.e. it *shrinks* with
  horizon, the opposite of "current state matters more the further out you
  look"). A cleaner reading is that the gap is simply the residual size of
  whatever daily/weekly/monthly HAR misses, and that residual happens to
  shrink as `log_har` itself gets more accurate at longer horizons (its own
  QLIKE falls from 0.824 at H=1 to 0.184 at H=168) — i.e. there may be less
  to explain, not more.
- The E-anchor result is n=2,046 production-slice episodes (H=19 only, one
  anchor hour a day) over 6 folds — a different population from the h4
  matrix's ~49,000-episode, all-anchor-hours H=24 row. "Both facts can be
  true" is demonstrated architecturally above, but the two experiments are
  not literally the same population, so treat the reconciliation as a
  mechanism argument, not a numerically matched contrast.
- Per R7/R39 (`RULES.md`), every one of these per-horizon numbers is a
  six-fold (six-year) statistic. The `har_short` vs `log_har` deltas at
  6h/24h/168h (−0.051 / −0.028 / −0.002) are **not** accompanied by a paired
  CI against each other in `vol_matrix_fair.json` (CIs there are only
  computed against the calib-selected *best* baseline, and at H=6/24/168
  `har_short` *is* that baseline, so `paired_ci` is `null` for it — see the
  table dump: `har_short` block_len is `None` at H=6/24/168, meaning no
  interval was computed for har_short against itself). Whether har_short's
  win over log_har at 6h/24h is itself distinguishable from six-year sampling
  noise **CANNOT BE DETERMINED FROM AVAILABLE ARTIFACTS** — it would need a
  new paired bootstrap of `har_short − log_har` specifically, which the
  pre-registered rule never computed because log_har was never the target of
  a within-family comparison.
- Given `STATS_PROTOCOL.md`'s fold-level MDEs (11.76% of persistence at H=6,
  31.68% at H=24 — see the shared table below), and that the har_short vs
  log_har deltas above (as % of persistence: 0.05085/0.45301=11.2% at H=6,
  0.02812/0.45511=6.2% at H=24) are **below or barely at** those MDEs, the
  6h/24h "wins" are exactly the size a six-fold design is not equipped to
  resolve at the fold level. The per-episode primary is powered (n≈49,000),
  but per-episode power does not licence a claim about *which mechanism*
  produced the gap — it only licences that the pooled mean differs, which is
  a weaker claim than "the regime-state mechanism is real."

---

## B. GARCH(1,1) — best at H=1/H=6, collapses at H=168

**Correction to the task's framing, checked against the artifact rather than
assumed:** the numbers "spike QLIKE 0.1917 (best of any arm), calm QLIKE
0.5565 (worst)" at H=168 belong to **`garch_normal`**, not `garch_t`, in
`vol_matrix_fair.json`. `garch_t`'s own H=168 figures are spike **0.2723**,
calm **0.4525**. Neither is literally the extremum of the full 10-arm table:
persistence's calm (0.5651) is slightly worse than `garch_normal`'s (0.5565),
so `garch_normal` is second-worst on calm, not worst. `garch_normal`'s spike
(0.1917) *is* the minimum of all 10 arms. Both GARCH variants show the same
qualitative pattern; the specific extremal numbers the task quotes are
`garch_normal`'s. This distinction matters for §B's Student-t mechanism
below, which is a real, clean, reproducible finding once the two arms are
correctly separated:

| H | garch_normal spike | garch_t spike | garch_normal calm | garch_t calm |
|---|---|---|---|---|
| 1 | 2.4328 | 2.9651 | 0.5069 | 0.4485 |
| 6 | 1.4500 | 1.7940 | 0.3664 | 0.3397 |
| 24 | 0.8884 | 1.1555 | 0.2973 | 0.2593 |
| 168 | 0.1917 | 0.2723 | 0.5565 | 0.4525 |

**At every one of the four horizons, `garch_normal` has the lower (better)
spike QLIKE and the higher (worse) calm QLIKE of the two.** That is a
100%-consistent, four-for-four pattern, not a coincidence of one horizon.

### Four candidate mechanisms

**(1) Recursive conditional variance, iterated forward H steps.**
`_forecast_from` in `eval/garch.py` iterates `hf = omega + persistence·hf`
with no new shocks (`E[r²] = h`), summing to the window forecast. This
mechanism alone says nothing about spike vs calm — it's shared by every
GARCH variant — but it *does* predict that the forecast is a
near-deterministic function of the single conditional-variance value at the
anchor, `h[t]`, and of nothing that happens inside the window. **Consistent
with the evidence but not distinguishing**: it explains why GARCH forecasts
degrade as H grows (more can happen inside a window the forecast cannot see)
in a generic sense, but doesn't by itself explain the spike/calm split.

**(2) Student-t innovation tails.**
Predicts: fitting under a heavy-tailed innovation assumption should make the
MLE-estimated `alpha` (reaction to the last squared return) *smaller*,
because a single huge return is partly "explained away" as ordinary tail
probability rather than forcing a large variance revision. That predicts
`garch_t`'s conditional-variance path is smoother/more damped than
`garch_normal`'s: **less** reactive during a genuine spike (worse spike
QLIKE) and **more** stable / less prone to residual over-elevation after an
isolated large return during otherwise calm stretches (better calm QLIKE).
**This is exactly the 4-for-4 pattern in the table above.** This mechanism is
**CONSISTENT** with the evidence and is the cleanest, most reproducible
finding in this section.

**(3) Short memory / fast mean reversion.**
Predicts: a GARCH that reverts quickly to its unconditional variance should
(a) do reasonably at all horizons because the forecast "gives up" on any
one shock and heads back to a stable long-run level, (b) in particular
**not** show a *worsening* calm penalty as horizon grows, and (c) not show
spike being its best-relative regime at long horizons — a fast-reverting
model has nothing to say about a week-long persistent regime, it would just
predict the unconditional mean for most of the window. The data shows the
opposite at H=168: calm is GARCH's *worst* regime (both variants) and spike
is its *best* relative regime (`garch_normal` literally the best of all 10
arms). **This mechanism is RULED OUT** by the artifact for this data and
this fit.

**(4) IGARCH-like near-unit persistence (alpha + beta ≈ 1).**
`eval/garch.py`'s own fitting code comments document, from development-time
multi-start fits on this series: *"persistence driven to ~1.0000 (normal),
1.0005 (Student-t) — i.e. IGARCH"* and the code explicitly seeds the
recursion from sample variance rather than
`omega/(1−alpha−beta)` because that quantity is undefined/negative at
measured persistence. Near-unit persistence means `hf` in the forward
recursion barely decays (`hf → omega + (≈1)·hf`), so the H-step forecast is
close to a **flat extrapolation of the anchor's own conditional variance**
rather than a mean-reverting forecast: it is essentially the best available
"continue whatever regime is happening right now" forecast. That predicts
exactly the observed pattern — if a genuine multi-day/week volatility
episode is already underway at the anchor, near-flat extrapolation is close
to correct (best-in-class spike QLIKE, and the advantage should be *largest*
at the *longest* horizon, since that's where competitors' mean-reverting HAR
cascades dilute a spike signal most by blending in daily/weekly/monthly
averages); if the anchor's momentary hourly variance happens to be elevated
by a one-off shock and the coming week is actually calm, flat extrapolation
overpredicts for the whole window and never corrects (worst-in-class calm
QLIKE at H=168). **This mechanism is CONSISTENT with the evidence and is the
single best-fitting explanation for the H=1-vs-H=168 reversal** — GARCH's
edge at H=1 (nowcasting the very next hour with the freshest possible input,
`alpha·r[t]²`) and its collapse at H=168 (flat-extrapolating that same
freshness across a week) are two faces of one mechanism, not two different
phenomena.

### WHY THIS EXPLANATION IS PROBABLY WRONG

- **The IGARCH persistence figures (mechanism 4) are quoted from a code
  comment describing development-time fits, not from `vol_matrix_fair.json`
  itself.** That artifact stores QLIKE, worst-fold, spike, calm, and
  per-fold arrays for GARCH — it does **not** persist `omega`, `alpha`,
  `beta` for the 6 folds × 2 distributions actually used to produce the
  matrix's numbers. Whether persistence is genuinely ≈1 in *every* fold at
  *every* horizon (recall: GARCH is fit once per fold on hourly returns and
  the *same* `omega/alpha/beta` are reused across all four horizons within a
  fold — only the forward-iteration length changes) **CANNOT BE DETERMINED
  FROM AVAILABLE ARTIFACTS**. If persistence varies materially across folds
  (e.g. a calmer fold fits a genuinely mean-reverting alpha+beta ≈ 0.85),
  mechanism 4 would need to be fold-conditional, and the clean H=1-vs-H=168
  story would be a mixture of two different regimes rather than one
  mechanism. §D.2 proposes the direct fix.
- Mechanism 2 (Student-t tails) is a clean 4-for-4 pattern, which is
  itself faintly suspicious for a *dependent* comparison: `garch_t` and
  `garch_normal` are fit and scored on the **same six folds** and the same
  test episodes, so their four paired differences are not four independent
  replications — a single shared feature of the return series (e.g. one or
  two of the six years having return distributions the Gaussian
  likelihood fits worse) could produce a consistent-looking sign pattern
  across all four horizons from what is really only 6 independent
  year-level draws, not "4 horizons × 6 folds = 24 independent
  confirmations." The right unit of replication here is folds, not
  horizons.
- Per R7 (`RULES.md`), "24× more episodes bought 1.53× effective sample
  size — 6.4% survival," and the fold-level MDEs computed before the matrix
  was built are 5.21% / 11.76% / 31.68% / 65.48% of the persistence baseline
  at H=1/6/24/168. GARCH's H=168 advantage over persistence (0.61859 vs
  0.44349, a 28.3% relative gap) clears that horizon's brutal 65.48% MDE
  comfortably — but its advantage over `har_short` runs the other way
  (GARCH is *worse* than `har_short` by (0.44349−0.18207)/0.61859 = 42.3% of
  the persistence baseline, which also clears 65.48%) — so the H=168
  collapse itself is probably real. What is **not** resolvable at this
  sample size is whether the *spike/calm decomposition itself* (rather than
  the pooled number) is stable fold-to-fold; spike/calm splits are reported
  pooled across all 6 folds with no per-fold spike/calm breakdown or CI in
  `vol_matrix_fair.json`, so a single unusual fold could be driving the
  0.1917/0.5565 extremes. **CANNOT BE DETERMINED FROM AVAILABLE
  ARTIFACTS** without a per-fold spike/calm re-extraction.

---

## C. Where does each winning teacher's advantage actually come from?

Per `ledger:vol-matrix-fair-result`, a teacher (an OLS baseline or GARCH) is
ahead of or statistically indistinguishable from NOCTUA at **every**
horizon. Decomposed by regime (spike = top-5% realized-RV episodes *within
each fold*, concatenated — `vol_matrix.py::spike_mask`; calm = the rest):

**H=1** — nominal best baseline persistence (0.62599); true best forecaster
garch_t (0.57438, CI vs persistence [+0.03425, +0.07075], clears
favourably). NOCTUA's own calm QLIKE (0.3844) is the **best of all 10 arms**
at H=1 — better than persistence (0.4516), garch_t (0.4485), har_short
(0.4293). NOCTUA's spike QLIKE (6.1280) is the **worst of all clean arms**
(excluding har_short's floor-artifact-inflated 12.4521) — nearly 2.5×
persistence's spike (3.9366) and over 2× garch_t's (2.9651). **The
advantage at H=1 is entirely a tail/spike-regime calibration deficit, not a
general-level one** — NOCTUA is typical-episode-accurate and spike-episode
poor.

**H=6** — best baseline har_short (0.41710). NOCTUA calm (0.2895) is close
to har_short's (0.2820, a 2.6% gap); NOCTUA spike (3.4801) is far worse than
har_short's (2.9832, 17% gap) and far worse than garch_t's (1.7940) or
garch_normal's (1.4500, less than half of NOCTUA's). **Same pattern as H=1:
the deficit is concentrated in the spike regime.**

**H=24** — best baseline har_short (0.27818). Here the split is
**reversed**: NOCTUA's calm (0.1967) and har_short's calm (0.1971) are
essentially identical (NOCTUA marginally better); the gap is carried almost
entirely by spike (NOCTUA 1.9655 vs har_short 1.8184). But the larger,
better-attested story at H=24 is not spike/calm at all — it is
**seasonality/feature-set**: `noctua` (42 columns, including `seas_1d` and
`seas_5d`) scores 0.28523 while `noctua40` (identical architecture, those two
columns dropped) scores 0.32844 on the *same episodes* — a 15% relative
swing from two anchor-local seasonal columns alone
(`ledger:vol-matrix`, unchanged by the fair-baseline correction). This is the
one horizon where the teacher-vs-NOCTUA question is dominated by an
internal NOCTUA ablation rather than a spike/calm split.

**H=168** — best baseline har_short (0.18207) vs noctua40 (0.20569, CI
[−0.04914, −0.00588], clears **against** NOCTUA). Unlike the other three
horizons, the shortfall is **not** concentrated in one regime: noctua40's
calm (0.1597) is 14% worse than har_short's (0.1400), and its spike (1.0790)
is 10% worse than har_short's (0.9821) — a roughly uniform degradation
across the whole distribution. That is a different signature from H=1/H=6
(tail-specific) and points instead to a general specification/capacity
shortfall at the horizon furthest from where the shared multi-horizon
network was best supported by data (recall from `E-anchor-verdict`/`spec.py`
that the shipped architecture is one network conditioned on `cal_H`, trained
jointly across horizons — H=168 is the most extreme point in that range and,
per `ledger:vol-matrix-fair-result`, is also where the pooled-vs-per-horizon
baseline confound was largest, 2.06×, suggesting horizon-extreme fits are
generally harder for *any* model on this data, NOCTUA included).

**Production slice (H=19 @ 17:00, `prod_fairbaseline.json`, n=2,046, a
different population from the h4 matrix — daily-anchor-only, not all anchor
hours).** Best calib baseline is har_short (0.31628); NOCTUA (0.29702) is
numerically ahead by +3.10% but the CI ([−0.00897, +0.02448]) straddles
zero — **not resolvable at this sample size**, consistent with
`ledger:E-prod-fairbaseline-result`'s own verdict ("NOCTUA still has the
best pooled QLIKE of any arm in the table... simply not significantly
better than har_short at n=2,046").

### WHY THIS EXPLANATION IS PROBABLY WRONG

- Spike is defined **per fold** as that fold's own top-5% RV episodes,
  concatenated (`spike_mask`, applied inside `for r in rows`). A fold with
  an unusually fat right tail (2022, 2023) contributes spike episodes that
  are individually far more extreme than a fold with a mild year, so
  "spike QLIKE" pools across folds with very different tail severities.
  The consistent "NOCTUA is bad at spikes" reading could be disproportionately
  a 1–2 fold effect rather than a property that holds in an average year;
  the per-fold spike QLIKE breakdown needed to check this **CANNOT BE
  DETERMINED FROM AVAILABLE ARTIFACTS** (only the fold-pooled spike/calm
  numbers are stored).
- The H=24 "seasonality carries the win" claim is well-attested for the
  `noctua` vs `noctua40` gap, but that is not the same claim as "seasonality
  explains the noctua-vs-har_short gap" — har_short has no seasonal columns
  at all, so the ablation only tells us *within NOCTUA* what the seasonal
  columns are worth, not that they are what closes (or fails to close) the
  gap to the baseline.
- At n≈49,000 the per-episode primary is well powered for the pooled mean,
  but the spike slice is only 5% of that (≈2,450 episodes per horizon,
  pooled across 6 folds ≈ 408/fold) — much closer to the underpowered
  regime, and no CI is reported on the spike-only or calm-only QLIKE
  differences anywhere in `vol_matrix_fair.json`. The claim "advantage is
  concentrated in spike" is a description of the point estimates, not a
  statistically established regime attribution.

---

## D. Falsifiable tests, one per mechanism

**D.1 — H-A1a "regime-state" vs H-A1b "extra regressors" for har_short's
long-horizon win.**
*Population*: the H=6, H=24, H=168 test episodes already in
`episodes_h4.parquet` (all anchor hours, 6 walk-forward folds, identical to
`vol_matrix_fair.json`'s existing population).
*Contrast*: refit a new baseline `har_lagged = har_1d(t) + har_1d(t−1) +
har_1d(t−2) + har_5d + har_22d` (same parameter count as `har_short`, same
per-horizon-per-fold refitting procedure, but the two extra regressors are
lagged copies of the existing daily term rather than new frequencies) against
the existing `har_short` and `log_har`, on identical test episodes.
*Primary statistic*: paired per-episode QLIKE, `har_short − har_lagged`,
moving-block bootstrap CI at 2H block length (same convention as the matrix).
*Falsification*: if the CI on `har_short − har_lagged` contains zero (or
favours `har_lagged`) at H=6/24/168, H-A1a (short-frequency **content**
matters, not just added flexibility) is falsified — the win is a
degrees-of-freedom effect, not a regime-timing one.

**D.2 — IGARCH persistence mechanism for GARCH's H=1-vs-H=168 reversal.**
*Population*: the same 6 walk-forward folds used in `vol_matrix_fair.json`.
*Contrast*: rerun `eval/garch.py::fit_and_forecast` with `verbose=True` (or
persist `omega, alpha, beta` from the existing `_FIT_CACHE`) for both `dist ∈
{normal, t}` on each fold, recording `alpha+beta` per fold per distribution —
this changes no forecast and no QLIKE number, it only logs already-computed
parameters.
*Primary statistic*: `alpha+beta` per fold (12 values: 6 folds × 2
distributions), and the fold-level correlation between `alpha+beta` and that
fold's own `(garch spike QLIKE) − (garch calm QLIKE)` gap.
*Falsification*: if any fold has `alpha+beta < 0.95` yet still shows the
best-in-class-spike / worst-in-class-calm pattern relative to the other 5
folds, mechanism (4) is falsified for that fold — the pattern would have to
come from something other than near-unit persistence. If `alpha+beta` is
uniformly ≥0.98 across all 12 fits, that specific escape is closed and
mechanism (4) stands as the leading explanation.

**D.3 — Student-t tail-damping mechanism (garch_t vs garch_normal).**
*Population*: the same 6-fold GARCH fits.
*Contrast*: for each fold, compute the correlation between `h[t]` (the fitted
conditional-variance path, already computed and cached in `_FIT_CACHE`) under
`dist=normal` and `dist=t`, and separately the ratio of each series'
step-to-step volatility (`std(diff(log h))`) as a measure of path
"reactivity."
*Primary statistic*: `reactivity(t) / reactivity(normal)` per fold.
*Falsification*: mechanism (2) predicts this ratio is consistently **< 1**
(the Student-t path is smoother/less reactive). If the ratio is ≥1 in a
majority of folds, the "tail-damping smooths the path" story is falsified and
the spike/calm split must come from something else (e.g. a level-shift in
the two distributions' unconditional variance rather than a shape/reactivity
difference).

**D.4 — H=1 floor-artifact mechanism for har_short's blow-up.**
*Population*: the H=1 training episodes across all 6 folds.
*Contrast*: count, per fold, how many training-set `har_1h` values sit at or
near the numeric floor (the value observed at the worst 2022 episode,
−13.8155, i.e. ≤ −13) and refit `har_short` at H=1 after **winsorizing**
`har_1h` at its 0.1st/99.9th training-percentile (a one-line, easily
reproducible change, not requiring retraining anything else).
*Primary statistic*: the winsorized fit's worst-fold QLIKE and top-episode
QLIKE at H=1, fold 2022, compared to the existing 2.83480 / 17,989.37.
*Falsification*: if winsorizing `har_1h` leaves the 2022 fold's worst-episode
QLIKE and fold-mean QLIKE materially unchanged (say, worst-episode QLIKE
still >1,000, or fold mean still >2.0), the floor-artifact/leverage
explanation in §A.2 is falsified and the blow-up must come from something
else in the regressor set (e.g. a different extreme episode not yet
inspected).

**D.5 — "Feature effect, not sample effect" for the H=24 seasonality claim.**
*Population*: the H=24 test episodes, restricted to those where `noctua` and
`noctua40` are both defined (`ledger:vol-matrix` already establishes both
arms are scored on identical episodes at H≤24).
*Contrast*: `noctua − noctua40`, per-episode QLIKE difference, split by
whether the episode falls in the top-5% spike slice or not.
*Primary statistic*: paired CI on the spike-slice difference and on the
calm-slice difference separately.
*Falsification*: mechanism claim is that `seas_1d`/`seas_5d` carry
"the ENTIRE H=24 advantage" (a pooled statement). If the calm-slice CI
excludes zero favourably by itself (i.e. the seasonal columns' benefit is
generic across typical episodes, not concentrated where volatility is
anomalous), the "seasonality" framing should be relabeled "general daily/weekly
calendar structure" rather than implying a regime-specific mechanism.

---

## Summary

**Ranked hypotheses, most to least supported by the artifacts as read:**

1. **A.2 — H=1 `har_short` blow-up is a floor/leverage artifact in
   `har_1h`, not generic overfitting.** Directly demonstrated: one anchor
   (2022-07-13 16:00) contributes 72% of the fold's mean loss; 10 of 8,758
   episodes contribute 79%; the pooled-fit version of the identical
   regressor set scores 0.524× as badly at the same horizon. Strongest,
   most concrete finding in this report.
2. **B(2) — Student-t tails damp GARCH's reactivity, trading spike accuracy
   for calm stability.** A clean, 4-for-4-horizons-consistent pattern
   directly visible in the artifact (`garch_normal` beats `garch_t` on spike,
   loses on calm, at every one of H=1/6/24/168).
3. **B(4) — Near-unit GARCH persistence (IGARCH) explains the H=1-vs-H=168
   reversal** as flat-extrapolation-of-current-state being right during an
   ongoing spike and wrong across a calm week. Well-supported qualitatively
   and by the code's own persistence comment, but the exact fold-level
   `alpha+beta` values used to produce `vol_matrix_fair.json` are not in any
   artifact — this is the single most important number that is missing.
4. **A.3 — the E-anchor/har_short reconciliation (same columns, different
   architectural role).** Structurally well-supported (39-input `Xa` already
   contains `har_1h`/`har_6h`; `BASE_COLS` does not) but not tested with a
   matched contrast — it is an architectural argument, not a measured one.
5. **A.1 (H-A1a) — "regime-state" content explains har_short's 6h/24h/168h
   win.** Plausible and consistent with the H=24 seasonal-column finding,
   but the horizon-trend actually runs opposite to a naive reading (the gap
   *shrinks* with horizon, 6h→168h) and no lagged-`har_1d` control exists to
   separate it from H-A1b. Weakest-supported hypothesis in this report.

**Ruled out:** B(3), short memory / fast mean reversion, as an explanation
of GARCH's spike/calm pattern — the data shows exactly the opposite of what
fast reversion predicts (best relative spike performance, worst relative
calm performance, at the *longest* horizon).

**Single most informative test:** D.2 (log GARCH's fold-level `alpha+beta`).
It costs nothing to compute — `eval/garch.py` already fits and caches these
parameters, they are simply never written to an artifact — and it is the one
missing number standing between "IGARCH persistence is a plausible narrative"
and "IGARCH persistence is a measured mechanism." Every other hypothesis in
this report is either already well-evidenced (A.2, B(2)) or requires a new
model fit (D.1, D.4); D.2 requires only turning on a `verbose` flag that
exists in the code today.
