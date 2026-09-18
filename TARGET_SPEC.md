# TARGET_SPEC — Phase 1

What the model predicts today, computed from `model/noctua/episodes.py`,
`model/noctua/splits.py`, `model/noctua/baselines.py`, `model/noctua/spec.py`
and `model/noctua/model.py`. What the requested scope (short + medium
horizons, three-class direction with an ex-ante neutral band) would require,
computed this session from `model/artifacts/episodes.parquet` and
`model/artifacts/btcusd_1h.parquet`. No modelling approach is proposed here —
this is a specification and a data-support audit, per the Phase 2 gate in
`model/PHASES.md`.

---

## 1. CURRENT — what the model predicts today

### 1.1 RV (realized volatility)

Built in two stages, both exact given the 5-minute grid divides the hour
evenly (`model/noctua/episodes.py:20-30`):

- **Per hour** (`build_hourly`): `rv5` = sum of squared 5-minute log returns
  within that hour — the RV-literature standard sampling frequency, chosen to
  trade off microstructure noise against estimation accuracy. A parallel
  `rv1` (1-minute squared returns) is computed but never fed to the model —
  it correlates 0.953 with the forward target and is deliberately excluded as
  forward-looking leakage (`episodes.py:213-216`).
- **Per episode** (`build_episodes`): `RV = sqrt(Σ rv5 over the H-hour forward
  window)` — i.e. the realized volatility (not variance) of the window,
  summing hourly realized variances then taking one square root at the end.

Two further decompositions are computed at the hourly level and available as
features (not as the target): realized semivariance (`rv5_pos`/`rv5_neg`,
Patton–Sheppard), bipower variation (`bpv5`, jump-robust continuous variance),
and realized quarticity (`rq5`, the noise-in-RV measure HARQ uses). None of
these change the RV target itself.

### 1.2 M_up / M_dn (excursion labels)

`M_up = max(0, log(running max of hourly HIGHS over [tau, tau+H)) − log(s_tau))`
`M_dn = min(0, log(running min of hourly LOWS  over [tau, tau+H)) − log(s_tau))`

where `s_tau` = close of the hour immediately before the anchor (the last
price known at decision time).

**The hourly high/low are themselves `max`/`min` of the underlying 1-minute
bar highs/lows** (`build_hourly`, lines 90-93: `np.maximum.at` / `np.minimum.at`
over every 1-minute bar in the hour). So `M_up`/`M_dn` inherit whatever
intra-minute extremes exist in the raw tick/bar data, including flagged bad
prints unless the `_clean` variant is used — and the model does not use the
clean variant (`M_up_clean`/`M_dn_clean` have zero references outside
`episodes.py` itself; confirmed by `AUDIT.md:675`). A single wick on a single
1-minute bar can set the label for an entire multi-hour or multi-day window.
This matters directly for any medium-horizon extension: a 3-month M_up is the
max of ~2,160 hourly highs, each of which is already the max of up to 60
1-minute highs — the label is dominated by the single worst tick in roughly
130,000 ticks, not by the window's typical behavior.

`episodes.py` also enforces the running extremum include the starting point
(`max(0, ·)` / `min(0, ·)`), which keeps the pathwise identities
`M_up ≥ max(0,R)` and `M_dn ≤ min(0,R)` exact — verified with zero violations
in the sanity check (`_sanity_checks`).

### 1.3 Stage A — what it predicts, in what units

`model/noctua/model.py:15-21`: Stage A predicts the distribution of

    y = log(RV) − 0.5·log(H)     ("log hourly volatility RATE")

i.e. RV normalized by the square root of the horizon so that forecasts at
different H are comparable on the same scale. It is an explicit Log-HAR
linear base (`BASE_COLS = [har_1d, har_5d, har_22d, cal_H,
cal_weekend_frac]`, `spec.py:36-38`) plus a gated residual MLP, initialized so
training starts exactly at the Log-HAR OLS solution.

### 1.4 Stage B — what it predicts

`model/noctua/model.py:18-26`: Stage B predicts the distribution of the
**standardized** functionals, conditional on the true volatility scale sigma:

    r    = R    / sigma
    m_up = M_up / sigma   ≥ 0
    m_dn = -M_dn / sigma  ≥ 0

as monotone quantile functions (cumulative softplus increments — quantile
crossing is structurally impossible). Sigma is a causal HAR-based reference
(`causal_har_1d_clipped`, clip bounds refit per fold on that fold's training
episodes only — `BASELINE_MANIFEST.md` §6), not the realized/future RV.
Touch probabilities are then produced by quadrature-mixing Stage B's
conditional law over Stage A's own predictive quantiles for sigma
(`model.py:27-32`) — deterministic, no Monte Carlo.

### 1.5 Quantile levels actually emitted

`model/noctua/spec.py:29-33`, 17 levels, deliberately dense in the tails
because an option seller lives at α = 1–5%:

    0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995

(median index = the entry nearest 0.5, i.e. the 0.50 level itself.)

### 1.6 Horizons available today

`DEFAULT_HORIZONS = (6, 12, 19, 24)` hours (`episodes.py:51`), verified
against the live `episodes.parquet` — episode counts by H: 6→127,080,
12→127,666, 19→127,853, 24→127,897 (510,496 total across all H, all anchor
hours). The **production** configuration is a single slice of this:
`anchor_hour == 17`, `H == 19` (settlement 12:00 UTC next day / 17:30 IST),
one non-overlapping episode per calendar day.

All four current horizons are ≤ 1 day — i.e. entirely inside the "short"
band the requested scope defines (intraday → 1 week). Nothing in the current
system reaches into "medium" (1 week → 3 months) at all.

---

## 2. REQUESTED SCOPE — data support, computed this session

Data span used below: hourly bars are **exactly gap-free** (every consecutive
`hour_ts` diff is 3600s, verified over all 128,031 hourly bars) from
2012-01-01 through 2026-08-10 00:00 UTC. Restricted to the in-sample window
used everywhere else in this repo (`SAMPLE_START = 2017-08-01`,
`splits.in_sample_mask`), that is **79,105 hourly anchors**, 2017-08-01
00:00 UTC → 2026-08-10 00:00 UTC. The static `test` split (`CALIB_END =
2024-07-01` onward) contains **18,481 hourly anchors**, 2024-07-01 00:00 UTC
→ 2026-08-10 00:00 UTC (770 days).

For each candidate horizon, "non-overlapping windows" = the number of
disjoint H-hour blocks that tile the span (`⌊n_hours / H⌋`), which is the
count a headline, non-overlapping evaluation (the convention this repo
already uses for its production slice) can actually report. "Per
walk-forward-fold test year" is the count that fits inside **one** of this
repo's actual walk-forward test folds (each fold's test set is exactly one
calendar year — see `SPLIT_MANIFEST.json`), because that is the evaluation
protocol `BASELINE_MANIFEST.md` and `model/eval/benchmark.py` actually run,
not a hypothetical pooled test window.

| horizon | H (hours) | non-overlap, full history | non-overlap, static test split (770 d) | non-overlap, per walk-forward fold (1 test year) | required embargo (h) | verdict |
|---|---:|---:|---:|---:|---:|---|
| 1 day | 24 | 3,296 | 770 | 365 | ≥24 | **supported** |
| 3 days | 72 | 1,098 | 256 | 121 | ≥72 | **supported** |
| 1 week | 168 | 470 | 110 | 52 | ≥168 | **supported, reduced power** |
| 2 weeks | 336 | 235 | 55 | 26 | ≥336 | **marginal** — usable for point forecasts (QLIKE/pinball), too thin per fold for conditional-coverage or DSC/UNC-with-bootstrap-CI tests |
| 1 month (30 d) | 720 | 109 | 25 | 12 | ≥720 | **BLOCKED BY DATA for anything beyond unconditional point comparison** — 25 independent test windows (12/year per fold) cannot support the calibration/discrimination machinery this repo's own protocol requires (Christoffersen conditional coverage, DSC/UNC vs. a shuffled null with a bootstrap CI — see `BASELINE_MANIFEST.md` §8 and `BENCHMARK.md` §6g, both of which already need hundreds of test episodes to get a non-degenerate CI at the *current*, far shorter horizons) |
| 3 months (90 d) | 2,160 | **36** | **8** | **4** | ≥2,160 | **BLOCKED BY DATA** — 36 non-overlapping windows in the entire 2017–2026 history, 8 in the whole static test period, 4 per walk-forward fold year. No hypothesis test, calibration check, or discrimination-vs-null comparison this repo runs anywhere else is meaningful on n=4–8. This matches `PHASES.md`'s own pre-registered estimate ("~35 non-overlapping windows") to within one window. |

Notes on the embargo column: the rule already in force
(`splits.py:52-54`, `time_splits`/`walk_forward_folds`) is `embargo_hours =
max(H)`, i.e. embargo must be **at least** the horizon so that no
train/calib window's label period can overlap a test window's feature period
across the boundary. For every horizon above 24h, this means the embargo
requirement grows past the current fixed 24h — a 90-day horizon needs a
90-day purge gap at every split boundary. That purge shrinks the *train/calib*
side (fewer usable training windows immediately before each boundary); it
does **not** by itself reduce the *test*-side counts in the table above,
because test is the terminal split (nothing trains on data after it) and the
static test start (2024-07-01) is unaffected by how large the embargo is.
The counts above are therefore already the post-embargo, effectively-
independent figures for the test side; the embargo's cost is paid entirely
out of the training data, which is a separate, real cost not sized here
because sizing it is a Phase 2/3 concern, not a spec concern.

**Intraday horizons** (sub-1-day, e.g. 1h/2h/4h) are not tabulated above
because they are already inside the currently-supported range — the existing
H=6 and H=12 configurations already produce ~127,000 episodes each, so
anything shorter is mechanically supported by the same hourly-anchor
machinery with even more non-overlapping windows, not fewer.

**Bottom line for Phase 2 scoping:** the "short" band (intraday → 1 week) is
supported by the data, with 1–2 week horizons already at reduced statistical
power relative to what this repo currently demonstrates at H≤24h. The
"medium" band (1 week → 3 months) as requested is **not uniformly
supportable**: 2 weeks is marginal, 1 month is blocked for anything but the
crudest point comparison, and 3 months is blocked outright. A model trained
at 90-day horizon could technically be *fit*; it could not be *evaluated* by
this repo's own evidentiary standard (walk-forward folds, DSC/UNC vs.
shuffled null with bootstrap CI, Christoffersen conditional coverage) with
anything close to the statistical power every other claim in `BENCHMARK.md`
and `BASELINE_MANIFEST.md` is held to.

---

## 3. DIRECTION target — neutral-band requirement, and what is already known

### 3.1 What an ex-ante neutral band requires

A three-class target (DOWN / NEUTRAL / UP) needs a threshold `c[t, h]` such
that the label is `UP` if `R > c[t,h]`, `DOWN` if `R < -c[t,h]`, `NEUTRAL`
otherwise — and **`c[t,h]` must be computable from information available at
time t**, i.e. no realized future price or realized future volatility may
enter it, on pain of exactly the leakage this repo's causality audit already
checks for (`BASELINE_MANIFEST.md` §9: "feature causality, 42 cols — 42/42
CAUSAL"). Concretely that means at minimum:

- **A transaction-cost floor.** The only concrete cost figure this repo has
  computed is the Delta Exchange options fee schedule in
  `RESEARCH_PLAN.md:551`: `min(0.03% notional, 10% of premium)` per leg. That
  figure is for options legs, not a spot/futures directional trade, so it is
  cited here as the one number this repo has actually measured, not as a
  ready-made cost basis for a direction target — adapting it (or replacing it
  with a spot/futures fee+spread figure) is a Phase 2 task, not resolved by
  this spec.
- **A volatility-scaled buffer**, e.g. `k · sigma_hat(t,h)` where
  `sigma_hat(t,h)` is Stage A's own causal forecast at time t (never the
  realized/future RV) — the natural choice given Stage B already standardizes
  everything by a causal sigma.
- Both terms must be refit/recomputed **per horizon h**, since a fixed cost
  or fixed vol buffer that is reasonable at H=6h is not reasonable at
  H=2160h.

No such threshold is defined anywhere in the current codebase; `eval/direction.py`
and `eval/kronos_direction.py` both score a binary sign, not a three-class,
neutral-banded target. Defining `c[t,h]` is scope for Phase 2, not something
this spec resolves.

### 3.2 Directional predictability has already been measured, extensively, and found near-zero

This is the load-bearing fact for scoping Phase 2: **direction is not an
open question this repo hasn't looked at — it has been measured repeatedly
and closed.**

- `model/BENCHMARK.md` §6g (amplification vs. barrier vs. direction, same
  episodes, same model): **"A Brier skill score of 20.3% against 4.98% on
  barrier discrimination and 0.18% on direction — the amplification question
  is 113× more answerable than the direction question and 4× more than the
  barrier question, on the same episodes, from the same model."**
- `model/eval/direction.py` exists specifically because an earlier two-row
  check found `NOCTUA log_loss 0.694344` against `constant_50 log_loss
  0.693147` — **NOCTUA lost to a coin flip** on sign, and the file's own
  stated purpose is to distinguish "model failure" from "efficient-market
  null" for direction specifically.
- `model/BENCHMARK.md` §5b: Kronos's own displayed "Upside Probability" is
  shown to be a 32-sample Monte Carlo proportion with an 8.8pp standard
  error at p=0.5, and when scored against realized outcomes it **loses 75%
  worse than a constant forecast** (log loss 1.20547 vs. 0.68967) and its
  discrimination does not clear a shuffled-input null. NOCTUA's own
  unpublished `prob_up` — deliberately not shipped, precisely because it was
  known unreliable — loses a much smaller 0.071 nats to a constant, i.e. it
  is closer to uninformative than actively harmful, but still not
  demonstrated skill.

Taken together: this repository's own measurements put directional skill at
roughly two orders of magnitude below its best-measured skill
(amplification, 20.3%) and well below its second-best (barrier
discrimination, 4.98%), at the horizons actually tested (6–24h). **This
TARGET_SPEC does not treat direction as an open opportunity.** Any Phase 2+
work on a three-class direction target should be scoped as: (a) testing
whether the *neutral band* construction itself (rather than a raw
binary sign) changes the finding, since a banded target removes the
hardest-to-call near-zero-return episodes that likely dominate the losses
above — this is a genuinely open sub-question the existing measurements do
not answer — not as a bet that raw directional skill was previously
unmeasured.

### 3.3 Direction and the horizon-support finding compound

Even setting the predictability question aside, a three-class direction
target at medium horizons inherits §2's data-support problem directly: a
3-month direction call has the same 36/8/4 (full-history/test/per-fold)
non-overlapping-window ceiling as the 3-month volatility target, because
both are read off the same non-overlapping tiling of the same hourly grid.
There is no version of "define a neutral band and test it at 3 months" that
escapes that ceiling.

---

## Paths written this session

- `TARGET_SPEC.md` (this file, repo root)
- `SPLIT_MANIFEST.json` (repo root)

*Educational research only. Not financial advice.*
