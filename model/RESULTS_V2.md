# NOCTUA v2 — capacity, committees, and a hierarchy that did not work

Three questions were asked and answered empirically. Two of the answers
contradict the premise they started from, which is why they are worth reading.

Protocol throughout is identical to v1 (`RESULTS.md`): expanding-window
walk-forward, embargoed splits, every number on the production slice (19 h
window opened 17:00 UTC, one non-overlapping episode per day, 2021-2026).

---

## 1. "Increase the parameters — we targeted 1 M and only used 50 k"

**Measured: more capacity makes it worse.** Monotone, on both metrics.

| hidden | params | QLIKE vs Log-HAR | p | barrier err |
|---|---|---|---|---|
| **32** | **6,378** | **−4.65 %** | **0.0000** | **2.706 pp** |
| 64 | 16,778 | −3.96 % | 0.0006 | 2.778 pp |
| 128 | 49,866 | −2.79 % | 0.0426 | 2.926 pp |
| 256 | 165,194 | −2.79 % | 0.0174 | 3.133 pp |
| 512 | 592,458 | −3.38 % | 0.0032 | 2.938 pp |

Shrinking the network **8×** nearly doubled the volatility edge (−2.79 % →
−4.65 %) and moved it from marginal to decisive (p = 0.043 → p < 0.0001).

The cause is sample size, and it is arithmetic rather than opinion. A 19-hour
window anchored hourly overlaps its neighbour by 18/19, so the 189,831 training
episodes carry only **2,498 independent observations** — a 76× inflation. At
width 128 the model already holds 20 parameters per effective observation; at
1 M it would hold ~400. Capacity buys variance here, not signal.

This also matches the published result the v1 plan was built on
(arXiv:2607.05291): across 50 assets, the *smallest* foundation model tested
(<1 M params) was the only one of nine to beat Log-HAR.

## 2. "Build many specialists and let them interact"

**Measured: yes — this is the real gain.** Four heterogeneous specialists,
pooled by Vincentization:

| specialist | what it knows | what it cannot do |
|---|---|---|
| neural | conditional, state-dependent shape | overfits; noisy in the body |
| gaussian | exact first-passage law, zero estimation variance | Gaussian tails are wrong for crypto |
| empirical | true fat tails, no parametric assumption | noisy at 1-in-200 |
| EVT (peaks-over-threshold) | principled tail extrapolation via GPD | says nothing about the body |

Pooling is **Vincentization** — averaging quantile *functions*, not CDFs.
Ranjan & Gneiting (2013) proved a linear pool of calibrated forecasts is
necessarily **overdispersed**, which would destroy the one thing this model is
good at. Quantile averaging also keeps monotonicity, so `safe_level` stays
exactly invertible.

| α | NOCTUA v1 | Gaussian | **Committee** |
|---|---|---|---|
| 1 % | 1.112 | 3.326 | **1.090** |
| 2 % | 1.423 | 3.527 | **1.336** |
| 5 % | 2.610 | 3.765 | **1.959** |
| 10 % | 3.257 | 2.710 | **2.131** |
| 20 % | 3.663 | **2.006** | 3.040 |
| 30 % | 3.777 | **3.693** | 5.175 |
| **mean** | 2.640 | 3.171 | **2.455** |
| **mean, α ≤ 10 %** | 2.100 | 3.332 | **1.629** |

In the range an option seller actually operates in, calibration error drops
**22 %**. Above α = 20 % the committee is worse and the Gaussian should be
used instead — reported, not hidden.

### The bug that decided it

The committee first came out at 3.199 pp — clearly *losing*. The cause was a
genuine modelling error, not tuning: NOCTUA's `safe_level` integrates over its
uncertainty about σ (32 quadrature atoms), while the three analytic specialists
conditioned on a single point estimate. **A predictive distribution built on a
point estimate of a parameter is under-dispersed relative to one that
integrates over it.** Pooling one hierarchical forecast with three
under-dispersed ones pulled every level too close to spot (α = 5 % was touched
7.1 % / 10.2 % of the time).

Mixing all specialists over the same σ atoms fixed it: 3.199 → **2.455**. The
correction widens the deep tail (1.16× at α = 0.5 %) and barely moves the body,
exactly as integrating over parameter uncertainty should.

## 3. "Make NOCTUA a parent that spawns specialist children"

**Measured: implemented rigorously, it does not help.**

The disciplined form of this is a hierarchical mixture of experts (Jacobs,
Jordan, Nowlan & Hinton 1991): a gating network reads market state — vol level,
vol-of-vol, position within the year, weekend fraction, jump share — and emits
per-episode weights over the children. It was initialised at exactly equal
weights (W = 0 ⇒ uniform softmax) so it starts at the flat committee and can
only be credited with what it adds.

It added nothing:

```
child        mean weight    sd ACROSS EPISODES
neural            0.919              0.0000
gaussian          0.038              0.0000
empirical         0.021              0.0000
evt               0.021              0.0000
```

The gate put 92 % on one child and **its weights do not vary with market state
at all**. Gated calibration error 2.667 pp — statistically the same as NOCTUA
alone (2.640) and worse than the flat committee (2.455).

Why it fails is interpretable, and it is an objective mismatch. The gate is
fitted by pinball loss, which rewards the single *sharpest* forecaster; the
metric that matters is *calibration*, which rewards the diversity the ensemble
provides. Given freedom to concentrate, the gate concentrated — the same
overfitting the capacity study predicted, arriving through a different door.

This is the "forecast combination puzzle" in its usual form: **equal weights are
hard to beat.** Both the level-dependent weights (§2) and the state-dependent
gate (§3) converged to roughly uniform or degenerate solutions. The gain comes
from *having* heterogeneous, properly-dispersed members — not from cleverness
about how to weight them.

---

## What v2 ships

- width **32** instead of 128 (6,378 params, 8× smaller)
- **3-seed** average — free variance reduction
- flat **equal-weight** committee of the four specialists, σ-mixed
- gating network **built, measured, and rejected**; code retained so the
  negative result stays reproducible

| | v1 shipped | **v2** |
|---|---|---|
| volatility QLIKE vs Log-HAR | −2.79 % (p = 0.043) | **−4.27 % (p = 0.0002)** |
| barrier err, α ≤ 10 % | 2.100 pp | **1.629 pp** |
| network params | 49,866 | **6,378 × 3 seeds** |

*Educational research only. Not financial advice.*
