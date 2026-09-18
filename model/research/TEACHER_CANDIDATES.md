# TEACHER_CANDIDATES — open-source model discovery + license/infrastructure audit

**Scope note:** this is a factual dossier for the "teacher" search described in
`TEACHER_ZOO.md`. It does not choose an architecture, does not train
anything, and does not modify anything under `model/` or `data/` other than
adding this file. Every field says `UNKNOWN` rather than guessing where the
available tools (Hugging Face MCP, `pip`, web search) could not establish a
fact with confidence.

**Audit date:** 2026-08-28. **Machine:** 4 CPU cores, ~15 GB RAM, no GPU.
**Egress:** crypto exchange APIs blocked (403 via proxy, not re-tested here —
this dossier does not touch exchange data at all); PyPI confirmed reachable
(`pip download --no-deps` succeeded for every candidate with a PyPI package,
see §Infra checks below); `pixi.prefix.dev` (TiRex-2's installer host) was
**not** verified reachable and is not on the documented CDN/proxy allowlist.

**Installed already:** numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, pyarrow 25,
torch 2.13.0+cu130, scikit-learn 1.9.0, arch. **Not installed:** xgboost,
lightgbm. These versions matter below — several candidates' PyPI metadata
pins *older* numpy/torch/pandas/sklearn and would try to downgrade this
environment on a real `pip install`.

**BTC evaluation frame this dossier was written against** (from
`TEACHER_ZOO.md`, for calibrating the adversarial paragraphs below): target
is `har_target(RV, H) = log(RV) − ½·log(H)`, i.e. a forward-window **log
volatility rate**, not a price level or a raw RV path; test years are
2021–2026 (walk-forward, embargoed); scoring is QLIKE on `σ = exp(ŷ)·√H`;
episodes are hourly-anchored and overlapping with horizons `H ∈ {1, 6, 24,
168}`. The task brief's numbers (~49,000 overlapping test episodes, ~512-step
context, ~6 independent yearly regimes) are used throughout.

---

## Infra checks performed

`pip download --no-deps -d . <pkg>` succeeded for all of: `chronos-forecasting`
(2.3.1), `timesfm` (2.0.2), `uni2ts` (2.0.0), `tirex-ts` (1.4.2), `toto-ts`
(0.2.0), `neuralforecast` (3.2.1). No PyPI package was found for TimeMixer,
TimeMixer++, TimeXer, TSMixer, TimesNet, DLinear, PatchTST, or iTransformer
individually — they ship as reference implementations inside
`thuml/Time-Series-Library` (MIT) and/or as one model class among many inside
`neuralforecast` (Apache-2.0), not as standalone packages. No PyPI package
was found for TiRex-2.

Dependency-pin conflicts found by inspecting each wheel's `METADATA` (not by
actually installing):

| package | pins that conflict with this environment |
|---|---|
| `uni2ts` (Moirai) | `numpy~=1.26.0` (have 2.4.6), `torch<2.5,>=2.1` (have 2.13.0), `gluonts~=0.14.3`, `jax[cpu]` |
| `toto-ts` (Toto) | `torch==2.7.0`, `numpy==1.26.4`, `pandas==2.2.3`, `scikit-learn==1.5.0`, `transformers==4.52.1`, `gluonts[torch]==0.16.2`, `datasets==2.17.1` — five+ **exact** pins |
| `chronos-forecasting` | `torch<3,>=2.2` — compatible |
| `timesfm` | `torch>=2.0.0` (extra) — compatible |
| `tirex-ts` | `torch`, `numpy`, `scikit-learn` unpinned — compatible; CUDA/xLSTM kernel only pulled in via optional `[cuda]` extra |
| `neuralforecast` | `torch>=2.9.1` — compatible |

A real (non `--no-deps`) `pip install uni2ts` or `pip install toto-ts` in this
environment would attempt to downgrade the already-installed torch and numpy
(and, for Toto, pandas and scikit-learn too), which is exactly the kind of
disruptive reinstall this task said not to trigger. This is flagged per
candidate below and reflected in the shortlist as **BLOCKED BY
INFRASTRUCTURE** where it is the deciding factor.

---

## Candidate records

### 1. PatchTST

1. **Name/paper:** PatchTST — "A Time Series is Worth 64 Words: Long-Term
   Forecasting with Transformers", Nie, Nguyen, Sinthong, Kalagnanam (IBM
   Research), **ICLR 2023**.
2. **Repo/PyPI:** official `github.com/yuqinie98/PatchTST`; also reimplemented
   in `thuml/Time-Series-Library` and `Nixtla/neuralforecast`. No standalone
   PyPI package; reached via `neuralforecast` (3.2.1, PyPI-confirmed) or by
   vendoring the reference repo.
3. **License code:** MIT (official repo and Time-Series-Library) /
   Apache-2.0 (neuralforecast). **License weights:** N/A — no general
   pretrained checkpoint is proposed here (see #6).
4. **Latest version/checkpoint:** N/A, architecture only; `neuralforecast`
   3.2.1 is the latest PyPI release as of this audit. No commit hash captured
   (not cloned in this session).
5. **Param count:** configurable, not fixed; typical long-horizon configs in
   the paper run roughly 1–10M parameters (patch length 16, `d_model` 128,
   2–3 encoder layers). For a single BTC-vol channel this would sit at the
   low end.
6. **Peak RAM, CPU inference, 8,760 seq × ctx 512:** FEASIBLE, low —
   sub-100MB weights, activation memory for a few-thousand-sequence batch on
   4 cores comfortably under 1–2GB.
7. **Max/native context length:** none architectural — patching makes long
   context cheap; the paper demonstrates up to 336–720 steps. 512 is trivial.
8. **Target formulation:** point forecast (MSE) by default; no native
   distributional head.
9. **Covariate support:** the original architecture is channel-independent
   (no dedicated exogenous channel); the `neuralforecast` wrapper bolts on
   `hist_exog`/`futr_exog`/`stat_exog` generically across all its models,
   which is a wrapper feature, not something PatchTST's paper designed for.
10. **Probabilistic output:** none native; quantile loss can be substituted
    since this would be trained from scratch anyway.
11. **Training-data provenance / leakage:** **LOW** — no public pretrained
    weights are being proposed; if used, the model is trained from scratch on
    our own BTC episodes, so no foundation-corpus leakage question applies.
    (IBM's HF `PatchTSTForPrediction` checkpoints, e.g. ETTh1-pretrain, exist
    but are not proposed for use here — they're pretrained only on ETT, not
    a general corpus, so irrelevant to leakage.)
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE** — small, well-supported,
    no dependency conflicts.
14. **PyPI install attempt:** `neuralforecast` 3.2.1 downloaded successfully
    via `pip download --no-deps`; `torch>=2.9.1` requirement is satisfied by
    the installed 2.13.0.

**WHY THIS SHOULD FAIL ON BTC.** PatchTST's whole contribution is
channel-independent patch tokenization tuned for smooth, seasonally
regular multivariate series like ETT/Weather/Traffic/Electricity, where the
signal has stable within-day and within-week periodicity that a fixed patch
length can exploit. BTC realized-vol is long-memory and regime-switching
rather than seasonal — its autocorrelation structure looks nothing like a
transformer/electricity load curve, so the inductive bias a fixed patch
length encodes (local semantic units repeating on a roughly constant period)
is not obviously present. The model was never designed to predict a
*log-vol-rate normalized by horizon*; it predicts levels of the input series
directly, so target engineering (the `−½·log(H)` term, the horizon-specific
targets) has to be bolted on outside the architecture and could silently
break the loss geometry the paper tuned patch sizes for. With ~49,000
overlapping hourly-anchored test episodes but only ~6 independent yearly
regimes, a small transformer with enough capacity to look sophisticated can
overfit whichever 1–2 regimes dominate the training window (e.g. a violent
2021 or 2022 drawdown) and then fail exactly when the next regime doesn't
resemble it — nothing in PatchTST's ETT/Traffic-style benchmarking would
have surfaced that failure mode, because those benchmarks don't have anything
like crypto's regime discontinuities.

---

### 2. iTransformer

1. **Name/paper:** iTransformer — "iTransformer: Inverted Transformers Are
   Effective for Time Series Forecasting", Liu et al. (Tsinghua/thuml),
   **ICLR 2024**.
2. **Repo/PyPI:** `github.com/thuml/iTransformer`; also in
   `thuml/Time-Series-Library` and `Nixtla/neuralforecast`. No standalone
   PyPI package; via `neuralforecast` 3.2.1.
3. **License code:** MIT. **License weights:** N/A, no general pretrained
   checkpoint.
4. **Latest version/checkpoint:** N/A (architecture); `neuralforecast` 3.2.1
   current. No commit hash captured.
5. **Param count:** scales with number of variates and `d_model`; for a
   single-variate (BTC-vol-only) setup this is small (a few M); grows if many
   covariate channels are added as extra "variate tokens."
6. **Peak RAM, CPU, 8,760×512:** FEASIBLE, low — same order as PatchTST.
7. **Max/native context length:** none architectural; the model attends
   across variates, not across time, so the time dimension is a fixed-size
   input embedding per variate — 512 is a non-issue.
8. **Target formulation:** point forecast (MSE); no native probabilistic
   head.
9. **Covariate support:** structurally interesting — because iTransformer
   embeds each *variate's entire history* as one token and attends across
   variates, adding covariates means adding more variate tokens. This gives
   it a natural (if unusual) way to use exogenous series, but it does not
   distinguish "known future" from "past-only" covariates; a future-known
   covariate would need manual masking/padding tricks not part of the
   original design.
10. **Probabilistic output:** none native.
11. **Training-data provenance / leakage:** **LOW** — trained from scratch
    if used; no proposed pretrained checkpoint.
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE**.
14. **PyPI install attempt:** via `neuralforecast` 3.2.1, same result as
    PatchTST above.

**WHY THIS SHOULD FAIL ON BTC.** iTransformer's central bet is that
cross-*variate* correlation (e.g. correlated load curves across many
electricity feeders, correlated exchange rates) is the dominant signal worth
attending over, and that per-variate temporal dynamics can be handled by a
simple feed-forward embedding of the whole lookback window. A single BTC
realized-vol series has exactly one target variate; iTransformer's core
mechanism (attention *across* variates) degenerates to nothing unless we feed
it many correlated covariate series, and even then the "variate token = one
embedded lookback window" design assumes each variate's dynamics are well
summarized by a linear projection of a fixed window — a poor fit for a
heavy-tailed, volatility-clustered series where the *shape* of recent
extremes (not just their linear projection) matters. iTransformer was
benchmarked on ETT/Traffic/Weather/Electricity/Exchange, i.e. series with
mild tails and slow-moving multivariate correlation; it has no demonstrated
mechanism for surviving a 2021-style speculative blow-off or a 2022-style
cascading deleveraging event, and because our overlapping episodes are all
drawn from only ~6 independent yearly regimes, a model that happens to
memorize "what November 2021 looked like" via its variate embeddings has no
way to generalize to a regime it has not seen — and the paper's benchmarks
never test that kind of out-of-regime generalization.

---

### 3. TimeMixer

1. **Name/paper:** TimeMixer — "TimeMixer: Decomposable Multiscale Mixing
   for Time Series Forecasting", Wang et al. (Ant Group/Tsinghua), **ICLR
   2024**.
2. **Repo/PyPI:** `github.com/kwuking/TimeMixer`; also in
   `thuml/Time-Series-Library` and `neuralforecast`. No standalone PyPI
   package.
3. **License code:** MIT. **License weights:** N/A, no pretrained checkpoint.
4. **Latest version/checkpoint:** N/A (architecture).
5. **Param count:** small, MLP-based multiscale decomposition; typically
   under ~5M for standard long-horizon configs.
6. **Peak RAM, CPU, 8,760×512:** FEASIBLE, low.
7. **Max/native context length:** configurable; paper tests 96–720; compute
   scales with number of downsampled scales × sequence length, still cheap
   at 512.
8. **Target formulation:** point forecast (MSE) primarily.
9. **Covariate support:** no dedicated exogenous channel in the original
   design (it's a decomposition + multiscale mixing model over the target
   series itself); `Time-Series-Library`'s wrapper can concatenate exogenous
   features at the input, a bolt-on rather than an architectural feature.
10. **Probabilistic output:** none native.
11. **Training-data provenance / leakage:** **LOW** — trained from scratch.
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE**.
14. **PyPI install attempt:** via `neuralforecast` 3.2.1 (its models module
    includes TimeMixer); same install result as above.

**WHY THIS SHOULD FAIL ON BTC.** TimeMixer's multiscale decomposition
assumes there are multiple *stable* periodicities (daily/weekly/etc.) whose
relative contribution to the signal can be learned once and mixed
consistently — this is a reasonable assumption for electricity/traffic/
weather but is shaky for BTC realized vol, where "24-hour" and "weekly"
structure exists but is weak, noisy, and itself regime-dependent (calm
periods show faint diurnal patterns tied to exchange liquidity hours; violent
periods are dominated by event-driven jumps that swamp any periodic
component). If the multiscale mixing weights are fit mostly during a
low-volatility training regime, the model will have learned to trust a
periodicity structure that a subsequent volatility regime largely erases —
and because our target is the *forward log-vol rate*, not the raw series
TimeMixer decomposes, any spurious periodic structure the model finds in
`log(RV)` history is liable to be an artifact of a particular funding/futures
expiry cycle in one regime rather than a real invariant. TimeMixer was never
evaluated on a target defined as a horizon-normalized forward *rate* over
long-memory, heavy-tailed data, and its ICLR-benchmark suite (ETT, Weather,
Traffic, Electricity, Solar, PEMS) contains nothing resembling a
regime-switching financial series.

---

### 4. TimeMixer++

1. **Name/paper:** TimeMixer++ — "TimeMixer++: A General Time Series Pattern
   Machine for Universal Predictive Analysis", Wang, Li, Shi, Ye, Mo, Lin,
   Ju, Chu, Jin, **ICLR 2025 (Oral)**, arXiv:2410.16032.
2. **Repo/PyPI:** **UNKNOWN** — could not confirm a distinct, maintained
   public repository for TimeMixer++ specifically (as opposed to the
   original TimeMixer repo) via the searches run in this session. No PyPI
   package found. Treat "no verified runnable implementation located" as a
   finding in itself, not an assumption of non-existence.
3. **License code:** UNKNOWN (no confirmed repo to read a LICENSE from).
   **License weights:** N/A — no pretrained zero-shot checkpoint was found;
   the paper evaluates the architecture per-task/per-dataset, not as a
   released foundation model.
4. **Latest version/checkpoint:** N/A.
5. **Param count:** UNKNOWN precisely. The method extends TimeMixer with
   "multi-resolution time imaging" (converting 1D series into 2D images via
   detected multi-periodicity) plus vision-style patch mixing across those
   images — this adds parameters and compute relative to TimeMixer, but by
   how much was not established here.
6. **Peak RAM, CPU, 8,760×512:** UNKNOWN precisely; the 1D→2D imaging step
   plus multi-resolution mixing is more compute-intensive than TimeMixer's
   pure-MLP mixing, so treat as **MARGINAL** rather than confidently
   feasible until a concrete implementation is located and profiled.
7. **Max/native context length:** UNKNOWN, paper/config-dependent.
8. **Target formulation:** point forecast is TimeMixer++'s default mode
   (it's evaluated across 8 tasks — forecasting, imputation, anomaly
   detection, classification — as a general "pattern machine," not as a
   dedicated probabilistic forecaster).
9. **Covariate support:** UNKNOWN / not a stated design goal of the paper.
10. **Probabilistic output:** none confirmed.
11. **Training-data provenance / leakage:** **LOW/UNKNOWN** — no confirmed
    public pretrained checkpoint exists to leak from; if used at all it would
    be trained from scratch.
12. **Honest eval mode:** trained-from-scratch, contingent on locating and
    verifying a runnable implementation first.
13. **CPU-only feasibility verdict:** **MARGINAL** — not primarily a compute
    judgment but a verifiability one: unlike every other architecture-style
    candidate in this dossier, this session could not confirm a maintained,
    runnable open-source implementation exists at all.
14. **PyPI install attempt:** not attempted — no package name identified.

**WHY THIS SHOULD FAIL ON BTC.** Even taking the paper's mechanism at face
value, TimeMixer++'s core trick — turning a 1D series into 2D "time images"
keyed on detected dominant periods, then mixing those images — inherits and
amplifies TimeMixer's periodicity assumption (see #3) rather than fixing it:
if the FFT/period-detection step cannot find a stable, dominant period in
BTC's noisy, regime-dependent autocorrelation structure (which realistic
crypto vol mostly can't offer outside of narrow calm windows), the "time
imaging" step manufactures a spurious 2D structure out of what is
substantially structureless or non-stationary noise, and the vision-style
mixing that follows will confidently overfit that artifact. Separately, and
more mundanely: a "general pattern machine" validated on eight *unrelated*
tasks (forecasting + imputation + anomaly detection + classification, mostly
on ETT/Weather/Traffic-style benchmarks per its lineage) is optimized for
breadth of applicability, not depth on any one hard forecasting problem —
there is no reason to expect its architecture search implicitly discovered
anything suited to a single, adversarially hard long-memory/heavy-tailed
target like forward BTC vol, and the complete absence of a verified public
implementation means any claim about its behavior here is currently
untestable rather than merely uncertain.

---

### 5. TimeXer

1. **Name/paper:** TimeXer — "TimeXer: Empowering Transformers for Time
   Series Forecasting with Exogenous Variables", Wang et al. (thuml),
   **NeurIPS 2024**.
2. **Repo/PyPI:** `github.com/thuml/TimeXer`; also in
   `thuml/Time-Series-Library` and `Nixtla/neuralforecast`. No standalone
   PyPI package.
3. **License code:** MIT. **License weights:** N/A, no pretrained checkpoint.
4. **Latest version/checkpoint:** N/A (architecture).
5. **Param count:** modest transformer with separate endogenous/exogenous
   tokenization; typically a few M parameters for standard configs.
6. **Peak RAM, CPU, 8,760×512:** FEASIBLE, low.
7. **Max/native context length:** configurable; paper tests up to 720 steps.
8. **Target formulation:** point forecast (MSE) by default.
9. **Covariate support:** **YES — this is TimeXer's entire reason to exist.**
   Native, purpose-built support for exogenous variables: endogenous series
   is patch-tokenized, exogenous series get their own variate-wise tokens,
   and cross-attention lets the endogenous target attend to exogenous
   context. Designed explicitly for scenarios where covariates are partially
   or fully known — the closest architectural fit, among the from-scratch
   candidates, to "feed it order-flow/on-chain/funding-rate covariates
   alongside BTC vol history."
10. **Probabilistic output:** none native (point forecast); a quantile loss
    could be substituted since this is trained from scratch.
11. **Training-data provenance / leakage:** **LOW** — trained from scratch,
    no foundation corpus involved.
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE**.
14. **PyPI install attempt:** via `neuralforecast` 3.2.1 (confirmed to
    include TimeXer per its model overview).

**WHY THIS SHOULD FAIL ON BTC.** TimeXer's exogenous-attention mechanism was
validated on covariates that are *causally upstream and reasonably stable* —
weather driving electricity load, calendar effects driving traffic. Whatever
covariates we'd feed it for BTC vol (order-flow imbalance, funding rates,
on-chain metrics, macro series) have a much weaker and more regime-dependent
causal relationship to forward realized volatility than temperature has to
power demand — in some regimes funding rate leads vol, in others it lags or
decouples entirely, and TimeXer's cross-attention has no mechanism to detect
*when* a given covariate has stopped being informative, so a shift in which
covariate matters (a classic feature of regime change) would look to the
model like ordinary noise rather than a signal it should discount. TimeXer
is also, like the rest of this transformer family, trained on a raw-level
forecasting objective in its home benchmarks, not the horizon-normalized
forward log-vol-rate target this project uses — nothing in NeurIPS
benchmarking exercised its loss geometry under that transform. And because
covariates increase the effective parameter count and the number of ways to
overfit, with ~49,000 overlapping episodes drawn from only ~6 independent
years, TimeXer is if anything *more* exposed to spurious covariate-vol
correlations that hold in-sample and vanish out-of-regime than a
covariate-free model would be.

---

### 6. TSMixer

1. **Name/paper:** two distinct papers share this name — flag the ambiguity
   explicitly. (a) Chen et al. (Google), "TSMixer: An All-MLP Architecture
   for Time Series Forecasting," **TMLR 2023**, arXiv:2303.06053. (b)
   Ekambaram et al. (IBM), "TSMixer: Lightweight MLP-Mixer Model for
   Multivariate Time Series Forecasting," **KDD 2023**. `neuralforecast`
   documents both a `TSMixer` and a `TSMixerx` ("x" = exogenous-extended)
   model, closer in spirit to the IBM lineage's exogenous handling.
2. **Repo/PyPI:** Google version — `github.com/google-research/google-research`
   (`tsmixer` subdirectory). IBM version — `github.com/ibm-granite/granite-tsfm`
   (also exposed via HF `transformers` as a `PatchTSMixer`-adjacent class,
   with some pretrained "granite-timeseries" checkpoints). No standalone
   `tsmixer` PyPI package found; reachable via `neuralforecast` 3.2.1.
3. **License code:** Apache-2.0 for both lineages. **License weights:** N/A
   for the from-scratch architecture; IBM's granite-timeseries pretrained
   checkpoints (if ever used) carry their own Apache-2.0 HF license but their
   pretraining corpus provenance was **not** audited in this session — treat
   as UNKNOWN, not cleared.
4. **Latest version/checkpoint:** N/A for the architecture; `neuralforecast`
   3.2.1 current.
5. **Param count:** small, all-MLP, typically well under 5M for standard
   configs.
6. **Peak RAM, CPU, 8,760×512:** FEASIBLE, low.
7. **Max/native context length:** configurable; papers test up to 720.
8. **Target formulation:** point forecast (MSE) primarily.
9. **Covariate support:** the `TSMixerx` / IBM lineage explicitly supports
   static and dynamic (including future-known) exogenous covariates — the
   "x" is literally for that.
10. **Probabilistic output:** none native.
11. **Training-data provenance / leakage:** **LOW** if trained from scratch
    (the recommended path here); **UNKNOWN** if a pretrained IBM
    granite-timeseries checkpoint were substituted instead — its training
    corpus was not enumerated in this session.
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE**.
14. **PyPI install attempt:** via `neuralforecast` 3.2.1, same result as
    above.

**WHY THIS SHOULD FAIL ON BTC.** TSMixer's headline claim (both lineages) is
that simple, cheap MLP mixing across time and feature dimensions matches or
beats attention on standard long-horizon benchmarks — which is really a
claim about those benchmarks having enough exploitable *linear and
low-order* structure (trend, seasonality, mild cross-channel correlation)
that attention's extra expressiveness buys nothing. BTC realized vol is
close to the adversarial case for that claim: its defining features (fat
tails, volatility clustering with long memory, abrupt regime breaks) are
exactly the kind of higher-order, non-stationary structure that mixing
linear projections along time and feature axes is not designed to capture —
an MLP mixer has no mechanism analogous to attention's content-based routing
that could let it "notice" a live regime break and reweight accordingly; it
just applies the same learned linear mix everywhere. Because it is this
cheap, it will *also* fit fastest and look deceptively good on the walk-
forward fold that resembles its training regime most closely, then miss
badly on a held-out regime that's structurally different — precisely the
scenario the ~6-independent-years ceiling on this dataset creates, and
precisely the failure mode Google/IBM's KDD/TMLR benchmarks (largely
non-financial, non-crypto, and not evaluated regime-by-regime) would never
have exposed.

---

### 7. TimesNet

1. **Name/paper:** TimesNet — "TimesNet: Temporal 2D-Variation Modeling for
   General Time Series Analysis", Wu et al. (Tsinghua/thuml), **ICLR 2023**.
2. **Repo/PyPI:** `github.com/thuml/Time-Series-Library` (flagship model of
   this repo) / `github.com/thuml/TimesNet`; also in `neuralforecast`. No
   standalone PyPI package.
3. **License code:** MIT. **License weights:** N/A, no pretrained checkpoint.
4. **Latest version/checkpoint:** N/A (architecture).
5. **Param count:** moderate — FFT-based period detection feeding
   Inception-style 2D convolution blocks; typically 5–15M for standard
   configs, the heaviest of the from-scratch architectures here (heavier
   than DLinear/PatchTST/TSMixer, comparable to or above TimeMixer).
6. **Peak RAM, CPU, 8,760×512 (inference):** FEASIBLE — inference alone is
   fine on 4 cores/15GB RAM. Note this verdict is inference-only; *training*
   cost (not asked for numerically here, but worth flagging) is noticeably
   higher than the MLP/linear family due to the 2D conv stack.
7. **Max/native context length:** configurable; benefits from longer context
   since FFT-based period detection needs several cycles to reliably
   estimate dominant periods — at 512 hourly steps (~21 days) there may not
   be enough repetitions of any weekly-scale structure to detect reliably.
8. **Target formulation:** point forecast (MSE) by default; TimesNet is
   explicitly a general "task machine" (forecasting + imputation +
   classification + anomaly detection), not a dedicated probabilistic
   forecaster.
9. **Covariate support:** no dedicated exogenous channel in the original
   design; `Time-Series-Library`'s generic wrapper can concatenate exogenous
   features as extra input channels.
10. **Probabilistic output:** none native.
11. **Training-data provenance / leakage:** **LOW** — trained from scratch.
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE** for inference; note the
    training-time cost caveat above.
14. **PyPI install attempt:** via `neuralforecast` 3.2.1.

**WHY THIS SHOULD FAIL ON BTC.** TimesNet's entire mechanism is built on the
premise that a time series has a small number of dominant, FFT-detectable
periods that, once found, can be reshaped into clean 2D grids where
convolution finds meaningful local structure. BTC realized vol simply does
not have this property in the way ICLR's ETT/Weather/Traffic/Electricity
benchmark suite does — its periodogram is dominated by broadband,
non-stationary power rather than a handful of sharp, stable spectral peaks,
so the FFT step is liable to lock onto a period that is either an artifact
of a particular training window (e.g. a weekly futures-expiry cadence that
existed in 2021 but not 2024) or simply noise, and then confidently apply 2D
convolution to a synthetic image built on that spurious structure — silent
failure, not a graceful degradation, because nothing in the architecture
checks whether the detected periods are actually reliable. With only ~512
hourly steps of context, there is barely enough data to estimate a
weekly-scale period at all, let alone verify its stability, and because
episodes overlap heavily and are drawn from ~6 independent regimes, whatever
period TimesNet locks onto during training is exactly the kind of spurious
in-sample pattern that a purely time-ordered walk-forward split (rather than
TimesNet's original random/blocked academic splits) is designed to expose.

---

### 8. DLinear

1. **Name/paper:** DLinear — "Are Transformers Effective for Time Series
   Forecasting?", Zeng, Chen, Zhang, Xu, **AAAI 2023**.
2. **Repo/PyPI:** `github.com/cure-lab/LTSF-Linear`; also in
   `thuml/Time-Series-Library` and `Nixtla/neuralforecast`. No standalone
   PyPI package.
3. **License code:** MIT. **License weights:** N/A — trivially cheap to
   train from scratch, no pretrained checkpoint is meaningful for this
   architecture.
4. **Latest version/checkpoint:** N/A (architecture, always trained
   per-task).
5. **Param count:** tiny — two linear layers over trend/seasonal-decomposed
   input, parameter count is `O(context_len × horizon)`; for context 512 and
   a short horizon this is on the order of tens of thousands to roughly 1M
   parameters at most. By far the smallest candidate in this dossier.
6. **Peak RAM, CPU, 8,760×512:** FEASIBLE, negligible — this is the cheapest
   candidate to run by a wide margin.
7. **Max/native context length:** no architectural limit at all; it is
   literally a linear map from the lookback window to the horizon.
8. **Target formulation:** point forecast — a single linear projection, no
   native distributional output whatsoever.
9. **Covariate support:** none natively — pure autoregressive linear model
   on the target's own history, channel-independent by design. Wrapper-level
   exogenous concatenation is possible but defeats the "why does simple
   linear regression already beat transformers" point of the paper.
10. **Probabilistic output:** none.
11. **Training-data provenance / leakage:** **LOW** — trivially trained from
    scratch, no foundation-model provenance question applies at all.
12. **Honest eval mode:** trained-from-scratch.
13. **CPU-only feasibility verdict:** **FEASIBLE** — the single cheapest,
    lowest-risk candidate to actually run in this environment.
14. **PyPI install attempt:** via `neuralforecast` 3.2.1.

**WHY THIS SHOULD FAIL ON BTC.** DLinear's entire argument is that most
long-horizon benchmark improvement claimed by complex transformers is
actually just well-tuned linear extrapolation of trend and seasonal
components — which makes it an excellent **sanity-check baseline** but a
poor **mechanism candidate** for something as structurally different as
forward log-vol-rate forecasting: BTC volatility is not trend+seasonal in
any meaningful sense (it's closer to a heavy-tailed, mean-reverting-but-
regime-dependent process), so DLinear's moving-average trend/seasonal split
is likely to decompose the log-vol series into a "trend" that is really just
slowly-decaying memory of the last big shock and a "seasonal" component that
is mostly noise, then linearly extrapolate both — which will look
reasonable in calm regimes and be badly wrong exactly at the volatility
spikes that matter most for a QLIKE-scored forecast (QLIKE punishes
underestimating realized vol heavily). Because it is linear and has almost
no capacity, DLinear cannot represent the asymmetric, convex response of
vol-of-vol to shocks at all — it will systematically underreact to the
tail events that dominate BTC's regime transitions. Its AAAI benchmark
victory over early transformers says nothing about its ability to model
heavy tails, because none of ETT/Traffic/Electricity/Exchange/Weather/ILI
have BTC-scale tail risk in the target variable.

---

### 9. Chronos-2

1. **Name/paper:** "Chronos-2: From Univariate to Universal Forecasting",
   Ansari et al. (Amazon), arXiv:2510.15821 (technical report; no confirmed
   peer-reviewed venue as of this audit).
2. **Repo/PyPI:** `github.com/amazon-science/chronos-forecasting`; PyPI
   package `chronos-forecasting`, confirmed downloadable, latest **2.3.1**.
3. **License code:** Apache-2.0. **License weights:** Apache-2.0 (HF card:
   `license: apache-2.0`) — code and weights match here, no divergence.
4. **Latest checkpoint:** `amazon/chronos-2` on HF. HF card last updated
   5 Jun 2026. **Commit/revision hash: UNKNOWN** — not captured in this
   session (would require a direct HF API/git call not made); the model is
   referenced by its `main` branch tag as of 2026-08-28.
5. **Param count:** 119.5M (single size for Chronos-2 as of this release —
   no small/base/large family the way Bolt has).
6. **Peak RAM, CPU inference, 8,760×512:** **FEASIBLE** — ~480MB fp32 weight
   footprint; encoder-only forward passes on ctx-512 batches, chunked, should
   stay well under a few GB peak RSS on 15GB RAM. CPU inference is
   explicitly supported per the model card ("supporting both GPU and CPU
   inference").
7. **Max/native context length:** **8,192** (per capability table). Max
   prediction length 1,024. Our context of 512 is well within range with
   large headroom.
8. **Target formulation:** multi-step-ahead **quantile** forecasts via a
   quantile-regression head; supports univariate, multivariate, and
   covariate-informed forecasting within one architecture (unified with
   in-context "group attention" across related series and covariates).
9. **Covariate support:** **YES, native** — both past-only and known-future
   real/categorical covariates, per the model card's capability table (the
   only candidate here besides TiRex-2 with this natively, and the only one
   confirmed installable via plain `pip`).
10. **Probabilistic output:** quantiles at caller-specified levels
    (deterministic quantile heads — not raw samples, not a fitted parametric
    distribution).
11. **Training-data provenance / leakage:** trained on (a) a subset of
    `autogluon/chronos_datasets` **excluding** the test portion of any
    dataset overlapping GIFT-Eval, (b) a subset of `Salesforce/
    GiftEvalPretrain`, (c) large-scale synthetic data. `chronos_datasets`'
    ~66 enumerable configs (inspected directly via HF in this session) —
    `dominick`, `electricity_15min`, `ercot`, `exchange_rate` (FX only, not
    crypto), `m4_*`, `m5`, `monash_*`, `solar_1h`, `taxi_*`,
    `training_corpus_kernel_synth_1m`/`tsmixup_10m` (synthetic), `ushcn`,
    `weatherbench_*`, `wiki_daily_100k`, `wind_farms_*` — contain **no
    BTC/crypto-named series**. `GiftEvalPretrain`'s internal series list
    could **not** be fully enumerated (its HF dataset-viewer failed with a
    500 error in this session), so its "Econ/Fin" domain contents are not
    independently confirmed crypto-free. **LEAKAGE RISK: MEDIUM** — no
    direct hit found in the parts that could be audited, but a meaningful
    fraction of the pretraining mixture (GiftEvalPretrain's Econ/Fin slice,
    plus any undisclosed synthetic augmentation) could not be ruled out.
12. **Honest eval mode:** **zero-shot only** — fine-tuning a 120M
    encoder-heavy model on 4 CPU cores with no GPU is not practically
    feasible within a reasonable time budget.
13. **CPU-only feasibility verdict:** **FEASIBLE** for zero-shot inference;
    **INFEASIBLE** for any fine-tuning path.
14. **PyPI install attempt:** `pip download --no-deps chronos-forecasting`
    succeeded (2.3.1). `torch<3,>=2.2` requirement is satisfied by the
    installed 2.13.0 — **no conflict**.

**WHY THIS SHOULD FAIL ON BTC.** Chronos-2's headline strength — long
context, multivariate and covariate-informed in-context learning, quantile
output — is trained and validated on GIFT-Eval/fev-bench/Chronos-Benchmark-II,
which are dominated by econ/energy/healthcare/nature/sales/transport/
web-cloud-ops series with comparatively tame tail behavior; none of those
benchmarks contain anything with BTC's combination of long memory *and*
sudden regime discontinuity *and* fat left/right tails in the vol series
itself. As a zero-shot model it has never seen our specific target
transform (`log(RV) − ½·log(H)`) — it will be asked to forecast a
quantity engineered specifically to make horizon comparison fair, which its
pretraining never explicitly optimized for; whatever "reasonable prior"
its quantile head encodes for a series shaped like this is an artifact of
whatever superficially similar-looking series existed in its training mix,
not evidence of BTC-specific understanding. Because in-context learning
models like Chronos-2 are explicitly designed to lean on the shape of the
provided context window, they are unusually exposed to volatility-of-
volatility: a context window that happens to end during a calm patch will
produce forecasts anchored to that calm regime with no signal telling the
model a regime shift is imminent — precisely the failure mode our
hourly-anchored, overlapping-episode, only-6-independent-years evaluation is
positioned to expose repeatedly rather than average away. Finally, the
MEDIUM leakage flag above means a genuinely strong zero-shot score here is
not fully trustworthy evidence of generalization until GiftEvalPretrain's
contents can be independently confirmed BTC-free.

---

### 10. Chronos-Bolt

1. **Name/paper:** same lineage as the original Chronos paper — "Chronos:
   Learning the Language of Time Series", Ansari et al. (Amazon),
   arXiv:2403.07815. Bolt itself does not appear to have a separate arXiv
   paper — it is documented via the Amazon blog and HF model cards as a
   distilled, direct-multi-step-regression variant of the original
   autoregressive Chronos-T5 models. **Separate peer-reviewed Bolt paper:
   UNKNOWN / not found.**
2. **Repo/PyPI:** same repo, `github.com/amazon-science/chronos-forecasting`;
   same PyPI package `chronos-forecasting` 2.3.1.
3. **License code:** Apache-2.0. **License weights:** Apache-2.0 for all
   four sizes (`amazon/chronos-bolt-{tiny,mini,small,base}`).
4. **Latest checkpoint:** `amazon/chronos-bolt-base`, HF card last updated
   21 Nov 2025. **Commit hash: UNKNOWN**, not captured.
5. **Param count:** **base = 205.3M** (per HF metadata — larger than
   Chronos-2's 119.5M despite the "Bolt" efficiency branding; the speed
   advantage comes from direct regression rather than token-by-token
   autoregression, not from being smaller). tiny/mini/small exact counts
   **UNKNOWN** (not individually queried), ordered tiny < mini < small <
   base.
6. **Peak RAM, CPU inference, 8,760×512:** **FEASIBLE** — base checkpoint
   ~820MB fp32 weights; CPU-fast inference is Bolt's specific selling point
   over the original autoregressive Chronos.
7. **Max/native context length:** **2,048** (per Chronos-2's comparison
   table). Max prediction length **64**.
8. **Target formulation:** direct multi-step **quantile** forecast — a
   regression head rather than autoregressive sampling.
9. **Covariate support:** **NO native support** — per Chronos-2's own
   capability table, Bolt is ❌ for both past-only and known-future
   covariates. External covariate regressors can be bolted on (e.g. via
   AutoGluon's tutorial) but this only models per-timestep effects, not
   effects across time, and is explicitly called out as a workaround, not a
   native capability.
10. **Probabilistic output:** quantiles.
11. **Training-data provenance / leakage:** trained on the original Chronos
    training mixture — the same `autogluon/chronos_datasets` audited above
    (no BTC/crypto series found among enumerable configs) plus the original
    paper's ~55-dataset public mixture with KernelSynth-generated synthetic
    series and TSMix augmentation. **LEAKAGE RISK: MEDIUM** — same rationale
    as Chronos-2 (no direct hit found, but full corpus provenance not
    exhaustively auditable from this session's tools).
12. **Honest eval mode:** **zero-shot** — fine-tuning is supported by the
    library, but even the base checkpoint (205M) is heavy to fine-tune on
    CPU only within a reasonable time budget.
13. **CPU-only feasibility verdict:** **FEASIBLE** for zero-shot inference —
    of all the foundation-model candidates, Bolt is explicitly marketed and
    architected for fast CPU inference (a large claimed speedup over the
    original autoregressive Chronos was seen in search results but was not
    independently re-verified here, so is not restated as fact).
14. **PyPI install attempt:** same package/result as Chronos-2 — downloadable,
    `torch<3,>=2.2` compatible with installed 2.13.0.

**WHY THIS SHOULD FAIL ON BTC.** Bolt trades away exactly the capability
most plausibly useful here — native covariate conditioning — in exchange for
CPU speed, so using it commits to a pure univariate zero-shot forecast of
BTC vol with no way to inform it about order flow, funding, or on-chain
signals short of a bolted-on external regressor the model card itself
describes as only capturing per-timestep effects. Its short native max
prediction length (64) and context (2,048) were tuned against GIFT-Eval/
Chronos-Benchmark-II horizons that are mostly much shorter and calmer than
what a 168-hour-ahead BTC vol forecast during a regime transition demands.
Being a regression-style direct multi-horizon quantile head (rather than
autoregressive sampling) means Bolt cannot represent horizon-to-horizon
dependence in its own uncertainty the way a sampling-based model can — its
quantiles at H=1 and H=168 are produced independently, which is a poor match
for a volatility process whose whole defining feature is that current vol is
highly informative about vol many steps ahead (long memory). And as with
Chronos-2, this is a zero-shot foundation model evaluated against a test
period (2021–2026) that overlaps the kind of "recent web-scale/financial
data" era its training corpus could plausibly draw from — the MEDIUM
leakage flag applies with the same caveat that GiftEvalPretrain/undisclosed
augmentation could not be fully audited here.

---

### 11. TimesFM

1. **Name/paper:** TimesFM — "A decoder-only foundation model for
   time-series forecasting", Das et al. (Google Research), **ICML 2024**,
   arXiv:2310.10688.
2. **Repo/PyPI:** `github.com/google-research/timesfm`; PyPI package
   `timesfm`, confirmed downloadable, latest **2.0.2** on the index — note
   this trails the newest HF checkpoints (2.5, 3.0), see below.
3. **License code:** Apache-2.0. **License weights:** Apache-2.0 for
   `google/timesfm-1.0-*` and `google/timesfm-2.5-*`; **but
   `google/timesfm-3.0-pytorch` (created 24 Aug 2026, four days before this
   audit) carries `license: other`, not Apache-2.0** — a confirmed
   license-drift between checkpoint generations that must be checked
   per-checkpoint before use, not assumed uniform across the TimesFM family.
   A third-party fine-tune, `pfnet/timesfm-1.0-200m-fin`, is
   `cc-by-nc-sa-4.0` — non-commercial, and notably finance-domain-tuned
   (flagged for awareness, not proposed for use here given its restrictive
   license and unaudited fine-tuning corpus).
4. **Latest checkpoint:** `google/timesfm-3.0-pytorch` is the newest by
   creation date, but license:other and released only 4 days before this
   audit — treat as unvetted. `google/timesfm-2.5-200m-pytorch` (updated
   2 Oct 2025, Apache-2.0) is the more prudent choice. **Commit hash:
   UNKNOWN**, not captured.
5. **Param count:** 200–231M depending on version (2.5-pytorch: 231.3M per
   HF metadata; 1.0: 200M).
6. **Peak RAM, CPU inference, 8,760×512:** **FEASIBLE** — ~920MB fp32
   weights, comparable order to Chronos-Bolt-base.
7. **Max/native context length:** 1.0 version natively 512; 2.5 version
   supports a configurable `max_context` (example usage shows 1024) — exact
   architectural ceiling for 2.5 **UNKNOWN precisely**, but ≥1024
   demonstrated. Our 512-step context sits right at the 1.0 model's native
   limit and comfortably inside 2.5's.
8. **Target formulation:** point forecast **and** quantile forecast in one
   call — the code example returns both a point forecast and a
   10th–90th-percentile decile array. Decoder-only, patch-based,
   autoregressive multi-horizon generation.
9. **Covariate support:** the core foundation model is **univariate only**;
   a separate covariates *tutorial/extension* exists in the repo using a
   hybrid approach (an external linear model on residuals), not a native
   architectural covariate channel. Treat as "no native support, external
   hack only" — weaker than Chronos-2 or TimeXer here.
10. **Probabilistic output:** quantiles (point + 10th–90th deciles).
11. **Training-data provenance / leakage:** the 2.5 checkpoint's card lists
    pretraining data as `GiftEvalPretrain` + **"Wikimedia Pageviews, cutoff
    Nov 2023"** + **"Google Trends top queries, cutoff EoY 2022"** + synthetic/
    augmented data. Wikipedia pageviews and Google Trends query-volume series
    are exactly the kind of broad web-attention corpus that plausibly
    includes a "Bitcoin"-topic pageview or search-interest series — this is
    **not** BTC price or realized-volatility data, but it is BTC-*topically*
    adjacent time series (public attention to Bitcoin, which itself
    correlates with realized volatility) that the model has plausibly seen
    during pretraining, via a channel none of the other candidates here have
    (search/attention time series rather than price/market time series).
    **LEAKAGE RISK: MEDIUM** — flagged specifically for this topical-
    adjacency channel, distinct from (and arguably a more novel finding
    than) the generic "public financial dataset" concern raised for the
    Chronos/TiRex/Toto family; no direct BTC price/RV series confirmed.
12. **Honest eval mode:** **zero-shot** — fine-tuning the 200–231M
    checkpoints is not CPU-practical here.
13. **CPU-only feasibility verdict:** **FEASIBLE** for zero-shot inference of
    the 1.0/2.5 checkpoints.
14. **PyPI install attempt:** `timesfm` 2.0.2 downloaded successfully via
    `pip download --no-deps`; `torch>=2.0.0` (extra) is satisfied. **Caveat:**
    the 2.5-pytorch model card itself says `pip install` support for the
    newest checkpoint API is "coming soon" and instructs `git clone` +
    `pip install -e .` instead — i.e. the pinned PyPI release (2.0.2) may not
    yet expose the 2.5/3.0 checkpoint-loading API cleanly. This is a minor,
    checkpoint-specific **infrastructure wrinkle** (the 1.0 checkpoint should
    work fine via the PyPI package; 2.5/3.0 may require building from
    source).

**WHY THIS SHOULD FAIL ON BTC.** TimesFM's decoder-only autoregressive
patch generation was tuned and evaluated primarily on Google's internal
mixture plus public benchmarks dominated by smooth, moderately seasonal
series (search trends, web traffic, standard forecasting-competition data);
it has no demonstrated track record on heavy-tailed, volatility-clustered
financial series, and autoregressive multi-step generation is specifically
prone to compounding error under exactly the kind of abrupt level/scale
shifts that BTC vol regime changes produce — an early-horizon overestimate
or underestimate of the "typical" scale propagates forward through
generation rather than being independently corrected step by step. The
model's own quantile output is trained to be well-calibrated on its training
mixture's notion of "typical" tail behavior, which is not calibrated to
BTC's much fatter tails — expect systematic underestimation of the extreme
quantiles precisely where QLIKE scoring punishes hardest. Its context length
(512-1024) is barely enough to see one full volatility cycle for BTC, whose
memory is genuinely long (multi-week to multi-month), so the model is being
asked to extrapolate from less history than the phenomenon's own memory
length. And the newly surfaced Google-Trends/Wikipedia-pageview leakage
channel (item 11 above) means a suspiciously good zero-shot score cannot be
fully trusted as evidence of a transferable volatility-forecasting
mechanism rather than memorized "Bitcoin was in the news a lot in
[period]" attention-volume correlation.

---

### 12. Moirai

1. **Name/paper:** Moirai — "Unified Training of Universal Time Series
   Forecasting Transformers", Woo et al. (Salesforce), **ICML 2024**,
   arXiv:2402.02592. Update: "Moirai 2.0: When less is more for time series
   forecasting," Liu, Aksu, et al., arXiv:2511.11698 (no confirmed
   peer-reviewed venue found for the 2.0 paper as of this audit).
2. **Repo/PyPI:** `github.com/SalesforceAIResearch/uni2ts`; PyPI package
   `uni2ts`, confirmed downloadable, latest **2.0.0**.
3. **License code:** Apache-2.0 (`uni2ts` repo). **License weights:
   `cc-by-nc-4.0`** for every `Salesforce/moirai-*` checkpoint on HF (1.0,
   1.1, MoE, and 2.0-R-small all confirmed `license: cc-by-nc-4.0` via HF
   metadata in this session) — **this is a direct, confirmed code/weights
   license divergence**: the inference code is permissively Apache-2.0, but
   the pretrained weights are non-commercial-only. The 2.0 model card also
   states explicitly: "This release is for research purposes only. The
   proprietary version of this model is used by Salesforce for business
   purposes." Any use of pretrained Moirai weights must respect the
   NC restriction.
4. **Latest checkpoint:** `Salesforce/moirai-2.0-R-small`, updated 29 Jan
   2026. **Commit hash: UNKNOWN**, not captured. Family also includes
   `moirai-1.0-R-{small,base,large}`, `moirai-1.1-R-{small,base,large}`,
   `moirai-moe-1.0-R-{small,base}`, an experimental
   `moirai-1.5-llama-kvcache`, and a Dec-2025 `moirai-agent` (Qwen2-based
   expert-routing approach, architecturally distinct from the rest of the
   family).
5. **Param count:** `moirai-2.0-R-small` = **11.4M** (confirmed via HF
   metadata). The 1.x R-family's small/base/large sizes are commonly cited
   in the literature as roughly 14M/91M/311M respectively, but these figures
   were **not individually re-confirmed via HF metadata in this session** —
   treat as approximate/UNKNOWN-precise.
6. **Peak RAM, CPU inference, 8,760×512:** **FEASIBLE** technically — even
   the largest 1.x checkpoint (~311M) is comfortably CPU-inferenceable on
   15GB RAM.
7. **Max/native context length:** not architecturally fixed the way a
   tokenizer-limited model is; patch-based encoder handles long/variable
   context. Exact practical ceiling **UNKNOWN**.
8. **Target formulation:** **differs by version** — Moirai 1.x used a
   flexible mixture-distribution output (student-t / negative-binomial /
   log-normal mixture, selected per series) for probabilistic forecasting.
   Moirai 2.0's README explicitly states it "switched from a distributional
   loss to a quantile loss formulation" — so 1.x and 2.0 checkpoints are not
   interchangeable in what they natively predict.
9. **Covariate support:** native past and future dynamic covariates plus
   static covariates — part of the original "any-variate attention" design
   (target and covariate series are treated as exchangeable variates within
   the same attention mechanism).
10. **Probabilistic output:** samples/mixture distribution (1.x) or
    quantiles (2.0), depending on checkpoint.
11. **Training-data provenance / leakage: HIGH — confirmed, not inferred.**
    This session directly inspected `Salesforce/lotsa_data`'s dataset
    structure via the HF MCP server and found a config named
    **`bitcoin_with_missing`** (18 series, ~359KB) — this is the Monash Time
    Series Forecasting Archive's Bitcoin dataset, folded into LOTSA, which
    is the pretraining corpus for the original Moirai (1.x) family as
    described in the Moirai paper. Public documentation of the Monash
    archive describes this dataset as ~18 short daily series of
    blockchain/on-chain metrics (transaction counts, hash rate, and related
    market metrics) rather than one long continuous BTC price series — so
    this is not a verbatim leak of "BTC realized volatility 2017–2024," but
    it is unambiguously BTC-market-derived daily data inside the
    pretraining mixture the model has seen. Moirai 2.0's card describes its
    pretraining as drawing on "non-leaking historical context" from GIFT-Eval
    splits plus "mixup data generated from non-leaking subsets of Chronos
    Dataset" plus "internal Salesforce operational data" — it is **not
    stated** whether LOTSA's `bitcoin_with_missing` specifically was
    filtered out of that "non-leaking subset" process, and the 1.x family
    (still the most commonly used/cited Moirai checkpoints) draws on LOTSA
    as originally described, unfiltered for this series as far as could be
    determined. **Verdict: HIGH for Moirai 1.x (direct, confirmed dataset
    match); MEDIUM-to-HIGH/UNKNOWN for Moirai 2.0 pending clarification of
    exactly which LOTSA subsets its "non-leaking" filter excludes.** This is
    the single clearest, most concrete leakage finding in this dossier.
12. **Honest eval mode:** if used at all, zero-shot only, with the leakage
    caveat disclosed prominently — not recommended as a clean zero-shot
    baseline given finding #11.
13. **CPU-only feasibility verdict:** technically feasible for
    small/base-size checkpoints, but see #14 — **effectively BLOCKED BY
    INFRASTRUCTURE** in this environment, compounding the leakage concern.
14. **PyPI install attempt:** `uni2ts` 2.0.0 downloaded successfully via
    `pip download --no-deps`, but its declared dependencies pin
    `numpy~=1.26.0` (installed: 2.4.6), `torch<2.5,>=2.1` (installed: 2.13.0),
    `gluonts~=0.14.3`, and `jax[cpu]` — a real `pip install uni2ts` (with
    dependencies) would attempt to downgrade this environment's already-
    installed torch and numpy, which the task instructed against triggering.
    **BLOCKED BY INFRASTRUCTURE**: installable in an isolated virtualenv,
    not cleanly inside this shared environment.

**WHY THIS SHOULD FAIL ON BTC — and why it should be excluded regardless.**
Independent of the mechanism critique, Moirai fails the single most
important precondition for a "teacher" in this project: **its pretraining
corpus contains Bitcoin-derived data**, confirmed directly in this session.
Any strong zero-shot score from Moirai on a BTC volatility task is
uninterpretable — it could reflect genuine transferable time-series
understanding, or it could simply reflect the model having partially
memorized Bitcoin's historical dynamics during pretraining, and there is no
way to distinguish the two from the outside. Even setting leakage aside, the
mechanism concerns from the rest of this family apply: any-variate attention
and mixture-distribution/quantile heads tuned across LOTSA's mostly
infrastructure/operational/energy/transport domains have no demonstrated
grounding in financial heavy-tailed dynamics, and the version-dependent
target formulation (distributional in 1.x, quantile in 2.0) means results
from one Moirai version cannot be meaningfully compared to another without
re-deriving the scoring pipeline. Given the confirmed leakage plus the
dependency-pin infrastructure block, Moirai should be treated as **excluded
from any zero-shot BTC benchmark**, and at most cited as a "known-
contaminated" reference point rather than a candidate teacher.

---

### 13. TiRex

1. **Name/paper:** TiRex — "TiRex: Zero-Shot Forecasting across Long and
   Short Horizons with Enhanced In-Context Learning", Auer, Podest, Klotz,
   Böck, Klambauer, Hochreiter (NX-AI / JKU Linz), **NeurIPS 2025**,
   arXiv:2505.23719.
2. **Repo/PyPI:** `github.com/NX-AI/tirex`; PyPI package `tirex-ts`,
   confirmed downloadable, latest **1.4.2**.
3. **License code:** per PyPI wheel metadata, **"NXAI COMMUNITY LICENSE
   AGREEMENT"** — a custom, non-OSI license, not MIT/Apache. **License
   weights:** HF card: `license: other`, `license_name:
   nx-ai-community-license` — same custom license for weights as for code
   (no divergence here, but importantly **not** a standard permissive
   license). The exact terms of the NX-AI community license were **not**
   independently read from the primary LICENSE file in this session — treat
   as restrictive/unverified until read, do not assume unencumbered
   commercial or redistribution rights.
4. **Latest checkpoint:** `NX-AI/TiRex`, updated 5 Feb 2026. A decontaminated
   variant `NX-AI/TiRex-1.1-gifteval` also exists (excludes GIFT-Eval
   overlap). **Commit hash: UNKNOWN**, not captured.
5. **Param count:** **35M** (per model card: "35M parameter pre-trained time
   series forecasting model based on xLSTM").
6. **Peak RAM, CPU inference, 8,760×512:** **FEASIBLE** — ~140MB fp32
   weights, the smallest of the foundation-model candidates by a wide
   margin. xLSTM's recurrent backbone (rather than quadratic self-attention)
   gives a favorable CPU compute profile that does not scale quadratically
   with context length.
7. **Max/native context length:** not hard-capped the way a fixed-length
   transformer tokenizer would be — xLSTM recurrence handles arbitrarily
   long context; the model is specifically marketed for both long and short
   horizons via "enhanced in-context learning." Exact practical ceiling
   **UNKNOWN**.
8. **Target formulation:** **both point estimates and quantile estimates**
   natively (per model card).
9. **Covariate support:** **NONE** in TiRex v1 — pure univariate zero-shot
   forecaster. (This is exactly the gap TiRex-2 was built to close.)
10. **Probabilistic output:** quantiles.
11. **Training-data provenance / leakage:** trained on `autogluon/
    chronos_datasets` + `Salesforce/GiftEvalPretrain` — the same two
    corpora examined for Chronos-2/Bolt/Toto. `chronos_datasets`' enumerable
    configs contain no BTC-named series (see #9 above); `GiftEvalPretrain`'s
    contents could not be fully enumerated (viewer error, as noted for
    Chronos-2). **LEAKAGE RISK: MEDIUM** — same rationale/caveat as
    Chronos-2/Bolt.
12. **Honest eval mode:** **zero-shot** — 35M is technically small enough
    that fine-tuning on CPU is conceivable in principle, but the library's
    optimized path is GPU-oriented CUDA sLSTM kernels (`[cuda]` extra); a
    CPU fallback path exists per the docs but is not the tuned/tested path
    for training — treat fine-tuning as **MARGINAL**, zero-shot as the
    honest default recommendation.
13. **CPU-only feasibility verdict:** **FEASIBLE** — the smallest and
    cheapest of all foundation-model candidates, and the base pip install
    has no conflicting hard version pins against the installed environment.
14. **PyPI install attempt:** `tirex-ts` 1.4.2 downloaded successfully via
    `pip download --no-deps`. Full dependency list: `torch`,
    `huggingface-hub`, `numpy`, `scikit-learn` — **all unpinned**, fully
    compatible with the installed versions. CUDA/xLSTM kernel and
    gluonts/hfdataset/notebook extras are all optional. **Cleanest install
    of any foundation-model candidate examined.**

**WHY THIS SHOULD FAIL ON BTC.** TiRex's xLSTM backbone and "enhanced
in-context learning" were validated primarily on GIFT-Eval and the Chronos
zero-shot leaderboard, benchmarks dominated by the same
energy/transport/nature/web-traffic-style series as the rest of this
foundation-model family — nothing there approximates BTC's combination of
fat tails, long memory, and abrupt cross-regime discontinuity. Its
recurrent (rather than attention-based) architecture is efficient, but
recurrence-based long-memory models are prone to gradually "forgetting"
distant context in exactly the situations where BTC vol most needs it —
a multi-week-old regime-defining shock that should still inform today's
forecast. TiRex has **no native covariate mechanism at all** (item 9 above),
so it cannot use order-flow, funding-rate, or on-chain signals even in
principle — it is committed to being a pure autoregressive vol-on-vol
forecaster, discarding exactly the kind of exogenous information that might
help distinguish "this spike is mechanical/liquidation-driven and will mean-
revert fast" from "this spike is the start of a new higher-vol regime." And
because it produces both point and quantile forecasts from a single
zero-shot in-context pass with no explicit regime-detection step, a context
window that happens to end mid-calm will anchor the forecast to that calm
regime with no signal of an impending shift — the same overlapping-episode,
few-independent-regimes evaluation described in the task brief is
positioned to expose this repeatedly.

---

### 14. TiRex-2

1. **Name/paper:** TiRex-2 — "TiRex-2: Generalizing TiRex to Multivariate
   Data and Streaming", Podest, Pichler, Bürger, Zólyomi, Voggenberger,
   Berghammer, Klotz, Böck, Klambauer, Hochreiter (NX-AI), arXiv:2607.01204
   — a very recent 2026 preprint (arXiv id and HF repo creation date, 16 Jun
   2026, both consistent with a mid-2026 release); **no confirmed
   peer-reviewed venue** as of this audit.
2. **Repo/PyPI:** `github.com/NX-AI/tirex-2`. **No PyPI package found** —
   the model card's own quickstart instructs installing via
   [Pixi](https://pixi.prefix.dev) (`curl -fsSL https://pixi.sh/install.sh |
   sh` then `git clone` + `pixi shell-hook`), not `pip`. `pixi.prefix.dev` is
   **not** on this environment's documented CDN/proxy allowlist and was not
   verified reachable in this session.
3. **License code:** **Apache-2.0** (per README front-matter) — notably
   **different from (more permissive than) TiRex v1's custom NX-AI
   community license.** **License weights:** Apache-2.0 (same, per HF card)
   — no code/weights divergence here, and a genuine licensing improvement
   over v1 if it can be used at all.
4. **Latest checkpoint:** `NX-AI/TiRex-2`, updated 21 Jul 2026. Decontaminated
   siblings exist: `NX-AI/TiRex-2-gifteval-zs` ("TiRex-2-g", excludes
   GiftEval pretrain+eval overlap) and `NX-AI/TiRex-2-fevbench` ("TiRex-2-f",
   excludes fev-bench eval overlap); `NX-AI/TiRex-2-gifteval-pretrain`
   ("TiRex-2-gp") is a comparison variant that *includes* GiftEval-Pretrain.
   **Commit hash: UNKNOWN**, not captured.
5. **Param count:** **"38.4M parameters in univariate mode and an additional
   44.1M parameters for multivariate forecasting"** — so roughly 38M active
   for a univariate BTC-vol-only setup, up to ~82M if run in its multivariate
   mode with covariates.
6. **Peak RAM, CPU inference, 8,760×512:** comparable order of magnitude to
   TiRex v1 (tens of millions of active params); the model card is
   explicitly CPU-first — its quickstart code literally shows
   `load_model("NX-AI/TiRex-2", device="cpu")` as the default example, with
   `example`/`example-cu128`/`example-cu126` as separate Pixi environments.
   Nominally **FEASIBLE** by parameter count and documented CPU support,
   contingent on getting the install to work at all (see #14).
7. **Max/native context length:** UNKNOWN exact number; same xLSTM-family
   recurrence caveat as v1 (no hard token-length cap the way attention-based
   models have).
8. **Target formulation:** point-forecast array returned by
   `model.forecast(..., output_type="numpy")` in the quickstart; exact
   distributional/quantile API details for v2 were **not independently
   confirmed** in this session (not re-read line by line from the paper) —
   treat quantile capability as **likely inherited from v1's design intent
   but UNKNOWN whether identical in API/coverage**.
9. **Covariate support:** **YES — this is the headline new capability vs.
   v1**: native conditioning on both past covariates and future-known
   covariates ("calendar features, holidays, promotions, or scheduled
   interventions" per the card), plus native multivariate joint forecasting
   of multiple target variates and a streaming/incremental-update mode.
10. **Probabilistic output:** UNKNOWN precise mechanism for v2 — not
    independently confirmed in this session.
11. **Training-data provenance / leakage: HIGH — confirmed, same evidentiary
    basis as Moirai.** The HF README front-matter lists TiRex-2's
    pretraining datasets as `autogluon/chronos_datasets` **and
    `Salesforce/lotsa_data`** — the latter is the exact LOTSA archive
    directly inspected in this session and confirmed to contain a
    `bitcoin_with_missing` config (see Moirai #11 for the full description
    of that dataset). **None of TiRex-2's three published decontaminated
    variants (g/gp/f) claim to exclude LOTSA's bitcoin dataset specifically
    — their decontamination is benchmark-targeted (GIFT-Eval overlap,
    fev-bench overlap), not BTC-targeted, so even the "clean" TiRex-2
    variants do not address this leakage risk for a BTC volatility task.**
    **LEAKAGE RISK: HIGH**, same caveat as Moirai that the Monash bitcoin
    dataset is ~18 daily on-chain/market-metric series rather than one
    continuous long price/vol series, but unambiguously BTC-market-derived
    data the model has seen.
12. **Honest eval mode:** if used at all, zero-shot with the leakage caveat
    prominently disclosed (same guidance as Moirai); no confirmed PyPI
    package means any fine-tuning tooling is also less turnkey/verified.
13. **CPU-only feasibility verdict:** nominally feasible per the model
    card's own explicit CPU documentation, but **effectively BLOCKED BY
    INFRASTRUCTURE** in practice (see #14) — could not be verified
    installable in this session at all.
14. **PyPI install attempt:** **not attempted — no PyPI package name could
    be identified.** The model's own documented install path requires the
    Pixi package/environment manager, whose installer host
    (`pixi.prefix.dev`) is outside the documented CDN allowlist for this
    container and was not confirmed reachable. This candidate's
    installability could not be verified in this session by any available
    method.

**WHY THIS SHOULD FAIL ON BTC — and why it should be excluded regardless.**
Like Moirai, TiRex-2 fails on provenance grounds independent of any
mechanism critique: its pretraining corpus is confirmed (via this session's
own direct inspection of `Salesforce/lotsa_data`) to include a Bitcoin-named
dataset, and none of NX-AI's published decontamination variants address
that specific series — a strong zero-shot BTC score from any TiRex-2
variant, including the "clean" ones, would be uninterpretable evidence of
generalization versus memorization. On mechanism: the new multivariate/
covariate/streaming machinery is unvalidated on anything resembling BTC vol
as of this very recent (mid-2026) preprint — a model whose main claimed
advance is "generalizing to multivariate data and streaming" for
observability/operational-style domains has no demonstrated track record on
heavy-tailed, regime-switching financial targets, and streaming/incremental
updating (designed for smoothly arriving new observations) is a poor
conceptual match for our hourly-anchored, overlapping-episode evaluation
protocol which deliberately re-anchors and embargoes rather than streams.
Finally, this candidate could not even be confirmed installable in this
environment — a required non-pip, non-allowlisted installer path — so any
claim about its BTC performance is currently untestable here regardless of
the leakage and mechanism concerns.

---

### 15. Toto

1. **Name/paper:** Toto — "This Time is Different: An Observability
   Perspective on Time Series Foundation Models", Cohen, Khwaja, Doubli, et
   al. (Datadog), original arXiv:2505.14766; a Toto-2.0 update references
   arXiv:2602.12147 and arXiv:2605.20119. The HF card's own citation lists
   the venue as "The Thirty-ninth Annual Conference on Neural Information
   Processing Systems" with `year={2026}` (i.e. **NeurIPS 2026**, per the
   card's self-reported bibtex — not independently cross-checked against the
   NeurIPS 2026 program in this session).
2. **Repo/PyPI:** `github.com/DataDog/toto`; PyPI package `toto-ts`,
   confirmed downloadable, latest **0.2.0**.
3. **License code:** Apache-2.0. **License weights:** Apache-2.0 (all
   `Datadog/Toto-*` HF repos confirmed `license: apache-2.0`) — no
   divergence.
4. **Latest checkpoint:** `Toto-Open-Base-1.0` (605MB, original release) plus
   a **Toto-2.0 family** spanning `Toto-2.0-{4m,22m,313m,1B,2.5B}`, a
   fine-tuned `Toto-2.0-2.5B-FT`, an ensemble/meta-learning
   `Toto-2.0-Family-and-Friends` (uses xgboost per its tags — note this is
   **not installed** in this environment), and an experimental multimodal
   `Toto-1.0-QA-Experimental` (VLM + time series, LoRA-adapted, out of scope
   for pure forecasting). **Commit hash: UNKNOWN**, not captured.
5. **Param count:** `Toto-Open-Base-1.0` = **151M** (per README's own
   checkpoint table). Toto-2.0 family spans **4M to 2.5B** — a huge range;
   only the smaller 2.0 sizes (4m/22m/313m) are plausible CPU candidates
   alongside 1.0-Base.
6. **Peak RAM, CPU inference, 8,760×512:** **FEASIBLE** for 1.0-Base
   (~600MB fp32 weights, comparable to Chronos-Bolt-base) and for 2.0-{4m,
   22m,313m}. **MARGINAL to INFEASIBLE** for 2.0-1B/2.5B on 15GB RAM with no
   GPU — the README recommends xFormers/flash-attention "for optimal speed
   and reduced memory usage," both of which are GPU-oriented optimizations;
   a CPU fallback exists but is explicitly not the tuned path, and the
   largest checkpoints' full-precision weight footprint alone (2.5B params
   ≈ 10GB fp32) leaves little headroom for activations on this machine.
7. **Max/native context length:** the README's own usage example uses
   **4,096 timesteps** as a natural working context across 7 variables —
   Toto is explicitly designed for long-context, high-dimensional
   observability workloads. Our 512-step context is well within range.
8. **Target formulation:** **full probabilistic** — explicitly a Student-T
   *mixture model* producing samples; `forecaster.forecast()` returns
   `forecast.median`, `forecast.samples` (256 by default in the example),
   and `forecast.quantile(q)` for arbitrary q. The richest native
   probabilistic output format among all six foundation-model candidates
   examined here.
9. **Covariate support:** Toto is a **multivariate** model (joint modeling
   of many observability metrics via "Proportional Factorized Space-Time
   Attention"), which gives it a way to use *other target-like series* as
   context, but this is joint multivariate forecasting rather than an
   explicit target/covariate split with dedicated known-future-covariate
   conditioning (unlike Chronos-2/TimeXer/TiRex-2). Treat as "past
   covariates via joint multivariate modeling only," not a native
   future-covariate mechanism.
10. **Probabilistic output:** **samples** (default 256) plus derived
    median/quantiles — genuine sample-based probabilistic forecasting.
11. **Training-data provenance / leakage:** trained on **~1 trillion points
    of Datadog's own internal observability metrics** (explicitly "no
    customer data" — infrastructure/APM telemetry: CPU, memory, request
    latency, and similar system metrics) **plus** `Salesforce/
    GiftEvalPretrain` **plus** `autogluon/chronos_datasets` **plus** roughly
    1/3 synthetic data. The two public corpora carry the same MEDIUM
    caveat established for Chronos/TiRex (no confirmed direct BTC series in
    the enumerable parts, GiftEvalPretrain not fully auditable). The
    Datadog-internal trillion-point corpus, however, is thematically about
    as far from BTC market microstructure as a pretraining corpus can be
    (system/infra telemetry, not financial or web-attention data) — **if
    anything this likely reduces aggregate leakage risk relative to
    Chronos/TiRex/TimesFM**, since a large majority of pretraining mass is
    off-topic for BTC, while not eliminating the flag from the shared public
    portion. **LEAKAGE RISK: MEDIUM**, with a note that it is the
    *least-concerning* MEDIUM among the foundation-model candidates given
    the composition of its corpus.
12. **Honest eval mode:** **zero-shot**, recommended for the CPU-feasible
    smaller checkpoints (1.0-Base, 2.0-{4m,22m,313m}).
13. **CPU-only feasibility verdict:** **FEASIBLE** for 1.0-Base and small
    2.0 variants; **MARGINAL/INFEASIBLE** for 2.0-1B/2.5B specifically.
14. **PyPI install attempt:** `toto-ts` 0.2.0 downloaded successfully via
    `pip download --no-deps`, but its declared dependencies carry the
    **most aggressive exact-version pinning of any candidate in this
    dossier**: `torch==2.7.0`, `numpy==1.26.4`, `pandas==2.2.3`,
    `scikit-learn==1.5.0`, `transformers==4.52.1`, `gluonts[torch]==0.16.2`,
    `datasets==2.17.1`, `lightning==2.3.3`, and more, all as hard `==` pins
    rather than ranges. A real `pip install toto-ts` (with dependencies)
    would attempt to simultaneously downgrade torch (2.13.0→2.7.0), numpy
    (2.4.6→1.26.4), pandas (3.0.5→2.2.3), and scikit-learn (1.9.0→1.5.0) in
    this environment. **BLOCKED BY INFRASTRUCTURE** — same category as
    Moirai/uni2ts, but more severe (five-plus hard exact pins vs. uni2ts's
    four range pins), making it the least tractable install of any candidate
    examined without a dedicated isolated environment.

**WHY THIS SHOULD FAIL ON BTC.** Toto's Proportional Factorized Space-Time
Attention and Student-T mixture output were built and validated for
observability metrics — CPU/memory/latency/error-rate telemetry — which
share some surface statistical properties with financial vol (spiky,
occasionally heavy-tailed, bursty) but arise from fundamentally different
generative processes: infra metrics are driven by discrete operational
events (deploys, traffic spikes, capacity limits) with comparatively
well-understood recovery dynamics, not by the reflexive, sentiment-driven,
liquidity-cascade dynamics that make crypto vol regimes persist and
compound the way they do. A Student-T mixture tuned to infra-metric tail
behavior may well *underestimate* how heavy BTC's tails actually get during
a genuine deleveraging cascade, or conversely produce spuriously wide
uncertainty bands calibrated to infra "incident" dynamics that don't map to
financial regime persistence. Toto's multivariate joint-attention design
also has no dedicated mechanism for known-future exogenous covariates
(item 9), so — despite being the most probabilistically expressive
candidate here — it cannot be cleanly given forward-looking information
(e.g., a scheduled macro release time) the way Chronos-2 or TimeXer could.
Even if the mechanism concerns were set aside, the installation itself is
the most fragile of any candidate in this dossier: the exact-pinned
dependency set means any attempt to actually run Toto in this shared
environment risks silently corrupting the numpy/pandas/torch/scikit-learn
versions every other part of this project depends on, which is a
disqualifying practical risk on top of the scientific ones.

---

## Shortlist (ranked by honest feasibility × plausible mechanism relevance to BTC volatility)

| rank | candidate | feasibility | mechanism relevance | one-line justification |
|---|---|---|---|---|
| 1 | **TimeXer** | FEASIBLE | High | Only from-scratch architecture purpose-built for exogenous/covariate conditioning; trains cheaply from scratch so no leakage question applies at all — the best-motivated mechanism candidate here, with the caveat that its covariate-weighting has no regime-change detector. |
| 2 | **DLinear** | FEASIBLE (cheapest of all) | Low-but-essential | Not expected to win — its job is to be the rigorous "can a two-parameter linear model already beat everything else" sanity check the AAAI paper itself demands; any teacher that can't beat DLinear on this target isn't a teacher. |
| 3 | **TiRex** | FEASIBLE (cleanest install of any foundation model) | Medium | Smallest, cheapest, best-behaved zero-shot install; native quantile output; MEDIUM leakage (unaudited GiftEvalPretrain slice only, no confirmed direct BTC hit) but zero native covariate support limits how useful it can be beyond a raw zero-shot reference point. |
| 4 | **Chronos-Bolt** | FEASIBLE | Medium | Fast, well-documented CPU path, native quantiles; MEDIUM leakage, same caveat as TiRex; no native covariates and short native horizon (64) limit fit to our longer-horizon episodes. |
| 5 | **Toto (1.0-Base / 2.0 small)** | FEASIBLE for small checkpoints, **BLOCKED BY INFRASTRUCTURE** for a real `pip install` in this shared environment (exact-pinned deps) | Medium | Richest probabilistic output (full sample-based mixture) and arguably the *lowest*-concern MEDIUM leakage of the foundation models (mostly off-topic infra pretraining corpus) — but the most fragile dependency footprint of any installable candidate; usable only inside an isolated venv. |
| 6 | **Chronos-2** | FEASIBLE | Medium-High | Only zero-shot candidate with full native past+future covariate support and the longest context margin; MEDIUM leakage (same GiftEvalPretrain caveat) and its in-context mechanism is specifically exposed to the "context ends in a calm regime" failure mode described above. |
| 7 | **PatchTST / iTransformer / TSMixer / TimeMixer** | FEASIBLE, small, cheap | Low-Medium | Standard from-scratch long-horizon architectures with no demonstrated grounding in heavy-tailed regime-switching data; useful as a capacity/architecture-diversity sweep alongside TimeXer/DLinear, but none was designed with a mechanism plausibly suited to BTC vol specifically. |
| 8 | **TimesFM** | FEASIBLE (1.0/2.5); newest 3.0 checkpoint's license and PyPI-readiness unvetted | Medium | Native point+quantile output and comfortable context margin, but no native covariates, autoregressive-generation error compounding under regime shifts, and a newly surfaced MEDIUM leakage channel via Wikipedia-pageview/Google-Trends "Bitcoin" topical-attention series in its pretraining mix. |
| 9 | **TimesNet** | FEASIBLE for inference, heavier to train than the rest of the from-scratch family | Low | FFT-based period detection is a poor fit for BTC's broadband, non-stationary spectrum at only 512 hourly steps of context; likeliest of the from-scratch models to lock onto a spurious, regime-specific "period." |
| 10 | **TimeMixer++** | **MARGINAL** — no verified runnable implementation located | Unknown | Cannot be confidently scheduled for any actual run in this project until a maintained, runnable open-source implementation is located and its license read; treat as a research-only entry pending that. |
| 11 | **Moirai** | **BLOCKED BY INFRASTRUCTURE** (hard numpy/torch/gluonts/jax pins conflict with this environment) **and independently disqualified by leakage** | N/A | **HIGH, confirmed leakage** — its LOTSA pretraining corpus contains a named Bitcoin dataset (`bitcoin_with_missing`), directly verified in this session; weights are also CC-BY-NC-4.0 (non-commercial) despite Apache-2.0 code. Exclude from any zero-shot BTC benchmark; cite only as a known-contaminated reference if at all. |
| 12 | **TiRex-2** | **BLOCKED BY INFRASTRUCTURE** (no pip package; requires an unvetted Pixi installer host) **and independently disqualified by leakage** | N/A | **HIGH, confirmed leakage** — same LOTSA `bitcoin_with_missing` exposure as Moirai, and none of its published "decontaminated" variants address it. Could not even be confirmed installable in this session. Exclude on both grounds. |

**Both infrastructure and leakage findings are called out explicitly per the
task's requirement:** Moirai and TiRex-2 are marked **BLOCKED BY
INFRASTRUCTURE** for a true, verifiable reason (dependency-pin conflicts and
a non-allowlisted installer host, respectively) that is separate from — and
in this dossier compounds rather than substitutes for — their independently
disqualifying **HIGH** leakage finding. Toto is BLOCKED BY INFRASTRUCTURE
for a real (non `--no-deps`) install in *this shared* environment
specifically, but is not leakage-disqualified and remains usable in an
isolated virtualenv if a project decides that path is worth the effort.
TimeMixer++'s MARGINAL rating reflects an inability to verify a runnable
implementation at all, not a compute or licensing judgment.
