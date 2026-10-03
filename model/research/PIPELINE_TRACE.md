# NOCTUA — the whole pipeline, traced end to end (2026-09-30)

Every boundary below says what crosses it, and **how the claim was checked by
running code**, not by reading it. Findings are at the end, each with its
status. Re-run the serving half any time with `NOCTUA_DEBUG=1` (see
`model/serve/README.md`); the trace records quoted here are from a live run on
2026-09-30 at 08:1x UTC.

```
 RESEARCH (offline, corpus on HF)                       SERVING (cron, live feed)
 ─────────────────────────────────                      ──────────────────────────
 HF dataset  ──hf_store──▶ btcusd_1min.parquet          Bitstamp 5-min ─┐ (Coinbase fallback)
   (pinned: regenerate.py refuses a different corpus)   serve/fetch ────┤ fetch / fetch.fallback
        │ episodes.build_hourly                                         ▼
        ▼                                               serve/history.get_hours
 btcusd_1h.parquet ──make_bundle──────────────────────▶  bundle (430 d) + live tail ─▶ history
        │ episodes.build_episodes                                        ▼
        ▼                                               predict.forecast
 episodes.parquet ─features─▶ features.parquet            anchor ▶ features ▶ flags ▶ prepare
        │ splits (walk-forward, production_mask)          ▶ predict ▶ vol_correction ▶ payload
        ▼                                                                ▼
 train_v2 ─▶ serve/noctua_v2.npz ◀─ add_*_anchor.py      data/noctua.json, data/kronos.json ─▶ write
        │     (frozen arrays, hashed)                                    ▼
        ▼                                               fetch-data.yml commits ─▶ dashboard src/data.js
 eval/* harnesses ─▶ artifacts/*.json ─▶ ledger.json                      (45-min window, else fallback)
                                                        forward-holdouts.yml ─▶ forward_*.py ─▶ lock files
```

## 1. Research chain

| step | code | output | checked by |
|---|---|---|---|
| corpus | `noctua/hf_store.py`, `noctua/regenerate.py` | `artifacts/btcusd_1min.parquet` (7,681,837 rows) | earlier cycle: download from the HF dataset, sha matched `research/hf_store.json` |
| hourly | `noctua/episodes.build_hourly` | `artifacts/btcusd_1h.parquet`, 2012-01-01 → 2026-08-09 14:00 | this trace: its 8,399 hours that overlap the served bundle are **bit-identical** to the bundle, every column |
| episodes | `noctua/episodes.build_episodes` | `episodes.parquet` | selftests of every consumer (46/46 pass) |
| features | `noctua/features.py` | `features.parquet`; serving calls the same `build_features` | `test_features` (precommit) |
| splits | `noctua/splits.py` | `walk_forward_folds`, `production_mask` = H 19 & 17:00 | used unchanged by every harness |
| model | `noctua/train_v2.py` → `serve/noctua_v2.npz` | 124 arrays (`load_model` trace) | `test_serving` |
| anchor increments | `noctua/add_{hour,weekend,dow,iv}_anchor.py` | arrays appended to the npz; flags ship OFF | serving gates: OFF bit-identical, ON moves by the algebra, gate can fail |
| evidence | `eval/*.py` → `artifacts/*.json` → `research/ledger.json` | 241 entries | `ledger --validate` 0 problems; pitfalls self-test |

## 2. Serving chain (every stage appears in the live `NOCTUA_DEBUG` trace)

| stage | what crossed it on the live run | invariant, and how checked |
|---|---|---|
| `load_model` | NOCTUA-v2, 124 arrays | — |
| `fetch` | `fetch_bitstamp`, 577 five-minute bars, tail 48 h | primary failure before a fallback success is now recorded (`fetch.fallback`, gate check) |
| `history` | 9,641 rows, 0 gaps, 41 new hours, bundle not rewritten | contiguity and ≥ 365 d enforced by `get_hours`; `test_history` |
| `anchor` | last closed hour 07:00, `exact: true`, not clamped | **not 17:00** — see F5 |
| `features` | 42 features, none non-finite | same `build_features` as training |
| `flags` | all four anchor increments off; functional "mean" | forward holdouts force every other flag off (selftests) |
| `prepare` | Xa 1×39, Xb 1×5, Xs 1×21; Log-HAR anchor −5.7165 | — |
| `predict` | blend_w 0.25; sigma_med 0.01433, sigma_mean 0.01812 | — |
| `vol_correction` | factor 0.9725 from **143** episodes (before the fix) | **the window was truncated** — F1, fixed: now 240 of 240 |
| `payload` | sigma 1.762 %, curves monotone both sides | `curves_monotone` computed in the trace |
| `write` | noctua.json 6,540 B, kronos.json 663 B | the trace changes no byte: HEAD's code and this code with the trace on write identical noctua.json (H 19 and 6) |

## 3. Publication and consumers

* `fetch-data.yml` runs `predict.py`, the news/sentiment scripts, then the
  smoke gate, and commits. **Every run in the last 3 days succeeded; GitHub
  fired the `*/30` schedule only 16 times, every 2.6–8.5 h** (F4).
* `src/data.js fetchKronos` reads `data/kronos.json` only if it is under
  **45 min** old; otherwise it scrapes the third-party Kronos demo page through
  free CORS proxies. `upside` from NOCTUA is pinned to 50 on purpose
  (`predict.to_legacy`); the demo's is directional and feeds strike skew and
  the conviction score (F4).
* `serve/app.py` (HF Space) renders `sigma_window_pct` and the curves.
* `forward-holdouts.yml` (daily 13:37 UTC) runs the five `forward_*.py`
  scorers; each counts only, below its N_MIN; the workflow commits lock files
  only.

## 4. Test sweep (this cycle)

| suite | result |
|---|---|
| `scripts/precommit.sh` | 20/20 gates (with `test_debug_trace`) |
| forward holdout selftests | 5/5 (7, 7, 4, 4, 5 checks) |
| every module `--selftest` in eval/, research/, noctua/ | 46/46 at the sweep; 47/47 with `eval/factor_window.py`, added after it (audit D) |
| `ruff --select F` beyond CI's F821 | 12 unused variables + 1 repeated dict key, each inspected: none is a live bug |
| `node scripts/smoke.js` | FAILS on data age only (F9) |

## 5. Findings

| # | finding | status |
|---|---|---|
| F1 | Served 60-day factor window was ~34–41 days: the 400-day bundle could not give the window's anchors their 365-day lookback | **FIXED** — `P4-factor-window-result`; bundle 430 d; gated in `test_adaptive` |
| F2 | `fetch_bars` dropped the primary venue's error when the fallback served | **FIXED** — recorded as `fetch.fallback` |
| F3 | The first live run with the trace on crashed (a foreign dict splatted into kwargs) | **FIXED** before commit; live-path check without network |
| F4 | The cron fires every ~5 h, not 30 min, so the dashboard showed NOCTUA's snapshot ~15 % of the time and the third-party Kronos demo (directional `upside`) the rest | **FIXED 2026-10-03** — one model by design: the snapshot is shown while its 19 h window is open, aged on every read from its anchor; after that the panel is offline, never the demo. Old cache keys dropped (they could hold demo values); labels name NOCTUA; the retail plan uses a neutral 50 when offline (what NOCTUA itself publishes). Gated by `scripts/test-noctua-source.js` (CI + precommit); verified in Chromium with a live and an expired snapshot |
| F5 | Published forecasts anchor at the last closed hour, ~95 % not 17:00 | known — `BENCHMARK.md` §6e measured the claim holds across anchors; `_next_anchor` is unused by design |
| F6 | `forecast` moves an anchor that is not a bar to the next bar without a word | now visible (`anchor.exact`, `anchor.clamped`); every in-repo caller passes exact bars |
| F7 | Dead code: `rv_q` in `predict.forecast` (unused since the first serving commit) | noted, left |
| F8 | `volatility_correction` applies a CLIPPED factor out of band although its reason says "treated as a data fault" | wording only; research's `served_log_factor` clips identically |
| F9 | `smoke.js` fails whenever the cron has not run for 6 h | known; `model-ci.yml` runs it last and says why |

**How the critical ones were found.** F1 and F2 came from the trace, not from
reading: a number (`n_episodes = 143`) that should have been ~240, and a
fallback that left no record. F3 came from running the new tool on the live
path its offline test could not reach. F4 came from measuring the cron's real
cadence and the dashboard's acceptance window together; each looked fine
alone. The common pattern: **a constant or window sized for one consumer and
silently short for a later one** — check it wherever a lookback or a freshness
window is added.
