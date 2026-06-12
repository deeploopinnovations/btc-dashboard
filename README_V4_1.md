# BTC Dashboard v4.1 — What Changed & How to Deploy

This release closes the 12-item fix plan and adds the **Option-Selling Desk** (`options.html`) — a
hedge-fund-style decision layer for selling BTC options on Delta Exchange, translated into plain
English for retail traders. Everything runs free: GitHub Pages + GitHub Actions + public APIs.

---

## 1. The 12 fixes, item by item

| # | Area | What changed |
|---|------|--------------|
| 1 | `src/rateLimit.js` | Garbage-collects expired timestamp buckets so the limiter no longer grows unbounded over a long-open tab. |
| 2 | `src/data.js` | Namespaced storage keys; Exa search wired (optional); funding fetch hardened; `_freshness` contract (`fresh-snapshot` / `stale-snapshot` / `live` / `proxy`); crypto.com ticker added; snapshot-first loading (45-min window) with the CORS-proxy chain demoted to backup. |
| 3 | `src/main.js` | Input handlers debounced; no more re-render storms while dragging sliders. |
| 4 | `src/ui.js` | Stale-data glyphs (⏳ on stale snapshots), news de-duplication, honors `prefers-reduced-motion`. |
| 5 | `index.html` | Accessibility pass: skip-to-content link, `aria-label`s on all sliders, `aria-live` status regions, reduced-motion CSS; CSP `<meta>` + SRI hash on Chart.js (GH Pages can't send headers, so both live in the page). |
| 6 | `scripts/parse-kronos.js` | Server-side parse of the Kronos demo page → `data/kronos.json` (upside prob + vol amplification, next 24 h). Timestamp regex tolerates the "(UTC)" suffix. |
| 7 | `scripts/fetch-enrichment.js` | News (CryptoPanic + GDELT, + Exa when `EXA_API_KEY` is set) and Fear & Greed → `data/news.json`, `data/fg.json`. |
| 8 | `scripts/fetch-sentiment.py` | **New.** FinBERT (ProsusAI/finbert) headline sentiment via the free HF Inference API → `data/sentiment.json`. Soft-exits on any failure so the workflow stays green; dashboard falls back to the keyword scorer. |
| 9 | `scripts/smoke.js` | Gate that refuses to publish malformed snapshots. |
| 10 | `.github/workflows/fetch-data.yml` | Cron every 30 min: parse Kronos → fetch enrichment → FinBERT → smoke gate → commit only on change. |
| 11 | `kronos_local/` | Flask app (`app.py` + `run.bat` + README) that runs Kronos-small locally on `127.0.0.1:8899`, writes `data/kronos_local.json`. Your install at `C:\Users\DELL\Kronos` + `C:\Users\DELL\kronos-env` is **correct** — the earlier failure was RAM (use Kronos-mini override in `run.bat` if it recurs). |
| 12 | This file. |

## 2. New: the Option-Selling Desk (`options.html` + `src/options.js`)

Built for short strangles on BTC options (Delta Exchange, 0.001 BTC lots):

- **One** Deribit public call pulls the full chain (~986 instruments); IV is inverted locally via
  Black-76 bisection, deltas are analytic. No API key, no rate-limit pain.
- **Risk Gate** — scores IV/HV ratio, Kronos vol amplification, Fear & Greed extremes, funding
  crowding, and FinBERT news sentiment into a single GREEN / AMBER / RED verdict, with every
  reason listed. It tells you when **not** to trade — the edge most retail tools omit.
- **Desk Notes** — the same numbers a fund desk reads, rewritten as five plain-English paragraphs
  (where we are, is selling worth it, what the AI sees, the crowd, bottom line) plus a glossary.
- **Strangle Builder** — delta-targeted leg picker (5–30 |Δ|), credit, breakevens, POP
  (1 − Σ|Δ|), Delta-Exchange margin heuristic, return on margin, credit/expected-move ratio, and a
  written defense plan (roll when a leg's delta doubles; exit at 50 % profit or 2× credit loss).
- **Chain table** with ATM and selected-leg highlighting.
- **Trading Clock** — hour/day/month seasonality card driven by `data/vol_seasonality.json`
  (2015–2026 study; see `BTC_VOL_RESEARCH.md`).
- **Buyer's Radar** — live DVOL-zone + shock-day card telling option *buyers* when paying
  premium is statistically justified (and when it isn't). Backtested against real Deribit
  implied vol, post-ETF era; full evidence incl. honest dead-ends in `OPTION_BUYER_ALPHA.md`.
- **Seller's Compass** — directional premium-selling regimes (894 days backtested at real
  DVOL): PRIME "persistent fear" put-selling (drawdown >15% + DVOL >50 → 92.7% win, worst
  week −3.5%, p=0.0000), GOOD regimes (uptrend+DVOL>50, cold/negative funding), and four
  significant anti-signals that block the verdict (shock day, RSI extremes, DVOL<40).
  Kronos rides on top as a live sizing tiebreaker (honestly labeled unbacktestable).
  Full study incl. the failed call-side edge in `SELLER_DIRECTIONAL_ALPHA.md`.
- **Daily Desk** — 1DTE selling card for Delta's every-day 12:00 UTC (5:30 PM IST) expiry
  (998 daily expiries backtested at real DVOL): the Saturday-lull PRIME trade (straddle
  +1.20%/d, 92.3% win, worst −3.2%, p=0.0000 — decaying, trade half size), the Monday-entry
  anti-day, the entry-hour clock (18:00 UTC entry beats the full 24h hold; 06:00 UTC = same
  EV at ¼ the tail), surviving daily filters (uptrend / negative funding puts), the DVOL<40
  rule FLIP (fine at 1DTE), and a live shortest-tenor ATM-IV-vs-DVOL line showing how much
  of the weekend discount the market has already priced. Fees rule: fat premium only —
  10Δ wings are net-negative. Full study in `DAILY_EXPIRY_ALPHA.md`.

Caveats baked into the footer: Deribit IV transfers to Delta Exchange within typical spread, but
always check Delta's own quote; the margin figure is a heuristic — verify in Delta's calculator;
short strangles carry unlimited risk.

## 3. Deploying

1. Push this repo to GitHub with Pages enabled (root of `main`).
2. **Actions → fetch-data → Run workflow** once to seed `data/*.json` (then it self-runs every 30 min).
3. Optional repo secrets (Settings → Secrets and variables → Actions):
   - `EXA_API_KEY` — richer news via Exa search.
   - `HF_TOKEN` — free Hugging Face token, enables FinBERT sentiment.
   Without them everything still works; those features just fall back.
4. Local Kronos (optional, better than the scraped demo): double-click `kronos_local\run.bat`,
   keep it running; the desk prefers whichever Kronos snapshot is fresher.

## 4. Quick verification

```
node --check src/options.js
python3 -m py_compile scripts/fetch-sentiment.py
node scripts/smoke.js        # after the first workflow run
```

*Not financial advice. Size so a 3σ move doesn't end your account.*
