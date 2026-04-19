# BTC Options Decision Desk · v4

> Decision-focused redesign of the KRONOS/RANGER dashboard, with a bulletproof Kronos scraper, the full PDF research baked in (HV20 · IV/HV20 regime · touch-probability curve · retail seller planner · session awareness · funding extremes), and an optional GitHub Actions upgrade that unlocks the premium MCP connectors you asked about.

---

## What changed vs v3

| Area | v3 | v4 |
|---|---|---|
| **Layout** | Everything competes for attention. You scroll to find the answer. | Hero **Decision Card** at top: verdict (TRADE OK / CAUTION / NO-TRADE) + confidence % + reasons/blockers, before anything else. |
| **Kronos scrape** | Single CORS proxy + fragile regex → went stale | **4 proxies in sequence** (corsproxy.io → allorigins → cors.lol → thingproxy) + **DOMParser** + 3-strategy fallback + freshness badge based on source timestamp |
| **IV/HV20 regime** | 3 buckets (green/amber/red) | **5 tiers with position sizing** (PDF §4.3): 100% / 70% / 40% / 20% / 0% based on exact ratio |
| **Move odds** | not shown | Conditional **next-24h odds table** that adapts to regime (normal: 26/56/76/87/93% · high-IV: 33/16/9/6/5%) |
| **Funding** | not tracked | Binance perp funding rate, flagged as blocker when extreme (PDF §3) |
| **Session** | static ribbon | Phase-aware with live "NOW" marker + tier-coded advice |
| **News fallback** | CryptoPanic only | CryptoPanic → GDELT (both proxied, no key) |
| **Fetching** | Sequential (slow) | **Promise.all parallel** — all independent endpoints at once |

---

## Live data sources (all free, all browser-reachable)

- **Binance** `/api/v3/ticker/24hr` + `/api/v3/klines` — price + hourly + daily (CORS-enabled, no key)
- **Binance** `/fapi/v1/premiumIndex` — perpetual funding rate (CORS-enabled)
- **Deribit** `/api/v2/public/get_book_summary_by_currency` — options chain + ATM IV
- **alternative.me** `/fng/` — Fear & Greed Index
- **Kronos demo** (scraped via proxy chain) — upside probability + vol amplification
- **CryptoPanic RSS** (proxied) — news fallback when no Exa key
- **GDELT DOC 2.0** (proxied) — secondary news fallback
- **Exa** (optional, with key) — best news sentiment if you set `window.EXA_API_KEY`

---

## About the MCP connectors (Crypto.com, BigData.com, Hugging Face, Exa, Massive Market Data)

You asked whether I could pull from these. Here's the honest story:

**MCP connectors run where Claude runs, not where your browser runs.** When we're chatting, I can call `Crypto.com:get_ticker`, `Bigdata.com:bigdata_market_tearsheet`, `Exa:web_search_exa`, etc. — that's the channel between Claude and those services. But GitHub Pages serves your dashboard as a pure static site to a user's browser, and browsers cannot call MCP servers directly.

**The solution: a GitHub Actions cron that caches MCP data into a JSON file** your dashboard reads. This gives you the "premium insight" of those connectors baked into a zero-cost static site.

### How to add it (optional, ~20 minutes)

1. Create `.github/workflows/fetch-data.yml`:

```yaml
name: Fetch enrichment data
on:
  schedule:
    - cron: '*/30 * * * *'   # every 30 min
  workflow_dispatch:
permissions:
  contents: write
jobs:
  fetch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Fetch Kronos + Crypto.com + market context
        run: |
          mkdir -p data
          # Kronos — fetch directly (no CORS on server side)
          curl -sL https://shiyu-coder.github.io/Kronos-demo/ > /tmp/kronos.html
          node scripts/parse-kronos.js /tmp/kronos.html > data/kronos.json
          # Binance / Crypto.com / Deribit / funding — fetch fresh
          curl -sL "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT" > data/price.json
          curl -sL "https://api.crypto.com/v2/public/get-ticker?instrument_name=BTC_USDT" > data/crypto_com.json
          curl -sL "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT" > data/funding.json
      - name: Commit snapshot
        run: |
          git config user.email "actions@github.com"
          git config user.name "GH Actions"
          git add data/
          git diff --staged --quiet || (git commit -m "auto: snapshot $(date -u +%FT%TZ)" && git push)
```

2. Add `scripts/parse-kronos.js` (simple Node script that parses the same HTML my `data.js` parses — use the `parseKronosHtml` logic from there).

3. In `data.js`, at the top of `fetchKronos`, try the cached file first:

```javascript
// v4.1 OPTIONAL — prefer server-side snapshot over browser proxy
try {
  const r = await fetch('./data/kronos.json?_=' + Date.now(), { signal: AbortSignal.timeout(3000) });
  if (r.ok) {
    const snap = await r.json();
    if (snap.upside != null) return { ...snap, proxy: 'github-actions' };
  }
} catch {}
// ...existing proxy-chain fallback
```

Advantages:
- **Zero CORS** — your own static file
- **Always fresh** (30-min max staleness, set by cron)
- **Bigger universe possible** — add BigData.com tearsheets, Hugging Face sentiment models, Exa news search, Massive Market Data endpoints — all via scripts that run server-side

If you want, tell me to proceed and I'll write the `parse-kronos.js` and a `fetch-enrichment.js` that hits BigData + Exa APIs given their keys.

---

## Deploy v4

```bash
cd C:\Users\DELL\BTC_Dashboard
# Replace the 6 files (keep deploy.yml, README.md, LICENSE, .gitignore from v3)
git add .
git commit -m "v4: decision-first UI + bulletproof Kronos + funding + odds tables"
git push origin main
```

GitHub Actions redeploys in ~30s. Hard-refresh the live URL.

---

## Smoke test results (local Node run)

All 9 tests pass:

```
TEST 1: HV20 from synthetic BTC data → annualised 23.0%, 1-day 1.2%
TEST 2: Regime tiers — 5 buckets with correct sizing multipliers 
TEST 3: Touch prob curve — 3× hv20_1d = 7%, 2× = 24%, 1.5× = 44%, 1× = 74%
TEST 4: Odds lookup — high-IV (33/16/9/6/5%) vs normal (26/56/76/87/93%)
TEST 5: Session context — pre-calm / calm / volatile phases work
TEST 6: Bearish plan → sells 60× $86K calls, 13.2% distance, 7% touch, $532 credit
TEST 7: Decision engine → TRADE OK, 100% confidence, full reasoning chain
TEST 8: NO-TRADE regime correctly blocks with 3 clear reasons
TEST 9: Kronos HTML parser correctly extracts 16.7% / 100% / 2026-04-18 17:00:25
```
