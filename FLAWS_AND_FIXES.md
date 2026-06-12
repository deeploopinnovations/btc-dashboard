# BTC Dashboard v4 — Flaws Audit & Fix Plan

Reviewed: `index.html`, `src/config.js`, `src/data.js`, `src/main.js`, `src/ui.js`, `src/charts.js`, `src/rateLimit.js`, `deploy.yml`, `.gitignore`, `README_V4.md` on 2026-04-19.

Scope: static site deployed via GitHub Pages at `https://deeploopinnovations.github.io/btc-dashboard/`. The live URL is blocked by our egress policy so findings are derived from source + search evidence, not browser behavior.

Conventions: P1 = correctness/security bug or accessibility blocker; P2 = UX / data quality; P3 = polish.

---

## 1. Correctness bugs (P1)

### 1.1 RateLimit garbage collector prefix collision
`src/rateLimit.js`:
```js
const cur = `${new Date().getUTCFullYear()}-${new Date().getUTCMonth()}`;
for (const k of Object.keys(s)) {
  if (!k.includes(cur)) { delete s[k]; changed = true; }
}
```
`getUTCMonth()` is 0-indexed. In January `cur = "2026-0"`, which `includes()` matches inside `"2026-0-…"`, `"2026-00"` (never produced), etc. But `cur = "2026-1"` (February) substring-matches `"2026-10-…"`, `"2026-11-…"`, `"2026-12-…"` (Nov, Dec of same year if another run produced them). Net effect: Feb's GC will preserve Nov/Dec keys; Jan's GC will preserve Oct/Nov/Dec. Small impact (stale counters) but a real correctness bug.

Fix: use a delimiter-aware comparison.
```js
const cur = `${y}-${mo}-`; // include trailing dash or store zero-padded month
if (!k.startsWith(`${api}_h_${cur}`) && !k.startsWith(...) ) ...
```
Simpler: normalize month to 2 digits everywhere and keep a prefix-safe GC.

### 1.2 GDELT news fetcher clobbers Exa cache
`fetchGdeltNews()` does `cacheSet('news', data, ...)`. When Exa later succeeds, the CryptoPanic/GDELT fallback's cached GDELT is overwritten, but during the GDELT-only branch it writes under the primary `'news'` key. This subtly causes future `cacheGet('news')` to return GDELT even when you'd prefer Exa on next refresh (if Exa's result hadn't yet cached).

Fix: namespace cache keys per source (`news_exa`, `news_cp`, `news_gdelt`) and a top-level `news_best` index.

### 1.3 Placeholder EXA key is treated as a real key
`src/config.js` ships `window.EXA_API_KEY = 'your-exa-api-key-here';`. `fetchNewsSentiment()` only checks `apiKey || null`. The literal placeholder is truthy, so the dashboard will call `api.exa.ai/search` with a bogus key, waste a call (and daily budget), 4xx, then fall through.

Fix: treat the placeholder / empty / known-bad patterns as "no key".
```js
const isPlaceholder = !apiKey || apiKey === 'your-exa-api-key-here' || apiKey.length < 16;
```

### 1.4 Funding threshold comment vs value mismatch
`data.js` funding flagger:
```js
// "Extreme" when magnitude > 0.01% per 8h = 0.03% daily.
flag: Math.abs(rate) > 0.0003 ? 'long-extreme' : …
```
`0.0003` is 0.03%/8h ≈ 0.09%/day, not 0.01%/8h. The comment and the code disagree. Either the comment is wrong (likely, since 0.01% is very low for "extreme") or the value should be `0.0001`.

Fix: align comment with value; document the intent (PDF §3 says ≥ 0.05%/8h for BTC perps during strong regimes).

### 1.5 Kronos source timestamp assumed UTC
`parseSourceTs()`:
```js
return Date.UTC(+m[1], +m[2]-1, +m[3], +m[4], +m[5], +m[6]);
```
The Kronos demo page ("Last Updated: 2026-04-18 17:00:25") does not declare a timezone in its HTML. Treating unknown-TZ timestamps as UTC can misreport freshness by up to ±12h. If the page is served from a server in UTC+8 (Tsinghua), "fresh" becomes "stale" prematurely.

Fix: inspect the demo's HTML for a `<meta>`/JSON island with TZ; if absent, add a server-side GH Action that re-emits timestamps in UTC (see enrichment plan §5).

### 1.6 Confidence-bar animation race
`ui.updateHero()` does:
```js
bar.style.width = '0%';
setTimeout(() => bar.style.width = decision.confidence + '%', 120);
```
If `refreshAll()` triggers twice in quick succession (user presses Refresh during an in-flight update, or slider change races the 30-min tick), multiple pending timers set conflicting widths.

Fix: keep a cancellable handle; or use `requestAnimationFrame` after forcing reflow.

### 1.7 Rate-limit race on parallel fetches
`RateLimit.canCall()` then `RateLimit.record()` are not atomic. `main.js` fires 7 `Promise.all` calls; if two happen to be the same API they both read stale counters before either records. With localStorage-based counters, the effective budget can overshoot by up to N parallel workers.

Fix: optimistically increment before the call; roll back on exception.

### 1.8 Refresh button lacks debounce
`<button onclick="refreshAll()">` calls straight into `refreshAll()` with no in-flight guard. Double-click → two concurrent fetch waves → the bug in 1.7 compounds.

Fix: disable the button while `refreshAll` is running; re-enable in a `finally`.

### 1.9 CryptoPanic RSS parsed with regex including CDATA
`itemRegex` can miss/garble items when the feed includes nested `<![CDATA[...]]>` with HTML. `DOMParser('text/xml')` is safer; fallback to regex only when XML parse fails.

### 1.10 Deribit option parsing discards partial rows silently
`fetchOptions()` keeps `.filter(Boolean)`, but doesn't log how many rows failed the regex or dropped due to missing IV. On an off-day this produces an empty chain and cascades into `findAtmIv() == null` → regime unknown.

Fix: aggregate counts of dropped rows and surface a debug badge when the options chain is thin.

---

## 2. Accessibility (P1 for real users)

Measured against WCAG 2.1 AA; findings likely to fail:

- **No `<main>` landmark.** Entire app is divs. Screen-reader users can't jump to content.
- **No `<h1>`.** The brand name uses `.brand-logo div`, so page has no document title in the outline.
- **Sliders labeled with adjacent `<div>`s, not `<label for>`.** `input[type=range]#rpLots` has no programmatic label. Read by screen readers as "slider, 60".
- **Button `↺ REFRESH` uses an icon character, not an accessible name.** Fine because the text follows, but `aria-label` should say "Refresh all data".
- **SVG gauge has no `role="img"` or `aria-label`.** Dial value is announced as unlabeled graphic.
- **Focus-visible outline is default browser UA.** On a dark theme the default focus ring can be invisible against `#1a1a24`. Add an explicit `:focus-visible` style.
- **Contrast issues.** `--muted: #71717a` on `--surface: #12121a` is ~4.45:1 — OK for body text, **fails AA for text < 14px** (the many 9-10px labels). Confidence sub-labels, session segment text, pulse-card sub-text all likely fail AA for small text (needs 4.5:1 for regular / 3:1 for large).
- **Color-only state for news dots** (pos/neg/neu). Add text like "Bullish"/"Bearish"/"Neutral" tooltip or visually hidden text.
- **Hero verdict change has no `aria-live`.** The most important element silently updates. Add `aria-live="polite"` and `aria-atomic="true"` to `#heroCard`.
- **"LIVE" pulse animation can't be disabled.** Users with vestibular disorders see the constant blink. Respect `prefers-reduced-motion`.
- **News links open in new tab with `target="_blank"` but not announced.** Add `rel="noopener noreferrer"` (noopener is there, noreferrer is missing) and an "(opens in new window)" visually hidden span.
- **Lang attribute is `en`** — good. Direction / currency / timezone not declared; fine.

Fix bundle:
- Add `<main>`, `<h1 class="sr-only">`, `<label>` elements for all sliders, ARIA on SVGs.
- Add `:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }`.
- Bump muted text from `#71717a` to `#9a9aa5` (≈6:1 on `#12121a`).
- Wrap animation `@keyframes blink` inside `@media (prefers-reduced-motion: no-preference)`.
- Add a skip link `<a href="#main" class="skip-link">Skip to content</a>`.

---

## 3. Security / privacy (P1)

- **No Content-Security-Policy meta tag.** A static site that uses four third-party CORS proxies is a particularly high-risk consumer of untrusted HTML. A reasonable CSP would prohibit inline scripts except your own, restrict `connect-src` to the expected APIs and proxies, and block framing.
- **No Subresource Integrity on Chart.js CDN.**
  ```html
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  ```
  If cdnjs is compromised or cache-poisoned, your dashboard runs attacker code. Add `integrity="sha384-…"` and `crossorigin="anonymous"`.
- **Google Fonts `@import` in CSS.** Beyond perf, some regulators (GDPR in Germany) treat GF as a third-party tracker. Consider self-hosting WOFF2 copies.
- **CORS proxies see every request.** `corsproxy.io`, `allorigins`, `cors.lol`, `thingproxy` can observe your upstream URLs and return arbitrary responses. This is the core reason to move Kronos scraping server-side (§5).
- **`target="_blank"` without `rel="noreferrer"`** on news links sends your Referer to third-party news sites.

---

## 4. UX / data-quality (P2)

- **No staleness indicator on pulse cards.** If Binance times out, the card silently shows yesterday's price from the 24h stale cache. Add a small "stale" glyph or gray-out when `data.ts` > 5 min old.
- **"Funding 8h" displayed but no rich explanation.** Retail users don't know what 0.01%/8h means. A tiny tooltip or "= $X/day on $10K" would help.
- **Timezone opacity.** Page displays IST, but Kronos source is in an unknown TZ (§1.5). A user in EST sees "17:00 IST" and "17:00 (source)" — ambiguous.
- **Session advice is hard-coded to Indian market hours.** A global dashboard should at least show phase labels that map to well-known crypto sessions (Asia/Europe/US).
- **News feed does not dedupe** across sources — if CryptoPanic and GDELT both carry the same Reuters story you'll see it twice. Use URL/host + title fuzzy-match to dedupe.
- **Kronos "stale / very-stale" badge** never tells the user what to do. Add action copy: "Stale — re-scrape or skip Kronos in decision".
- **The 48h hourly chart legend is missing y-axis title / tooltip format.** Numeric "$86K" format hides the real number on hover.
- **Refresh button doesn't show a spinner or progress.** The fetch takes 3–10s; the only feedback is the clock staying the same.
- **Empty state for the News card** reads "No news loaded." Should differentiate between "not fetched yet" and "all sources offline".
- **Pulse strip for "Kronos Upside"** uses green/amber/red buckets of `<45 / 45–55 / ≥55`. A dead 50/50 reading is colored amber (neutral) which is fine — but the same 50 from Kronos being offline (defaulted to 50 in `computeSentiment`) is indistinguishable. Pass through a distinct "no data" state.

---

## 5. Connector integration (the headline ask)

MCP servers (Crypto.com, BigData.com, Hugging Face, Exa, Massive Market Data) run where Claude runs — **not** in your browser. A static GitHub Pages site cannot call them directly. The right pattern, already sketched in `README_V4.md` but not implemented, is:

1. **GitHub Actions cron** runs every 10–30 minutes on ubuntu-latest.
2. It calls the external APIs directly (no CORS) using repo secrets.
3. It writes `data/*.json` snapshots into the repo.
4. The dashboard's `DataLayer.fetch*` tries `./data/<src>.json` first and falls back to the browser proxy chain.

Implementation matrix (what each connector actually contributes, and how to fetch it without MCPs at runtime):

| Data source | What you gain | How to fetch server-side | Secret needed |
|---|---|---|---|
| Crypto.com Exchange REST | BTC/USDT ticker, 1m/1h candles, orderbook depth — independent of Binance | `GET https://api.crypto.com/exchange/v1/public/get-tickers?instrument_name=BTC_USDT` | none |
| Kronos demo page | Upside prob, vol amp, source timestamp | Direct GET — no CORS server-side | none |
| Kronos local model (Hugging Face) | Actual forecast from your own machine | Python job (too heavy for free GH runners for every 30m), run locally → publish JSON | HF token optional |
| BigData.com | News tearsheet + market sentiment | MCP-only; keep locally, write JSON into repo via `gh` push, OR use bigdata public API | `BIGDATA_API_KEY` |
| Exa search | High-quality recent BTC news + summaries | `POST https://api.exa.ai/search` | `EXA_API_KEY` |
| Massive Market Data | ETF flows, macro, on-chain metrics | `GET https://api.mmd.xyz/*` (varies) | `MMD_API_KEY` |
| Hugging Face FinBERT sentiment | Proper sentiment vs keyword-hack | `transformers` pipeline in GH Action — or HF Inference API call | `HF_TOKEN` |

Action item: a `.github/workflows/fetch-data.yml` + three node scripts + a small Python step that calls HF Inference API for FinBERT sentiment. See `scripts/` (to be created).

Risks:
- **Commit spam.** Every 30 min → 48 commits/day. Mitigate with `[skip ci]` messages and a `.gitignore` on old snapshots (keep only current). Or push to an orphan `data` branch.
- **Rate limits** on free tiers — mitigate with staggered crons (every 30m for price/funding, every 3h for news).
- **Secrets exposure.** Never echo them to logs. Use repository secrets only.

---

## 6. Local dashboard (CODEX) integration

You mentioned the CODEX dashboard has a "better backend" with fewer news limitations. Two patterns to consider adopting from CODEX (not visible to me — infer from your description):

- **Proxy/aggregator server** running at `127.0.0.1:8787` that fans out to news sources and serves a normalized JSON feed. You can publish the same endpoint spec as a GH Action output, so the GH dashboard gets the same shape.
- **If CODEX runs a local Python job against Kronos**, we reuse that by having it write `C:\Users\DELL\BTC_Dashboard\data\kronos.json` on a schedule; commits flow to Pages.

When you're ready, mount `C:\Users\DELL\BTC_Dashboard_CODEX` into this session and I'll port the better backend pieces.

---

## 7. Performance (P3)

- `<link>` fonts from Google Fonts via CSS `@import` is a render-blocking third-party request.
- No `<link rel="preconnect">` to Binance/Deribit/alternative.me.
- Chart.js is full 205 kB. You use one chart type (line). Build a tiny custom SVG line chart (< 5 kB) or tree-shake Chart.js.
- `refreshAll()` fires 7 parallel fetches but waits for news (proxied) serially before updating state — could be in parallel too.
- `charts.js` creates `new Chart(...)` on every refresh instead of calling `chart.update()`.

---

## 8. Testing

`README_V4.md` says "All 9 tests pass" but there is no `tests/` or `package.json` committed. Either the file is missing from the repo or the claim is aspirational. Add a `scripts/smoke.js` the user can run locally via `node scripts/smoke.js` to sanity-check the compute engines.

---

## 9. Fix plan (what I'll apply)

I will apply the following changes in this session:

1. **Patch `rateLimit.js`** — fix GC prefix collision, add atomic `beginCall/endCall`.
2. **Patch `data.js`** — namespace news cache, handle EXA placeholder, fix funding threshold comment, add staleness flag to every fetcher, include Crypto.com as a second price source, prefer `./data/*.json` snapshot when present.
3. **Patch `main.js`** — debounce refresh, cancel pending animations, surface staleness flags.
4. **Patch `ui.js`** — add stale glyph on pulse cards, ARIA live on hero, clearer news source labels, dedupe news items, respect `prefers-reduced-motion`.
5. **Patch `index.html`** — add `<main>`, `<h1 class="sr-only">`, skip link, `aria-label`s on SVG, labels on sliders, CSP meta, SRI on Chart.js, `rel="noopener noreferrer"` on news.
6. **Add `scripts/parse-kronos.js`** — server-side Kronos parser (no CORS).
7. **Add `scripts/fetch-enrichment.js`** — server-side enrichment via Crypto.com / Exa / BigData / MMD.
8. **Add `scripts/fetch-sentiment.py`** — FinBERT sentiment via HF Inference API.
9. **Add `scripts/smoke.js`** — node smoke test for the quant engines.
10. **Add `.github/workflows/fetch-data.yml`** — cron that publishes `data/*.json` into the repo.
11. **Add `kronos_local/`** — Gradio UI + requirements + PowerShell installer/verifier.
12. **Add `README_V4_1.md`** — what changed and how to deploy.

I will **not** push or commit. All changes land inside `C:\Users\DELL\BTC_Dashboard`; you run `git add . && git commit && git push`.
