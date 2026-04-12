/**
 * main.js
 * Orchestrates all data fetching, UI updates, and auto-refresh loops.
 * Enforces rate limits before every fetch.
 */

// ── STATE ───────────────────────────────────────────────────────────────────
let state = {
  price:    null,
  hourly:   null,
  daily:    null,
  fg:       null,
  kronos:   null,
  news:     null,
  options:  null,
  ranger:   null,
  sentiment:null,
};

// ── FULL REFRESH ─────────────────────────────────────────────────────────────
async function refreshAll() {
  console.log('[Dashboard] refreshAll() called');
  UI.updateTimestamp();

  // Update rate limit display first (always allowed, pure localStorage read)
  UI.updateRateLimitDisplay(RateLimit.getStats());

  // 1. Fetch price (Crypto.com, very generous limits)
  state.price = await DataLayer.fetchPrice();
  if (state.price) UI.updatePriceMetrics(state.price);

  // 2. Fetch hourly candles for chart
  state.hourly = await DataLayer.fetchHourly();
  if (state.hourly?.length) Charts.renderPriceChart(state.hourly);

  // 3. Fetch daily candles for ATR
  state.daily = await DataLayer.fetchDaily();

  // 4. Fear & Greed (1x/day limit enforced inside DataLayer)
  state.fg = await DataLayer.fetchFearGreed();

  // 5. Compute RANGER (pure math, no API)
  if (state.daily?.length && state.price) {
    state.ranger = DataLayer.computeRanger(state.daily, state.fg);
    UI.updateRanger(state.ranger, state.price.price);
  }

  // 6. Kronos (1x/day enforced, reads cached if already fetched today)
  state.kronos = await DataLayer.fetchKronos();

  // 7. News sentiment (1x/hr via Exa, enforced inside DataLayer)
  state.news = await DataLayer.fetchNewsSentiment();
  UI.updateNewsFeed(state.news);

  // 8. Options (Deribit, generous limits)
  state.options = await DataLayer.fetchOptions();

  // 9. Composite sentiment
  if (state.price) {
    state.sentiment = DataLayer.computeSentiment(state.news, state.kronos, state.fg);
    UI.updateSentimentGauges(state.sentiment);
    UI.updateScoreRing(state.sentiment);
    Charts.renderGauges(state.sentiment);
    UI.updateSentimentSources(state.sentiment);

    // 10. Update banner and direction
    if (state.ranger) {
      UI.updateBanner(state.price.price, state.ranger, state.kronos, state.sentiment);
      const { callStrike, putStrike } = UI.updateStrikes(state.price.price, state.ranger, state.kronos?.upside || 36.7);
      Charts.renderRangeVisual(state.price.price, putStrike, callStrike);
      UI.updateSignals(state.price.price, state.ranger, state.kronos, state.sentiment, state.fg);
    }
  }

  // 11. Theta chart and timing bar (always fresh, no API)
  Charts.renderThetaChart();
  UI.updateTimingBar();

  // 12. Final rate limit display update
  UI.updateRateLimitDisplay(RateLimit.getStats());

  console.log('[Dashboard] refreshAll() complete');
}

// ── PRICE-ONLY REFRESH (every 60s, very cheap) ───────────────────────────────
async function refreshPrice() {
  state.price = await DataLayer.fetchPrice();
  if (!state.price) return;
  UI.updatePriceMetrics(state.price);
  UI.updateTimestamp();

  if (state.ranger && state.sentiment) {
    UI.updateBanner(state.price.price, state.ranger, state.kronos, state.sentiment);
    const { callStrike, putStrike } = UI.updateStrikes(state.price.price, state.ranger, state.kronos?.upside || 36.7);
    Charts.renderRangeVisual(state.price.price, putStrike, callStrike);
  }
  UI.updateRateLimitDisplay(RateLimit.getStats());
}

// ── AUTO-REFRESH LOOPS ────────────────────────────────────────────────────────
function startRefreshLoops() {
  // Price: every 60 seconds (Crypto.com has very generous free limits)
  setInterval(refreshPrice, 60_000);

  // Full refresh: every 60 minutes (aligns with Exa 1/hr limit perfectly)
  setInterval(refreshAll, 3_600_000);

  // Theta chart: every 5 minutes (pure calculation, no API)
  setInterval(() => Charts.renderThetaChart(), 300_000);
}

// ── GLOBAL REFRESH BUTTON ─────────────────────────────────────────────────────
// Already wired in HTML: onclick="refreshAll()"

// ── BOOT ─────────────────────────────────────────────────────────────────────
(async function boot() {
  console.log('[Dashboard] Booting…');
  await refreshAll();
  startRefreshLoops();
  console.log('[Dashboard] Ready. Price refreshes every 60s, full refresh every 60min.');
})();
