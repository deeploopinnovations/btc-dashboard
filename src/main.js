/**
 * main.js  (v3)
 * Orchestrates all data fetching, UI updates, auto-refresh loops.
 * Adds retail seller plan computation on every full refresh.
 */

let state = {
  price: null, hourly: null, daily: null, fg: null, kronos: null, news: null,
  options: null, ranger: null, sentiment: null,
  // new:
  hv20: null, atmInfo: null, regime: null, retailPlan: null,
};

async function refreshAll() {
  console.log('[Dashboard] refreshAll()');
  UI.updateTimestamp();
  UI.updateRateLimitDisplay(RateLimit.getStats());

  // 1. Price (Binance — CORS-safe)
  state.price = await DataLayer.fetchPrice();
  if (state.price) UI.updatePriceMetrics(state.price);

  // 2. Hourly candles
  state.hourly = await DataLayer.fetchHourly();
  if (state.hourly?.length) Charts.renderPriceChart(state.hourly);

  // 3. Daily candles — 30 bars, enough for HV20 + ATR7
  state.daily = await DataLayer.fetchDaily();

  // 4. HV20 from daily closes
  if (state.daily?.length >= 21) {
    state.hv20 = DataLayer.computeHV20(state.daily);
  }

  // 5. Fear & Greed
  state.fg = await DataLayer.fetchFearGreed();

  // 6. RANGER (classic formula)
  if (state.daily?.length && state.price) {
    state.ranger = DataLayer.computeRanger(state.daily, state.fg);
    UI.updateRanger(state.ranger, state.price.price);
  }

  // 7. Kronos (via CORS proxy)
  state.kronos = await DataLayer.fetchKronos();

  // 8. Options chain (Deribit)
  state.options = await DataLayer.fetchOptions();

  // 9. ATM IV + IV/HV20 regime
  if (state.options && state.price && state.hv20) {
    state.atmInfo = DataLayer.findAtmIv(state.options, state.price.price);
    if (state.atmInfo) {
      state.regime = DataLayer.classifyRegime(state.atmInfo.atmIv, state.hv20.annualised);
    }
  }

  // 10. News
  state.news = await DataLayer.fetchNewsSentiment();
  UI.updateNewsFeed(state.news);

  // 11. Composite sentiment
  state.sentiment = DataLayer.computeSentiment(state.news, state.kronos, state.fg, state.regime);
  UI.updateSentimentGauges(state.sentiment, state.kronos, state.fg, state.regime);
  UI.updateScoreRing(state.sentiment);
  Charts.renderGauges(state.sentiment, state.kronos, state.regime);
  UI.updateSentimentSources(state.sentiment, state.kronos, state.fg);

  // 12. Direction banner + strikes + signals
  if (state.price && state.ranger) {
    UI.updateBanner(state.price.price, state.ranger, state.kronos, state.sentiment, state.regime);
    const { callStrike, putStrike } = UI.updateStrikes(state.price.price, state.ranger, state.kronos?.upside || 50);
    Charts.renderRangeVisual(state.price.price, putStrike, callStrike);
    UI.updateSignals(state.price.price, state.ranger, state.kronos, state.sentiment, state.fg, state.regime, state.hv20);
  }

  // 13. Retail seller plan (the PDF strategy)
  if (state.price && state.options && state.atmInfo && state.hv20 && state.regime && state.kronos) {
    state.retailPlan = DataLayer.buildRetailPlan({
      price:        state.price.price,
      options:      state.options,
      atmInfo:      state.atmInfo,
      hv20:         state.hv20,
      kronosUpside: state.kronos.upside,
      regime:       state.regime,
      shortLots:      parseInt(document.getElementById('rpLots')?.value)   || 60,
      touchThreshold: parseFloat(document.getElementById('rpTouch')?.value || '0.10'),
    });
    UI.updateRetailPlan(state.retailPlan, state.price.price, state.hv20, state.regime, state.atmInfo);
  } else {
    UI.updateRetailPlan(null, state.price?.price, state.hv20, state.regime, state.atmInfo);
  }

  // 14. Supporting visuals
  Charts.renderThetaChart();
  UI.updateTimingBar();
  UI.updateRateLimitDisplay(RateLimit.getStats());

  console.log('[Dashboard] refreshAll complete.', {
    price: state.price?.price, kronos: state.kronos?.upside,
    hv20: state.hv20?.annualised?.toFixed(2), regime: state.regime?.label,
    retailOk: state.retailPlan?.ok,
  });
}

async function refreshPrice() {
  state.price = await DataLayer.fetchPrice();
  if (!state.price) return;
  UI.updatePriceMetrics(state.price);
  UI.updateTimestamp();
  if (state.ranger && state.sentiment) {
    UI.updateBanner(state.price.price, state.ranger, state.kronos, state.sentiment, state.regime);
    const { callStrike, putStrike } = UI.updateStrikes(state.price.price, state.ranger, state.kronos?.upside || 50);
    Charts.renderRangeVisual(state.price.price, putStrike, callStrike);
  }
  UI.updateRateLimitDisplay(RateLimit.getStats());
}

function startRefreshLoops() {
  setInterval(refreshPrice, 60_000);           // price every 60s
  setInterval(refreshAll,   30 * 60_000);       // full refresh every 30 min
  setInterval(() => Charts.renderThetaChart(), 300_000);
}

// Hooked by retail-seller slider inputs
function onRetailSliderChange() {
  if (!state.price || !state.options || !state.atmInfo) return;
  state.retailPlan = DataLayer.buildRetailPlan({
    price:        state.price.price,
    options:      state.options,
    atmInfo:      state.atmInfo,
    hv20:         state.hv20,
    kronosUpside: state.kronos?.upside,
    regime:       state.regime,
    shortLots:      parseInt(document.getElementById('rpLots')?.value)   || 60,
    touchThreshold: parseFloat(document.getElementById('rpTouch')?.value || '0.10'),
  });
  UI.updateRetailPlan(state.retailPlan, state.price.price, state.hv20, state.regime, state.atmInfo);
}

(async function boot() {
  console.log('[Dashboard] Booting v3 (CORS-safe + retail mode)…');
  await refreshAll();
  startRefreshLoops();
  console.log('[Dashboard] Ready. Price ~60s, full refresh ~30min.');
})();
