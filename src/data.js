/**
 * data.js
 * All data fetching. Every external call checks RateLimit first.
 * Results are cached in localStorage with TTL.
 */
const DataLayer = (() => {

  // ── CACHE HELPERS ────────────────────────────────────────────────────────
  function cacheGet(key) {
    try {
      const item = JSON.parse(localStorage.getItem('btc_cache_' + key));
      if (item && Date.now() < item.expires) return item.data;
    } catch {}
    return null;
  }

  function cacheSet(key, data, ttlMs) {
    try {
      localStorage.setItem('btc_cache_' + key, JSON.stringify({ data, expires: Date.now() + ttlMs }));
    } catch {}
  }

  // ── CRYPTO.COM PRICE ─────────────────────────────────────────────────────
  async function fetchPrice() {
    const cached = cacheGet('price');
    if (cached) return cached;

    const check = RateLimit.canCall('cryptoCom');
    if (!check.allowed) {
      console.warn('[RateLimit]', check.reason);
      return cacheGet('price_stale');
    }

    try {
      const r = await fetch('https://api.crypto.com/v2/public/get-ticker?instrument_name=BTC_USDT');
      const j = await r.json();
      const t = j.result?.data;
      if (!t) throw new Error('No data');
      const data = {
        price:  parseFloat(t.a),
        high:   parseFloat(t.h),
        low:    parseFloat(t.l),
        change: parseFloat(t.c),
        vol:    parseFloat(t.v),
        volUsd: parseFloat(t.vv),
        ts:     Date.now(),
      };
      RateLimit.record('cryptoCom');
      cacheSet('price', data, 60_000);        // 60s TTL
      cacheSet('price_stale', data, 86400_000); // stale fallback
      return data;
    } catch (e) {
      console.error('[fetchPrice]', e);
      return cacheGet('price_stale') || { price: 71164, high: 73814, low: 71075, change: -0.0243, vol: 2447, volUsd: 177524263, ts: Date.now() };
    }
  }

  // ── CRYPTO.COM OHLCV (1H) ────────────────────────────────────────────────
  async function fetchHourly() {
    const cached = cacheGet('hourly');
    if (cached) return cached;

    const check = RateLimit.canCall('cryptoCom');
    if (!check.allowed) return cacheGet('hourly_stale') || [];

    try {
      const r = await fetch('https://api.crypto.com/v2/public/get-candlestick?instrument_name=BTC_USDT&timeframe=1h');
      const j = await r.json();
      const raw = j.result?.data || [];
      const data = raw.slice(-24).map(c => ({
        t: c.t, o: parseFloat(c.o), h: parseFloat(c.h),
        l: parseFloat(c.l), c: parseFloat(c.c), v: parseFloat(c.v),
      }));
      RateLimit.record('cryptoCom');
      cacheSet('hourly', data, 300_000);
      cacheSet('hourly_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchHourly]', e);
      return cacheGet('hourly_stale') || [];
    }
  }

  // ── CRYPTO.COM DAILY (for ATR-7) ─────────────────────────────────────────
  async function fetchDaily() {
    const cached = cacheGet('daily');
    if (cached) return cached;

    const check = RateLimit.canCall('cryptoCom');
    if (!check.allowed) return cacheGet('daily_stale') || [];

    try {
      const r = await fetch('https://api.crypto.com/v2/public/get-candlestick?instrument_name=BTC_USDT&timeframe=1D');
      const j = await r.json();
      const raw = j.result?.data || [];
      const data = raw.slice(-8).map(c => ({
        h: parseFloat(c.h), l: parseFloat(c.l), c: parseFloat(c.c),
      }));
      RateLimit.record('cryptoCom');
      cacheSet('daily', data, 3600_000);
      cacheSet('daily_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchDaily]', e);
      return cacheGet('daily_stale') || [
        {h:73814,l:72514,c:73049},{h:73476,l:71413,c:72962},{h:73156,l:70459,c:71791},
        {h:72875,l:70695,c:71072},{h:72782,l:67725,c:71922},{h:70373,l:68307,c:68851},{h:69156,l:66609,c:69028}
      ];
    }
  }

  // ── FEAR & GREED ─────────────────────────────────────────────────────────
  async function fetchFearGreed() {
    const cached = cacheGet('fg');
    if (cached) return cached;

    const check = RateLimit.canCall('fearGreed');
    if (!check.allowed) return cacheGet('fg_stale') || { value: 38, label: 'Fear' };

    try {
      const r = await fetch('https://api.alternative.me/fng/?limit=1&format=json');
      const j = await r.json();
      const v = j.data?.[0];
      if (!v) throw new Error('No data');
      const data = { value: parseInt(v.value), label: v.value_classification };
      RateLimit.record('fearGreed');
      cacheSet('fg', data, 3600_000 * 6); // 6h TTL
      cacheSet('fg_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchFearGreed]', e);
      return cacheGet('fg_stale') || { value: 38, label: 'Fear' };
    }
  }

  // ── DERIBIT OPTIONS ───────────────────────────────────────────────────────
  async function fetchOptions() {
    const cached = cacheGet('options');
    if (cached) return cached;

    const check = RateLimit.canCall('deribit');
    if (!check.allowed) return cacheGet('options_stale') || null;

    try {
      const r = await fetch(
        'https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option',
        { signal: AbortSignal.timeout(8000) }
      );
      const j = await r.json();
      const rows = j.result || [];
      // Extract strike from instrument name e.g. BTC-12APR26-72000-C
      const parsed = rows
        .map(row => {
          const m = row.instrument_name.match(/BTC-\d{1,2}\w{3}\d{2}-(\d+)-([CP])/);
          if (!m) return null;
          return {
            strike: parseInt(m[1]),
            type: m[2],
            oi: row.open_interest || 0,
            vol: row.volume || 0,
            mark: row.mark_price || 0,
            expiry: row.instrument_name.split('-')[1],
          };
        })
        .filter(Boolean)
        .filter(r => r.strike >= 55000 && r.strike <= 95000);

      RateLimit.record('deribit');
      cacheSet('options', parsed, 600_000); // 10 min TTL
      cacheSet('options_stale', parsed, 86400_000);
      return parsed;
    } catch (e) {
      console.error('[fetchOptions]', e);
      return cacheGet('options_stale') || null;
    }
  }

  // ── KRONOS DEMO (public JSON endpoint) ───────────────────────────────────
  // Kronos publishes a public demo page; we scrape the key metrics once per day at 09:00 IST
  async function fetchKronos() {
    const cached = cacheGet('kronos');
    if (cached) return cached;

    // Only run at/after 09:00 IST
    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    if (nowIST.getHours() < 9) {
      return cacheGet('kronos_stale') || { upside: 36.7, volAmp: 33.3, ts: null };
    }

    const check = RateLimit.canCall('kronos');
    if (!check.allowed) return cacheGet('kronos_stale') || { upside: 36.7, volAmp: 33.3, ts: null };

    try {
      // Kronos publishes metrics on their live demo page as inline text
      const r = await fetch('https://shiyu-coder.github.io/Kronos-demo/', { signal: AbortSignal.timeout(8000) });
      const html = await r.text();

      // Parse upside probability
      const upsideMatch = html.match(/(\d+\.?\d*)\s*%[\s\S]*?higher than the last known price/);
      const volMatch    = html.match(/(\d+\.?\d*)\s*%[\s\S]*?recent historical volatility/);

      const upside = upsideMatch ? parseFloat(upsideMatch[1]) : 36.7;
      const volAmp = volMatch    ? parseFloat(volMatch[1])    : 33.3;

      const data = { upside, volAmp, ts: Date.now() };
      RateLimit.record('kronos');
      cacheSet('kronos', data, 3600_000 * 8); // 8h TTL (only update once daily anyway)
      cacheSet('kronos_stale', data, 86400_000 * 2);
      return data;
    } catch (e) {
      console.error('[fetchKronos]', e);
      return cacheGet('kronos_stale') || { upside: 36.7, volAmp: 33.3, ts: null };
    }
  }

  // ── NEWS SENTIMENT (Exa API — 1 call/hr max) ─────────────────────────────
  // NOTE: In a pure static site, the Exa API key would be exposed in the browser.
  // For production, route through a Cloudflare Worker or Vercel Edge Function.
  // For local/personal use, the key is read from window.EXA_API_KEY set in a config.js.
  async function fetchNewsSentiment() {
    const cached = cacheGet('news');
    if (cached) return cached;

    const check = RateLimit.canCall('exa');
    if (!check.allowed) {
      console.warn('[Exa RateLimit]', check.reason);
      return cacheGet('news_stale') || getStaticNews();
    }

    // No API key = use static fallback (for GitHub Pages public deployment)
    const apiKey = window.EXA_API_KEY || null;
    if (!apiKey) {
      console.info('[Exa] No API key configured — using static news cache');
      return cacheGet('news_stale') || getStaticNews();
    }

    try {
      const r = await fetch('https://api.exa.ai/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
        body: JSON.stringify({
          query: 'Bitcoin BTC price sentiment analysis today',
          numResults: 8,
          useAutoprompt: true,
          startPublishedDate: new Date(Date.now() - 86400_000).toISOString(),
        }),
        signal: AbortSignal.timeout(8000),
      });
      const j = await r.json();
      const items = (j.results || []).map(item => ({
        headline: item.title,
        url: item.url,
        src: new URL(item.url).hostname.replace('www.',''),
        date: item.publishedDate,
        // Simple keyword sentiment
        sent: scoreSentiment(item.title + ' ' + (item.text||'')),
      }));

      const data = { items, ts: Date.now() };
      RateLimit.record('exa');
      cacheSet('news', data, 3600_000);       // 1hr TTL — perfectly aligns with 1/hr limit
      cacheSet('news_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchNewsSentiment]', e);
      return cacheGet('news_stale') || getStaticNews();
    }
  }

  // ── KEYWORD SENTIMENT SCORER ─────────────────────────────────────────────
  // Free, runs in-browser, no API needed. Weights keywords for financial context.
  function scoreSentiment(text) {
    const t = text.toLowerCase();
    const bullish = ['bullish','rally','surge','breakout','recover','buy','etf inflow','institutional','adoption','breakthrough','higher','gain','green','pump','above','rebound','ceasefire'];
    const bearish = ['bearish','crash','drop','fall','bear','sell','liquidat','fear','panic','below','loss','red','dump','warning','risk','decline','bottom','correction','capitulat'];

    let score = 0;
    bullish.forEach(w => { if (t.includes(w)) score += 1; });
    bearish.forEach(w => { if (t.includes(w)) score -= 1; });

    if (score > 0)  return 'pos';
    if (score < 0)  return 'neg';
    return 'neu';
  }

  // ── STATIC NEWS FALLBACK (last fetched from BigData/Exa in session) ───────
  function getStaticNews() {
    return {
      items: [
        { headline: 'Bitcoin could hit $78K first, then crash to $54K — analyst warns', src: 'Crypto Wire', sent: 'neg' },
        { headline: '$4B in crypto longs at risk if BTC closes $67,180 CME gap', src: 'Crypto Briefing', sent: 'neg' },
        { headline: 'BlackRock crypto portfolio drops $20B in Q1 2026 as BTC falls', src: 'Crypto Briefing', sent: 'neg' },
        { headline: 'Bitcoin sees largest short deleveraging of 2026 — $52.7M liquidated', src: 'Crypto Briefing', sent: 'pos' },
        { headline: '$471M ETF inflows; Strategy buys 45K BTC in 30 days', src: 'Benzinga', sent: 'pos' },
        { headline: 'Bitcoin liquidity halves since Sep 2025, worsens in April 2026', src: 'Crypto Briefing', sent: 'neg' },
        { headline: "CryptoQuant: $55K bear market bottom possible in late 2026", src: 'CoinTelegraph', sent: 'neg' },
        { headline: "BTC April win rate 69% historically but 2026 conditions differ", src: 'Yahoo Finance', sent: 'neu' },
      ],
      ts: Date.now() - 3600_000,
    };
  }

  // ── COMPUTE RANGER ───────────────────────────────────────────────────────
  function computeRanger(dailyCandles, fg) {
    const ranges = dailyCandles.slice(-7).map(c => (c.h - c.l) / ((c.h + c.l) / 2) * 100);
    const atr7 = ranges.reduce((a, b) => a + b, 0) / ranges.length;

    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const dow = nowIST.getDay(); // 0=Sun
    const dowMults = [1.12, 1.08, 1.02, 0.98, 0.97, 1.05, 1.10];
    const dowMult = dowMults[dow];

    const ivHv = 1.40;   // estimated; replace with live Deribit IV if available
    const volRatio = 0.75; // approximate today's vol vs avg
    const fgFactor = 1 + (50 - (fg?.value || 38)) / 200;

    const raw = atr7 * Math.pow(ivHv, 0.20) * Math.pow(volRatio, 0.12) * Math.pow(dowMult, 0.05) * fgFactor;
    const safe = raw * 2.2;

    return { atr7, ivHv, volRatio, dowMult, fgFactor, raw, safe };
  }

  // ── COMPUTE SENTIMENT SCORE ───────────────────────────────────────────────
  function computeSentiment(news, kronos, fg) {
    const newsItems = news?.items || [];
    const newsPos = newsItems.filter(i => i.sent === 'pos').length;
    const newsNeg = newsItems.filter(i => i.sent === 'neg').length;
    const newsScore = newsItems.length > 0 ? (newsPos / newsItems.length * 100) : 40;

    const kronosScore  = kronos?.upside  || 36.7;
    const fgScore      = fg?.value       || 38;
    const analystScore = 41; // static from research
    const macroRisk    = 68; // static from news analysis

    // Weighted composite (0=max bearish, 100=max bullish)
    const composite = Math.round(
      kronosScore  * 0.30 +
      fgScore      * 0.25 +
      newsScore    * 0.25 +
      analystScore * 0.10 +
      (100 - macroRisk) * 0.10
    );

    return { composite, newsScore: Math.round(newsScore), kronosScore, fgScore, analystScore, macroRisk };
  }

  return { fetchPrice, fetchHourly, fetchDaily, fetchFearGreed, fetchOptions, fetchKronos, fetchNewsSentiment, computeRanger, computeSentiment };
})();
