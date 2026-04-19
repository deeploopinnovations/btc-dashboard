/**
 * data.js  (v4 — bulletproof Kronos + full PDF quant stack)
 * =====================================================================
 * NEW IN V4:
 *   • Kronos scraping: 4 proxy fallbacks in sequence with DOMParser
 *     (not fragile regex). Fresh-staleness badge based on source timestamp.
 *   • Funding rate (Binance fapi) — free, CORS-safe
 *   • Tiered sizing multiplier (PDF §4.3): 1.0× / 0.6× / 0.3× / skip
 *   • Next-day move odds lookup (PDF backtest on IV/HV20 > 1.76 regime)
 *   • Session context (IST calm/volatile/transition)
 *   • GDELT free news fallback (no key, CORS-proxied)
 *   • All HV20 / ATM-IV / regime / touch-prob / retail-planner retained
 *
 * Dashboard runs in the browser, so only CORS-safe free public endpoints
 * are callable. MCP connectors (BigData, Hugging Face, Exa premium,
 * Massive Market Data) cannot run client-side without keys — an optional
 * GitHub Actions snapshot cron is described in README_V4.md.
 */
const DataLayer = (() => {

  // ── CORS PROXY CHAIN (free, no-key, in priority order) ─────────────────
  // Tried one after another until one succeeds. Covers the case where a
  // proxy goes down, rate-limits us, or returns a stale cached page.
  const CORS_PROXIES = [
    { name: 'corsproxy.io',   wrap: u => 'https://corsproxy.io/?' + encodeURIComponent(u) },
    { name: 'allorigins',     wrap: u => 'https://api.allorigins.win/raw?url=' + encodeURIComponent(u) },
    { name: 'cors.lol',       wrap: u => 'https://api.cors.lol/?url=' + encodeURIComponent(u) },
    { name: 'thingproxy',     wrap: u => 'https://thingproxy.freeboard.io/fetch/' + u },
  ];

  async function fetchViaProxyChain(targetUrl, timeoutMs = 8000) {
    const errors = [];
    for (const p of CORS_PROXIES) {
      try {
        const controller = new AbortController();
        const to = setTimeout(() => controller.abort(), timeoutMs);
        const r = await fetch(p.wrap(targetUrl), { signal: controller.signal });
        clearTimeout(to);
        if (!r.ok) { errors.push(`${p.name}: HTTP ${r.status}`); continue; }
        const txt = await r.text();
        if (!txt || txt.length < 200) { errors.push(`${p.name}: empty`); continue; }
        return { text: txt, proxy: p.name };
      } catch (e) {
        errors.push(`${p.name}: ${e.message || 'err'}`);
      }
    }
    throw new Error('All proxies failed: ' + errors.join(' | '));
  }

  // ── CACHE HELPERS ──────────────────────────────────────────────────────
  function cacheGet(key) {
    try {
      const item = JSON.parse(localStorage.getItem('btc_cache_v4_' + key));
      if (item && Date.now() < item.expires) return item.data;
    } catch {}
    return null;
  }
  function cacheSet(key, data, ttlMs) {
    try {
      localStorage.setItem('btc_cache_v4_' + key,
        JSON.stringify({ data, expires: Date.now() + ttlMs }));
    } catch {}
  }

  // ══════════════════════════════════════════════════════════════════════
  //                          PRICE / CANDLES
  // ══════════════════════════════════════════════════════════════════════

  async function fetchPrice() {
    const cached = cacheGet('price');
    if (cached) return cached;
    if (!RateLimit.canCall('binance').allowed) return cacheGet('price_stale');
    try {
      const r = await fetch('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT',
        { signal: AbortSignal.timeout(8000) });
      const t = await r.json();
      if (!t.lastPrice) throw new Error('No data');
      const data = {
        price:  parseFloat(t.lastPrice),
        high:   parseFloat(t.highPrice),
        low:    parseFloat(t.lowPrice),
        change: parseFloat(t.priceChangePercent) / 100,
        vol:    parseFloat(t.volume),
        volUsd: parseFloat(t.quoteVolume),
        ts:     Date.now(),
      };
      RateLimit.record('binance');
      cacheSet('price', data, 60_000);
      cacheSet('price_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchPrice]', e);
      return cacheGet('price_stale');
    }
  }

  async function fetchHourly() {
    const cached = cacheGet('hourly');
    if (cached) return cached;
    if (!RateLimit.canCall('binance').allowed) return cacheGet('hourly_stale') || [];
    try {
      const r = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=48',
        { signal: AbortSignal.timeout(8000) });
      const raw = await r.json();
      const data = raw.map(k => ({
        t: k[0], o: +k[1], h: +k[2], l: +k[3], c: +k[4], v: +k[5],
      }));
      RateLimit.record('binance');
      cacheSet('hourly', data, 300_000);
      cacheSet('hourly_stale', data, 86400_000);
      return data;
    } catch (e) { console.error('[fetchHourly]', e); return cacheGet('hourly_stale') || []; }
  }

  async function fetchDaily() {
    const cached = cacheGet('daily');
    if (cached) return cached;
    if (!RateLimit.canCall('binance').allowed) return cacheGet('daily_stale') || [];
    try {
      const r = await fetch('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=60',
        { signal: AbortSignal.timeout(8000) });
      const raw = await r.json();
      const data = raw.map(k => ({
        t: k[0], o: +k[1], h: +k[2], l: +k[3], c: +k[4], v: +k[5],
      }));
      RateLimit.record('binance');
      cacheSet('daily', data, 3600_000);
      cacheSet('daily_stale', data, 86400_000 * 2);
      return data;
    } catch (e) { console.error('[fetchDaily]', e); return cacheGet('daily_stale') || []; }
  }

  // ══════════════════════════════════════════════════════════════════════
  //                     FUNDING RATE (NEW — PDF §4)
  // ══════════════════════════════════════════════════════════════════════
  // Perpetual funding. "Extreme" when magnitude > 0.01% per 8h = 0.03% daily.
  // PDF calls out funding extremes as a macro regime flag alongside IV/HV.
  async function fetchFunding() {
    const cached = cacheGet('funding');
    if (cached) return cached;
    if (!RateLimit.canCall('binance').allowed) return cacheGet('funding_stale');
    try {
      const r = await fetch('https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT',
        { signal: AbortSignal.timeout(8000) });
      const j = await r.json();
      const rate = parseFloat(j.lastFundingRate);   // fraction (e.g. 0.0001 = 0.01%)
      const data = {
        rate,
        ratePct:       rate * 100,                  // %
        annualizedPct: rate * 3 * 365 * 100,        // 3× daily × 365 (funding paid every 8h)
        markPrice:     parseFloat(j.markPrice),
        nextFundingMs: j.nextFundingTime,
        flag:          Math.abs(rate) > 0.0003 ? (rate > 0 ? 'long-extreme' : 'short-extreme')
                     : Math.abs(rate) > 0.0001 ? (rate > 0 ? 'long-heavy'   : 'short-heavy')
                     : 'neutral',
        ts: Date.now(),
      };
      RateLimit.record('binance');
      cacheSet('funding', data, 600_000);
      cacheSet('funding_stale', data, 86400_000);
      return data;
    } catch (e) { console.error('[fetchFunding]', e); return cacheGet('funding_stale'); }
  }

  // ══════════════════════════════════════════════════════════════════════
  //                       FEAR & GREED / OPTIONS
  // ══════════════════════════════════════════════════════════════════════

  async function fetchFearGreed() {
    const cached = cacheGet('fg');
    if (cached) return cached;
    if (!RateLimit.canCall('fearGreed').allowed) return cacheGet('fg_stale');
    try {
      const r = await fetch('https://api.alternative.me/fng/?limit=1&format=json',
        { signal: AbortSignal.timeout(8000) });
      const j = await r.json();
      const v = j.data?.[0];
      if (!v) throw new Error('No data');
      const data = { value: parseInt(v.value), label: v.value_classification, ts: Date.now() };
      RateLimit.record('fearGreed');
      cacheSet('fg', data, 3600_000 * 6);
      cacheSet('fg_stale', data, 86400_000);
      return data;
    } catch (e) { console.error('[fetchFearGreed]', e); return cacheGet('fg_stale'); }
  }

  async function fetchOptions() {
    const cached = cacheGet('options');
    if (cached) return cached;
    if (!RateLimit.canCall('deribit').allowed) return cacheGet('options_stale');
    try {
      const r = await fetch(
        'https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option',
        { signal: AbortSignal.timeout(10000) });
      const j = await r.json();
      const rows = j.result || [];
      const parsed = rows.map(row => {
        const m = row.instrument_name.match(/BTC-(\d{1,2}\w{3}\d{2})-(\d+)-([CP])/);
        if (!m) return null;
        return {
          name:       row.instrument_name,
          expiry:     m[1],
          strike:     parseInt(m[2]),
          type:       m[3],
          oi:         row.open_interest || 0,
          vol:        row.volume || 0,
          mark:       row.mark_price || 0,
          markIv:     row.mark_iv || 0,
          bidIv:      row.bid_iv || 0,
          askIv:      row.ask_iv || 0,
          underlying: row.underlying_price || 0,
        };
      }).filter(Boolean);
      RateLimit.record('deribit');
      cacheSet('options', parsed, 600_000);
      cacheSet('options_stale', parsed, 86400_000);
      return parsed;
    } catch (e) { console.error('[fetchOptions]', e); return cacheGet('options_stale'); }
  }

  // ══════════════════════════════════════════════════════════════════════
  //      KRONOS SCRAPER (BULLETPROOF — 4 proxies + DOMParser)
  // ══════════════════════════════════════════════════════════════════════
  // The live page is rendered as static HTML with the two metrics appearing
  // in deterministic sections. Instead of hoping one regex works, we:
  //   1. Try each CORS proxy in sequence until we get HTML
  //   2. Parse HTML with DOMParser (robust to whitespace/tag shifts)
  //   3. Extract by heading → next large % number (3 strategies)
  //   4. Sanity-check (0 ≤ upside ≤ 100, timestamp parseable)
  //   5. Expose freshness: how stale is the source timestamp vs now?

  async function fetchKronos() {
    const cached = cacheGet('kronos');
    if (cached) return cached;
    if (!RateLimit.canCall('kronos').allowed) return cacheGet('kronos_stale');

    const target = 'https://shiyu-coder.github.io/Kronos-demo/';
    let html, proxyUsed;
    try {
      const res = await fetchViaProxyChain(target, 9000);
      html = res.text; proxyUsed = res.proxy;
    } catch (e) {
      console.error('[fetchKronos] proxy chain failed', e);
      return cacheGet('kronos_stale');
    }

    const parsed = parseKronosHtml(html);
    if (!parsed) {
      console.error('[fetchKronos] parse failed, html length=', html.length);
      return cacheGet('kronos_stale');
    }

    // Compute freshness based on source timestamp
    const srcMs = parseSourceTs(parsed.sourceTs);
    const ageHrs = srcMs ? (Date.now() - srcMs) / 3600_000 : null;
    const freshness = ageHrs === null ? 'unknown'
                    : ageHrs < 2   ? 'fresh'
                    : ageHrs < 8   ? 'recent'
                    : ageHrs < 24  ? 'stale'
                    :                'very-stale';

    const data = {
      upside:    parsed.upside,
      volAmp:    parsed.volAmp,
      sourceTs:  parsed.sourceTs,
      sourceMs:  srcMs,
      ageHrs,
      freshness,
      fetchedAt: Date.now(),
      proxy:     proxyUsed,
    };
    RateLimit.record('kronos');
    cacheSet('kronos', data, 3600_000 * 3);            // 3h browser cache
    cacheSet('kronos_stale', data, 86400_000 * 3);
    return data;
  }

  // Kronos HTML parser — strict, defensive, multiple strategies
  function parseKronosHtml(html) {
    // STRATEGY A: DOMParser — look for h3 headings, find following percentage
    try {
      const doc = new DOMParser().parseFromString(html, 'text/html');
      const headings = [...doc.querySelectorAll('h1,h2,h3,h4,h5,p,strong')];
      let upside = null, volAmp = null;
      for (const h of headings) {
        const hText = (h.textContent || '').toLowerCase();
        if (upside === null && hText.includes('upside probability')) {
          // Look at siblings for a "XX.X%" number
          const pct = findNearbyPercent(h);
          if (pct !== null && pct >= 0 && pct <= 100) upside = pct;
        }
        if (volAmp === null && hText.includes('volatility amplification')) {
          const pct = findNearbyPercent(h);
          if (pct !== null && pct >= 0 && pct <= 100) volAmp = pct;
        }
      }
      // Timestamp
      let sourceTs = null;
      const bodyTxt = doc.body?.textContent || '';
      const tsM = bodyTxt.match(/Last Updated[^:]*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})/i);
      if (tsM) sourceTs = tsM[1].trim();

      if (upside !== null && volAmp !== null) {
        return { upside, volAmp, sourceTs, strategy: 'domparser' };
      }
    } catch (e) { console.warn('[parseKronos] DOM strategy failed', e); }

    // STRATEGY B: Labeled-section regex — percent immediately after the label
    // "Upside Probability (Next 24h)</h3>\n16.7%"
    try {
      const labelRe = /Upside\s+Probability[\s\S]{0,200}?(\d+(?:\.\d+)?)\s*%/i;
      const upMatch  = html.match(labelRe);
      const volRe    = /Volatility\s+Amplification[\s\S]{0,200}?(\d+(?:\.\d+)?)\s*%/i;
      const vlMatch  = html.match(volRe);
      const tsM      = html.match(/Last Updated[^:]*:\s*(?:<[^>]+>)?([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})/i);
      if (upMatch && vlMatch) {
        return {
          upside: parseFloat(upMatch[1]),
          volAmp: parseFloat(vlMatch[1]),
          sourceTs: tsM ? tsM[1].trim() : null,
          strategy: 'label-regex',
        };
      }
    } catch (e) { console.warn('[parseKronos] label-regex failed', e); }

    // STRATEGY C: Legacy long-context regex (our v3 fallback)
    try {
      const upM = html.match(/([\d.]+)\s*%[\s\S]{0,400}?higher than the last known price/i);
      const vlM = html.match(/([\d.]+)\s*%[\s\S]{0,400}?recent historical volatility/i);
      if (upM && vlM) {
        const tsM = html.match(/Last Updated[^:]*:\s*(?:<[^>]+>)?([^<*\n]+)/i);
        return {
          upside: parseFloat(upM[1]),
          volAmp: parseFloat(vlM[1]),
          sourceTs: tsM ? tsM[1].trim() : null,
          strategy: 'legacy-regex',
        };
      }
    } catch (e) { console.warn('[parseKronos] legacy failed', e); }

    return null;
  }

  // Helper: starting from a header element, walk following siblings looking
  // for the first "XX.X%" number (ignores the header's own text).
  function findNearbyPercent(startEl) {
    let el = startEl;
    for (let i = 0; i < 12 && el; i++) {
      // Check siblings first
      let sib = el.nextElementSibling;
      for (let j = 0; j < 6 && sib; j++) {
        const txt = (sib.textContent || '').trim();
        const m = txt.match(/^(\d+(?:\.\d+)?)\s*%\s*$/) || txt.match(/(\d+(?:\.\d+)?)\s*%/);
        if (m) {
          const n = parseFloat(m[1]);
          if (n >= 0 && n <= 100) return n;
        }
        sib = sib.nextElementSibling;
      }
      el = el.parentElement;
    }
    return null;
  }

  function parseSourceTs(s) {
    if (!s) return null;
    // "2026-04-18 17:00:25" → treated as UTC
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$/);
    if (!m) return null;
    return Date.UTC(+m[1], +m[2]-1, +m[3], +m[4], +m[5], +m[6]);
  }

  // ══════════════════════════════════════════════════════════════════════
  //                          NEWS (Exa → CryptoPanic → GDELT fallback)
  // ══════════════════════════════════════════════════════════════════════

  async function fetchNewsSentiment() {
    const cached = cacheGet('news');
    if (cached) return cached;

    const apiKey = window.EXA_API_KEY || null;
    if (!apiKey) {
      // No key → use free public news feeds via CORS proxy
      const cp = await fetchCryptoPanicNews();
      if (cp?.items?.length) return cp;
      return await fetchGdeltNews();
    }

    if (!RateLimit.canCall('exa').allowed)
      return cacheGet('news_stale') || await fetchCryptoPanicNews();

    try {
      const r = await fetch('https://api.exa.ai/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
        body: JSON.stringify({
          query: 'Bitcoin BTC price news today analysis',
          numResults: 10,
          useAutoprompt: true,
          startPublishedDate: new Date(Date.now() - 86400_000).toISOString(),
        }),
        signal: AbortSignal.timeout(10000),
      });
      const j = await r.json();
      const items = (j.results || []).map(it => ({
        headline: it.title,
        url:      it.url,
        src:      it.url ? new URL(it.url).hostname.replace('www.','') : 'unknown',
        date:     it.publishedDate,
        sent:     scoreSentiment(it.title + ' ' + (it.text || '')),
      }));
      const data = { items, ts: Date.now(), source: 'Exa' };
      RateLimit.record('exa');
      cacheSet('news', data, 3600_000);
      cacheSet('news_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchNewsSentiment]', e);
      return cacheGet('news_stale') || await fetchCryptoPanicNews();
    }
  }

  async function fetchCryptoPanicNews() {
    const cached = cacheGet('cpnews');
    if (cached) return cached;
    try {
      const res = await fetchViaProxyChain('https://cryptopanic.com/news/rss/?currencies=BTC', 9000);
      const xml = res.text;
      const itemRegex = /<item>[\s\S]*?<title>([\s\S]*?)<\/title>[\s\S]*?<link>([\s\S]*?)<\/link>[\s\S]*?<\/item>/g;
      const items = [];
      let m;
      while ((m = itemRegex.exec(xml)) && items.length < 10) {
        const title = m[1].replace(/<!\[CDATA\[|\]\]>/g, '').trim();
        const link  = m[2].replace(/<!\[CDATA\[|\]\]>/g, '').trim();
        const src   = (() => { try { return new URL(link).hostname.replace('www.','').split('/')[0]; } catch { return 'CryptoPanic'; } })();
        items.push({ headline: title, url: link, src, sent: scoreSentiment(title) });
      }
      const data = { items, ts: Date.now(), source: 'CryptoPanic RSS' };
      cacheSet('cpnews', data, 3600_000);
      cacheSet('news_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchCryptoPanicNews]', e);
      return cacheGet('news_stale') || { items: [], ts: Date.now(), source: 'offline' };
    }
  }

  // GDELT DOC 2.0 — free, no key (returns JSON, but CORS is strict so we proxy)
  async function fetchGdeltNews() {
    try {
      const url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin%20BTC&mode=ArtList&format=json&maxrecords=10&sort=DateDesc';
      const res = await fetchViaProxyChain(url, 9000);
      const j = JSON.parse(res.text);
      const items = (j.articles || []).slice(0, 10).map(a => ({
        headline: a.title,
        url:      a.url,
        src:      a.domain || 'GDELT',
        date:     a.seendate,
        sent:     scoreSentiment(a.title),
      }));
      const data = { items, ts: Date.now(), source: 'GDELT' };
      cacheSet('news', data, 3600_000);
      cacheSet('news_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchGdeltNews]', e);
      return cacheGet('news_stale') || { items: [], ts: Date.now(), source: 'offline' };
    }
  }

  function scoreSentiment(text) {
    const t = (text || '').toLowerCase();
    const bull = ['bullish','rally','surge','breakout','recover','buy','inflow','institutional',
                  'adoption','higher','gain','green','pump','above','rebound','ath','all-time high',
                  'soar','jump','spike','optimistic','accumulat','bull case','upgrade'];
    const bear = ['bearish','crash','drop','fall','bear','sell','liquidat','fear','panic','below',
                  'loss','red','dump','warning','risk','decline','bottom','correction','capitulat',
                  'plunge','tumble','slide','downgrade','weakness'];
    let s = 0;
    bull.forEach(w => { if (t.includes(w)) s++; });
    bear.forEach(w => { if (t.includes(w)) s--; });
    return s > 0 ? 'pos' : s < 0 ? 'neg' : 'neu';
  }

  // ══════════════════════════════════════════════════════════════════════
  //                        QUANTITATIVE ENGINES
  // ══════════════════════════════════════════════════════════════════════

  // HV20 (20-day annualised realised vol from log returns)
  function computeHV20(dailyCandles) {
    if (!dailyCandles || dailyCandles.length < 21) return null;
    const closes = dailyCandles.slice(-21).map(c => c.c);
    const logRets = [];
    for (let i = 1; i < closes.length; i++) logRets.push(Math.log(closes[i] / closes[i-1]));
    const mean = logRets.reduce((a,b) => a+b, 0) / logRets.length;
    const variance = logRets.reduce((s, r) => s + (r-mean)**2, 0) / (logRets.length - 1);
    const dailyStd = Math.sqrt(variance);
    const annualised = dailyStd * Math.sqrt(365) * 100;
    const oneDay = annualised / Math.sqrt(365);
    return { annualised, oneDay, dailyStd, n: logRets.length };
  }

  // HV20 historical series — for the sparkline + trend
  function computeHV20Series(dailyCandles) {
    if (!dailyCandles || dailyCandles.length < 25) return [];
    const closes = dailyCandles.map(c => c.c);
    const logRets = [];
    for (let i = 1; i < closes.length; i++) logRets.push(Math.log(closes[i] / closes[i-1]));
    const out = [];
    for (let i = 19; i < logRets.length; i++) {
      const w = logRets.slice(i-19, i+1);
      const mean = w.reduce((a,b)=>a+b,0) / w.length;
      const v = w.reduce((s,r)=>s+(r-mean)**2,0) / (w.length-1);
      out.push({
        t: dailyCandles[i+1].t,
        hv20: Math.sqrt(v) * Math.sqrt(365) * 100,
      });
    }
    return out;
  }

  // ATR-7 (simple)
  function computeATR7(dailyCandles) {
    if (!dailyCandles || dailyCandles.length < 7) return null;
    const last7 = dailyCandles.slice(-7);
    const ranges = last7.map(c => (c.h - c.l) / ((c.h + c.l) / 2) * 100);
    return ranges.reduce((a,b) => a+b, 0) / ranges.length;
  }

  // ATM IV from Deribit chain (nearest expiry, strike closest to spot)
  function findAtmIv(options, spot) {
    if (!options?.length) return null;
    const expiryMap = {};
    options.forEach(o => { (expiryMap[o.expiry] ||= []).push(o); });
    const parseExp = s => {
      const m = s.match(/(\d{1,2})(\w{3})(\d{2})/);
      if (!m) return Infinity;
      const months = {JAN:0,FEB:1,MAR:2,APR:3,MAY:4,JUN:5,JUL:6,AUG:7,SEP:8,OCT:9,NOV:10,DEC:11};
      return new Date(2000 + +m[3], months[m[2].toUpperCase()], +m[1]).getTime();
    };
    const now = Date.now();
    const expiries = Object.keys(expiryMap)
      .map(e => ({ exp: e, t: parseExp(e) }))
      .filter(x => x.t > now)
      .sort((a,b) => a.t - b.t);
    if (!expiries.length) return null;
    const nearest = expiryMap[expiries[0].exp];
    const strikes = [...new Set(nearest.map(o => o.strike))]
      .sort((a,b) => Math.abs(a-spot) - Math.abs(b-spot));
    for (const k of strikes) {
      const call = nearest.find(o => o.strike === k && o.type === 'C');
      const put  = nearest.find(o => o.strike === k && o.type === 'P');
      if (call && put && (call.markIv || put.markIv)) {
        const iv = (call.markIv + put.markIv) / 2;
        return {
          atmStrike:       k,
          atmIv:           iv,
          callMark:        call.mark * (call.underlying || spot),
          putMark:         put.mark  * (put.underlying  || spot),
          straddleCost:    (call.mark + put.mark) * (call.underlying || spot),
          straddleCostPct: ((call.mark + put.mark) * 100),
          expiry:          expiries[0].exp,
          daysToExpiry:    (expiries[0].t - now) / 86400_000,
        };
      }
    }
    return null;
  }

  // Classic RANGER (kept for back-compat with existing UI pieces)
  function computeRanger(dailyCandles, fg, ivHvRatio) {
    const atr7 = computeATR7(dailyCandles);
    if (!atr7) return null;
    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const dow = nowIST.getDay();
    const dowMults = [1.12, 1.08, 1.02, 0.98, 0.97, 1.05, 1.10];
    const dowMult = dowMults[dow];
    const ivHv = ivHvRatio || 1.40;
    const volRatio = 0.85;
    const fgFactor = 1 + (50 - (fg?.value || 50)) / 200;
    const raw  = atr7 * Math.pow(ivHv, 0.20) * Math.pow(volRatio, 0.12) * Math.pow(dowMult, 0.05) * fgFactor;
    const safe = raw * 2.2;
    return { atr7, ivHv, volRatio, dowMult, fgFactor, raw, safe };
  }

  // IV/HV20 REGIME (PDF §2: danger threshold = 1.76, rounded to 1.8)
  function classifyRegime(atmIvPct, hv20Ann) {
    if (!atmIvPct || !hv20Ann) return { ratio: null, regime: 'unknown', label: '—', allowTrade: false, sizing: 0 };
    const ratio = atmIvPct / hv20Ann;
    let regime, label, allowTrade, sizing;
    if      (ratio < 1.2) { regime = 'green';     label = 'CALM';       allowTrade = true;  sizing = 1.0;  }
    else if (ratio < 1.4) { regime = 'green';     label = 'NORMAL';     allowTrade = true;  sizing = 0.7;  }
    else if (ratio < 1.6) { regime = 'amber';     label = 'CAUTION';    allowTrade = true;  sizing = 0.4;  }
    else if (ratio < 1.8) { regime = 'amber-dark';label = 'REDUCED';    allowTrade = true;  sizing = 0.2;  }
    else                  { regime = 'red';       label = 'NO-TRADE';   allowTrade = false; sizing = 0;    }
    return { ratio, regime, label, allowTrade, sizing, ivPct: atmIvPct, hv20: hv20Ann };
  }

  // Next-day move odds (PDF §1 backtest — conditional on IV/HV20 > 1.76)
  function nextDayMoveOdds(ratio) {
    if (ratio == null) return null;
    if (ratio > 1.76) {
      // HIGH-IV regime — fat tails
      return {
        regimeType: 'high-iv',
        description: 'Elevated IV/HV20 regime — realised vol likely to overshoot',
        odds: [
          { move: '≥ 2%', prob: 0.33 },
          { move: '≥ 4%', prob: 0.16 },
          { move: '≥ 6%', prob: 0.087 },
          { move: '≥ 8%', prob: 0.061 },
          { move: '≥ 10%', prob: 0.045 },
        ],
        daysPct: 10,
      };
    }
    // NORMAL regime — typical distribution (from PDF §5 "repeatable behaviour")
    return {
      regimeType: 'normal',
      description: 'Normal regime — volatility-clustered, tails contained',
      odds: [
        { move: 'Range ≤ 1× hv20_1d', prob: 0.26 },
        { move: 'Range ≤ 1.5× hv20_1d', prob: 0.56 },
        { move: 'Range ≤ 2× hv20_1d', prob: 0.76 },
        { move: 'Range ≤ 2.5× hv20_1d', prob: 0.87 },
        { move: 'Range ≤ 3× hv20_1d', prob: 0.93 },
      ],
      daysPct: 90,
    };
  }

  // Touch probability (PDF §5 empirical: k × hv20_1d buckets)
  function touchProbability(distancePct, hv20_1d) {
    if (!hv20_1d || hv20_1d <= 0) return null;
    const k = distancePct / hv20_1d;
    if (k >= 3.0) return 0.07;
    if (k >= 2.5) return 0.13;
    if (k >= 2.0) return 0.24;
    if (k >= 1.5) return 0.44;
    if (k >= 1.0) return 0.74;
    return 0.90;
  }

  // Session context (PDF §5 + existing calm_period_analysis)
  function computeSessionContext() {
    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const h = nowIST.getHours() + nowIST.getMinutes() / 60;
    let phase, advice, tier;
    if      (h >= 5.5  && h < 8.5)  { phase = 'Pre-Calm';           advice = 'Wait for calm window (08:30–12:30 IST) for tight spreads.'; tier = 'neutral'; }
    else if (h >= 8.5  && h < 12.5) { phase = 'CALM ⭐ (best entry)';advice = 'Ideal execution window. Run IV/HV20 + Kronos checks now.'; tier = 'best'; }
    else if (h >= 12.5 && h < 14)   { phase = 'Post-Calm';          advice = 'Still relatively calm. OK to enter but vol rising soon.'; tier = 'ok'; }
    else if (h >= 14   && h < 17.5) { phase = 'Pre-Volatile';       advice = 'Secondary entry OK 16:30–17:20 for next-day structure.';    tier = 'warn'; }
    else if (h >= 17.5 && h < 18.5) { phase = 'Expiry Transition';  advice = '17:30 IST Delta expiry. Avoid new entries on old structure.'; tier = 'skip'; }
    else if (h >= 18.5 || h < 0.5)  { phase = 'VOLATILE (EU+US)';   advice = 'Highest realised vol window — DO NOT enter new short premium.'; tier = 'skip'; }
    else                            { phase = 'Late-Night';         advice = 'Asian illiquid hours. Monitor only, don\'t trade.'; tier = 'neutral'; }
    return { phase, advice, tier, istHour: h };
  }

  // Retail seller planner (from PDF §2)
  function buildRetailPlan({
    price, options, atmInfo, hv20, kronosUpside, regime,
    shortLots = 60, safetyFactor = 1.15, touchThreshold = 0.10,
  }) {
    if (!price || !atmInfo || !hv20 || !options) {
      return { ok: false, reason: 'Missing inputs (price/ATM/HV20/options)' };
    }
    if (!regime.allowTrade) {
      return { ok: false, reason: `IV/HV20 = ${regime.ratio?.toFixed(2)} → ${regime.label}. Skip today.` };
    }
    const direction = kronosUpside >= 55 ? 'bullish' : kronosUpside <= 45 ? 'bearish' : 'neutral';
    if (direction === 'neutral') {
      return { ok: false, reason: `Kronos ${kronosUpside}% ≈ 50/50. No directional edge — use symmetric condor instead.` };
    }
    const sellSide = direction === 'bullish' ? 'P' : 'C';
    const sellSideLabel = direction === 'bullish' ? 'PUTS (below spot)' : 'CALLS (above spot)';
    const reqPremPerLot = (atmInfo.straddleCost * safetyFactor) / shortLots;
    const candidates = options
      .filter(o => o.expiry === atmInfo.expiry && o.type === sellSide)
      .map(o => ({
        ...o,
        premium: o.mark * (o.underlying || price),
        distPct: ((o.strike - price) / price) * 100,
        absDist: Math.abs(((o.strike - price) / price) * 100),
      }))
      .filter(o => sellSide === 'P' ? o.strike < price : o.strike > price);
    const viable = candidates.filter(o => o.premium >= reqPremPerLot);
    if (!viable.length) {
      return {
        ok: false,
        reason: `No OTM ${sellSideLabel} pay ≥ $${reqPremPerLot.toFixed(2)}/lot required to finance ${shortLots}-lot wing.`,
        direction, sellSide, atmInfo, reqPremPerLot, candidates: candidates.slice(0, 5),
      };
    }
    const scored = viable.map(o => ({ ...o, touchProb: touchProbability(o.absDist, hv20.oneDay) }))
      .filter(o => o.touchProb !== null && o.touchProb <= touchThreshold)
      .sort((a,b) => b.absDist - a.absDist);
    if (!scored.length) {
      return {
        ok: false,
        reason: `All premium-viable strikes have touch probability > ${(touchThreshold*100).toFixed(0)}%. Market too volatile for this structure today.`,
        direction, sellSide, atmInfo, reqPremPerLot,
        candidates: viable.slice(0, 5).map(o => ({ ...o, touchProb: touchProbability(o.absDist, hv20.oneDay) })),
      };
    }
    const best = scored[0];
    const totalShortPremium = best.premium * shortLots;
    const netCredit = totalShortPremium - atmInfo.straddleCost;
    return {
      ok: true, direction, sellSide, sellSideLabel, atmInfo,
      shortStrike: best.strike,
      shortDistancePct: best.absDist,
      shortPremiumPerLot: best.premium,
      shortLots,
      totalShortPremium,
      straddleCost: atmInfo.straddleCost,
      netCredit,
      touchProb: best.touchProb,
      reqPremPerLot,
      regime,
      alternatives: scored.slice(1, 4),
    };
  }

  // Composite sentiment score
  function computeSentiment(news, kronos, fg, regime) {
    const items = news?.items || [];
    const newsPos = items.filter(i => i.sent === 'pos').length;
    const newsNeg = items.filter(i => i.sent === 'neg').length;
    const newsNeu = items.filter(i => i.sent === 'neu').length;
    const newsScore = items.length
      ? Math.round((newsPos / Math.max(1, newsPos + newsNeg + newsNeu)) * 100)
      : 40;
    const kronosScore   = kronos?.upside || 50;
    const fgScore       = fg?.value || 50;
    const volAmpScore   = kronos?.volAmp || 50;
    const regimePenalty = regime?.regime === 'red' ? 20
                        : regime?.regime === 'amber-dark' ? 10
                        : regime?.regime === 'amber' ? 5 : 0;
    const composite = Math.round(
      kronosScore * 0.35 + fgScore * 0.25 + newsScore * 0.25 +
      (100 - volAmpScore) * 0.15 - regimePenalty
    );
    return {
      composite: Math.max(0, Math.min(100, composite)),
      newsScore, kronosScore, fgScore, volAmpScore,
      newsPos, newsNeg, total: items.length,
    };
  }

  // ═══════════════════════════════════════════════════════════════════════
  //                   MASTER DECISION ENGINE (NEW)
  // ═══════════════════════════════════════════════════════════════════════
  // Pulls it all together. Returns { verdict, confidence, reasons[], actionPath }
  // Used to populate the hero decision card.
  function buildDecision({ price, hv20, regime, kronos, retailPlan, session, funding, sentiment }) {
    const reasons = [];
    const blockers = [];

    // Trade allowed by IV/HV20 regime?
    if (!regime?.allowTrade) {
      blockers.push(`IV/HV20 = ${regime?.ratio?.toFixed(2) || '?'} (${regime?.label}) — PDF rule: no short premium above 1.8`);
    } else {
      reasons.push(`IV/HV20 = ${regime.ratio.toFixed(2)} → ${regime.label} (sizing: ${(regime.sizing*100).toFixed(0)}%)`);
    }

    // Session check
    if (session?.tier === 'skip') {
      blockers.push(`Session: ${session.phase} — avoid new entries now`);
    } else if (session?.tier === 'best') {
      reasons.push(`Session: ${session.phase} ✓ (ideal)`);
    } else {
      reasons.push(`Session: ${session?.phase || '—'}`);
    }

    // Funding regime
    if (funding?.flag === 'long-extreme' || funding?.flag === 'short-extreme') {
      blockers.push(`Perp funding extreme (${funding.ratePct.toFixed(4)}%) — crowd positioning risk`);
    } else if (funding) {
      reasons.push(`Funding: ${funding.ratePct.toFixed(4)}% (${funding.flag})`);
    }

    // Kronos freshness
    if (kronos?.freshness === 'very-stale') {
      blockers.push(`Kronos last updated >${kronos.ageHrs?.toFixed(0)}h ago — signal stale`);
    } else if (kronos) {
      reasons.push(`Kronos: ${kronos.upside.toFixed(1)}% upside / ${kronos.volAmp.toFixed(1)}% vol-amp (${kronos.freshness})`);
    }

    // Directional clarity
    if (kronos && Math.abs(kronos.upside - 50) < 5) {
      blockers.push(`Kronos ${kronos.upside.toFixed(1)}% ≈ 50/50 — no directional edge for asymmetric wing`);
    }

    // Plan viability
    const canTrade = retailPlan?.ok === true;

    let verdict, verdictClass;
    if (blockers.length >= 2) {
      verdict = 'NO-TRADE'; verdictClass = 'nt';
    } else if (blockers.length === 1) {
      verdict = canTrade ? 'CAUTION' : 'NO-TRADE';
      verdictClass = canTrade ? 'cau' : 'nt';
    } else {
      verdict = canTrade ? 'TRADE OK' : 'WAIT';
      verdictClass = canTrade ? 'go' : 'cau';
    }

    // Confidence = 0-100 based on positive signals vs blockers
    const confidence = Math.max(0, Math.min(100,
      100 - blockers.length * 25 - (kronos?.freshness === 'stale' ? 10 : 0)
      + (regime?.sizing || 0) * 30
      - (kronos ? Math.abs(50 - kronos.upside) < 5 ? 15 : 0 : 15)
    ));

    return {
      verdict, verdictClass, confidence,
      reasons, blockers,
      canTrade,
      direction: kronos?.upside >= 55 ? 'bullish' : kronos?.upside <= 45 ? 'bearish' : 'neutral',
      tradeStructure: canTrade && retailPlan.ok
        ? `1× long $${retailPlan.atmInfo.atmStrike} straddle + ${retailPlan.shortLots}× short $${retailPlan.shortStrike} ${retailPlan.sellSide}`
        : null,
    };
  }

  return {
    // fetchers
    fetchPrice, fetchHourly, fetchDaily, fetchFearGreed,
    fetchOptions, fetchKronos, fetchNewsSentiment, fetchFunding,
    // quant engines
    computeHV20, computeHV20Series, computeATR7,
    findAtmIv, computeRanger, classifyRegime,
    touchProbability, buildRetailPlan, computeSentiment,
    nextDayMoveOdds, computeSessionContext, buildDecision,
    // util
    scoreSentiment,
  };
})();
