/**
 * data.js  (v3 — CORS-safe + retail seller mode)
 * ─────────────────────────────────────────────────────────────────────────
 * FIXED:
 *   - Crypto.com blocked CORS from browser → replaced with Binance (Access-Control-Allow-Origin: *)
 *   - Kronos demo blocked CORS → routed through corsproxy.io (free, no key)
 *   - All stale hardcoded values removed in favour of transparent timestamps
 *
 * ADDED (from retail-seller research PDF):
 *   - HV20 computation from BTCUSDT daily closes (annualised, log-returns)
 *   - IV/HV20 filter (red ≥1.8, amber 1.5–1.8, green <1.5)
 *   - Retail seller strategy engine: ATM straddle cost + 60× OTM wing financing
 *   - Touch probability estimator based on 2.5–3× hv20_1d bands
 *
 * Every external call still goes through RateLimit.
 */
const DataLayer = (() => {

  // ── CORS PROXY (free, no key) ──────────────────────────────────────────
  // Used ONLY for pages that don't send CORS headers (Kronos demo HTML).
  // Binance & CoinGecko return CORS headers natively, so no proxy needed.
  const CORS_PROXY = 'https://corsproxy.io/?';
  const proxied = url => CORS_PROXY + encodeURIComponent(url);

  // ── CACHE HELPERS ──────────────────────────────────────────────────────
  function cacheGet(key) {
    try {
      const item = JSON.parse(localStorage.getItem('btc_cache_' + key));
      if (item && Date.now() < item.expires) return item.data;
    } catch {}
    return null;
  }
  function cacheSet(key, data, ttlMs) {
    try {
      localStorage.setItem('btc_cache_' + key,
        JSON.stringify({ data, expires: Date.now() + ttlMs }));
    } catch {}
  }

  // ── BINANCE PRICE (replaces Crypto.com — CORS-enabled) ─────────────────
  // Binance public endpoints return Access-Control-Allow-Origin: *
  async function fetchPrice() {
    const cached = cacheGet('price');
    if (cached) return cached;

    const check = RateLimit.canCall('binance');
    if (!check.allowed) return cacheGet('price_stale');

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

  // ── BINANCE HOURLY CANDLES (24 bars) ───────────────────────────────────
  async function fetchHourly() {
    const cached = cacheGet('hourly');
    if (cached) return cached;

    const check = RateLimit.canCall('binance');
    if (!check.allowed) return cacheGet('hourly_stale') || [];

    try {
      const r = await fetch(
        'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=24',
        { signal: AbortSignal.timeout(8000) });
      const raw = await r.json();
      // Binance kline: [openTime, o, h, l, c, v, closeTime, quoteVolume, ...]
      const data = raw.map(k => ({
        t: k[0], o: parseFloat(k[1]), h: parseFloat(k[2]),
        l: parseFloat(k[3]), c: parseFloat(k[4]), v: parseFloat(k[5]),
      }));
      RateLimit.record('binance');
      cacheSet('hourly', data, 300_000);
      cacheSet('hourly_stale', data, 86400_000);
      return data;
    } catch (e) {
      console.error('[fetchHourly]', e);
      return cacheGet('hourly_stale') || [];
    }
  }

  // ── BINANCE DAILY CANDLES (30 bars — enough for HV20 + ATR7) ───────────
  async function fetchDaily() {
    const cached = cacheGet('daily');
    if (cached) return cached;

    const check = RateLimit.canCall('binance');
    if (!check.allowed) return cacheGet('daily_stale') || [];

    try {
      const r = await fetch(
        'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=30',
        { signal: AbortSignal.timeout(8000) });
      const raw = await r.json();
      const data = raw.map(k => ({
        t: k[0], o: parseFloat(k[1]), h: parseFloat(k[2]),
        l: parseFloat(k[3]), c: parseFloat(k[4]), v: parseFloat(k[5]),
      }));
      RateLimit.record('binance');
      cacheSet('daily', data, 3600_000);
      cacheSet('daily_stale', data, 86400_000 * 2);
      return data;
    } catch (e) {
      console.error('[fetchDaily]', e);
      return cacheGet('daily_stale') || [];
    }
  }

  // ── FEAR & GREED (alternative.me — CORS-enabled) ───────────────────────
  async function fetchFearGreed() {
    const cached = cacheGet('fg');
    if (cached) return cached;

    const check = RateLimit.canCall('fearGreed');
    if (!check.allowed) return cacheGet('fg_stale');

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
    } catch (e) {
      console.error('[fetchFearGreed]', e);
      return cacheGet('fg_stale');
    }
  }

  // ── DERIBIT OPTIONS (CORS-enabled for public endpoints) ────────────────
  async function fetchOptions() {
    const cached = cacheGet('options');
    if (cached) return cached;

    const check = RateLimit.canCall('deribit');
    if (!check.allowed) return cacheGet('options_stale');

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
          name:   row.instrument_name,
          expiry: m[1],
          strike: parseInt(m[2]),
          type:   m[3],              // 'C' or 'P'
          oi:     row.open_interest || 0,
          vol:    row.volume || 0,
          mark:   row.mark_price || 0,
          markIv: row.mark_iv || 0,  // IV as %
          bidIv:  row.bid_iv || 0,
          askIv:  row.ask_iv || 0,
          underlying: row.underlying_price || 0,
        };
      }).filter(Boolean);

      RateLimit.record('deribit');
      cacheSet('options', parsed, 600_000);
      cacheSet('options_stale', parsed, 86400_000);
      return parsed;
    } catch (e) {
      console.error('[fetchOptions]', e);
      return cacheGet('options_stale');
    }
  }

  // ── KRONOS DEMO (CORS proxy — parses live page) ────────────────────────
  async function fetchKronos() {
    const cached = cacheGet('kronos');
    if (cached) return cached;

    const check = RateLimit.canCall('kronos');
    if (!check.allowed) return cacheGet('kronos_stale');

    try {
      // corsproxy.io is free + fast; fallback to allorigins if it fails
      let html = null;
      try {
        const r = await fetch(proxied('https://shiyu-coder.github.io/Kronos-demo/'),
          { signal: AbortSignal.timeout(10000) });
        html = await r.text();
      } catch {
        const r2 = await fetch(
          'https://api.allorigins.win/raw?url=' + encodeURIComponent('https://shiyu-coder.github.io/Kronos-demo/'),
          { signal: AbortSignal.timeout(10000) });
        html = await r2.text();
      }
      if (!html || html.length < 500) throw new Error('Empty response');

      // Parse: upside probability + vol amplification from the demo HTML
      const upM = html.match(/([\d.]+)\s*%[\s\S]{0,400}?higher than the last known price/i);
      const vlM = html.match(/([\d.]+)\s*%[\s\S]{0,400}?recent historical volatility/i);
      const tsM = html.match(/Last Updated[^:]*:\s*<[^>]+>([^<]+)</i)
               || html.match(/Last Updated[^:]*:\s*\*\*([^*]+)\*\*/i);

      if (!upM || !vlM) throw new Error('Parse failed');
      const data = {
        upside:       parseFloat(upM[1]),
        volAmp:       parseFloat(vlM[1]),
        sourceTs:     tsM ? tsM[1].trim() : null,
        fetchedAt:    Date.now(),
      };
      RateLimit.record('kronos');
      cacheSet('kronos', data, 3600_000 * 4);   // 4h TTL
      cacheSet('kronos_stale', data, 86400_000 * 3);
      return data;
    } catch (e) {
      console.error('[fetchKronos]', e);
      return cacheGet('kronos_stale');
    }
  }

  // ── NEWS SENTIMENT via Exa (optional key) ──────────────────────────────
  async function fetchNewsSentiment() {
    const cached = cacheGet('news');
    if (cached) return cached;

    const apiKey = window.EXA_API_KEY || null;
    if (!apiKey) {
      // No key → use CryptoPanic public RSS via CORS proxy as fallback
      return await fetchCryptoPanicNews();
    }

    const check = RateLimit.canCall('exa');
    if (!check.allowed) return cacheGet('news_stale') || await fetchCryptoPanicNews();

    try {
      const r = await fetch('https://api.exa.ai/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
        body: JSON.stringify({
          query: 'Bitcoin BTC price news today analysis',
          numResults: 8,
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

  // ── FREE NEWS FALLBACK (CryptoPanic public RSS via CORS proxy) ─────────
  async function fetchCryptoPanicNews() {
    const cached = cacheGet('cpnews');
    if (cached) return cached;

    try {
      const url = 'https://cryptopanic.com/news/rss/?currencies=BTC';
      const r = await fetch(proxied(url), { signal: AbortSignal.timeout(10000) });
      const xml = await r.text();
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

  // ── KEYWORD SENTIMENT SCORER ───────────────────────────────────────────
  function scoreSentiment(text) {
    const t = (text || '').toLowerCase();
    const bull = ['bullish','rally','surge','breakout','recover','buy','inflow','institutional',
                  'adoption','higher','gain','green','pump','above','rebound','ath','all-time high',
                  'soar','jump','spike','optimistic','accumulat'];
    const bear = ['bearish','crash','drop','fall','bear','sell','liquidat','fear','panic','below',
                  'loss','red','dump','warning','risk','decline','bottom','correction','capitulat',
                  'plunge','tumble','slide'];
    let s = 0;
    bull.forEach(w => { if (t.includes(w)) s++; });
    bear.forEach(w => { if (t.includes(w)) s--; });
    return s > 0 ? 'pos' : s < 0 ? 'neg' : 'neu';
  }

  // ════════════════════════════════════════════════════════════════════════
  //                     QUANTITATIVE ENGINES
  // ════════════════════════════════════════════════════════════════════════

  // ── HV20 (20-day annualised realised vol from log returns) ─────────────
  // Per research PDF: 20-day HV has best correlation (~0.465) with 30-day forward realised vol
  function computeHV20(dailyCandles) {
    if (!dailyCandles || dailyCandles.length < 21) return null;
    const closes = dailyCandles.slice(-21).map(c => c.c);
    const logRets = [];
    for (let i = 1; i < closes.length; i++) {
      logRets.push(Math.log(closes[i] / closes[i-1]));
    }
    const mean = logRets.reduce((a,b) => a+b, 0) / logRets.length;
    const variance = logRets.reduce((s, r) => s + (r-mean)**2, 0) / (logRets.length - 1);
    const dailyStd = Math.sqrt(variance);
    const hv20Annualised = dailyStd * Math.sqrt(365) * 100;   // %
    const hv20_1d        = hv20Annualised / Math.sqrt(365);   // expected 1-day move %
    return { annualised: hv20Annualised, oneDay: hv20_1d, n: logRets.length };
  }

  // ── ATM IV from Deribit chain (for the 24h expiry) ─────────────────────
  function findAtmIv(options, spot) {
    if (!options || !options.length) return null;

    // Filter nearest expiry only
    const expiryMap = {};
    options.forEach(o => {
      if (!expiryMap[o.expiry]) expiryMap[o.expiry] = [];
      expiryMap[o.expiry].push(o);
    });
    // Parse expiry dates to find nearest future
    const parseExp = s => {
      const m = s.match(/(\d{1,2})(\w{3})(\d{2})/);
      if (!m) return Infinity;
      const months = {JAN:0,FEB:1,MAR:2,APR:3,MAY:4,JUN:5,JUL:6,AUG:7,SEP:8,OCT:9,NOV:10,DEC:11};
      return new Date(2000 + parseInt(m[3]), months[m[2].toUpperCase()], parseInt(m[1])).getTime();
    };
    const now = Date.now();
    const expiries = Object.keys(expiryMap)
      .map(e => ({ exp: e, t: parseExp(e) }))
      .filter(x => x.t > now)
      .sort((a,b) => a.t - b.t);

    if (!expiries.length) return null;
    const nearest = expiryMap[expiries[0].exp];

    // Find strike closest to spot with both C and P
    const strikes = [...new Set(nearest.map(o => o.strike))].sort((a,b) => Math.abs(a-spot) - Math.abs(b-spot));
    for (const k of strikes) {
      const call = nearest.find(o => o.strike === k && o.type === 'C');
      const put  = nearest.find(o => o.strike === k && o.type === 'P');
      if (call && put && (call.markIv || put.markIv)) {
        const iv = (call.markIv + put.markIv) / 2;
        return {
          atmStrike:      k,
          atmIv:          iv,
          callMark:       call.mark * (call.underlying || spot),  // Deribit quotes premium in BTC
          putMark:        put.mark  * (put.underlying  || spot),
          straddleCost:   (call.mark + put.mark) * (call.underlying || spot),
          straddleCostPct: ((call.mark + put.mark) * 100),        // as % of spot (mark is in BTC)
          expiry:         expiries[0].exp,
          daysToExpiry:   (expiries[0].t - now) / 86400_000,
        };
      }
    }
    return null;
  }

  // ── RANGER (existing formula, kept for back-compat) ────────────────────
  function computeRanger(dailyCandles, fg) {
    if (!dailyCandles || dailyCandles.length < 7) return null;
    const last7 = dailyCandles.slice(-7);
    const ranges = last7.map(c => (c.h - c.l) / ((c.h + c.l) / 2) * 100);
    const atr7 = ranges.reduce((a,b) => a+b, 0) / ranges.length;

    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const dow = nowIST.getDay();
    const dowMults = [1.12, 1.08, 1.02, 0.98, 0.97, 1.05, 1.10];
    const dowMult = dowMults[dow];

    const ivHv = 1.40;    // placeholder (replaced by real IV/HV20 in retail mode below)
    const volRatio = 0.85;
    const fgFactor = 1 + (50 - (fg?.value || 50)) / 200;

    const raw  = atr7 * Math.pow(ivHv, 0.20) * Math.pow(volRatio, 0.12) * Math.pow(dowMult, 0.05) * fgFactor;
    const safe = raw * 2.2;
    return { atr7, ivHv, volRatio, dowMult, fgFactor, raw, safe };
  }

  // ── IV/HV20 REGIME CLASSIFIER (from research PDF) ──────────────────────
  // < 1.5  → GREEN  (normal sizing)
  // 1.5–1.8 → AMBER (reduced size)
  // ≥ 1.8  → RED    (no trade — realised vol likely to overshoot)
  function classifyRegime(atmIvPct, hv20Ann) {
    if (!atmIvPct || !hv20Ann) return { ratio: null, regime: 'unknown', label: '—', allowTrade: false };
    const ratio = atmIvPct / hv20Ann;
    let regime, label, allowTrade;
    if (ratio < 1.5)       { regime = 'green';  label = 'NORMAL';     allowTrade = true;  }
    else if (ratio < 1.8)  { regime = 'amber';  label = 'REDUCED';    allowTrade = true;  }
    else                   { regime = 'red';    label = 'NO-TRADE';   allowTrade = false; }
    return { ratio, regime, label, allowTrade, ivPct: atmIvPct, hv20: hv20Ann };
  }

  // ── TOUCH PROBABILITY (k × hv20_1d model from PDF) ────────────────────
  // PDF backtest: P(next-day range ≤ k × hv20_1d): 1.5×=56%, 2×=76%, 2.5×=87%, 3×=93%
  function touchProbability(distancePct, hv20_1d) {
    if (!hv20_1d || hv20_1d <= 0) return null;
    const k = distancePct / hv20_1d;
    // Piecewise linear from PDF empirical distribution
    if (k >= 3.0) return 0.07;   // 7% touch
    if (k >= 2.5) return 0.13;
    if (k >= 2.0) return 0.24;
    if (k >= 1.5) return 0.44;
    if (k >= 1.0) return 0.74;
    return 0.90;
  }

  // ── RETAIL SELLER STRATEGY ENGINE ──────────────────────────────────────
  // From PDF: 1× long ATM straddle + 60× short far-OTM wings on the LIKELY side
  //   to move AGAINST (bullish view → sell OTM puts; bearish → sell OTM calls)
  //
  // Solves for OTM strike where:
  //   (a) 60 × short_premium ≥ 1.1× × ATM_straddle_cost  (collects net credit)
  //   (b) touch probability ≤ user threshold (default 10%)
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

    // Direction from Kronos — pick which side to sell
    const direction = kronosUpside >= 55 ? 'bullish' : kronosUpside <= 45 ? 'bearish' : 'neutral';
    if (direction === 'neutral') {
      return { ok: false, reason: `Kronos ${kronosUpside}% ≈ 50/50. No directional edge — use symmetric condor instead.` };
    }

    const sellSide = direction === 'bullish' ? 'P' : 'C';   // bullish → sell puts; bearish → sell calls
    const sellSideLabel = direction === 'bullish' ? 'PUTS (below spot)' : 'CALLS (above spot)';

    // Required per-lot premium to "finance" the straddle
    const reqPremPerLot = (atmInfo.straddleCost * safetyFactor) / shortLots;

    // Filter nearest-expiry options on the chosen side
    const candidates = options
      .filter(o => o.expiry === atmInfo.expiry && o.type === sellSide)
      .map(o => ({
        ...o,
        premium: o.mark * (o.underlying || price),     // $ premium per lot
        distPct: ((o.strike - price) / price) * 100,   // +ve for calls, -ve for puts
        absDist: Math.abs(((o.strike - price) / price) * 100),
      }))
      .filter(o => {
        if (sellSide === 'P') return o.strike < price;  // OTM puts only
        return o.strike > price;                         // OTM calls only
      });

    // Only keep strikes with enough premium
    const viable = candidates.filter(o => o.premium >= reqPremPerLot);
    if (!viable.length) {
      return {
        ok: false,
        reason: `No OTM ${sellSideLabel} pay ≥ $${reqPremPerLot.toFixed(2)}/lot required to finance ${shortLots}-lot wing.`,
        direction, sellSide, atmInfo, reqPremPerLot, candidates: candidates.slice(0, 5),
      };
    }

    // Among viable, check touch probability and pick FURTHEST strike
    // that still meets premium requirement AND has touch ≤ threshold
    const scored = viable.map(o => ({
      ...o,
      touchProb: touchProbability(o.absDist, hv20.oneDay),
    }))
    .filter(o => o.touchProb !== null && o.touchProb <= touchThreshold)
    .sort((a,b) => b.absDist - a.absDist);  // furthest first

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
      ok: true,
      direction,
      sellSide,
      sellSideLabel,
      atmInfo,
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

  // ── COMPOSITE SENTIMENT SCORE ──────────────────────────────────────────
  function computeSentiment(news, kronos, fg, regime) {
    const items = news?.items || [];
    const newsPos = items.filter(i => i.sent === 'pos').length;
    const newsNeg = items.filter(i => i.sent === 'neg').length;
    const newsScore = items.length > 0
      ? Math.round((newsPos / Math.max(1, newsPos + newsNeg + items.filter(i => i.sent==='neu').length)) * 100)
      : 40;

    const kronosScore   = kronos?.upside || 50;
    const fgScore       = fg?.value || 50;
    const volAmpScore   = kronos?.volAmp || 50;
    const regimePenalty = regime?.regime === 'red' ? 20 : regime?.regime === 'amber' ? 10 : 0;

    const composite = Math.round(
      kronosScore * 0.35 +
      fgScore     * 0.25 +
      newsScore   * 0.25 +
      (100 - volAmpScore) * 0.15
      - regimePenalty
    );

    return {
      composite: Math.max(0, Math.min(100, composite)),
      newsScore, kronosScore, fgScore, volAmpScore,
      newsPos, newsNeg, total: items.length,
    };
  }

  return {
    // fetchers
    fetchPrice, fetchHourly, fetchDaily, fetchFearGreed,
    fetchOptions, fetchKronos, fetchNewsSentiment,
    // quant engines
    computeHV20, findAtmIv, computeRanger, classifyRegime,
    touchProbability, buildRetailPlan, computeSentiment,
    // util
    scoreSentiment,
  };
})();
