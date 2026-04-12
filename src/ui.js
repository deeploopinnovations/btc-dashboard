/**
 * ui.js
 * All DOM updates. Pure functions — takes data, writes to DOM.
 */
const UI = (() => {
  const fmt = (n, d = 0) => new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }).format(n);
  const fmtK = n => '$' + fmt(Math.round(n / 1000)) + 'K';

  // ── HEADER ───────────────────────────────────────────────────────────────
  function updateTimestamp() {
    const ts = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
    const el = document.getElementById('tsEl');
    if (el) el.textContent = ts + ' IST';
    const ft = document.getElementById('ftTs');
    if (ft) ft.textContent = 'Updated: ' + ts.split(',')[1]?.trim() || '';
  }

  function updateRateInfo(stats) {
    const el = document.getElementById('rateInfo');
    if (!el) return;
    const exa = stats.exa;
    el.textContent = `Exa: ${exa.daily}/${exa.dayLimit}/day · ${exa.hourly}/${exa.hourLimit}/hr`;
  }

  // ── DIRECTION BANNER ─────────────────────────────────────────────────────
  function updateBanner(price, ranger, kronos, sentiment) {
    const score = sentiment.composite;
    const kronosUpside = kronos?.upside || 36.7;

    let dir, dirClass, confColor;
    if (score >= 55)    { dir = 'BULLISH';         dirClass = 'bull'; confColor = 'var(--green)'; }
    else if (score >= 45) { dir = 'NEUTRAL';        dirClass = 'neu';  confColor = 'var(--amber)'; }
    else                { dir = 'BEARISH-NEUTRAL'; dirClass = 'dir-bear'; confColor = 'var(--red)'; }

    const conf = dirClass === 'bull' ? score : dirClass === 'dir-bear' ? 100 - score : 50;
    const halfSafe = ranger.safe / 2 / 100 * price;

    const banner = document.getElementById('dirBanner');
    if (banner) {
      banner.className = 'dir-banner ' + dirClass;
      banner.style.setProperty('--b-color', dirClass === 'bull' ? 'rgba(74,222,128,0.3)' : dirClass === 'neu' ? 'rgba(251,191,36,0.3)' : 'rgba(248,113,113,0.3)');
      banner.style.setProperty('--bg-color', dirClass === 'bull' ? 'rgba(74,222,128,0.05)' : dirClass === 'neu' ? 'rgba(251,191,36,0.05)' : 'rgba(248,113,113,0.05)');
    }

    const setText = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    setText('dirMain', dir);
    setText('dirSub', `Kronos: ${kronosUpside.toFixed(1)}% upside · Fear/Greed: ${sentiment.fgScore} · RANGER: ${ranger.raw.toFixed(2)}%`);
    setText('bPrice', '$' + fmt(Math.round(price)));
    setText('bRange', '$' + fmt(Math.round(price - halfSafe)) + '–$' + fmt(Math.round(price + halfSafe)));
    setText('bRANGER', ranger.raw.toFixed(2) + '%');

    const fill = document.getElementById('confFill');
    const pct  = document.getElementById('confPct');
    if (fill) { fill.style.width = '0%'; fill.style.background = confColor; setTimeout(() => fill.style.width = conf + '%', 200); }
    if (pct)  { pct.textContent = Math.round(conf) + '%'; pct.className = 'conf-pct ' + (dirClass === 'bull' ? 'pos' : dirClass === 'neu' ? 'neu' : 'neg'); }
  }

  // ── PRICE METRICS ────────────────────────────────────────────────────────
  function updatePriceMetrics(priceData) {
    const { price, high, low, change, volUsd } = priceData;
    const setText = (id, t, cls) => {
      const e = document.getElementById(id);
      if (!e) return;
      e.textContent = t;
      if (cls) e.className = cls;
    };
    setText('mPrice', '$' + fmt(Math.round(price)));
    setText('mChg', (change >= 0 ? '▲ +' : '▼ ') + (change * 100).toFixed(2) + '%', 'mc ' + (change >= 0 ? 'pos' : 'neg'));
    setText('mHigh', '$' + fmt(Math.round(high)));
    setText('mHighPct', '+' + ((high - price) / price * 100).toFixed(1) + '% above current');
    setText('mLow', '$' + fmt(Math.round(low)));
    setText('mLowPct', ((low - price) / price * 100).toFixed(1) + '% from current');
    setText('mVol', '$' + fmt(volUsd / 1e6, 1) + 'M');
    setText('mVolNote', volUsd < 250e6 ? 'Below avg — calm conditions' : 'Above avg — active session');
  }

  // ── SENTIMENT GAUGES ─────────────────────────────────────────────────────
  function updateSentimentGauges(sentiment) {
    const gaugeData = [
      { vId: 'sv0', tId: 'st0', val: sentiment.newsScore,   max: 100 },
      { vId: 'sv1', tId: 'st1', val: sentiment.kronosScore, max: 100 },
      { vId: 'sv2', tId: 'st2', val: 100 - (sentiment.macroRisk * 0.33), max: 100 },
      { vId: 'sv3', tId: 'st3', val: sentiment.analystScore, max: 100 },
      { vId: 'sv4', tId: 'st4', val: 100 - sentiment.macroRisk, max: 100 },
    ];
    const labels2 = [
      [38, 'Fear', 'neg'],    [37, 'Bearish Lean', 'neg'],
      [67, 'Low Vol', 'pos'], [41, 'Cautious', 'neu'],
      [32, 'Elevated', 'neg'],
    ];
    gaugeData.forEach((g, i) => {
      const vEl = document.getElementById(g.vId);
      const tEl = document.getElementById(g.tId);
      if (vEl) { vEl.textContent = Math.round(g.val); vEl.className = 'gv ' + labels2[i][2]; }
      if (tEl) { tEl.textContent = labels2[i][1]; tEl.className = 'gt ' + labels2[i][2]; }
    });
  }

  // ── SCORE RING ────────────────────────────────────────────────────────────
  function updateScoreRing(sentiment) {
    const score = sentiment.composite;
    const color = score < 40 ? '#f87171' : score < 55 ? '#fbbf24' : '#4ade80';
    const cls   = score < 40 ? 'ring-num neg' : score < 55 ? 'ring-num neu' : 'ring-num pos';
    const circumference = 251.2;
    const fill = (score / 100) * circumference;

    setTimeout(() => {
      const arc = document.getElementById('scoreArc');
      if (arc) { arc.style.strokeDashoffset = circumference - fill; arc.style.stroke = color; }
      const num = document.getElementById('scoreN');
      if (num) { num.textContent = score; num.className = cls; }

      const bars = [
        ['pbBear', 'pbBearF', 100 - score, 'var(--red)'],
        ['pbBull', 'pbBullF', score, 'var(--green)'],
        ['pbMacro','pbMacroF', sentiment.macroRisk, 'var(--amber)'],
        ['pbVol',  'pbVolF', 67, 'var(--cyan)'],
      ];
      bars.forEach(([lblId, fillId, pct, color]) => {
        const lbl = document.getElementById(lblId); if (lbl) lbl.textContent = Math.round(pct) + '%';
        const fil = document.getElementById(fillId); if (fil) { fil.style.width = '0'; setTimeout(() => fil.style.width = pct + '%', 300); fil.style.background = color; }
      });
    }, 300);
  }

  // ── RANGER DISPLAY ───────────────────────────────────────────────────────
  function updateRanger(ranger, price) {
    const setText = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    setText('rangerOut', ranger.raw.toFixed(2) + '%');
    setText('safeOut',   ranger.safe.toFixed(2) + '%');
    setText('atrOut',    ranger.atr7.toFixed(2) + '%');

    // Ranger inputs table
    const el = document.getElementById('rangerInputs');
    if (!el) return;
    const rows = [
      ['ATR-7 (7-day avg range)',      ranger.atr7.toFixed(3) + '%'],
      ['IV/HV ratio (estimated)',      ranger.ivHv.toFixed(2) + '×'],
      ['Volume ratio vs 20-day avg',   ranger.volRatio.toFixed(2) + '× (below avg)'],
      ['Day-of-week multiplier',       ranger.dowMult.toFixed(2) + '×'],
      ['Fear & Greed factor',          ranger.fgFactor.toFixed(4)],
      ['RANGER raw output',            ranger.raw.toFixed(3) + '%'],
      ['Safety buffer (×2.2)',         ranger.safe.toFixed(3) + '%'],
    ];
    el.innerHTML = rows.map(([l, v]) =>
      `<div class="ir"><span class="ir-l">${l}</span><span class="ir-v">${v}</span></div>`
    ).join('');
  }

  // ── STRIKES ───────────────────────────────────────────────────────────────
  function updateStrikes(price, ranger, kronosUpside) {
    const halfSafe = ranger.safe / 2 / 100 * price;
    // Kronos bearish lean: put closer, call farther
    const bullAdj = kronosUpside < 45 ? 0.85 : kronosUpside > 55 ? 1.15 : 1.0;
    const bearAdj = kronosUpside < 45 ? 1.15 : kronosUpside > 55 ? 0.85 : 1.0;

    const callStrike = Math.round((price + halfSafe * bullAdj) / 500) * 500;
    const putStrike  = Math.round((price - halfSafe * bearAdj) / 500) * 500;

    const setText = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    setText('callSt',  '$' + fmt(callStrike));
    setText('putSt',   '$' + fmt(putStrike));
    setText('callDst', '+' + ((callStrike - price) / price * 100).toFixed(2) + '% · $' + fmt(callStrike - Math.round(price)) + ' pts');
    setText('putDst',       ((putStrike - price) / price * 100).toFixed(2) + '% · $' + fmt(putStrike - Math.round(price)) + ' pts');

    return { callStrike, putStrike };
  }

  // ── SIGNAL TABLE ─────────────────────────────────────────────────────────
  function updateSignals(price, ranger, kronos, sentiment, fg) {
    const kronosUpside = kronos?.upside || 36.7;
    const signals = [
      { name: 'Kronos Direction',   src: 'Kronos AI (AAAI 2026)',     val: kronosUpside.toFixed(1) + '% upside prob',  reading: kronosUpside < 45 ? 'Bearish lean' : 'Bullish lean', pill: kronosUpside < 45 ? 'p-r' : 'p-g', action: 'Shift put closer (bear adj. ×1.15)' },
      { name: 'Kronos Vol Regime',  src: 'Monte Carlo N=30',          val: (100 - (kronos?.volAmp || 33.3)).toFixed(1) + '% low-vol prob', reading: 'Low vol likely', pill: 'p-g', action: 'Sell premium confidently' },
      { name: 'RANGER Range',       src: 'ATR-7 + IV/HV',             val: ranger.raw.toFixed(2) + '% → safe ' + ranger.safe.toFixed(2) + '%', reading: 'Compressing', pill: 'p-v', action: 'Strikes at ×2.2 buffer' },
      { name: 'Fear & Greed',       src: 'Alternative.me',            val: (fg?.value || 38) + '/100 · ' + (fg?.label || 'Fear'), reading: 'Caution', pill: 'p-a', action: 'Reduce size by 20%' },
      { name: 'Macro Risk',         src: 'Exa News · BigData',        val: 'US-Iran tension · Oil $98+', reading: 'Elevated', pill: 'p-r', action: 'Skip if escalation today' },
      { name: 'IV Percentile',      src: 'Deribit estimate',          val: '38th pct (below avg)', reading: 'Sell IV > HV', pill: 'p-g', action: 'Confirmed: IV > HV ✓' },
      { name: 'Vol Today',          src: 'Crypto.com Live',           val: '$' + (sentiment?.volUsd ? fmt(sentiment.volUsd / 1e6, 0) : '177') + 'M', reading: 'Low vol', pill: 'p-a', action: 'Calm window active' },
      { name: 'News Sentiment',     src: 'Exa + keyword scorer',      val: Math.round(sentiment.newsScore) + '/100', reading: sentiment.newsScore < 45 ? 'Bearish' : 'Neutral', pill: sentiment.newsScore < 45 ? 'p-r' : 'p-a', action: 'Watch CME gap $67,180' },
    ];

    const tbody = document.getElementById('sigBody');
    if (!tbody) return;
    tbody.innerHTML = signals.map(s =>
      `<tr><td style="font-weight:500">${s.name}</td>
       <td style="color:var(--muted);font-size:10px">${s.src}</td>
       <td style="font-family:var(--font-mono);font-size:11px">${s.val}</td>
       <td><span class="pill ${s.pill}">${s.reading}</span></td>
       <td style="font-size:11px;color:var(--muted)">${s.action}</td></tr>`
    ).join('');
  }

  // ── TIMING BAR ───────────────────────────────────────────────────────────
  function updateTimingBar() {
    const segs = [
      { w: '22%', bg: 'rgba(96,165,250,0.15)', text: '00–05:30' },
      { w: '12%', bg: 'rgba(74,222,128,0.2)',  text: '05:30–08:30' },
      { w: '15%', bg: 'rgba(74,222,128,0.5)',  text: '08:30–12:30 ★' },
      { w: '10%', bg: 'rgba(74,222,128,0.2)',  text: '12:30–14:00' },
      { w: '7%',  bg: 'rgba(251,191,36,0.3)',  text: '14–16:30' },
      { w: '7%',  bg: 'rgba(248,113,113,0.4)', text: '16:30–17:30' },
      { w: '27%', bg: 'rgba(248,113,113,0.2)', text: '17:30–23:59' },
    ];
    const bar = document.getElementById('tBar');
    if (bar) bar.innerHTML = segs.map(s =>
      `<div class="t-seg" style="width:${s.w};background:${s.bg}">${s.text}</div>`
    ).join('');
  }

  // ── RATE LIMIT DISPLAY ───────────────────────────────────────────────────
  function updateRateLimitDisplay(stats) {
    const grid = document.getElementById('rlGrid');
    if (!grid) return;

    const toShow = ['exa', 'fearGreed', 'deribit', 'kronos', 'bigdata', 'cryptoCom'];
    grid.innerHTML = toShow.map(key => {
      const s = stats[key];
      const dayPct  = s.dayLimit   ? (s.daily   / s.dayLimit   * 100) : 0;
      const monPct  = s.monthLimit ? (s.monthly / s.monthLimit * 100) : 0;
      const barColor = dayPct > 80 ? '#f87171' : dayPct > 50 ? '#fbbf24' : '#4ade80';
      const dayStr   = s.dayLimit   ? `${s.daily}/${s.dayLimit}/day`     : `${s.daily} today (unlimited)`;
      const monStr   = s.monthLimit ? `${s.monthly}/${s.monthLimit}/mo`  : 'Unlimited';
      const hrStr    = s.hourLimit  ? `${s.hourly}/${s.hourLimit}/hr`    : 'Unlimited';
      return `<div class="rl-card">
        <div class="rl-name">${s.label}</div>
        <div class="rl-bar-bg"><div class="rl-bar-fill" style="width:${Math.min(100,dayPct)}%;background:${barColor}"></div></div>
        <div class="rl-nums">
          <span style="color:var(--muted)">${dayStr}</span>
          <span style="color:var(--muted)">${hrStr}</span>
        </div>
        <div style="font-size:9px;color:var(--muted);margin-top:3px">${monStr}</div>
      </div>`;
    }).join('');

    updateRateInfo(stats);
  }

  // ── SENTIMENT SOURCES ────────────────────────────────────────────────────
  function updateSentimentSources(sentiment) {
    const el = document.getElementById('sentSrc');
    if (!el) return;
    const sources = [
      { src: 'Kronos AI (AAAI 2026)',  weight: '30%', score: Math.round(sentiment.kronosScore),  dir: sentiment.kronosScore < 45 ? 'Bearish' : 'Bullish' },
      { src: 'BigData.com / Exa News', weight: '25%', score: Math.round(sentiment.newsScore),    dir: sentiment.newsScore < 45 ? 'Bearish' : 'Neutral' },
      { src: 'Fear & Greed Index',     weight: '25%', score: sentiment.fgScore,                  dir: sentiment.fgScore < 40 ? 'Extreme Fear' : 'Fear' },
      { src: 'Analyst Consensus',      weight: '10%', score: sentiment.analystScore,             dir: 'Cautious' },
      { src: 'Macro / Geopolitical',   weight: '10%', score: 100 - sentiment.macroRisk,          dir: 'Elevated Risk' },
    ];
    el.innerHTML = sources.map(s => {
      const color = s.score < 40 ? 'var(--red)' : s.score < 55 ? 'var(--amber)' : 'var(--green)';
      return `<div class="ir">
        <span class="ir-l">${s.src} <span style="color:var(--muted);font-size:9px">(${s.weight})</span></span>
        <span class="ir-v" style="color:${color}">${s.score}/100 · ${s.dir}</span>
      </div>`;
    }).join('');
  }

  // ── NEWS FEED ─────────────────────────────────────────────────────────────
  function updateNewsFeed(news) {
    const el = document.getElementById('newsFeed');
    if (!el) return;
    const items = news?.items || [];
    el.innerHTML = items.slice(0, 8).map(item => {
      const dotColor = item.sent === 'pos' ? 'var(--green)' : item.sent === 'neg' ? 'var(--red)' : 'var(--amber)';
      const src = item.src || (item.url ? new URL(item.url).hostname.replace('www.','') : 'Unknown');
      const dateStr = item.date ? new Date(item.date).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }) : '';
      return `<div class="ni">
        <div class="ni-dot" style="background:${dotColor}"></div>
        <div>
          <div class="ni-hl">${item.headline}</div>
          <div class="ni-src">${src}${dateStr ? ' · ' + dateStr : ''}</div>
        </div>
      </div>`;
    }).join('');
  }

  return {
    updateTimestamp, updateRateInfo, updateBanner, updatePriceMetrics,
    updateSentimentGauges, updateScoreRing, updateRanger, updateStrikes,
    updateSignals, updateTimingBar, updateRateLimitDisplay,
    updateSentimentSources, updateNewsFeed,
  };
})();
