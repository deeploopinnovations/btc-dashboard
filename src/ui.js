/**
 * ui.js  (v3)
 * All DOM updates. Adds retail seller panel + regime-aware visuals.
 */
const UI = (() => {
  const fmt  = (n, d = 0) => new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }).format(n);
  const fmtK = n => '$' + fmt(Math.round(n / 1000)) + 'K';
  const pct  = (n, d = 2) => (n >= 0 ? '+' : '') + n.toFixed(d) + '%';

  // ── HEADER ───────────────────────────────────────────────────────────────
  function updateTimestamp() {
    const ts = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour12: false });
    const el = document.getElementById('tsEl');
    if (el) el.textContent = ts + ' IST';
    const ft = document.getElementById('ftTs');
    if (ft) ft.textContent = 'Updated: ' + (ts.split(',')[1]?.trim() || '');
  }

  function updateRateInfo(stats) {
    const el = document.getElementById('rateInfo');
    if (!el) return;
    const exa = stats.exa;
    el.textContent = `Exa: ${exa.daily}/${exa.dayLimit}/day · ${exa.hourly}/${exa.hourLimit}/hr`;
  }

  // ── DIRECTION BANNER ─────────────────────────────────────────────────────
  function updateBanner(price, ranger, kronos, sentiment, regime) {
    const score = sentiment.composite;
    const ku = kronos?.upside ?? 50;

    let dir, dirClass, confColor;
    if      (score >= 58) { dir = 'BULLISH';          dirClass = 'bull';     confColor = 'var(--green)'; }
    else if (score >= 48) { dir = 'NEUTRAL';          dirClass = 'neu';      confColor = 'var(--amber)'; }
    else                  { dir = 'BEARISH-NEUTRAL';  dirClass = 'dir-bear'; confColor = 'var(--red)';   }

    const conf = dirClass === 'bull' ? score : dirClass === 'dir-bear' ? 100 - score : 50;
    const halfSafe = ranger.safe / 2 / 100 * price;

    const banner = document.getElementById('dirBanner');
    if (banner) {
      banner.className = 'dir-banner ' + dirClass;
      banner.style.setProperty('--b-color',  dirClass === 'bull' ? 'rgba(74,222,128,0.3)' : dirClass === 'neu' ? 'rgba(251,191,36,0.3)' : 'rgba(248,113,113,0.3)');
      banner.style.setProperty('--bg-color', dirClass === 'bull' ? 'rgba(74,222,128,0.05)' : dirClass === 'neu' ? 'rgba(251,191,36,0.05)' : 'rgba(248,113,113,0.05)');
    }

    const set = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    set('dirMain', dir);

    const regimeStr = regime ? ` · IV/HV20 ${regime.ratio?.toFixed(2)} (${regime.label})` : '';
    set('dirSub',   `Kronos ${ku.toFixed(1)}% upside · F&G ${sentiment.fgScore} · RANGER ${ranger.raw.toFixed(2)}%${regimeStr}`);
    set('bPrice',   '$' + fmt(Math.round(price)));
    set('bRange',   '$' + fmt(Math.round(price - halfSafe)) + '–$' + fmt(Math.round(price + halfSafe)));
    set('bRANGER',  ranger.raw.toFixed(2) + '%');

    const fill = document.getElementById('confFill');
    const pctEl = document.getElementById('confPct');
    if (fill)  { fill.style.width = '0%'; fill.style.background = confColor; setTimeout(() => fill.style.width = conf + '%', 200); }
    if (pctEl) { pctEl.textContent = Math.round(conf) + '%'; pctEl.className = 'conf-pct ' + (dirClass === 'bull' ? 'pos' : dirClass === 'neu' ? 'neu' : 'neg'); }
  }

  // ── PRICE METRICS ────────────────────────────────────────────────────────
  function updatePriceMetrics(priceData) {
    if (!priceData) return;
    const { price, high, low, change, volUsd } = priceData;
    const set = (id, t, cls) => { const e = document.getElementById(id); if (!e) return; e.textContent = t; if (cls) e.className = cls; };
    set('mPrice',    '$' + fmt(Math.round(price)));
    set('mChg',      (change >= 0 ? '▲ +' : '▼ ') + (change * 100).toFixed(2) + '%', 'mc ' + (change >= 0 ? 'pos' : 'neg'));
    set('mHigh',     '$' + fmt(Math.round(high)));
    set('mHighPct',  '+' + ((high - price) / price * 100).toFixed(1) + '% above current');
    set('mLow',      '$' + fmt(Math.round(low)));
    set('mLowPct',   ((low - price) / price * 100).toFixed(1) + '% from current');
    set('mVol',      '$' + fmt(volUsd / 1e6, 1) + 'M');
    set('mVolNote',  volUsd < 250e9 ? 'Below avg — calm conditions' : 'Above avg — active session');
  }

  // ── SENTIMENT GAUGES ─────────────────────────────────────────────────────
  function updateSentimentGauges(sentiment, kronos, fg, regime) {
    const newsScore   = sentiment?.newsScore   ?? 0;
    const kronosScore = sentiment?.kronosScore ?? 50;
    const volAmp      = kronos?.volAmp         ?? 50;
    const fgScore     = fg?.value              ?? 50;
    const regimeInv   = regime?.ratio ? Math.max(0, Math.min(100, 100 - (regime.ratio - 1) * 100)) : 50;

    const gauges = [
      { vId: 'sv0', tId: 'st0', val: newsScore,        label: newsScore < 40 ? 'Bearish' : newsScore < 60 ? 'Mixed' : 'Bullish',   cls: newsScore < 40 ? 'neg' : newsScore < 60 ? 'neu' : 'pos' },
      { vId: 'sv1', tId: 'st1', val: Math.round(kronosScore), label: kronosScore < 40 ? 'Bearish Lean' : kronosScore < 60 ? 'Neutral' : 'Bullish Lean', cls: kronosScore < 40 ? 'neg' : kronosScore < 60 ? 'neu' : 'pos' },
      { vId: 'sv2', tId: 'st2', val: Math.round(volAmp),      label: volAmp > 60 ? 'HIGH VOL' : volAmp > 40 ? 'Normal' : 'Low Vol',               cls: volAmp > 60 ? 'neg' : volAmp > 40 ? 'neu' : 'pos' },
      { vId: 'sv3', tId: 'st3', val: fgScore,                 label: fg?.label || '—',                                                            cls: fgScore < 40 ? 'neg' : fgScore < 60 ? 'neu' : 'pos' },
      { vId: 'sv4', tId: 'st4', val: Math.round(regimeInv),   label: regime?.label || '—',                                                        cls: regime?.regime === 'green' ? 'pos' : regime?.regime === 'amber' ? 'neu' : 'neg' },
    ];
    gauges.forEach(g => {
      const v = document.getElementById(g.vId); if (v) { v.textContent = g.val; v.className = 'gv ' + g.cls; }
      const t = document.getElementById(g.tId); if (t) { t.textContent = g.label; t.className = 'gt ' + g.cls; }
    });
  }

  // ── SCORE RING ───────────────────────────────────────────────────────────
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
        ['pbBear',  'pbBearF',  100 - score,        'var(--red)'],
        ['pbBull',  'pbBullF',  score,              'var(--green)'],
        ['pbMacro', 'pbMacroF', sentiment.volAmpScore || 50, 'var(--amber)'],
        ['pbVol',   'pbVolF',   sentiment.newsScore, 'var(--cyan)'],
      ];
      bars.forEach(([lblId, fillId, p, c]) => {
        const lbl = document.getElementById(lblId); if (lbl) lbl.textContent = Math.round(p) + '%';
        const fil = document.getElementById(fillId); if (fil) { fil.style.width = '0'; setTimeout(() => fil.style.width = p + '%', 300); fil.style.background = c; }
      });
    }, 200);
  }

  // ── RANGER DISPLAY ───────────────────────────────────────────────────────
  function updateRanger(ranger, price) {
    const set = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    set('rangerOut', ranger.raw.toFixed(2) + '%');
    set('safeOut',   ranger.safe.toFixed(2) + '%');
    set('atrOut',    ranger.atr7.toFixed(2) + '%');

    const el = document.getElementById('rangerInputs');
    if (!el) return;
    const rows = [
      ['ATR-7 (7-day avg range)',      ranger.atr7.toFixed(3) + '%'],
      ['IV/HV placeholder',            ranger.ivHv.toFixed(2) + '× (see IV/HV20 below)'],
      ['Volume ratio',                 ranger.volRatio.toFixed(2) + '×'],
      ['Day-of-week multiplier',       ranger.dowMult.toFixed(2) + '×'],
      ['Fear & Greed factor',          ranger.fgFactor.toFixed(4)],
      ['RANGER raw',                   ranger.raw.toFixed(3) + '%'],
      ['Safe buffer (×2.2)',           ranger.safe.toFixed(3) + '%'],
    ];
    el.innerHTML = rows.map(([l, v]) =>
      `<div class="ir"><span class="ir-l">${l}</span><span class="ir-v">${v}</span></div>`
    ).join('');
  }

  // ── STRIKES (classic panel) ──────────────────────────────────────────────
  function updateStrikes(price, ranger, kronosUpside) {
    const halfSafe = ranger.safe / 2 / 100 * price;
    const bullAdj  = kronosUpside < 45 ? 0.85 : kronosUpside > 55 ? 1.15 : 1.0;
    const bearAdj  = kronosUpside < 45 ? 1.15 : kronosUpside > 55 ? 0.85 : 1.0;
    const callStrike = Math.round((price + halfSafe * bullAdj) / 500) * 500;
    const putStrike  = Math.round((price - halfSafe * bearAdj) / 500) * 500;

    const set = (id, t) => { const e = document.getElementById(id); if (e) e.textContent = t; };
    set('callSt',  '$' + fmt(callStrike));
    set('putSt',   '$' + fmt(putStrike));
    set('callDst', '+' + ((callStrike - price) / price * 100).toFixed(2) + '% · $' + fmt(callStrike - Math.round(price)) + ' pts');
    set('putDst',       ((putStrike - price) / price * 100).toFixed(2) + '% · $' + fmt(putStrike - Math.round(price)) + ' pts');
    return { callStrike, putStrike };
  }

  // ── SIGNAL TABLE ─────────────────────────────────────────────────────────
  function updateSignals(price, ranger, kronos, sentiment, fg, regime, hv20) {
    const ku = kronos?.upside || 50;
    const va = kronos?.volAmp || 50;
    const signals = [
      { name: 'Kronos Direction',  src: 'Kronos AI (AAAI 2026)',  val: ku.toFixed(1) + '% upside',
        reading: ku < 45 ? 'Bearish' : ku > 55 ? 'Bullish' : 'Neutral', pill: ku < 45 ? 'p-r' : ku > 55 ? 'p-g' : 'p-a',
        action: ku < 45 ? 'Sell OTM calls' : ku > 55 ? 'Sell OTM puts' : 'No directional trade' },
      { name: 'Kronos Vol Regime', src: 'Monte Carlo N=30',       val: va.toFixed(1) + '% vol amp prob',
        reading: va > 60 ? 'HIGH VOL AHEAD' : va > 40 ? 'Normal' : 'Low Vol', pill: va > 60 ? 'p-r' : va > 40 ? 'p-a' : 'p-g',
        action: va > 60 ? 'Widen strikes or skip' : 'Standard sizing OK' },
      { name: 'HV20 (20-day realised)', src: 'Binance daily closes', val: hv20 ? hv20.annualised.toFixed(1) + '% ann · ' + hv20.oneDay.toFixed(2) + '%/day' : '—',
        reading: hv20 ? (hv20.annualised > 70 ? 'Elevated' : 'Normal') : '—', pill: hv20 && hv20.annualised > 70 ? 'p-a' : 'p-v',
        action: 'Use for IV/HV filter' },
      { name: 'IV/HV20 Regime',    src: 'Deribit ATM / HV20',     val: regime?.ratio ? regime.ratio.toFixed(2) + '×' : '—',
        reading: regime?.label || '—', pill: regime?.regime === 'green' ? 'p-g' : regime?.regime === 'amber' ? 'p-a' : 'p-r',
        action: regime?.allowTrade ? 'Trade allowed' : 'NO-TRADE today' },
      { name: 'RANGER Range',      src: 'ATR-7 + DOW + F&G',      val: ranger.raw.toFixed(2) + '% raw',
        reading: 'Safe ±' + (ranger.safe/2).toFixed(2) + '%', pill: 'p-v', action: 'Strikes at ×2.2 buffer' },
      { name: 'Fear & Greed',      src: 'alternative.me',          val: (fg?.value || 50) + ' · ' + (fg?.label || '—'),
        reading: (fg?.value || 50) < 30 ? 'Extreme' : 'OK', pill: (fg?.value || 50) < 30 ? 'p-a' : 'p-g', action: 'Weight in composite' },
      { name: 'News Sentiment',    src: sentiment?.total ? `${sentiment.total} items` : 'cached', val: sentiment.newsScore + '/100',
        reading: sentiment.newsScore < 40 ? 'Bearish' : sentiment.newsScore < 60 ? 'Mixed' : 'Bullish', pill: sentiment.newsScore < 40 ? 'p-r' : sentiment.newsScore < 60 ? 'p-a' : 'p-g',
        action: 'Monitor flows' },
      { name: 'Composite Score',   src: 'Weighted (Kronos+F&G+News+Vol)', val: sentiment.composite + '/100',
        reading: sentiment.composite < 40 ? 'Bearish' : sentiment.composite < 55 ? 'Neutral' : 'Bullish',
        pill: sentiment.composite < 40 ? 'p-r' : sentiment.composite < 55 ? 'p-a' : 'p-g', action: 'Final conviction' },
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
    const toShow = ['exa', 'fearGreed', 'deribit', 'kronos', 'binance'];
    grid.innerHTML = toShow.map(key => {
      const s = stats[key]; if (!s) return '';
      const dayPct = s.dayLimit ? (s.daily / s.dayLimit * 100) : 0;
      const barColor = dayPct > 80 ? '#f87171' : dayPct > 50 ? '#fbbf24' : '#4ade80';
      const dayStr = s.dayLimit ? `${s.daily}/${s.dayLimit}/day`  : `${s.daily} today`;
      const monStr = s.monthLimit ? `${s.monthly}/${s.monthLimit}/mo` : 'Unlimited';
      const hrStr  = s.hourLimit  ? `${s.hourly}/${s.hourLimit}/hr` : 'Unlimited';
      return `<div class="rl-card">
        <div class="rl-name">${s.label}</div>
        <div class="rl-bar-bg"><div class="rl-bar-fill" style="width:${Math.min(100,dayPct)}%;background:${barColor}"></div></div>
        <div class="rl-nums"><span style="color:var(--muted)">${dayStr}</span><span style="color:var(--muted)">${hrStr}</span></div>
        <div style="font-size:9px;color:var(--muted);margin-top:3px">${monStr}</div>
      </div>`;
    }).join('');
    updateRateInfo(stats);
  }

  // ── SENTIMENT SOURCES ────────────────────────────────────────────────────
  function updateSentimentSources(sentiment, kronos, fg) {
    const el = document.getElementById('sentSrc');
    if (!el) return;
    const sources = [
      { src: 'Kronos AI',         weight: '35%', score: Math.round(sentiment.kronosScore), dir: sentiment.kronosScore < 45 ? 'Bearish' : sentiment.kronosScore > 55 ? 'Bullish' : 'Neutral' },
      { src: 'News (Exa/CryptoPanic)', weight: '25%', score: sentiment.newsScore, dir: sentiment.newsScore < 40 ? 'Bearish' : sentiment.newsScore > 60 ? 'Bullish' : 'Mixed' },
      { src: 'Fear & Greed',      weight: '25%', score: sentiment.fgScore, dir: fg?.label || '—' },
      { src: 'Kronos Vol Amp',    weight: '15%', score: Math.round(100 - (kronos?.volAmp || 50)), dir: (kronos?.volAmp || 50) > 60 ? 'HIGH VOL' : 'Calm' },
    ];
    el.innerHTML = sources.map(s => {
      const color = s.score < 40 ? 'var(--red)' : s.score < 55 ? 'var(--amber)' : 'var(--green)';
      return `<div class="ir">
        <span class="ir-l">${s.src} <span style="color:var(--muted);font-size:9px">(${s.weight})</span></span>
        <span class="ir-v" style="color:${color}">${s.score}/100 · ${s.dir}</span>
      </div>`;
    }).join('');
  }

  // ── NEWS FEED ────────────────────────────────────────────────────────────
  function updateNewsFeed(news) {
    const el = document.getElementById('newsFeed');
    if (!el) return;
    const items = news?.items || [];
    const header = `<div style="font-size:9px;color:var(--muted);margin-bottom:6px">Source: ${news?.source || '—'} · ${items.length} items</div>`;
    if (!items.length) { el.innerHTML = header + '<div style="font-size:11px;color:var(--muted)">No news loaded.</div>'; return; }
    el.innerHTML = header + items.slice(0, 8).map(item => {
      const dot = item.sent === 'pos' ? 'var(--green)' : item.sent === 'neg' ? 'var(--red)' : 'var(--amber)';
      const src = item.src || 'Unknown';
      return `<div class="ni">
        <div class="ni-dot" style="background:${dot}"></div>
        <div><div class="ni-hl">${escapeHtml(item.headline)}</div><div class="ni-src">${src}</div></div>
      </div>`;
    }).join('');
  }

  // ── RETAIL SELLER PLAN PANEL  ────────────────────────────────────────────
  function updateRetailPlan(plan, price, hv20, regime, atmInfo) {
    const el = document.getElementById('retailPanel');
    if (!el) return;

    // Top metrics row
    const topRow = `
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px">
        <div class="rp-metric">
          <div class="rp-l">HV20 (annualised)</div>
          <div class="rp-v">${hv20 ? hv20.annualised.toFixed(1) + '%' : '—'}</div>
          <div class="rp-s">${hv20 ? '1-day: ' + hv20.oneDay.toFixed(2) + '%' : ''}</div>
        </div>
        <div class="rp-metric">
          <div class="rp-l">ATM IV (Deribit)</div>
          <div class="rp-v">${atmInfo ? atmInfo.atmIv.toFixed(1) + '%' : '—'}</div>
          <div class="rp-s">${atmInfo ? '$' + fmt(atmInfo.atmStrike) + ' · ' + atmInfo.expiry : ''}</div>
        </div>
        <div class="rp-metric" style="border:1px solid ${regime?.regime === 'green' ? 'rgba(74,222,128,0.35)' : regime?.regime === 'amber' ? 'rgba(251,191,36,0.35)' : 'rgba(248,113,113,0.35)'}">
          <div class="rp-l">IV/HV20 Regime</div>
          <div class="rp-v" style="color:${regime?.regime === 'green' ? 'var(--green)' : regime?.regime === 'amber' ? 'var(--amber)' : 'var(--red)'}">${regime?.ratio ? regime.ratio.toFixed(2) + '×' : '—'}</div>
          <div class="rp-s">${regime?.label || '—'}</div>
        </div>
        <div class="rp-metric">
          <div class="rp-l">ATM Straddle Cost</div>
          <div class="rp-v">${atmInfo ? '$' + fmt(atmInfo.straddleCost, 0) : '—'}</div>
          <div class="rp-s">${atmInfo ? 'per 1 lot · ' + (atmInfo.straddleCost/price*100).toFixed(2) + '% of spot' : ''}</div>
        </div>
      </div>`;

    // Sliders
    const controls = `
      <div style="display:flex;gap:16px;flex-wrap:wrap;padding:10px 0 14px;border-top:1px solid var(--border)">
        <label style="font-size:10px;color:var(--muted);display:flex;flex-direction:column;gap:4px;min-width:140px">
          Short lots (hedge size)
          <input type="range" id="rpLots" min="20" max="100" step="5" value="60" oninput="document.getElementById('rpLotsOut').textContent=this.value; onRetailSliderChange()" style="width:100%">
          <span style="color:var(--accent);font-family:var(--font-mono);font-size:12px"><span id="rpLotsOut">60</span> lots</span>
        </label>
        <label style="font-size:10px;color:var(--muted);display:flex;flex-direction:column;gap:4px;min-width:140px">
          Max touch probability
          <input type="range" id="rpTouch" min="0.03" max="0.20" step="0.01" value="0.10" oninput="document.getElementById('rpTouchOut').textContent=(parseFloat(this.value)*100).toFixed(0)+'%'; onRetailSliderChange()" style="width:100%">
          <span style="color:var(--accent);font-family:var(--font-mono);font-size:12px"><span id="rpTouchOut">10%</span></span>
        </label>
        <div style="font-size:10px;color:var(--muted);align-self:center">
          Direction input: <b style="color:${plan?.direction === 'bullish' ? 'var(--green)' : plan?.direction === 'bearish' ? 'var(--red)' : 'var(--amber)'}">${plan?.direction?.toUpperCase() || 'NEUTRAL'}</b>
          <br>(from Kronos upside probability)
        </div>
      </div>`;

    // Plan output
    let planBody;
    if (!plan) {
      planBody = `<div class="rp-warn">Waiting for inputs (price / options / HV20)…</div>`;
    } else if (!plan.ok) {
      planBody = `<div class="rp-warn rp-red">
        <div style="font-size:14px;font-weight:700;margin-bottom:6px">⛔ NO-TRADE</div>
        <div style="font-size:12px">${plan.reason}</div>
      </div>`;
      if (plan.candidates?.length) {
        planBody += `<div style="font-size:10px;color:var(--muted);margin-top:10px">Premium-viable strikes with their touch probabilities:</div>`;
        planBody += `<table class="sig-tbl" style="margin-top:6px"><thead><tr><th>Strike</th><th>Distance</th><th>Premium/lot</th><th>Touch prob</th></tr></thead><tbody>` +
          plan.candidates.map(c => `<tr><td>$${fmt(c.strike)}</td><td>${c.absDist?.toFixed(2)}%</td><td>$${fmt(c.premium, 2)}</td><td>${c.touchProb ? (c.touchProb*100).toFixed(0)+'%' : '—'}</td></tr>`).join('')
          + `</tbody></table>`;
      }
    } else {
      // Valid plan!
      const netCreditClass = plan.netCredit > 0 ? 'pos' : 'neg';
      planBody = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
          <div class="rp-card rp-green">
            <div class="rp-l">LEG 1: Long Straddle</div>
            <div style="font-size:14px;font-weight:700;margin:3px 0">Buy 1× ATM $${fmt(plan.atmInfo.atmStrike)} C + P</div>
            <div class="rp-s">Cost: <b>$${fmt(plan.straddleCost, 0)}</b> · Expiry: ${plan.atmInfo.expiry}</div>
          </div>
          <div class="rp-card rp-green">
            <div class="rp-l">LEG 2: Short OTM Wing (${plan.sellSideLabel})</div>
            <div style="font-size:14px;font-weight:700;margin:3px 0">Sell ${plan.shortLots}× $${fmt(plan.shortStrike)} ${plan.sellSide}</div>
            <div class="rp-s">Premium: <b>$${fmt(plan.shortPremiumPerLot, 2)}</b>/lot · Total: <b>$${fmt(plan.totalShortPremium, 0)}</b></div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">
          <div class="rp-metric"><div class="rp-l">Distance from Spot</div><div class="rp-v">${plan.shortDistancePct.toFixed(2)}%</div><div class="rp-s">${plan.shortDistancePct > 15 ? 'Very safe' : plan.shortDistancePct > 10 ? 'Safe' : 'Moderate'}</div></div>
          <div class="rp-metric"><div class="rp-l">Touch Probability</div><div class="rp-v">${(plan.touchProb*100).toFixed(0)}%</div><div class="rp-s">HV20 × ${(plan.shortDistancePct/plan.atmInfo.straddleCost*100).toFixed(1)}</div></div>
          <div class="rp-metric"><div class="rp-l">Net Credit</div><div class="rp-v ${netCreditClass}">${plan.netCredit >= 0 ? '+' : ''}$${fmt(plan.netCredit, 0)}</div><div class="rp-s">after financing straddle</div></div>
          <div class="rp-metric"><div class="rp-l">Required/lot</div><div class="rp-v">$${plan.reqPremPerLot.toFixed(2)}</div><div class="rp-s">to break even × ${((plan.shortPremiumPerLot/plan.reqPremPerLot)*100).toFixed(0)}% coverage</div></div>
        </div>
        <div style="font-size:11px;color:var(--amber);padding:10px;background:rgba(251,191,36,0.06);border:1px solid rgba(251,191,36,0.25);border-radius:6px">
          ⚠️ <b>Risk note:</b> Shorting ${plan.shortLots}× uncapped is extreme-gamma. If BTC touches $${fmt(plan.shortStrike)} (${plan.shortDistancePct.toFixed(1)}% move), your short wing blows up. Use Delta's <b>strategy builder</b> to add cheap long protection further OTM and cap max loss to &lt;10% of equity.
        </div>`;

      if (plan.alternatives?.length) {
        planBody += `<div style="font-size:10px;color:var(--muted);margin-top:10px">Alternative strikes (also valid, closer to spot):</div>`;
        planBody += `<table class="sig-tbl" style="margin-top:4px"><thead><tr><th>Strike</th><th>Distance</th><th>Premium</th><th>Touch</th><th>Net credit</th></tr></thead><tbody>` +
          plan.alternatives.map(a => {
            const nc = (a.premium * plan.shortLots) - plan.straddleCost;
            return `<tr><td>$${fmt(a.strike)}</td><td>${a.absDist.toFixed(2)}%</td><td>$${fmt(a.premium,2)}</td><td>${(a.touchProb*100).toFixed(0)}%</td><td style="color:${nc>=0?'var(--green)':'var(--red)'}">$${fmt(nc, 0)}</td></tr>`;
          }).join('')
          + `</tbody></table>`;
      }
    }

    el.innerHTML = topRow + controls + planBody;
  }

  // ── UTIL ─────────────────────────────────────────────────────────────────
  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  return {
    updateTimestamp, updateRateInfo, updateBanner, updatePriceMetrics,
    updateSentimentGauges, updateScoreRing, updateRanger, updateStrikes,
    updateSignals, updateTimingBar, updateRateLimitDisplay,
    updateSentimentSources, updateNewsFeed, updateRetailPlan,
  };
})();
