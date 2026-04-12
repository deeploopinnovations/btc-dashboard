/**
 * charts.js
 * All Chart.js and canvas rendering (gauges, price, theta).
 */
const Charts = (() => {
  let priceChart = null;
  let thetaChart = null;

  // ── SEMI-CIRCLE GAUGE ────────────────────────────────────────────────────
  function drawGauge(canvasId, value, maxVal, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = 72, H = 36, cx = 36, cy = 36, r = 28;
    ctx.clearRect(0, 0, W, H);
    // Background arc
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 7; ctx.stroke();
    // Fill arc
    const fillAngle = Math.PI + (value / maxVal) * Math.PI;
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, fillAngle);
    ctx.strokeStyle = color; ctx.lineWidth = 7; ctx.lineCap = 'round'; ctx.stroke();
  }

  function renderGauges(sentiment) {
    const gauges = [
      { id: 'g0', val: sentiment.newsScore,   max: 100, color: sentiment.newsScore   < 45 ? '#f87171' : sentiment.newsScore < 60 ? '#fbbf24' : '#4ade80' },
      { id: 'g1', val: sentiment.kronosScore, max: 100, color: sentiment.kronosScore < 45 ? '#f87171' : sentiment.kronosScore < 60 ? '#fbbf24' : '#4ade80' },
      { id: 'g2', val: 100 - (sentiment.macroRisk * 0.33), max: 100, color: '#4ade80' }, // vol amp inverted
      { id: 'g3', val: sentiment.analystScore, max: 100, color: '#fbbf24' },
      { id: 'g4', val: 100 - sentiment.macroRisk, max: 100, color: '#f87171' },
    ];
    gauges.forEach(g => drawGauge(g.id, g.val, g.max, g.color));
  }

  // ── PRICE CHART ──────────────────────────────────────────────────────────
  function renderPriceChart(hourlyData) {
    const ctx = document.getElementById('priceC');
    if (!ctx) return;

    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const labels = hourlyData.map(c => {
      const d = new Date(c.t);
      const ist = new Date(d.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
      return ist.getHours() + ':' + String(ist.getMinutes()).padStart(2, '0');
    });

    if (priceChart) { priceChart.destroy(); priceChart = null; }

    priceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'BTC/USDT 1H',
          data: hourlyData.map(c => c.c),
          borderColor: '#818cf8',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          fill: true,
          backgroundColor: (context) => {
            const chart = context.chart;
            const { ctx: c, chartArea } = chart;
            if (!chartArea) return 'transparent';
            const gradient = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
            gradient.addColorStop(0, 'rgba(129,140,248,0.12)');
            gradient.addColorStop(1, 'rgba(129,140,248,0.01)');
            return gradient;
          },
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => '$' + c.parsed.y.toLocaleString() } } },
        scales: {
          x: { ticks: { font: { size: 8 }, maxRotation: 45, color: '#71717a' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { ticks: { font: { size: 9 }, callback: v => '$' + Math.round(v / 1000) + 'K', color: '#71717a' }, grid: { color: 'rgba(255,255,255,0.03)' } },
        },
      },
    });
  }

  // ── THETA DECAY CHART ────────────────────────────────────────────────────
  function renderThetaChart() {
    const ctx = document.getElementById('thetaC');
    if (!ctx) return;

    const nowIST = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
    const sessionStart = new Date(nowIST); sessionStart.setHours(9, 0, 0, 0);
    const sessionEnd   = new Date(nowIST); sessionEnd.setHours(17, 30, 0, 0);
    const totalMins = (sessionEnd - sessionStart) / 60000;
    const elapsedMins = Math.max(0, Math.min(totalMins, (nowIST - sessionStart) / 60000));
    const nowFrac = elapsedMins / totalMins;

    const N = 50;
    const labels = Array.from({ length: N }, (_, i) => {
      const mins = (i / N) * totalMins;
      const t = new Date(sessionStart.getTime() + mins * 60000);
      return t.getHours() + ':' + String(t.getMinutes()).padStart(2, '0');
    });

    // Exponential theta decay (slow early, fast near expiry)
    const decayData = Array.from({ length: N }, (_, i) => {
      const t = i / N;
      return parseFloat((100 * (1 - Math.exp(-5 * t * t))).toFixed(2));
    });

    const nowIdx = Math.round(nowFrac * N);

    if (thetaChart) { thetaChart.destroy(); thetaChart = null; }

    thetaChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Elapsed decay',
            data: decayData.map((v, i) => i <= nowIdx ? v : null),
            borderColor: '#fbbf24', borderWidth: 2, pointRadius: 0, tension: 0.4,
            fill: true, backgroundColor: 'rgba(251,191,36,0.08)',
          },
          {
            label: 'Future decay',
            data: decayData.map((v, i) => i >= nowIdx ? v : null),
            borderColor: 'rgba(251,191,36,0.25)', borderWidth: 1.5, pointRadius: 0, tension: 0.4,
            borderDash: [4, 4], fill: false,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: c => c.parsed.y?.toFixed(1) + '% decayed' } },
        },
        scales: {
          x: { ticks: { font: { size: 8 }, maxTicksLimit: 6, color: '#71717a' }, grid: { color: 'rgba(255,255,255,0.03)' } },
          y: { min: 0, max: 100, ticks: { font: { size: 8 }, callback: v => v + '%', color: '#71717a' }, grid: { color: 'rgba(255,255,255,0.03)' } },
        },
      },
      plugins: [{
        afterDraw(chart) {
          const { ctx: c, scales: { x, y } } = chart;
          const px = x.getPixelForValue(nowIdx);
          c.save();
          c.setLineDash([3, 3]);
          c.strokeStyle = '#818cf8';
          c.lineWidth = 1.5;
          c.beginPath(); c.moveTo(px, y.top); c.lineTo(px, y.bottom); c.stroke();
          c.restore();
        },
      }],
    });

    // Update "NOW" label
    const el = document.getElementById('nowLbl');
    if (el) el.textContent = nowIST.getHours() + ':' + String(nowIST.getMinutes()).padStart(2,'0') + ' IST ↑';
  }

  // ── RANGE VISUAL ─────────────────────────────────────────────────────────
  function renderRangeVisual(price, putStrike, callStrike) {
    const track = document.getElementById('rangeTrack');
    if (!track) return;

    const minP = putStrike  * 0.984;
    const maxP = callStrike * 1.016;
    const range = maxP - minP;
    const pct = v => ((v - minP) / range * 100).toFixed(2);

    const putPct  = pct(putStrike);
    const callPct = pct(callStrike);
    const curPct  = pct(price);

    const bearZ  = document.getElementById('bearZ');
    const safeZ  = document.getElementById('safeZ');
    const bullZ  = document.getElementById('bullZ');
    const needle = document.getElementById('needleEl');
    const pmPut  = document.getElementById('pmPut');
    const pmCall = document.getElementById('pmCall');
    const pmCur  = document.getElementById('pmCur');

    if (bearZ) { bearZ.style.left = '0'; bearZ.style.width = putPct + '%'; bearZ.style.background = 'rgba(248,113,113,0.12)'; bearZ.style.border = '1px solid rgba(248,113,113,0.25)'; bearZ.style.color = '#f87171'; bearZ.textContent = 'BEAR'; }
    if (safeZ) { safeZ.style.left = putPct + '%'; safeZ.style.width = (callPct - putPct) + '%'; safeZ.style.background = 'rgba(74,222,128,0.08)'; safeZ.style.border = '1px solid rgba(74,222,128,0.22)'; safeZ.style.color = '#4ade80'; safeZ.textContent = 'SAFE ZONE'; }
    if (bullZ) { bullZ.style.left = callPct + '%'; bullZ.style.width = (100 - callPct) + '%'; bullZ.style.background = 'rgba(248,113,113,0.12)'; bullZ.style.border = '1px solid rgba(248,113,113,0.25)'; bullZ.style.color = '#f87171'; bullZ.textContent = 'BULL'; }
    if (needle) needle.style.left = curPct + '%';
    if (pmPut)  { pmPut.style.left  = putPct + '%';  pmPut.textContent  = '$' + putStrike.toLocaleString(); }
    if (pmCall) { pmCall.style.left = callPct + '%'; pmCall.textContent = '$' + callStrike.toLocaleString(); }
    if (pmCur)  { pmCur.style.left  = curPct + '%';  pmCur.textContent  = '▼ $' + Math.round(price).toLocaleString(); }
  }

  return { renderGauges, renderPriceChart, renderThetaChart, renderRangeVisual };
})();
