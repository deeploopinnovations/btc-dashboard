/**
 * rateLimit.js (v4)
 * localStorage-tracked budget enforcer. Kronos bumped to 8/day since
 * we call on every 30-min full refresh (≈48/day would breach free proxies).
 */
const RateLimit = (() => {
  const BUDGETS = {
    exa:        { monthly: 1000, daily: 33,   hourly: 1,   label: 'Exa Search'     },
    binance:    { monthly: null, daily: null, hourly: 180, label: 'Binance public' },
    deribit:    { monthly: null, daily: null, hourly: 60,  label: 'Deribit options'},
    fearGreed:  { monthly: null, daily: 2,    hourly: 1,   label: 'Fear & Greed'   },
    kronos:     { monthly: null, daily: 8,    hourly: 2,   label: 'Kronos demo'    },
  };

  const KEY = 'btc_rl_v4';

  function getStore() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
  }
  function saveStore(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch {} }

  function windows() {
    const n = new Date();
    return {
      hourKey: `${n.getUTCFullYear()}-${n.getUTCMonth()}-${n.getUTCDate()}-${n.getUTCHours()}`,
      dayKey:  `${n.getUTCFullYear()}-${n.getUTCMonth()}-${n.getUTCDate()}`,
      monKey:  `${n.getUTCFullYear()}-${n.getUTCMonth()}`,
    };
  }

  function canCall(apiKey) {
    const b = BUDGETS[apiKey];
    if (!b) return { allowed: true, reason: null };
    const s = getStore();
    const w = windows();
    const h = s[`${apiKey}_h_${w.hourKey}`] || 0;
    const d = s[`${apiKey}_d_${w.dayKey}`] || 0;
    const m = s[`${apiKey}_m_${w.monKey}`] || 0;
    if (b.hourly !== null && h >= b.hourly)
      return { allowed: false, reason: `${b.label}: hourly (${b.hourly}/hr) reached` };
    if (b.daily !== null && d >= b.daily)
      return { allowed: false, reason: `${b.label}: daily (${b.daily}/day) reached` };
    if (b.monthly !== null && m >= b.monthly)
      return { allowed: false, reason: `${b.label}: monthly (${b.monthly}) reached` };
    return { allowed: true, reason: null };
  }

  function record(apiKey) {
    const s = getStore();
    const w = windows();
    s[`${apiKey}_h_${w.hourKey}`] = (s[`${apiKey}_h_${w.hourKey}`] || 0) + 1;
    s[`${apiKey}_d_${w.dayKey}`]  = (s[`${apiKey}_d_${w.dayKey}`]  || 0) + 1;
    s[`${apiKey}_m_${w.monKey}`]  = (s[`${apiKey}_m_${w.monKey}`]  || 0) + 1;
    saveStore(s);
  }

  function getStats() {
    const s = getStore();
    const w = windows();
    const out = {};
    for (const [k, b] of Object.entries(BUDGETS)) {
      out[k] = {
        label:      b.label,
        hourly:     s[`${k}_h_${w.hourKey}`] || 0,
        daily:      s[`${k}_d_${w.dayKey}`]  || 0,
        monthly:    s[`${k}_m_${w.monKey}`]  || 0,
        hourLimit:  b.hourly,
        dayLimit:   b.daily,
        monthLimit: b.monthly,
      };
    }
    return out;
  }

  // Garbage-collect old keys (keep current month only)
  try {
    const s = getStore();
    const cur = `${new Date().getUTCFullYear()}-${new Date().getUTCMonth()}`;
    let changed = false;
    for (const k of Object.keys(s)) {
      // keep any key that references current month/day/hour
      if (!k.includes(cur)) { delete s[k]; changed = true; }
    }
    if (changed) saveStore(s);
  } catch {}

  return { canCall, record, getStats, BUDGETS };
})();
