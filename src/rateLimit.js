/**
 * rateLimit.js (v3)
 * Enforces monthly/daily/hourly API budgets via localStorage.
 */
const RateLimit = (() => {
  const BUDGETS = {
    exa:        { monthly: 1000, daily: 33,   hourly: 1,  label: 'Exa Search'     },
    binance:    { monthly: null, daily: null, hourly: 120, label: 'Binance (public)' },
    deribit:    { monthly: null, daily: null, hourly: 60,  label: 'Deribit Options' },
    fearGreed:  { monthly: null, daily: 2,    hourly: 1,  label: 'Fear & Greed'   },
    kronos:     { monthly: null, daily: 8,    hourly: 2,  label: 'Kronos Demo'    },
    bigdata:    { monthly: null, daily: 2,    hourly: 1,  label: 'BigData.com'    },
    cryptoCom:  { monthly: null, daily: null, hourly: 60,  label: 'Crypto.com (deprecated)' },
  };

  const KEY = 'btc_dash_rate_v3';

  function getStore() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
  }
  function saveStore(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch {} }

  function getWindows() {
    const now = new Date();
    return {
      hourKey: `${now.getUTCFullYear()}-${now.getUTCMonth()}-${now.getUTCDate()}-${now.getUTCHours()}`,
      dayKey:  `${now.getUTCFullYear()}-${now.getUTCMonth()}-${now.getUTCDate()}`,
      monKey:  `${now.getUTCFullYear()}-${now.getUTCMonth()}`,
    };
  }

  function canCall(apiKey) {
    const b = BUDGETS[apiKey];
    if (!b) return { allowed: true, reason: null };
    const store = getStore();
    const { hourKey, dayKey, monKey } = getWindows();
    const hourlyUsed  = store[`${apiKey}_h_${hourKey}`]  || 0;
    const dailyUsed   = store[`${apiKey}_d_${dayKey}`]   || 0;
    const monthlyUsed = store[`${apiKey}_m_${monKey}`]   || 0;
    if (b.hourly  !== null && hourlyUsed  >= b.hourly)
      return { allowed: false, reason: `${b.label}: hourly limit (${b.hourly}/hr) reached.` };
    if (b.daily   !== null && dailyUsed   >= b.daily)
      return { allowed: false, reason: `${b.label}: daily limit (${b.daily}/day) reached.` };
    if (b.monthly !== null && monthlyUsed >= b.monthly)
      return { allowed: false, reason: `${b.label}: monthly limit (${b.monthly}/mo) reached.` };
    return { allowed: true, reason: null };
  }

  function record(apiKey) {
    const store = getStore();
    const { hourKey, dayKey, monKey } = getWindows();
    store[`${apiKey}_h_${hourKey}`]  = (store[`${apiKey}_h_${hourKey}`]  || 0) + 1;
    store[`${apiKey}_d_${dayKey}`]   = (store[`${apiKey}_d_${dayKey}`]   || 0) + 1;
    store[`${apiKey}_m_${monKey}`]   = (store[`${apiKey}_m_${monKey}`]   || 0) + 1;
    saveStore(store);
  }

  function getStats() {
    const store = getStore();
    const { hourKey, dayKey, monKey } = getWindows();
    const stats = {};
    for (const [key, b] of Object.entries(BUDGETS)) {
      stats[key] = {
        label:      b.label,
        hourly:     store[`${key}_h_${hourKey}`]  || 0,
        daily:      store[`${key}_d_${dayKey}`]   || 0,
        monthly:    store[`${key}_m_${monKey}`]   || 0,
        hourLimit:  b.hourly,
        dayLimit:   b.daily,
        monthLimit: b.monthly,
      };
    }
    return stats;
  }

  // Cleanup old keys
  try {
    const store = getStore();
    const cutoff = Date.now() - 32 * 86400000;
    const toDel = [];
    for (const k of Object.keys(store)) {
      const segs = k.split('_');
      if (segs.length < 3) continue;
      // best-effort: drop any key that doesn't match current month
      const monK = `${new Date().getUTCFullYear()}-${new Date().getUTCMonth()}`;
      if (!k.includes(monK)) {
        // keep last 2 months
        toDel.push(k);
      }
    }
    // Keep things small but don't blindly delete everything
    if (Object.keys(store).length > 200) {
      toDel.slice(0, 50).forEach(k => delete store[k]);
      saveStore(store);
    }
  } catch {}

  return { canCall, record, getStats, BUDGETS };
})();
