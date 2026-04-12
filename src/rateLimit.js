/**
 * rateLimit.js
 * Enforces monthly/daily/hourly API budgets via localStorage.
 * Every API call MUST go through this gatekeeper.
 */
const RateLimit = (() => {
  // ── BUDGETS ────────────────────────────────────────────────────────────────
  const BUDGETS = {
    exa:        { monthly: 1000, daily: 33,  hourly: 1,  label: 'Exa Search'       },
    cryptoCom:  { monthly: null, daily: null, hourly: 60, label: 'Crypto.com'       },
    deribit:    { monthly: null, daily: null, hourly: 60, label: 'Deribit Options'  },
    fearGreed:  { monthly: null, daily: 1,   hourly: 1,  label: 'Fear & Greed'     },
    bigdata:    { monthly: null, daily: 2,   hourly: 1,  label: 'BigData.com'      },
    kronos:     { monthly: null, daily: 1,   hourly: 1,  label: 'Kronos AI'        },
    binance:    { monthly: null, daily: null, hourly: 60, label: 'Binance OHLCV'   },
  };

  const KEY = 'btc_dash_rate_v2';

  function getStore() {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
    catch { return {}; }
  }

  function saveStore(s) {
    try { localStorage.setItem(KEY, JSON.stringify(s)); } catch {}
  }

  function getWindows() {
    const now = new Date();
    return {
      hourKey: `${now.getUTCFullYear()}-${now.getUTCMonth()}-${now.getUTCDate()}-${now.getUTCHours()}`,
      dayKey:  `${now.getUTCFullYear()}-${now.getUTCMonth()}-${now.getUTCDate()}`,
      monKey:  `${now.getUTCFullYear()}-${now.getUTCMonth()}`,
    };
  }

  /**
   * Check if we can call this API.
   * Returns { allowed: bool, reason: string|null }
   */
  function canCall(apiKey) {
    const b = BUDGETS[apiKey];
    if (!b) return { allowed: true, reason: null };

    const store = getStore();
    const { hourKey, dayKey, monKey } = getWindows();

    const hourlyUsed  = (store[`${apiKey}_h_${hourKey}`]  || 0);
    const dailyUsed   = (store[`${apiKey}_d_${dayKey}`]   || 0);
    const monthlyUsed = (store[`${apiKey}_m_${monKey}`]   || 0);

    if (b.hourly  !== null && hourlyUsed  >= b.hourly)
      return { allowed: false, reason: `${b.label}: hourly limit (${b.hourly}/hr) reached. Next call in ${60 - new Date().getUTCMinutes()} min.` };
    if (b.daily   !== null && dailyUsed   >= b.daily)
      return { allowed: false, reason: `${b.label}: daily limit (${b.daily}/day) reached. Resets at midnight UTC.` };
    if (b.monthly !== null && monthlyUsed >= b.monthly)
      return { allowed: false, reason: `${b.label}: monthly limit (${b.monthly}/mo) reached!` };

    return { allowed: true, reason: null };
  }

  /**
   * Record a call. Call AFTER successful API response.
   */
  function record(apiKey) {
    const store = getStore();
    const { hourKey, dayKey, monKey } = getWindows();

    store[`${apiKey}_h_${hourKey}`]  = (store[`${apiKey}_h_${hourKey}`]  || 0) + 1;
    store[`${apiKey}_d_${dayKey}`]   = (store[`${apiKey}_d_${dayKey}`]   || 0) + 1;
    store[`${apiKey}_m_${monKey}`]   = (store[`${apiKey}_m_${monKey}`]   || 0) + 1;

    saveStore(store);
  }

  /**
   * Get current usage stats for all APIs
   */
  function getStats() {
    const store = getStore();
    const { hourKey, dayKey, monKey } = getWindows();
    const stats = {};
    for (const [key, b] of Object.entries(BUDGETS)) {
      stats[key] = {
        label:        b.label,
        hourly:       store[`${key}_h_${hourKey}`]  || 0,
        daily:        store[`${key}_d_${dayKey}`]   || 0,
        monthly:      store[`${key}_m_${monKey}`]   || 0,
        hourLimit:    b.hourly,
        dayLimit:     b.daily,
        monthLimit:   b.monthly,
      };
    }
    return stats;
  }

  /**
   * Clean up old keys (> 32 days) to prevent localStorage bloat
   */
  function cleanup() {
    try {
      const store = getStore();
      const now = new Date();
      const cutoff = new Date(now - 32 * 86400000);
      const toDelete = [];
      for (const k of Object.keys(store)) {
        // key format: apiKey_h_YYYY-M-D-H or _d_YYYY-M-D or _m_YYYY-M
        const parts = k.split('_');
        if (parts.length < 3) continue;
        const datePart = parts.slice(2).join('-');
        const segments = datePart.split('-').map(Number);
        let date;
        if (segments.length === 4) date = new Date(Date.UTC(segments[0], segments[1], segments[2]));
        else if (segments.length === 3) date = new Date(Date.UTC(segments[0], segments[1], segments[2]));
        else if (segments.length === 2) date = new Date(Date.UTC(segments[0], segments[1], 1));
        if (date && date < cutoff) toDelete.push(k);
      }
      toDelete.forEach(k => delete store[k]);
      saveStore(store);
    } catch {}
  }

  // Run cleanup on load
  cleanup();

  return { canCall, record, getStats, BUDGETS };
})();
