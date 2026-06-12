/* =========================================================================
   BTC Option-Selling Desk — src/options.js (v1)
   One Deribit call for the full chain; IV inverted locally via Black-76
   bisection; analytic deltas; delta-targeted short-strangle builder with a
   Delta-Exchange margin heuristic; regime gate from Kronos + F&G + funding.
   Free resources only. No keys. CORS-clean endpoints:
     - api.deribit.com (Access-Control-Allow-Origin: *)
     - api.binance.com / fapi.binance.com
     - local ./data/*.json snapshots (committed by GH Actions)
   ========================================================================= */
'use strict';

/* ------------------------------ math: Black-76 -------------------------- */
function normCdf(x) {
  // Abramowitz & Stegun 7.1.26 — |err| < 7.5e-8, plenty for IV work
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  let p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
          t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

function black76(F, K, T, sigma, isCall) {
  if (T <= 0 || sigma <= 0) return Math.max(isCall ? F - K : K - F, 0);
  const sT = sigma * Math.sqrt(T);
  const d1 = (Math.log(F / K) + 0.5 * sT * sT) / sT;
  const d2 = d1 - sT;
  return isCall ? F * normCdf(d1) - K * normCdf(d2)
                : K * normCdf(-d2) - F * normCdf(-d1);
}

function black76Delta(F, K, T, sigma, isCall) {
  if (T <= 0 || sigma <= 0) return isCall ? (F > K ? 1 : 0) : (F < K ? -1 : 0);
  const sT = sigma * Math.sqrt(T);
  const d1 = (Math.log(F / K) + 0.5 * sT * sT) / sT;
  return isCall ? normCdf(d1) : normCdf(d1) - 1;
}

function impliedVol(price, F, K, T, isCall) {
  // bisection: robust, monotone in sigma; 60 iters ≈ 1e-9 precision
  if (!(price > 0) || !(F > 0) || !(K > 0) || !(T > 0)) return null;
  const intrinsic = Math.max(isCall ? F - K : K - F, 0);
  if (price <= intrinsic + 1e-9) return null;          // at/below intrinsic
  let lo = 0.005, hi = 5.0;
  if (black76(F, K, T, hi, isCall) < price) return null; // absurd mark
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    (black76(F, K, T, mid, isCall) > price) ? hi = mid : lo = mid;
  }
  return (lo + hi) / 2;
}

/* ------------------------------ state ----------------------------------- */
const S = {
  spot: null, hv20: null, funding: null,
  kronos: null, fng: null,
  chainByExpiry: new Map(),   // expiryMs -> [{strike, callMark, putMark, callOi, putOi, iv..., delta...}]
  selectedExpiry: null,
};

const $ = id => document.getElementById(id);
const fmt$ = v => v == null ? '—' : '$' + Math.round(v).toLocaleString();
const fmtPct = (v, d = 1) => v == null ? '—' : (v * 100).toFixed(d) + '%';

/* ------------------------------ fetchers --------------------------------- */
async function jget(url, opts) {
  const r = await fetch(url, Object.assign({ cache: 'no-store' }, opts));
  if (!r.ok) throw new Error(url + ' -> HTTP ' + r.status);
  return r.json();
}

async function fetchChain() {
  // Single call: every BTC option's mark, OI, underlying — we solve IV ourselves.
  const j = await jget('https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option');
  const rows = j.result || [];
  const byExp = new Map();
  let spot = null;
  for (const r of rows) {
    // instrument_name: BTC-27JUN26-60000-C
    const m = /^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$/.exec(r.instrument_name);
    if (!m) continue;
    const months = {JAN:0,FEB:1,MAR:2,APR:3,MAY:4,JUN:5,JUL:6,AUG:7,SEP:8,OCT:9,NOV:10,DEC:11};
    const expMs = Date.UTC(2000 + +m[3], months[m[2]], +m[1], 8, 0, 0); // Deribit expiry 08:00 UTC
    if (expMs < Date.now() + 30 * 60_000) continue;                      // skip expiring <30m
    const strike = +m[4];
    if (r.underlying_price > 0) spot = r.underlying_price;
    if (!byExp.has(expMs)) byExp.set(expMs, new Map());
    const chain = byExp.get(expMs);
    if (!chain.has(strike)) chain.set(strike, { strike });
    const row = chain.get(strike);
    const markUsd = (r.mark_price || 0) * (r.underlying_price || 0);     // BTC-denominated mark -> USD
    if (m[5] === 'C') { row.callMark = markUsd; row.callOi = r.open_interest || 0; }
    else              { row.putMark  = markUsd; row.putOi  = r.open_interest || 0; }
  }
  S.spot = spot;
  S.chainByExpiry = new Map(
    [...byExp.entries()].sort((a, b) => a[0] - b[0])
      .map(([exp, m]) => [exp, [...m.values()].sort((a, b) => a.strike - b.strike)])
  );
}

async function fetchHv20() {
  const k = await jget('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=22');
  const closes = k.map(c => +c[4]);
  const rets = [];
  for (let i = 1; i < closes.length; i++) rets.push(Math.log(closes[i] / closes[i - 1]));
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const varr = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length - 1);
  S.hv20 = Math.sqrt(varr) * Math.sqrt(365);
}

async function fetchFunding() {
  const j = await jget('https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT');
  S.funding = parseFloat(j.lastFundingRate);
}

async function fetchSnapshots() {
  // GH-Actions snapshots; tolerate absence (e.g. fresh clone).
  try { S.kronos = await jget('data/kronos.json?_=' + Date.now()); } catch (_) {}
  try {
    const local = await jget('data/kronos_local.json?_=' + Date.now());
    // prefer local model output when fresher
    if (local && (!S.kronos || (local._updatedMs || 0) > (S.kronos._updatedMs || 0))) S.kronos = local;
  } catch (_) {}
  try { S.fng = await jget('data/fg.json?_=' + Date.now()); } catch (_) {}
  try { S.finbert = await jget('data/sentiment.json?_=' + Date.now()); } catch (_) {}
}

/* --------------------------- chain enrichment ---------------------------- */
function enrich(expiryMs) {
  const T = (expiryMs - Date.now()) / (365 * 86400_000);
  const F = S.spot;
  const rows = S.chainByExpiry.get(expiryMs) || [];
  for (const r of rows) {
    r.callIv = r.callMark != null ? impliedVol(r.callMark, F, r.strike, T, true)  : null;
    r.putIv  = r.putMark  != null ? impliedVol(r.putMark,  F, r.strike, T, false) : null;
    r.callDelta = r.callIv ? black76Delta(F, r.strike, T, r.callIv, true)  : null;
    r.putDelta  = r.putIv  ? black76Delta(F, r.strike, T, r.putIv,  false) : null;
  }
  return { rows, T };
}

function atmIv(rows) {
  if (!rows.length || !S.spot) return null;
  const atm = rows.reduce((a, b) =>
    Math.abs(b.strike - S.spot) < Math.abs(a.strike - S.spot) ? b : a);
  const ivs = [atm.callIv, atm.putIv].filter(v => v != null);
  return ivs.length ? ivs.reduce((a, b) => a + b, 0) / ivs.length : null;
}

/* --------------------------- strangle builder ---------------------------- */
function pickLeg(rows, targetAbsDelta, side) {
  // side 'P': delta in (-1,0); 'C': delta in (0,1). Pick OTM leg nearest target.
  let best = null, bestErr = Infinity;
  for (const r of rows) {
    const d = side === 'P' ? r.putDelta : r.callDelta;
    const mark = side === 'P' ? r.putMark : r.callMark;
    if (d == null || mark == null || mark <= 0) continue;
    if (side === 'P' && r.strike >= S.spot) continue;   // OTM only
    if (side === 'C' && r.strike <= S.spot) continue;
    const err = Math.abs(Math.abs(d) - targetAbsDelta);
    if (err < bestErr) { bestErr = err; best = r; }
  }
  return best;
}

function deltaExMarginPerLeg(strike, premiumUsd, isPut) {
  // Delta Exchange short-option heuristic (documented in footer; VERIFY in
  // their calculator): max(15% * spot - OTM distance, 7.5% * spot) + premium.
  const spot = S.spot;
  const otm = isPut ? Math.max(spot - strike, 0) : Math.max(strike - spot, 0);
  return Math.max(0.15 * spot - otm, 0.075 * spot) + premiumUsd;
}

/* ------------------------------ risk gate -------------------------------- */
function computeGate(atm) {
  const why = [];
  let score = 0; // positive = favorable to sell premium

  const ivhv = (atm && S.hv20) ? atm / S.hv20 : null;
  if (ivhv != null) {
    if (ivhv >= 1.25) { score += 2; why.push(`IV/HV ${ivhv.toFixed(2)} — fat premium vs realized (edge to sellers)`); }
    else if (ivhv >= 1.05) { score += 1; why.push(`IV/HV ${ivhv.toFixed(2)} — modest vol-risk premium`); }
    else { score -= 2; why.push(`IV/HV ${ivhv.toFixed(2)} — options are CHEAP vs realized; selling has no statistical edge`); }
  }

  const volAmp = S.kronos?.volAmp;
  if (volAmp != null) {
    if (volAmp >= 80) { score -= 2; why.push(`Kronos vol-amplification ${volAmp}% — model expects realized vol to EXPAND; short gamma is dangerous`); }
    else if (volAmp >= 60) { score -= 1; why.push(`Kronos vol-amplification ${volAmp}% — elevated expansion risk`); }
    else { score += 1; why.push(`Kronos vol-amplification ${volAmp}% — vol expected calm`); }
  }

  const up = S.kronos?.upside;
  if (up != null && (up >= 70 || up <= 30)) {
    score -= 1; why.push(`Kronos directional skew (upside ${up}%) — delta-neutral strangles fight a directional model`);
  }

  const fv = S.fng?.value;
  if (fv != null) {
    if (fv <= 15 || fv >= 88) { score -= 1; why.push(`Fear & Greed ${fv} — extreme readings precede outsized moves; widen strikes or cut size`); }
    else { score += 1; why.push(`Fear & Greed ${fv} — mid-regime, mean-reversion friendly`); }
  }

  const fb = S.finbert;
  if (fb && fb.score != null && Math.abs(fb.score) > 0.35) {
    score -= 1;
    why.push(`FinBERT news sentiment ${fb.score > 0 ? '+' : ''}${fb.score} (${fb.label}, ${fb.n} headlines) — strong one-sided news flow fuels trends, the enemy of strangles`);
  } else if (fb && fb.score != null) {
    why.push(`FinBERT news sentiment ${fb.score > 0 ? '+' : ''}${fb.score} (${fb.label}) — news flow balanced`);
  }

  if (S.funding != null && Math.abs(S.funding) > 0.0003) {
    score -= 1; why.push(`Funding ${(S.funding * 100).toFixed(4)}%/8h — crowded perp positioning, squeeze risk`);
  } else if (S.funding != null) {
    why.push(`Funding ${(S.funding * 100).toFixed(4)}%/8h — neutral positioning`);
  }

  let cls, label;
  if (score >= 2)      { cls = 'v-sell';    label = 'GREEN — conditions favor selling premium (size normally)'; }
  else if (score >= 0) { cls = 'v-caution'; label = 'AMBER — sell only wide strikes at reduced size'; }
  else                 { cls = 'v-stand';   label = 'RED — stand down / buy-side or spreads only'; }
  return { score, cls, label, why, ivhv };
}

/* --------------------------- desk notes (plain English) ------------------ */
/* The translation layer: turns the same numbers a fund desk reads into
   sentences a first-year retail trader can act on. No jargon unexplained. */
function renderDeskNotes(gate, atm, em, T) {
  const el = document.getElementById('deskNotes');
  if (!el) return;
  const p = [];

  // 1. Where we are
  if (S.spot) {
    const emPct = em != null ? (em / S.spot * 100).toFixed(1) : null;
    p.push(`<b>Where we are.</b> Bitcoin trades at <b>${fmt$(S.spot)}</b>. ` +
      (emPct != null
        ? `The options market is pricing a normal move of about <b>±${emPct}%</b> (±${fmt$(em)}) between now and this expiry. Think of that as the market's own weather forecast — roughly 2 days out of 3, price should stay inside that band.`
        : `Chain data is still loading, so no expected-move estimate yet.`));
  }

  // 2. Is premium rich or cheap?
  if (gate.ivhv != null) {
    if (gate.ivhv >= 1.15) {
      p.push(`<b>Is selling worth it?</b> Options are currently priced <b>${((gate.ivhv - 1) * 100).toFixed(0)}% richer</b> than how much Bitcoin has actually been moving (IV ${fmtPct(atm)} vs realized ${fmtPct(S.hv20)}). That gap is the <i>vol-risk premium</i> — the "insurance markup" you collect as a seller. Today the markup exists.`);
    } else if (gate.ivhv >= 1.0) {
      p.push(`<b>Is selling worth it?</b> Options are priced only slightly above realized movement (IV ${fmtPct(atm)} vs ${fmtPct(S.hv20)}). The seller's edge is thin — like selling insurance at nearly cost price. Acceptable, not exciting.`);
    } else {
      p.push(`<b>Is selling worth it?</b> <span class="neg">No.</span> Options are priced <i>cheaper</i> than Bitcoin's actual movement (IV ${fmtPct(atm)} vs ${fmtPct(S.hv20)}). Selling here is selling insurance below cost — the statistical edge belongs to buyers today.`);
    }
  }

  // 3. The AI forecast
  if (S.kronos?.volAmp != null) {
    const va = S.kronos.volAmp, up = S.kronos.upside;
    if (va >= 80) {
      p.push(`<b>What the AI sees.</b> The Kronos model (trained on 12B financial data points) gives a <b class="neg">${va}% chance volatility EXPANDS</b> in the next 24h${up != null ? ` and a ${up}% chance price ends higher` : ''}. Expanding volatility is the one thing that hurts option sellers most — it's the storm warning. When this number is above 80, funds cut their short-vol books, not grow them.`);
    } else if (va >= 60) {
      p.push(`<b>What the AI sees.</b> Kronos puts vol-expansion odds at <b class="warn">${va}%</b>${up != null ? ` (upside ${up}%)` : ''} — choppier than ideal. Sellers should go wider on strikes and smaller on size.`);
    } else {
      p.push(`<b>What the AI sees.</b> Kronos expects calm: only ${va}% odds of volatility expanding${up != null ? `, upside ${up}%` : ''}. Quiet tape is a premium-seller's best friend.`);
    }
  }

  // 4. Crowd + news
  const crowd = [];
  if (S.fng?.value != null) {
    const fv = S.fng.value;
    crowd.push(fv <= 20 ? `the crowd is in <b class="neg">${S.fng.label || 'Extreme Fear'}</b> (${fv}/100) — historically the zone of violent snap-back rallies`
      : fv >= 80 ? `the crowd is in <b class="warn">${S.fng.label || 'Extreme Greed'}</b> (${fv}/100) — euphoria precedes air-pockets`
      : `crowd mood is mid-range (${fv}/100) — no emotional extreme to fade or fear`);
  }
  if (S.finbert?.score != null) {
    crowd.push(`AI-read news sentiment (FinBERT) is <b>${S.finbert.label}</b> (${S.finbert.score > 0 ? '+' : ''}${S.finbert.score})`);
  }
  if (S.funding != null) {
    crowd.push(Math.abs(S.funding) > 0.0003
      ? `perp funding at ${(S.funding * 100).toFixed(4)}%/8h shows a crowded ${S.funding > 0 ? 'long' : 'short'} side — squeeze fuel`
      : `perp funding is neutral — no crowded side to squeeze`);
  }
  if (crowd.length) p.push(`<b>The crowd.</b> ${crowd.join('; ')}.`);

  // 5. The instruction
  if (gate.cls === 'v-sell') {
    p.push(`<b>Bottom line.</b> <span class="pos">Conditions favor selling premium.</span> The builder below has picked strikes a fund desk would recognize: far enough out to win ~${document.getElementById('pop')?.textContent || '70%+'} of the time, close enough to be paid for the risk. Enter, set the exit rules, and let the math work.`);
  } else if (gate.cls === 'v-caution') {
    p.push(`<b>Bottom line.</b> <span class="warn">Tradeable, but on half rations.</span> Sell wider strikes (drop target |Δ| to 0.10), cut lots in half, and take profits early at 50% of credit. The edge is there but the weather is unsettled.`);
  } else {
    p.push(`<b>Bottom line.</b> <span class="neg">Stand down.</span> This is a day to NOT sell naked options — the desk's most profitable trades are often the ones never placed. If you must trade, use defined-risk spreads (buy a further wing against each short leg) so a wild move can't hurt you beyond a known amount. Re-check tomorrow; regimes flip fast.`);
  }

  el.innerHTML = p.map(x => `<p style="margin:0 0 9px">${x}</p>`).join('');

  const g = document.getElementById('deskGlossary');
  if (g) g.innerHTML =
    `<b>30-second glossary:</b> <i>IV</i> = how big a move options are charging for · ` +
    `<i>Realized/HV</i> = how big moves have actually been · ` +
    `<i>Δ (delta)</i> ≈ odds an option finishes in-the-money (0.15Δ ≈ 15%) · ` +
    `<i>POP</i> = probability the whole trade profits · ` +
    `<i>Strangle</i> = sell one put below + one call above; you win if price stays between them.`;
}

/* ------------------------------ rendering -------------------------------- */
function expLabel(ms) {
  const d = new Date(ms);
  const days = ((ms - Date.now()) / 86400_000).toFixed(1);
  return d.toISOString().slice(0, 10) + ` (${days}d)`;
}

function renderAll() {
  const expiry = S.selectedExpiry;
  const { rows, T } = enrich(expiry);
  const atm = atmIv(rows);

  $('spot').textContent = fmt$(S.spot);
  $('atmIv').textContent = fmtPct(atm);
  $('hv20').textContent = fmtPct(S.hv20);
  const gate = computeGate(atm);
  $('ivhv').textContent = gate.ivhv != null ? gate.ivhv.toFixed(2) + '×' : '—';
  $('ivhv').className = gate.ivhv >= 1.15 ? 'pos' : gate.ivhv <= 1.0 ? 'neg' : 'warn';
  const em = (atm != null) ? S.spot * atm * Math.sqrt(T) : null;
  $('expMove').textContent = em != null ? '±' + fmt$(em).slice(1) : '—';
  $('funding').textContent = S.funding != null ? (S.funding * 100).toFixed(4) + '% /8h' : '—';

  $('krUp').textContent  = S.kronos?.upside  != null ? S.kronos.upside + '%'  : 'n/a (run snapshot or kronos_local)';
  $('krVol').textContent = S.kronos?.volAmp != null ? S.kronos.volAmp + '%' : 'n/a';
  $('krVol').className   = (S.kronos?.volAmp ?? 0) >= 80 ? 'neg' : (S.kronos?.volAmp ?? 0) >= 60 ? 'warn' : 'pos';
  $('fng').textContent   = S.fng?.value != null ? `${S.fng.value} · ${S.fng.label || ''}` : 'n/a';
  $('vrp').textContent   = gate.ivhv != null ? (gate.ivhv >= 1.15 ? 'PRESENT' : gate.ivhv >= 1.0 ? 'THIN' : 'ABSENT') : '—';

  const v = $('verdict');
  v.textContent = gate.label;
  v.className = 'verdict ' + gate.cls;
  $('verdictWhy').innerHTML = gate.why.map(w => '• ' + w).join('<br>');

  renderStrangle(rows, T, em);
  renderChain(rows, expiry);
  renderDeskNotes(gate, atm, em, T);   // after renderStrangle: Desk Notes reads the live POP figure
}

function renderStrangle(rows, T, em) {
  const target = +$('deltaSlider').value / 100;
  const lots = +$('lotSlider').value;
  const sizeBtc = lots * 0.001;                       // Delta Exchange BTC option lot = 0.001 BTC
  $('deltaVal').textContent = target.toFixed(2);
  $('lotVal').textContent = lots;

  const put = pickLeg(rows, target, 'P');
  const call = pickLeg(rows, target, 'C');
  if (!put || !call) {
    $('putLeg').textContent = $('callLeg').textContent = 'insufficient chain liquidity';
    return;
  }
  S._legs = { put: put.strike, call: call.strike };

  const credit = (put.putMark + call.callMark) * sizeBtc;
  const beLo = put.strike - (put.putMark + call.callMark);
  const beHi = call.strike + (put.putMark + call.callMark);
  const pop = 1 - (Math.abs(put.putDelta) + Math.abs(call.callDelta));

  // margin: full on the expensive leg, 50% on the cheaper (strangle benefit heuristic)
  const mP = deltaExMarginPerLeg(put.strike, put.putMark, true)  * sizeBtc;
  const mC = deltaExMarginPerLeg(call.strike, call.callMark, false) * sizeBtc;
  const margin = Math.max(mP, mC) + 0.5 * Math.min(mP, mC);

  $('putLeg').textContent = `${put.strike.toLocaleString()} P @ ${fmt$(put.putMark)}  (Δ ${put.putDelta.toFixed(2)}, IV ${fmtPct(put.putIv)})`;
  $('callLeg').textContent = `${call.strike.toLocaleString()} C @ ${fmt$(call.callMark)}  (Δ +${call.callDelta.toFixed(2)}, IV ${fmtPct(call.callIv)})`;
  $('credit').textContent = fmt$(credit) + `  (${sizeBtc} BTC notional/leg)`;
  $('breakevens').textContent = `${fmt$(beLo)}  /  ${fmt$(beHi)}`;
  $('pop').textContent = fmtPct(pop, 0);
  $('margin').textContent = fmt$(margin);
  $('rom').textContent = margin > 0 ? fmtPct(credit / margin) : '—';
  $('cem').textContent = em ? ((put.putMark + call.callMark) / em).toFixed(2) + '× of 1σ move' : '—';

  const wingWidthPct = ((call.strike - put.strike) / S.spot * 100).toFixed(1);
  $('legNotes').textContent =
    `Strikes span ${wingWidthPct}% of spot. Defense plan: roll the tested leg when its delta doubles, ` +
    `or close the structure at 50% of max profit / 2× credit loss — whichever comes first.`;
}

function renderChain(rows, expiry) {
  $('chainExpiry').textContent = '· ' + expLabel(expiry);
  const tb = $('chainTbl').querySelector('tbody');
  const atmStrike = rows.length ? rows.reduce((a, b) =>
    Math.abs(b.strike - S.spot) < Math.abs(a.strike - S.spot) ? b : a).strike : null;
  tb.innerHTML = rows
    .filter(r => Math.abs(r.strike - S.spot) / S.spot < 0.35)  // ±35% window
    .map(r => {
      const cls = r.strike === atmStrike ? 'atm'
        : (S._legs && (r.strike === S._legs.put || r.strike === S._legs.call)) ? 'leg' : '';
      return `<tr class="${cls}">
        <td>${r.callMark != null ? Math.round(r.callMark) : ''}</td>
        <td>${r.callIv != null ? (r.callIv * 100).toFixed(1) : ''}</td>
        <td>${r.callDelta != null ? r.callDelta.toFixed(2) : ''}</td>
        <td>${r.callOi ? r.callOi.toFixed(0) : ''}</td>
        <td style="text-align:center;font-weight:700">${r.strike.toLocaleString()}</td>
        <td>${r.putOi ? r.putOi.toFixed(0) : ''}</td>
        <td>${r.putDelta != null ? r.putDelta.toFixed(2) : ''}</td>
        <td>${r.putIv != null ? (r.putIv * 100).toFixed(1) : ''}</td>
        <td>${r.putMark != null ? Math.round(r.putMark) : ''}</td>
      </tr>`;
    }).join('');
}

/* ------------------------------ bootstrap -------------------------------- */
let inFlight = false;
async function refresh() {
  if (inFlight) return;
  inFlight = true;
  $('status').textContent = 'loading chain…';
  try {
    await Promise.allSettled([fetchChain(), fetchHv20(), fetchFunding(), fetchSnapshots()]);
    if (!S.chainByExpiry.size) throw new Error('empty chain');
    const sel = $('expirySel');
    const keep = S.selectedExpiry;
    sel.innerHTML = [...S.chainByExpiry.keys()].slice(0, 8)
      .map(ms => `<option value="${ms}">${expLabel(ms)}</option>`).join('');
    S.selectedExpiry = (keep && S.chainByExpiry.has(keep)) ? keep : +sel.options[0].value;
    sel.value = S.selectedExpiry;
    renderAll();
    $('status').textContent = 'updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    console.error(e);
    $('status').textContent = 'load failed — ' + e.message;
  } finally {
    inFlight = false;
  }
}

let uiDebounce = null;
function onUi() { clearTimeout(uiDebounce); uiDebounce = setTimeout(renderAll, 80); }

document.addEventListener('DOMContentLoaded', () => {
  $('btnRefresh').addEventListener('click', refresh);
  $('expirySel').addEventListener('change', e => { S.selectedExpiry = +e.target.value; renderAll(); });
  $('deltaSlider').addEventListener('input', onUi);
  $('lotSlider').addEventListener('input', onUi);
  refresh();
  setInterval(refresh, 5 * 60_000);   // 5-min auto refresh, well inside free limits
});
