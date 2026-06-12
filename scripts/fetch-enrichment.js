#!/usr/bin/env node
/**
 * scripts/fetch-enrichment.js  (v4.1)
 * =====================================================================
 * GH Actions-side news + fear/greed enrichment. Server-side fetches hit
 * the sources directly (no CORS proxies, no browser rate budgets) and
 * write static snapshots the dashboard prefers over live scraping:
 *
 *   data/news.json  — merged CryptoPanic RSS + GDELT, deduped, sentiment-scored
 *   data/fg.json    — alternative.me Fear & Greed index
 *
 * This is the pattern ported from the CODEX backend (live_sources.py):
 * direct fetch + merge, instead of the GitHub-Pages CORS-proxy roulette.
 *
 * Node >= 18 (global fetch). No dependencies. Optional env: EXA_API_KEY
 * (repo secret) upgrades news quality via Exa search.
 */
const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'data');

// ── shared helpers (mirrors of data.js scoreSentiment / dedupeNews) ──────
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

function dedupeNews(items) {
  const seen = new Set();
  const out = [];
  for (const it of items || []) {
    let host = '';
    try { host = new URL(it.url).hostname.replace(/^www\./, ''); } catch { /* keep '' */ }
    const titleKey = (it.headline || '').toLowerCase()
      .replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 60);
    const key = host + '|' + titleKey;
    const titleOnlyKey = 't|' + titleKey;
    if (seen.has(key) || (titleKey.length > 20 && seen.has(titleOnlyKey))) continue;
    seen.add(key); seen.add(titleOnlyKey);
    out.push(it);
  }
  return out;
}

async function get(url, ms = 15000, headers = {}) {
  const r = await fetch(url, {
    signal: AbortSignal.timeout(ms),
    headers: { 'User-Agent': 'btc-dashboard-snapshot/1.0', ...headers },
  });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${url}`);
  return r;
}

// ── sources ───────────────────────────────────────────────────────────────
async function fetchCryptoPanic() {
  const xml = await (await get('https://cryptopanic.com/news/rss/?currencies=BTC')).text();
  const itemRegex = /<item>[\s\S]*?<title>([\s\S]*?)<\/title>[\s\S]*?<link>([\s\S]*?)<\/link>[\s\S]*?<\/item>/g;
  const items = [];
  let m;
  while ((m = itemRegex.exec(xml)) && items.length < 15) {
    const title = m[1].replace(/<!\[CDATA\[|\]\]>/g, '').trim();
    const link  = m[2].replace(/<!\[CDATA\[|\]\]>/g, '').trim();
    let src = 'CryptoPanic';
    try { src = new URL(link).hostname.replace(/^www\./, ''); } catch { /* default */ }
    items.push({ headline: title, url: link, src, sent: scoreSentiment(title) });
  }
  return items;
}

async function fetchGdelt() {
  const url = 'https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin%20BTC&mode=ArtList&format=json&maxrecords=15&sort=DateDesc';
  const j = await (await get(url)).json();
  return (j.articles || []).slice(0, 15).map(a => ({
    headline: a.title, url: a.url, src: a.domain || 'GDELT', date: a.seendate,
    sent: scoreSentiment(a.title),
  }));
}

async function fetchExa(apiKey) {
  const r = await fetch('https://api.exa.ai/search', {
    method: 'POST',
    signal: AbortSignal.timeout(15000),
    headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
    body: JSON.stringify({
      query: 'Bitcoin BTC price market news today',
      numResults: 10, type: 'auto', category: 'news',
      startPublishedDate: new Date(Date.now() - 86400_000).toISOString(),
    }),
  });
  if (!r.ok) throw new Error(`Exa HTTP ${r.status}`);
  const j = await r.json();
  return (j.results || []).map(x => ({
    headline: x.title, url: x.url,
    src: (() => { try { return new URL(x.url).hostname.replace(/^www\./, ''); } catch { return 'Exa'; } })(),
    date: x.publishedDate, sent: scoreSentiment(x.title),
  }));
}

async function fetchFearGreed() {
  const j = await (await get('https://api.alternative.me/fng/?limit=2')).json();
  const cur = j?.data?.[0];
  if (!cur) throw new Error('no FG data');
  return {
    value: parseInt(cur.value, 10),
    label: cur.value_classification,
    prev: j.data[1] ? parseInt(j.data[1].value, 10) : null,
    srcTs: parseInt(cur.timestamp, 10) * 1000,
    ts: Date.now(),
    _updatedMs: Date.now(),
  };
}

// ── main ──────────────────────────────────────────────────────────────────
(async () => {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  let exitCode = 0;

  // News: gather every source independently; one failure must not sink the rest.
  const sources = [];
  const exaKey = process.env.EXA_API_KEY && process.env.EXA_API_KEY.length > 20
    ? process.env.EXA_API_KEY : null;
  const tasks = [
    ['CryptoPanic', fetchCryptoPanic()],
    ['GDELT', fetchGdelt()],
  ];
  if (exaKey) tasks.push(['Exa', fetchExa(exaKey)]);

  const settled = await Promise.allSettled(tasks.map(([, p]) => p));
  let merged = [];
  settled.forEach((res, i) => {
    const name = tasks[i][0];
    if (res.status === 'fulfilled' && res.value.length) {
      console.log(`[enrichment] ${name}: ${res.value.length} items`);
      sources.push(name);
      merged = merged.concat(res.value);
    } else {
      console.error(`[enrichment] ${name} failed:`, res.reason?.message || 'empty');
    }
  });

  if (merged.length) {
    const news = {
      items: dedupeNews(merged).slice(0, 20),
      ts: Date.now(),
      source: sources.join('+'),
      _updatedMs: Date.now(),
    };
    fs.writeFileSync(path.join(DATA_DIR, 'news.json'), JSON.stringify(news, null, 2) + '\n');
    console.log(`[enrichment] wrote news.json (${news.items.length} items from ${news.source})`);
  } else {
    console.error('[enrichment] all news sources failed — keeping previous news.json');
    exitCode = 0; // deliberate: stale snapshot beats a red workflow
  }

  // Fear & Greed
  try {
    const fg = await fetchFearGreed();
    fs.writeFileSync(path.join(DATA_DIR, 'fg.json'), JSON.stringify(fg, null, 2) + '\n');
    console.log(`[enrichment] wrote fg.json (${fg.value} · ${fg.label})`);
  } catch (e) {
    console.error('[enrichment] fear/greed failed:', e.message);
  }

  process.exit(exitCode);
})();
