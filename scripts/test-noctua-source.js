#!/usr/bin/env node
// scripts/test-noctua-source.js
// =====================================================================
// The dashboard's forecast panel shows ONE model by design: NOCTUA's
// snapshot (data/kronos.json, written by the fetch-data cron). It used to
// accept that snapshot only when under 45 minutes old and otherwise scrape a
// third-party Kronos demo through free CORS proxies -- and with the cron
// firing every ~5 h that meant NOCTUA ~15% of the time and a DIRECTIONAL demo
// number (which NOCTUA deliberately pins to 50) the rest
// (model/research/PIPELINE_TRACE.md, finding F4).
//
// Runs the real src/data.js in a sandbox with a stubbed fetch and asserts:
//   - a snapshot is used while its 19-hour forecast window is open, however
//     long the cron gap, with its age computed NOW from the anchor time (the
//     file's own ageHrs/freshness are frozen at write time)
//   - once the window has closed, or the snapshot is missing or malformed,
//     the panel is offline (null) -- never another model
//   - no request ever leaves for the demo page or a CORS proxy
//   - a value cached by the old code (possibly the demo's) is never served
//
//   node scripts/test-noctua-source.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SRC = fs.readFileSync(path.join(__dirname, '..', 'src', 'data.js'), 'utf8');
const HOUR = 3600_000;
let failures = 0;

function check(name, ok, detail = '') {
  console.log(`  [${ok ? 'ok  ' : 'FAIL'}] ${name}${detail ? ' -- ' + detail : ''}`);
  if (!ok) failures++;
}

// One sandbox per case: fresh localStorage, a fetch that serves `snapshot`
// for ./data/kronos.json and records every URL requested.
function sandbox({ snapshot, status = 200, storage = {} } = {}) {
  const urls = [];
  const store = { ...storage };
  const ctx = {
    console: { log() {}, warn() {}, error() {} },
    Date, JSON, Math, Promise, AbortSignal, AbortController, URL, setTimeout, clearTimeout,
    encodeURIComponent, parseFloat, parseInt, isFinite, Number, String, Array, Object, Error,
    localStorage: {
      getItem: k => (k in store ? store[k] : null),
      setItem: (k, v) => { store[k] = String(v); },
      removeItem: k => { delete store[k]; },
    },
    RateLimit: { canCall: () => ({ allowed: true }), record() {} },
    fetch: async (url) => {
      urls.push(String(url));
      if (String(url).startsWith('./data/kronos.json')) {
        if (snapshot === undefined) throw new Error('network down');
        return { ok: status === 200, json: async () => JSON.parse(JSON.stringify(snapshot)) };
      }
      // what the old code scraped: the demo page, padded past its 200-char minimum
      const demo = '<html><h3>Upside Probability (Next 24h)</h3><p>77.0%</p><h3>Volatility Amplification</h3><p>12.0%</p>' + ' '.repeat(300) + '</html>';
      return { ok: true, text: async () => demo,
               json: async () => ({}) };
    },
  };
  vm.createContext(ctx);
  vm.runInContext(SRC + '\n;this.DataLayer = DataLayer;', ctx);
  return { DL: ctx.DataLayer, urls, store };
}

function snap(anchorAgoH, extra = {}) {
  const anchorMs = Date.now() - anchorAgoH * HOUR;
  return {
    upside: 50.0, p_up_raw: 50.4, volAmp: 55.6,
    sourceTs: new Date(anchorMs).toISOString().slice(0, 19).replace('T', ' '),
    sourceMs: anchorMs, tz: 'UTC', ageHrs: 0.0, freshness: 'fresh',
    fetchedAt: anchorMs + 1.3 * HOUR, proxy: 'noctua-local',
    _updatedMs: anchorMs + 1.3 * HOUR, model: 'NOCTUA-v2', ...extra,
  };
}

const offModel = urls => urls.filter(u => !u.startsWith('./data/'));

(async () => {
  console.log('noctua-source dashboard test');

  // 1. just published
  let s = sandbox({ snapshot: snap(1.5) });
  let k = await s.DL.fetchKronos();
  check('fresh snapshot is shown', k && k.model === 'NOCTUA-v2' && k.upside === 50,
        k ? `${k.model} upside ${k.upside}` : 'null');
  check('age is computed now from the anchor, not read from the file',
        k && Math.abs(k.ageHrs - 1.5) < 0.05 && k.freshness === 'fresh',
        k ? `ageHrs ${k.ageHrs?.toFixed(2)} ${k.freshness}` : '');

  // 2. the longest cron gap seen (8.5 h) plus the publication lag: still NOCTUA
  s = sandbox({ snapshot: snap(9.8) });
  k = await s.DL.fetchKronos();
  check('a 9.8 h old snapshot (longest cron gap) is still shown, labelled stale',
        k && k.model === 'NOCTUA-v2' && k.freshness === 'stale',
        k ? `${k.freshness}, ${k.ageHrs?.toFixed(1)} h` : 'null');
  check('no request to the demo or a proxy while a snapshot is valid',
        offModel(s.urls).length === 0, offModel(s.urls).join(', '));

  // 3. the forecast window (anchor + 19 h) has closed: offline, not the demo
  s = sandbox({ snapshot: snap(19.5) });
  k = await s.DL.fetchKronos();
  check('after the 19 h window closes the panel is offline', k === null,
        k ? `${k.model} ${k.upside}` : 'null');
  check('...and nothing is fetched from the demo or a proxy',
        offModel(s.urls).length === 0, offModel(s.urls).join(', '));

  // 4. snapshot unreachable / HTTP error / malformed
  for (const [name, opts] of [['network failure', { snapshot: undefined }],
                              ['HTTP 404', { snapshot: snap(1), status: 404 }],
                              ['no upside field', { snapshot: snap(1, { upside: undefined }) }]]) {
    s = sandbox(opts);
    k = await s.DL.fetchKronos();
    check(`${name}: offline, no other model`, k === null && offModel(s.urls).length === 0,
          `result ${k === null ? 'null' : JSON.stringify(k).slice(0, 60)}; other urls ${offModel(s.urls).length}`);
  }

  // 5. a value cached by the previous code (possibly the demo's) must not come back
  const old = { data: { upside: 77, volAmp: 12, freshness: 'fresh', proxy: 'corsproxy.io' },
                expires: Date.now() + 3 * HOUR };
  s = sandbox({ snapshot: undefined, storage: {
    'btc_cache_v4_kronos': JSON.stringify(old), 'btc_cache_v4_kronos_stale': JSON.stringify(old) } });
  k = await s.DL.fetchKronos();
  check('an old cached demo value is never served', k === null, k ? `served upside ${k.upside}` : 'null');

  // 6. the age keeps moving: a value cached at publish is re-aged on the next read
  s = sandbox({ snapshot: snap(1.0) });
  await s.DL.fetchKronos();
  const realNow = Date.now;
  Date.now = () => realNow() + 7 * HOUR;
  try { k = await s.DL.fetchKronos(); } finally { Date.now = realNow; }
  check('a cached snapshot is re-aged on every read', k && Math.abs(k.ageHrs - 8.0) < 0.05
        && k.freshness === 'stale', k ? `${k.ageHrs?.toFixed(2)} h ${k.freshness}` : 'null');

  // 7. the gate can fail: the OLD acceptance rule would have rejected case 2
  const oldRule = 45 * 60_000;
  check('gate-can-fail: the old 45-min rule rejects the 9.8 h snapshot',
        Date.now() - snap(9.8)._updatedMs > oldRule);

  console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
