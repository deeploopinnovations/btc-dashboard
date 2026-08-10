#!/usr/bin/env node
/**
 * scripts/parse-kronos.js  (v4.1)  ** RETIRED -- NO LONGER RUN BY CI **
 *
 * The public Kronos demo stopped updating; its last genuine source timestamp
 * was 2026-07-04 11:00:26 UTC and this script had been re-committing a fossil
 * ever since. data/kronos.json is now produced by model/serve/predict.py
 * (NOCTUA). Kept for reference only.
 * =====================================================================
 * GH Actions-side Kronos demo scraper. Runs server-side (no CORS), parses
 * "Upside Probability (Next 24h)" and "Volatility Amplification (Next 24h)"
 * from https://shiyu-coder.github.io/Kronos-demo/ and writes data/kronos.json
 * in exactly the shape the browser's fetchKronos() produces, plus _updatedMs
 * so tryLoadSnapshot() can age-gate it (45-min window).
 *
 * Node >= 18 (global fetch). No dependencies.
 */
const fs = require('fs');
const path = require('path');

const TARGET = 'https://shiyu-coder.github.io/Kronos-demo/';
const OUT = path.join(__dirname, '..', 'data', 'kronos.json');

function fail(msg) { console.error('[parse-kronos] FATAL:', msg); process.exit(1); }

// Mirror of data.js parseKronosHtml() strategies B/C (regex-based; no DOM here).
function parseKronosHtml(html) {
  const text = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ');

  const num = '([0-9]{1,3}(?:\\.[0-9]+)?)\\s*%';
  const upM = text.match(new RegExp('Upside\\s+Probability[^%]{0,80}?' + num, 'i'));
  const vaM = text.match(new RegExp('Volatility\\s+Amplification[^%]{0,80}?' + num, 'i'));
  if (!upM || !vaM) return null;

  const upside = parseFloat(upM[1]);
  const volAmp = parseFloat(vaM[1]);
  if (!(upside >= 0 && upside <= 100)) return null;
  if (!(volAmp >= 0 && volAmp <= 100)) return null;

  // Timestamp: demo page renders e.g. "Last update: 2026-06-12 08:00" (UTC).
  // Page renders: "Last Updated (UTC): 2026-06-12 04:00:25" — allow up to a
  // dozen non-digit chars (the "(UTC):" token) between the label and the date.
  const tsM = text.match(/(?:last\s+updat\w*|updated?)[^0-9]{0,16}([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}(?::[0-9]{2})?)/i);
  return { upside, volAmp, sourceTs: tsM ? tsM[1] : null };
}

function parseSourceTsUTC(s) {
  if (!s) return null;
  const m = s.match(/([0-9]{4})-([0-9]{2})-([0-9]{2})[ T]([0-9]{2}):([0-9]{2})(?::([0-9]{2}))?/);
  if (!m) return null;
  return Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] || 0));
}

(async () => {
  let html;
  try {
    const r = await fetch(TARGET, { signal: AbortSignal.timeout(20000), headers: { 'User-Agent': 'btc-dashboard-snapshot/1.0' } });
    if (!r.ok) fail(`HTTP ${r.status} from Kronos demo`);
    html = await r.text();
  } catch (e) { fail('fetch failed: ' + e.message); }

  const parsed = parseKronosHtml(html);
  if (!parsed) {
    // Don't fail the whole workflow on a parse miss — keep yesterday's file.
    console.error('[parse-kronos] parse failed (page layout changed?). html length =', html.length);
    process.exit(0);
  }

  const srcMs = parseSourceTsUTC(parsed.sourceTs);
  const ageHrs = srcMs ? (Date.now() - srcMs) / 3600_000 : null;
  const freshness = (ageHrs === null || ageHrs < -0.25) ? 'unknown'
                  : ageHrs < 2  ? 'fresh'
                  : ageHrs < 8  ? 'recent'
                  : ageHrs < 24 ? 'stale'
                  :               'very-stale';

  const data = {
    upside: parsed.upside,
    volAmp: parsed.volAmp,
    sourceTs: parsed.sourceTs,
    sourceMs: srcMs,
    tz: 'UTC',
    ageHrs,
    freshness,
    fetchedAt: Date.now(),
    proxy: 'gh-actions',
    _updatedMs: Date.now(),
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(data, null, 2) + '\n');
  console.log('[parse-kronos] wrote', OUT, `upside=${data.upside}% volAmp=${data.volAmp}% freshness=${freshness}`);
})();
