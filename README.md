# ₿ KRONOS/RANGER — BTC Live Options Intelligence Dashboard

> Real-time BTC options signal dashboard combining **Kronos AI** (AAAI 2026), **RANGER formula**, news sentiment, and live market data. Fully static — runs entirely in the browser. No server needed. Zero cost.

---

## 🚀 Live Dashboard

**[👉 Click here to open the live dashboard](https://YOUR_USERNAME.github.io/btc-dashboard/)**

> Replace `YOUR_USERNAME` with your GitHub username after deploying (see instructions below).
> The dashboard loads instantly and pulls live data directly from free public APIs.

---

## 📊 What It Shows

| Feature | Source | Refresh |
|---|---|---|
| BTC price + 24H high/low | Crypto.com public API | Every 60 sec |
| RANGER range prediction | ATR-7 formula + IV/HV | Every 60 min |
| Kronos AI direction signal | AAAI 2026 live demo | Once at 09:00 IST |
| News sentiment | Exa API (keyword scored) | Every 60 min |
| Fear & Greed Index | Alternative.me | Once per day |
| Options OI + max pain | Deribit public API | Every 10 min |
| Theta decay visualization | Pure calculation | Every 5 min |
| Combined conviction score | Weighted composite | Every 60 min |
| Recommended strikes (Call/Put) | RANGER × 2.2 + Kronos adj. | Every 60 min |

---

## 🛡️ Rate Limit Protection

Every API call passes through a **hard budget enforcer** (built into `src/rateLimit.js`). Limits are tracked in `localStorage` by hour, day, and month. No API can ever be over-called.

| API | Free Limit | Our Usage | Safety Margin |
|---|---|---|---|
| **Exa Search** | 1,000/month · 33/day | **1/hour = 24/day max** | 27% buffer daily |
| **Alternative.me** | Unlimited (polite) | 1/day | N/A |
| **Crypto.com** | Unlimited public | 1/minute (price) | N/A |
| **Deribit** | Unlimited public | 1/10 min (options) | N/A |
| **Kronos Demo** | Unlimited (scrape) | 1/day at 09:00 IST | N/A |
| **BigData.com** | Connector quota | 2/day | Conservative |
| **Binance OHLCV** | Unlimited public | On demand | N/A |

The dashboard shows a **live rate limit panel** so you always know exactly how many calls have been used.

---

## 📦 Tech Stack (All Free)

- **No backend** — pure HTML + CSS + JavaScript
- **No build step** — open `index.html` directly or serve with GitHub Pages
- **No framework** — vanilla JS in 5 clean modules
- **Chart.js 4.4** (CDN) — price + theta charts
- **Google Fonts** (CDN) — Space Mono + DM Sans
- All data sources: free public APIs or free tiers

---

## 🌐 Deploy to GitHub Pages (5 minutes)

### Step 1 — Create your repo
```bash
# Option A: GitHub CLI
gh repo create btc-dashboard --public --clone
cd btc-dashboard

# Option B: Go to github.com → New repository → name: btc-dashboard → Public
```

### Step 2 — Add the files
```bash
# Copy all files from this repo into your new repo folder
git add .
git commit -m "Initial deploy: BTC KRONOS/RANGER dashboard"
git push origin main
```

### Step 3 — Enable GitHub Pages
1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Source**: select `Deploy from a branch`
4. Branch: `main` · Folder: `/ (root)`
5. Click **Save**

### Step 4 — Your live URL
```
https://YOUR_USERNAME.github.io/btc-dashboard/
```

**Update the README link** at the top of this file with your actual URL.

⏱️ First deploy takes ~2 minutes. After that, every `git push` auto-redeploys in ~30 seconds.

---

## 🔑 Optional: Add Exa API Key

The dashboard works without an Exa API key (falls back to cached news). To enable live news sentiment:

1. Get a free key at [exa.ai](https://exa.ai) (1,000 searches/month free)
2. Create `src/config.js`:
```javascript
// src/config.js — DO NOT COMMIT THIS FILE if key is private
window.EXA_API_KEY = 'your-key-here';
```
3. Add to `.gitignore`:
```
src/config.js
```
4. Add to `index.html` before other scripts:
```html
<script src="src/config.js"></script>
```

> ⚠️ **For public repos**: never commit your API key. The dashboard works fine without it using cached static news.

---

## 📁 Project Structure

```
btc-dashboard/
├── index.html          ← Main dashboard UI (open this in browser)
├── src/
│   ├── rateLimit.js    ← API budget enforcer (localStorage-based)
│   ├── data.js         ← All data fetching with caching + rate limiting
│   ├── charts.js       ← Chart.js renders (price, theta, gauges)
│   ├── ui.js           ← All DOM updates
│   └── main.js         ← Orchestrator + auto-refresh loops
└── README.md           ← This file
```

---

## 🤖 About Kronos

Kronos is the first open-source foundation model for financial candlesticks (K-lines), trained on 12 billion records from 45 exchanges. Published at **AAAI 2026** by Tsinghua University.

- 📄 Paper: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- 💻 GitHub: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- 🤗 Models: [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)
- 🔴 Live demo: [Kronos-demo](https://shiyu-coder.github.io/Kronos-demo/)

---

## 📐 RANGER Formula

```
RANGER = ATR₇ × (IV/HV)^0.20 × (Vol/AvgVol)^0.12 × DOW^0.05 × FG_Factor

Where:
  ATR₇      = 7-day average true range (%)
  IV/HV     = implied vol ÷ historical vol (estimated 1.40×)
  Vol/AvgVol = today's volume ÷ 20-day average
  DOW       = day-of-week multiplier (Sun=1.12, Mon=1.08 ... Thu=0.97)
  FG_Factor = 1 + (50 − FearGreedIndex) / 200

Safe strangle width = RANGER × 2.2 (×2.2 buffer for options strikes)
```

Kronos direction bias adjusts the put/call strikes:
- Kronos upside < 45% → put strike × 0.85 (closer), call strike × 1.15 (farther)  
- Kronos upside > 55% → inverse

---

## ⚠️ Disclaimer

This dashboard is for educational and research purposes only. It does not constitute financial advice. Options trading involves significant risk and is not suitable for all investors. Always conduct your own research and consult a financial advisor before making trading decisions.

---

## 📜 License

MIT License — free to use, modify, and deploy.
