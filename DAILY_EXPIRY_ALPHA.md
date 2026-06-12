# Daily-Expiry Alpha — Selling BTC 1-Day Options on Delta Exchange

**Settlement: every day 12:00 UTC = 5:30 PM IST.** 998 tradable days (Sep 2023 – Jun 2026) priced with Black-Scholes at each day's *real* Deribit DVOL, settled at the actual next-noon price built from 8.8 years of hourly candles. All P&L is % of spot per trade. Every claim below survived a 10,000-sample bootstrap unless marked otherwise.

---

## 0) The honest headline

Daily expiry changes the physics. At 7 days you are a **vega** trader (fear premium mean-reverts); at 1 day you are a **gamma** trader (you win unless today is the day something happens). Three things emerged that the weekly study could never see, because they live *inside* the week and *inside* the day:

1. **The Saturday lull is the single largest, cleanest edge found in this entire project** — bigger per unit of risk than the weekly "persistent fear" trade.
2. **WHEN you enter matters as much as WHAT you sell.** The 24h window after expiry is back-loaded: the US session (first 6 hours, 5:30 PM–11:30 PM IST) carries 31% of the day's variance. Entering later sells the quiet hours and skips the loud ones.
3. **The weekly PRIME "persistent fear" trade does NOT transfer to 1 day** (+0.01%/d, not significant). Fear regimes whipsaw daily; the weekly edge needed 7 days of mean-reversion to cash the premium. Don't sell dailies into a drawdown expecting the weekly magic.

---

## 1) Baselines — sell every single day, no filter

| Instrument (1DTE, entry 12:00 UTC) | EV/day | Win % | Worst day | CVaR-5% | Avg premium |
|---|---|---|---|---|---|
| 25Δ put | +0.113% | 86.3% | −16.1% | −3.4% | 0.40% |
| 25Δ call | +0.095% | 85.4% | −10.3% | −3.2% | 0.39% |
| 10Δ put | +0.016% | 94.3% | −14.6% | −2.0% | — |
| 10Δ call | +0.027% | 94.0% | −8.8% | −1.8% | — |
| 25Δ strangle | +0.208% | 76.8% | −15.6% | −4.1% | 0.79% |
| **ATM straddle** | **+0.400%** | 70.1% | −16.1% | −4.6% | 2.12% |

Read this carefully:

- **10Δ "safe" wings are a trap at 1 day.** They win 94% of the time and earn almost nothing (+0.02%/d) — after fees they are roughly break-even, and the worst day is still −15%. All risk, no pay. At daily expiry, sell **closer to the money or don't sell at all**.
- The ATM straddle collects the most VRP per day because the variance risk premium is fractionally largest at the shortest tenor — but it eats the full tail.
- The put/call EV gap (2.2× at weekly) shrinks to 1.2× at daily. Skew matters less when there's only one day for a crash to happen.

## 2) THE find: the weekend lull

Entering **Saturday 12:00 UTC (Sat 5:30 PM IST), expiring Sunday noon**:

| Trade (Sat entry) | EV/day | Win % | Worst | p-value |
|---|---|---|---|---|
| **ATM straddle** | **+1.202%** | **92.3%** | **−3.2%** | 0.0000 |
| 25Δ strangle | +0.656% | 93.0% | −2.8% | 0.0000 |
| 25Δ put | +0.328% | 96.5% | −3.2% | 0.0002 |
| Friday entry straddle (Fri→Sat) | +0.654% | 73.9% | −5.7% | 0.0330 |
| Fri+Sat combined straddle | +0.928% | — | — | 0.0000 |

**Why it works (the structural proof, 8.8 years of hourly data):** the Sat-noon→Sun-noon window realizes only **33–45% of weekday volatility** every single year since 2022. No US session, no macro prints, no ETF flows — the market is literally closed everywhere that matters. An option priced anywhere near weekday vol for that window is massively overpriced.

**Why nobody arbitrages it away fully:** market makers do discount weekend IV — but not by the ~75% that the *variance* ratio (0.45² ≈ 0.20) justifies. Cutting IV that much looks insane on a screen, so the discount stops halfway.

**The honest decay warning:** Sat straddle EV by year: 2023 +1.50% → 2024 +1.43% → 2025 +1.18% → **2026 +0.55%** (and the realized Sat/weekday ratio rose to 0.71 this year). The edge is real, structural, and *shrinking* — either being arbitraged or 2026 weekends are genuinely busier. Trade it at half the size your backtest courage suggests, and watch the dashboard's live IV before pulling the trigger.

**The mirror anti-signal:** **Monday entry is the worst day of the week** (strangle −0.095%/d, p=0.006). Mon-noon→Tue-noon reprices the whole weekend's news through a full US session. Skip Mondays entirely. Sunday entry carries the single worst day in the sample (−16.1% on Aug 4 2024, the yen-carry crash weekend) — the weekend lull does NOT extend to Sunday→Monday.

## 3) The second find: entry hour beats entry signal

Same straddle, same noon expiry, different entry hour (priced at DVOL, settled at real prices):

| Entry (UTC / IST) | Hours left | EV/trade | Win % | Worst | EV per hour at risk |
|---|---|---|---|---|---|
| 12:00 / 5:30 PM | 24h | +0.400% | 70.1% | −16.1% | 0.017%/h |
| **18:00 / 11:30 PM** | 18h | **+0.560%** | 74.4% | −13.2% | 0.031%/h |
| 00:00 / 5:30 AM | 12h | +0.483% | 77.7% | −9.7% | 0.040%/h |
| **06:00 / 11:30 AM** | 6h | +0.398% | 81.5% | **−4.3%** | **0.066%/h** |
| 09:00 / 2:30 PM | 3h | +0.290% | 82.8% | −4.7% | 0.097%/h |

The mechanism (8.8-year hourly variance map): the day's variance is front-loaded into the US session right after listing — 31% burns off in the first 6 hours, 55% by midnight UTC. **An entry at 11:30 PM IST collects MORE total premium-vs-realized than the full 24h hold (+0.56% vs +0.40%) while skipping the most explosive hours.** And the 6-hour morning entry (11:30 AM IST → expiry) earns the same as the full day with one-quarter of the worst loss.

Caveat: this assumes options are priced in calendar time off a flat IV. To the extent market makers price "event time" (cheaper overnight IV), the late-entry edge shrinks — check the live chain's actual IV at your entry hour; if overnight IV is already crushed 40%+ below day IV, the edge is priced in.

## 4) Which weekly regimes survive at 1 day — and which die

| Weekly regime | At 7 days | At 1 day | Verdict |
|---|---|---|---|
| PRIME persistent fear (dd90<−15, DVOL>50): put | +0.95%/wk p=0.0000 | +0.01%/d p=0.11 | **DIES.** Weekly-only trade. |
| Uptrend (>MA100) + DVOL>50: put | +0.81%/wk | +0.23%/d p=0.004 | **Survives.** Best daily put filter. |
| Uptrend alone: put | +0.50%/wk | +0.21%/d p=0.006 | Survives. |
| Funding 7d < 0: put | +0.81%/wk | +0.26%/d p=0.034 | Survives (92% win, worst −3.5%). |
| RSI<30: put (anti-signal) | −0.78%/wk | **−0.41%/d p=0.009** | **Survives — still poison.** |
| RSI>70: call (anti-signal) | −0.62%/wk | −0.03%/d p=0.11 | Weakens; mild caution only. |
| DVOL<40: strangle (anti-signal) | −0.47%/wk | **+0.25%/d — not a loser!** | **FLIPS.** Low IV still overprices a single day. The DVOL<40 no-sell rule applies to WEEKLIES, not dailies. |
| Sell into crash day (shock) | −1.04%/wk | n too small at 1d | Keep the rule anyway. |

The DVOL<40 flip is the most practically useful: **on quiet weeks when the Seller's Compass says "stand aside" on weeklies, the daily strangle is still positive.** The variance risk premium never fully disappears at the 1-day tenor — there is always someone paying for overnight protection.

## 5) Fees — the tax that weekly sellers ignore and daily sellers can't

Delta Exchange model: taker fee = min(0.03% notional, 10% of premium) per leg, plus settlement fee on ITM legs.

| Trade | Gross EV/d | Est. fees | Net EV/d |
|---|---|---|---|
| 25Δ put | +0.113% | 0.037% | **+0.076%** (fees eat 33%!) |
| 25Δ strangle | +0.208% | 0.075% | +0.133% |
| ATM straddle | +0.400% | 0.075% | **+0.325%** |
| Sat straddle | +1.202% | 0.075% | **+1.127%** |

Rule that falls out: **at daily expiry, sell fat premium (ATM/25Δ), never thin premium (10Δ)** — fees are a fixed-ish cost, so the thinner the premium, the larger the bite. The 10Δ wings are net-negative after fees. Use limit orders (maker) to halve this.

## 6) The Daily Playbook (5:30 PM IST decision, every day)

1. **Is tomorrow's window Sat-noon→Sun-noon?** (i.e., today is Saturday) → **PRIME DAILY: sell the ATM straddle or 25Δ strangle.** Best trade of the week. Friday entry is the junior version.
2. **Is today Monday, or did BTC just crash/shock, or RSI<30?** → **STAND ASIDE.** Monday repricing + fresh panic are the two ways daily sellers die.
3. **Otherwise, directional:** uptrend above MA100 (especially with DVOL>50) or funding flushed negative → sell the 25Δ put.
4. **Otherwise, neutral:** sell the 25Δ strangle — yes, even when DVOL<40 (daily-only exception).
5. **Entry-time refinement:** patient? Wait until 11:30 PM IST — historically more EV, smaller bombs. Nervous? The 11:30 AM IST 6-hour entry earns nearly the full-day EV at a quarter of the tail. Check live IV first: if the late-session IV is already crushed, skip.
6. **Sizing:** worst day in sample was −16% of notional on a straddle (a Sunday). Size so that a −16% day is annoying, not fatal. The Aug-2024 weekend proves "weekend = safe" has one exception per cycle: **the lull trade is Saturday only, never Sunday.**

Compounding honestly: the Sat trade alone ≈ +1.1% net/week. Add ~4 weekday trades at ~+0.1–0.3% net when filters allow ≈ +0.5–1.0%/week more. That's in the same range as the weekly Compass — but with 5× more decisions, 5× more fee drag, and 5× more chances to break your own rules. Daily selling is not magic money; it's the same VRP harvested in smaller, more frequent, more disciplined bites.

## 7) Caveats (read twice)

- **Flat-DVOL pricing.** Real daily options trade on their own IV, usually below DVOL on weekends and overnight. The Sat and late-entry EVs above are **upper bounds**; the dashboard card compares live 1-day IV to DVOL so you can see how much of the edge the market has already taken.
- 998 priced days = ~142 of each weekday. The Sat result is p=0.0000 and structurally confirmed by 8.8 years of realized vol, but 2026 shows decay — respect it.
- 2026's Sat/weekday vol ratio (0.71) is the highest since 2021. If that persists, the weekend edge halves again.
- Slippage on Delta's daily book can exceed fees in thin hours; always quote-check both legs before selling a strangle.
- No 2022-style bear market in the priced sample.
- Kronos: live overlay only (no forecast archive exists to backtest) — use as a sizing tiebreaker on the directional put days, same as the weekly Compass.

*Companion studies: `SELLER_DIRECTIONAL_ALPHA.md` (weekly), `OPTION_BUYER_ALPHA.md` (buyer side), `daily_expiry_alpha.png` (chart).*
