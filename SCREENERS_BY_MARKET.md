# Screeners by Market — Reference

Which stock screeners to use in each market, synthesized from the repo's docs and
code across `main`, `karz`, and the archived scan scripts. Educational/research
only — **not investment advice.**

## How to read this

The evidence comes in **three layers** that answer different questions. Don't
conflate them:

| Layer | Source | Answers |
|---|---|---|
| **Implemented** | `full_*_market_scan.py` headers/code (archive/single-market-scan-scripts) | What each market's scanner *runs today* |
| **Documented** | `README.md`, `FEATURES.md`, `DATA_SOURCES.md` (main) | What's *possible* per market (gated by data) |
| **Backtested** | `BACKTEST_FINDINGS.md` (karz) | What actually had *edge* out-of-sample |

**Structural fact that drives everything:** only **US** (SEC EDGAR XBRL) and
**India** (Trendlyne → screener.in → NSE/BSE bhavcopy) have a real fundamentals
feed. Every other market is **yfinance-only**, so fundamental screeners there are
best-effort (Europe is the exception — its scan computes a real P/E from yfinance).

---

## 1. The screeners

| Screener | Idea | Data | Runs in |
|---|---|---|---|
| **Darvas Box** | Breakout above a consolidation box near 52-wk high | OHLCV only | All markets (universal) |
| **Golden Crossover** | 50-DMA crosses above 200-DMA | OHLCV only | All markets; hard-required in bull-regime auto-screener |
| **Piotroski F-Score** | 9-point financial-health check (≥7 = strong) | Fundamentals (simplified 0–6 variant runs on OHLCV) | US, IN, EU, JP, KR, SGX |
| **Coffee Can** | Buy-and-hold quality: Rev CAGR >10%, ROCE/ROE >15%, D/E <1, +NI, +FCF, MCap floor | Fundamentals | US, IN, EU, JP, KR |
| **Magic Formula** | Greenblatt: high ROIC (>25%) + earnings yield (>15%) | Fundamentals (quarterly) | **US + IN only** |
| **Bull Cartel** | Momentum+quality: QoQ sales >15%, profit >20% | Fundamentals (quarterly) | **US + IN only** |
| **PEGY** | PEG + dividend yield valuation | Fundamentals | EU, SGX |
| **Cash Conversion Cycle** | Working-capital efficiency | Fundamentals | US (SEC), IN (screener.in) |
| **Pegu Score / Sarvas Scan** | Composite score + BUY scan | Fundamentals (R pipeline) | **India only** (separate R codebase) |
| **ML L1/L2/L3** | Supervised classifier / unsupervised discovery / RL agent | OHLCV features (+fundamentals if present) | IN + US demonstrated |

Common thresholds: **Darvas** confirm=3, current bar excluded; **Piotroski** ≥7
strong; **"Triple Hit" = Darvas BREAKOUT_BUY + Piotroski ≥7 + Coffee Can PASS**;
Darvas/Buffett 0–7 scale **BUY ≥5 · WATCH ≥3**. Fundamentals run only on Darvas
breakout candidates (API-frugal). Banks/NBFCs are excluded from ROIC/ROCE in India.

---

## 2. What each market's scanner runs (implemented)

| Market | Screeners live in the scan | Recommended combo | MCap floor |
|---|---|---|---|
| **India (NSE/BSE)** | Darvas · Golden Cross · Piotroski · Coffee Can · Magic Formula · Bull Cartel — **all 6** | **Triple Hit** + **Multi-Screen** (≥3 of 6) | ₹500 Cr |
| **US (NYSE/NASDAQ)** | Darvas · Golden Cross · Piotroski · Coffee Can · Magic Formula · Bull Cartel — **all 6** | Triple Hit | $1B |
| **Europe** | Darvas · Piotroski · Coffee Can · **PEGY** | Triple Hit | €1B |
| **Japan (TSE)** | Darvas · Piotroski · Coffee Can | Triple Hit | ¥100B |
| **Korea (KRX)** | Darvas · Piotroski · Coffee Can | Triple Hit | ₩100B |
| **Singapore (SGX)** | Darvas · Piotroski · **PEGY** | Value / Momentum / Income workflows | S$500M |
| **China, HK, TW, CA, AU, SA, ZA, BR, TH, ID, MY, MX, IL** | Darvas · Golden Cross (price/momentum only — yfinance has no usable fundamentals) | — | — |
| **India (R side)** | **Pegu Score · Sarvas Scan** | — | — |

---

## 3. What actually had edge (the only real backtest)

`BACKTEST_FINDINGS.md` (10-year LTM, 2016→2026, 21-day forward horizon, 5 markets,
survivorship-caveated) tested **technical screens only** — mean over market, in
percentage points of edge:

- **Mean-reversion (`rsi_oversold`) wins in US, Korea, China, Japan** — #1 in all
  four, positive in all five (KR +2.0pp, US +1.9pp, JP win-rate 0.51, CN +1.1pp).
- **India is the momentum exception** — `near_high` (+1.3pp) and `momentum`
  (+1.2pp) win there, but are **negative** in US/KR/CN.
- **The Golden Cross was the worst screen in every market** — negative/near-zero
  edge everywhere; the classic 50/200 cross added no out-of-sample value.
- Fundamentals: over 10y, **US mildly penalizes leverage** (D/E vs return
  ρ = −0.135, p = 0.027); liquidity + small size dominate.

**Important:** none of the *fundamental* screeners (Piotroski, Coffee Can, Magic
Formula, Bull Cartel, Pegu, Sarvas) are backtested by name anywhere in the repo —
they are implemented and thresholded, but their per-market performance is asserted,
not measured. The "expansion screener" (capex/debt 8-D→11-D model) is US-only and
largely **synthetic/projected** (own F1 = 0.0000 on synthetic data) — a prototype,
not evidence.

---

## 4. Recommendation per market

- **India** — richest coverage and the one market where both fundamentals *and*
  momentum are justified (momentum edge is backtest-confirmed). Use the full
  6-screener **Multi-Screen** + the R **Pegu/Sarvas** overlay.
- **US** — full fundamental stack works (SEC EDGAR). But for *timing*, the backtest
  favors **buying oversold**, not breakouts/golden-crosses — consider an
  RSI-oversold entry filter on top of the Triple Hit.
- **Japan / Korea / China** — thin fundamentals (yfinance-only); backtest says
  **mean-reversion, not momentum**. Use Darvas+Piotroski for *selection*, favor
  oversold *entries*, de-emphasize Golden Cross.
- **Europe** — the only non-US/IN market with a usable P/E (the EU scan computes it
  from yfinance), so Piotroski + Coffee Can + PEGY are legitimate; not in the
  backtest, so treat momentum signals cautiously.
- **Everywhere** — the Golden Cross is documented but backtested as the weakest
  screen; don't weight it heavily.

---

## 5. Session verification (2026-07-09) — a real, point-in-time re-check

I re-ran a **Darvas-breakout** backtest against the stored 10-year daily OHLCV
(`cache_seed/ltm/<MKT>.parquet`): close crossing above its trailing 60-day box top,
21-day forward return vs the market's all-days mean (edge in pp), ~400-ticker
sample/market, point-in-time (no look-ahead). Result **matches `BACKTEST_FINDINGS.md`**:

| Market | Breakout edge | Read |
|---|---:|---|
| India | **+1.6 pp** | breakout works (momentum-friendly) ✅ |
| Europe | **+0.4 pp** | mildly positive |
| Korea | +0.0 pp | neutral |
| Japan | −0.2 pp | breakout hurts (mean-reversion) |
| China | −0.7 pp | breakout hurts |
| US | −6.5 pp | breakout hurts most |

(Absolute return levels are survivorship-inflated — LTM holds only currently-listed
names — so read the **sign and ordering**, not the magnitude.)

**The fundamental screeners could NOT be backtested:** the repo stores only
*current-snapshot* fundamentals (`cache_seed/fundamentals/{US,IN}.parquet`), not
point-in-time quarterly financials, so a Piotroski/Coffee-Can/Magic-Formula backtest
would require look-ahead — the exact flaw the "expansion screener" docs fell into.
Closing that gap needs a historical quarterly-financials collection first (US via
SEC EDGAR XBRL is the most tractable). Until then, the fundamental screeners remain
implemented-and-thresholded but **empirically unvalidated by name.**

---

*Sources (via `git show <ref>:<file>`): `main` — README.md, FEATURES.md,
HOW_IT_WORKS.md, AUTO_SCREENER.md, DATA_SOURCES.md, EXCHANGE_COVERAGE.md; root
~/CLAUDE.md (Darvas/Buffett + Piotroski thresholds, Pegu/Sarvas R pipeline); `karz`
— BACKTEST_FINDINGS.md (the only real backtest), PRICE_CORRELATION_FINDINGS.md,
F1_TUNING_METHODOLOGY_REPORT.md; `archive/single-market-scan-scripts` —
full_{us,indian,japan,korea,european}_market_scan.py headers, SGX_FULL_SCAN_README.md.*
