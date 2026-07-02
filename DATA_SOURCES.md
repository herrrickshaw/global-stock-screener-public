# Data sources — used capability & unused potential

Every external data source the system references, what we **currently use**, and the
**unused potential** each holds for future analysis. All sources are free/official.

> ⚠️ Educational / research only. Respect each provider's Terms of Service & rate limits.

---

## Price & volume (OHLCV)

### 1. NSE + BSE bhavcopy — `bhavcopy_history.py`, `bhavcopy_store.py`
- **Source:** `nse` lib (NSE EOD) + `bseindia.com/download/BhavCopy/Equity/` (BSE CSV)
- **Currently used:** daily OHLCV for the full IN equity universe → LTM/STM seeds; the official, rate-limit-free EOD path.
- **Unused potential:** the *full* bhavcopy carries **delivery quantity & delivery %**, number of trades, VWAP, series/segment, and 52-week high/low fields — none consumed yet. Delivery% is a strong accumulation/distribution signal; deliverable-turnover would sharpen the liquidity tiers and feed a "smart-money" screen.

### 2. Yahoo Finance — `data_sources.py` (`yfinance`)
- **Currently used:** daily OHLC for the 19 non-IN markets (close/high/low/volume) → returns, SMA, RSI.
- **Unused potential:** **splits & dividends** (total-return series), **intraday/60-min** bars, **sector/industry, market cap, shares outstanding, beta**, and basic **fundamentals (trailing P/E, forward P/E, margins)** are all in the yfinance payload but dropped — could give dividend-adjusted performance, cross-market sector rotation, and P/E without SEC/screener.

### 3. Stooq — `data_sources.py` (fallback)
- **Currently used:** OHLC fallback when Yahoo rate-limits.
- **Unused potential:** **index & FX history, commodities**, and long (20-yr+) daily histories — a free path to deepen the LTM well beyond the current ~1yr and to add macro/FX context.

---

## Fundamentals

### 4. SEC EDGAR (XBRL companyfacts) — `sec_fundamentals.py`
- **Source:** `data.sec.gov/api/xbrl/companyfacts`, `sec.gov/files/company_tickers*.json`
- **Currently used (US):** net income, revenue, ROE/ROA, CFO, EBIT, current ratio, debt, margins, asset turnover, shares, EPS growth → Piotroski, ROCE, Magic Formula, decision engine.
- **Unused potential:** the **full XBRL fact set** — segment revenue, R&D, capex history, inventory/receivables/payables (full CCC), share buybacks, deferred revenue, per-quarter series, **8-K/10-K filing dates & text** — enables quarterly-trend models, accrual/earnings-quality flags, and event studies around filings.

### 5. screener.in — public screens — `screener_in.py`, `public_screens.py`
- **Currently used (IN):** 26 curated popular screens (Bull Cartel, Piotroski, Magic Formula, CCC, …) as validation ground-truth + the CCC metric.
- **Unused potential:** each screen page exposes **ROCE/ROE/OPM/sales & profit growth/P/E/dividend columns** we only partially parse; the **query definitions** themselves (login) encode analyst logic we could replicate; peer/sector pages give **sector medians** for relative valuation.

### 5b. Trendlyne — authenticated (`trendlyne_auth.py`)
- **Currently used (IN):** per-stock P/E, PB, ROE, ROA, debt/equity, dividend yield,
  OPM, revenue/profit growth, Piotroski, market cap — the primary IN fundamentals feed
  (higher throughput than screener.in's export). Creds via `.env` only.
- **Unused potential:** SWOT/DVM scores, shareholding & pledge trends, forecasts,
  results calendar, superstar-investor holdings.

### 6. screener.in — authenticated export — `screener_in_auth.py`
- **Currently used (IN):** per-company Excel export → net income, ROE, ROCE, debt, growth for the promoted watchlist.
- **Unused potential:** the export's **10-year annual + quarterly sheets** (balance sheet, cash flow, ratios, shareholding pattern incl. **promoter holding & pledge**, FII/DII holding) — the richest untapped IN fundamentals source; would power promoter-pledge red-flags, holding-trend and 10-yr compounding screens.

### 7. Damodaran datasets — `reference_data.py`
- **Currently used:** company list + reference tables (beta, margins, P/E, ROE) as reference seeds.
- **Unused potential:** **industry cost of capital / ERP / country risk premia, tax rates, revenue multiples** — a ready-made framework for DCF/intrinsic-value scoring and country-risk-adjusted screening.

### 8. Fama-French — `reference_data.py`
- **Currently used:** FF3 factor table stored as a reference seed (not yet in signals).
- **Unused potential:** **factor regressions** (market/size/value, plus momentum/quality if FF5 added) → factor-exposure attribution per stock and factor-tilted portfolio construction.

---

## Universes & reference

### 9. Exchange universe feeds — `universe_sources.py`
- **Sources:** JPX (`jpx.co.jp`), KRX (`kind.krx.co.kr`), SGX (`api.sgx.com`), Euronext (`live.euronext.com`), Eastmoney (`push2.eastmoney.com`, China).
- **Currently used:** the list of tradeable tickers per market (to fetch OHLCV).
- **Unused potential:** these endpoints also carry **sector/industry classification, listing date, board/segment, free-float, index membership, and (SGX/Eastmoney) live quotes & fundamentals** — sector tagging alone would unlock cross-market sector-rotation and the doc's sector-boom detector for all 20 markets.

### 10. `investpy` — referenced in `universe_sources.py`
- **Currently used:** supplementary universe/listing lookups.
- **Unused potential:** **economic calendar, bonds, ETFs, indices, commodities, country macro** — a macro overlay for regime detection.

### 11. nsepython — `market_calendar.py`, `screener_in.py`
- **Currently used:** NSE trading-holiday list (calendar) + CCC screen helper.
- **Unused potential:** **live quotes, option chain / India VIX, bulk & block deals, FII/DII cash-segment flows, corporate actions/announcements, index constituents** — VIX for regime, bulk/block + FII flows for the doc's "smart-money" and sector-flow signals.

---

## News / sentiment (currently archived)

### 12. RSS + VADER — `archive/sentiment-and-research-experiments`
- **Currently used:** none on `main` (moved to an archive branch).
- **Unused potential:** headline **sentiment per ticker** (token-free) as a discovery/timing overlay; the doc's Layer-2 "policy score" (Budget/PLI mentions) could be built from the same feed.

---

## Summary — biggest untapped wins

| Rank | Opportunity | Source(s) | Enables |
|------|-------------|-----------|---------|
| 1 | **Delivery % / deliverable turnover** | bhavcopy full (1) | accumulation signal, sharper liquidity |
| 2 | **10-yr + quarterly fundamentals, promoter pledge** | screener.in export (6) | compounding & red-flag screens |
| 3 | **Sector/industry classification (all 20 mkts)** | universes (9), yfinance (2) | sector-rotation, boom-phase detector |
| 4 | **FII/DII flows, bulk/block deals, VIX** | nsepython (11) | smart-money & regime signals |
| 5 | **Dividend/split-adjusted total return** | yfinance (2), stooq (3) | correct long-horizon performance |
| 6 | **Factor regressions (FF)** | Fama-French (8) | factor attribution & tilts |
| 7 | **Deeper history (20-yr)** | stooq (3), SEC (4) | fills the 5-yr LTM, robust backtests |

See `DATA_AND_MODULES.md` (module guide) and `BLOOMBERG_SOURCES.md` (Bloomberg-equivalent map).
