# 📈 Global Stock Screener

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herrrickshaw/global-stock-screener/blob/main/colab_quickstart.ipynb)

A 20-market equity screener that **ships with ~1 year of cached price data** (≈40,000 symbols across the world's major exchanges), so you can clone and get results in minutes — no API keys, no LLM tokens.

> ⚠️ **Educational / research only. NOT investment advice.** Screener output is a mechanical filter, not a buy/sell signal. Past performance does not guarantee future returns. Consult a SEBI-registered (or your jurisdiction's) advisor.

---

## ▶️ Run in Google Colab (no setup)

```python
!pip install -q lmdb xlrd yfinance nsepython bseindia nse feedparser vaderSentiment investpy
!git lfs install && git clone --depth 1 https://github.com/herrrickshaw/global-stock-screener.git
%cd global-stock-screener && git lfs pull
!python run.py --strategy darvas --market IN          # → prints results + saves CSV
```

Or open **`colab_quickstart.ipynb`** and Runtime → Run all.

## ▶️ Run locally

```bash
git lfs install && git clone https://github.com/herrrickshaw/global-stock-screener.git
cd global-stock-screener && git lfs pull
pip install -r requirements.txt
python run.py                         # global momentum leaderboard
python run.py --strategy garp --market US --top 25
python run.py --brief                 # full Daily Market Brief → results/daily_brief.html
python run.py --list                  # all strategies & markets
```

`run.py` auto-bootstraps the cached data into a fast local store on first use, then
prints results and saves them to `results/`.

---

## What you get

- **20 markets:** IN, US, JP, KR, CN, HK, TW, CA, AU, UK, DE, SA, BR, CH, ZA, SE, FI, DK, SG, Euronext
- **11 strategies:** `piotroski, coffee_can, magic_formula, bluest_blue_chips, debt_reduction, dividend_yield, golden_crossover, loss_to_profit, garp, darvas, cash_conversion_cycle`
- **Liquidity tiers** (High/Medium/Low) on every result + a fast liquidity pre-filter
- **Cash Conversion Cycle** for India (screener.in) and US (SEC EDGAR)
- **Custom screens** on your own parameters (returns, RSI, drawdown, volatility, CCC, …)
- **Global momentum leaderboard** + **20-market 5-year scoreboard**
- **Token-free daily mailer** (`send_mailer.py` + `daily_pipeline.sh`)

```python
import os; os.environ["BHAV_CACHE"] = "/content/cache"
import screener_kit as kit
kit.bootstrap()
kit.screen("darvas", "IN", top=15, min_turnover_usd=1_000_000)
kit.custom_screen({"ccc": ("<", 0), "above_200dma": ("==", True)}, "IN")
```

## Data sources (all free / official)

Exchange bhavcopy (NSE/BSE), SEC EDGAR (US fundamentals), JPX/KRX/SGX/Euronext/
Eastmoney (universes), Yahoo→Stooq (prices, with fallback), Damodaran & Fama-French
(reference), screener.in (India CCC), RSS + VADER (news sentiment — not an LLM).

See **`DATA_AND_MODULES.md`** for the full module guide and **`BLOOMBERG_SOURCES.md`**
for the Bloomberg-equivalent public-source map.
