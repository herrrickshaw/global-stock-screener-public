# Global Stock Screener

A 20-market equity screener that ships with roughly one year of cached price data (about 40,000 symbols
across the major exchanges), so a fresh clone can produce results without API keys or LLM tokens. It runs
11 screening strategies, tags every result with a liquidity tier, and can build a Daily Market Brief (HTML)
that a token-free mailer sends. Outputs are CSV files and HTML under `results/`.

> Status: research-only, public copy of a private repository. Educational use only; not investment advice.
> Screener output is a mechanical filter, not a buy/sell signal. Left out of this copy and its history:
> screener.in data (the 26 public-screen exports in `cache_seed/public_screens/`,
> `cache_seed/india_ccc_screen.parquet` and the India fundamentals cache `cache_seed/fundamentals/IN.parquet`),
> which may not be redistributed, and an unrelated automation export. The collectors that fetch screener.in
> data are included; they need your own access and credentials, and the India screens run without those
> files until you fetch your own.

New to the repo? [How to use this repo with AI (Claude, Gemini, ChatGPT)](HOW_TO_USE_WITH_AI.md).
Colab: [`colab_quickstart.ipynb`](colab_quickstart.ipynb) (Runtime -> Run all).

![Architecture](architecture.svg)

## What it does
- Screens **20 markets**: IN, US, JP, KR, CN, HK, TW, CA, AU, UK, DE, SA, BR, CH, ZA, SE, FI, DK, SG, EU (Euronext).
- Runs **11 strategies**: `piotroski, coffee_can, magic_formula, bluest_blue_chips, debt_reduction,
  dividend_yield, golden_crossover, loss_to_profit, garp, darvas, cash_conversion_cycle`.
- Adds **liquidity tiers** (High/Medium/Low, per-market thresholds on 20-day median USD turnover) to every
  result, plus a fast liquidity pre-filter.
- Computes **Cash Conversion Cycle** for India (screener.in) and the US (SEC EDGAR).
- Supports **custom screens** on your own rules (returns, RSI, drawdown, volatility, CCC, ...), a global
  momentum leaderboard and a 20-market 5-year scoreboard.
- Provides a **token-free daily mailer** (`send_mailer.py` + `daily_pipeline.sh`) built from the cached data.

## Quick start
```bash
git lfs install && git clone https://github.com/herrrickshaw/global-stock-screener.git
cd global-stock-screener && git lfs pull        # price seeds are Git LFS objects
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --strategy darvas --market IN     # prints results and saves a CSV in results/
```
More invocations:
```bash
python run.py                                   # global momentum leaderboard
python run.py --strategy garp --market US --top 25
python run.py --brief                           # Daily Market Brief -> results/daily_brief.html
python run.py --list                            # all strategies and markets
python preflight.py                             # check LFS data, packages, serving views (--ensure to fix)
```
From Python:
```python
import screener_kit as kit
kit.bootstrap()
kit.screen("darvas", "IN", top=15, min_turnover_usd=1_000_000)
kit.custom_screen({"ccc": ("<", 0), "above_200dma": ("==", True)}, "IN")
```

## How it works
`run.py` bootstraps the committed seeds into a local LMDB store on first use, then screens.

| Stage | Module | Output |
|---|---|---|
| Ingest India EOD | `bhavcopy_history.py`, `bhavcopy_store.py` (LMDB) | per-symbol OHLCV |
| Ingest other markets | `universe_sources.py`, `fetch_market_ohlc.py`, `data_sources.py` (Yahoo, Stooq fallback) | `cache_seed/*.parquet` |
| Shared frame helpers | `frames.py` | long/dict conversions, compact zstd parquet |
| Facade | `screener_kit.py` | `bootstrap / load / get / update / screen / custom_screen` |
| Strategies | `strategies/` (`base.py` + one module per strategy) | `Result` rows (pass, score, metrics) |
| Liquidity | `liquidity.py` | USD turnover index and High/Medium/Low tiers |
| Custom rules | `custom_screener.py` | ranked frame for rules like `{"rsi14": ("<", 35)}` |
| Leaderboards | `run_global_analysis.py`, `market_performance.py` | momentum / 5-year scoreboards |
| Brief and mail | `build_mailer.py`, `send_mailer.py`, `daily_pipeline.sh` | HTML brief, SMTP send |

Design notes:
- Every strategy module exposes `META` and `screen(StockData) -> Result | None`; `needs` says whether it
  uses price only or fundamentals, so runners can skip strategies lacking data.
- The Darvas strategy flags a breakout within 10% of the 52-week high on at least 1.5x the prior 20-day
  average volume; the 20-day average excludes the current bar.
- `query.py` (DuckDB) and `partition_ltm.py` (Hive `market=/year=` partitioning) provide SQL over the seeds.

## Repository layout
| Path | Purpose |
|---|---|
| `run.py`, `screener_kit.py`, `preflight.py` | CLI entry, Python facade, readiness check |
| `strategies/` | 11 strategy modules and the shared `StockData`/`Result` contract |
| `custom_screener.py`, `liquidity.py`, `frames.py` | custom rules, liquidity tiers, OHLCV frame helpers |
| `bhavcopy_*.py`, `data_sources.py`, `universe_sources.py`, `sec_fundamentals.py`, `screener_in.py` | data collectors |
| `build_mailer.py`, `send_mailer.py`, `daily_pipeline.sh`, `*.plist` | daily brief and launchd/cron schedule |
| `src/`, `api.py`, `serving_layer.py` | newer package layout and optional FastAPI layer |
| `tests/` | pytest suite (synthetic data, no network) |
| `cache_seed/`, `reference_seed/` | committed seeds (Git LFS), architecture.svg |
| `*.md` | design and operations notes; see Documentation below |

## Data sources
| Source | What is taken | Access route | Licence / caveat |
|---|---|---|---|
| NSE/BSE bhavcopy | India EOD prices | official files | exchange terms apply |
| SEC EDGAR | US fundamentals | official API | public |
| JPX, KRX, SGX, Euronext, Eastmoney | listed universes | public lists | check each site's terms |
| Yahoo Finance, Stooq | global prices (Yahoo, then Stooq fallback) | `yfinance`, HTTP | unofficial; personal/research use |
| Damodaran, Fama-French | reference datasets | download | academic sources |
| screener.in | India CCC and fundamentals | your own login | not redistributable; excluded from this copy |
| RSS feeds + VADER | news sentiment (not an LLM) | `feedparser` | per-publisher terms |

## Configuration
- `BHAV_CACHE`: directory for the local store (default `~/Downloads/data/bhavcopy_cache`).
- screener.in credentials via environment variables, see `SCREENER_ACCESS.md`. The mailer reads `GMAIL_USER`, `GMAIL_APP_PASSWORD` and
  `MAIL_TO` (defaults to `GMAIL_USER`). No secret is committed; never commit yours.

## Testing
```bash
pip install pytest && pytest -q        # 159 passed, 1 skipped; no network
ruff check . && black --check .        # as run in CI
```
`tests/test_refactor_characterisation.py` pins strategy, custom-screen, liquidity and frame outputs to a
golden snapshot (`tests/fixtures/golden_screening.json`) built from synthetic data.

## Documentation
| Doc | Contents |
|---|---|
| `FEATURES.md`, `HOW_IT_WORKS.md` | capability catalogue; bootstrap -> screen -> refresh walk-through |
| `ARCHITECTURE.md`, `DATA_SCHEMA_AND_ARCHITECTURE.md` | layers, flows, schemas |
| `DATA_AND_MODULES.md`, `DATA_SOURCES.md`, `EXCHANGE_COVERAGE.md` | module guide, sources, the 20 markets |
| `SCREENER_ACCESS.md`, `SECURITY.md` | screener.in credentials; integrity checks |
| `TOGAF.md` / `governance.py`, `AGILE.md` / `backlog.py` | EA alignment and WSJF backlog (executable) |
| `DEPENDENCIES.md` (`dep_map.py`), `BRANCHES.md` | code/data blast radius; branch map |

## Limitations and known issues
- Mechanical screens only; past performance does not guarantee future returns.
- The repository is large (about 240 top-level files, many legacy scripts and status notes); several
  collectors (`full_*_market_scan.py`, `quarterly_*`, `phased_expansion_*`) duplicate logic and are
  candidates for consolidation.
- Price seeds need Git LFS; without `git lfs pull` the seeds are pointer files.
- Yahoo/Stooq data is unofficial and can be adjusted or missing for recent splits; verify before relying on it.
- The mailer's send path has no already-sent guard; do not run it twice for the same session.
- Freshness should be judged by trading session, not file modification time.

## Related repositories
- Source repository (private original): `herrrickshaw/global-stock-screener`.

## Licence
No licence file yet; all rights reserved.
