# Features — capability catalog

Everything the system does, grouped, with the module and a one-line usage. See
`AGILE.md` for the WSJF backlog and `TOGAF.md` for the architecture principles.

> Educational / research only. NOT investment advice.

## Data platform
| Capability | Module | Run |
|-----------|--------|-----|
| Two-tier memory (5y LTM / 1y STM) | `market_memory` | `python market_memory.py --status` |
| Linkage: manifest, memoised load, incremental store | `datalink` | `python datalink.py --status` |
| Materialised serving views + CDC deltas | `serving_layer` | `python serving_layer.py --refresh` |
| NoSQL per-symbol store (B+tree) | `bhavcopy_store` | `python bhavcopy_store.py --build` |
| Multi-market trading calendar | `market_holidays` | `python market_holidays.py` |
| DuckDB SQL over the parquet seeds | `query` | `python query.py "SELECT ..."` |
| Year-partitioned LTM (pruned queries) | `partition_ltm` | `python partition_ltm.py` |
| Master company list (38k, 20 markets) | `export_company_list` | `python export_company_list.py` |

## Screening
| Capability | Module | Run |
|-----------|--------|-----|
| 11 built-in strategies | `strategies/` + `screener_kit` | `python run.py --strategy darvas --market IN` |
| Screener.in price-metric formulas on the DB | `screen_metrics` | `python screen_metrics.py --market IN` |
| Fundamental formulas (Piotroski/ROCE/…) | `fundamental_metrics` | `python fundamental_metrics.py --market US` |
| Validation vs Screener.in popular screens | `validation` + `public_screens` | `python validation.py --market IN` |
| **Delivery-% accumulation signal** | `delivery_signal` | `python delivery_signal.py --fetch` |
| **Sector classification + rotation** | `sector_map` | `python sector_map.py --fetch --market IN` |

## ML & signals
| Capability | Module | Run |
|-----------|--------|-----|
| Supervised forward-return classifier (L1) | `ml_supervised` | `python ml_supervised.py --market IN --train` |
| Unsupervised discovery + RL correction (L2) | `auto_screener` | `python auto_screener.py --market IN` |
| RL trade agent (L3) | `rl_trader` | `python rl_trader.py --market IN --train` |
| **FII/DII + India-VIX regime** | `flows_regime` | `python flows_regime.py --fetch` |

## Investor tools
| Capability | Module | Run |
|-----------|--------|-----|
| Upload holdings/tax → Buy/Hold/Sell | `portfolio_evaluator` | `python portfolio_evaluator.py --file holdings.xlsx` |
| Multi-factor decision engine (+ IMPAIRED, DVM, pledge, accumulation) | `decision_engine` | (used by the evaluator) |
| Company-name → ticker matching | `name_matcher` | `python name_matcher.py "Reliance Industries"` |
| Delisted/renamed ticker aliases | `ticker_aliases` | `python ticker_aliases.py ZOMATO` |
| **Watchlist decision-delta (what changed)** | `watchlist_delta` | `python watchlist_delta.py --snapshot --market IN` |

## Data sources (auth via `.env` only)
US = SEC EDGAR · IN = **Trendlyne** (P/E, ROE, DVM, pledge) → screener.in → yfinance ·
others = yfinance. See `trendlyne_auth`, `sec_fundamentals`, `screener_in*`, `DATA_SOURCES.md`.

## Ops, governance & process
| Capability | Module | Run |
|-----------|--------|-----|
| Autonomous daily pipeline | `daily_memory.sh` | cron/launchd |
| **Daily market brief** | `daily_brief` | `python daily_brief.py --mail` |
| CRUD pipeline (filter-driven tiering) | `pipeline` | `python pipeline.py --sync` |
| Initial-conditions bootstrap/health | `preflight` | `python preflight.py --ensure` |
| Tamper-evidence (SHA-256 manifest) | `integrity` | `python integrity.py --verify` |
| Interdependency map (code + data) | `dep_map` | `python dep_map.py` |
| TOGAF governance conformance | `governance` | `python governance.py` |
| SAFe WSJF backlog / PI planning | `backlog` | `python backlog.py --pi` |

**Delivered this program (bold above):** delivery signal + decision factor, sector
rotation + boom-gating, Trendlyne source + DVM/pledge, ticker aliases, P/E clamp,
daily brief, FII/DII+VIX regime, watchlist delta — 11 features, all behind the
protected-PR flow (lint + tests + integrity + governance green).
