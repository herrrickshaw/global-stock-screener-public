# Data inventory & linkage

Every data asset in the repo, how it connects, and the linkages added to cut
processing/fetching time.

## 1. Data assets

### Committed (ship with the repo, Git LFS)
| Asset | What | Size |
|-------|------|------|
| `cache_seed/cleaned_long.parquet` | India (NSE+BSE) 1-yr OHLCV seed — **STM** | ~15 MB |
| `cache_seed/cleaned_long_<MKT>.parquet` ×19 | 1-yr OHLCV seed per market — **STM** | ~125 MB |
| `cache_seed/ltm/<MKT>.parquet` | 5-yr OHLCV archive — **LTM** (accumulates over time) | grows |
| `cache_seed/liquidity_index.parquet` | per-symbol Turnover_USD + tier | ~0.6 MB |
| `cache_seed/india_ccc_screen.parquet` | India Cash-Conversion-Cycle screen snapshot | small |
| `cache_seed/market_liquidity.parquet`, `global_highlights.parquet`, `market_performance_5y.parquet` | derived reports | small |
| `cache_seed/nse_holidays.json`, `no_data_dates.json` | calendar + negative caches | tiny |
| `reference_seed/damodaran_*.parquet`, `french_ff3.parquet` | fundamentals/factor reference | ~1.8 MB |

### Runtime only (not committed — under `$BHAV_CACHE`)
| Asset | What |
|-------|------|
| `assembled_long.parquet` | raw consolidated (Date,Symbol,OHLCV,_exch) — India speed layer |
| `cleaned_long*.parquet` (live copies) | working copies the daily jobs update |
| `ohlcv.lmdb` (~464 MB) | per-symbol NoSQL store — O(1) single-symbol reads |
| `data_manifest.json`, `.store_state.json` | the linkage index + store signatures (gitignored) |

## 2. How it flows

```
universes/bhavcopy/yahoo─┐
                         ├─ market_memory.py ── LTM (5y) ──derive── STM (1y seeds)
bhavcopy_history.py ─────┘                                              │
                                                                        ▼
                                        datalink.py  ◄── manifest / memo / signatures
                                             │
                     ┌───────────────────────┼────────────────────────┐
                     ▼                        ▼                         ▼
             load_market() memo        ccc_map_cached()        build_store_incremental()
                     │                        │                         │
                     ▼                        ▼                         ▼
              screener_kit.load ──► _stocks ──► strategies/*      ohlcv.lmdb ──► kit.get()
                                        │
                                  liquidity.py (index) ──► _add_liquidity()
```

## 3. Linkages added (to reduce processing/fetching) — `datalink.py`

| Linkage | Before | After |
|---------|--------|-------|
| **Memoized loader** `load_market()` | every `screen()` re-read + re-grouped the whole parquet (~2.3 s each); 11 screeners ≈ 25 s of pure loading | first load ~2.3 s, every repeat **0.1 ms** (signature-keyed, auto-invalidates when the seed changes) |
| **Cached CCC** `ccc_map_cached()` | India CCC hit screener.in **over the network on every India screen** | reads the committed parquet; refetches only when missing/older than 1 day |
| **Incremental store** `build_store_incremental()` | daily LMDB rebuild re-ingested all **38 k symbols / 477 MB (~30 s)** even if one market changed | rebuilds only markets whose signature changed; **no-op** when nothing changed |
| **Manifest** `data_manifest.json` | each module re-opened parquets to learn coverage | one JSON index: market, tier, symbols, date span, rows, signature |

`screener_kit.load()` and `_india_ccc()` now route through the memo/CCC cache, so
the whole screening stack benefits automatically. All are signature-keyed on file
size+mtime, so correctness is preserved — a changed seed transparently invalidates
its cache entry.

```bash
python3 datalink.py --manifest   # (re)build the index
python3 datalink.py --status     # coverage table
python3 datalink.py --store      # incremental LMDB refresh (changed markets only)
```
