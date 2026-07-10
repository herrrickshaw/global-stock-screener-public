# ML Factor Screener — learned per-market factor weights

A new, ML/AI-based screener (`ml_factor_screener.py`) that **learns how much each
factor matters in each market** from 10 years of price history, then filters on the
weighted composite. Educational/research only — **not investment advice.**

## Why it's new

Every existing screener in the repo **hand-sets its weights**:

| Existing screener | Weighting |
|---|---|
| PEGU (India, `pegu_sarvas_analysis.R`) | Valuation 30 / Quality 30 / Growth 25 / Safety 15 — **hand-set** threshold ladders |
| PEGU (German/Europe, `german_pegu_score.py`) | Ecosystem 40 / Coverage 30 / Suite 20 / Universe 10 — **hand-set** |
| Expansion screener (8-D→11-D) | Capex 24–31 / FCF 22 / Debt ↓10–14 … — fit on **synthetic** data (own F1 = 0.0) |
| `ml_supervised.py` | XGBoost/HistGBM **classifier** — learns, but doesn't expose factor weights |
| `ml_signal_engine.py` | Ridge, but refit every prediction, weights not persisted |

**None derive interpretable factor weights per market from real data.** This one does.

## Pipeline (per market, from `cache_seed/ltm/<MKT>.parquet` — 10y OHLCV, 20 markets)

1. **Factors** — computed point-in-time per (ticker, date):
   `mom_63`, `mom_21` (momentum) · `rsi14` (mean-reversion) · `trend` (dist. above 200-DMA)
   · `near_high`, `near_low` (6-mo box) · `vol21` (realised volatility).
2. **Weight (the AI step)** — fit **Ridge** (α=10) of the *standardised* factors on the
   forward return (default 21d), forward returns winsorised to 1–99 pct so lottery /
   survivorship movers don't let raw volatility masquerade as a return factor. The
   Ridge **coefficients are the learned, signed factor weights**, reported per market
   with the in-sample information coefficient (IC).
3. **Filter — by liquidity tier** — apply the learned weights to each stock's *latest*
   factors → composite score, split each market into **High / Medium / Low** liquidity
   (terciles of 63-day average dollar volume), and take the top-N movers *within each
   tier*. High-tier = tradeable large names; Low-tier = volatile small/OTC movers.
   Output is ordered High → Medium → Low.

Regime-adaptive **by construction**: the same factor gets a positive weight where it
paid and a negative one where it didn't — no manual regime rules.

## What it learns (21d horizon, in-sample IC)

| Market | IC | Reads as |
|---|---:|---|
| EU | 0.15 | **trend + short-momentum** pay (trend-following) |
| UK | 0.12 | trend + oversold pay; avoid 6-mo extremes |
| CN | 0.08 | **oversold (rsi)** pays; momentum reverses |
| IN | 0.07 | **trend (above-200DMA)** pays (momentum-friendly) |
| JP/KR | 0.06 | mild; KR = short-term reversal + oversold |
| US | 0.04 | **buy oversold**, short-term mean-reversion |

The cross-market divergence matches the repo's real backtest (`BACKTEST_FINDINGS.md`):
India/Europe trend-following, US/China/Korea mean-reverting. The screener recovers
this **without being told** — the weights come from the data.

## Outputs
- `data/curated/ml_factor_weights.parquet` — per-market learned factor weights + IC.
- `data/curated/ml_factor_picks.parquet` — top-N picks per market (score + factors).

## Usage
```
python3 ml_factor_screener.py                     # all markets, 21d, top 20
python3 ml_factor_screener.py --markets US,IN,EU --top 15
python3 ml_factor_screener.py --horizon 63 --weights-only
```

## Caveats (honest)
- **IC is in-sample** (weights fit on all history, then read back). It's directional
  evidence of factor structure, not a validated trading edge — a proper version would
  **walk-forward** (train on rolling past, score the unseen future). ICs are weak
  (0.03–0.15), consistent with the repo's finding that forward returns are barely
  predictable (R² ≈ 0).
- **Price factors only** — the fundamental factors (Piotroski/Coffee-Can) can be added
  for **US** once `sec_history_collector.py` finishes (point-in-time SEC financials);
  other markets lack a historical-financials source.
- **Survivorship** — `ltm` holds only currently-listed names, which flatters every
  backtest here.
