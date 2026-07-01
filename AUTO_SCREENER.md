# Hybrid auto-screener + multi-market calendar

Implements the three-layer design from *ML_Stock_Screening_System.docx*, adapted
to run on the precomputed serving views (fast, no heavy training loop).

## Three layers (`auto_screener.py`)

| Layer | Doc | Here |
|-------|-----|------|
| **Supervised anchor** | XGBoost on labelled returns | `known_good(market)` — the "known universe": symbols the existing screens would flag, derived from the deep serving view via each strategy's published gates (darvas near-high breakout, golden crossover, coffee-can momentum) + live screener.in CCC for IN. These are the labels. |
| **Unsupervised discovery** | UMAP + HDBSCAN + KMeans | `discover()` — StandardScaler + KMeans over the serving features (RSI, drawdown, multi-horizon returns, trend flags) restricted to liquid names. The cluster most **enriched with known-good, liquid, high-return** stocks becomes a *new screen*: its 10th–90th-percentile feature bounds → a rule. Adapts to **market conditions** (regime) and **liquidity**. |
| **Reinforcement correction** | PPO agent | `rl_refine()` — kicks in only when the discovery **deviates too far** from the known universe (reward < threshold or deviation > 0.5). A reward-driven policy search perturbs the thresholds, keeping changes that raise `reward = 2·overlap + 0.5·liquidity + 0.01·return − 1.5·deviation` (the doc's reward shape), pulling the screen back toward the validated universe. |

Regime-aware: **Bear** tape hard-requires `Above200DMA` + caps drawdown (no falling
knives); **Bull** tape requires `GoldenCross` and lets momentum breathe.

```bash
python3 auto_screener.py --market IN --top 15
python3 auto_screener.py --market US
```

Output: printed regime + discovered rule + metrics + top picks, saved to
`cache_seed/discovered_screens/<MKT>.json` (gitignored, regenerated).

Example (IN, Bear tape): discovered a cluster with **100% overlap** vs 530 known-good
names, reward 3.17, no RL needed. When the anchor is thin the RL corrector engages
(observed reward 0.27 → 1.96).

## Multi-market holiday calendar (`market_holidays.py`)

Trading-day calendar for **all 20 markets** — weekend rules (Saudi rests Fri/Sat,
others Sat/Sun) + 2025–2026 exchange holiday sets (IN pulls the live NSE list).

```python
from market_holidays import is_trading_day, trading_days, next_trading_day, is_market_open_today
```

Wired into `market_memory.update_all`: markets **closed today are skipped** (no
wasted fetch/processing). E.g. on 1 Jul, HK (SAR Day) and CA (Canada Day) are
skipped automatically.

> ⚠️ Research/education only. Discovered screens are historical associations, not
> predictions or advice. Human review always wins (per the doc's guardrails).
