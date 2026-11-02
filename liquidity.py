#!/usr/bin/env python3
# liquidity.py
# ============
# Liquidity pre-filter — rank/screen every symbol by USD trading turnover so the
# expensive strategy logic only runs on TRADABLE names. Computes each stock's
# 20-day median turnover (Close×Volume), FX-converts to USD, and caches a small
# index (cache_seed/liquidity_index.parquet) for instant filtering.
#
#   from liquidity import liquid_symbols, build_index, turnover
#   syms = liquid_symbols("IN", min_usd=1_000_000)   # only liquid IN names
#
# Why it's faster: screens like custom_screen/run_global_analysis otherwise touch
# the full universe (e.g. India 8,931 stocks, most untradable micro-caps). Filtering
# to liquid names first cuts that to the few hundred that matter — often a 10–20×
# reduction in stocks scored.

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

INDEX_PATH = Path(__file__).parent / "cache_seed" / "liquidity_index.parquet"

# market -> currency; LSE quotes pence, JSE cents → divide turnover by unit
CCY = {
    "US": "USD",
    "IN": "INR",
    "CN": "CNY",
    "JP": "JPY",
    "EU": "EUR",
    "HK": "HKD",
    "KR": "KRW",
    "TW": "TWD",
    "CA": "CAD",
    "UK": "GBP",
    "DE": "EUR",
    "AU": "AUD",
    "SG": "SGD",
    "SA": "SAR",
    "BR": "BRL",
    "CH": "CHF",
    "ZA": "ZAR",
    "SE": "SEK",
    "FI": "EUR",
    "DK": "DKK",
}
UNIT = {"UK": 100.0, "ZA": 100.0}


TURNOVER_WINDOW = 20  # trading days used for the median turnover


def _fx_rates(currencies) -> Dict[str, Optional[float]]:
    """Units of each currency per 1 USD (live Yahoo quotes; None where unavailable)."""
    import yfinance as yf

    need = sorted(set(currencies) - {"USD"})
    fx: Dict[str, Optional[float]] = {"USD": 1.0}
    try:
        d = yf.download([f"{c}=X" for c in need], period="5d", progress=False)["Close"]
        for c in need:
            try:
                fx[c] = float(d[f"{c}=X"].dropna().iloc[-1])
            except Exception:  # missing column / empty series: treat as no rate
                log.warning("no FX quote for %s", c)
                fx[c] = None
    except Exception:  # network / yfinance failures: markets without a rate are skipped
        log.warning("FX download failed", exc_info=True)
        for c in need:
            fx[c] = None
    return fx


def median_turnover_usd(df: pd.DataFrame, rate: float, unit: float = 1.0) -> Optional[float]:
    """20-day median of Close*Volume converted to USD (None if too short / no volume).

    `rate` is local-currency units per USD; `unit` is the quote sub-unit (pence=100)."""
    if df is None or len(df) < TURNOVER_WINDOW or "Volume" not in df.columns:
        return None
    t = float((df["Close"] * df["Volume"]).tail(TURNOVER_WINDOW).median())
    return (t / unit) / rate if t > 0 else None


def build_index(
    verbose: bool = True,
    loader: Optional[Callable[[str], Dict[str, pd.DataFrame]]] = None,
    fx: Optional[Dict[str, Optional[float]]] = None,
    markets: Optional[Iterable[str]] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute per-symbol 20-day median turnover (USD) across all market seeds.

    loader/fx/markets/out_path default to the live seeds, Yahoo FX and INDEX_PATH;
    they exist so the logic can be exercised with synthetic data."""
    if loader is None or markets is None:
        import screener_kit as kit

        loader = loader or kit.load
        markets = markets if markets is not None else kit.MARKETS
    fx = fx if fx is not None else _fx_rates(CCY.values())
    out_path = out_path or INDEX_PATH
    rows = []
    for m in markets:
        rate = fx.get(CCY.get(m))
        if not rate:
            continue
        unit = UNIT.get(m, 1.0)
        for s, df in loader(m).items():
            t = median_turnover_usd(df, rate, unit)
            if t is not None:
                rows.append(
                    {
                        "Symbol": s,
                        "Market": m,
                        "turnover_usd": t,
                        "ltp": float(df["Close"].iloc[-1]),
                    }
                )
        if verbose:
            print(f"  {m}: indexed {sum(1 for r in rows if r['Market'] == m)} stocks")
    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx.to_parquet(out_path, compression="zstd", index=False)
    if verbose:
        print(f"  liquidity index: {len(idx)} symbols → {out_path.name}")
    return idx


def _load_index() -> pd.DataFrame:
    return pd.read_parquet(INDEX_PATH) if INDEX_PATH.exists() else pd.DataFrame()


# liquidity tiers by USD median daily turnover (global defaults)
HIGH = 10_000_000  # ≥ $10M/day  → High
MED = 1_000_000  # $1M–$10M/day → Medium  (< $1M → Low)

# Per-market overrides (high, medium) — smaller/thinner markets use lower bars so
# "High/Medium/Low" is meaningful *within* each market. Tune freely.
MARKET_TIERS = {
    "US": (20_000_000, 2_000_000),
    "CN": (20_000_000, 2_000_000),
    "JP": (10_000_000, 1_000_000),
    "EU": (10_000_000, 1_000_000),
    "HK": (5_000_000, 500_000),
    "TW": (5_000_000, 500_000),
    "KR": (5_000_000, 500_000),
    "UK": (5_000_000, 500_000),
    "DE": (5_000_000, 500_000),
    "CA": (3_000_000, 300_000),
    "AU": (3_000_000, 300_000),
    "IN": (5_000_000, 500_000),
    "BR": (3_000_000, 300_000),
    "SA": (3_000_000, 300_000),
    "CH": (3_000_000, 300_000),
    "SG": (2_000_000, 200_000),
    "ZA": (2_000_000, 200_000),
    "SE": (2_000_000, 200_000),
    "FI": (1_000_000, 100_000),
    "DK": (1_000_000, 100_000),
}


def tier(turnover_usd: Optional[float], market: Optional[str] = None) -> str:
    # pd.isna also catches NaN (e.g. a symbol absent from the turnover index,
    # which arrives as float('nan') via DataFrame.map, not None) — previously
    # only `is None` was checked, so unmapped/untracked symbols silently fell
    # through to "Low" instead of "Unknown" (NaN >= anything is False, so the
    # High/Medium comparisons below both fail and it defaults to Low).
    if turnover_usd is None or pd.isna(turnover_usd):
        return "Unknown"
    hi, med = MARKET_TIERS.get(market, (HIGH, MED))
    if turnover_usd >= hi:
        return "High"
    if turnover_usd >= med:
        return "Medium"
    return "Low"


_CACHE_MAP = {}


def _turnover_map() -> Dict[str, float]:
    global _CACHE_MAP
    if not _CACHE_MAP:
        idx = _load_index()
        if not idx.empty:
            _CACHE_MAP = dict(zip(idx["Symbol"], idx["turnover_usd"]))
    return _CACHE_MAP


def annotate(df: pd.DataFrame, symbol_col: str = "Symbol") -> pd.DataFrame:
    """Add Turnover_USD and Liquidity (High/Medium/Low) columns to a result frame."""
    if df is None or df.empty or symbol_col not in df.columns:
        return df
    tmap = _turnover_map()
    df = df.copy()
    df["Turnover_USD"] = df[symbol_col].map(tmap).round(0)
    if "Market" in df.columns:
        df["Liquidity"] = [tier(t, m) for t, m in zip(df["Turnover_USD"], df["Market"])]
    else:
        df["Liquidity"] = df["Turnover_USD"].map(tier)
    return df


def turnover(symbol: str) -> Optional[float]:
    """USD median daily turnover for one symbol (from the cached index)."""
    idx = _load_index()
    hit = idx[idx["Symbol"] == symbol]
    return float(hit.iloc[0]["turnover_usd"]) if not hit.empty else None


def liquid_symbols(
    market: Optional[str] = None, min_usd: float = 1_000_000, top: Optional[int] = None
) -> List[str]:
    """Symbols trading at least `min_usd`/day, optionally limited to a market and
    the top-N most liquid. This is the fast pre-filter for screens."""
    idx = _load_index()
    if idx.empty:
        return []
    if market:
        idx = idx[idx["Market"] == market]
    idx = idx[idx["turnover_usd"] >= min_usd].sort_values("turnover_usd", ascending=False)
    if top:
        idx = idx.head(top)
    return idx["Symbol"].tolist()


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build_index()
    idx = _load_index()
    if not idx.empty:
        print(f"\nindex: {len(idx)} symbols")
        for m in idx["Market"].unique():
            sub = idx[idx["Market"] == m]
            liq = (sub["turnover_usd"] >= 1e6).sum()
            print(f"  {m}: {len(sub):>5} total, {liq:>4} liquid (≥$1M/day)")
