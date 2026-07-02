#!/usr/bin/env python3
# total_return.py
# ===============
# F-06 — dividend/split-adjusted TOTAL return (price appreciation + reinvested
# dividends), the correct basis for long-horizon performance vs price-only return.
#
#   total_return_from(prices, dividends)   # pure calc (per-period arrays)
#   total_return("RELIANCE", "IN")         # best-effort via yfinance adjusted close
#
#   python3 total_return.py --symbol RELIANCE --market IN
#
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import warnings
from typing import List, Optional

import numpy as np

warnings.filterwarnings("ignore")

_SUFFIX = {
    "IN": ".NS",
    "US": "",
    "JP": ".T",
    "KR": ".KS",
    "HK": ".HK",
    "TW": ".TW",
    "AU": ".AX",
    "UK": ".L",
    "DE": ".DE",
    "CA": ".TO",
    "SG": ".SI",
    "BR": ".SA",
    "CH": ".SW",
    "ZA": ".JO",
    "SE": ".ST",
    "FI": ".HE",
    "DK": ".CO",
    "EU": ".PA",
    "CN": ".SS",
    "SA": ".SR",
}


def total_return_from(prices: List[float], dividends: Optional[List[float]] = None) -> float:
    """Total return % over the series: reinvest each period's dividend at that period's
    price, compounding the resulting growth factors."""
    p = np.asarray(prices, float)
    if len(p) < 2 or p[0] <= 0:
        return 0.0
    d = np.zeros(len(p)) if dividends is None else np.asarray(dividends, float)
    growth = 1.0
    for i in range(1, len(p)):
        if p[i - 1] <= 0:
            continue
        growth *= (p[i] + d[i]) / p[i - 1]  # price change + dividend reinvested
    return round((growth - 1) * 100, 2)


def price_return_from(prices: List[float]) -> float:
    p = np.asarray(prices, float)
    return round((p[-1] / p[0] - 1) * 100, 2) if len(p) >= 2 and p[0] > 0 else 0.0


def total_return(symbol: str, market: str = "IN", period: str = "1y") -> dict:
    """Best-effort dividend-adjusted vs price-only return via yfinance."""
    try:
        import yfinance as yf

        h = yf.Ticker(f"{symbol}{_SUFFIX.get(market, '')}").history(
            period=period, auto_adjust=False
        )
        if h.empty:
            return {}
        close = h["Close"].to_numpy("float64")
        divs = h["Dividends"].to_numpy("float64") if "Dividends" in h else None
        return {
            "symbol": symbol,
            "price_return": price_return_from(close),
            "total_return": total_return_from(close, divs),
            "dividend_uplift": round(total_return_from(close, divs) - price_return_from(close), 2),
        }
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dividend-adjusted total return")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--market", default="IN")
    ap.add_argument("--period", default="1y")
    args = ap.parse_args()
    r = total_return(args.symbol, args.market, args.period)
    if r:
        print(
            f"{r['symbol']}: price {r['price_return']}% | total {r['total_return']}% "
            f"| dividend uplift {r['dividend_uplift']}%"
        )
    else:
        print("no data (yfinance throttled/unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
