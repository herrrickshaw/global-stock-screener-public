"""Synthetic data builders shared by the group-A (backtest) tests."""

import numpy as np
import pandas as pd


def price_panel(symbols, n=420, seed=0, start="2018-01-01", drift=0.0008, vol=0.02):
    """Long OHLCV panel: Date, High, Close, Volume, Symbol (ltm schema)."""
    rng = np.random.default_rng(seed)
    frames = []
    for k, s in enumerate(symbols):
        r = rng.normal(drift, vol, n)
        close = 100 * np.exp(np.cumsum(r)) * (1 + 0.1 * k)
        frames.append(
            pd.DataFrame(
                {
                    "Date": pd.date_range(start, periods=n, freq="B"),
                    "High": close * 1.01,
                    "Close": close,
                    "Volume": rng.integers(1_000, 5_000, n).astype(float),
                    "Symbol": s,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def fund_hist(tickers, years=7, start_year=2018, seed=1):
    """Annual fundamentals history (yfinance/SEC-like schema) with a filed date."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, t in enumerate(tickers):
        for y in range(years):
            ta = 1000 + 50 * y + 10 * k
            rev = 800 * (1.12**y) + 5 * k
            rows.append(
                {
                    "ticker": t,
                    "fy_end": f"{start_year + y}-12-31",
                    "filed": f"{start_year + y + 1}-02-20",
                    "net_income": rev * (0.08 + 0.02 * rng.random()),
                    "revenue": rev,
                    "gross_profit": rev * (0.35 + 0.01 * y),
                    "cfo": rev * 0.12,
                    "ebit": rev * 0.15,
                    "total_assets": ta,
                    "current_assets": ta * 0.4 + 5 * y,
                    "current_liabilities": ta * 0.2,
                    "long_term_debt": ta * (0.3 - 0.02 * y),
                    "equity": ta * 0.5,
                    "shares": 100.0,
                }
            )
    return pd.DataFrame(rows)


def lin_prices(symbols, start="2018-01-01", n=400, base=100.0, slope=1.0):
    """Daily (calendar) prices with Close = base + slope*index, for exact return maths."""
    frames = []
    for s in symbols:
        idx = np.arange(n, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "Date": pd.date_range(start, periods=n, freq="D"),
                    "High": base + slope * idx + 1.0,
                    "Close": base + slope * idx,
                    "Volume": 1000.0,
                    "Symbol": s,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
