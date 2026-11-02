from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import custom_screener
import frames
import liquidity
import strategies
from strategies.base import StockData


def mk(seed, n=300, trend=0.001):

    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(trend, 0.02, n)))
    idx = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Volume": r.integers(1_000, 100_000, n).astype(float),
        },
        index=idx,
    )
    df.iloc[-1, df.columns.get_loc("Volume")] *= 5
    return df


def build_snapshot() -> dict:
    """Deterministic outputs of strategies/custom_screener/liquidity/frames on synthetic data."""
    warnings.filterwarnings("ignore")
    fund = [
        dict(
            net_income=5,
            roa=0.08,
            roa_prev=0.05,
            cfo=9,
            debt_to_assets=0.2,
            debt_to_assets_prev=0.3,
            current_ratio=2,
            current_ratio_prev=1.5,
            shares=10,
            shares_prev=10,
            gross_margin=0.4,
            gross_margin_prev=0.35,
            asset_turnover=1.1,
            asset_turnover_prev=1.0,
            eps_growth=25,
            pe=18,
            industry_pe=22,
            roe=22,
            dividend_yield=3.1,
            revenue=1000,
            cogs=600,
            inventory=100,
            receivables=80,
            payables=60,
        ),
        dict(net_income=-1, roa=-0.01, pe=-5, eps_growth=3),
        {},
    ]
    res = {}
    stocks = []
    for i in range(6):
        s = StockData(
            f"S{i}",
            "IN" if i % 2 else "US",
            ohlcv=mk(i, trend=0.002 if i < 3 else -0.001),
            fundamentals=fund[i % 3],
        )
        stocks.append(s)
    stocks.append(StockData("SHORT", "IN", ohlcv=mk(9, n=40)))
    stocks.append(StockData("NODATA", "US"))
    for s in stocks:
        for only in (None, "price"):
            r = strategies.run_all(s, only)
            res[f"{s.symbol}|{only}"] = {k: v.row() for k, v in r.items()}
        res[f"{s.symbol}|metrics"] = custom_screener.compute_metrics(s)
    crit = {"above_200dma": ("==", True), "rsi14": ("<", 90)}
    res["screen"] = json.loads(
        custom_screener.screen(stocks, crit, rank_by="ret_126", top=5).to_json(orient="split")
    )
    res["screen_callable"] = json.loads(
        custom_screener.screen(stocks, lambda m: m.get("ltp", 0) > 0, rank_by="ltp").to_json(
            orient="split"
        )
    )
    try:
        custom_screener.evaluate(stocks[0], {"ltp": ("~", 1)})
    except ValueError as e:
        res["badop"] = str(e)
    res["tiers"] = [
        liquidity.tier(t, m)
        for t in (None, float("nan"), 0, 4e5, 6e5, 3e7)
        for m in (None, "US", "IN", "XX")
    ]
    df = pd.DataFrame({"Symbol": ["A", "B"], "Market": ["US", "IN"]})
    saved_map = dict(liquidity._CACHE_MAP)
    liquidity._CACHE_MAP.clear()
    liquidity._CACHE_MAP.update({"A": 2.5e7, "B": 1e5})
    res["annotate"] = json.loads(liquidity.annotate(df).to_json(orient="split"))
    res["annotate_nomarket"] = json.loads(
        liquidity.annotate(df[["Symbol"]]).to_json(orient="split")
    )
    liquidity._CACHE_MAP.clear()
    liquidity._CACHE_MAP.update(saved_map)
    hist = {s.symbol: s.ohlcv for s in stocks[:3]}
    lg = frames.to_long(hist)
    res["to_long"] = {
        "dtypes": {k: str(v) for k, v in lg.dtypes.items()},
        "head": json.loads(lg.head(5).to_json(orient="split", date_format="iso")),
    }
    back = frames.long_to_dict(lg)
    res["roundtrip"] = {k: float(v["Close"].sum()) for k, v in back.items()}
    return res
