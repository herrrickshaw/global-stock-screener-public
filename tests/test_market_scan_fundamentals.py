"""Characterisation of fundamental_scan() in the Europe/Japan/Korea scanners.

Golden values (tests/fixtures/golden_fundamental_scans.json) were captured from the code
before pass-2 splitting. Fake yfinance Ticker objects with synthetic statements only.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GOLDEN = Path(__file__).parent / "fixtures" / "golden_fundamental_scans.json"
COLS = pd.to_datetime(["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"])


def _frame(rows: dict) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in rows.items()}, index=COLS).T


def _statements(kind: str):
    if kind == "strong":
        inc = _frame(
            {
                "Net Income": [150, 120, 100, 80],
                "Total Revenue": [1500, 1200, 1000, 800],
                "Gross Profit": [700, 500, 400, 300],
                "EBIT": [400, 330, 280, 200],
            }
        )
        bal = _frame(
            {
                "Total Assets": [1000, 1000, 900, 800],
                "Long Term Debt": [100, 200, 200, 200],
                "Current Assets": [500, 400, 380, 300],
                "Current Liabilities": [200, 220, 200, 180],
                "Share Issued": [100, 100, 100, 100],
                "Stockholders Equity": [700, 600, 500, 400],
            }
        )
        cf = _frame(
            {
                "Operating Cash Flow": [300, 250, 200, 150],
                "Free Cash Flow": [200, 150, 100, 50],
            }
        )
    elif kind == "weak":
        inc = _frame(
            {
                "Net Income": [-5, 20, 30, 40],
                "Total Revenue": [800, 900, 1000, 1100],
                "Gross Profit": [100, 200, 300, 400],
                "Operating Income": [10, 20, 30, 40],
            }
        )
        bal = _frame(
            {
                "Total Assets": [1000, 900, 900, 900],
                "Long Term Debt": [400, 300, 200, 100],
                "Total Current Assets": [200, 300, 300, 300],
                "Total Current Liabilities": [300, 200, 200, 200],
                "Share Issued": [120, 100, 100, 100],
                "Total Stockholder Equity": [100, 200, 300, 400],
            }
        )
        cf = _frame(
            {
                "Total Cash From Operating Activities": [-10, 5, 8, 9],
                "Capital Expenditure": [-30, -20, -20, -20],
            }
        )
    elif kind == "no_balance":
        inc = _frame({"Net Income": [10, 9, 8, 7], "Total Revenue": [100, 90, 80, 70]})
        bal, cf = None, None
    else:
        raise ValueError(kind)
    return inc, bal, cf


class _FastInfo:
    def __init__(self, mcap):
        self.market_cap = mcap


class FakeTicker:
    def __init__(self, kind, info=None, mcap=5e11, raise_info=False, raise_fast=False):
        self.kind, self._info, self._mcap = kind, info, mcap
        self._raise_info, self._raise_fast = raise_info, raise_fast
        if kind != "none":
            inc, bal, cf = _statements(kind)
            self.income_stmt, self.balance_sheet, self.cash_flow = inc, bal, cf
        else:
            self.income_stmt = self.balance_sheet = self.cash_flow = None

    @property
    def info(self):
        if self._raise_info:
            raise RuntimeError("info down")
        return self._info

    @property
    def fast_info(self):
        if self._raise_fast:
            raise RuntimeError("fast_info down")
        return _FastInfo(self._mcap)


INFO_FULL = {
    "trailingPE": 15.2,
    "pegRatio": 1.3,
    "dividendYield": 0.03,
    "priceToBook": 2.345,
    "sector": "Industrials",
    "marketCap": 6e9,
    "debtToEquity": 55,
}
INFO_ODD = {
    "regularMarketPrice": 50.0,
    "trailingEps": 2.5,
    "dividendYield": 2.5,
    "debtToEquity": 500,
    "marketCap": 5e8,
}

SCENARIOS = {
    "strong_full_info": dict(kind="strong", info=INFO_FULL),
    "strong_odd_info": dict(kind="strong", info=INFO_ODD),
    "strong_small_cap": dict(kind="strong", info={}, mcap=1e9),
    "strong_fast_raises": dict(kind="strong", info=INFO_FULL, raise_fast=True),
    "strong_info_raises": dict(kind="strong", info=INFO_FULL, raise_info=True),
    "weak": dict(kind="weak", info=INFO_FULL),
    "no_balance": dict(kind="no_balance", info=INFO_FULL),
    "no_statements": dict(kind="none", info=INFO_FULL),
}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _load(name, monkeypatch, tmp_path):
    import importlib

    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "pykrx", types.SimpleNamespace(stock=types.SimpleNamespace()))
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _call(market, mod, scenario):
    tk = FakeTicker(**SCENARIOS[scenario])
    mod.yf.Ticker = lambda *_a, **_k: tk
    if market == "europe":
        return mod.fundamental_scan("ABC.DE")
    if market == "japan":
        return mod.fundamental_scan("1234.T")
    return mod.fundamental_scan("005930", ".KS")


@pytest.mark.parametrize("market", ["europe", "japan", "korea"])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_fundamental_scan_matches_golden(market, scenario, tmp_path, monkeypatch):
    mod = _load(
        f"full_{'european' if market == 'europe' else market}_market_scan", monkeypatch, tmp_path
    )
    monkeypatch.setattr(mod.yf, "Ticker", mod.yf.Ticker)  # restore after test
    if market == "europe":
        monkeypatch.setattr(mod, "_emerging_fundamentals", lambda s: None)
    out = _clean(_call(market, mod, scenario))
    key = f"{market}/{scenario}"
    data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if key not in data and os.environ.get("UPDATE_GOLDEN"):
        data[key] = out
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True))
    assert json.loads(json.dumps(out)) == json.loads(GOLDEN.read_text())[key]


def test_europe_emerging_fallback(tmp_path, monkeypatch):
    mod = _load("full_european_market_scan", monkeypatch, tmp_path)
    monkeypatch.setattr(mod.yf, "Ticker", mod.yf.Ticker)
    tk = FakeTicker("none", info={})
    mod.yf.Ticker = lambda *_a, **_k: tk
    monkeypatch.setattr(
        mod, "_emerging_fundamentals", lambda s: {"f_score": 6, "cc_qualifies": "PASS"}
    )
    out = mod.fundamental_scan("XYZ.IS")
    key = "europe/emerging_fallback"
    data = json.loads(GOLDEN.read_text())
    if key not in data and os.environ.get("UPDATE_GOLDEN"):
        data[key] = _clean(out)
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True))
    assert json.loads(json.dumps(_clean(out))) == json.loads(GOLDEN.read_text())[key]
