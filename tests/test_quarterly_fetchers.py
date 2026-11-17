"""Characterisation of QuarterlyAnalyzer.fetch_company_data and the collector's
fetch_quarterly_data on fake yfinance tickers (synthetic statements, no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import quarterly_data_analyzer as qa
import quarterly_data_collector as qc

GOLDEN = Path(__file__).parent / "fixtures" / "golden_quarterly_fetchers.json"


def _stmt(rows: dict, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = pd.to_datetime(pd.date_range("2025-12-31", periods=n, freq="-3ME"))
    return pd.DataFrame({k: rng.uniform(lo, hi, n) for k, (lo, hi) in rows.items()}, index=cols).T


IS_ROWS = {
    "Total Revenue": (900, 1500),
    "Operating Income": (100, 300),
    "Net Income": (50, 200),
    "Interest Expense": (5, 20),
}
CF_ROWS = {"Operating Cash Flow": (80, 250), "Capital Expenditure": (-90, -10)}
BS_ROWS = {
    "Total Assets": (4000, 5000),
    "Total Debt": (500, 900),
    "Total Stockholder Equity": (1500, 2500),
    "Current Assets": (900, 1200),
    "Current Liabilities": (500, 800),
}


class FakeTicker:
    def __init__(self, kind: str):
        self.kind = kind
        n = {"full": 8, "short": 3, "partial": 6}.get(kind, 8)
        if kind == "none":
            self.quarterly_income_stmt = self.quarterly_cashflow = None
            self.quarterly_balance_sheet = None
            return
        is_rows = {k: v for k, v in IS_ROWS.items() if kind != "partial" or k != "Interest Expense"}
        bs_rows = {k: v for k, v in BS_ROWS.items() if kind != "partial" or k != "Current Assets"}
        self.quarterly_income_stmt = _stmt(is_rows, n, 1)
        self.quarterly_cashflow = _stmt(CF_ROWS, n, 2)
        self.quarterly_balance_sheet = _stmt(bs_rows, n, 3)
        if kind == "zero_equity":
            self.quarterly_balance_sheet.loc["Total Stockholder Equity"] = 0.0
        if kind == "nan_cells":
            self.quarterly_income_stmt.iloc[0, 1] = np.nan
            self.quarterly_cashflow.iloc[1, 2] = np.nan


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return [_clean(r) for r in o.to_dict("records")]
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 9)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


KINDS = ["full", "short", "partial", "none", "zero_equity", "nan_cells"]


def test_fetchers_match_golden(monkeypatch):
    snap = {}
    for kind in KINDS:
        tk = FakeTicker(kind)
        monkeypatch.setattr(qa.yf, "Ticker", lambda *_a, _tk=tk, **_k: _tk)
        res = qa.QuarterlyAnalyzer().fetch_company_data("AAA")
        res = _clean(res)
        if res:
            for q in res["quarters"]:
                q.pop("quarter_label", None)  # strftime %q is platform-specific
        snap[f"analyzer/{kind}"] = res
        monkeypatch.setattr(qc.yf, "Ticker", lambda *_a, _tk=tk, **_k: _tk)
        collector = next(
            v
            for v in vars(qc).values()
            if isinstance(v, type) and hasattr(v, "fetch_quarterly_data")
        ).__new__(
            next(
                v
                for v in vars(qc).values()
                if isinstance(v, type) and hasattr(v, "fetch_quarterly_data")
            )
        )
        snap[f"collector/{kind}"] = _clean(collector.fetch_quarterly_data("AAA"))
    monkeypatch.setattr(qa.yf, "Ticker", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    snap["analyzer/raises"] = qa.QuarterlyAnalyzer().fetch_company_data("AAA")
    snap = json.loads(json.dumps(snap))
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
