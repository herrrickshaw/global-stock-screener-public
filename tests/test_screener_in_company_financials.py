"""Characterisation of screener_in_auth.company_financials on stub exports (no network)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd

import screener_in_auth as sia

GOLDEN = Path(__file__).parent / "fixtures" / "golden_company_financials.json"


def _export(rows: list) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Data Sheet", header=False, index=False)
    return buf.getvalue()


FULL = [
    ["META", "x"],
    ["Net profit", 120, 100, 90, 80, 70, 60],
    ["Sales", 1500, 1300, 1100, 1000, 900, 800],
    ["Interest", 30, 28, 25],
    ["Profit before tax", 180, 150, 130],
    ["EPS in Rs", "12.5", "10", "9"],
    ["Borrowings", 400, 380, 350],
    ["Reserves", 900, 800, 700],
    ["Equity Capital", 100, 100, 100],
    ["Other Liabilities", 200, 180, 160],
    ["Net Block", 600, 550, 500],
    ["Inventory", 150, 140],
    ["Trade receivables", 210, 190],
    ["Cash & Bank", 80, 70],
    ["Cash from Operating Activity", 140, 120],
    ["No. of Equity Shares", 10, 10],
    ["Dividend Amount", 20, 18, 0],
]
SPARSE = [["Net profit", 10], ["Sales", 100]]
EMPTY_LABELS = [["nothing", 1]]
PAGE = {
    "roe_page": 18.5,
    "roce_page": 22.1,
    "market_cap_cr": 12000.0,
    "current_price": 450.0,
    "pe": 20.5,
    "book_value": 90.0,
    "dividend_yield": 1.1,
}


def test_company_financials_matches_golden(monkeypatch):
    snap = {}
    cases = {"full": _export(FULL), "sparse": _export(SPARSE), "no_labels": _export(EMPTY_LABELS)}
    for name, raw in cases.items():
        for page_name, page in (("page", PAGE), ("nopage", {})):
            monkeypatch.setattr(sia, "company_export_bytes", lambda s, _r=raw: _r)
            monkeypatch.setattr(sia, "page_ratios", lambda s, _p=page: dict(_p))
            snap[f"{name}/{page_name}"] = sia.company_financials("ABC")
    monkeypatch.setattr(sia, "company_export_bytes", lambda s: b"")
    snap["empty_bytes"] = sia.company_financials("ABC")
    monkeypatch.setattr(sia, "company_export_bytes", lambda s: b"garbage")
    snap["corrupt"] = sia.company_financials("ABC")
    snap = json.loads(json.dumps(snap))
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
