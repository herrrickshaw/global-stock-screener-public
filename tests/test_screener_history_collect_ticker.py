"""Characterisation of screener_history_collector.collect_ticker row derivation.

The Excel export is a stub workbook; _extract_annual / industry_of are faked so the test
pins the derivation of EBIT, gross margin, capital employed and ROCE (and the error paths).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import screener_history_collector as shc

GOLDEN = Path(__file__).parent / "fixtures" / "golden_collect_ticker.json"
NAN = float("nan")
D = pd.Timestamp


def _xlsx_bytes() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([["x", 1], ["y", 2]]).to_excel(
            w, sheet_name="Data Sheet", header=False, index=False
        )
    return buf.getvalue()


ANNUAL = {
    "mfr": {
        D("2024-03-31"): dict(
            revenue=1000.0,
            profit_before_tax=200.0,
            interest=40.0,
            raw_material_cost=500.0,
            change_in_inventory=-20.0,
            power_and_fuel=30.0,
            other_mfr_exp=NAN,
            equity_capital=100.0,
            reserves=400.0,
            borrowings=300.0,
        ),
        D("2023-03-31"): dict(
            revenue=900.0,
            profit_before_tax=150.0,
            interest=None,
            raw_material_cost=250.0,
            equity_capital=100.0,
            reserves=300.0,
            borrowings=200.0,
        ),
    },
    "services": {
        D("2024-03-31"): dict(
            revenue=2000.0,
            profit_before_tax=500.0,
            interest=0.0,
            raw_material_cost=10.0,
            equity_capital=50.0,
            reserves=900.0,
            borrowings=0.0,
        ),
    },
    "lender": {
        D("2024-03-31"): dict(
            revenue=100.0,
            profit_before_tax=30.0,
            interest=50.0,
            raw_material_cost=None,
            equity_capital=10.0,
            reserves=60.0,
            borrowings=500.0,
        ),
    },
    "negative_net_worth": {
        D("2024-03-31"): dict(
            revenue=50.0,
            profit_before_tax=5.0,
            interest=1.0,
            raw_material_cost=40.0,
            equity_capital=10.0,
            reserves=-302.0,
            borrowings=420.0,
        ),
    },
    "zero_revenue": {
        D("2024-03-31"): dict(
            revenue=0.0, profit_before_tax=5.0, interest=1.0, raw_material_cost=4.0
        ),
    },
    "empty": {},
}
INDUSTRY = {"lender": "Housing Finance", "mfr": "Auto Components"}


class _Sess:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def company_export_bytes(self, ticker):
        if self.behaviour == "block":
            raise RuntimeError("Max retries exceeded")
        if self.behaviour == "other_error":
            raise ValueError("boom")
        if self.behaviour == "empty":
            return b""
        if self.behaviour == "corrupt":
            return b"not an excel file"
        return _xlsx_bytes()


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return "NaN"
    return o


def test_collect_ticker_matches_golden(monkeypatch):
    snap = {}
    for key in ANNUAL:
        monkeypatch.setattr(shc, "_extract_annual", lambda df, _k=key: ANNUAL[_k])
        monkeypatch.setattr(shc, "industry_of", lambda sess, t, _k=key: INDUSTRY.get(_k))
        snap[key] = _clean(shc.collect_ticker(_Sess("ok"), key.upper()))
    for beh in ("empty", "corrupt", "other_error"):
        snap[beh] = shc.collect_ticker(_Sess(beh), "T")
    with pytest.raises(shc.Blocked):
        shc.collect_ticker(_Sess("block"), "T")
    snap = json.loads(json.dumps(snap))
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert snap == json.loads(GOLDEN.read_text())
