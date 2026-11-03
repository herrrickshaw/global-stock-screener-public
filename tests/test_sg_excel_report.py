"""Characterisation of sg_stock_daily_report._write_excel_report (workbook contents + styling)."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

from openpyxl import load_workbook

GOLDEN = Path(__file__).parent / "fixtures" / "golden_sg_excel_report.json"


def _results():
    full = {
        "symbol": "D05.SI",
        "quote": {
            "name": "DBS",
            "sector": "Financials",
            "lastPrice": 40.0,
            "previousClose": 39.0,
            "trailingPE": 9.5,
            "priceToBook": 1.4,
            "dividendYield": 0.05,
            "52wHigh": 42.0,
            "52wLow": 30.0,
            "200dAvg": 36.0,
            "marketCap": 1.1e11,
            "volume": 5_000_000,
        },
        "scans": {
            "darvas": {
                "signal": "BREAKOUT_BUY",
                "box_top": 39.5,
                "box_bottom": 37.0,
                "position_in_box_pct": 100.0,
                "upside_to_top_pct": -1.2,
            },
            "piotroski": {"f_score": 7, "interpretation": "Strong"},
            "pegy": {
                "peg_ratio": 1.1,
                "pegy_adjusted": 0.8,
                "dividend_yield_%": 5.0,
                "breakout": {"signal": "ABOVE_200MA", "distance_from_200ma_%": 11.1},
            },
        },
    }
    sparse = {"symbol": "ZZZ", "quote": {"name": "Sparse"}}
    return [full, sparse]


def _rgb(color):
    v = getattr(color, "rgb", None)
    return v if isinstance(v, str) else None


def _dump(path: Path) -> dict:
    wb = load_workbook(path)
    out = {}
    for ws in wb.worksheets:
        cells = {}
        for r in ws.iter_rows():
            for c in r:
                if c.value is not None or _rgb(c.fill.fgColor) not in (None, "00000000"):
                    cells[c.coordinate] = [
                        c.value,
                        c.font.b,
                        c.font.sz,
                        _rgb(c.font.color),
                        _rgb(c.fill.fgColor),
                        c.alignment.horizontal,
                    ]
        out[ws.title] = {
            "cells": cells,
            "merged": sorted(str(m) for m in ws.merged_cells.ranges),
            "widths": {k: round(v.width, 3) for k, v in sorted(ws.column_dimensions.items())},
        }
    return {"order": wb.sheetnames, "sheets": out}


def test_write_excel_report_matches_golden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("sg_stock_daily_report", None)
    mod = importlib.import_module("sg_stock_daily_report")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", out_dir)
    snap = {}
    for scans in (False, True):
        for f in out_dir.glob("*.xlsx"):
            f.unlink()
        mod._write_excel_report(_results(), include_scans=scans)
        files = list(out_dir.glob("sg_stock_report_*.xlsx"))
        assert len(files) == 1
        snap[f"scans={scans}"] = _dump(files[0])
    if not GOLDEN.exists() and os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    assert json.loads(json.dumps(snap)) == json.loads(GOLDEN.read_text())
