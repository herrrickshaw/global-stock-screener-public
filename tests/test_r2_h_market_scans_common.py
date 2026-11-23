"""Offline tests for the Japan / Korea / Europe full-market scanners (group H).

Each module is imported fresh (cwd = tmp_path so the module-level ``./xxx_scan`` dir lands
there) and ``sys.modules`` is restored afterwards.
"""

import importlib
import sys
import types

import pandas as pd
import pytest

NAMES = ["full_japan_market_scan", "full_korea_market_scan", "full_european_market_scan"]


@pytest.fixture
def load(tmp_path, monkeypatch):
    saved = {}

    def _load(name):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(
            sys.modules, "pykrx", types.SimpleNamespace(stock=types.SimpleNamespace())
        )
        saved.setdefault(name, sys.modules.pop(name, None))
        mod = importlib.import_module(name)
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        monkeypatch.setattr(mod, "DOWNLOAD_DIR", out)
        if hasattr(mod, "_CACHE_FILE"):
            monkeypatch.setattr(mod, "_CACHE_FILE", out / ".symbols_cache.json")
        monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
        return mod

    yield _load
    for name, orig in saved.items():
        sys.modules.pop(name, None)
        if orig is not None:
            sys.modules[name] = orig


def _df(highs, lows, closes):
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes})


H = [10, 9, 8, 7, 6, 5, 5, 5]
L = [3, 3, 3, 3, 3, 4, 4, 4]


def _box(last_close, highs=H, lows=L):
    return _df(highs + [last_close + 1], lows + [last_close - 1], [1] * len(highs) + [last_close])


# ── darvas ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", NAMES)
def test_darvas_insufficient(load, name):
    m = load(name)
    for df in (None, pd.DataFrame(), _df([1] * 7, [1] * 7, [1] * 7)):
        r = m.compute_darvas_box(df)
        assert r == {"signal": "INSUFFICIENT_DATA", "box_top": None, "box_bottom": None}
    # confirm param moves the minimum: confirm=1 -> 6 rows enough
    assert m.compute_darvas_box(_df([1] * 6, [1] * 6, [1] * 6), confirm=1)["signal"] != (
        "INSUFFICIENT_DATA"
    )
    assert m.compute_darvas_box(_df([1] * 8, [1] * 8, [1] * 8))["signal"] != "INSUFFICIENT_DATA"


@pytest.mark.parametrize("name", NAMES)
def test_darvas_signals_and_numbers(load, name):
    m = load(name)
    r = m.compute_darvas_box(_box(12))
    assert r["signal"] == "BREAKOUT_BUY"
    assert r["box_top"] == 6 and r["box_bottom"] == 3
    assert r["upside_pct"] == -50.0 and r["pos_in_box"] == 300.0
    assert r["current_price"] == 12
    r = m.compute_darvas_box(_box(1))
    assert r["signal"] == "BREAKDOWN_SELL"
    assert r["upside_pct"] == 500.0 and r["pos_in_box"] == -66.7
    r = m.compute_darvas_box(_box(4))
    assert r["signal"] == "IN_BOX"
    assert r["upside_pct"] == 50.0 and r["pos_in_box"] == 33.3
    # strict boundaries: == top and == bottom are inside the box
    assert m.compute_darvas_box(_box(6))["signal"] == "IN_BOX"
    assert m.compute_darvas_box(_box(3))["signal"] == "IN_BOX"
    assert m.compute_darvas_box(_box(7))["signal"] == "BREAKOUT_BUY"
    assert m.compute_darvas_box(_box(2))["signal"] == "BREAKDOWN_SELL"


@pytest.mark.parametrize("name", NAMES)
def test_darvas_current_bar_excluded_from_box(load, name):
    m = load(name)
    # a huge last-bar high must not become the box top
    df = _box(4)
    df.loc[df.index[-1], "High"] = 1000
    assert m.compute_darvas_box(df)["box_top"] == 6


@pytest.mark.parametrize("name", NAMES)
def test_darvas_no_box_rising(load, name):
    m = load(name)
    n = 9
    r = m.compute_darvas_box(_df(list(range(1, n + 1)), [1] * n, [5] * n))
    assert r["signal"] == "NO_BOX" and r["box_top"] is None and r["box_bottom"] is None


@pytest.mark.parametrize("name", NAMES)
def test_darvas_fallback_min_low_and_zero_lows(load, name):
    m = load(name)
    # decreasing lows: no confirmed bottom -> min of the segment
    r = m.compute_darvas_box(_box(4, lows=[9, 9, 9, 9, 4, 3, 2, 1]))
    assert r["box_bottom"] == 1 and r["box_top"] == 6
    # all lows zero in the segment -> NO_BOX but top reported
    r = m.compute_darvas_box(_df(H + [9], [0] * 9, [1] * 8 + [4]))
    assert r["signal"] == "NO_BOX" and r["box_top"] == 6 and r["box_bottom"] is None


@pytest.mark.parametrize("name", NAMES)
def test_darvas_zero_range_zero_price_and_garbage(load, name):
    m = load(name)
    r = m.compute_darvas_box(_box(6, lows=[6, 6, 6, 6, 6, 7, 7, 7]))
    assert r["box_top"] == r["box_bottom"] == 6 and r["pos_in_box"] == 0
    r = m.compute_darvas_box(_df(H + [1], L + [0], [1] * 8 + [0]))
    assert r["signal"] == "BREAKDOWN_SELL" and r["upside_pct"] == 0
    df = _box(4)
    df["High"] = df["High"].astype(object)
    df.loc[0, "High"] = "x"  # coerced to 0 and skipped
    assert m.compute_darvas_box(df)["signal"] in ("IN_BOX", "NO_BOX")


# ── style sheet ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,fn", [(NAMES[0], "style_sheet"), (NAMES[1], "style_sheet")])
def test_style_sheet_formats(load, name, fn):
    from openpyxl import Workbook

    m = load(name)
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Rev_CAGR%", "MCap", "LTP", "Plain", "Txt"])
    ws.append(["a", 1.5, 2.0, 3.0, 4.0, "x"])
    ws.append(["b", 1, 2, 3, 4, "y"])
    getattr(m, fn)(ws)
    assert ws["A1"].font.bold and ws["A1"].fill.start_color.rgb.endswith("1A3A5C")
    assert ws["B2"].number_format == '0.00"%"'
    assert ws["C2"].number_format == "#,##0"
    assert ws["D2"].number_format == "#,##0"
    assert ws["E2"].number_format == "0.00"
    assert ws["F2"].alignment.horizontal == "left" and ws["B2"].alignment.horizontal == "right"
    assert ws["A2"].fill.fill_type is None and ws["A3"].fill.fill_type == "solid"
    assert ws.row_dimensions[1].height == 28 and ws.row_dimensions[2].height == 20
    assert ws.column_dimensions["B"].width == 13  # max(len("Rev_CAGR%")+4, 12)
    assert ws.column_dimensions["F"].width == 12
    ws["F2"] = "x" * 30
    getattr(m, fn)(ws)
    assert ws.column_dimensions["F"].width == 34


@pytest.mark.parametrize("name", NAMES[:2])
def test_style_sheet_noop_without_openpyxl(load, monkeypatch, name):
    m = load(name)
    monkeypatch.setattr(m, "OPENPYXL_OK", False)
    assert m.style_sheet(object()) is None  # would raise if it touched ws


def test_europe_style_sheet(load, monkeypatch):
    from openpyxl import Workbook

    m = load(NAMES[2])
    wb = Workbook()
    ws = wb.active
    ws.append(["Symbol", "Change%", "Market_Cap_B", "P_E_Ratio", "LTP", "Piotroski_Score", "Txt"])
    ws.append(["A", 1.0, 2.0, 3.0, 4.0, 7, "x"])
    ws.append(["B", -1.0, 2.0, 3.0, 4.0, 8, "y"])
    m.style_excel_sheet(ws)
    assert ws["A1"].font.bold
    assert ws.row_dimensions[1].height == 28
    assert ws.views.sheetView[0].showGridLines is True
    monkeypatch.setattr(m, "OPENPYXL_OK", False)
    assert m.style_excel_sheet(object()) is None


# ── scan_common-derived helpers ──────────────────────────────────────────────────
def test_args_defaults_and_flags(load):
    j = load(NAMES[0])
    a = j._parse_args([])
    assert (a.top, a.no_scans, a.workers) == (0, False, j.MAX_WORKERS)
    a = j._parse_args(["--top", "3", "--no-scans", "--workers", "2"])
    assert (a.top, a.no_scans, a.workers) == (3, True, 2)
    k = load(NAMES[1])
    a = k._parse_args([])
    assert (a.kospi_only, a.top, a.workers) == (False, 0, k.MAX_WORKERS)
    assert k._parse_args(["--kospi-only"]).kospi_only is True
    e = load(NAMES[2])
    a = e._parse_args([])
    assert (a.top, a.no_scans, a.universe, a.label, a.batch) == (0, False, None, None, 120)
    a = e._parse_args(["--top", "5", "--universe", "u.csv", "--label", "x", "--batch", "7"])
    assert (a.top, a.universe, a.label, a.batch) == (5, "u.csv", "x", 7)


@pytest.mark.parametrize("name", NAMES)
def test_darvas_zero_high_is_skipped_as_box_top(load, name):
    m = load(name)
    r = m.compute_darvas_box(_box(4, highs=[10, 9, 8, 7, 0, 5, 5, 5]))
    assert r["box_top"] == 7
