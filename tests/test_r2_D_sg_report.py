"""Offline tests for sg_stock_daily_report.py (fake yfinance Ticker objects, cwd in tmp_path)."""

import importlib
import runpy
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "sg_stock_daily_report", raising=False)
    fake_yf = types.ModuleType("yfinance")
    fake_yf.Ticker = lambda s: FakeTicker(sym=s)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    mod = importlib.import_module("sg_stock_daily_report")
    monkeypatch.setattr(mod, "DOWNLOAD_DIR", tmp_path / "dl")
    (tmp_path / "dl").mkdir()
    yield mod
    sys.modules.pop("sg_stock_daily_report", None)


class FakeFI:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Boom:
    def __getattr__(self, name):
        raise RuntimeError("fi boom")


class FakeTicker:
    def __init__(
        self, sym="D05.SI", fi=None, info=None, hist=None, actions=None, calendar=None, **frames
    ):
        self.sym = sym
        self.fast_info = fi if fi is not None else FakeFI()
        self._info = info if info is not None else {}
        self._hist = hist
        self.actions = actions
        self._calendar = calendar
        self.__dict__.update(frames)
        self.history_calls = []

    @property
    def info(self):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info

    @property
    def calendar(self):
        if isinstance(self._calendar, Exception):
            raise self._calendar
        return self._calendar

    def history(self, period="1mo", auto_adjust=True):
        self.history_calls.append(period)
        if isinstance(self._hist, Exception):
            raise self._hist
        return self._hist


# ---------- formatting ----------


def test_fmt(sg):
    assert sg.fmt(1234.567) == "S$1,234.57"
    assert sg.fmt(1.5, prefix="", decimals=1) == "1.5"
    assert sg.fmt(0) == "S$0.00"
    assert sg.fmt("abc") == "abc"
    assert sg.fmt(None) == "N/A" and sg.fmt("") == "N/A"


@pytest.mark.parametrize(
    "v,exp",
    [
        (1e12, "S$1.00T"),
        (999e9, "S$999.00B"),
        (1e9, "S$1.00B"),
        (1e9 - 1e4, "S$999.99M"),
        (2.5e6, "S$2.50M"),
        (999999, "S$1000.00K"),
        (1000, "S$1.00K"),
        (999, "S$999.00"),
        (-2e9, "S$-2.00B"),
        (0, "S$0.00"),
        (None, "N/A"),
        ("x", "N/A"),
    ],
)
def test_fmt_large(sg, v, exp):
    assert sg.fmt_large(v) == exp


def test_pct(sg):
    assert sg.pct(1.234) == "\033[92m▲ 1.23%\033[0m"
    assert sg.pct(0) == "\033[92m▲ 0.00%\033[0m"
    assert sg.pct(-2) == "\033[91m▼ 2.00%\033[0m"
    assert sg.pct(None) == "N/A" and sg.pct("x") == "N/A"


def test_section_and_row(sg, capsys):
    sg.section("hello")
    out = capsys.readouterr().out
    assert out == "\n" + "-" * 60 + "\n  HELLO\n" + "-" * 60 + "\n"
    sg.row("Lbl", "val")
    assert capsys.readouterr().out == f"  {'Lbl':<30} val\n"
    sg.row("L", "v", width=3)
    assert capsys.readouterr().out == "  L   v\n"


def test_get_ticker_suffix(sg):
    assert sg._get_ticker(" d05 ").sym == "D05.SI"
    assert sg._get_ticker("bn4.si").sym == "BN4.SI"


# ---------- financial frames ----------


def test_yf_financials(sg):
    df = pd.DataFrame({"a": [1]})
    t = FakeTicker(income_stmt=pd.DataFrame(), financials=df, balance_sheet=df, cashflow=df)
    inc, bal, cf = sg._yf_financials(t)
    assert inc is df and bal is df and cf is df  # empty income_stmt skipped -> falls through
    t = FakeTicker(income_stmt="notdf", cash_flow=df)
    inc, bal, cf = sg._yf_financials(t)
    assert inc is None and bal is None and cf is df
    assert sg._yf_financials(FakeTicker()) == (None, None, None)


def test_row(sg):
    df = pd.DataFrame({"c0": [5.0, np.nan, 7.0], "c1": [2.0, 3.0, "bad"]}, index=["A", "B", "C"]).T
    df.columns = ["A", "B", "C"]
    df = pd.DataFrame([[5.0, 2.0], [np.nan, 3.0], [7.0, "bad"]], index=["A", "B", "C"])
    assert sg._row(df, "A") == 5.0 and sg._row(df, "A", col=1) == 2.0
    assert sg._row(df, "B") is None  # NaN
    assert sg._row(df, "Z", "A") == 5.0  # first matching name
    assert sg._row(df, "A", col=5) is None  # IndexError swallowed
    assert sg._row(df, "C", col=1) is None  # ValueError swallowed
    assert sg._row(None, "A") is None and sg._row(pd.DataFrame(), "A") is None
    assert sg._row(df, "C", "A", col=1) == 2.0  # failed name falls through to next


# ---------- quote / history / actions ----------


def test_fetch_quote_prefers_fast_info(sg):
    fi = FakeFI(
        last_price=10.0,
        previous_close=9.0,
        open=9.5,
        day_high=11.0,
        day_low=8.0,
        last_volume=100,
        three_month_average_volume=200,
        market_cap=1e9,
        year_high=12.0,
        year_low=5.0,
        year_change=0.1,
        fifty_day_average=9.0,
        two_hundred_day_average=8.5,
        currency="SGD",
        exchange="SES",
    )
    info = {
        "currentPrice": 99.0,
        "previousClose": 98.0,
        "trailingPE": 12.0,
        "shortName": "DBS",
        "sector": "Fin",
        "industry": "Banks",
        "country": "SG",
        "dividendYield": 0.05,
        "beta": 1.1,
        "exchange": "SGX",
    }
    q = sg.fetch_quote(FakeTicker(fi=fi, info=info))
    assert q["lastPrice"] == 10.0 and q["previousClose"] == 9.0 and q["volume"] == 100
    assert q["52wHigh"] == 12.0 and q["200dAvg"] == 8.5 and q["ytdChange"] == 0.1
    assert (
        q["trailingPE"] == 12.0 and q["name"] == "DBS" and q["exchange"] == "SGX"
    )  # info exchange first
    assert q["currency"] == "SGD" and q["dividendYield"] == 0.05


def test_fetch_quote_info_fallbacks_and_defaults(sg):
    info = {
        "regularMarketPrice": 5.0,
        "averageVolume": 10,
        "fiftyTwoWeekHigh": 7.0,
        "fiftyTwoWeekLow": 1.0,
        "longName": "Long",
        "dayHigh": 6.0,
    }
    q = sg.fetch_quote(FakeTicker(info=info))
    assert (
        q["lastPrice"] == 5.0
        and q["avgVolume"] == 10
        and q["52wHigh"] == 7.0
        and q["52wLow"] == 1.0
    )
    assert q["name"] == "Long" and q["dayHigh"] == 6.0
    assert (
        q["currency"] == "SGD"
        and q["exchange"] == "SGX"
        and q["sector"] == "—"
        and q["country"] == "SG"
    )
    assert q["ytdChange"] is None
    q2 = sg.fetch_quote(FakeTicker(info={"currentPrice": 1.0, "regularMarketPrice": 2.0}))
    assert q2["lastPrice"] == 1.0  # currentPrice before regularMarketPrice
    fi = FakeFI(last_price=0, exchange="XYZ")
    q3 = sg.fetch_quote(FakeTicker(fi=fi, info={"currentPrice": 4.0}))
    assert q3["lastPrice"] == 4.0 and q3["exchange"] == "XYZ"  # falsy fast_info value falls through


def test_fetch_quote_survives_errors(sg):
    q = sg.fetch_quote(FakeTicker(fi=_Boom(), info=RuntimeError("no info")))
    assert q["lastPrice"] is None and q["name"] == "—" and q["currency"] == "SGD"
    q = sg.fetch_quote(FakeTicker(info=None))  # info or {} path
    assert q["trailingPE"] is None


def test_fetch_historical(sg, capsys):
    df = pd.DataFrame({"Close": [1.0]})
    t = FakeTicker(hist=df)
    assert sg.fetch_historical(t) is df and t.history_calls == ["1mo"]
    sg.fetch_historical(t, period="6mo")
    assert t.history_calls[-1] == "6mo"
    assert sg.fetch_historical(FakeTicker(hist=None)).empty
    out = sg.fetch_historical(FakeTicker(hist=RuntimeError("bad")))
    assert out.empty and "[Historical error] bad" in capsys.readouterr().out


def test_fetch_corporate_actions(sg):
    idx = pd.to_datetime([f"2024-0{i}-01" for i in range(1, 9)])
    acts = pd.DataFrame(
        {"Dividends": [0, 1, 2, 0, 3, 4, 5, 6], "Stock Splits": [0, 0, 2, 0, 0, 0, 3, 0]}, index=idx
    )
    cal = {
        "Earnings Date": ["2024-09-01", "x"],
        "Ex-Dividend Date": "2024-08-01",
        "Dividend Date": "2024-08-20",
    }
    r = sg.fetch_corporate_actions(FakeTicker(actions=acts, calendar=cal))
    assert len(r["dividends"]) == 5 and r["dividends"][-1] == {"date": "2024-08-01", "amount": 6}
    assert r["dividends"][0]["amount"] == 2  # tail(5) of 6 positive dividends drops the first
    assert r["splits"] == [{"date": "2024-03-01", "ratio": 2}, {"date": "2024-07-01", "ratio": 3}]
    assert r["calendar"] == {
        "next_earnings": "2024-09-01",
        "ex_dividend_date": "2024-08-01",
        "dividend_date": "2024-08-20",
    }


def test_fetch_corporate_actions_empty_and_errors(sg):
    r = sg.fetch_corporate_actions(FakeTicker(actions=None, calendar=None))
    assert r == {
        "dividends": [],
        "splits": [],
        "calendar": {"next_earnings": "—", "ex_dividend_date": "—", "dividend_date": "—"},
    }
    r = sg.fetch_corporate_actions(
        FakeTicker(actions=pd.DataFrame(), calendar={"Earnings Date": []})
    )
    assert r["dividends"] == [] and r["calendar"]["next_earnings"] == "—"
    r = sg.fetch_corporate_actions(
        FakeTicker(actions=pd.DataFrame({"x": [1]}), calendar=RuntimeError())
    )
    assert r["dividends"] == [] and r["calendar"] == {}  # both errors swallowed


# ---------- darvas helpers ----------


def test_darvas_result_and_find_col(sg):
    r = sg._darvas_result("NO_BOX", "n", box_top=5)
    assert r == {"signal": "NO_BOX", "box_top": 5, "box_bottom": None, "note": "n"}
    df = pd.DataFrame(columns=["Date", "ch_trade_high_price", "HIGH", "Adj High"])
    assert sg._find_col(df, ["High"]) == "ch_trade_high_price"  # first column in frame order
    assert sg._find_col(df, ["Zzz", "Date"]) == "Date"  # candidate order wins
    assert sg._find_col(df, ["Nope"]) is None


def test_find_box_top(sg):
    # confirm=3: need candidate followed by 3 strictly lower highs; most recent wins
    assert sg._find_box_top([10, 9, 8, 7], 3) == (0, 10)
    assert sg._find_box_top([10, 9, 8, 10, 9, 8, 7], 3) == (3, 10)
    assert sg._find_box_top([10, 9, 10, 7], 3) == (None, None)  # not strictly lower
    assert sg._find_box_top([5, 5, 5, 5], 3) == (None, None)  # equal is not lower
    assert sg._find_box_top([0, 0, 0, 0], 3) == (None, None)  # zero candidates skipped
    assert sg._find_box_top([10, 9, 8], 3) == (None, None)  # window too short
    assert sg._find_box_top([], 3) == (None, None)


def test_find_box_bottom(sg):
    assert sg._find_box_bottom([5, 6, 7, 8], 3) == 5
    assert sg._find_box_bottom([5, 6, 4, 8, 9, 10], 3) == 4  # 5 not confirmed (4<5) -> 4 confirmed
    assert sg._find_box_bottom([5, 6, 5], 3) == 5  # shorter than confirm -> min positive low
    assert sg._find_box_bottom([0, 0], 3) is None
    assert sg._find_box_bottom([0, 2, 3, 4, 5], 3) == 2  # zero skipped
    assert sg._find_box_bottom([3, 2, 1, 0.5], 3) == 0.5  # nothing confirmed -> min


def test_classify_vs_box(sg):
    assert sg._classify_vs_box(10.01, 10, 5) == "BREAKOUT_BUY"
    assert sg._classify_vs_box(10, 10, 5) == "IN_BOX"
    assert sg._classify_vs_box(5, 10, 5) == "IN_BOX"
    assert sg._classify_vs_box(4.99, 10, 5) == "BREAKDOWN_SELL"


def _ohlc(highs, lows, closes):
    return pd.DataFrame({"High": highs, "Low": lows, "Close": closes})


def test_compute_darvas_edge_cases(sg):
    assert sg.compute_darvas_box(None)["note"] == "Empty OHLC DataFrame"
    assert sg.compute_darvas_box(pd.DataFrame())["signal"] == "INSUFFICIENT_DATA"
    r = sg.compute_darvas_box(pd.DataFrame({"X": [1] * 10}))
    assert r["note"] == "Could not identify OHLC columns"
    r = sg.compute_darvas_box(_ohlc([1] * 7, [1] * 7, [1] * 7))
    assert r["signal"] == "INSUFFICIENT_DATA" and "Need ≥ 8 bars; got 7" in r["note"]
    r = sg.compute_darvas_box(_ohlc([1] * 8, [1] * 8, [1] * 8), confirm=3)
    assert r["signal"] == "NO_BOX" and r["note"] == "No confirmed box top"
    assert (
        sg.compute_darvas_box(_ohlc([1] * 6, [1] * 6, [1] * 6), confirm=1)["signal"] == "NO_BOX"
    )  # confirm param honoured


def test_compute_darvas_signals_and_numbers(sg):
    highs = [20, 18, 17, 16, 15, 15, 15, 15, 99]
    lows = [10, 11, 12, 13, 14, 14, 14, 14, 1]
    base = _ohlc(highs, lows, [15] * 8 + [25])
    r = sg.compute_darvas_box(base)
    # most recent confirmed top is 16 (idx 3; followed by three lower highs); bottom 13
    assert r["signal"] == "BREAKOUT_BUY" and r["box_top"] == 16 and r["box_bottom"] == 13
    assert r["current_price"] == 25 and r["box_range"] == 3
    assert r["position_in_box_pct"] == 400.0 and r["upside_to_top_pct"] == -36.0
    assert r["confirm_days"] == 3 and r["data_points"] == 9  # current bar's high 99 / low 1 ignored
    r = sg.compute_darvas_box(_ohlc(highs, lows, [15] * 8 + [9]))
    assert r["signal"] == "BREAKDOWN_SELL" and r["position_in_box_pct"] == -133.3
    r = sg.compute_darvas_box(_ohlc(highs, lows, [15] * 8 + [15]))
    assert (
        r["signal"] == "IN_BOX"
        and r["position_in_box_pct"] == 66.7
        and r["upside_to_top_pct"] == 6.67
    )


def test_compute_darvas_zero_range_and_no_bottom(sg):
    # box top 8, every low 8 -> bottom 8, range 0 -> position 0
    r = sg.compute_darvas_box(_ohlc([10, 9, 8, 7, 7, 7, 7, 7, 7], [8] * 9, [5] * 8 + [10]))
    assert r["box_range"] == 0 and r["position_in_box_pct"] == 0
    # all lows zero -> bottom None -> NO_BOX with box_top retained
    r = sg.compute_darvas_box(_ohlc([10, 9, 8, 7, 7, 7, 7, 7, 7], [0] * 9, [5] * 9))
    assert (
        r["signal"] == "NO_BOX"
        and r["box_top"] == 8
        and r["note"] == "Could not confirm a box bottom"
    )
    # zero current price: upside guarded
    r = sg.compute_darvas_box(
        _ohlc(
            [20, 18, 17, 16, 15, 15, 15, 15, 15],
            [10, 11, 12, 13, 14, 14, 14, 14, 14],
            [15] * 8 + [0],
        )
    )
    assert r["upside_to_top_pct"] == 0 and r["signal"] == "BREAKDOWN_SELL"


def test_compute_darvas_alt_columns_and_nan(sg):
    df = pd.DataFrame(
        {
            "CH_TRADE_HIGH_PRICE": [20, 18, 17, 16, 15, 15, 15, 15, 15],
            "CH_TRADE_LOW_PRICE": [10, 11, 12, 13, 14, 14, 14, 14, 14],
            "CH_CLOSING_PRICE": [15, 15, 15, 15, 15, 15, 15, 15, np.nan],
        }
    )
    r = sg.compute_darvas_box(df)
    assert r["signal"] == "BREAKDOWN_SELL" and r["current_price"] == 0  # NaN -> 0 via fillna


def test_display_darvas(sg, capsys):
    sg.display_darvas_box(
        {
            "signal": "BREAKOUT_BUY",
            "box_top": 20,
            "box_bottom": 10,
            "current_price": 25,
            "box_range": 10,
            "position_in_box_pct": 150.0,
            "upside_to_top_pct": -20.0,
        }
    )
    out = capsys.readouterr().out
    assert "BREAKOUT BUY" in out and "S$20.00" in out and "150.0%" in out and "-20.00%" in out
    for sig, txt in [
        ("BREAKDOWN_SELL", "BREAKDOWN SELL"),
        ("IN_BOX", "IN BOX"),
        ("NO_BOX", "No confirmed Darvas box"),
        ("INSUFFICIENT_DATA", "Insufficient data"),
        ("WEIRD", "WEIRD"),
    ]:
        sg.display_darvas_box({"signal": sig})
        o = capsys.readouterr().out
        assert txt in o and "Box Top" not in o
    sg.display_darvas_box({})
    assert "Signal: N/A" in capsys.readouterr().out


# ---------- piotroski ----------


def _fin(rows):
    return pd.DataFrame(rows).T if False else pd.DataFrame({k: v for k, v in rows.items()}).T


def _frames(strong=True):
    inc = pd.DataFrame(
        {
            "Net Income": [100.0, 50.0],
            "Total Revenue": [1000.0, 900.0],
            "Gross Profit": [500.0, 400.0],
        }
    ).T
    bal = pd.DataFrame(
        {
            "Total Assets": [1000.0, 1000.0],
            "Long Term Debt": [100.0, 200.0],
            "Current Assets": [500.0, 400.0],
            "Current Liabilities": [250.0, 250.0],
            "Share Issued": [10.0, 10.0],
        }
    ).T
    cf = pd.DataFrame({"Operating Cash Flow": [200.0, 100.0]}).T
    return inc, bal, cf


def test_piotroski_perfect(sg):
    inc, bal, cf = _frames()
    t = FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf)
    r = sg.compute_piotroski_score(t, "D05.SI")
    assert r["f_score"] == 9 and r["interpretation"] == "STRONG" and r["symbol"] == "D05.SI"
    assert all(v == 1 for v in r["component_scores"].values()) and len(r["component_scores"]) == 9
    d = r["details"]
    assert d["ROA_current_%"] == 10.0 and d["ROA_prev_%"] == 5.0 and d["OCF_current_$M"] == 0.0
    assert (
        d["LTD_ratio_curr_%"] == 10.0
        and d["CurrentRatio_curr"] == 2.0
        and d["GrossMargin_curr_%"] == 50.0
    )


def test_piotroski_no_income(sg):
    assert sg.compute_piotroski_score(FakeTicker(), "X") == {
        "symbol": "X",
        "error": "No income statement data available",
    }


def test_piotroski_weak_and_branches(sg):
    inc = pd.DataFrame(
        {
            "Net Income": [-10.0, 50.0],
            "Total Revenue": [800.0, 900.0],
            "Gross Profit": [300.0, 400.0],
        }
    ).T
    bal = pd.DataFrame(
        {
            "Total Assets": [1000.0, 1000.0],
            "Long Term Debt": [300.0, 200.0],
            "Current Assets": [300.0, 400.0],
            "Current Liabilities": [250.0, 250.0],
            "Share Issued": [12.0, 10.0],
        }
    ).T
    cf = pd.DataFrame({"Total Cash From Operating Activities": [-5.0, 10.0]}).T
    r = sg.compute_piotroski_score(
        FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf), "W"
    )
    # only F4 fires: negative OCF/assets (-0.005) still exceeds a more negative ROA (-0.01)
    assert r["f_score"] == 1 and r["interpretation"] == "WEAK"
    assert (
        r["component_scores"]["F4_Accruals"] == 1 and r["component_scores"]["F2_OCF_positive"] == 0
    )
    assert r["component_scores"]["F7_No_dilution"] == 0  # shares increased


def test_piotroski_missing_data_defaults(sg):
    inc = pd.DataFrame({"Net Income": [100.0]}).T  # only one period, no revenue
    r = sg.compute_piotroski_score(FakeTicker(income_stmt=inc), "M")
    s = r["component_scores"]
    assert s["F7_No_dilution"] == 1  # unknown share counts => no dilution
    assert s["F1_ROA_positive"] == 0  # no assets
    assert sum(s.values()) == 1 and r["interpretation"] == "WEAK"
    assert r["details"]["ROA_current_%"] == "N/A" and r["details"]["GrossMargin_curr_%"] == "N/A"


def test_piotroski_interpretation_boundaries(sg, monkeypatch):
    inc, bal, cf = _frames()
    # break F9 (asset turnover) and F8 -> 7 points: STRONG at exactly 7
    inc7 = inc.copy()
    inc7.loc["Gross Profit"] = [300.0, 400.0]  # margin down -> F8=0
    inc7.loc["Total Revenue"] = [800.0, 900.0]  # turnover down -> F9=0
    r = sg.compute_piotroski_score(FakeTicker(income_stmt=inc7, balance_sheet=bal, cash_flow=cf))
    assert r["f_score"] == 7 and r["interpretation"] == "STRONG"
    bal6 = bal.copy()
    bal6.loc["Share Issued"] = [12.0, 10.0]  # F7=0 -> 6
    r = sg.compute_piotroski_score(FakeTicker(income_stmt=inc7, balance_sheet=bal6, cash_flow=cf))
    assert r["f_score"] == 6 and r["interpretation"] == "MODERATE"
    bal4 = bal6.copy()
    bal4.loc["Current Assets"] = [300.0, 400.0]  # F6=0 -> 5
    bal4.loc["Long Term Debt"] = [300.0, 200.0]  # F5=0 -> 4
    r = sg.compute_piotroski_score(FakeTicker(income_stmt=inc7, balance_sheet=bal4, cash_flow=cf))
    assert r["f_score"] == 4 and r["interpretation"] == "MODERATE"
    cf0 = pd.DataFrame({"Operating Cash Flow": [-1.0, 100.0]}).T  # F2=0,F4=0 -> 2
    r = sg.compute_piotroski_score(FakeTicker(income_stmt=inc7, balance_sheet=bal4, cash_flow=cf0))
    assert r["f_score"] == 2 and r["interpretation"] == "WEAK"


def test_piotroski_f4_accruals_boundary(sg):
    inc, bal, cf = _frames()
    cf_eq = pd.DataFrame(
        {"Operating Cash Flow": [100.0, 100.0]}
    ).T  # OCF/A == ROA -> not strictly greater
    r = sg.compute_piotroski_score(FakeTicker(income_stmt=inc, balance_sheet=bal, cash_flow=cf_eq))
    assert (
        r["component_scores"]["F4_Accruals"] == 0 and r["component_scores"]["F2_OCF_positive"] == 1
    )


def test_display_piotroski(sg, capsys):
    sg.display_piotroski_score({"error": "boom"})
    assert "boom" in capsys.readouterr().out
    for score, col in [(9, "92"), (7, "92"), (6, "93"), (4, "93"), (3, "91")]:
        sg.display_piotroski_score({"f_score": score, "interpretation": "I"})
        out = capsys.readouterr().out
        assert f"\033[{col}m{score}/9" in out and "— I" in out


# ---------- pegy ----------


def _pq(**info):
    return FakeTicker(hist=pd.DataFrame(), info=info)


def test_pegy_error(sg):
    assert sg.compute_pegy_and_breakout(FakeTicker(hist=RuntimeError()), "X") == {
        "symbol": "X",
        "error": "Data fetch failed",
    }


def test_pegy_uptrend(sg):
    r = sg.compute_pegy_and_breakout(
        _pq(
            currentPrice=110.0,
            twoHundredDayAverage=100.0,
            trailingPE=15.234,
            pegRatio=1.5,
            dividendYield=0.04,
        ),
        "S",
    )
    assert r["cmp"] == 110.0 and r["ma_200"] == 100.0 and r["pe_ratio"] == 15.23
    assert r["peg_ratio"] == 1.5 and r["dividend_yield_%"] == 4.0 and r["pegy_adjusted"] == 5.5
    assert r["breakout"] == {"distance_from_200ma_%": 10.0, "signal": "Above 200-day MA - Uptrend"}


@pytest.mark.parametrize(
    "cmp,sig",
    [
        (105.0, "Near 200-day MA - Consolidation"),
        (105.01, "Above 200-day MA - Uptrend"),
        (95.0, "Near 200-day MA - Consolidation"),
        (94.99, "Below 200-day MA - Downtrend"),
        (100.0, "Near 200-day MA - Consolidation"),
    ],
)
def test_pegy_breakout_thresholds(sg, cmp, sig):
    r = sg.compute_pegy_and_breakout(_pq(currentPrice=cmp, twoHundredDayAverage=100.0), "S")
    assert r["breakout"]["signal"] == sig


def test_pegy_missing_inputs(sg):
    r = sg.compute_pegy_and_breakout(_pq(), "S")
    assert r["cmp"] is None and r["pe_ratio"] is None and r["peg_ratio"] is None
    assert (
        r["dividend_yield_%"] is None
        and r["pegy_adjusted"] is None
        and r["ma_200"] is None
        and r["breakout"] == {}
    )
    r = sg.compute_pegy_and_breakout(_pq(currentPrice=10.0, pegRatio=2.0), "S")
    assert (
        r["pegy_adjusted"] == 2.0 and r["breakout"] == {}
    )  # no yield -> PEG only; no MA -> no breakout
    r = sg.compute_pegy_and_breakout(_pq(currentPrice=100.0, twoHundredDayAverage=100.0), "S")
    assert r["breakout"] == {
        "distance_from_200ma_%": None,
        "signal": "Near 200-day MA - Consolidation",
    }


def test_display_pegy_ok(sg, capsys):
    sg.display_pegy_and_breakout(
        {
            "cmp": 110.0,
            "pe_ratio": 15.0,
            "peg_ratio": 1.5,
            "dividend_yield_%": 4.0,
            "pegy_adjusted": 5.5,
            "ma_200": 100.0,
            "breakout": {"distance_from_200ma_%": 10.0, "signal": "Above"},
        }
    )
    out = capsys.readouterr().out
    assert (
        "S$110.00" in out and "4.00%" in out and "5.5" in out and "10.00%" in out and "Above" in out
    )
    sg.display_pegy_and_breakout({"error": "bad"})
    assert "bad" in capsys.readouterr().out
    sg.display_pegy_and_breakout({"dividend_yield_%": 0.0, "breakout": {}})
    out = capsys.readouterr().out
    assert "N/A" in out and "Distance" not in out


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="sg_stock_daily_report.py display_pegy_and_breakout: compute_pegy_and_breakout returns dividend_yield_% = None for non-payers, "
    "and f'{None:.2f}' raises (the .get default 0 is bypassed because the key exists)",
)
def test_display_pegy_none_dividend_yield(sg, capsys):
    r = sg.compute_pegy_and_breakout(_pq(currentPrice=10.0), "S")
    assert r["dividend_yield_%"] is None
    sg.display_pegy_and_breakout(r)
    assert "0.00%" in capsys.readouterr().out


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="sg_stock_daily_report.py display_pegy_and_breakout: distance_from_200ma_% is None when CMP == 200-day MA, and f'{None:.2f}' raises",
)
def test_display_pegy_none_distance(sg, capsys):
    r = sg.compute_pegy_and_breakout(
        _pq(currentPrice=100.0, twoHundredDayAverage=100.0, dividendYield=0.01), "S"
    )
    sg.display_pegy_and_breakout(r)
    assert "Near 200-day" in capsys.readouterr().out


# ---------- displays ----------


def test_display_price_summary(sg, capsys):
    q = {
        "name": "DBS",
        "exchange": "SGX",
        "sector": "Fin",
        "currency": "SGD",
        "lastPrice": 110.0,
        "previousClose": 100.0,
        "dayHigh": 111.0,
        "dayLow": 99.0,
        "52wHigh": 120.0,
        "52wLow": 80.0,
        "200dAvg": 95.0,
        "volume": 1234567.0,
        "marketCap": 3.2e9,
        "trailingPE": 10.0,
        "forwardPE": 9.0,
        "priceToBook": 1.5,
        "dividendYield": 0.055,
    }
    sg.display_price_summary(q)
    out = capsys.readouterr().out
    assert "DBS" in out and "S$110.00" in out and "S$10.00" in out and "▲ 10.00%" in out
    assert "1,234,567" in out and "S$3.20B" in out and "5.50%" in out
    sg.display_price_summary({})
    out = capsys.readouterr().out
    assert out.count("N/A") >= 8 and "Currency" in out and "SGD" in out
    sg.display_price_summary({"lastPrice": 90.0, "previousClose": 100.0})
    assert "▼ 10.00%" in capsys.readouterr().out
    sg.display_price_summary({"lastPrice": 100.0, "previousClose": 100.0})
    assert "Change" in capsys.readouterr().out  # zero change -> chg falsy, pct N/A


def test_display_corporate_actions(sg, capsys):
    a = {
        "calendar": {"ex_dividend_date": "2024-01-01", "dividend_date": "2024-02-01"},
        "dividends": [{"date": "d1", "amount": 0.5}, {"date": "d2", "amount": 0.7}],
    }
    sg.display_corporate_actions(a)
    out = capsys.readouterr().out
    assert "2024-01-01" in out and "2024-02-01" in out and "Recent Dividends" in out
    assert out.index("d2") < out.index("d1")  # newest first
    sg.display_corporate_actions({})
    out = capsys.readouterr().out
    assert "—" in out and "Recent Dividends" not in out


# ---------- run / batch / scans ----------


def _install_fake_ticker(sg, monkeypatch, hist=None):
    hist = (
        hist
        if hist is not None
        else _ohlc(
            [20, 18, 17, 16, 15, 15, 15, 15, 15], [10, 11, 12, 13, 14, 14, 14, 14, 14], [15] * 9
        )
    )
    monkeypatch.setattr(
        sg,
        "_get_ticker",
        lambda s: FakeTicker(
            sym=s,
            hist=hist,
            info={
                "currentPrice": 15.0,
                "previousClose": 14.0,
                "shortName": "N",
                "dividendYield": 0.02,
            },
        ),
    )


def test_run_text_no_scans(sg, monkeypatch, capsys):
    _install_fake_ticker(sg, monkeypatch)
    d = sg.run(" d05 ")
    out = capsys.readouterr().out
    assert d["symbol"] == "D05.SI" and d["scans"] == {} and d["quote"]["lastPrice"] == 15.0
    assert (
        "SINGAPORE STOCK REPORT — D05.SI" in out
        and "LIVE QUOTE" in out
        and "CORPORATE ACTIONS" in out
    )
    assert "DARVAS" not in out and "Data sourced from Yahoo Finance" in out
    datetime.fromisoformat(d["timestamp"])


def test_run_text_with_scans_and_excel_mode(sg, monkeypatch, capsys):
    _install_fake_ticker(sg, monkeypatch)
    monkeypatch.setattr(
        sg,
        "compute_piotroski_score",
        lambda t, s: {"symbol": s, "f_score": 5, "interpretation": "MODERATE"},
    )
    d = sg.run("BN4.SI", run_scans=True)
    out = capsys.readouterr().out
    assert d["symbol"] == "BN4.SI" and set(d["scans"]) == {"darvas", "piotroski", "pegy"}
    assert d["scans"]["darvas"]["signal"] == "IN_BOX"
    assert (
        "Running quantitative scans" in out and "PIOTROSKI F-SCORE" in out and "PEGY RATIO" in out
    )
    d = sg.run("BN4", output_format="excel", run_scans=True)
    out = capsys.readouterr().out
    assert "LIVE QUOTE" not in out and "DARVAS BOX SCAN" not in out  # no text display for excel
    assert d["scans"]["pegy"]["cmp"] == 15.0


def test_run_batch(sg, monkeypatch, capsys, tmp_path):
    calls = []

    def fake_run(sym, output_format="text", run_scans=False):
        calls.append((sym, output_format, run_scans))
        if sym == "BAD.SI":
            raise RuntimeError("nope")
        return {"symbol": sym, "quote": {"name": sym}, "scans": {}}

    monkeypatch.setattr(sg, "run", fake_run)
    wrote = []
    monkeypatch.setattr(sg, "_write_excel_report", lambda r, s: wrote.append((len(r), s)))
    res = sg.run_batch(["A.SI", "BAD.SI", "C.SI"], output_format="excel", run_scans=True)
    out = capsys.readouterr().out
    assert [r["symbol"] for r in res] == ["A.SI", "C.SI"] and wrote == [(2, True)]
    assert (
        "[02/3] BAD" in out
        and "BAD failed: nope" in out
        and "2 OK, 1 failed" in out
        and "Failed: BAD" in out
    )
    assert calls[0] == ("A.SI", "excel", True)
    res = sg.run_batch(["A.SI"], output_format="text")
    assert (
        wrote == [(2, True)] and "Failed:" not in capsys.readouterr().out
    )  # no excel in text mode
    sg.run_batch(["BAD.SI"], output_format="excel")
    assert wrote == [(2, True)]  # nothing succeeded -> no workbook
    sg.run_batch()
    assert len(calls) == 3 + 1 + 1 + len(sg.STI_30)


def test_run_scans_only(sg, monkeypatch, capsys):
    _install_fake_ticker(sg, monkeypatch)
    monkeypatch.setattr(sg, "compute_piotroski_score", lambda t, s: {"symbol": s, "error": "none"})
    r = sg.run_scans_only("d05")
    out = capsys.readouterr().out
    assert r["symbol"] == "D05.SI" and set(r) == {"symbol", "darvas", "piotroski", "pegy"}
    assert "SCANS — D05.SI" in out and "DARVAS BOX SCAN" in out and "none" in out
    assert sg.run_scans_only("U11.SI")["symbol"] == "U11.SI"


def test_summary_helpers(sg):
    assert sg._change_pct({"lastPrice": 110, "previousClose": 100}) == pytest.approx(10.0)
    assert (
        sg._change_pct({"lastPrice": 110}) is None
        and sg._change_pct({"lastPrice": 0, "previousClose": 5}) is None
    )
    assert sg._div_yield_pct({"dividendYield": 0.05}) == pytest.approx(5.0)
    assert sg._div_yield_pct({}) is None and sg._div_yield_pct({"dividendYield": 0}) is None


# ---------- CLI ----------


def _run_cli(sg, monkeypatch, argv):
    calls = {}
    fake_yf = sys.modules["yfinance"]
    monkeypatch.setattr(sys, "argv", ["sg"] + argv)
    # run the script as __main__ with run/run_batch replaced by recorders
    import builtins  # noqa: F401

    src = open(sg.__file__).read()
    g = {"__name__": "__main__", "__file__": sg.__file__}
    code = compile(src, sg.__file__, "exec")
    # pre-exec defs, then swap recorders before the main block: split at the CLI marker
    head, tail = src.split('if __name__ == "__main__":')
    exec(compile(head, sg.__file__, "exec"), g)
    g["run"] = lambda *a, **k: calls.setdefault("run", (a, k))
    g["run_batch"] = lambda *a, **k: calls.setdefault("batch", (a, k))
    exec(compile("if True:" + tail, sg.__file__, "exec"), g)
    del code, fake_yf
    return calls, g


def test_cli_default_symbol(sg, monkeypatch, capsys):
    calls, _ = _run_cli(sg, monkeypatch, [])
    assert calls["run"] == (("D05",), {"output_format": "text", "run_scans": False})
    assert "defaulting to D05" in capsys.readouterr().out


def test_cli_symbol_scans_excel(sg, monkeypatch):
    calls, _ = _run_cli(sg, monkeypatch, ["U11", "--scans", "--output", "excel"])
    assert calls["run"] == (("U11",), {"output_format": "excel", "run_scans": True})


@pytest.mark.parametrize(
    "flag,name", [("--sti30", "STI_30"), ("--sti-top15", "STI_TOP_15"), ("--sgx-all", "SGX_ALL")]
)
def test_cli_batches(sg, monkeypatch, flag, name):
    calls, g = _run_cli(sg, monkeypatch, [flag, "--scans"])
    assert "run" not in calls
    assert calls["batch"] == ((), {"symbols": g[name], "output_format": "text", "run_scans": True})


def test_cli_sgx_all_wins_over_sti30(sg, monkeypatch):
    calls, g = _run_cli(sg, monkeypatch, ["--sti30", "--sgx-all"])
    assert calls["batch"][1]["symbols"] is g["SGX_ALL"]


def test_cli_real_main_block(sg, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["sg", "--sti-top15", "--output", "text"])
    monkeypatch.setattr(
        sys.modules["yfinance"], "Ticker", lambda s: FakeTicker(sym=s, hist=pd.DataFrame(), info={})
    )
    runpy.run_path(sg.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "SG BATCH REPORT — 15 stocks" in out and "15 OK, 0 failed" in out


def test_import_without_yfinance_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "sg_stock_daily_report", raising=False)
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(SystemExit) as e:
        importlib.import_module("sg_stock_daily_report")
    assert "pip install yfinance" in str(e.value)
    monkeypatch.setitem(sys.modules, "yfinance", types.ModuleType("yfinance"))
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(SystemExit) as e:
        importlib.import_module("sg_stock_daily_report")
    assert "pip install openpyxl" in str(e.value)
    sys.modules.pop("sg_stock_daily_report", None)
