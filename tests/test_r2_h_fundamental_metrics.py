import sys
import types

import numpy as np
import pandas as pd
import pytest

import fundamental_metrics as fm


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "CACHE", tmp_path)
    return tmp_path


def _mod(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def test_fetch_one_us(monkeypatch):
    _mod(monkeypatch, "sec_fundamentals", fundamentals=lambda s: {"pe": 1} if s == "A" else None)
    assert fm._fetch_one("US", "A") == {"pe": 1}
    assert fm._fetch_one("US", "B") == {}


def test_fetch_one_in_order(monkeypatch):
    calls = []

    def tl(s):
        calls.append("tl")
        return {}

    def sc(s):
        calls.append("sc")
        return {"roe": 3}

    monkeypatch.setattr(fm, "_in_trendlyne", tl)
    monkeypatch.setattr(fm, "_in_screener", sc)
    assert fm._fetch_one("IN", "X") == {"roe": 3}
    assert calls == ["tl", "sc"]


def test_fetch_one_in_exception_falls_to_yf(monkeypatch):
    def boom(s):
        raise RuntimeError

    monkeypatch.setattr(fm, "_in_trendlyne", boom)
    monkeypatch.setattr(fm, "_in_screener", boom)
    monkeypatch.setattr(fm, "_yf_fundamentals", lambda m, s: {"m": m, "s": s})
    assert fm._fetch_one("IN", "X") == {"m": "IN", "s": "X"}


def test_fetch_one_in_first_wins(monkeypatch):
    monkeypatch.setattr(fm, "_in_trendlyne", lambda s: {"a": 1})
    monkeypatch.setattr(fm, "_in_screener", lambda s: pytest.fail("no"))
    assert fm._fetch_one("IN", "X") == {"a": 1}


def test_fetch_one_other_market(monkeypatch):
    monkeypatch.setattr(fm, "_yf_fundamentals", lambda m, s: {"m": m})
    assert fm._fetch_one("JP", "X") == {"m": "JP"}


def test_in_trendlyne_caches_session(monkeypatch):
    n = {"s": 0}

    def sess():
        n["s"] += 1
        return "SESS"

    _mod(
        monkeypatch,
        "trendlyne_auth",
        session=sess,
        fundamentals=lambda s, ss: {"x": ss} if s == "A" else None,
    )
    monkeypatch.setattr(fm, "_TL_SESSION", None)
    assert fm._in_trendlyne("A") == {"x": "SESS"}
    assert fm._in_trendlyne("B") == {}
    assert n["s"] == 1


def test_in_screener(monkeypatch):
    _mod(
        monkeypatch, "screener_in_auth", company_financials=lambda s: {"a": 1} if s == "A" else None
    )
    assert fm._in_screener("A") == {"a": 1}
    assert fm._in_screener("B") == {}


class _T:
    def __init__(self, info):
        self.info = info


def _yf(monkeypatch, info, seen=None):
    def Ticker(sym):
        if seen is not None:
            seen.append(sym)
        if isinstance(info, Exception):
            raise info
        return _T(info)

    _mod(monkeypatch, "yfinance", Ticker=Ticker)


def test_yf_fundamentals_mapping(monkeypatch):
    seen = []
    _yf(
        monkeypatch,
        {
            "trailingPE": 10.0,
            "returnOnEquity": 0.2,
            "debtToEquity": 150.0,
            "netIncomeToCommon": 5,
            "totalRevenue": 7,
            "trailingEps": 2,
            "marketCap": 100,
            "grossMargins": 0.4,
            "earningsGrowth": 0.1,
        },
        seen,
    )
    out = fm._yf_fundamentals("JP", "7203")
    assert seen == ["7203.T"]
    assert out["source"] == "yfinance"
    assert out["pe"] == 10.0
    assert out["roe"] == pytest.approx(20.0)
    assert out["debt_to_equity"] == pytest.approx(1.5)
    assert out["gross_margin"] == pytest.approx(40.0)
    assert out["profit_growth"] == pytest.approx(10.0)
    assert out["market_cap"] == 100


def test_yf_fundamentals_de_boundary_and_fallbacks(monkeypatch):
    seen = []
    _yf(monkeypatch, {"forwardPE": float("nan"), "debtToEquity": 5, "trailingPE": ""}, seen)
    out = fm._yf_fundamentals("ZZ", "AAA")  # unknown suffix
    assert seen == ["AAA"]
    assert out["debt_to_equity"] == 5  # not > 5 -> unchanged
    assert "pe" not in out and "roe" not in out and "gross_margin" not in out
    _yf(monkeypatch, {"forwardPE": 7, "debtToEquity": 5.01})
    out = fm._yf_fundamentals("US", "A")
    assert out["pe"] == 7
    assert out["debt_to_equity"] == pytest.approx(0.0501)


def test_yf_fundamentals_empty_and_error(monkeypatch):
    _yf(monkeypatch, {})
    assert fm._yf_fundamentals("JP", "A") == {}
    _yf(monkeypatch, RuntimeError("x"))
    assert fm._yf_fundamentals("JP", "A") == {}


@pytest.fixture
def serving(monkeypatch):
    def install(df):
        _mod(monkeypatch, "serving_layer", serving=lambda m: df)

    return install


def test_liquid_symbols(serving):
    serving(
        pd.DataFrame({"Symbol": ["A", "B", "C"], "Turnover_USD": [1, 3, 2], "Close": [1, 2, 3]})
    )
    assert fm._liquid_symbols("US", 2) == ["B", "C"]
    serving(pd.DataFrame({"Symbol": ["A", "B"]}))
    assert fm._liquid_symbols("US", 1) == ["A"]
    serving(pd.DataFrame())
    assert fm._liquid_symbols("US", 1) == []


def test_prices(serving):
    serving(pd.DataFrame({"Symbol": ["A", "B"], "Close": [1.5, 2.5]}))
    assert fm._prices("US") == {"A": 1.5, "B": 2.5}
    serving(pd.DataFrame())
    assert fm._prices("US") == {}


def test_load_fundamentals_fetch_cache_and_price(cache, serving, monkeypatch, capsys):
    serving(
        pd.DataFrame(
            {
                "Symbol": ["A", "B", "C"],
                "Turnover_USD": [3, 2, 1],
                "Close": [10.0, 20.0, 30.0],
            }
        )
    )
    fetched = []

    def fo(m, s):
        fetched.append(s)
        return {"shares": 2.0, "pe": 1} if s != "B" else {}

    monkeypatch.setattr(fm, "_fetch_one", fo)
    out = fm.load_fundamentals("JP", limit=2)
    assert fetched == ["A", "B"]
    assert list(out["Symbol"]) == ["A"]
    assert out["market_cap"].iloc[0] == 20.0
    assert "+1 fetched, 1 total cached" in capsys.readouterr().out
    # second call skips A (cached) and fetches B? B had no data -> retried
    fetched.clear()
    out = fm.load_fundamentals("JP", limit=5, verbose=False)
    assert fetched == ["B", "C"]
    assert sorted(out["Symbol"]) == ["A", "C"]
    assert capsys.readouterr().out == ""


def test_load_fundamentals_explicit_symbols_and_checkpoint(cache, serving, monkeypatch, capsys):
    serving(pd.DataFrame())
    syms = [f"S{i}" for i in range(26)]
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {"pe": 1})
    sleeps = []
    monkeypatch.setattr(fm.time, "sleep", lambda t: sleeps.append(t))
    out = fm.load_fundamentals("US", limit=100, symbols=syms)
    assert len(out) == 26
    assert "fetched 25/26" in capsys.readouterr().out
    assert sleeps == [0.12] * 26
    assert "price" in out and out["price"].isna().all()


def test_load_fundamentals_nothing(cache, serving):
    serving(pd.DataFrame())
    out = fm.load_fundamentals("JP", verbose=False)
    assert out.empty


def test_load_fundamentals_limit_applies_after_filter(cache, serving, monkeypatch):
    serving(pd.DataFrame())
    monkeypatch.setattr(fm, "_fetch_one", lambda m, s: {"pe": 1})
    out = fm.load_fundamentals("JP", limit=2, symbols=["a", "b", "c"], verbose=False)
    assert sorted(out["Symbol"]) == ["a", "b"]


def test_piotroski_each_point_and_missing():
    assert fm._piotroski({}) == 0
    assert fm._piotroski({"net_income": 1}) == 1
    assert fm._piotroski({"net_income": 0}) == 0
    assert fm._piotroski({"roa": 0.1}) == 1
    assert fm._piotroski({"roa": 0}) == 0
    assert fm._piotroski({"cfo": 1}) == 1
    assert fm._piotroski({"cfo": 2, "net_income": 1}) == 3
    assert fm._piotroski({"cfo": 1, "net_income": 1}) == 2  # not strictly greater
    assert fm._piotroski({"debt_to_assets": 1, "debt_to_assets_prev": 2}) == 1
    assert fm._piotroski({"debt_to_assets": 2, "debt_to_assets_prev": 2}) == 0
    assert fm._piotroski({"current_ratio": 3, "current_ratio_prev": 2}) == 1
    assert fm._piotroski({"current_ratio": 2, "current_ratio_prev": 2}) == 0
    assert fm._piotroski({"shares": 2, "shares_prev": 2}) == 1
    assert fm._piotroski({"shares": 3, "shares_prev": 2}) == 0
    assert fm._piotroski({"gross_margin": 3, "gross_margin_prev": 2}) == 1
    assert fm._piotroski({"gross_margin": 2, "gross_margin_prev": 2}) == 0
    assert fm._piotroski({"asset_turnover": 3, "asset_turnover_prev": 2}) == 1
    assert fm._piotroski({"asset_turnover": 2, "asset_turnover_prev": 2}) == 0
    assert fm._piotroski({"net_income": float("nan"), "roa": None}) == 0


def test_piotroski_perfect():
    r = dict(
        net_income=1, roa=1, cfo=2, debt_to_assets=1, debt_to_assets_prev=2,
        current_ratio=2, current_ratio_prev=1, shares=1, shares_prev=1,
        gross_margin=2, gross_margin_prev=1, asset_turnover=2, asset_turnover_prev=1,
    )  # fmt: skip
    assert fm._piotroski(r) == 9


def test_roce_and_helpers():
    assert fm._roce({"ebit": 30, "total_assets": 100}) == 30
    assert fm._roce({"ebit": 30, "assets": 200}) == 15
    assert fm._roce({"ebit": 30, "total_assets": 0, "assets": 100}) == 30
    assert fm._roce({"ebit": 0, "roa": 4}) == 4
    assert fm._roce({}) is None
    assert fm._first_val([5, 6]) == 5
    assert fm._first_val(()) is None
    assert fm._first_val(np.array([7.0])) == 7.0
    assert fm._first_val(np.array([])) is None
    assert fm._first_val(None) is None
    assert fm._first_val(float("nan")) is None
    assert fm._first_val(3) == 3
    assert fm._earnings_yield({"ebit": 8, "market_cap": 100}) == 8
    assert fm._earnings_yield({"ebit": 8}) is None
    assert fm._earnings_yield({"ebit": 0, "market_cap": 100}) is None


def test_fcf_yield():
    assert fm._fcf_yield({"cfo": 10, "capex_history": [4], "market_cap": 100}) == 6
    assert fm._fcf_yield({"cfo": 10, "market_cap": 100}) == 10
    assert fm._fcf_yield({"cfo": 10, "capex_history": [0], "market_cap": 100}) == 10
    assert fm._fcf_yield({"market_cap": 100}) is None
    assert fm._fcf_yield({"cfo": 1, "market_cap": 0}) is None
    assert fm._fcf_yield({"cfo": 1}) is None
    assert fm._fcf_yield({"cfo": 1, "market_cap": float("nan")}) is None
    assert fm._fcf_yield({"cfo": 0, "market_cap": 100}) == 0


def test_dividend_yield():
    assert fm._dividend_yield({"dividend_history": [-3], "market_cap": 100}) == 3
    assert fm._dividend_yield({"dividend_history": [3], "market_cap": 100}) == 3
    assert fm._dividend_yield({"dividend_history": [0], "market_cap": 100}) is None
    assert fm._dividend_yield({"dividend_history": [3]}) is None
    assert fm._dividend_yield({"dividend_history": [3], "market_cap": float("nan")}) is None
    assert fm._dividend_yield({"market_cap": 5}) is None


def _frame():
    return pd.DataFrame(
        [
            {"Symbol": "A", "net_income": 1, "roa": 1, "cfo": 2, "ebit": 30, "total_assets": 100,
             "market_cap": 100, "roe": 25, "debt_to_equity": 0.05, "eps_growth": 20,
             "capex_history": [1], "dividend_history": [4]},
            {"Symbol": "B", "net_income": -1, "ebit": 10, "total_assets": 100, "market_cap": 200,
             "roe": 10, "debt_to_equity": 0.4, "eps_growth": 5},
        ]
    )  # fmt: skip


def test_enrich_and_screens():
    df = fm._enrich(_frame())
    assert df["piotroski"].tolist() == [4, 0]
    assert df["roce"].tolist() == [30, 10]
    m = {k: v(df).fillna(False).tolist() for k, v in fm.FUND_SCREENS.items()}
    assert m["high_roce"] == [True, False]
    assert m["high_roe"] == [True, False]
    assert m["debt_free"] == [True, False]
    assert m["low_debt_equity"] == [True, False]
    assert m["magic_formula"] == [True, False]
    assert m["fcf_yield"] == [False, False]
    assert m["high_dividend"] == [True, False]
    assert m["eps_growth"] == [True, False]
    assert m["coffee_can"] == [True, False]
    assert m["piotroski_9"] == [False, False]
    # boundaries
    d = pd.DataFrame({"roce": [20.0, 20.1], "piotroski": [7, 8]})
    assert fm.FUND_SCREENS["high_roce"](d).tolist() == [False, True]
    assert fm.FUND_SCREENS["piotroski_9"](d).tolist() == [False, True]


def test_run_all(cache, serving, monkeypatch, capsys):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit, verbose: _frame())
    counts = fm.run_all("US")
    assert counts["high_roce"] == 1 and counts["debt_free"] == 1
    out = capsys.readouterr().out
    assert "fundamental screens on US (2 names" in out
    assert "high_roce" in out
    counts = fm.run_all("US", verbose=False)
    assert capsys.readouterr().out == ""
    assert set(counts) == set(fm.FUND_SCREENS)


def test_run_all_empty(monkeypatch, capsys):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit, verbose: pd.DataFrame())
    assert fm.run_all("US") == {}
    assert "no fundamentals for US" in capsys.readouterr().out
    assert fm.run_all("US", verbose=False) == {}
    assert capsys.readouterr().out == ""


def test_run_all_predicate_exception(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit, verbose: _frame())
    monkeypatch.setitem(fm.FUND_SCREENS, "boom", lambda d: 1 / 0)
    counts = fm.run_all("US", verbose=False)
    assert counts["boom"] == 0


def test_run_screen(monkeypatch):
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit: _frame())
    out = fm.run_screen("high_roce", "US")
    assert out["Symbol"].tolist() == ["A"]
    assert "piotroski" in out.columns and "roce" in out.columns
    assert fm.run_screen("nope", "US").empty
    # sorted by piotroski for other screens and top handling
    out = fm.run_screen("low_debt_equity", "US", top=1)
    assert out["Symbol"].tolist() == ["A"]
    out = fm.run_screen("low_debt_equity", "US", top=None)
    assert out["Symbol"].tolist() == ["A"]
    monkeypatch.setattr(fm, "load_fundamentals", lambda m, limit: pd.DataFrame())
    assert fm.run_screen("high_roce", "US").empty


def test_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fm", "--market", "IN", "--limit", "3"])
    seen = {}
    monkeypatch.setattr(fm, "run_all", lambda m, limit: seen.update(m=m, limit=limit))
    assert fm.main() == 0
    assert seen == {"m": "IN", "limit": 3}
    monkeypatch.setattr(sys, "argv", ["fm", "--screen", "high_roce", "--top", "2"])
    monkeypatch.setattr(fm, "run_screen", lambda n, m, limit, top: _frame().head(top))
    assert fm.main() == 0
    out = capsys.readouterr().out
    assert "high_roce on US — 2 names" in out
    monkeypatch.setattr(fm, "run_screen", lambda n, m, limit, top: pd.DataFrame())
    assert fm.main() == 0
    assert "none / no fundamentals" in capsys.readouterr().out
