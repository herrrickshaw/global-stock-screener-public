import types

import numpy as np
import pandas as pd
import pytest

import quarterly_data_analyzer as qa

A = qa.QuarterlyAnalyzer()
D = pd.Timestamp("2024-03-31")


def _frame(rows, dates=(D,)):
    return pd.DataFrame(rows, index=list(dates)).T


def test_val():
    df = _frame({"Total Revenue": [5.0]})
    assert A._val(df, "Total Revenue", D) == 5.0
    assert A._val(df, "Missing", D) is None


def test_income_metrics():
    df = _frame(
        {
            "Total Revenue": [200.0],
            "Operating Income": [50.0],
            "Net Income": [20.0],
            "Interest Expense": [3.0],
        }
    )
    m, st = {}, {}
    A._income_metrics(df, D, m, st)
    assert m == {
        "revenue": 200.0,
        "operating_income": 50.0,
        "net_income": 20.0,
        "interest_expense": 3.0,
        "oi_margin": 25.0,
        "ni_margin": 10.0,
    }
    assert st["total_revenue"] == 200.0 and st["net_income"] == 20.0


def test_income_metrics_missing_and_zero():
    df = _frame({"Total Revenue": [100.0], "Operating Income": [0.0], "Net Income": [-5.0]})
    m, st = {}, {}
    A._income_metrics(df, D, m, st)
    assert "oi_margin" not in m  # zero op income is falsy
    assert m["ni_margin"] == -5.0  # negative still computed
    assert m["interest_expense"] is None
    m, st = {}, {}
    A._income_metrics(_frame({"Operating Income": [5.0]}), D, m, st)
    assert "oi_margin" not in m and "ni_margin" not in m


def test_cashflow_metrics():
    df = _frame({"Operating Cash Flow": [100.0], "Capital Expenditure": [-30.0]})
    m = {}
    A._cashflow_metrics(df, D, m, {"total_revenue": 200.0})
    assert m == {"ocf": 100.0, "capex": 30.0, "fcf": 70.0, "fcf_margin": 35.0}
    m = {}
    A._cashflow_metrics(df, D, m, {"total_revenue": 0})
    assert m["fcf"] == 70.0 and m["fcf_margin"] is None
    m = {}
    A._cashflow_metrics(df, D, m, {"total_revenue": None})
    assert m["fcf_margin"] is None
    # positive capex is abs()'d too
    m = {}
    A._cashflow_metrics(
        _frame({"Operating Cash Flow": [10.0], "Capital Expenditure": [4.0]}),
        D,
        m,
        {"total_revenue": 10.0},
    )
    assert m["fcf"] == 6.0 and m["capex"] == 4.0
    m = {}
    A._cashflow_metrics(_frame({"Operating Cash Flow": [10.0]}), D, m, {"total_revenue": 10.0})
    assert m == {"ocf": 10.0, "capex": None}
    m = {}
    A._cashflow_metrics(
        _frame({"Capital Expenditure": [0.0], "Operating Cash Flow": [10.0]}), D, m, {}
    )
    assert m["capex"] is None and "fcf" not in m


def _bs(**kw):
    return _frame({k.replace("_", " "): [v] for k, v in kw.items()})


def test_balance_metrics_full():
    df = _frame(
        {
            "Total Assets": [1000.0],
            "Total Debt": [200.0],
            "Total Stockholder Equity": [400.0],
            "Current Assets": [300.0],
            "Current Liabilities": [150.0],
        }
    )
    m, st = {}, {"total_revenue": 500.0}
    A._balance_metrics(df, D, m, st)
    assert m["current_ratio"] == 2.0
    assert m["de_ratio"] == 0.5
    assert m["asset_turnover"] == 0.5
    assert st["total_debt"] == 200.0 and st["total_equity"] == 400.0


def test_balance_metrics_edges():
    # no debt -> de 0; zero/neg equity -> no de_ratio; missing revenue -> asset_turnover None
    m, st = {}, {"total_revenue": None}
    A._balance_metrics(
        _frame({"Total Assets": [10.0], "Total Stockholder Equity": [5.0]}), D, m, st
    )
    assert m["de_ratio"] == 0 and m["asset_turnover"] is None
    assert m["current_ratio"] is None
    for eq in (0.0, -1.0):
        m = {}
        A._balance_metrics(
            _frame({"Total Assets": [-1.0], "Total Debt": [1.0], "Total Stockholder Equity": [eq]}),
            D, m, {"total_revenue": 1.0},
        )  # fmt: skip
        assert "de_ratio" not in m and "asset_turnover" not in m
    m = {}
    A._balance_metrics(_frame({"Current Assets": [1.0]}), D, m, {"total_revenue": 1.0})
    assert m["current_ratio"] is None  # needs both rows


def test_roic_metric():
    m = {}
    st = {"operating_income": 100.0, "total_debt": 300.0, "total_equity": 100.0, "net_income": 60.0}
    A._roic_metric(m, st)
    assert m["roic"] == pytest.approx(60.0 / 400 * 100)  # NOPAT = 100*(1-(1-.6)) = 60
    m = {}
    A._roic_metric(m, {**st, "operating_income": -10.0})
    assert m["roic"] == pytest.approx(-10 * 0.75 * 1 / 400 * 100)
    for missing in ("operating_income", "total_debt", "total_equity"):
        m = {}
        A._roic_metric(m, {**st, missing: 0})
        assert m == {}
    m = {}
    A._roic_metric(m, {**st, "total_debt": -500.0})  # invested capital <= 0
    assert m == {}


# ── fetch_company_data ───────────────────────────────────────────────────────────
class T:
    def __init__(self, is_, cf, bs):
        self.quarterly_income_stmt, self.quarterly_cashflow, self.quarterly_balance_sheet = (
            is_,
            cf,
            bs,
        )


def _patch_ticker(monkeypatch, ticker=None, exc=None):
    def Ticker(sym):
        if exc:
            raise exc
        return ticker

    monkeypatch.setattr(qa, "yf", types.SimpleNamespace(Ticker=Ticker))


def _stmts(n):
    dates = pd.date_range("2025-12-31", periods=n, freq="-3ME")
    is_ = pd.DataFrame(
        {
            d: {"Total Revenue": 100.0 + i, "Operating Income": 20.0, "Net Income": 10.0}
            for i, d in enumerate(dates)
        }
    )
    cf = pd.DataFrame(
        {d: {"Operating Cash Flow": 15.0, "Capital Expenditure": -5.0} for d in dates}
    )
    bs = pd.DataFrame(
        {
            d: {"Total Assets": 500.0, "Total Debt": 100.0, "Total Stockholder Equity": 200.0}
            for d in dates
        }
    )
    return is_, cf, bs


def test_fetch_none_inputs(monkeypatch):
    is_, cf, bs = _stmts(5)
    for args in ((None, cf, bs), (is_, None, bs), (is_, cf, None)):
        _patch_ticker(monkeypatch, T(*args))
        assert A.fetch_company_data("X") is None


def test_fetch_too_few_quarters(monkeypatch):
    _patch_ticker(monkeypatch, T(*_stmts(3)))
    assert A.fetch_company_data("X") is None
    _patch_ticker(monkeypatch, T(*_stmts(4)))
    out = A.fetch_company_data("X")
    assert out is not None and len(out["quarters"]) == 4


def test_fetch_caps_at_20_and_computes(monkeypatch):
    _patch_ticker(monkeypatch, T(*_stmts(24)))
    out = A.fetch_company_data("X")
    assert out["ticker"] == "X"
    assert len(out["quarters"]) == 20
    q = out["quarters"][0]
    assert q["revenue"] == 100.0 and q["oi_margin"] == 20.0 and q["fcf"] == 10.0
    assert q["de_ratio"] == 0.5
    assert q["roic"] == pytest.approx(20 * 0.5 / 300 * 100)  # NOPAT = 20*(1-(1-0.5)) = 10
    assert q["quarter_label"].startswith("2025-Q")
    assert out["trends"]["latest_de"] == 0.5


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="quarterly_data_analyzer.py:~123 strftime('%Y-Q%q') -- %q is not a strftime "
    "directive, so labels come out as '2025-Qq' instead of '2025-Q4'",
)
def test_quarter_label_should_name_the_quarter(monkeypatch):
    _patch_ticker(monkeypatch, T(*_stmts(4)))
    out = A.fetch_company_data("X")
    assert out["quarters"][0]["quarter_label"] == "2025-Q4"


def test_fetch_section_failures_are_isolated(monkeypatch):
    is_, cf, bs = _stmts(5)
    cf = cf.iloc[0:0]  # no rows -> sections that need them just yield None
    _patch_ticker(monkeypatch, T(is_, cf, bs))
    out = A.fetch_company_data("X")
    assert out["quarters"][0]["ocf"] is None and out["quarters"][0]["revenue"] == 100.0


def test_fetch_section_exception_swallowed(monkeypatch):
    _patch_ticker(monkeypatch, T(*_stmts(5)))

    def boom(self, *a, **k):
        raise RuntimeError

    monkeypatch.setattr(qa.QuarterlyAnalyzer, "_cashflow_metrics", boom)
    monkeypatch.setattr(qa.QuarterlyAnalyzer, "_roic_metric", staticmethod(boom))
    out = A.fetch_company_data("X")
    assert "ocf" not in out["quarters"][0] and "roic" not in out["quarters"][0]
    assert out["quarters"][0]["revenue"] == 100.0


def test_fetch_ticker_exception(monkeypatch):
    _patch_ticker(monkeypatch, exc=RuntimeError("net"))
    assert A.fetch_company_data("X") is None


# ── trends ───────────────────────────────────────────────────────────────────────
def _q(i, **kw):
    return {"date": pd.Timestamp("2024-01-01") - pd.Timedelta(days=90 * i), **kw}


def test_trends_too_few():
    assert A._calculate_trends([_q(i) for i in range(3)]) == {}


def test_trends_values():
    qs = [
        _q(0, revenue=200.0, capex=20.0, total_debt=50.0, fcf=10.0, ocf=30.0, oi_margin=25.0,
           ni_margin=12.0, roic=9.0, de_ratio=0.4, fcf_margin=5.0, asset_turnover=0.7,
           interest_expense=2.0),
        _q(1, revenue=150.0, capex=10.0, total_debt=60.0, fcf=20.0, ocf=40.0, oi_margin=22.0,
           ni_margin=11.0, roic=8.0),
        _q(2, revenue=120.0),
        _q(3, revenue=100.0, capex=5.0, total_debt=40.0, fcf=0, ocf=20.0, oi_margin=20.0,
           ni_margin=10.0, roic=5.0),
    ]  # fmt: skip
    t = A._calculate_trends(list(reversed(qs)))  # input order must not matter
    assert t["revenue_cagr"] == pytest.approx(((200 / 100) ** (1 / 5) - 1) * 100)
    assert t["capex_cagr"] == pytest.approx(((20 / 5) ** (1 / 5) - 1) * 100)
    assert t["debt_cagr"] == pytest.approx(((50 / 40) ** (1 / 5) - 1) * 100)
    assert t["avg_fcf"] == 15.0  # zero fcf excluded
    assert t["avg_ocf"] == 30.0
    assert t["avg_capex"] == pytest.approx(35 / 3)
    assert t["oi_margin_change"] == 5.0
    assert t["ni_margin_change"] == 2.0
    assert t["roic_change"] == 4.0
    assert t["latest_de"] == 0.4 and t["latest_fcf_margin"] == 5.0
    assert t["latest_asset_turnover"] == 0.7 and t["latest_roic"] == 9.0
    assert t["interest_coverage"] == pytest.approx(30.0 / 2.01)


def test_trends_sparse():
    qs = [_q(i) for i in range(4)]
    t = A._calculate_trends(qs)
    assert "revenue_cagr" not in t and "oi_margin_change" not in t
    assert np.isnan(t["avg_fcf"])
    assert "interest_coverage" not in t
    assert t["latest_de"] is None
    # needs both ocf and interest
    qs[0].update(ocf=10.0)
    assert "interest_coverage" not in A._calculate_trends(qs)


def test_trends_cagr_nonpositive_oldest():
    qs = [_q(0, revenue=10.0), _q(1, revenue=5.0), _q(2), _q(3, revenue=-3.0)]
    # negative revenue is truthy -> used as oldest -> safe_cagr None
    assert A._calculate_trends(qs)["revenue_cagr"] is None


# ── scoring ──────────────────────────────────────────────────────────────────────
def test_score_empty():
    assert A.score_company(None) == 0
    assert A.score_company({}) == 0
    assert A.score_company({"trends": {}}) == 0


@pytest.mark.parametrize(
    "trend,pts",
    [
        ({"debt_cagr": 0.1}, 10),
        ({"debt_cagr": 14.9}, 10),
        ({"debt_cagr": 15.0}, 0),
        ({"debt_cagr": -1.0}, 0),
        ({"capex_cagr": 15.1}, 20),
        ({"capex_cagr": 15.0}, 0),
        ({"revenue_cagr": 10.1}, 15),
        ({"revenue_cagr": 10.0}, 0),
        ({"ni_margin_change": -1.9}, 15),
        ({"ni_margin_change": -2.0}, 0),
        ({"avg_fcf": 1.0}, 15),
        ({"avg_fcf": -1.0}, 0),
        ({"latest_de": 0.5}, 10),
        ({"latest_de": 1.0}, 0),
        ({"latest_fcf_margin": 2.1}, 5),
        ({"latest_fcf_margin": 2.0}, 0),
        ({"roic_change": 0.1}, 5),
        ({"roic_change": 0.0}, 0),
    ],
)
def test_score_each_rule(trend, pts):
    assert A.score_company({"trends": trend}) == pts


def test_score_total_capped():
    t = {
        "debt_cagr": 5, "capex_cagr": 20, "revenue_cagr": 20, "ni_margin_change": 1,
        "avg_fcf": 5, "latest_de": 0.5, "latest_fcf_margin": 5, "roic_change": 1,
    }  # fmt: skip
    assert A.score_company({"trends": t}) == 95


# ── analyze_portfolio ────────────────────────────────────────────────────────────
def test_analyze_portfolio(monkeypatch, capsys):
    good = {
        "revenue_cagr": 20.0, "capex_cagr": 30.0, "debt_cagr": 5.0, "avg_fcf": 2e9,
        "roic_change": 1.0, "latest_de": 0.5, "ni_margin_change": 1.0, "latest_fcf_margin": 9,
    }  # fmt: skip
    weak = {
        "revenue_cagr": 1.0,
        "capex_cagr": 0.0,
        "debt_cagr": 0.0,
        "avg_fcf": 0,
        "roic_change": -1.0,
    }
    mid = {
        "revenue_cagr": 20.0,
        "capex_cagr": 20.0,
        "debt_cagr": 3.0,
        "roic_change": -1.0,
    }  # 10+20+15=45...
    data = {
        "GOOD": {"trends": good},
        "WEAK": {"trends": weak},
        "MID": {"trends": {**mid, "avg_fcf": 1.0}},
    }

    def fetch(self, t):
        return data.get(t)

    monkeypatch.setattr(qa.QuarterlyAnalyzer, "fetch_company_data", fetch)
    df = qa.QuarterlyAnalyzer().analyze_portfolio(["GOOD", "NONE", "WEAK", "MID"])
    out = capsys.readouterr().out
    assert df["ticker"].tolist() == ["GOOD", "MID", "WEAK"]
    assert df["score"].tolist() == [95, 60, 0]
    assert "3/4 successful" in out
    assert "High expansion (75+): 1" in out
    assert "Medium expansion (50-75): 1" in out
    assert "Low expansion (<50): 1" in out
    assert "Companies with improving ROIC: 1" in out
    assert "$2.00B" in out and "20.0%" in out
    assert "N/A" in out  # WEAK has falsy cagr
    assert "Avg revenue CAGR: 13.7%" in out


def test_analyze_portfolio_boundaries(monkeypatch, capsys):
    # scores exactly 75 -> high; exactly 50 -> medium
    t75 = {
        "debt_cagr": 5,
        "capex_cagr": 20,
        "revenue_cagr": 20,
        "avg_fcf": 1,
        "ni_margin_change": 1,
    }  # 10+20+15+15+15=75
    t50 = {"capex_cagr": 20, "revenue_cagr": 20, "avg_fcf": 1, "roic_change": -1}  # 20+15+15=50
    data = {"A": {"trends": t75}, "B": {"trends": t50}}
    monkeypatch.setattr(qa.QuarterlyAnalyzer, "fetch_company_data", lambda self, t: data[t])
    df = qa.QuarterlyAnalyzer().analyze_portfolio(["A", "B"])
    assert df["score"].tolist() == [75, 50]
    out = capsys.readouterr().out
    assert "High expansion (75+): 1" in out
    assert "Medium expansion (50-75): 1" in out
    assert "Low expansion (<50): 0" in out
