import types

import numpy as np
import pandas as pd
import pytest

import quarterly_data_collector as qc


@pytest.fixture
def col():
    return qc.QuarterlyDataCollector()


def quarters(n, start="2020-03-31"):
    # yfinance order: newest first
    return list(pd.date_range(start, periods=n, freq="QE"))[::-1]


def stmts(n, rev=lambda i: 100.0 + i, capex=True):
    qs = quarters(n)
    idx = lambda d: {c: d[k] for k, c in enumerate(qs)}  # noqa: E731
    r = list(range(n))
    inc = pd.DataFrame(
        {
            "Total Revenue": [rev(n - 1 - k) for k in r],
            "Operating Income": [20.0] * n,
            "Net Income": [10.0] * n,
        },
        index=qs,
    ).T
    cfd = {"Operating Cash Flow": [50.0] * n}
    if capex:
        cfd["Capital Expenditure"] = [-5.0] * n
    cf = pd.DataFrame(cfd, index=qs).T
    bs = pd.DataFrame(
        {
            "Total Assets": [1000.0] * n,
            "Total Debt": [200.0] * n,
            "Total Stockholder Equity": [400.0] * n,
        },
        index=qs,
    ).T
    return types.SimpleNamespace(
        quarterly_cashflow=cf, quarterly_income_stmt=inc, quarterly_balance_sheet=bs
    )


def test_select_sample_companies(col):
    allt = col.select_sample_companies(1000)
    assert len(allt) >= 60 and len(set(allt)) == len(allt)
    assert col.select_sample_companies() == allt[:60]
    assert col.select_sample_companies(5) == allt[:5]
    assert col.select_sample_companies(0) == []


def test_val(col):
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    assert col._val(df, "y", "a") == 2
    assert col._val(df, "zz", "a") is None


def test_quarter_metrics_fields_and_label(col):
    s = stmts(1)
    q = s.quarterly_cashflow.columns[0]
    m = col._quarter_metrics(
        "T", q, s.quarterly_income_stmt, s.quarterly_cashflow, s.quarterly_balance_sheet
    )
    assert m == {
        "ticker": "T",
        "quarter": "2020-Q1",
        "date": q,
        "revenue": 100.0,
        "operating_income": 20.0,
        "net_income": 10.0,
        "ocf": 50.0,
        "capex": 5.0,  # abs of negative outflow
        "total_assets": 1000.0,
        "total_debt": 200.0,
        "total_equity": 400.0,
    }
    for month, qn in [(1, 1), (3, 1), (4, 2), (6, 2), (7, 3), (9, 3), (10, 4), (12, 4)]:
        ts = pd.Timestamp(2021, month, 15)
        out = col._quarter_metrics("T", ts, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        assert out["quarter"] == f"2021-Q{qn}"


def test_quarter_metrics_missing_rows_and_blocks(col):
    q = pd.Timestamp("2020-03-31")
    m = col._quarter_metrics("T", q, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert all(m[k] is None for k in ("revenue", "ocf", "capex", "total_assets", "total_equity"))
    # a missing column raises inside the try -> that block alone is nulled
    s = stmts(1)
    other = pd.Timestamp("1999-01-01")
    m = col._quarter_metrics(
        "T", other, s.quarterly_income_stmt, s.quarterly_cashflow, s.quarterly_balance_sheet
    )
    assert m["revenue"] is None and m["ocf"] is None and m["capex"] is None
    assert m["total_assets"] is None and m["total_debt"] is None


def test_last_20_quarters(col):
    s = stmts(25)
    cf, inc, bs = col._last_20_quarters(s)
    assert cf.shape[1] == inc.shape[1] == bs.shape[1] == 20
    assert cf.columns[0] == s.quarterly_cashflow.columns[0]  # newest kept
    s = stmts(20)
    assert col._last_20_quarters(s)[0].shape[1] == 20
    s.quarterly_balance_sheet = pd.DataFrame()
    assert col._last_20_quarters(s) is None
    s = stmts(3)
    s.quarterly_cashflow = None
    assert col._last_20_quarters(s) is None
    s = stmts(3)
    s.quarterly_income_stmt = None
    assert col._last_20_quarters(s) is None


def patch_yf(monkeypatch, stock=None, exc=None):
    def Ticker(t):
        if exc:
            raise exc
        return stock

    monkeypatch.setattr(qc.yf, "Ticker", Ticker)


def test_fetch_quarterly_data_sorted_and_capped(col, monkeypatch):
    patch_yf(monkeypatch, stmts(25))
    df = col.fetch_quarterly_data("T")
    assert len(df) == 20
    assert df["date"].is_monotonic_increasing and df.index.tolist() == list(range(20))
    assert set(df["ticker"]) == {"T"}


def test_fetch_quarterly_data_empty_paths(col, monkeypatch):
    s = stmts(3)
    s.quarterly_cashflow = pd.DataFrame()
    patch_yf(monkeypatch, s)
    assert col.fetch_quarterly_data("T").empty
    patch_yf(monkeypatch, exc=RuntimeError("net"))
    assert col.fetch_quarterly_data("T").empty
    bad = types.SimpleNamespace()  # AttributeError inside inner try -> swallowed
    patch_yf(monkeypatch, bad)
    assert col.fetch_quarterly_data("T").empty


def frame(n=8, rev=None, **over):
    d = pd.DataFrame(
        {
            "revenue": rev if rev is not None else [100.0 * (1.1**i) for i in range(n)],
            "capex": [10.0] * n,
            "total_debt": [50.0] * n,
            "ocf": [30.0] * n,
            "total_equity": [100.0] * n,
            "operating_income": [20.0] * n,
            "net_income": [10.0] * n,
        }
    )
    for k, v in over.items():
        d[k] = v
    return d


def test_calculate_metrics_small_inputs(col):
    assert col.calculate_metrics(pd.DataFrame()) == {}
    assert col.calculate_metrics(frame(3)) == {}
    assert col.calculate_metrics(frame(4)) != {}  # boundary: exactly 4 rows


def test_calculate_metrics_basic(col):
    m = col.calculate_metrics(frame(8))
    assert m["avg_ocf"] == 30 and m["avg_capex"] == 10 and m["avg_fcf"] == 20
    assert m["latest_de"] == 0.5
    assert m["oi_margin_trend"] == pytest.approx(0) or "oi_margin_trend" in m
    assert m["capex_cagr"] == pytest.approx(0.0)  # flat capex
    assert m["debt_cagr"] == pytest.approx(0.0)


def test_calculate_metrics_latest_de_conditions(col):
    assert "latest_de" not in col.calculate_metrics(frame(5, total_equity=0.0))
    assert "latest_de" not in col.calculate_metrics(frame(5, total_equity=-5.0))
    m = col.calculate_metrics(
        frame(5, total_equity=[1, 1, 1, 1, 100.0], total_debt=[0, 0, 0, 0, 25.0])
    )
    assert m["latest_de"] == 0.25


def test_calculate_metrics_margin_trends(col):
    rev = [100.0, 100.0, 100.0, 200.0]
    oi = [20.0, 20.0, 20.0, 20.0]  # margins 20,20,20,10
    ni = [10.0, 10.0, 10.0, 40.0]  # 10,10,10,20
    m = col.calculate_metrics(frame(4, rev=rev, operating_income=oi, net_income=ni))
    assert m["oi_margin_trend"] == pytest.approx(10.0)  # first - last
    assert m["ni_margin_trend"] == pytest.approx(-10.0)


def test_calculate_metrics_nan_handling(col):
    d = frame(6)
    d["capex"] = [np.nan] * 5 + [4.0]
    d["total_debt"] = [np.nan] * 6
    m = col.calculate_metrics(d)
    assert "capex_cagr" not in m  # only one non-NaN value (>=2 required)
    assert "debt_cagr" not in m
    d["capex"] = [np.nan] * 4 + [4.0, 4.0]
    assert "capex_cagr" in col.calculate_metrics(d)


def test_calculate_metrics_cagr_nonpositive_start(col):
    # newest revenue (the "start" value in the source) <= 0 -> None
    m = col.calculate_metrics(frame(4, rev=[100.0, 100.0, 100.0, 0.0]))
    assert m["revenue_cagr"] is None
    m = col.calculate_metrics(frame(4, rev=[100.0, 100.0, 100.0, -5.0]))
    assert m["revenue_cagr"] is None
    m = col.calculate_metrics(frame(4, rev=[100.0, 100.0, 100.0, 1.0]))
    assert m["revenue_cagr"] is not None


def test_calculate_metrics_swallows_missing_columns(col):
    d = frame(5).drop(columns=["ocf"])
    m = col.calculate_metrics(d)
    assert "revenue_cagr" in m and "avg_ocf" not in m


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="quarterly_data_collector.py:~207: cagr(revenue.iloc[-1], revenue.iloc[0]) passes "
    "the NEWEST value as start and the OLDEST as end (df is sorted ascending by date), "
    "so a growing company reports negative CAGR",
)
def test_revenue_cagr_positive_for_growing_company(col):
    rev = [100.0 * (1.2**i) for i in range(20)]
    m = col.calculate_metrics(frame(20, rev=rev))
    assert m["revenue_cagr"] > 0


def test_collect_data(col, monkeypatch, capsys):
    frames = {
        "A": frame(6).assign(date=pd.date_range("2020", periods=6, freq="QE")),
        "B": pd.DataFrame(),
    }
    monkeypatch.setattr(col, "fetch_quarterly_data", lambda t: frames[t])
    out = col.collect_data(["A", "B"])
    assert list(out) == ["A"] and set(out["A"]) == {"quarterly_data", "metrics"}
    txt = capsys.readouterr().out
    assert "Collection complete: 1/2 companies" in txt and "COMPANIES TO ANALYZE: 2" in txt
    assert "~40 quarterly" in txt


def test_analyze_trends_scores(col, capsys):
    def e(**m):
        return {"metrics": m}

    collected = {
        "ALL": e(revenue_cagr=10.01, capex_cagr=15.01, avg_fcf=1, latest_de=0.99),  # 100
        "EDGE": e(revenue_cagr=10, capex_cagr=15, avg_fcf=0, latest_de=1),  # 0 (strict >)
        "MID": e(revenue_cagr=11, capex_cagr=16),  # 50
        "THREE": e(revenue_cagr=11, capex_cagr=16, avg_fcf=1),  # 75
        "ZERODE": e(revenue_cagr=11, latest_de=0),  # 25 (0 < de excluded)
        "NONE": {"metrics": {}},
    }
    r = col.analyze_trends(collected)
    assert r is collected
    out = capsys.readouterr().out
    assert "High expansion (≥75): 2 companies" in out
    assert "Medium expansion (50-75): 1 companies" in out
    assert "Low expansion (<50): 2 companies" in out
    lines = [ln for ln in out.splitlines() if "Score:" in ln]
    assert lines[0].strip().startswith("1. ALL") and "100/100" in lines[0]
    assert len(lines) == 5


def test_analyze_trends_top10_and_empty(col, capsys):
    collected = {f"T{i:02d}": {"metrics": {"revenue_cagr": 20}} for i in range(15)}
    col.analyze_trends(collected)
    assert len([ln for ln in capsys.readouterr().out.splitlines() if "Score:" in ln]) == 10
    col.analyze_trends({})
    assert "High expansion" not in capsys.readouterr().out


def test_main_block(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(
        qc.yf,
        "Ticker",
        lambda t: types.SimpleNamespace(
            quarterly_cashflow=None, quarterly_income_stmt=None, quarterly_balance_sheet=None
        ),
    )
    runpy.run_path("quarterly_data_collector.py", run_name="__main__")
    out = capsys.readouterr().out
    assert "PHASE 1 COMPLETE" in out and "Companies analyzed: 0" in out
