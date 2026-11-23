from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import daily_price_collector as dpc


@pytest.fixture
def col():
    return dpc.DailyPriceCollector()


def ohlcv(n, start_price=100.0, step=1.0, vol=1000):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
    close = start_price + step * np.arange(n)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": vol}, index=idx
    )


def test_init_window(col):
    assert (col.end_date - col.start_date) == timedelta(days=1825)
    assert abs((datetime.now() - col.end_date).total_seconds()) < 60


def test_select_sample_companies(col):
    allt = col.select_sample_companies(1000)
    assert len(allt) >= 60 and len(set(allt)) == len(allt)
    assert col.select_sample_companies() == allt[:60]
    assert col.select_sample_companies(3) == allt[:3]
    assert col.select_sample_companies(0) == []


def patch_history(monkeypatch, df=None, exc=None):
    calls = []

    class T:
        def history(self, start, end):
            calls.append((start, end))
            if exc:
                raise exc
            return df.copy() if df is not None else pd.DataFrame()

    monkeypatch.setattr(dpc.yf, "Ticker", lambda t: T())
    return calls


def test_fetch_price_data_columns_and_values(col, monkeypatch):
    calls = patch_history(monkeypatch, ohlcv(120))
    df = col.fetch_price_data("AAA")
    assert calls == [(col.start_date, col.end_date)]
    assert "Date" in df.columns and set(df["ticker"]) == {"AAA"}
    assert df["returns"].iloc[1] == pytest.approx(1.0)  # 100 -> 101
    assert df["log_returns"].iloc[1] == pytest.approx(np.log(101 / 100) * 100)
    assert np.isnan(df["returns"].iloc[0])
    assert df["volume_ma_20"].iloc[18] != df["volume_ma_20"].iloc[18]  # NaN before window fills
    assert df["volume_ma_20"].iloc[19] == 1000
    assert df["price_momentum_20"].iloc[20] == pytest.approx((120 / 100 - 1) * 100)
    assert np.isnan(df["price_momentum_20"].iloc[19])
    assert df["volatility_20"].iloc[:20].isna().all() and df["volatility_20"].iloc[20] > 0


def test_fetch_price_data_min_rows_boundary(col, monkeypatch):
    patch_history(monkeypatch, ohlcv(99))
    assert col.fetch_price_data("A").empty
    patch_history(monkeypatch, ohlcv(100))
    assert len(col.fetch_price_data("A")) == 100
    patch_history(monkeypatch, pd.DataFrame())
    assert col.fetch_price_data("A").empty
    patch_history(monkeypatch, exc=RuntimeError("net"))
    assert col.fetch_price_data("A").empty


def test_collect_prices(col, monkeypatch, capsys):
    frames = {"A": ohlcv(10), "B": pd.DataFrame(), "C": ohlcv(30)}
    monkeypatch.setattr(col, "fetch_price_data", lambda t: frames[t])
    out = col.collect_prices(["A", "B", "C"])
    assert list(out) == ["A", "C"]
    txt = capsys.readouterr().out
    assert "Collection complete: 2/3 companies" in txt
    assert "Total records collected: 40" in txt and "Average records per company: 20" in txt
    assert "~5,475 daily records" in txt
    assert col.start_date.strftime("%Y-%m-%d") in txt


def test_collect_prices_all_fail_average_zero(col, monkeypatch, capsys):
    monkeypatch.setattr(col, "fetch_price_data", lambda t: pd.DataFrame())
    assert col.collect_prices(["A"]) == {}
    assert "Average records per company: 0" in capsys.readouterr().out


def with_metrics(n, **kw):
    df = ohlcv(n, **kw)
    df["returns"] = df["Close"].pct_change() * 100
    df["price_momentum_20"] = df["Close"].pct_change(20) * 100
    df["volatility_20"] = df["returns"].rolling(20).std()
    return df.reset_index()


def test_analyze_price_metrics_stats(col, capsys):
    a = with_metrics(60)
    a["volatility_20"] = 2.0
    a["returns"] = 1.0
    a["price_momentum_20"] = 5.0
    b = with_metrics(60)
    b["volatility_20"] = 4.0
    b["returns"] = -1.0
    b["price_momentum_20"] = -3.0
    c = with_metrics(60)
    c["volatility_20"] = 6.0
    c["returns"] = 0.0
    c["price_momentum_20"] = 0.0
    collected = {"A": a, "B": b, "C": c, "EMPTY": pd.DataFrame()}
    r = col.analyze_price_metrics(collected)
    assert r is collected
    out = capsys.readouterr().out
    assert (
        "Mean: 4.00%" in out
        and "Median: 4.00%" in out
        and "High: 6.00%" in out
        and "Low: 2.00%" in out
    )
    assert "Mean: 0.000%" in out and "Positive days ratio: 33.3%" in out  # strictly > 0
    assert "Positive momentum: 1 companies" in out and "Negative momentum: 1 companies" in out


def test_analyze_price_metrics_all_nan_and_empty(col, capsys):
    d = with_metrics(5)
    d["volatility_20"] = np.nan
    d["returns"] = np.nan
    d["price_momentum_20"] = np.nan
    col.analyze_price_metrics({"A": d})
    out = capsys.readouterr().out
    assert "Volatility (20-day rolling)" not in out and "Daily Returns:" not in out
    assert "20-Day Momentum" not in out
    col.analyze_price_metrics({})
    assert "PRICE CHARACTERISTICS" in capsys.readouterr().out


def test_prepare_for_correlation_values(col, capsys):
    df = with_metrics(300, start_price=100.0, step=1.0)  # 100 .. 399
    out = col.prepare_for_correlation({"A": df})
    (r,) = out.to_dict("records")
    assert r["ticker"] == "A" and r["latest_price"] == 399.0
    assert r["total_return"] == pytest.approx(299.0)
    assert r["price_5yr_cagr"] == pytest.approx((3.99**0.2 - 1) * 100)
    assert r["max_drawdown"] == 0.0  # monotonic rise
    assert r["momentum_1y"] == pytest.approx(df["price_momentum_20"].tail(252).mean())
    assert r["momentum_2y"] == 0 and r["volatility_3y"] == 0  # len <= 504 / 756
    assert r["volatility_1y"] == pytest.approx(df["volatility_20"].tail(252).mean())
    assert r["sharpe_ratio"] == pytest.approx(
        df["returns"].mean() / df["returns"].std() * np.sqrt(252)
    )
    txt = capsys.readouterr().out
    assert "Top performers by 5-year price CAGR" in txt and "A      - CAGR:" in txt


def test_prepare_for_correlation_window_boundaries(col):
    e252 = col.prepare_for_correlation({"A": with_metrics(252)}).iloc[0]
    assert e252["momentum_1y"] == 0 and e252["volatility_1y"] == 0  # needs > 252
    n253 = col.prepare_for_correlation({"A": with_metrics(253)}).iloc[0]
    assert n253["momentum_1y"] != 0 and n253["volatility_1y"] != 0
    e504 = col.prepare_for_correlation({"A": with_metrics(504)}).iloc[0]
    assert e504["momentum_2y"] == 0
    n505 = col.prepare_for_correlation({"A": with_metrics(505)}).iloc[0]
    assert n505["momentum_2y"] != 0
    assert col.prepare_for_correlation({"A": with_metrics(756)}).iloc[0]["volatility_3y"] == 0
    assert col.prepare_for_correlation({"A": with_metrics(757)}).iloc[0]["volatility_3y"] != 0


def test_prepare_for_correlation_drawdown_and_degenerate(col):
    d = with_metrics(5)
    d["Close"] = [100.0, 200.0, 100.0, 150.0, 50.0]
    d["returns"] = d["Close"].pct_change() * 100
    r = col.prepare_for_correlation({"A": d}).iloc[0]
    assert r["max_drawdown"] == pytest.approx(-75.0)  # 200 -> 50
    assert r["total_return"] == pytest.approx(-50.0)
    # flat returns -> std 0 -> sharpe 0
    flat = with_metrics(5)
    flat["Close"] = 10.0
    flat["returns"] = 0.0
    assert col.prepare_for_correlation({"A": flat}).iloc[0]["sharpe_ratio"] == 0
    # non-positive first price -> cagr 0 and total_return 0
    z = with_metrics(5)
    z["Close"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    r = col.prepare_for_correlation({"A": z}).iloc[0]
    assert r["price_5yr_cagr"] == 0 and r["total_return"] == 0


def test_prepare_for_correlation_empty_and_top5(col, capsys):
    out = col.prepare_for_correlation({"E": pd.DataFrame()})
    assert out.empty and "Top performers" not in capsys.readouterr().out
    data = {f"T{i}": with_metrics(30, start_price=100.0, step=float(i)) for i in range(1, 8)}
    out = col.prepare_for_correlation(data)
    assert len(out) == 7
    txt = capsys.readouterr().out
    lines = [ln for ln in txt.splitlines() if "CAGR:" in ln and "Sharpe" in ln]
    assert len(lines) == 5 and lines[0].strip().startswith("T7")


def test_main_block(monkeypatch, capsys):
    import runpy

    class T:
        def history(self, start, end):
            return ohlcv(120)

    monkeypatch.setattr(dpc.yf, "Ticker", lambda t: T())
    runpy.run_path("daily_price_collector.py", run_name="__main__")
    out = capsys.readouterr().out
    assert "PHASE 2 COMPLETE" in out and "Total price records: " in out
