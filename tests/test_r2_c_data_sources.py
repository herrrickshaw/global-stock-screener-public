"""Offline tests for data_sources (fake requests.Session / stock_utils)."""

import datetime as dt
import sys
import types

import pandas as pd
import pytest

import data_sources as ds


def csv_text(n=70, header="Date,Open,High,Low,Close,Volume"):
    d = pd.date_range("2024-01-01", periods=n, freq="D")
    rows = [f"{x.date()},10,11,9,{10 + i},100" for i, x in enumerate(d)]
    return header + "\n" + "\n".join(rows) + "\n"


class Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status


class Sess:
    def __init__(self, responder):
        self.responder = responder
        self.headers = {}
        self.calls = []

    def get(self, url, **kw):
        self.calls.append((url, kw))
        return self.responder(url, kw)


def install(monkeypatch, responder):
    sess = Sess(responder)
    monkeypatch.setattr(ds.requests, "Session", lambda: sess)
    return sess


@pytest.mark.parametrize(
    "t,exp",
    [
        ("7203.T", "7203.jp"),
        ("RIO.L", "rio.uk"),
        ("ADS.DE", "ads.de"),
        ("SAP.F", "sap.de"),
        ("AIR.PA", "air.fr"),
        ("ASML.AS", "asml.nl"),
        ("KBC.BR", "kbc.be"),
        ("ENI.MI", "eni.it"),
        ("SAN.MC", "san.es"),
        ("0005.HK", "0005.hk"),
        ("NESN.SW", "nesn.ch"),
        ("VOLV.ST", "volv.se"),
        ("AAPL", "aapl.us"),
        ("RELIANCE.NS", None),
        ("D05.SI", None),
    ],
)
def test_stooq_symbol(t, exp):
    assert ds._stooq_symbol(t) == exp


@pytest.mark.parametrize(
    "t,exp",
    [
        ("7203.T", "7203.TSE"),
        ("ADS.DE", "ADS.XETRA"),
        ("TCS.NS", "TCS.NSE"),
        ("D05.SI", "D05.SG"),
        ("005930.KS", "005930.KO"),
        ("AAPL", "AAPL.US"),
        ("X.ZZ", None),
    ],
)
def test_eodhd_symbol(t, exp):
    assert ds._eodhd_symbol(t) == exp


def test_period_to_from():
    today = dt.date.today()
    assert ds._period_to_from("5d") == (today - dt.timedelta(days=10)).isoformat()
    assert ds._period_to_from("6mo") == (today - dt.timedelta(days=6 * 31 + 5)).isoformat()
    assert ds._period_to_from("1y") == (today - dt.timedelta(days=371)).isoformat()
    assert ds._period_to_from("max") is None
    assert ds._period_to_from("") is None
    assert ds._period_to_from(None) is None


def test_stooq_fetch(monkeypatch, capsys):
    def responder(url, kw):
        if "aapl" in url:
            return Resp(csv_text(70))
        if "msft" in url:
            return Resp("<html>", 200)
        if "tsla" in url:
            return Resp(csv_text(70), 500)
        if "nvda" in url:
            return Resp("Date,Open,High,Low,Volume\n2024-01-01,1,1,1,1\n")
        if "amd" in url:
            return Resp("Date,Open,High,Low,Close,Volume\n")
        if "intc" in url:
            return Resp(csv_text(59))
        raise RuntimeError("net")

    sess = install(monkeypatch, responder)
    out = ds.stooq_fetch(
        ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "INTC", "ERR", "X.NS"], min_bars=60
    )
    assert set(out) == {"AAPL"}
    assert "User-Agent" in sess.headers
    assert len(sess.calls) == 7  # .NS skipped before any request
    assert sess.calls[0][0] == "https://stooq.com/q/d/l/?s=aapl.us&i=d"
    assert sess.calls[0][1]["timeout"] == 20
    assert list(out["AAPL"].columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert "stooq: 1/8 fetched" in capsys.readouterr().out
    ds.stooq_fetch(["AAPL"], verbose=False)
    assert capsys.readouterr().out == ""


def test_stooq_min_bars_boundary(monkeypatch):
    install(monkeypatch, lambda u, k: Resp(csv_text(60)))
    assert "A" in ds.stooq_fetch(["A"], min_bars=60, verbose=False)
    assert "A" not in ds.stooq_fetch(["A"], min_bars=61, verbose=False)


def test_stooq_without_cleaner(monkeypatch):
    monkeypatch.setattr(ds, "clean_ohlcv", None)
    install(monkeypatch, lambda u, k: Resp(csv_text(5)))
    assert "A" in ds.stooq_fetch(["A"], min_bars=5, verbose=False)
    assert "A" not in ds.stooq_fetch(["A"], min_bars=6, verbose=False)


def test_stooq_cleaner_returns_none(monkeypatch):
    monkeypatch.setattr(ds, "clean_ohlcv", lambda df, ticker, min_bars: None)
    install(monkeypatch, lambda u, k: Resp(csv_text(70)))
    assert ds.stooq_fetch(["A"], verbose=False) == {}


def test_eodhd_dormant(monkeypatch):
    monkeypatch.delenv("EODHD_API_KEY", raising=False)
    sess = install(monkeypatch, lambda u, k: Resp(csv_text()))
    assert ds.eodhd_fetch(["AAPL"]) == {}
    assert sess.calls == []


def test_eodhd_fetch(monkeypatch, capsys):
    monkeypatch.setenv("EODHD_API_KEY", "dummy-test-key")

    def responder(url, kw):
        if url.endswith("/AAPL.US"):
            return Resp(csv_text(70))
        if url.endswith("/BAD.US"):
            return Resp("html", 200)
        if url.endswith("/ERR.US"):
            raise RuntimeError("x")
        if url.endswith("/NOC.US"):
            return Resp("Date,Open\n2024-01-01,1\n")
        if url.endswith("/EMP.US"):
            return Resp("Date,Open,High,Low,Close,Volume\n")
        return Resp("", 404)

    sess = install(monkeypatch, responder)
    out = ds.eodhd_fetch(["AAPL", "BAD", "ERR", "NOC", "EMP", "X.ZZ", "NF"], period="1y")
    assert set(out) == {"AAPL"}
    url, kw = sess.calls[0]
    assert url == "https://eodhd.com/api/eod/AAPL.US"
    assert kw["params"]["fmt"] == "csv" and kw["params"]["period"] == "d"
    assert kw["params"]["from"] == ds._period_to_from("1y")
    assert len(sess.calls) == 6
    assert "eodhd: 1/7 fetched" in capsys.readouterr().out
    sess.calls.clear()
    ds.eodhd_fetch(["AAPL"], period="max", verbose=False)
    assert "from" not in sess.calls[0][1]["params"]


def test_eodhd_no_cleaner_and_short(monkeypatch):
    monkeypatch.setenv("EODHD_API_KEY", "dummy-test-key")
    monkeypatch.setattr(ds, "clean_ohlcv", None)
    install(monkeypatch, lambda u, k: Resp(csv_text(10)))
    assert "A" in ds.eodhd_fetch(["A"], min_bars=10, verbose=False)
    assert ds.eodhd_fetch(["A"], min_bars=11, verbose=False) == {}


def test_yahoo_fetch(monkeypatch):
    seen = {}

    def bulk(tickers, **kw):
        seen.update(kw, tickers=tickers)
        return {"A": 1}

    monkeypatch.setattr(ds, "_yahoo_bulk", bulk)
    assert ds.yahoo_fetch(["A"], period="3mo", min_bars=7, verbose=False) == {"A": 1}
    assert seen == {
        "tickers": ["A"],
        "period": "3mo",
        "batch_size": 80,
        "min_bars": 7,
        "verbose": False,
    }
    monkeypatch.setattr(ds, "_yahoo_bulk", None)
    assert ds.yahoo_fetch(["A"]) == {}


def test_groww_fetch(monkeypatch):
    mod = types.ModuleType("groww_source")
    seen = {}
    mod.fetch = lambda t, **kw: seen.update(kw) or {"G": 1}
    monkeypatch.setitem(sys.modules, "groww_source", mod)
    assert ds.groww_fetch(["G"], period="2y", min_bars=9, verbose=False) == {"G": 1}
    assert seen == {"period": "2y", "min_bars": 9, "verbose": False}

    def boom(*a, **k):
        raise RuntimeError("403")

    mod.fetch = boom
    assert ds.groww_fetch(["G"]) == {}


def test_fetch_fallback_chain(monkeypatch, capsys):
    calls = []

    def mk(name, hits):
        def fn(tickers, **kw):
            calls.append((name, list(tickers), kw))
            return {t: name for t in tickers if t in hits}

        return fn

    monkeypatch.setattr(
        ds,
        "SOURCES",
        {"yahoo": mk("yahoo", {"A"}), "stooq": mk("stooq", {"B"}), "eodhd": mk("eodhd", set())},
    )
    out = ds.fetch(
        ["A", "B", "C", "A"], order=("yahoo", "bogus", "stooq", "eodhd"), period="2y", min_bars=5
    )
    assert out == {"A": "yahoo", "B": "stooq"}
    assert calls[0][1] == ["A", "B", "C"]  # de-duplicated, order kept
    assert calls[1][1] == ["B", "C"]
    assert calls[2][1] == ["C"]
    assert calls[0][2] == {"min_bars": 5, "verbose": True, "period": "2y"}
    assert "period" not in calls[1][2]  # stooq takes no period
    assert calls[2][2]["period"] == "2y"
    assert ds.get_last_stats() == {
        "requested": 3,
        "filled": 2,
        "by_source": {"yahoo": 1, "stooq": 1, "eodhd": 0},
    }
    text = capsys.readouterr().out
    assert "source 'yahoo': attempting 3 tickers" in text
    assert "multi-source total: 2/4 (1 unresolved)" in text
    ds.LAST_STATS["junk"] = 1
    s = ds.get_last_stats()
    s["x"] = 1
    assert "x" not in ds.LAST_STATS


def test_fetch_stops_when_nothing_pending(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ds,
        "SOURCES",
        {
            "yahoo": lambda t, **k: calls.append("y") or {x: 1 for x in t},
            "stooq": lambda t, **k: calls.append("s") or {},
        },
    )
    ds.fetch(["A"], order=("yahoo", "stooq"), verbose=False)
    assert calls == ["y"]
    assert ds.fetch([], order=("yahoo",), verbose=False) == {}
    assert ds.get_last_stats()["requested"] == 0


def test_fetch_groww_gets_period(monkeypatch):
    seen = {}
    monkeypatch.setattr(ds, "SOURCES", {"groww": lambda t, **k: seen.update(k) or {}})
    ds.fetch(["A"], order=("groww",), period="9mo", verbose=False)
    assert seen["period"] == "9mo"
