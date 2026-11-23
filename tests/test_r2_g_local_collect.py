import sys
import types

import pandas as pd
import pytest

from tests.r2g_support import fake_module, loader  # noqa: F401

D1, D2 = pd.Timestamp("2023-03-31"), pd.Timestamp("2024-03-31")


def stmt(rows):
    return pd.DataFrame(rows, index=[D1, D2]).T


def full():
    return {
        "inc": stmt({"Net Income": [1, 2], "Total Revenue": [10, float("nan")]}),
        "bal": stmt({"Total Assets": [100, 200], "Ordinary Shares Number": [5, 6]}),
        "cash": stmt({"Cash Flow From Continuing Operating Activities": [7, 8]}),
    }


@pytest.fixture
def lc(loader, tmp_path, monkeypatch):  # noqa: F811
    data = {}
    calls = []

    def Ticker(tk, session=None):
        calls.append(tk)
        d = data.get(tk)
        if isinstance(d, Exception):
            raise d
        d = d or {}
        return types.SimpleNamespace(
            income_stmt=d.get("inc"), balance_sheet=d.get("bal"), cashflow=d.get("cash")
        )

    sessions = []
    creq = fake_module(
        "curl_cffi.requests", Session=lambda impersonate: sessions.append(impersonate) or object()
    )
    m = loader(
        "local_collect",
        {
            "yfinance": fake_module("yfinance", Ticker=Ticker),
            "curl_cffi": fake_module("curl_cffi", requests=creq),
            "curl_cffi.requests": creq,
        },
    )
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(m, "TICKERS", str(tmp_path / "t.csv"))
    m.data, m.calls, m.sessions, m.tmp = data, calls, sessions, tmp_path
    return m


def test_sess_cached(lc):
    s = lc._sess()
    assert lc._sess() is s and lc.sessions == ["chrome"]


def test_throttle(lc, monkeypatch):
    sl = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sl.append(s))
    monkeypatch.setattr(lc.time, "time", lambda: 50.0)
    lc._last[0] = 49.95
    lc._throttle()
    assert sl == [pytest.approx(lc.MIN_INTERVAL - 0.05)]
    sl.clear()
    lc._last[0] = 40.0
    lc._throttle()
    assert sl == [] and lc._last[0] == 50.0


def test_pick(lc):
    df = stmt({"A": [1, float("nan")]})
    assert lc._pick(None, ["A"]) == {} and lc._pick(pd.DataFrame(), ["A"]) == {}
    assert lc._pick(df, ["Z"]) == {}
    assert lc._pick(df, ["Z", "A"]) == {D1: 1.0}


def test_fetch_rows_and_symbol(lc):
    lc.data["REL.NS"] = full()
    rows = lc.fetch("REL.NS", True)
    assert [r["ticker"] for r in rows] == ["REL", "REL"]
    assert rows[0]["fy_end"] == "2023-03-31" and rows[0]["filed"] == "2023-06-29"
    assert rows[0]["cfo"] == 7.0 and rows[0]["shares"] == 5.0
    assert "revenue" not in rows[1]
    assert lc.fetch("REL.NS", False)[0]["ticker"] == "REL.NS"


def test_fetch_empty_retries_then_returns_empty(lc):
    lc.data["E"] = {"inc": pd.DataFrame(), "bal": None}
    assert lc.fetch("E", False, retries=2) == []
    assert lc.calls.count("E") == 2
    lc.calls.clear()
    lc.data["X"] = RuntimeError("boom")
    assert lc.fetch("X", False) == [] and lc.calls.count("X") == 3
    # income empty but balance present is enough
    lc.data["B"] = {"inc": None, "bal": stmt({"Total Assets": [1, 2]})}
    assert len(lc.fetch("B", False)) == 2


def run_collect(lc, market, tickers, **kw):
    lc.collect_market(market, tickers, **kw)


def test_collect_market_write_resume(lc, capsys):
    lc.data["A.NS"] = full()
    lc.data["B.NS"] = {}
    run_collect(lc, "IN", ["A.NS", "B.NS"], workers=2)
    out = capsys.readouterr().out
    assert "IN: 2 names, 0 done, 2 to fetch" in out and "IN: done — 1 names with data" in out
    path = lc.tmp / "out" / "IN.parquet"
    assert set(pd.read_parquet(path)["ticker"]) == {"A"}
    lc.data["B.NS"] = full()
    run_collect(lc, "IN", ["A.NS", "B.NS"])
    assert "2 names, 1 done, 1 to fetch" in capsys.readouterr().out
    assert set(pd.read_parquet(path)["ticker"]) == {"A", "B"}
    # non-bare market keeps full ticker key
    lc.data["X.L"] = full()
    run_collect(lc, "UK", ["X.L"])
    assert set(pd.read_parquet(lc.tmp / "out" / "UK.parquet")["ticker"]) == {"X.L"}
    run_collect(lc, "UK", ["x.l"])
    assert "1 done, 0 to fetch" in capsys.readouterr().out.splitlines()[-2]


def test_collect_market_old_schema_and_nothing(lc, capsys):
    (lc.tmp / "out").mkdir()
    pd.DataFrame({"ticker": ["A"], "net_income": [1.0]}).to_parquet(lc.tmp / "out" / "IN.parquet")
    lc.data["A.NS"] = {}
    run_collect(lc, "IN", ["A.NS"])
    assert "1 names, 0 done, 1 to fetch" in capsys.readouterr().out
    run_collect(lc, "JP", [])  # no todo, no frame, no write
    assert not (lc.tmp / "out" / "JP.parquet").exists()


def test_collect_market_future_error_and_checkpoint(lc, monkeypatch, capsys):
    monkeypatch.setattr(lc, "fetch", lambda tk, b: (_ for _ in ()).throw(RuntimeError("x")))
    run_collect(lc, "JP", ["a"])
    assert "done — 0 names" in capsys.readouterr().out
    writes = []
    orig = pd.DataFrame.to_parquet
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, *a, **k: (writes.append(len(self)), orig(self, *a, **k))[1],
    )
    monkeypatch.setattr(lc, "fetch", lambda tk, b: [{"ticker": tk, "total_assets": 1.0}])
    run_collect(lc, "JP", [f"t{i}" for i in range(201)], workers=1)
    assert len(writes) == 2 and writes[-1] == 201


def test_main(lc, monkeypatch, capsys):
    pd.DataFrame(
        {"market": ["US", "JP", "IN", "IN"], "yf_ticker": ["A", "7203.T", "X.NS", "Y.NS"]}
    ).to_csv(lc.TICKERS, index=False)
    got = []
    real = lc.collect_market
    monkeypatch.setattr(lc, "collect_market", lambda m, t, w: got.append((m, t, w)))
    monkeypatch.setattr(sys, "argv", ["x"])
    lc.main()
    assert got == [("IN", ["X.NS", "Y.NS"], 8), ("JP", ["7203.T"], 8)]  # US skipped
    got.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--markets", "us, jp", "--workers", "2"])
    lc.main()
    assert got == [("US", ["A"], 2), ("JP", ["7203.T"], 2)]  # explicit list is not filtered
    assert "ALL DONE — coverage:" in capsys.readouterr().out
    lc.data["7203.T"] = full()
    real("JP", ["7203.T"], 1)
    capsys.readouterr()
    lc.main()
    assert "JP: 1 tickers, 2 rows" in capsys.readouterr().out
