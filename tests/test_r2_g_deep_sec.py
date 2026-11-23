import sys

import pandas as pd
import pytest

from tests.r2g_support import fake_module, loader  # noqa: F401


# ------------------------------------------------------------ deep_prices
class Hist:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []

    def Ticker(self, tk, session=None):
        outer = self

        class T:
            def history(self, period, auto_adjust):
                outer.calls.append((tk, period, auto_adjust))
                b = outer.behavior(tk)
                if isinstance(b, Exception):
                    raise b
                return b

        return T()


def hist_df(n=3, tz=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC" if tz else None, name="Date")
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10, "Dividends": 0.0},
        index=idx,
    )


@pytest.fixture
def dp(loader, tmp_path, monkeypatch):  # noqa: F811
    yf = fake_module("yfinance")
    yf.h = Hist(lambda tk: hist_df())
    yf.Ticker = lambda tk, session=None: yf.h.Ticker(tk, session)
    sessions = []

    class Sess:
        def __init__(self, impersonate):
            sessions.append(impersonate)

    creq = fake_module("curl_cffi.requests", Session=Sess)
    cc = fake_module("curl_cffi", requests=creq)
    m = loader("deep_prices", {"yfinance": yf, "curl_cffi": cc, "curl_cffi.requests": creq})
    m.yf_fake = yf
    m.sessions = sessions
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setattr(m, "LTM_OUT", str(tmp_path / "ltm"))
    return m


def test_sess_is_per_thread_cached(dp):
    s1 = dp._sess()
    assert dp._sess() is s1
    assert dp.sessions == ["chrome"]
    import threading

    box = []
    t = threading.Thread(target=lambda: box.append(dp._sess()))
    t.start()
    t.join()
    assert box[0] is not s1 and len(dp.sessions) == 2


def test_throttle_sleeps_only_when_too_soon(dp, monkeypatch):
    sl = []
    monkeypatch.setattr(dp.time, "sleep", lambda s: sl.append(s))
    t = {"now": 100.0}
    monkeypatch.setattr(dp.time, "time", lambda: t["now"])
    dp._last[0] = 99.95  # 0.05 elapsed
    dp._throttle()
    assert sl and sl[0] == pytest.approx(dp.MIN_INTERVAL - 0.05)
    sl.clear()
    dp._last[0] = 99.0  # 1s elapsed
    dp._throttle()
    assert sl == []
    assert dp._last[0] == 100.0


def test_fetch_ok_normalises(dp):
    h = dp.fetch("aapl")
    assert list(h.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
    assert h["Date"].dt.tz is None and set(h["Symbol"]) == {"AAPL"}
    assert dp.yf_fake.h.calls == [("aapl", "10y", True)]


def test_fetch_retries_then_none_and_recovers(dp):
    n = {"c": 0}

    def b(tk):
        n["c"] += 1
        return hist_df() if n["c"] == 3 else pd.DataFrame()

    dp.yf_fake.h.behavior = b
    assert dp.fetch("X") is not None and n["c"] == 3
    n["c"] = 0
    dp.yf_fake.h.behavior = lambda tk: (n.__setitem__("c", n["c"] + 1), None)[1]
    assert dp.fetch("X", retries=2) is None and n["c"] == 2
    dp.yf_fake.h.behavior = lambda tk: RuntimeError("x")
    assert dp.fetch("X", retries=1) is None


def test_collect_market_fresh_and_resume(dp, tmp_path, capsys):
    dp.collect_market("US", ["a", "b"], workers=2)
    out = pd.read_parquet(tmp_path / "ltm" / "US.parquet")
    assert sorted(out["Symbol"].unique()) == ["A", "B"]
    assert "US: 2 names, 0 deep, 2 to fetch" in capsys.readouterr().out
    # shallow panel (3 bars each, median<=327) is re-collected, not reused
    dp.yf_fake.h.calls.clear()
    dp.collect_market("US", ["a", "b"])
    assert len(dp.yf_fake.h.calls) == 2


def test_collect_market_reuses_deep_and_boundary(dp, tmp_path):
    (tmp_path / "ltm").mkdir()
    deep = pd.DataFrame({"Symbol": ["A"] * 328 + ["B"] * 328, "Date": pd.Timestamp("2024-01-01")})
    deep.to_parquet(tmp_path / "ltm" / "US.parquet")
    dp.collect_market("US", ["a", "c"])
    assert [c[0] for c in dp.yf_fake.h.calls] == ["c"]
    out = pd.read_parquet(tmp_path / "ltm" / "US.parquet")
    assert len(out) == 328 * 2 + 3
    # exactly 327 median -> not deep -> refetch all
    dp.yf_fake.h.calls.clear()
    pd.DataFrame({"Symbol": ["A"] * 327, "Date": pd.Timestamp("2024-01-01")}).to_parquet(
        tmp_path / "ltm" / "US.parquet"
    )
    dp.collect_market("US", ["a"])
    assert len(dp.yf_fake.h.calls) == 1
    # no Symbol column -> median 0 -> refetch
    dp.yf_fake.h.calls.clear()
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "ltm" / "US.parquet")
    dp.collect_market("US", ["a"])
    assert len(dp.yf_fake.h.calls) == 1


def test_collect_market_failed_fetch_and_no_todo(dp, tmp_path, capsys):
    dp.yf_fake.h.behavior = lambda tk: RuntimeError("x")
    dp.collect_market("US", ["a"])
    assert not (tmp_path / "ltm" / "US.parquet").exists()
    assert "done — 0 names" in capsys.readouterr().out
    dp.collect_market("US", [])
    assert "0 to fetch" in capsys.readouterr().out


def test_collect_market_future_exception_swallowed(dp, monkeypatch, capsys):
    monkeypatch.setattr(dp, "fetch", lambda tk: (_ for _ in ()).throw(RuntimeError("x")))
    dp.collect_market("US", ["a"])
    assert "done — 0 names" in capsys.readouterr().out


def test_collect_market_checkpoint_every_200(dp, tmp_path, monkeypatch):
    writes = []
    orig = pd.DataFrame.to_parquet
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet", lambda self, *a, **k: (writes.append(1), orig(self, *a, **k))[1]
    )
    monkeypatch.setattr(dp, "fetch", lambda tk: hist_df(1).reset_index().assign(Symbol=tk))
    dp.collect_market("US", [f"t{i}" for i in range(201)], workers=1)
    assert len(writes) == 2  # at n==200 and n==201


def test_main(dp, tmp_path, monkeypatch, capsys):
    csv = tmp_path / "t.csv"
    pd.DataFrame({"market": ["US", "IN", "IN"], "yf_ticker": ["a", "b", "c"]}).to_csv(
        csv, index=False
    )
    monkeypatch.setattr(dp, "TICKERS", str(csv))
    got = []
    real_collect = dp.collect_market
    monkeypatch.setattr(dp, "collect_market", lambda m, t, w: got.append((m, t, w)))
    monkeypatch.setattr(sys, "argv", ["x"])
    dp.main()
    assert got == [("IN", ["b", "c"], 10), ("US", ["a"], 10)]
    got.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--markets", "us, in", "--workers", "3"])
    dp.main()
    assert got == [("US", ["a"], 3), ("IN", ["b", "c"], 3)]
    got.clear()
    monkeypatch.setattr(sys, "argv", ["x", "--shallow"])
    dp.main()
    assert [g[0] for g in got] == dp.SHALLOW
    out = capsys.readouterr().out
    assert "DONE — depth check:" in out
    real_collect("US", ["a"])  # create a parquet for the depth line
    capsys.readouterr()
    dp.main()
    assert "US.parquet"[:-8] + ": 1 syms, 3 med bars" in capsys.readouterr().out


# ------------------------------------------------------------ sec_fundamentals
@pytest.fixture
def sec(loader):  # noqa: F811
    return loader("sec_fundamentals")


def fact(vals, unit="USD", form="10-K", fp="FY"):
    return {"units": {unit: [{"end": e, "val": v, "form": form, "fp": fp} for e, v in vals]}}


def test_annual_filters_dedups_orders(sec):
    node = {
        "units": {
            "USD": [
                {"end": "2021-12-31", "val": 1, "form": "10-K", "fp": "FY"},
                {"end": "2022-12-31", "val": 2, "form": "10-K", "fp": "FY"},
                {"end": "2022-12-31", "val": 3, "form": "10-K", "fp": "FY"},  # later wins
                {"end": "2023-12-31", "val": 9, "form": "10-Q", "fp": "FY"},
                {"end": "2023-12-31", "val": 9, "form": "10-K", "fp": "Q1"},
                {"end": "2023-12-31", "val": None, "form": "10-K", "fp": "FY"},
                {"end": "2020-12-31", "val": 0, "form": "10-K", "fp": "FY"},
            ]
        }
    }
    facts = {"us-gaap": {"C": node}}
    assert sec._annual(facts, "C") == [3, 1, 0]
    assert sec._annual(facts, "C", n=2) == [3, 1]
    assert sec._annual(facts, "missing") == []
    assert sec._annual({}, "C") == []


def test_first_picks_first_nonempty(sec):
    facts = {"us-gaap": {"B": fact([("2022", 5)])}}
    assert sec._first(facts, "A", "B", "C") == [5]
    assert sec._first(facts, "A") == []


def test_small_helpers(sec):
    assert sec._ratio(6, 3) == 2
    assert sec._ratio(0, 3) == 0
    assert sec._ratio(None, 3) is None
    assert sec._ratio(1, 0) is None and sec._ratio(1, None) is None
    assert sec._margin(50, 100) == 50.0
    assert sec._margin(0, 100) == 0.0
    assert sec._margin(100, 100) == 100.0
    assert sec._margin(101, 100) is None and sec._margin(-1, 100) is None
    assert sec._margin(None, 1) is None and sec._margin(1, 0) is None
    assert sec._margin(1, 3) == 33.33
    assert sec._pct(110, 100) == pytest.approx(10)
    assert sec._pct(-90, -100) == pytest.approx(10)
    assert sec._pct(None, 1) is None and sec._pct(1, 0) is None and sec._pct(1, None) is None
    assert sec._roe_history([10, 20, 30], [100, 0, 300]) == [10.0, 10.0]
    assert sec._roe_history([10], [0]) is None
    assert sec._roe_history([], [1]) is None


class Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http")

    def json(self):
        return self._d


def test_ticker_cik(sec, monkeypatch):
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, h=headers, t=timeout)
        return Resp(
            {"0": {"ticker": "aapl", "cik_str": 320193}, "1": {"ticker": "KO", "cik_str": 21344}}
        )

    monkeypatch.setattr(sec.requests, "get", get)
    assert sec._ticker_cik() == {"AAPL": "0000320193", "KO": "0000021344"}
    assert (
        seen["t"] == 30
        and "User-Agent" in seen["h"]
        and seen["url"].endswith("company_tickers.json")
    )
    sec._ticker_cik()  # cached
    monkeypatch.setattr(sec.requests, "get", lambda *a, **k: 1 / 0)
    assert "AAPL" in sec._ticker_cik()


def test_ticker_cik_http_error(sec, monkeypatch):
    monkeypatch.setattr(sec.requests, "get", lambda *a, **k: Resp({}, 500))
    with pytest.raises(RuntimeError):
        sec._ticker_cik()


def test_companyfacts_paths(sec, monkeypatch):
    monkeypatch.setattr(sec, "_ticker_cik", lambda: {"AAPL": "0000320193"})
    assert sec.companyfacts("zzz") is None
    urls = []

    def get(url, headers, timeout):
        urls.append(url)
        return Resp({"facts": {"us-gaap": {}}})

    monkeypatch.setattr(sec.requests, "get", get)
    assert sec.companyfacts("aapl") == {"us-gaap": {}}
    assert urls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]
    sec.companyfacts.cache_clear()
    monkeypatch.setattr(sec.requests, "get", lambda *a, **k: Resp({}, 404))
    assert sec.companyfacts("AAPL") is None
    sec.companyfacts.cache_clear()
    monkeypatch.setattr(sec.requests, "get", lambda *a, **k: 1 / 0)
    assert sec.companyfacts("AAPL") is None
    sec.companyfacts.cache_clear()
    monkeypatch.setattr(sec.requests, "get", lambda *a, **k: Resp({}))  # no "facts" key
    assert sec.companyfacts("AAPL") == {}


def facts_full():
    y = lambda *v: [(f"20{22 - i}-12-31", x) for i, x in enumerate(v)]  # noqa: E731
    g = {
        "NetIncomeLoss": fact(y(20, 10, 5, 1)),
        "Revenues": fact(y(200, 100, 50, 40)),
        "Assets": fact(y(400, 200, 100, 100)),
        "Liabilities": fact(y(100, 80, 50, 50)),
        "StockholdersEquity": fact(y(200, 100, 50, 50)),
        "AssetsCurrent": fact(y(60, 40)),
        "LiabilitiesCurrent": fact(y(30, 40)),
        "NetCashProvidedByUsedInOperatingActivities": fact(y(25)),
        "OperatingIncomeLoss": fact(y(30)),
        "GrossProfit": fact(y(80, 60)),
        "CommonStockSharesOutstanding": fact(y(10, 11), unit="shares"),
        "EarningsPerShareBasic": fact(y(2.0, 1.0)),
        "LongTermDebt": fact(y(5, 6)),
        "PaymentsToAcquirePropertyPlantAndEquipment": fact(y(3)),
        "PaymentsOfDividends": fact(y(1, 1)),
        "InventoryNet": fact(y(7)),
        "AccountsReceivableNetCurrent": fact(y(8)),
        "AccountsPayableCurrent": fact(y(9)),
        "CostOfRevenue": fact(y(120)),
    }
    return {"us-gaap": g}


def test_fundamentals_empty(sec, monkeypatch):
    monkeypatch.setattr(sec, "companyfacts", lambda t: None)
    assert sec.fundamentals("X") == {}
    monkeypatch.setattr(sec, "companyfacts", lambda t: {})
    assert sec.fundamentals("X") == {}


def test_fundamentals_full(sec, monkeypatch):
    monkeypatch.setattr(sec, "companyfacts", lambda t: facts_full())
    f = sec.fundamentals("X")
    assert f["source"] == "SEC-EDGAR"
    assert f["net_income"] == 20 and f["net_income_prev"] == 10
    assert f["revenue"] == 200 and f["revenue_5y_ago"] == 40
    assert f["roa"] == pytest.approx(5.0) and f["roa_prev"] == pytest.approx(5.0)
    assert f["roe"] == pytest.approx(10.0)
    assert f["current_ratio"] == 2 and f["current_ratio_prev"] == 1
    assert f["debt_to_assets"] == 0.25 and f["debt_to_assets_prev"] == 0.4
    assert f["debt_to_equity"] == 0.5
    assert f["gross_margin"] == 40.0 and f["gross_margin_prev"] == 60.0
    assert f["asset_turnover"] == 0.5 and f["asset_turnover_prev"] == 0.5
    assert f["shares"] == 10 and f["shares_prev"] == 11
    assert f["eps_ttm"] == 2.0 and f["eps_growth"] == pytest.approx(100.0)
    assert (
        f["debt_history"] == [5, 6]
        and f["capex_history"] == [3]
        and f["dividend_history"] == [1, 1]
    )
    assert f["roe_history"] == [10.0, 10.0, 10.0, 2.0]
    assert (f["cfo"], f["ebit"]) == (25, 30)
    assert (f["inventory"], f["receivables"], f["payables"], f["cogs"]) == (7, 8, 9, 120)


def test_fundamentals_sparse_drops_none(sec, monkeypatch):
    sparse = {
        "us-gaap": {"NetIncomeLoss": fact([("2022-12-31", 5)]), "Assets": fact([("2022-12-31", 0)])}
    }
    monkeypatch.setattr(sec, "companyfacts", lambda t: sparse)
    f = sec.fundamentals("X")
    assert f == {"source": "SEC-EDGAR", "net_income": 5}


def test_fundamentals_zero_equity_and_short_rev(sec, monkeypatch):
    d = {
        "us-gaap": {
            "NetIncomeLoss": fact([("2022-12-31", 5)]),
            "StockholdersEquity": fact([("2022-12-31", 0)]),
            "Revenues": fact([("2022-12-31", 10), ("2021-12-31", 7)]),
        }
    }
    monkeypatch.setattr(sec, "companyfacts", lambda t: d)
    f = sec.fundamentals("X")
    assert "roe" not in f  # equity 0 -> falsy -> None
    assert f["revenue_5y_ago"] == 7  # fewer than 4 rows -> last element
    d["us-gaap"]["StockholdersEquity"] = fact([("2022-12-31", 50)])
    sec.companyfacts = lambda t: d
    assert sec.fundamentals("X")["roe"] == pytest.approx(10.0)


def test_sec_main(sec, monkeypatch, capsys):
    import runpy

    monkeypatch.setattr(
        sec.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    monkeypatch.setattr(sys, "argv", ["sec_fundamentals.py", "ZZZ"])
    with pytest.raises(RuntimeError):
        runpy.run_path("sec_fundamentals.py", run_name="__main__")


def test_sec_main_prints(monkeypatch, capsys):
    import runpy

    import requests

    class R:
        status_code = 200

        def __init__(self, url):
            self.url = url

        def raise_for_status(self):
            pass

        def json(self):
            if "company_tickers" in self.url:
                return {"0": {"ticker": "AAA", "cik_str": 1}}
            return {"facts": facts_full()}

    monkeypatch.setattr(requests, "get", lambda url, **k: R(url))
    monkeypatch.setattr(sys, "argv", ["sec_fundamentals.py", "AAA", "ZZZ"])
    runpy.run_path("sec_fundamentals.py", run_name="__main__")
    out = capsys.readouterr().out
    assert "AAA:  \n" in out and "ZZZ:  no data" in out
    assert "net_income" in out and "gross_margin       40.0" in out
