"""Offline tests for universe_sources (fake requests + lazily imported modules)."""

import runpy
import sys
import types

import numpy as np
import pandas as pd
import pytest

import universe_sources as us


class Resp:
    def __init__(self, payload=None, content=b"", status=200):
        self.payload, self.content, self.status = payload, content, status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


def fake_mod(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def test_us_sec(monkeypatch):
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, timeout=timeout, headers=headers)
        return Resp(
            {
                "0": {"ticker": "aapl"},
                "1": {"ticker": "BRK.B"},
                "2": {"ticker": ""},
                "3": {},
                "4": {"ticker": "AAPL"},
            }
        )

    monkeypatch.setattr(us.requests, "get", get)
    assert us.us_sec() == ["AAPL", "BRK-B"]
    assert seen["timeout"] == 30 and "sec.gov" in seen["url"]
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: Resp(status=500))
    with pytest.raises(RuntimeError):
        us.us_sec()


def test_sg_sgx(monkeypatch):
    prices = [
        {"nc": "D05", "type": "stocks"},
        {"nc": "A17U", "type": "reits"},
        {"nc": "CJLU", "type": "businesstrusts"},
        {"nc": "ETF1", "type": "etfs"},
        {"nc": "", "type": "stocks"},
        {"type": "stocks"},
        {"nc": "D05", "type": "stocks"},
    ]
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: Resp({"data": {"prices": prices}}))
    assert us.sg_sgx() == ["A17U.SI", "CJLU.SI", "D05.SI"]


def test_cn_eastmoney_pagination(monkeypatch):
    pages = {
        1: {"total": 5, "diff": [{"f12": "600000", "f13": 1}, {"f12": "000001", "f13": 0}]},
        2: {"total": 5, "diff": {"a": {"f12": "600001", "f13": 1}, "b": {"f12": None, "f13": 1}}},
        3: {"total": 5, "diff": [{"f12": "000002", "f13": 0}, {"f12": "000002", "f13": 0}]},
    }
    calls = []

    def get(url, params, headers, timeout):
        calls.append((url, params["pn"]))
        return Resp({"data": pages[params["pn"]]})

    monkeypatch.setattr(us.requests, "get", get)
    assert us.cn_eastmoney() == ["000001.SZ", "000002.SZ", "600000.SS", "600001.SS"]
    assert [c[1] for c in calls] == [1, 2, 3]  # stops once len(out) >= total
    assert calls[0][0] == "https://push2.eastmoney.com/api/qt/clist/get"


def test_cn_eastmoney_host_fallback_and_stops(monkeypatch):
    calls = []

    # without `total`, loop continues until diff empty -> make 2nd page empty
    state = {"n": 0}

    def get2(url, params, headers, timeout):
        calls.append(url)
        if "82." in url or url.startswith("https://push2.east"):
            raise OSError("down")
        state["n"] += 1
        diff = [{"f12": "1", "f13": 0}] if state["n"] == 1 else []
        return Resp({"data": {"diff": diff}})

    monkeypatch.setattr(us.requests, "get", get2)
    assert us.cn_eastmoney() == ["1.SZ"]
    assert "http://80.push2.eastmoney.com/api/qt/clist/get" in calls


def test_cn_eastmoney_empty_and_all_hosts_fail(monkeypatch):
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: Resp({"data": None}))
    assert us.cn_eastmoney() == []
    n = []
    monkeypatch.setattr(
        us.requests, "get", lambda *a, **k: n.append(1) or (_ for _ in ()).throw(OSError())
    )
    assert us.cn_eastmoney() == [] and len(n) == 4  # every host tried
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: Resp(ValueError("bad json")))
    assert us.cn_eastmoney() == []


def test_jp_jpx(monkeypatch):
    jpx = [
        {"code": "7203"},
        {"yf_ticker": "6758.T", "code": "6758"},
        {"code": ""},
        {"yf_ticker": "9984.T"},
    ]
    mod = fake_mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=lambda: jpx,
        fetch_tse_universe_kabupy=lambda: pytest.fail("fallback unused"),
    )
    assert us.jp_jpx() == ["6758.T", "7203.T", "9984.T"]

    def boom():
        raise RuntimeError("x")

    mod.fetch_tse_universe_jpx = boom
    mod.fetch_tse_universe_kabupy = lambda: [{"code": "1111"}]
    assert us.jp_jpx() == ["1111.T"]
    mod.fetch_tse_universe_jpx = lambda: []
    assert us.jp_jpx() == ["1111.T"]


def test_kr_krx(monkeypatch):
    fake_mod(
        monkeypatch,
        "full_korea_market_scan",
        build_krx_universe=lambda: [
            {"code": "005930"},
            {"code": "035720", "yf_suffix": ".KQ"},
            {"code": ""},
            {"yf_suffix": ".KS"},
        ],
    )
    assert us.kr_krx() == ["005930.KS", "035720.KQ"]


def test_in_bhavcopy(monkeypatch):
    store = fake_mod(monkeypatch, "bhavcopy_store", symbols=lambda: ["B", "A"])
    hist = fake_mod(monkeypatch, "bhavcopy_history", fetch_history=lambda verbose: {"Z": 1, "Y": 2})
    assert us.in_bhavcopy() == ["A", "B"]
    store.symbols = lambda: []
    assert us.in_bhavcopy() == ["Y", "Z"]

    def boom():
        raise RuntimeError

    store.symbols = boom
    assert us.in_bhavcopy() == ["Y", "Z"]
    seen = {}
    hist.fetch_history = lambda verbose: seen.setdefault("v", verbose) or {}
    store.symbols = lambda: None
    us.in_bhavcopy()
    assert seen["v"] is False


EURONEXT_CSV = (
    "Name;ISIN;Symbol;Market\n"
    "A;FR1;AIR;Euronext Paris, Euronext Growth\n"
    "B;NL1;ASML;Euronext Amsterdam\n"
    "C;;NOISIN;Euronext Paris\n"
    "D;IT1;ENI;Borsa Italiana\n"
    "E;XX1;ZZZ;Unknown Venue\n"
    "F;NO1; EQNR ;Oslo Børs\n"
    "G;FR2;AIR;Euronext Paris\n"
    "H;PT1;;Euronext Lisbon\n"
)


def test_euronext_equities(monkeypatch):
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, headers=headers, timeout=timeout)
        return Resp(content=("﻿" + EURONEXT_CSV).encode("utf-8"))

    monkeypatch.setattr(us.requests, "get", get)
    assert us.euronext_equities() == ["AIR.PA", "ASML.AS", "ENI.MI", "EQNR.OL"]
    assert "mics=XPAR,XAMS" in seen["url"] and seen["timeout"] == 40
    assert seen["headers"]["Referer"].startswith("https://live.euronext.com")
    us.euronext_equities(mics="XPAR")
    assert seen["url"].endswith("mics=XPAR")
    monkeypatch.setattr(us.requests, "get", lambda *a, **k: Resp(status=403))
    with pytest.raises(RuntimeError):
        us.euronext_equities()


def test_investpy_universe(monkeypatch):
    monkeypatch.setitem(sys.modules, "investpy", None)
    assert us.investpy_universe() is None
    calls = []

    def get_stocks(country):
        calls.append(country)
        if country == "bad":
            raise RuntimeError("x")
        return pd.DataFrame({"symbol": [country]})

    fake_mod(monkeypatch, "investpy", get_stocks=get_stocks)
    df = us.investpy_universe(countries=("france", "bad", "italy"))
    assert list(df["symbol"]) == ["france", "italy"] and list(df.index) == [0, 1]
    assert calls == ["france", "bad", "italy"]
    assert us.investpy_universe(countries=("bad",)) is None
    assert us.investpy_universe(countries=()) is None
    calls.clear()
    us.investpy_universe()
    assert len(calls) == 16 and calls[0] == "france"


def test_eu_curated_and_eu(monkeypatch):
    fake_mod(
        monkeypatch,
        "full_european_market_scan",
        EURO_STOXX_50_META={"ASML.AS": 1, "NEW.XX": 2},
    )
    cur = us.eu_curated()
    assert cur == sorted(set(cur)) and "NEW.XX" in cur and "HSBA.L" in cur and "SAP.DE" in cur
    assert cur.count("ASML.AS") == 1
    monkeypatch.setattr(us, "euronext_equities", lambda: ["AIR.PA", "EXTRA.PA"])
    full = us.eu()
    assert "EXTRA.PA" in full and "NEW.XX" in full and full == sorted(set(full))

    def boom():
        raise RuntimeError

    monkeypatch.setattr(us, "euronext_equities", boom)
    assert "EXTRA.PA" not in us.eu() and "HSBA.L" in us.eu()


def damo(monkeypatch, df):
    fake_mod(monkeypatch, "reference_data", damodaran_companies=lambda: df)


def test_from_damodaran(monkeypatch):
    df = pd.DataFrame(
        {
            "Country": [
                "Hong Kong",
                "Hong Kong",
                "Hong Kong",
                "Hong Kong",
                "Hong Kong",
                "Sweden",
                "Sweden",
                "Japan",
            ],
            "Exchange": ["SEHK", "SEHK", "SEHK", "OTHER", "SEHK", "OM", "OM", "TSE"],
            "Ticker": ["5", "700", "", "9", "5", "VOLV-B", "ERIC-B", "7203"],
        }
    )
    damo(monkeypatch, df)
    assert us.from_damodaran("Hong Kong", {"SEHK"}) == ["0005.HK", "0700.HK"]
    # without allowed set the unknown exchange has no suffix -> skipped anyway
    assert us.from_damodaran("Hong Kong") == ["0005.HK", "0700.HK"]
    # country suffix wins over exchange code and ignores allowed_exch=None
    assert us.from_damodaran("Sweden") == ["ERIC-B.ST", "VOLV-B.ST"]
    # allowed_exch excludes everything
    assert us.from_damodaran("Sweden", {"XXX"}) == []
    assert us.from_damodaran("Japan") == []  # no suffix mapping for TSE
    assert us.from_damodaran("Nowhere") == []


def test_from_damodaran_none_and_nonhk(monkeypatch):
    damo(monkeypatch, None)
    assert us.from_damodaran("Canada") == []
    df = pd.DataFrame(
        {"Country": ["Canada", "Canada"], "Exchange": ["TSX", " TSXV "], "Ticker": [" RY ", "ABC"]}
    )
    damo(monkeypatch, df)
    assert us.from_damodaran("Canada", {"TSX", "TSXV"}) == [
        "ABC.V",
        "RY.TO",
    ]  # exch stripped, no zfill


def test_providers_and_get_universe(monkeypatch):
    calls = []
    monkeypatch.setattr(us, "from_damodaran", lambda c, a=None: calls.append((c, a)) or [c])
    assert us.get_universe("hk") == ["Hong Kong"]
    assert calls[-1] == ("Hong Kong", {"SEHK"})
    expected = {
        "TW": ("Taiwan", {"TWSE", "TPEX"}),
        "CA": ("Canada", {"TSX", "TSXV", "CNSX"}),
        "AU": ("Australia", {"ASX"}),
        "BR": ("Brazil", {"BOVESPA"}),
        "ZA": ("South Africa", {"JSE"}),
        "SA": ("Saudi Arabia", {"SASE"}),
        "SE": ("Sweden", None),
        "DK": ("Denmark", None),
        "FI": ("Finland", None),
        "UK": ("United Kingdom", {"LSE", "AIM"}),
        "DE": ("Germany", {"XTRA", "DB", "HMSE"}),
        "CH": ("Switzerland", {"SWX"}),
        "TH": ("Thailand", {"SET"}),
        "ID": ("Indonesia", {"IDX"}),
        "MY": ("Malaysia", {"KLSE"}),
        "MX": ("Mexico", {"BMV"}),
        "IL": ("Israel", {"TASE"}),
    }
    for m, exp in expected.items():
        calls.clear()
        us.get_universe(m)
        assert calls == [exp], m
    assert {"US", "IN", "JP", "KR", "SG", "CN", "EU"} <= set(us.PROVIDERS)
    with pytest.raises(KeyError):
        us.get_universe("nope")


def test_main_block(monkeypatch, capsys):
    # run as __main__ with the real module's globals patched via a stub requests
    monkeypatch.setattr(sys, "argv", ["universe_sources.py", "xx", "us"])
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp({"0": {"ticker": "x"}}))
    runpy.run_path(us.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "XX: ERROR" in out
    assert "US:      1 tickers   e.g. ['X']" in out


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="universe_sources.py:~283 `not tkr` is False for a NaN ticker (what read_excel / "
    "pandas>=3 produce for blank cells), so 'nan' is emitted as a ticker",
)
def test_from_damodaran_nan_ticker_is_skipped(monkeypatch):
    df = pd.DataFrame(
        {
            "Country": ["Hong Kong", "Hong Kong"],
            "Exchange": ["SEHK", "SEHK"],
            "Ticker": ["5", np.nan],
        }
    )
    damo(monkeypatch, df)
    assert us.from_damodaran("Hong Kong", {"SEHK"}) == ["0005.HK"]
