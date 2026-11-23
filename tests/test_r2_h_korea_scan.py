import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests.test_r2_h_market_scans_common import load  # noqa: F401

NAME = "full_korea_market_scan"


@pytest.fixture
def m(load):  # noqa: F811
    return load(NAME)


# ── cache / universe ─────────────────────────────────────────────────────────────
def test_cache_roundtrip(m):
    import json
    import time

    assert m._load_cache() is None
    m._save_cache([{"code": "1", "market": "KOSPI"}])
    assert m._load_cache() == [{"code": "1", "market": "KOSPI"}]
    d = json.loads(m._CACHE_FILE.read_text())
    d["ts"] = time.time() - m.SYMBOL_CACHE_TTL - 1
    m._CACHE_FILE.write_text(json.dumps(d))
    assert m._load_cache() is None
    m._CACHE_FILE.write_text("{bad")
    assert m._load_cache() is None


def test_save_cache_swallows(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_CACHE_FILE", tmp_path / "x" / "y.json")
    m._save_cache([1])


KIND_HTML = """<table><thead><tr><th>회사명</th><th>종목코드</th><th>업종</th></tr></thead>
<tbody>
<tr><td>Samsung</td><td>5930</td><td>Semi</td></tr>
<tr><td>Bad</td><td>ABC</td><td>x</td></tr>
<tr><td>Long</td><td>1234567</td><td>x</td></tr>
<tr><td>Hynix</td><td>660</td><td>Semi</td></tr>
</tbody></table>"""


def _kind(monkeypatch, m, text=KIND_HTML, exc=None, seen=None):
    # pd.read_html needs lxml/bs4, which CI does not install -> parse the fixture by hand
    def read_html(buf, encoding=None):
        assert encoding == "euc-kr"
        t = buf.read()
        if "<tr>" not in t:
            raise ValueError("No tables found")
        import re

        rows = [
            re.findall(r"<t[dh]>(.*?)</t[dh]>", r) for r in re.findall(r"<tr>(.*?)</tr>", t, re.S)
        ]
        return [pd.DataFrame(rows[1:], columns=rows[0])]

    monkeypatch.setattr(m.pd, "read_html", read_html)

    class R:
        def __init__(self):
            self.text = text

        def raise_for_status(self):
            if exc:
                raise exc

    def get(url, params=None, headers=None, timeout=None):
        if seen is not None:
            seen.append((url, params, headers, timeout))
        return R()

    monkeypatch.setattr(m.requests, "get", get)


def test_fetch_kind_list(m, monkeypatch):
    seen = []
    _kind(monkeypatch, m, seen=seen)
    out = m._fetch_kind_list("stockMkt", "KOSPI", ".KS")
    assert out == [
        {"code": "005930", "name": "Samsung", "sector": "Semi", "market": "KOSPI", "yf_suffix": ".KS"},
        {"code": "000660", "name": "Hynix", "sector": "Semi", "market": "KOSPI", "yf_suffix": ".KS"},
    ]  # fmt: skip
    url, params, headers, timeout = seen[0]
    assert url == m._KIND_URL and params["marketType"] == "stockMkt" and timeout == 20
    assert params["method"] == "download" and headers["Referer"].startswith("https://kind.krx")


def test_fetch_kind_list_errors(m, monkeypatch, capsys):
    _kind(monkeypatch, m, exc=RuntimeError("503"))
    assert m._fetch_kind_list("x", "KOSDAQ", ".KQ") == []
    assert "KIND fetch error (KOSDAQ): 503" in capsys.readouterr().out
    _kind(monkeypatch, m, text="<p>no tables</p>")
    assert m._fetch_kind_list("x", "KOSPI", ".KS") == []


def _stub_lists(m, monkeypatch, kospi, kosdaq):
    calls = []

    def f(param, label, sfx):
        calls.append(label)
        return list(kospi if label == "KOSPI" else kosdaq)

    monkeypatch.setattr(m, "_fetch_kind_list", f)
    return calls


def test_build_universe_both_markets_and_cached(m, monkeypatch):
    a = {"code": "1", "market": "KOSPI"}
    b = {"code": "2", "market": "KOSDAQ"}
    calls = _stub_lists(m, monkeypatch, [a], [b])
    assert m.build_krx_universe() == [a, b]
    assert calls == ["KOSPI", "KOSDAQ"]
    assert m._load_cache() == [a, b]
    calls.clear()
    assert m.build_krx_universe() == [a, b] and calls == []  # cache hit
    assert m.build_krx_universe(kospi_only=True) == [a]


def test_build_universe_kospi_only_skips_kosdaq(m, monkeypatch):
    a = {"code": "1", "market": "KOSPI"}
    calls = _stub_lists(m, monkeypatch, [a], [{"code": "2", "market": "KOSDAQ"}])
    assert m.build_krx_universe(kospi_only=True) == [a]
    assert calls == ["KOSPI"]


def test_build_universe_empty_exits(m, monkeypatch):
    _stub_lists(m, monkeypatch, [], [])
    with pytest.raises(SystemExit) as e:
        m.build_krx_universe()
    assert "Could not fetch any KRX tickers" in str(e.value)


# ── ohlc ─────────────────────────────────────────────────────────────────────────
def test_normalise_cols(m):
    df = pd.DataFrame(columns=["시가", "고가", "저가", "종가", "거래량", "등락률"])
    assert list(m._normalise_krx_cols(df).columns) == [
        "Open", "High", "Low", "Close", "Volume", "등락률",
    ]  # fmt: skip
    assert list(m._normalise_krx_cols(pd.DataFrame(columns=["High"])).columns) == ["High"]


def _krx_df(n, cols=("고가", "저가", "종가")):
    return pd.DataFrame({c: np.arange(1, n + 1, dtype=float) for c in cols})


def test_fetch_krx_ohlc(m, monkeypatch, capsys):
    calls = []
    ok_n = m.DARVAS_CONFIRM + 5
    data = {
        "000001": _krx_df(ok_n),
        "000002": _krx_df(ok_n - 1),  # too short
        "000003": None,
        "000004": pd.DataFrame(),
        "000005": _krx_df(ok_n, cols=("고가", "저가")),  # missing Close
        "000006": "boom",
    }

    def get(f, t, code):
        calls.append((f, t, code))
        v = data[code]
        if isinstance(v, str):
            raise RuntimeError
        return v

    monkeypatch.setattr(m, "krx", types.SimpleNamespace(get_market_ohlcv_by_date=get))
    sleeps = []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleeps.append(s))
    out = m.fetch_krx_ohlc([{"code": c} for c in data], months=2)
    assert list(out) == ["000001"] and list(out["000001"].columns) == ["High", "Low", "Close"]
    f, t, _ = calls[0]
    from datetime import datetime

    assert (datetime.strptime(t, "%Y%m%d") - datetime.strptime(f, "%Y%m%d")).days == 62
    assert len(sleeps) == 4  # the None/empty frames `continue` before the sleep
    assert "1 tickers with usable OHLC" in capsys.readouterr().out


def test_fetch_krx_ohlc_progress_every_200(m, monkeypatch, capsys):
    monkeypatch.setattr(
        m, "krx", types.SimpleNamespace(get_market_ohlcv_by_date=lambda f, t, c: _krx_df(2))
    )
    m.fetch_krx_ohlc([{"code": f"{i:06d}"} for i in range(200)])
    assert "200/200 fetched (0 usable)" in capsys.readouterr().out


# ── fundamental_scan ─────────────────────────────────────────────────────────────
FR = pd.DataFrame({"2024": [1.0]}, index=["Net Income"])


class _T:
    def __init__(self, inc=None):
        self.income_stmt, self.balance_sheet, self.cash_flow = inc, FR, FR


def test_fundscan_paths(m, monkeypatch):
    seen = []
    monkeypatch.setattr(m.yf, "Ticker", lambda s: seen.append(s) or _T())
    r = m.fundamental_scan("005930", ".KS")
    assert seen == ["005930.KS"] and r["error"] == "no_financial_data"
    assert r["code"] == "005930" and r["yf_ticker"] == "005930.KS"

    def boom(s):
        raise RuntimeError("y" * 100)

    monkeypatch.setattr(m.yf, "Ticker", boom)
    assert m.fundamental_scan("1", ".KQ")["error"] == "y" * 80


@pytest.mark.parametrize("score,strong", [(6, False), (7, True)])
def test_fundscan_strong_and_cc(m, monkeypatch, score, strong):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _T(FR))
    monkeypatch.setattr(m, "piotroski_f_score", lambda i, b, c: score)
    seen = {}

    def cc(inc, bal, cf, t, floor):
        seen["floor"] = floor
        return {"cc_qualifies": "PASS"}, 3.3e10

    monkeypatch.setattr(m, "coffee_can_core", cc)
    r = m.fundamental_scan("1", ".KS")
    assert r["f_strong"] is strong
    if strong:
        assert seen["floor"] == 1e11 and r["market_cap_b_krw"] == 330.0
        assert r["cc_qualifies"] == "PASS"
    else:
        assert r["cc_score"] == "—" and "market_cap_b_krw" not in r


def test_fundscan_failures_swallowed(m, monkeypatch):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _T(FR))

    def boom(*a):
        raise ValueError

    monkeypatch.setattr(m, "piotroski_f_score", boom)
    assert m.fundamental_scan("1", ".KS")["f_score"] is None
    monkeypatch.setattr(m, "piotroski_f_score", lambda i, b, c: 9)
    monkeypatch.setattr(m, "coffee_can_core", boom)
    r = m.fundamental_scan("1", ".KS")
    assert r["f_score"] == 9 and r["cc_qualifies"] == "FAIL"
    monkeypatch.setattr(m, "coffee_can_core", lambda *a: ({}, None))
    assert m.fundamental_scan("1", ".KS")["market_cap_b_krw"] is None


# ── technical ────────────────────────────────────────────────────────────────────
def _px(closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c})


def test_technical_row(m):
    info = {"name": "Samsung", "market": "KOSPI"}
    row, darv = m._technical_row("005930", info, _px([100.0] * 199 + [120.0]))
    assert row["Code"] == "005930" and row["Name"] == "Samsung" and row["Market"] == "KOSPI"
    assert row["LTP_KRW"] == 120 and row["Change%"] == 20.0
    assert row["200_Day_MA"] == 100.0 and row["Distance_to_200MA%"] == 20.0
    assert row["Trend_Signal"] == "Above 200MA (Uptrend)"
    assert row["Darvas_Signal"] == darv["signal"]
    row, _ = m._technical_row("1", {}, _px([100.0] * 30))
    assert row["200_Day_MA"] is None and row["Trend_Signal"] == "Need 200+ days"
    assert row["Name"] == ""


def test_technical_stage(m, monkeypatch):
    sig = {"1": "BREAKOUT_BUY", "2": "BREAKDOWN_SELL", "3": "IN_BOX", "4": "BREAKOUT_BUY"}

    def fake(code, info, df):
        return {"Code": code, "Darvas_Signal": sig[code]}, {"signal": sig[code]}

    monkeypatch.setattr(m, "_technical_row", fake)
    uni = [{"code": c, "yf_suffix": ".KQ"} for c in ("1", "2", "3", "4", "5")]
    uni[3].pop("yf_suffix")
    ohlc = {c: object() for c in "1234"}  # "5" has no OHLC
    allr, dv, br = m._technical_stage(uni, ohlc)
    assert [r["Code"] for r in allr] == ["1", "2", "3", "4"]
    assert [r["Code"] for r in dv] == ["1", "2", "4"]
    assert br == [("1", ".KQ"), ("4", ".KS")]  # default suffix .KS


def test_fundamental_row_and_cap(m, monkeypatch, capsys):
    tech = {"Name": "n", "Market": "KOSPI", "LTP_KRW": 5}
    row = m._fundamental_row("1", ".KS", tech, {"f_score": 8, "f_strong": True})
    assert row["YF_Ticker"] == "1.KS" and row["Piotroski_Strong"] == "YES"
    assert row["CoffeeCan"] == "FAIL" and row["CC_Score"] == "—" and row["Error"] == ""
    assert m._fundamental_row("1", ".KS", tech, {})["Piotroski_Strong"] == "NO"

    monkeypatch.setattr(m, "MAX_FUND_CANDIDATES", 2)
    rows = [
        {"Code": "1", "Upside_to_Top%": -9.0},
        {"Code": "2", "Upside_to_Top%": -1.0},
        {"Code": "3", "Upside_to_Top%": None},
        {"Code": "4", "Upside_to_Top%": 0.5},
    ]
    info = [(c, ".KS") for c in "1234"]
    assert m._cap_breakouts(info, rows) == [("4", ".KS"), ("2", ".KS")]
    assert "capped to 2" in capsys.readouterr().out
    assert m._cap_breakouts(info[:2], rows) == info[:2]


def test_fundamental_stage(m, monkeypatch, capsys):
    def fs(code, sfx):
        if code == "9":
            raise RuntimeError("bad")
        return {"f_score": 8, "f_strong": code == "1", "cc_qualifies": "PASS"}

    monkeypatch.setattr(m, "fundamental_scan", fs)
    darvas = [{"Code": c} for c in "129"]
    info = [("9", ".KS"), ("1", ".KS"), ("2", ".KQ")]
    fund, triple, got = m._fundamental_stage(info, darvas, 1)
    assert sorted(r["Code"] for r in fund) == ["1", "2"]
    assert [r["Code"] for r in triple] == ["1"]
    assert got == info
    out = capsys.readouterr().out
    assert "9: error — bad" in out and "3/3 done  (triple hits: 1)" in out


def test_save_and_summary(m, capsys):
    rows = [{"Change%": 1.0, "Code": "a"}, {"Change%": 5.0, "Code": "b"}]
    path = m._save_workbook(rows, rows, [], [])
    assert path.name.startswith("korea_market_scan_") and path.exists()
    assert pd.read_excel(path, sheet_name="All_Stocks")["Code"].tolist() == ["b", "a"]
    triple = [
        {"Code": "1", "Name": "L", "Piotroski_Score": 7, "CC_Score": "6/6", "Market": "KOSPI", "LTP_KRW": 1000.0},
        {"Code": "2", "Name": "H", "Piotroski_Score": 9, "CC_Score": "6/6", "Market": "KOSDAQ", "LTP_KRW": 2000.0},
    ]  # fmt: skip
    m._print_summary([1], [("a", ".KS")], [1], triple, "P.xlsx")
    out = capsys.readouterr().out
    assert out.index("[2] H") < out.index("[1] L") and "₩2,000" in out
    assert "KRX stocks scanned:    1" in out
    m._print_summary([], [], [], [], "P")
    assert "No Triple Hit stocks found today." in capsys.readouterr().out


def test_main_flags(m, monkeypatch, capsys):
    uni = [
        {"code": f"{i:06d}", "name": "n", "market": "KOSPI" if i else "KOSDAQ", "yf_suffix": ".KS"}
        for i in range(3)
    ]
    seen = {}

    def bu(kospi_only=False):
        seen["kospi_only"] = kospi_only
        return list(uni)

    monkeypatch.setattr(m, "build_krx_universe", bu)
    monkeypatch.setattr(
        m, "fetch_krx_ohlc", lambda u, months=3: seen.update(n=len(u), months=months) or {}
    )
    monkeypatch.setattr(sys, "argv", ["x", "--kospi-only", "--top", "2", "--no-scans"])
    m.main()
    out = capsys.readouterr().out
    assert seen == {"kospi_only": True, "n": 2, "months": 3}
    assert "(limited to first 2 tickers)" in out
    assert "Total: 2 tickers (1 KOSPI + 1 KOSDAQ)" in out
    assert "Stage 4 — Skipped" in out
