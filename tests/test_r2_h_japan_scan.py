import io
import json
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest

from tests.test_r2_h_market_scans_common import load  # noqa: F401

NAME = "full_japan_market_scan"


@pytest.fixture
def m(load):  # noqa: F811
    return load(NAME)


# ── cache ────────────────────────────────────────────────────────────────────────
def test_cache_roundtrip_and_ttl(m, monkeypatch):
    assert m._load_cache() is None  # no file
    m._save_cache([{"code": "1"}])
    assert m._load_cache() == [{"code": "1"}]
    data = json.loads(m._CACHE_FILE.read_text())
    ttl = m.SYMBOL_CACHE_TTL
    data["ts"] = time.time() - ttl + 100  # still fresh
    m._CACHE_FILE.write_text(json.dumps(data))
    assert m._load_cache() == [{"code": "1"}]
    data["ts"] = time.time() - ttl - 100  # stale
    m._CACHE_FILE.write_text(json.dumps(data))
    assert m._load_cache() is None
    m._CACHE_FILE.write_text("not json")
    assert m._load_cache() is None
    m._CACHE_FILE.write_text(json.dumps({"tickers": [1]}))  # missing ts -> epoch 0 -> stale
    assert m._load_cache() is None


def test_save_cache_swallows_errors(m, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_CACHE_FILE", tmp_path / "no" / "such" / "f.json")
    m._save_cache([1])  # must not raise
    assert not (tmp_path / "no").exists()


# ── kabupy ───────────────────────────────────────────────────────────────────────
def _kabupy(monkeypatch, issues=None, exc=None):
    mod = types.ModuleType("kabupy")

    class Jpx:
        @property
        def issues(self):
            if exc:
                raise exc
            return issues

    mod.Jpx = Jpx
    monkeypatch.setitem(sys.modules, "kabupy", mod)


def test_kabupy_filters_and_normalises(m, monkeypatch, capsys):
    _kabupy(
        monkeypatch,
        [
            {"category": "プライム（内国株式）", "security_code": "7203.0", "name": "Toyota",
             "33_industry_category": "Auto"},
            {"category": "スタンダード（内国株式）", "security_code": "98", "name": "Small",
             "17_industry_category": "Misc"},
            {"category": "ETF", "security_code": "1306", "name": "ETF"},
            {"category": "（内国株式）", "security_code": "ABCD", "name": "Bad"},
            {"category": "（内国株式）", "security_code": "12345", "name": "TooLong"},
            {"category": "（内国株式）", "security_code": "6758"},
        ],
    )  # fmt: skip
    out = m.fetch_tse_universe_kabupy()
    assert out == [
        {"code": "7203", "name": "Toyota", "sector": "Auto", "market": "プライム"},
        {"code": "0098", "name": "Small", "sector": "Misc", "market": "スタンダード"},
        {"code": "6758", "name": "", "sector": "—", "market": ""},
    ]
    assert "kabupy → 3 TSE domestic equities" in capsys.readouterr().out


def test_kabupy_empty_and_errors(m, monkeypatch, capsys):
    _kabupy(monkeypatch, [])
    assert m.fetch_tse_universe_kabupy() == []
    assert "TSE domestic" not in capsys.readouterr().out
    monkeypatch.setitem(sys.modules, "kabupy", None)  # import raises ImportError
    assert m.fetch_tse_universe_kabupy() == []
    assert "kabupy not installed" in capsys.readouterr().out
    _kabupy(monkeypatch, exc=RuntimeError("boom"))
    assert m.fetch_tse_universe_kabupy() == []
    assert "kabupy error: boom" in capsys.readouterr().out


# ── JPX excel ────────────────────────────────────────────────────────────────────
def _xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _resp(content, status=200):
    class R:
        def __init__(self):
            self.content = content

        def raise_for_status(self):
            if status != 200:
                raise RuntimeError(f"HTTP {status}")

    return R()


def _get(monkeypatch, m, content, status=200, seen=None):
    def get(url, headers=None, timeout=None):
        if seen is not None:
            seen.append((url, headers, timeout))
        return _resp(content, status)

    monkeypatch.setattr(m.requests, "get", get)


JPX = pd.DataFrame(
    {
        "コード": ["7203", "1301.0", "99", "XXXX", "6758"],
        "銘柄名": ["Toyota", "Kyokuyo", "Tiny", "Bad", "Sony"],
        "33業種区分": ["Auto", "Fish", "Misc", "x", "Elec"],
        "市場・商品区分": [
            "プライム（内国株式）",
            "スタンダード（内国株式）",
            "ETF・ETN",
            "プライム（内国株式）",
            "グロース（内国株式）",
        ],
    }
)


def test_jpx_filters_markets_and_codes(m, monkeypatch):
    seen = []
    _get(monkeypatch, m, _xlsx(JPX), seen=seen)
    out = m.fetch_tse_universe_jpx()
    assert [(s["code"], s["name"], s["sector"]) for s in out] == [
        ("7203", "Toyota", "Auto"),
        ("1301", "Kyokuyo", "Fish"),
    ]
    assert seen[0][0] == m.JPX_MASTER_URL and seen[0][2] == 30
    assert "Mozilla" in seen[0][1]["User-Agent"]


def test_jpx_without_market_or_name_or_sector(m, monkeypatch):
    df = pd.DataFrame({"Code": ["7203", "99"]})
    _get(monkeypatch, m, _xlsx(df))
    out = m.fetch_tse_universe_jpx()
    assert out == [
        {"code": "7203", "name": "", "sector": "—"},
        {"code": "0099", "name": "", "sector": "—"},
    ]


def test_jpx_no_code_column(m, monkeypatch, capsys):
    _get(monkeypatch, m, _xlsx(pd.DataFrame({"foo": [1]})))
    assert m.fetch_tse_universe_jpx() == []
    assert "Cannot identify code column" in capsys.readouterr().out


def test_jpx_errors(m, monkeypatch, capsys):
    _get(monkeypatch, m, b"", status=503)
    assert m.fetch_tse_universe_jpx() == []
    assert "JPX fetch error: HTTP 503" in capsys.readouterr().out
    _get(monkeypatch, m, b"not an excel file")
    assert m.fetch_tse_universe_jpx() == []


# ── universe ─────────────────────────────────────────────────────────────────────
def test_universe_cache_hit(m, monkeypatch, capsys):
    m._save_cache([{"yf_ticker": "1.T"}])
    monkeypatch.setattr(m, "fetch_tse_universe_kabupy", lambda: pytest.fail("no"))
    assert m.build_ticker_universe() == [{"yf_ticker": "1.T"}]
    assert "cache hit: 1 TSE tickers" in capsys.readouterr().out


def test_universe_empty_cache_is_a_miss(m, monkeypatch):
    m._save_cache([])
    monkeypatch.setattr(m, "fetch_tse_universe_kabupy", lambda: [])
    monkeypatch.setattr(
        m, "fetch_tse_universe_jpx", lambda: [{"code": "1111", "name": "n", "sector": "s"}]
    )
    assert m.build_ticker_universe()[0]["yf_ticker"] == "1111.T"


def test_universe_kabupy_dedupes_and_caches(m, monkeypatch):
    rows = [
        {"code": "1", "name": "a", "sector": "s"},
        {"code": "1", "name": "dup", "sector": "s"},
        {"code": "2", "name": "b", "sector": "t"},
    ]
    monkeypatch.setattr(m, "fetch_tse_universe_kabupy", lambda: rows)
    monkeypatch.setattr(m, "fetch_tse_universe_jpx", lambda: pytest.fail("no"))
    out = m.build_ticker_universe()
    assert out == [
        {"yf_ticker": "1.T", "code": "1", "name": "a", "sector": "s"},
        {"yf_ticker": "2.T", "code": "2", "name": "b", "sector": "t"},
    ]
    assert m._load_cache() == out


def test_universe_falls_back_to_jpx_then_exits(m, monkeypatch):
    monkeypatch.setattr(m, "fetch_tse_universe_kabupy", lambda: [])
    monkeypatch.setattr(m, "fetch_tse_universe_jpx", lambda: [])
    with pytest.raises(SystemExit) as e:
        m.build_ticker_universe()
    assert "Could not fetch TSE universe" in str(e.value)


# ── bulk download ────────────────────────────────────────────────────────────────
def _ohlc(n=10):
    return pd.DataFrame({"Close": np.arange(n, dtype=float), "High": 1.0}, index=pd.RangeIndex(n))


def _multi(tickers, n=10):
    return pd.concat({t: _ohlc(n) for t in tickers}, axis=1).swaplevel(axis=1)


def test_bulk_download_batches_multiindex(m, monkeypatch, capsys):
    monkeypatch.setattr(m, "BATCH_SIZE", 2)
    calls, sleeps = [], []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleeps.append(s))

    def dl(batch, **kw):
        calls.append((list(batch), kw))
        return _multi(batch, n=m.DARVAS_CONFIRM + 5 if batch[0] == "C.T" else 10)

    monkeypatch.setattr(m.yf, "download", dl)
    out = m.bulk_download_ohlc(["A.T", "B.T", "C.T", "D.T", "E.T"], period="6mo")
    assert [c[0] for c in calls] == [["A.T", "B.T"], ["C.T", "D.T"], ["E.T"]]
    assert calls[0][1]["period"] == "6mo" and calls[0][1]["auto_adjust"] is True
    assert sorted(out) == ["A.T", "B.T", "C.T", "D.T", "E.T"]
    assert sleeps == [m.SLEEP_BETWEEN] * 2  # none after the last batch
    assert "Batch 1/3" in capsys.readouterr().out


def test_bulk_download_short_and_missing_filtered(m, monkeypatch):
    monkeypatch.setattr(m.yf, "download", lambda b, **k: _multi(["A.T"], n=m.DARVAS_CONFIRM + 4))
    assert m.bulk_download_ohlc(["A.T", "ZZ.T"]) == {}  # too short + KeyError ticker


def test_bulk_download_single_flat_empty_and_error(m, monkeypatch, capsys):
    flat = _ohlc(3)
    monkeypatch.setattr(m.yf, "download", lambda b, **k: flat)
    out = m.bulk_download_ohlc(["A.T"])
    assert list(out) == ["A.T"] and out["A.T"] is flat
    monkeypatch.setattr(m.yf, "download", lambda b, **k: pd.DataFrame())
    assert m.bulk_download_ohlc(["A.T"]) == {}
    assert "empty" in capsys.readouterr().out

    def boom(b, **k):
        raise RuntimeError("rate")

    monkeypatch.setattr(m.yf, "download", boom)
    assert m.bulk_download_ohlc(["A.T"]) == {}
    assert "ERROR — rate" in capsys.readouterr().out


# ── fundamental_scan ─────────────────────────────────────────────────────────────
class _T:
    def __init__(self, inc=None, bal=None, cf=None):
        self.income_stmt, self.balance_sheet, self.cash_flow = inc, bal, cf


FR = pd.DataFrame({"2024": [1.0]}, index=["Net Income"])


def test_fundscan_no_data_and_exception(m, monkeypatch):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _T())
    r = m.fundamental_scan("7203.T")
    assert (
        r["error"] == "no_financial_data" and r["f_score"] is None and r["cc_qualifies"] == "FAIL"
    )

    def boom(s):
        raise RuntimeError("x" * 200)

    monkeypatch.setattr(m.yf, "Ticker", boom)
    r = m.fundamental_scan("7203.T")
    assert r["error"] == "x" * 80


def test_fundscan_falls_back_to_financials_attr(m, monkeypatch):
    t = types.SimpleNamespace(income_stmt=pd.DataFrame(), financials=FR, balance_sheet=FR,
                              cash_flow=None, cashflow=FR)  # fmt: skip
    monkeypatch.setattr(m.yf, "Ticker", lambda s: t)
    monkeypatch.setattr(m, "piotroski_f_score", lambda i, b, c: 3)
    r = m.fundamental_scan("1.T")
    assert r["f_score"] == 3 and r["f_strong"] is False and r["cc_score"] == "—"


def test_fundscan_piotroski_failure_returns_early(m, monkeypatch):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _T(FR, FR, FR))

    def boom(*a):
        raise ValueError

    monkeypatch.setattr(m, "piotroski_f_score", boom)
    r = m.fundamental_scan("1.T")
    assert r["f_score"] is None and "f_strong" not in r and r["error"] is None


@pytest.mark.parametrize("score,strong", [(6, False), (7, True), (9, True)])
def test_fundscan_strong_threshold_and_coffee_can(m, monkeypatch, score, strong):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _T(FR, FR, FR))
    monkeypatch.setattr(m, "piotroski_f_score", lambda i, b, c: score)
    seen = {}

    def cc(inc, bal, cf, ticker, floor):
        seen["floor"] = floor
        return {"cc_qualifies": "PASS", "cc_score": "6/6"}, 2.5e11

    monkeypatch.setattr(m, "coffee_can_core", cc)
    r = m.fundamental_scan("1.T")
    assert r["f_strong"] is strong
    if strong:
        assert seen["floor"] == 1e11
        assert r["cc_qualifies"] == "PASS" and r["market_cap_b_jpy"] == 250.0
    else:
        assert "market_cap_b_jpy" not in r and r["cc_score"] == "—"
        assert r["cc_qualifies"] == "FAIL"


def test_fundscan_coffee_can_failure_and_no_mcap(m, monkeypatch):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _T(FR, FR, FR))
    monkeypatch.setattr(m, "piotroski_f_score", lambda i, b, c: 8)

    def boom(*a):
        raise ValueError

    monkeypatch.setattr(m, "coffee_can_core", boom)
    r = m.fundamental_scan("1.T")
    assert r["f_score"] == 8 and r["cc_qualifies"] == "FAIL"
    monkeypatch.setattr(m, "coffee_can_core", lambda *a: ({}, 0))
    assert m.fundamental_scan("1.T")["market_cap_b_jpy"] is None


# ── technical rows ───────────────────────────────────────────────────────────────
def _px(closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c})


def test_technical_row_ma200(m):
    closes = [100.0] * 199 + [120.0]
    row, darv = m._technical_row(
        "7203.T", _px(closes), {"code": "7203", "name": "T", "sector": "A"}
    )
    assert row["LTP_JPY"] == 120 and row["Change%"] == 20.0
    assert row["200_Day_MA"] == 100.0  # round((199*100+120)/200 = 100.1, 0)
    assert row["Distance_to_200MA%"] == 20.0
    assert row["Trend_Signal"] == "Above 200MA (Uptrend)"
    assert row["Code"] == "7203" and row["Name"] == "T" and row["Sector"] == "A"
    assert row["Darvas_Signal"] == darv["signal"]


def test_technical_row_short_history_and_defaults(m):
    row, _ = m._technical_row("1301.T", _px([100.0] * 30), {})
    assert row["200_Day_MA"] is None and row["Distance_to_200MA%"] is None
    assert row["Trend_Signal"] == "Insufficient History"
    assert row["Code"] == "1301" and row["Name"] == "" and row["Sector"] == "—"
    assert row["Change%"] == 0.0
    # one row only: prev None
    df = pd.DataFrame({"High": [1.0], "Low": [1.0], "Close": [5.0]})
    row, darv = m._technical_row("1.T", df, {})
    assert row["Change%"] is None and darv["signal"] == "INSUFFICIENT_DATA"
    # no valid closes
    df = pd.DataFrame({"High": [1.0] * 9, "Low": [1.0] * 9, "Close": [np.nan] * 9})
    row, _ = m._technical_row("1.T", df, {})
    assert row["LTP_JPY"] is None and row["Change%"] is None


def test_technical_stage_partitions(m, monkeypatch):
    sig = {"A.T": "BREAKOUT_BUY", "B.T": "BREAKDOWN_SELL", "C.T": "IN_BOX"}

    def fake(tkr, df, info):
        return {"YF_Ticker": tkr, "Darvas_Signal": sig[tkr]}, {"signal": sig[tkr]}

    monkeypatch.setattr(m, "_technical_row", fake)
    allr, dv, br = m._technical_stage({k: None for k in sig}, {})
    assert [r["YF_Ticker"] for r in allr] == ["A.T", "B.T", "C.T"]
    assert [r["YF_Ticker"] for r in dv] == ["A.T", "B.T"]
    assert br == ["A.T"]
    assert dv[0] is not allr[0]  # copied


def test_fundamental_row(m):
    tech = {"Code": "1", "Name": "n", "Sector": "s", "LTP_JPY": 5, "Darvas_Signal": "X"}
    row = m._fundamental_row("1.T", tech, {"f_score": 8, "f_strong": True, "cc_qualifies": "PASS"})
    assert row["Piotroski_Strong"] == "YES" and row["CoffeeCan"] == "PASS"
    assert row["CC_Score"] == "—" and row["Error"] == "" and row["Piotroski_Score"] == 8
    row = m._fundamental_row("1.T", tech, {})
    assert row["Piotroski_Strong"] == "NO" and row["CoffeeCan"] == "FAIL"


def test_cap_breakouts(m, monkeypatch, capsys):
    monkeypatch.setattr(m, "MAX_FUND_CANDIDATES", 2)
    rows = [
        {"YF_Ticker": "A", "Upside_to_Top%": -10.0},
        {"YF_Ticker": "B", "Upside_to_Top%": -1.0},
        {"YF_Ticker": "C", "Upside_to_Top%": None},
        {"YF_Ticker": "D", "Upside_to_Top%": 0.5},
    ]
    assert m._cap_breakouts(["A", "B", "C", "D"], rows) == ["D", "B"]
    assert "capped to 2" in capsys.readouterr().out
    assert m._cap_breakouts(["A", "C"], rows) == ["A", "C"]  # == cap -> untouched
    assert capsys.readouterr().out == ""


def test_fundamental_stage(m, monkeypatch, capsys):
    def fs(tkr):
        if tkr == "ERR.T":
            raise RuntimeError("bad")
        return {"f_score": 8, "f_strong": tkr != "WEAK.T", "cc_qualifies": "PASS"}

    monkeypatch.setattr(m, "fundamental_scan", fs)
    darvas = [{"YF_Ticker": t, "Code": t[:-2]} for t in ("OK.T", "WEAK.T", "ERR.T")]
    fund, triple, tick = m._fundamental_stage(["ERR.T", "OK.T", "WEAK.T"], darvas, 1)
    assert sorted(r["YF_Ticker"] for r in fund) == ["OK.T", "WEAK.T"]
    assert [r["YF_Ticker"] for r in triple] == ["OK.T"]
    assert triple[0] is not fund[[r["YF_Ticker"] for r in fund].index("OK.T")]
    out = capsys.readouterr().out
    assert "ERR.T: error — bad" in out and "3/3 done  (triple hits: 1)" in out


# ── output ───────────────────────────────────────────────────────────────────────
def test_save_workbook(m):
    rows = [{"Change%": 1.0, "Code": "a"}, {"Change%": 5.0, "Code": "b"}]
    path = m._save_workbook(rows, rows[:1], [], [])
    assert path.parent == m.DOWNLOAD_DIR and path.name.startswith("japan_market_scan_")
    sheets = pd.read_excel(path, sheet_name=None)
    assert list(sheets) == ["All_Stocks", "Darvas_Signals", "Fundamentals", "Triple_Hits"]
    assert sheets["All_Stocks"]["Code"].tolist() == ["b", "a"]  # sorted by Change% desc
    assert sheets["Fundamentals"].empty


def test_print_summary(m, capsys):
    triple = [
        {"Code": "1", "Name": "Lo", "Piotroski_Score": 7, "CC_Score": "6/6", "LTP_JPY": 1000.0},
        {"Code": "2", "Name": "Hi", "Piotroski_Score": 9, "CC_Score": "6/6", "LTP_JPY": 2000.0},
        {"Code": "3", "Name": "None", "Piotroski_Score": None, "CC_Score": "-", "LTP_JPY": 5.0},
    ]
    m._print_summary([1, 2], ["a"], [1, 2, 3], triple, "P.xlsx")
    out = capsys.readouterr().out
    assert "TSE stocks scanned:    2" in out and "TRIPLE HITS:         3" in out
    assert out.index("[2] Hi") < out.index("[1] Lo") < out.index("[3] None")
    assert "¥2,000" in out and "P.xlsx" in out
    m._print_summary([], [], [], [], "P")
    assert "No Triple Hit stocks found today." in capsys.readouterr().out


# ── main ─────────────────────────────────────────────────────────────────────────
def _wire_main(m, monkeypatch, n=3):
    uni = [
        {"yf_ticker": f"{1000+i}.T", "code": f"{1000+i}", "name": "n", "sector": "s"}
        for i in range(n)
    ]
    monkeypatch.setattr(m, "build_ticker_universe", lambda: uni)
    seen = {}

    def dl(tk, period="3mo"):
        seen["tk"], seen["period"] = list(tk), period
        return {t: _px([100.0] * 30) for t in tk}

    monkeypatch.setattr(m, "bulk_download_ohlc", dl)
    return seen


def test_main_top_and_no_scans(m, monkeypatch, capsys):
    seen = _wire_main(m, monkeypatch)
    monkeypatch.setattr(m, "_fundamental_stage", lambda *a: pytest.fail("skipped"))
    monkeypatch.setattr(sys, "argv", ["x", "--top", "2", "--no-scans"])
    m.main()
    out = capsys.readouterr().out
    assert seen["tk"] == ["1000.T", "1001.T"] and seen["period"] == "3mo"
    assert "(limited to first 2 tickers)" in out and "Stage 4 — Skipped" in out
    assert len(list(m.DOWNLOAD_DIR.glob("*.xlsx"))) == 1


def test_main_runs_fundamentals_when_breakouts(m, monkeypatch, capsys):
    _wire_main(m, monkeypatch)
    monkeypatch.setattr(m, "_technical_stage", lambda o, mt: ([{"Darvas_Signal": "BREAKOUT_BUY", "Code": "1"}],
                                                              [{"Darvas_Signal": "BREAKOUT_BUY"}], ["A.T"]))  # fmt: skip
    called = {}

    def fstage(b, d, w):
        called["args"] = (b, w)
        return [], [], b

    monkeypatch.setattr(m, "_fundamental_stage", fstage)
    monkeypatch.setattr(sys, "argv", ["x", "--workers", "3"])
    m.main()
    assert called["args"] == (["A.T"], 3)
    # no breakouts -> stage skipped even without --no-scans
    called.clear()
    monkeypatch.setattr(m, "_technical_stage", lambda o, mt: ([], [], []))
    m.main()
    assert called == {} and "Stage 4 — Skipped" in capsys.readouterr().out
