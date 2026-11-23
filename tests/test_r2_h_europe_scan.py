import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests.test_r2_h_market_scans_common import load  # noqa: F401

NAME = "full_european_market_scan"


@pytest.fixture
def m(load, monkeypatch):  # noqa: F811
    mod = load(NAME)
    monkeypatch.setattr(mod, "_EMERG_FUND", {"mod": None, "checked": False})
    return mod


def _ef(monkeypatch, source="FMP", result=None, exc=None, avail_exc=None):
    mod = types.ModuleType("emerging_fundamentals")

    def avail():
        if avail_exc:
            raise avail_exc
        return source

    def get(sym):
        if exc:
            raise exc
        return result

    mod.available_source, mod.get_fundamentals = avail, get
    monkeypatch.setitem(sys.modules, "emerging_fundamentals", mod)
    return mod


# ── emerging fundamentals gate ───────────────────────────────────────────────────
@pytest.mark.parametrize("sym", ["SAP.DE", "AKBNK.is.x", "ADS.L", ""])
def test_emerging_only_for_is_pr_bd(m, monkeypatch, sym):
    _ef(monkeypatch, result={"f_score": 5})
    assert m._emerging_fundamentals(sym) is None
    assert m._EMERG_FUND["checked"] is False  # not even probed


@pytest.mark.parametrize("sym", ["AKBNK.IS", "cez.pr", "OTP.BD"])
def test_emerging_suffixes_case_insensitive(m, monkeypatch, sym):
    _ef(monkeypatch, result={"f_score": 5})
    assert m._emerging_fundamentals(sym) == {"f_score": 5}


def test_emerging_no_provider_cached(m, monkeypatch):
    _ef(monkeypatch, source=None, result={"f_score": 1})
    assert m._emerging_fundamentals("A.IS") is None
    assert m._EMERG_FUND == {"mod": None, "checked": True}
    _ef(monkeypatch, source="FMP", result={"f_score": 1})  # probed once only
    assert m._emerging_fundamentals("A.IS") is None


def test_emerging_probe_and_call_errors(m, monkeypatch):
    _ef(monkeypatch, avail_exc=RuntimeError())
    assert m._emerging_fundamentals("A.IS") is None
    assert m._EMERG_FUND["checked"] is True and m._EMERG_FUND["mod"] is None
    m._EMERG_FUND.update(mod=None, checked=False)
    _ef(monkeypatch, exc=RuntimeError())
    assert m._emerging_fundamentals("A.IS") is None


# ── coffee can / valuation ───────────────────────────────────────────────────────
def _stmt(rows, ncols=3):
    cols = pd.date_range("2024-12-31", periods=ncols, freq="-12ME")
    return pd.DataFrame({k: list(v) for k, v in rows.items()}, index=range(ncols)).T.set_axis(
        cols, axis=1
    )


class _Ticker:
    def __init__(self, mcap=None):
        self.fast_info = types.SimpleNamespace(market_cap=mcap)


def _fin():
    inc = _stmt(
        {"Total Revenue": [200, 150, 100], "EBIT": [50, 40, 30], "Net Income": [20, 15, 10]}
    )
    bal = _stmt(
        {
            "Total Assets": [200, 200, 200],
            "Current Liabilities": [100, 100, 100],
            "Long Term Debt": [20, 20, 20],
            "Stockholders Equity": [100, 100, 100],
        }
    )
    cf = _stmt({"Free Cash Flow": [5, 5, 5], "Operating Cash Flow": [8, 8, 8]})
    return inc, bal, cf


def test_coffee_can_pass_and_state(m):
    inc, bal, cf = _fin()
    state = {}
    out = m._coffee_can(inc, bal, cf, {"debtToEquity": 50, "marketCap": 5e9}, _Ticker(), state)
    assert state["mcap"] == 5e9
    assert out["cc_score"] == "6/6" and out["cc_qualifies"] == "PASS"
    assert out["cc_debt_equity"] == 0.5
    assert out["cc_rev_cagr"] is not None and out["cc_roce_avg"] is not None


def test_coffee_can_de_scaling_and_fallback(m):
    inc, bal, cf = _fin()

    def de(info):
        return m._coffee_can(inc, bal, cf, info, _Ticker(), {})["cc_debt_equity"]

    assert de({"debtToEquity": 10, "marketCap": 1e9}) == 10  # <=10 taken as ratio
    assert de({"debtToEquity": 50, "marketCap": 1e9}) == 0.5  # >10 is a percent
    assert de({"marketCap": 1e9}) == 0.2  # LTD / equity from statements
    # statements without equity -> de None -> C3 fails
    _, bal2, _ = _fin()
    bal2 = bal2.drop(index="Stockholders Equity")
    out = m._coffee_can(inc, bal2, cf, {"marketCap": 1e9}, _Ticker(), {})
    assert out["cc_debt_equity"] is None and out["cc_score"] == "5/6"
    bal3 = bal.copy()
    bal3.loc["Stockholders Equity"] = 0
    assert m._coffee_can(inc, bal3, cf, {"marketCap": 1e9}, _Ticker(), {})["cc_score"] == "5/6"


def test_coffee_can_mcap_floor_and_fast_info(m):
    inc, bal, cf = _fin()

    def score(info, tk=None):
        s = {}
        out = m._coffee_can(inc, bal, cf, info, tk or _Ticker(), s)
        return int(out["cc_score"].split("/")[0]), s["mcap"]

    assert score({"debtToEquity": 0.5, "marketCap": 1e9}) == (6, 1e9)
    assert score({"debtToEquity": 0.5, "marketCap": 1e9 - 1}) == (5, 1e9 - 1)
    assert score({"debtToEquity": 0.5}, _Ticker(2e9)) == (6, 2e9)  # fast_info fallback


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="full_european_market_scan.py:242 getattr(fast_info, 'market_cap', 0) returns None "
    "(attribute present but None) so `mcap >= 1e9` raises and the whole Coffee Can result is lost",
)
def test_coffee_can_unknown_market_cap_should_just_fail_c4(m):
    inc, bal, cf = _fin()
    out = m._coffee_can(inc, bal, cf, {"debtToEquity": 0.5}, _Ticker(None), {})
    assert out["cc_score"] == "5/6"


def test_valuation(m):
    res = {}
    m._valuation(
        {"trailingPE": 12.345, "pegRatio": 1.5, "dividendYield": 0.03, "priceToBook": 2.345,
         "sector": "Tech"}, 5e9, res)  # fmt: skip
    assert res == {
        "sector": "Tech", "market_cap_b": 5.0, "pe_ratio": 12.35, "pb_ratio": 2.35,
        "dividend_yield": 3.0, "pegy_ratio": 4.5,
    }  # fmt: skip


def test_valuation_dividend_threshold_and_fallbacks(m):
    def dy(v):
        r = {}
        m._valuation({"trailingPE": 5, "dividendYield": v}, 1e9, r)
        return r["dividend_yield"]

    assert dy(0.14) == 14.0  # fraction -> percent
    assert dy(0.15) == 0.15  # already a percent
    assert dy(3.2) == 3.2
    assert dy(None) == 0.0
    r = {}
    m._valuation({"forwardPE": 8}, 0, r)
    assert r["pe_ratio"] == 8 and r["market_cap_b"] is None and r["pegy_ratio"] is None
    assert r["pb_ratio"] is None and r["sector"] is None
    r = {}
    m._valuation({"regularMarketPrice": 50, "trailingEps": 5}, 1e9, r)
    assert r["pe_ratio"] == 10.0
    r = {}
    m._valuation({"regularMarketPrice": 50}, 1e9, r)  # eps missing -> /1
    assert r["pe_ratio"] == 50.0
    r = {}
    m._valuation({"trailingPE": -3}, 1e9, r)
    assert r["pe_ratio"] is None
    r = {}
    m._valuation({"trailingPE": 5, "pegRatio": 0, "dividendYield": 0.0}, 1e9, r)
    assert r["pegy_ratio"] is None  # 0.0 -> falsy


def test_valuation_raises_but_keeps_sector(m):
    r = {}
    with pytest.raises(TypeError):
        m._valuation({"trailingPE": 5, "sector": "S", "pegRatio": None, "priceToBook": "x"}, 1e9, r)
    assert "sector" in r


# ── fundamental_scan ─────────────────────────────────────────────────────────────
class _FT:
    def __init__(self, inc, info=None, info_exc=None):
        self.income_stmt, self.balance_sheet, self.cash_flow = inc, None, None
        self.fast_info = types.SimpleNamespace(market_cap=0)
        self._info, self._exc = info, info_exc

    @property
    def info(self):
        if self._exc:
            raise self._exc
        return self._info


def test_fundscan_info_failure_and_emerging_fallback(m, monkeypatch):
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _FT(None, info_exc=RuntimeError()))
    monkeypatch.setattr(m, "_emerging_fundamentals", lambda s: {"f_score": 7, "source": "FMP"})
    r = m.fundamental_scan("A.IS")
    assert r["f_score"] == 7 and r["source"] == "FMP" and r["symbol"] == "A.IS"
    monkeypatch.setattr(m, "_emerging_fundamentals", lambda s: None)
    r = m.fundamental_scan("A.IS")
    assert r["error"] == "no_financial_statements" and r["f_score"] is None
    monkeypatch.setattr(m.yf, "Ticker", lambda s: _FT(pd.DataFrame()))
    assert m.fundamental_scan("A.DE")["error"] == "no_financial_statements"


def test_fundscan_ticker_error_and_stage_failures(m, monkeypatch):
    def boom(s):
        raise RuntimeError("z" * 100)

    monkeypatch.setattr(m.yf, "Ticker", boom)
    assert m.fundamental_scan("A.DE")["error"] == "z" * 80
    inc, bal, cf = _fin()
    t = _FT(inc, info={"debtToEquity": 0.5, "marketCap": 2e9, "trailingPE": 9, "sector": "S"})
    t.balance_sheet, t.cash_flow = bal, cf
    monkeypatch.setattr(m.yf, "Ticker", lambda s: t)
    monkeypatch.setattr(m, "piotroski_f_score", lambda i, b, c: 8)
    r = m.fundamental_scan("A.DE")
    assert r["f_score"] == 8 and r["cc_qualifies"] == "PASS"
    assert r["pe_ratio"] == 9 and r["sector"] == "S" and r["market_cap_b"] == 2.0

    def boom2(*a, **k):
        raise ValueError

    monkeypatch.setattr(m, "piotroski_f_score", boom2)
    monkeypatch.setattr(m, "_coffee_can", boom2)
    monkeypatch.setattr(m, "_valuation", boom2)
    r = m.fundamental_scan("A.DE")
    assert r["f_score"] is None and r["cc_qualifies"] == "FAIL" and "pe_ratio" not in r


# ── load_universe ────────────────────────────────────────────────────────────────
def test_load_universe_columns(m, tmp_path):
    p = tmp_path / "u.csv"
    p.write_text(
        "yf_ticker,name,index,exchange\nSAP.DE,SAP,DAX,XETRA\n nan ,x,y,z\n,e,e,e\nAIR.PA,Airbus,CAC,PAR\n"
    )
    meta = m.load_universe(str(p))
    assert meta == {"SAP.DE": ("SAP", "XETRA"), "AIR.PA": ("Airbus", "PAR")}
    assert list(meta) == ["SAP.DE", "AIR.PA"]


def test_load_universe_fallback_columns(m, tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("sym,label\nA.DE,Alpha\n")
    assert m.load_universe(str(p)) == {"A.DE": ("Alpha", "—")}
    p.write_text("sym\nB.DE\n")
    assert m.load_universe(str(p)) == {"B.DE": ("B.DE", "—")}


# ── banner ───────────────────────────────────────────────────────────────────────
def test_banner(m, monkeypatch, capsys):
    _ef(monkeypatch, source="FMP")
    m._print_banner("lbl", 7)
    out = capsys.readouterr().out
    assert "EUROPEAN MARKET SCANNER — lbl  (7 tickers)" in out and "FMP active" in out
    _ef(monkeypatch, source=None)
    m._print_banner("lbl", 7)
    assert "momentum-only" in capsys.readouterr().out
    _ef(monkeypatch, avail_exc=RuntimeError())
    m._print_banner("lbl", 7)
    out = capsys.readouterr().out
    assert "Emerging-market" not in out and "EUROPEAN MARKET SCANNER" in out


# ── download ─────────────────────────────────────────────────────────────────────
def _ohlc(n=25, base=100.0):
    c = np.full(n, base)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": c})


def test_download_chunk_ok_and_params(m, monkeypatch):
    seen = []

    def dl(chunk, **kw):
        seen.append((chunk, kw))
        return "RAW"

    monkeypatch.setattr(m.yf, "download", dl)
    assert m._download_chunk(["A"], 1) == "RAW"
    assert seen[0][1] == {"period": "1y", "group_by": "ticker", "auto_adjust": True,
                          "threads": True, "progress": False}  # fmt: skip


@pytest.mark.parametrize("msg", ["Rate limited!", "429 Too Many Requests"])
def test_download_chunk_retries_once_on_rate_limit(m, monkeypatch, capsys, msg):
    sleeps, calls = [], []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleeps.append(s))

    def dl(chunk, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(msg)
        return "RAW"

    monkeypatch.setattr(m.yf, "download", dl)
    assert m._download_chunk(["A"], 3) == "RAW"
    assert sleeps == [20] and len(calls) == 2
    assert "batch 3 rate-limited" in capsys.readouterr().out


def test_download_chunk_gives_up(m, monkeypatch, capsys):
    calls = []

    def always(chunk, **kw):
        calls.append(1)
        raise RuntimeError("Rate limited")

    monkeypatch.setattr(m.yf, "download", always)
    assert m._download_chunk(["A"], 1) is None
    assert len(calls) == 2  # one retry only
    assert "download failed" in capsys.readouterr().out
    calls.clear()

    def other(chunk, **kw):
        calls.append(1)
        raise RuntimeError("x" * 100)

    monkeypatch.setattr(m.yf, "download", other)
    assert m._download_chunk(["A"], 1) is None
    assert len(calls) == 2  # non-rate-limit error: loop still runs the second attempt


def test_split_chunk_multi_and_flat(m):
    raw = pd.concat({"A": _ohlc(), "B": _ohlc(), "E": _ohlc() * np.nan}, axis=1)
    out = m._split_chunk(raw, ["A", "B", "E", "MISSING"])
    assert sorted(out) == ["A", "B"]
    flat = _ohlc()
    assert list(m._split_chunk(flat, ["A"])) == ["A"]
    assert m._split_chunk(flat, ["A", "B"]) == {}  # ambiguous flat frame
    assert m._split_chunk(_ohlc() * np.nan, ["A"]) == {}


def test_download_ohlc_chunks_and_empty(m, monkeypatch, capsys):
    sleeps = []
    monkeypatch.setattr(m.time, "sleep", lambda s: sleeps.append(s))
    chunks = []

    def dc(chunk, no):
        chunks.append((list(chunk), no))
        if no == 2:
            return None
        if no == 3:
            return pd.DataFrame()
        return pd.concat({s: _ohlc() for s in chunk}, axis=1)

    monkeypatch.setattr(m, "_download_chunk", dc)
    out = m._download_ohlc(["A", "B", "C", "D", "E", "F", "G"], 2)
    assert chunks == [(["A", "B"], 1), (["C", "D"], 2), (["E", "F"], 3), (["G"], 4)]
    assert sorted(out) == ["A", "B", "G"]
    assert sleeps == [2, 2, 2, 2]
    o = capsys.readouterr().out
    assert "batch 1/4 — 2 tickers" in o and "batch 4/4 — 3 tickers" in o


# ── technical ────────────────────────────────────────────────────────────────────
def test_technical_row(m):
    closes = [100.0] * 199 + [120.0]
    c = np.array(closes)
    df = pd.DataFrame({"High": c * 1.01, "Low": c * 0.99, "Close": c})
    row, darv = m._technical_row("SAP.DE", df, {"SAP.DE": ("SAP", "Tech")})
    assert row["Name"] == "SAP" and row["Sector"] == "Tech"
    assert row["LTP"] == 120.0 and row["Change%"] == 20.0
    assert row["200_Day_MA"] == 100.1 and row["Distance_to_200MA%"] == 19.88
    assert row["Trend_Signal"] == "Above 200MA (Uptrend)"
    assert row["Dividend_Yield%"] == 0.0 and row["P_E_Ratio"] is None
    assert row["Darvas_Signal"] == darv["signal"]
    short = pd.DataFrame({"High": [1.0] * 25, "Low": [1.0] * 25, "Close": [1.0] * 25})
    row, _ = m._technical_row("ZZ", short, {})
    assert row["Name"] == "ZZ" and row["Sector"] == "—" and row["200_Day_MA"] is None
    assert row["Trend_Signal"] == "Insufficient History" and row["Change%"] == 0.0
    one = pd.DataFrame({"High": [1.0], "Low": [1.0], "Close": [4.0]})
    assert m._technical_row("Z", one, {})[0]["Change%"] is None


def test_technical_stage(m, monkeypatch, capsys):
    sig = {"A": "BREAKOUT_BUY", "B": "BREAKDOWN_SELL", "C": "IN_BOX"}

    def fake(sym, df, meta):
        if sym == "BOOM":
            raise RuntimeError("x")
        return {"Symbol": sym, "Darvas_Signal": sig[sym]}, {"signal": sig[sym]}

    monkeypatch.setattr(m, "_technical_row", fake)
    big, small = _ohlc(20), _ohlc(19)
    ohlc = {"A": big, "B": big, "C": big, "S": small, "E": pd.DataFrame(), "BOOM": big}
    allm, br = m._technical_stage(["A", "B", "C", "S", "E", "N", "BOOM"], ohlc, {})
    assert list(allm) == ["A", "B", "C"] and br == ["A"]
    assert "Technical error on BOOM: x" in capsys.readouterr().out


def test_fundamental_row_prefers_scan_sector(m):
    tech = {"Name": "n", "Sector": "Exch", "LTP": 5}
    assert m._fundamental_row("A", tech, {"sector": "Real"})["Sector"] == "Real"
    assert m._fundamental_row("A", tech, {"sector": None})["Sector"] == "Exch"
    assert m._fundamental_row("A", tech, {"f_score": 8})["Piotroski_Score"] == 8


def test_fundamental_stage(m, monkeypatch, capsys):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)

    def fs(sym):
        if sym == "ERR":
            raise RuntimeError("bad")
        return {"f_score": {"HI": 7, "LO": 6, "PASS0": 0}[sym], "cc_qualifies": "FAIL" if sym == "LO" else "PASS",
                "market_cap_b": 1.5, "pe_ratio": 9.0, "pb_ratio": 1.1, "dividend_yield": 2.0}  # fmt: skip

    monkeypatch.setattr(m, "fundamental_scan", fs)
    allm = {s: {"Symbol": s} for s in ("ERR", "HI", "LO", "PASS0")}
    fund, triple = m._fundamental_stage(["ERR", "HI", "LO", "PASS0"], allm)
    assert sorted(r["Symbol"] for r in fund) == ["HI", "LO", "PASS0"]
    assert [r["Symbol"] for r in triple] == ["HI"]  # >=7 & PASS; f_score 0 is falsy
    assert allm["HI"]["Market_Cap_B"] == 1.5 and allm["HI"]["P_E_Ratio"] == 9.0
    assert allm["HI"]["P_B_Ratio"] == 1.1 and allm["HI"]["Dividend_Yield%"] == 2.0
    out = capsys.readouterr().out
    assert "Fundamental error on ERR: bad" in out and "4/4 processed" in out


def test_fundamental_stage_five_step_progress(m, monkeypatch, capsys):
    monkeypatch.setattr(m, "MAX_WORKERS", 1)
    monkeypatch.setattr(m, "fundamental_scan", lambda s: {"f_score": 1})
    syms = [f"S{i}" for i in range(6)]
    m._fundamental_stage(syms, {s: {} for s in syms})
    out = capsys.readouterr().out
    assert "5/6 processed" in out and "6/6 processed" in out and "4/6 processed" not in out


def test_save_workbook_names(m):
    rows = [{"Change%": 1.0, "Symbol": "a"}, {"Change%": 2.0, "Symbol": "b"}]
    p = m._save_workbook("european", rows, [], [], [])
    assert p.name.startswith("european_market_scan_2") or p.name.startswith("european_market_scan_")
    assert "european_market_scan_european" not in p.name
    p2 = m._save_workbook("broad", rows, rows[:1], [], [])
    assert p2.name.startswith("european_market_scan_broad_")
    p3 = m._save_workbook(None, rows, [], [], [])
    assert not p3.name.startswith("european_market_scan_None")
    assert pd.read_excel(p2, sheet_name="All_Stocks")["Symbol"].tolist() == ["b", "a"]


def test_print_summary(m, capsys):
    th = [
        {"Symbol": "LO", "Name": "Low", "Piotroski_Score": 7, "CoffeeCan_Score": "6/6", "LTP": 1.0},
        {
            "Symbol": "HI",
            "Name": "High",
            "Piotroski_Score": 9,
            "CoffeeCan_Score": "6/6",
            "LTP": 2.0,
        },
    ]
    m._print_summary([1, 2], ["A"], [1], th, "P.xlsx")
    out = capsys.readouterr().out
    assert out.index("HI") < out.index("LO") and "TRIPLE HITS FOUND:          2" in out
    m._print_summary([], [], [], [], "P")
    assert "No Triple Hit stocks found today." in capsys.readouterr().out


# ── main ─────────────────────────────────────────────────────────────────────────
def test_main_exits_without_data(m, monkeypatch):
    monkeypatch.setattr(m, "_print_banner", lambda *a: None)
    monkeypatch.setattr(m, "_download_ohlc", lambda s, b: {})
    monkeypatch.setattr(sys, "argv", ["x", "--top", "2"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert "No OHLC data downloaded" in str(e.value)


def test_main_universe_label_and_batch(m, monkeypatch, tmp_path, capsys):
    p = tmp_path / "u.csv"
    p.write_text("yf_ticker,name,exchange\nAAA.DE,A,X\nBBB.DE,B,Y\nCCC.DE,C,Z\n")
    got = {}
    monkeypatch.setattr(m, "_print_banner", lambda label, n: got.update(label=label, n=n))

    def dl(symbols, batch):
        got.update(symbols=list(symbols), batch=batch)
        return {s: _ohlc(30) for s in symbols}

    monkeypatch.setattr(m, "_download_ohlc", dl)
    saved = {}
    monkeypatch.setattr(m, "_save_workbook", lambda label, *a: saved.update(label=label) or "P")
    monkeypatch.setattr(m, "_fundamental_stage", lambda *a: pytest.fail("no breakouts"))
    monkeypatch.setattr(sys, "argv", ["x", "--universe", str(p), "--top", "2", "--batch", "9"])
    m.main()
    assert got == {"label": "european_broad", "n": 2, "symbols": ["AAA.DE", "BBB.DE"], "batch": 9}
    assert saved["label"] == "european_broad"
    assert "Stage 4 — Skipped" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "--universe", str(p), "--label", "mine", "--no-scans"])
    m.main()
    assert saved["label"] == "mine" and got["symbols"] == ["AAA.DE", "BBB.DE", "CCC.DE"]
