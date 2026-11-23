import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests._r2i import fresh


def _ohlc(n=5, start="2026-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    c = np.arange(1, n + 1, dtype=float) + 10
    return pd.DataFrame(
        {"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": np.full(n, 1000)}, index=idx
    )


# ---------------- build_market_seeds ----------------
@pytest.fixture
def bms(monkeypatch, tmp_path):
    return fresh(monkeypatch, tmp_path, "build_market_seeds")


def test_import_without_stock_utils(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "stock_utils", None)
    assert fresh(monkeypatch, tmp_path, "build_market_seeds").clean_ohlcv is None


def test_market_of(bms):
    f = bms._market_of
    assert f("AAPL.parquet") == "US"
    assert f("RELIANCE.NS.parquet") == "IN"
    assert f("500325.BO.parquet") == "IN"
    assert f("7203.T.parquet") == "JP"
    assert f("005930.KS.parquet") == "KR"
    assert f("XYZ.ZZ.parquet") is None
    assert f("BRK.B") is None  # name with a dot but unknown suffix
    assert f("AAPL") == "US"


def test_build_missing_dir(bms, tmp_path):
    bms.OHLC_DIR = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        bms.build(out=tmp_path / "o")


def _populate(bms, tmp_path, recent=True):
    d = tmp_path / "ohlc"
    d.mkdir()
    start = (pd.Timestamp.today().normalize() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    old = "2020-01-01"
    _ohlc(5, start if recent else old).to_parquet(d / "AAPL.parquet")
    _ohlc(5, start).to_parquet(d / "TCS.NS.parquet")
    _ohlc(5, start).to_parquet(d / "X.ZZ.parquet")  # unknown market → skipped
    (d / "BAD.L.parquet").write_text("not parquet")  # unreadable → skipped
    _ohlc(5, start).drop(columns=["Close"]).to_parquet(d / "NOCLOSE.T.parquet")
    pd.DataFrame(columns=["Close"]).to_parquet(d / "EMPTY.DE.parquet")
    bms.OHLC_DIR = d
    return d


def test_build_writes_seeds(bms, tmp_path, capsys, monkeypatch):
    _populate(bms, tmp_path)
    monkeypatch.setattr(bms, "clean_ohlcv", None)
    out = tmp_path / "out"
    s = bms.build(days=400, out=out, verbose=True)
    assert set(s) == {"US", "IN"}
    assert s["US"]["symbols"] == 1 and s["US"]["rows"] == 5
    assert s["US"]["file"] == "cleaned_long_US.parquet"
    assert (out / "cleaned_long_IN.parquet").exists()
    assert "scanning 6 cached tickers" in capsys.readouterr().out


def test_build_quiet_and_cutoff(bms, tmp_path, capsys, monkeypatch):
    _populate(bms, tmp_path, recent=False)  # AAPL has only 2020 data → filtered out
    monkeypatch.setattr(bms, "clean_ohlcv", None)
    s = bms.build(days=400, out=tmp_path / "o", verbose=False)
    assert set(s) == {"IN"}
    assert capsys.readouterr().out == ""


def test_build_uses_cleaner(bms, tmp_path, monkeypatch):
    _populate(bms, tmp_path)
    calls = []

    def clean(d, ticker=None, min_bars=None):
        calls.append((ticker, min_bars))
        return None if ticker == "AAPL" else d

    monkeypatch.setattr(bms, "clean_ohlcv", clean)
    s = bms.build(out=tmp_path / "o", verbose=False)
    assert set(s) == {"IN"}
    assert ("AAPL", 1) in calls and ("TCS.NS", 1) in calls


# ---------------- fetch_market_ohlc ----------------
@pytest.fixture
def fmo(monkeypatch, tmp_path):
    return fresh(monkeypatch, tmp_path, "fetch_market_ohlc")


def test_tickers_cn_dedup_and_unknown(fmo, monkeypatch):
    monkeypatch.setattr(fmo, "CN_TICKERS", ["A.SS", "B.SZ", "A.SS"])
    assert fmo._tickers("CN") == ["A.SS", "B.SZ"]
    with pytest.raises(ValueError, match="unknown market XX"):
        fmo._tickers("XX")


def _mod(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def test_tickers_eu_sg(fmo, monkeypatch):
    _mod(monkeypatch, "full_european_market_scan", EURO_STOXX_50_META={"A.PA": 1, "B.DE": 2})
    _mod(monkeypatch, "sg_stock_daily_report", SGX_ALL=["Z.SI", "Y.SI", "Z.SI"])
    assert fmo._tickers("EU") == ["A.PA", "B.DE"]
    assert fmo._tickers("SG") == ["Z.SI", "Y.SI"]


def test_tickers_jp(fmo, monkeypatch):
    uni = [{"code": 1301}, {"yf_ticker": "7203.T", "code": 7203}, {"name": "skip"}, {"code": 0}]
    _mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=lambda: uni,
        fetch_tse_universe_kabupy=lambda: [{"code": 9}],
    )
    assert fmo._tickers("JP") == ["1301.T", "7203.T"]

    def boom():
        raise RuntimeError

    _mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=boom,
        fetch_tse_universe_kabupy=lambda: [{"code": 9}],
    )
    assert fmo._tickers("JP") == ["9.T"]
    _mod(
        monkeypatch,
        "full_japan_market_scan",
        fetch_tse_universe_jpx=lambda: [],
        fetch_tse_universe_kabupy=lambda: [{"code": 8}],
    )
    assert fmo._tickers("JP") == ["8.T"]


def test_tickers_kr(fmo, monkeypatch):
    _mod(
        monkeypatch,
        "full_korea_market_scan",
        build_krx_universe=lambda: [
            {"code": "005930"},
            {"code": "035720", "yf_suffix": ".KQ"},
            {"name": "nocode"},
        ],
    )
    assert fmo._tickers("KR") == ["005930.KS", "035720.KQ"]


def test_write_seed_paths(fmo, tmp_path, monkeypatch):
    monkeypatch.setattr(fmo, "SEED_DIR", tmp_path / "seed")
    monkeypatch.setattr(fmo, "clean_ohlcv", lambda d, ticker=None, min_bars=None: None)
    assert fmo._write_seed("CN", {"A": _ohlc()}) == {"market": "CN", "symbols": 0}
    seen = []

    def clean(d, ticker=None, min_bars=None):
        seen.append(min_bars)
        return d if ticker == "A" else d.iloc[0:0]

    monkeypatch.setattr(fmo, "clean_ohlcv", clean)
    r = fmo._write_seed("CN", {"A": _ohlc(), "B": _ohlc()})
    assert seen == [60, 60]
    assert r["symbols"] == 1 and r["rows"] == 5 and r["file"] == "cleaned_long_CN.parquet"
    assert (tmp_path / "seed" / "cleaned_long_CN.parquet").exists()


def test_run(fmo, monkeypatch, capsys):
    monkeypatch.setattr(fmo, "PROVIDERS", {"TH": 1})
    monkeypatch.setattr(
        fmo,
        "get_universe",
        lambda m: (_ for _ in ()).throw(RuntimeError("down")) if m == "TH" else [],
    )
    got = {}

    def mf(tickers, order, period, min_bars, verbose):
        got.update(t=tickers, order=order, period=period, mb=min_bars)
        return {"x": 1}

    monkeypatch.setattr(fmo, "multi_fetch", mf)
    monkeypatch.setattr(fmo, "_write_seed", lambda m, h: {"market": m, "symbols": 3})
    monkeypatch.setattr(fmo, "_tickers", lambda m: ["Q"])
    s = fmo.run(["TH", "CN"])
    assert s == {"CN": {"market": "CN", "symbols": 3}}
    assert got == {"t": ["Q"], "order": ("yahoo", "stooq"), "period": "1y", "mb": 60}
    out = capsys.readouterr().out
    assert "universe fetch failed: down" in out and "1 tickers" in out


def test_run_uses_provider_universe(fmo, monkeypatch):
    monkeypatch.setattr(fmo, "PROVIDERS", {"TH": 1})
    monkeypatch.setattr(fmo, "get_universe", lambda m: ["T1", "T2"])
    seen = []
    monkeypatch.setattr(fmo, "multi_fetch", lambda t, **k: seen.append(t) or {})
    monkeypatch.setattr(fmo, "_write_seed", lambda m, h: {})
    fmo.run(["TH"])
    assert seen == [["T1", "T2"]]
    monkeypatch.setattr(fmo, "USE_FULL_UNIVERSE", False)
    monkeypatch.setattr(fmo, "_tickers", lambda m: ["L"])
    fmo.run(["TH"])
    assert seen[-1] == ["L"]


# ---------------- export_company_list ----------------
@pytest.fixture
def ecl(monkeypatch, tmp_path):
    m = fresh(monkeypatch, tmp_path, "export_company_list")
    monkeypatch.setattr(m, "OUT_DIR", tmp_path / "cache_seed")
    (tmp_path / "cache_seed").mkdir()
    return m


def _serving(rows):
    return pd.DataFrame(
        {
            "Symbol": [r[0] for r in rows],
            "Close": [r[1] for r in rows],
            "Turnover_USD": [r[2] for r in rows],
            "Liquidity": "A",
            "LastDate": "2026-01-01",
            "Bars": 10,
            "Above200DMA": True,
            "Ret252": 0.1,
            "Extra": 1,
        }
    )


def test_names(ecl, monkeypatch):
    m = types.ModuleType("symbol_master")
    m.load_master = lambda: pd.DataFrame({"symbol": ["TCS", 5], "name": ["Tata", "Five"]})
    monkeypatch.setitem(sys.modules, "symbol_master", m)
    assert ecl._names() == {"TCS": "Tata", "5": "Five"}
    m.load_master = lambda: None
    assert ecl._names() == {}
    m.load_master = lambda: pd.DataFrame()
    assert ecl._names() == {}
    m.load_master = lambda: pd.DataFrame({"symbol": ["a"]})
    assert ecl._names() == {}

    def boom():
        raise RuntimeError

    m.load_master = boom
    assert ecl._names() == {}


def test_build(ecl, monkeypatch):
    monkeypatch.setattr(ecl, "_names", lambda: {"TCS": "Tata"})
    data = {
        "IN": _serving([("TCS", 1, 5.0), ("INFY", 2, 9.0)]),
        "US": _serving([("AAPL", 3, 1.0)]),
        "ZZ": _serving([("Q", 3, 1.0)]),
        "JP": pd.DataFrame(),
    }
    monkeypatch.setattr(ecl.sl, "serving", lambda m: data[m])
    df = ecl.build(["JP", "IN", "US", "ZZ"])
    assert list(df.columns)[:5] == ["Market", "Symbol", "Name", "Exchange", "Currency"]
    assert "Extra" not in df.columns
    # sort: Market asc, turnover desc
    assert list(df["Market"]) == ["IN", "IN", "US", "ZZ"]
    assert list(df["Symbol"][:2]) == ["INFY", "TCS"]
    r = df[df["Symbol"] == "TCS"].iloc[0]
    assert (r["Name"], r["Exchange"], r["Currency"]) == ("Tata", "NSE/BSE", "INR")
    z = df[df["Symbol"] == "Q"].iloc[0]
    assert (z["Exchange"], z["Currency"], z["Name"]) == ("ZZ", "", "")


def test_build_empty_and_default_markets(ecl, monkeypatch):
    monkeypatch.setattr(ecl, "_names", lambda: {})
    monkeypatch.setattr(ecl.datalink, "MARKETS", ["IN"])
    monkeypatch.setattr(
        ecl.sl, "serving", lambda m: _serving([("A", 1, 1.0)]) if m == "IN" else pd.DataFrame()
    )
    assert list(ecl.build()["Market"]) == ["IN"]
    monkeypatch.setattr(ecl.sl, "serving", lambda m: pd.DataFrame())
    assert ecl.build().empty


def test_main(ecl, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--market", "IN"])
    monkeypatch.setattr(ecl, "_names", lambda: {"A": "Alpha"})
    monkeypatch.setattr(ecl.sl, "serving", lambda m: _serving([("A", 1, 1.0), ("B", 1, 2.0)]))
    assert ecl.main() == 0
    out = capsys.readouterr().out
    assert "2 companies across 1 markets (1 with names)" in out
    assert (ecl.OUT_DIR / "company_list.csv").exists()
    assert (ecl.OUT_DIR / "company_list.parquet").exists()
    assert "IN" in out


def test_main_empty(ecl, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    monkeypatch.setattr(ecl, "build", lambda m: pd.DataFrame())
    assert ecl.main() == 1
    assert "no serving views" in capsys.readouterr().out


# ---------------- query ----------------
@pytest.fixture
def q(monkeypatch, tmp_path):
    m = fresh(monkeypatch, tmp_path, "query")
    monkeypatch.setattr(m, "HERE", tmp_path)
    monkeypatch.setattr(m, "_CON", None)
    monkeypatch.setattr(m, "LTM_GLOB", str(tmp_path / "cache_seed" / "ltm" / "*.parquet"))
    monkeypatch.setitem(sys.modules, "duckdb", None)  # import duckdb -> ImportError
    return m


def test_require_missing(q):
    with pytest.raises(RuntimeError, match="needs DuckDB"):
        q._require()
    with pytest.raises(RuntimeError, match="needs DuckDB"):
        q.con()


class _Cur:
    def __init__(self, log):
        self.log = log

    def execute(self, s):
        self.log.append(s)
        return self

    def fetchdf(self):
        return pd.DataFrame({"a": [1]})


def _fake_duck(monkeypatch, log):
    d = types.ModuleType("duckdb")
    d.connect = lambda p: log.append("connect:" + p) or _Cur(log)
    monkeypatch.setitem(sys.modules, "duckdb", d)


def test_con_no_data(q, monkeypatch, tmp_path):
    log = []
    _fake_duck(monkeypatch, log)
    monkeypatch.setattr(q, "LTM_GLOB", str(tmp_path / "cache_seed" / "ltm" / "*.parquet"))
    c = q.con()
    assert log == ["connect::memory:"]
    assert q.con() is c  # cached


def test_con_single_file_views(q, monkeypatch, tmp_path):
    log = []
    _fake_duck(monkeypatch, log)
    (tmp_path / "cache_seed" / "ltm").mkdir(parents=True)
    (tmp_path / "cache_seed" / "ltm" / "IN.parquet").write_text("x")
    (tmp_path / "cache_seed" / "serving").mkdir()
    (tmp_path / "cache_seed" / "serving" / "IN.parquet").write_text("x")
    monkeypatch.setattr(q, "LTM_GLOB", str(tmp_path / "cache_seed" / "ltm" / "*.parquet"))
    q.con()
    assert any("CREATE VIEW ltm" in s and "filename=true" in s for s in log)
    assert any("CREATE VIEW serving" in s for s in log)


def test_con_hive_view(q, monkeypatch, tmp_path):
    log = []
    _fake_duck(monkeypatch, log)
    p = tmp_path / "cache_seed" / "ltm_parts" / "market=IN"
    p.mkdir(parents=True)
    (p / "a.parquet").write_text("x")
    q.con()
    assert any("hive_partitioning=true" in s for s in log)
    assert not any("filename=true" in s for s in log)


def test_con_empty_hive_dir_falls_back(q, monkeypatch, tmp_path):
    log = []
    _fake_duck(monkeypatch, log)
    (tmp_path / "cache_seed" / "ltm_parts").mkdir(parents=True)
    q.con()
    assert not any("CREATE VIEW ltm" in s for s in log)


def test_sql_bars_latest(q, monkeypatch):
    log = []
    _fake_duck(monkeypatch, log)
    assert q.sql("SELECT 1")["a"][0] == 1
    q.bars("TCS")
    q.bars("TCS", "IN", start="2025-01-01", end="2025-02-01")
    q.latest("IN", where="Close > 1")
    q.latest("US")
    sqls = [s for s in log if s.startswith("SELECT")]
    assert "WHERE Symbol = 'TCS' ORDER BY Date" in sqls[1]
    assert (
        "WHERE Symbol = 'TCS' AND Market = 'IN' AND Date >= DATE '2025-01-01' "
        "AND Date <= DATE '2025-02-01' ORDER BY Date"
    ) in sqls[2]
    assert "Market = 'IN'" in sqls[3] and "rn = 1 AND (Close > 1)" in sqls[3]
    assert "rn = 1 AND (TRUE)" in sqls[4]


def test_query_main(q, monkeypatch, capsys):
    log = []
    _fake_duck(monkeypatch, log)
    monkeypatch.setattr(sys, "argv", ["q"])
    assert q.main() == 0
    assert "count(DISTINCT Symbol)" in log[-1]
    monkeypatch.setattr(sys, "argv", ["q", "SELECT 7"])
    assert q.main() == 0
    assert log[-1] == "SELECT 7"
    assert "a" in capsys.readouterr().out
