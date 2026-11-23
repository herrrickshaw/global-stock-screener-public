import datetime as dt
import sys

import pandas as pd
import pytest

import bulk_seed
import groww_source as gs
from tests.r2g_support import fake_module


# ---------------------------------------------------------------- bulk_seed
def test_normalise_aliases_and_cleaning():
    df = pd.DataFrame(
        {
            "TckrSymb": [" nse:abc ", "BSE:XYZ", "bad", "q"],
            "TradDt": ["2024-01-02", "2024-01-02", "notadate", "2024-01-03"],
            "OpnPric": [1, 2, 3, 4],
            "HghPric": [1, 2, 3, 4],
            "LwPric": [1, 2, 3, 4],
            "ClsPric": [10, "x", 3, None],
            "TotTrdQty": [5, 6, 7, 8],
            "extra": [0, 0, 0, 0],
        }
    )
    out = bulk_seed._normalise(df)
    assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
    # row2: Close non-numeric -> dropped; row3: bad date; row4: Close NaN
    assert list(out["Symbol"]) == ["ABC"]
    assert out["Close"].iloc[0] == 10


def test_normalise_missing_columns_gives_empty():
    out = bulk_seed._normalise(pd.DataFrame({"Date": ["2024-01-01"], "Close": [1]}))
    assert out.empty and "Symbol" in out.columns


class FakeMM:
    def __init__(self, ltm=None):
        self.ltm = ltm or {}
        self.commits = []

    def read(self, market, tier):
        assert tier == "ltm"
        return self.ltm

    def to_long(self, d):
        return pd.concat([x.assign(Symbol=k) for k, x in d.items()])

    def commit(self, market, long, verbose=True):
        self.commits.append((market, long))
        return {"market": market, "rows": len(long)}


@pytest.fixture
def fmm(monkeypatch):
    f = FakeMM({"AAA": 1, "BBB": 2})
    monkeypatch.setattr(bulk_seed, "mm", f)
    return f


def test_from_yfinance_no_symbols(fmm):
    fmm.ltm = {}
    assert bulk_seed.from_yfinance("IN") == {"market": "IN", "error": "no symbols to backfill"}


def test_from_yfinance_strips_suffix(fmm, monkeypatch, capsys):
    seen = {}

    def bd(tickers, period, batch_size, min_bars, verbose):
        seen.update(t=tickers, p=period, b=batch_size, m=min_bars)
        return {"AAA.NS": pd.DataFrame({"Close": [1.0]}), "OTHER": pd.DataFrame({"Close": [2.0]})}

    monkeypatch.setitem(sys.modules, "stock_utils", fake_module("stock_utils", bulk_download=bd))
    res = bulk_seed.from_yfinance("IN")
    assert seen == {"t": ["AAA.NS", "BBB.NS"], "p": "5y", "b": 100, "m": 1}
    assert res["fetched"] == 2 and res["rows"] == 2
    assert set(fmm.commits[0][1]["Symbol"]) == {"AAA", "OTHER"}
    assert "yfinance backfill: 2 IN tickers, period=5y" in capsys.readouterr().out
    # empty suffix keeps tickers as is; explicit symbols bypass mm.read
    seen.clear()
    bulk_seed.from_yfinance("US", symbols=["Z"], suffix="", period="1y", batch=7, verbose=False)
    assert seen["t"] == ["Z"] and seen["b"] == 7 and seen["p"] == "1y"


def test_from_yfinance_nothing_fetched(fmm, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "stock_utils", fake_module("stock_utils", bulk_download=lambda *a, **k: {})
    )
    assert bulk_seed.from_yfinance("IN", verbose=False) == {"market": "IN", "fetched": 0}
    assert fmm.commits == []


def _csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


GOOD = {
    "symbol": ["A", "A", "B"],
    "date": ["2024-01-01", "2024-01-01", "2024-01-01"],
    "open": [1, 1, 2],
    "high": [1, 1, 2],
    "low": [1, 1, 2],
    "close": [1, 1, 2],
    "volume": [1, 1, 2],
}


def test_from_csv_missing_and_unparseable(fmm, tmp_path):
    assert "error" in bulk_seed.from_csv("IN", str(tmp_path / "nope.csv"))
    (tmp_path / "bad.csv").write_text("")  # EmptyDataError swallowed
    _csv(tmp_path / "nocols.csv", {"a": [1]})
    r = bulk_seed.from_csv("IN", str(tmp_path))
    assert r == {"market": "IN", "error": "no parseable rows (check columns)"}


def test_from_csv_dir_dedup_and_file(fmm, tmp_path, capsys):
    _csv(tmp_path / "a.csv", GOOD)
    (tmp_path / "z.csv").write_text("")
    r = bulk_seed.from_csv("IN", str(tmp_path))
    assert r["rows_imported"] == 2  # dup (Date, Symbol) dropped
    assert "2 rows, 2 symbols" in capsys.readouterr().out
    r = bulk_seed.from_csv("IN", str(tmp_path / "a.csv"), verbose=False)
    assert r["rows_imported"] == 2 and capsys.readouterr().out == ""


def test_main_paths(fmm, monkeypatch, tmp_path, capsys):
    calls = {}
    monkeypatch.setattr(
        bulk_seed, "from_yfinance", lambda m, **k: calls.update(yf=(m, k)) or {"y": 1}
    )
    monkeypatch.setattr(bulk_seed, "from_csv", lambda m, p: calls.update(csv=(m, p)) or {"c": 1})
    monkeypatch.setattr(
        sys, "argv", ["x", "--source", "yfinance", "--limit", "1", "--period", "2y"]
    )
    assert bulk_seed.main() == 0
    assert calls["yf"] == ("IN", {"symbols": ["AAA"], "period": "2y", "suffix": ".NS"})
    monkeypatch.setattr(sys, "argv", ["x", "--source", "yfinance"])
    bulk_seed.main()
    assert calls["yf"][1]["symbols"] == ["AAA", "BBB"]
    monkeypatch.setattr(sys, "argv", ["x", "--source", "csv"])
    assert bulk_seed.main() == 1
    assert "--path required" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "--source", "csv", "--path", "p", "--market", "US"])
    assert bulk_seed.main() == 0
    assert calls["csv"] == ("US", "p")
    assert "result: {'c': 1}" in capsys.readouterr().out


# ---------------------------------------------------------------- groww
class FakeGroww:
    EXCHANGE_NSE = "NSE"
    SEGMENT_CASH = "CASH"
    CANDLE_INTERVAL_DAY = "1day"
    minted = []
    fail_mint = False
    candles = {}

    def __init__(self, tok):
        self.tok = tok
        self.calls = []

    @classmethod
    def get_access_token(cls, api_key, secret):
        cls.minted.append((api_key, secret))
        if cls.fail_mint:
            raise RuntimeError("no")
        return "tok-" + str(len(cls.minted))

    def get_all_instruments(self):
        return pd.DataFrame({"a": [1, 2]})

    def get_historical_candles(self, **kw):
        self.calls.append(kw)
        v = FakeGroww.candles.get(kw["groww_symbol"])
        if isinstance(v, Exception):
            raise v
        return v


@pytest.fixture
def gw(monkeypatch, tmp_path):
    FakeGroww.minted = []
    FakeGroww.fail_mint = False
    FakeGroww.candles = {}
    monkeypatch.setattr(gs, "_ENV", tmp_path / ".env")
    monkeypatch.setattr(gs, "_TOKEN_CACHE", {})
    monkeypatch.delenv("GROWW_API_KEY", raising=False)
    monkeypatch.delenv("GROWW_API_SECRET", raising=False)
    monkeypatch.setitem(sys.modules, "growwapi", fake_module("growwapi", GrowwAPI=FakeGroww))
    return gs


def test_load_env(gw, tmp_path):
    assert gw._load_env() == {}
    (tmp_path / ".env").write_text("#C=1\nA=b=c\nnoeq\nD=\n")
    assert gw._load_env() == {"A": "b=c", "D": ""}


def test_access_token_missing_keys(gw, monkeypatch):
    assert gw._access_token() is None
    monkeypatch.setenv("GROWW_API_KEY", "k")
    assert gw._access_token() is None  # secret missing


def test_access_token_mint_cache_and_env_file(gw, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("GROWW_API_KEY=fk\nGROWW_API_SECRET=fs\n")
    assert gw._access_token() == "tok-1"
    assert gw._access_token() == "tok-1"  # cached, not re-minted
    assert FakeGroww.minted == [("fk", "fs")]
    gw._TOKEN_CACHE["day"] = dt.date.today() - dt.timedelta(days=1)
    assert gw._access_token() == "tok-2"


def test_access_token_env_vars_and_failure(gw, monkeypatch):
    monkeypatch.setenv("GROWW_API_KEY", "ek")
    monkeypatch.setenv("GROWW_API_SECRET", "es")
    FakeGroww.fail_mint = True
    assert gw._access_token() is None
    assert gw._TOKEN_CACHE == {}
    FakeGroww.fail_mint = False
    assert gw._access_token() == "tok-2"
    assert FakeGroww.minted[0] == ("ek", "es")


def test_client_and_instruments(gw, monkeypatch):
    assert gw._client() is None
    assert gw.instruments().empty
    monkeypatch.setenv("GROWW_API_KEY", "k")
    monkeypatch.setenv("GROWW_API_SECRET", "s")
    assert isinstance(gw._client(), FakeGroww)
    assert len(gw.instruments()) == 2
    monkeypatch.setattr(FakeGroww, "get_all_instruments", lambda self: 1 / 0)
    assert gw.instruments().empty
    monkeypatch.setitem(sys.modules, "growwapi", None)  # SDK import fails
    assert gw._client() is None


def test_groww_symbol():
    assert gs._groww_symbol("reliance.NS") == "NSE-RELIANCE"
    assert gs._groww_symbol("abc.BO") == "NSE-ABC"
    assert gs._groww_symbol("TCS") == "NSE-TCS"


def _candles(n):
    return {"candles": [[1700000000 + i * 86400, 1, 2, 0.5, 1.5, 100] for i in range(n)]}


def test_fetch_no_client(gw):
    assert gw.fetch(["A"]) == {}


def test_fetch_filters_and_window(gw, monkeypatch, capsys):
    monkeypatch.setenv("GROWW_API_KEY", "k")
    monkeypatch.setenv("GROWW_API_SECRET", "s")
    FakeGroww.candles = {
        "NSE-OK": _candles(60),
        "NSE-SHORT": _candles(59),
        "NSE-NONE": None,
        "NSE-EMPTY": {"candles": []},
        "NSE-ERR": RuntimeError("403"),
        "NSE-LIST": _candles(60)["candles"],
    }
    out = gw.fetch(["OK.NS", "SHORT", "NONE", "EMPTY", "ERR", "LIST"])
    assert sorted(out) == ["LIST", "OK.NS"]
    df = out["OK.NS"]
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.name == "Date" and len(df) == 60
    assert "groww: 2/6 fetched" in capsys.readouterr().out


def test_fetch_period_windows_and_quiet(gw, monkeypatch, capsys):
    monkeypatch.setenv("GROWW_API_KEY", "k")
    monkeypatch.setenv("GROWW_API_SECRET", "s")
    FakeGroww.candles = {"NSE-A": _candles(1)}
    captured = []
    orig = FakeGroww.get_historical_candles

    def spy(self, **kw):
        captured.append(kw)
        return orig(self, **kw)

    monkeypatch.setattr(FakeGroww, "get_historical_candles", spy)
    today = dt.date.today()
    for period, days in [("1y", 371), ("2mo", 67), ("10d", 15), ("garbage", 366), (None, 371)]:
        captured.clear()
        out = gw.fetch(["A"], period=period, min_bars=1, verbose=False)
        assert "A" in out
        want = (today - dt.timedelta(days=days)).strftime("%Y-%m-%d 09:15:00")
        assert captured[0]["start_time"] == want
        assert captured[0]["end_time"] == today.strftime("%Y-%m-%d 15:30:00")
        assert captured[0]["exchange"] == "NSE"
    assert capsys.readouterr().out == ""


def test_fetch_sdk_import_fails_after_client(gw, monkeypatch):
    monkeypatch.setattr(gs, "_client", lambda: object())
    monkeypatch.setitem(sys.modules, "growwapi", None)
    assert gs.fetch(["A"]) == {}


def test_groww_main(gw, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert gw.main() == 0
    out = capsys.readouterr().out
    assert "auth: no keys / SDK" in out and "ohlc: 0/1" in out and "instruments" not in out
    monkeypatch.setenv("GROWW_API_KEY", "k")
    monkeypatch.setenv("GROWW_API_SECRET", "s")
    monkeypatch.setattr(sys, "argv", ["x", "--instruments", "--ohlc", "A", "B"])
    gw.main()
    out = capsys.readouterr().out
    assert "auth: OK (token minted)" in out and "instruments: 2 rows" in out and "ohlc: 0/2" in out
