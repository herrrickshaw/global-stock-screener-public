"""Offline tests for market_data_cache (HOME redirected; fake yfinance)."""

import importlib
import json
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mdc(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    saved = sys.modules.pop("market_data_cache", None)
    mod = importlib.import_module("market_data_cache")
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    yield mod
    sys.modules.pop("market_data_cache", None)
    if saved is not None:
        sys.modules["market_data_cache"] = saved


def ohlc(n=5, start="2024-01-01", base=100.0):
    idx = pd.date_range(start, periods=n, freq="D")
    c = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": [1000.0] * n}, index=idx
    )


class FakeYF:
    def __init__(self):
        self.calls = []
        self.handler = lambda *a, **k: ohlc()
        self.tickers = {}

    def download(self, tickers, **kw):
        self.calls.append((tickers, kw))
        return self.handler(tickers, **kw)

    def Ticker(self, t):  # noqa: N802
        return self.tickers[t]


@pytest.fixture
def yf(mdc, monkeypatch):
    f = FakeYF()
    monkeypatch.setattr(mdc, "yf", f, raising=False)
    monkeypatch.setattr(mdc, "_YF_OK", True)
    return f


def test_dirs_created(mdc):
    for d in (mdc.OHLC_DIR, mdc.FUND_DIR, mdc.INDEX_DIR, mdc.META_DIR):
        assert d.is_dir()


def test_meta_roundtrip_and_corrupt(mdc):
    c = mdc.MarketCache(verbose=False)
    assert c._meta == {}
    df = ohlc(3)
    c._update_meta("k", mdc.OHLC_DIR / "x", df)
    m = c._meta["k"]
    assert m["rows"] == 3 and m["from"] == "2024-01-01" and m["to"] == "2024-01-03"
    assert json.loads(mdc.META_FILE.read_text())["k"]["rows"] == 3
    assert mdc.MarketCache()._meta["k"]["rows"] == 3
    c._update_meta("e", mdc.OHLC_DIR / "x", pd.DataFrame())
    assert c._meta["e"]["from"] is None and c._meta["e"]["to"] is None
    mdc.META_FILE.write_text("{not json")
    assert mdc.MarketCache()._meta == {}


def test_is_stale(mdc):
    c = mdc.MarketCache(verbose=False)
    assert c._is_stale("nokey", 1) is True
    c._meta["a"] = {"updated": (datetime.now() - timedelta(hours=2)).isoformat()}
    assert c._is_stale("a", 1) is True
    assert c._is_stale("a", 3) is False
    c._meta["b"] = {"updated": "garbage"}
    assert c._is_stale("b", 1) is True
    c._meta["c"] = {}
    assert c._is_stale("c", 1) is True


def test_last_cached_date(mdc):
    c = mdc.MarketCache(verbose=False)
    assert c._last_cached_date("x") is None
    c._meta["x"] = {"to": "2024-02-03"}
    assert c._last_cached_date("x") == date(2024, 2, 3)
    c._meta["y"] = {"to": "bad"}
    assert c._last_cached_date("y") is None
    c._meta["z"] = {"to": None}
    assert c._last_cached_date("z") is None


def test_ohlc_path_sanitised(mdc):
    c = mdc.MarketCache(verbose=False)
    assert c._ohlc_path("A/B:C.NS").name == "A_B_C.NS.parquet"


def test_get_ohlc_cold_start(mdc, yf, capsys):
    c = mdc.MarketCache(verbose=True)
    df = c.get_ohlc("ABC.NS")
    assert len(df) == 5
    assert c._ohlc_path("ABC.NS").exists()
    assert c._meta["ohlc:ABC.NS"]["rows"] == 5
    out = capsys.readouterr().out
    assert "Cache created: ABC.NS 5 rows (2024-01-01 – 2024-01-05)" in out
    start = yf.calls[0][1]["start"]
    assert start == date.today() - timedelta(days=mdc.OHLC_YEARS * 365 + 30)


def test_get_ohlc_cold_start_empty(mdc, yf):
    yf.handler = lambda *a, **k: pd.DataFrame()
    c = mdc.MarketCache(verbose=False)
    assert c.get_ohlc("ABC.NS").empty
    assert not c._ohlc_path("ABC.NS").exists()


def test_get_ohlc_fresh_memory_and_disk(mdc, yf):
    c = mdc.MarketCache(verbose=False)
    c.get_ohlc("ABC.NS")
    n = len(yf.calls)
    c2 = mdc.MarketCache(verbose=False)
    d1 = c2.get_ohlc("ABC.NS")  # disk
    d2 = c2.get_ohlc("ABC.NS")  # memory
    assert d1 is d2 and len(d1) == 5
    assert len(yf.calls) == n


def _stale(c, key, last="2024-01-05", hours=100):
    c._meta[key]["updated"] = (datetime.now() - timedelta(hours=hours)).isoformat()
    c._meta[key]["to"] = last
    c._save_meta()


def test_get_ohlc_incremental(mdc, yf, capsys):
    c = mdc.MarketCache(verbose=True)
    c.get_ohlc("ABC.NS")
    _stale(c, "ohlc:ABC.NS")
    yf.handler = lambda *a, **k: ohlc(2, start="2024-01-05", base=500.0)  # overlaps last day
    capsys.readouterr()
    df = c.get_ohlc("ABC.NS")
    assert len(df) == 6
    assert yf.calls[-1][1]["start"] == date(2024, 1, 6)
    assert df.loc["2024-01-05", "Close"] == 500.0  # keep=last
    assert "Cache updated: ABC.NS +2 rows (total 6)" in capsys.readouterr().out


def test_get_ohlc_incremental_no_new_rows(mdc, yf):
    c = mdc.MarketCache(verbose=False)
    c.get_ohlc("ABC.NS")
    _stale(c, "ohlc:ABC.NS")
    yf.handler = lambda *a, **k: pd.DataFrame()
    assert len(c.get_ohlc("ABC.NS")) == 5


def test_get_ohlc_up_to_date_boundary(mdc, yf):
    c = mdc.MarketCache(verbose=False)
    c.get_ohlc("ABC.NS")
    n = len(yf.calls)
    # last cached = yesterday -> start == today -> nothing to fetch
    _stale(c, "ohlc:ABC.NS", last=(date.today() - timedelta(days=1)).isoformat())
    assert len(c.get_ohlc("ABC.NS")) == 5
    assert len(yf.calls) == n
    # last cached = 2 days ago -> start == yesterday < today -> fetches
    _stale(c, "ohlc:ABC.NS", last=(date.today() - timedelta(days=2)).isoformat())
    yf.handler = lambda *a, **k: pd.DataFrame()
    c.get_ohlc("ABC.NS")
    assert len(yf.calls) == n + 3  # empty result retried 3x
    assert yf.calls[-1][1]["start"] == date.today() - timedelta(days=1)


def test_get_ohlc_stale_without_last_date_returns_none(mdc, yf):
    """Stale cache whose meta lacks 'to' falls through every branch -> None."""
    c = mdc.MarketCache(verbose=False)
    c.get_ohlc("ABC.NS")
    _stale(c, "ohlc:ABC.NS", last=None)
    assert c.get_ohlc("ABC.NS") is None


def test_get_ohlc_force_redownloads(mdc, yf):
    c = mdc.MarketCache(verbose=False)
    c.get_ohlc("ABC.NS")
    n = len(yf.calls)
    c.get_ohlc("ABC.NS", force=True)
    assert len(yf.calls) == n + 1


def test_yf_download_single(mdc, yf):
    c = mdc.MarketCache(verbose=False)
    raw = ohlc(3)
    raw["Extra"] = 1
    yf.handler = lambda *a, **k: raw
    out = c._yf_download_single("T", date(2024, 1, 1))
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # multiindex
    mi = pd.concat({"T": ohlc(3)}, axis=1).swaplevel(axis=1)
    yf.handler = lambda *a, **k: mi
    assert len(c._yf_download_single("T", date(2024, 1, 1))) == 3
    # drops all-nan rows
    r = ohlc(3)
    r.iloc[1] = np.nan
    yf.handler = lambda *a, **k: r
    assert len(c._yf_download_single("T", date(2024, 1, 1))) == 2
    # empty retries 3 times then returns empty
    yf.calls.clear()
    yf.handler = lambda *a, **k: pd.DataFrame()
    assert c._yf_download_single("T", date(2024, 1, 1)).empty
    assert len(yf.calls) == 3


def test_yf_download_single_errors(mdc, yf, monkeypatch):
    c = mdc.MarketCache(verbose=False)
    sleeps = []
    monkeypatch.setattr(mdc.time, "sleep", sleeps.append)

    def rate(*a, **k):
        raise RuntimeError("429 Too Many")

    yf.handler = rate
    assert c._yf_download_single("T", date(2024, 1, 1)).empty
    assert sleeps == [15, 30, 45]
    assert len(yf.calls) == 3

    yf.calls.clear()

    def other(*a, **k):
        raise ValueError("bad")

    yf.handler = other
    assert c._yf_download_single("T", date(2024, 1, 1)).empty
    assert len(yf.calls) == 1  # break on non-rate error

    yf.calls.clear()

    def rate2(*a, **k):
        raise RuntimeError("Rate limited")

    yf.handler = rate2
    c._yf_download_single("T", date(2024, 1, 1))
    assert len(yf.calls) == 3


def test_no_yf(mdc, monkeypatch):
    monkeypatch.setattr(mdc, "_YF_OK", False)
    c = mdc.MarketCache(verbose=False)
    assert c._yf_download_single("T", date(2024, 1, 1)).empty
    assert c._bulk_yf_download(["T"], date(2024, 1, 1)) == {}
    assert c.get_financials("T") == {}


def _multi(tickers, n=3):
    return pd.concat({f: ohlc(n) for f in ["Open"]}, axis=1)  # unused helper


def multi_raw(tickers, n=3):
    frames = {}
    for t in tickers:
        frames[t] = ohlc(n)
    df = pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)
    return df


def test_bulk_download_multiindex(mdc, yf, capsys):
    c = mdc.MarketCache(verbose=False)
    yf.handler = lambda batch, **k: multi_raw(["A", "B"])
    out = c._bulk_yf_download(["A", "B", "C"], date(2024, 1, 1))
    assert set(out) == {"A", "B"}  # C -> KeyError swallowed
    text = capsys.readouterr().out
    assert "Downloading batch 1/1 (3 tickers)" in text
    assert "OK (2/3)" in text and "Bulk download complete: 2/3 tickers" in text
    assert yf.calls[0][1]["threads"] is True


def test_bulk_download_batches_and_single(mdc, yf, monkeypatch):
    c = mdc.MarketCache(verbose=False)
    monkeypatch.setattr(mdc, "BATCH_SIZE", 2)
    sleeps = []
    monkeypatch.setattr(mdc.time, "sleep", sleeps.append)
    yf.handler = lambda batch, **k: ohlc()  # flat frame -> first of batch
    out = c._bulk_yf_download(["A", "B", "C"], date(2024, 1, 1))
    assert set(out) == {"A", "C"}
    assert len(yf.calls) == 2
    assert sleeps == [mdc.BATCH_SLEEP]  # sleep only between batches


def test_bulk_download_empty_and_errors(mdc, yf, monkeypatch, capsys):
    c = mdc.MarketCache(verbose=False)
    yf.handler = lambda *a, **k: pd.DataFrame()
    assert c._bulk_yf_download(["A"], date(2024, 1, 1)) == {}
    assert "empty" in capsys.readouterr().out
    sleeps = []
    monkeypatch.setattr(mdc.time, "sleep", sleeps.append)

    def rate(*a, **k):
        raise RuntimeError("Too Many Requests")

    yf.handler = rate
    yf.calls.clear()
    c._bulk_yf_download(["A"], date(2024, 1, 1))
    assert sleeps == [30, 60, 90] and len(yf.calls) == 3
    assert "Rate limited — waiting 30s" in capsys.readouterr().out

    def other(*a, **k):
        raise ValueError("oops")

    yf.handler = other
    yf.calls.clear()
    c._bulk_yf_download(["A"], date(2024, 1, 1))
    assert len(yf.calls) == 1
    assert "ERROR: oops" in capsys.readouterr().out
    for msg in ("Rate", "429"):
        yf.calls.clear()
        yf.handler = lambda *a, m=msg, **k: (_ for _ in ()).throw(RuntimeError(m))
        c._bulk_yf_download(["A"], date(2024, 1, 1))
        assert len(yf.calls) == 3


def test_clean_ohlc(mdc):
    assert mdc.MarketCache._clean_ohlc(pd.DataFrame()).empty
    df = ohlc(4)
    df.index = [str(i.date()) for i in df.index]
    df.index = list(df.index[:3]) + ["not a date"]
    out = mdc.MarketCache._clean_ohlc(df)
    assert isinstance(out.index, pd.DatetimeIndex) and len(out) == 3
    # shuffled -> sorted
    d = ohlc(5).iloc[[3, 0, 4, 1, 2]]
    out = mdc.MarketCache._clean_ohlc(d)
    assert out.index.is_monotonic_increasing


def test_clean_ohlc_central_none_and_fallback(mdc, monkeypatch):
    monkeypatch.setattr(mdc, "_clean_ohlcv_central", lambda df, min_bars: None)
    out = mdc.MarketCache._clean_ohlc(ohlc(3))
    assert out.empty
    monkeypatch.setattr(mdc, "_clean_ohlcv_central", None)
    df = ohlc(4)
    df.iloc[1, 0] = np.nan
    dup = pd.concat([df, df.iloc[[0]].assign(Close=999.0)]).sort_index(kind="stable")
    out = mdc.MarketCache._clean_ohlc(dup)
    assert len(out) == 4 and out.iloc[0]["Close"] == 999.0  # keep last
    assert out.iloc[1]["Open"] == 100.0  # ffilled


def test_write_parquet_atomic_and_failure(mdc, tmp_path, monkeypatch):
    p = tmp_path / "w.parquet"
    mdc.MarketCache._write_parquet(p, ohlc(2))
    assert p.exists() and not p.with_suffix(".tmp").exists()

    def boom(self, tmp, **kw):
        tmp = type(p)(tmp)
        tmp.write_text("partial")
        raise OSError("disk")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    q = tmp_path / "q.parquet"
    with pytest.raises(OSError):
        mdc.MarketCache._write_parquet(q, ohlc(2))
    assert not q.with_suffix(".tmp").exists() and not q.exists()

    def boom2(self, tmp, **kw):
        raise OSError("early")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom2)
    with pytest.raises(OSError):
        mdc.MarketCache._write_parquet(q, ohlc(2))


def test_bulk_all_paths(mdc, yf, capsys):
    c = mdc.MarketCache(verbose=True)
    # pre-populate fresh F (disk), stale S (incremental), unreadable U
    for t in ("F", "S"):
        df = ohlc(5)
        p = c._ohlc_path(t)
        c._write_parquet(p, df)
        c._update_meta(f"ohlc:{t}", p, df)
    _stale(c, "ohlc:S")
    c._ohlc_path("U").write_text("not parquet")
    c._update_meta("ohlc:U", c._ohlc_path("U"), ohlc(1))
    c._mem["M"] = ohlc(2)
    c._ohlc_path("M").write_bytes(b"")  # exists, in memory
    c._update_meta("ohlc:M", c._ohlc_path("M"), ohlc(1))

    def handler(batch, **k):
        names = list(batch)
        return multi_raw(names, n=2)

    yf.handler = handler
    res = c.get_ohlc_bulk(["F", "S", "U", "M", "N"])
    assert set(res) == {"F", "S", "U", "M", "N"}
    assert res["M"] is c._mem["M"]
    out = capsys.readouterr().out
    assert "Cache: 3 fresh | 1 incremental | 1 cold-start" in out
    assert "Cache result: 5/5 tickers loaded" in out
    # incremental start window = today - 5 days
    starts = sorted(str(call[1]["start"]) for call in yf.calls)
    assert str(date.today() - timedelta(days=5)) in starts
    assert len(res["S"]) >= 5


def test_bulk_force_and_incremental_fallbacks(mdc, yf):
    c = mdc.MarketCache(verbose=False)
    df = ohlc(5)
    for t in ("A", "B"):
        p = c._ohlc_path(t)
        c._write_parquet(p, df)
        c._update_meta(f"ohlc:{t}", p, df)
    # no new rows for A -> existing; B's download returns nothing too
    yf.handler = lambda *a, **k: pd.DataFrame()
    res = c.get_ohlc_bulk(["A", "B"], force=True)
    assert set(res) == {"A", "B"} and len(res["A"]) == 5


def test_bulk_incremental_exception_uses_downloaded(mdc, yf, monkeypatch):
    c = mdc.MarketCache(verbose=False)
    df = ohlc(5)
    p = c._ohlc_path("A")
    c._write_parquet(p, df)
    c._update_meta("ohlc:A", p, df)
    monkeypatch.setattr(c, "_bulk_yf_download", lambda t, start: {"A": ohlc(2)})

    def bad(d):
        raise RuntimeError("clean")

    monkeypatch.setattr(c, "_clean_ohlc", bad)
    res = c.get_ohlc_bulk(["A"], force=True)
    assert len(res["A"]) == 2
    monkeypatch.setattr(c, "_bulk_yf_download", lambda t, start: {})
    res = c.get_ohlc_bulk(["A"], force=True)
    assert len(res["A"]) == 5  # existing returned, no exception


def test_bulk_incremental_missing_file_path(mdc, yf, monkeypatch):
    c = mdc.MarketCache(verbose=False)
    monkeypatch.setattr(c, "_bulk_yf_download", lambda t, start: {})
    # path doesn't exist -> goes to needs_full; nothing downloaded -> not in result
    assert c.get_ohlc_bulk(["X"]) == {}


class FakeTicker:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def fin(cols=("2023-12-31", "2022-12-31")):
    return pd.DataFrame({c: [1.0, 2.0] for c in cols}, index=["Rev", "NI"])


def test_first_df(mdc):
    t = FakeTicker(a=None, b=pd.DataFrame(), c="notdf", d=fin())
    assert mdc._first_df(t, "a", "b", "c", "missing") is None
    assert mdc._first_df(t, "a", "d") is t.d


def test_get_financials_annual_and_cache(mdc, yf):
    t = FakeTicker(
        income_stmt=pd.DataFrame(),
        financials=fin(),
        balance_sheet=fin(),
        cashflow=fin(),
        cash_flow=None,
    )
    yf.tickers["T"] = t
    c = mdc.MarketCache(verbose=False)
    out = c.get_financials("T")
    assert set(out) == {"income_stmt", "balance_sheet", "cash_flow"}
    assert c._meta["fund:T:annual"]["sheets"] == ["income_stmt", "balance_sheet", "cash_flow"]
    # now served from parquet without touching yf
    yf.tickers.clear()
    again = c.get_financials("T")
    assert set(again) == {"income_stmt", "balance_sheet", "cash_flow"}
    assert isinstance(again["income_stmt"].columns, pd.DatetimeIndex)
    assert "_sheet" not in again["income_stmt"].columns
    assert list(again["income_stmt"].index) == ["Rev", "NI"]
    # force bypasses cache -> KeyError inside try -> {}
    assert c.get_financials("T", force=True) == {}


def test_get_financials_quarterly_and_empty(mdc, yf):
    yf.tickers["Q"] = FakeTicker(quarterly_financials=fin())
    c = mdc.MarketCache(verbose=False)
    out = c.get_financials("Q", freq="quarterly")
    assert set(out) == {"quarterly_income"}
    yf.tickers["E"] = FakeTicker()
    assert c.get_financials("E") == {}
    assert "fund:E:annual" not in c._meta


def test_get_financials_corrupt_cache_refetches(mdc, yf, capsys):
    c = mdc.MarketCache(verbose=True)
    path = mdc.FUND_DIR / "T_annual.parquet"
    path.write_text("junk")
    c._meta["fund:T:annual"] = {"updated": datetime.now().isoformat()}
    yf.tickers["T"] = FakeTicker(financials=fin())
    assert set(c.get_financials("T")) == {"income_stmt"}
    yf.tickers.clear()
    assert c.get_financials("T", force=True) == {}
    assert "Financial fetch failed for T" in capsys.readouterr().out
    q = mdc.MarketCache(verbose=False)
    assert q.get_financials("T", force=True) == {}


def test_get_financials_stale_boundary(mdc, yf):
    yf.tickers["T"] = FakeTicker(financials=fin())
    c = mdc.MarketCache(verbose=False)
    c.get_financials("T")
    yf.tickers["T"] = FakeTicker(financials=fin(), balance_sheet=fin())
    # 6 days old: still fresh (7 day threshold)
    c._meta["fund:T:annual"]["updated"] = (datetime.now() - timedelta(days=6)).isoformat()
    assert set(c.get_financials("T")) == {"income_stmt"}
    c._meta["fund:T:annual"]["updated"] = (datetime.now() - timedelta(days=8)).isoformat()
    assert set(c.get_financials("T")) == {"income_stmt", "balance_sheet"}


def idx_data(n=300):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    c = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0}, index=idx)


def test_get_index(mdc, yf, capsys):
    yf.handler = lambda *a, **k: idx_data()
    c = mdc.MarketCache(verbose=True)
    df = c.get_index("^NSEI")
    assert {"dma50", "dma200", "dma200_sl", "vol20"} <= set(df.columns)
    assert df["dma50"].iloc[49] == pytest.approx(df["Close"].iloc[:50].mean())
    assert np.isnan(df["dma200"].iloc[198]) and not np.isnan(df["dma200"].iloc[199])
    assert df["dma200_sl"].iloc[-1] == pytest.approx(5.0)
    assert (mdc.INDEX_DIR / "NSEI.parquet").exists()
    assert c._meta["index:^NSEI"]["rows"] == 300
    assert "Index cached: ^NSEI 300 bars" in capsys.readouterr().out
    assert yf.calls[0][1]["start"] == date.today() - timedelta(days=mdc.INDEX_YEARS * 365 + 30)
    n = len(yf.calls)
    again = c.get_index("^NSEI")
    assert len(yf.calls) == n and len(again) == 300
    c.get_index("^NSEI", force=True)
    assert len(yf.calls) == n + 1
    c._meta["index:^NSEI"]["updated"] = (datetime.now() - timedelta(hours=21)).isoformat()
    c.get_index("^NSEI")
    assert len(yf.calls) == n + 2


def test_get_index_empty(mdc, yf):
    yf.handler = lambda *a, **k: pd.DataFrame()
    assert mdc.MarketCache(verbose=False).get_index("^GSPC").empty


def test_report(mdc, capsys):
    c = mdc.MarketCache(verbose=False)
    assert c.report().empty
    assert "Cache is empty" in capsys.readouterr().out
    now = datetime.now()
    c._meta = {
        "ohlc:A": {
            "rows": 5,
            "from": "a",
            "to": "b",
            "updated": (now - timedelta(minutes=30)).isoformat(),
        },
        "ohlc:B": {"updated": (now - timedelta(hours=5)).isoformat()},
        "fund:C:annual": {"updated": (now - timedelta(days=3)).isoformat()},
        "index:D": {"updated": "bad"},
        "index:E": {},
    }
    df = c.report()
    assert list(df["Age"]) == ["30m", "5h", "3d", "—", "—"]
    assert list(df["Rows"]) == [5, "—", "—", "—", "—"]
    out = capsys.readouterr().out
    assert "2 OHLC | 1 fundamental | 2 index" in out
    c._meta = {"k" * 60: {}}
    assert len(c.report()["Key"].iloc[0]) == 50


def test_purge(mdc, yf, capsys):
    c = mdc.MarketCache(verbose=False)
    for t in ("AAA", "BBB"):
        df = ohlc(3)
        p = c._ohlc_path(t)
        c._write_parquet(p, df)
        c._update_meta(f"ohlc:{t}", p, df)
        for fr in ("annual", "quarterly"):
            (mdc.FUND_DIR / f"{t}_{fr}.parquet").write_text("x")
    c.purge("AAA")
    assert not c._ohlc_path("AAA").exists() and c._ohlc_path("BBB").exists()
    assert not (mdc.FUND_DIR / "AAA_annual.parquet").exists()
    assert not (mdc.FUND_DIR / "AAA_quarterly.parquet").exists()
    assert (mdc.FUND_DIR / "BBB_annual.parquet").exists()
    assert list(c._meta) == ["ohlc:BBB"]
    c.purge()
    assert c._meta == {} and not list(mdc.OHLC_DIR.glob("*.parquet"))
    assert not list(mdc.FUND_DIR.glob("*.parquet"))
    assert "Cache wiped" in capsys.readouterr().out


def test_get_cache_singleton(mdc):
    mdc._CACHE_INSTANCE = None
    a = mdc.get_cache()
    assert a is mdc.get_cache() and a.verbose is True


def test_warm_cache(mdc, monkeypatch, capsys):
    calls = []

    class FC:
        def get_index(self, s):
            calls.append(("idx", s))

        def get_ohlc_bulk(self, s):
            calls.append(("bulk", s))

        def report(self):
            calls.append(("report",))

    monkeypatch.setattr(mdc, "get_cache", lambda: FC())
    mdc.warm_cache(["A"] * 250)
    assert calls[:2] == [("idx", "^NSEI"), ("idx", "^GSPC")]
    assert calls[2][0] == "bulk" and calls[3] == ("report",)
    out = capsys.readouterr().out
    assert "Warming cache: 250 stocks, 5-year window" in out
    assert "~6 minutes" in out and "Cache warm" in out
    calls.clear()
    mdc.warm_cache(["A"], index_tickers=["^X"])
    assert calls[0] == ("idx", "^X") and calls[1][0] == "bulk"
