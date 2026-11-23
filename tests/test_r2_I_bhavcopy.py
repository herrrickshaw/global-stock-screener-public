import datetime as dt
import json
import runpy
import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests._r2i import fresh


def _ohlcv(n=5, start="2026-01-05", base=10.0):
    idx = pd.date_range(start, periods=n, freq="D")
    c = base + np.arange(n, dtype=float)
    return pd.DataFrame(
        {"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": np.arange(n) + 100},
        index=idx,
    )


# ---------------- bhavcopy_store ----------------
@pytest.fixture
def st(monkeypatch, tmp_path):
    m = fresh(monkeypatch, tmp_path, "bhavcopy_store", {"BHAV_CACHE": tmp_path / "bc"})
    m.CACHE.mkdir(parents=True, exist_ok=True)
    return m


def test_ser_de_roundtrip_types(st):
    df = _ohlcv()
    df.index.name = "Date"
    df["Volume"] = ["1", "2", "x", None, "5"]
    out = st._de(st._ser(df))
    assert list(out["Volume"]) == [1, 2, 0, 0, 5]  # coerce + fillna(0)
    assert out["Volume"].dtype == "int64"
    assert out["Close"].dtype == "float64"
    assert out.index.is_monotonic_increasing
    # float32 downcast loses precision beyond ~7 digits
    df2 = _ohlcv()
    df2.index.name = "Date"
    df2["Close"] = 1.123456789012
    assert out is not None and st._de(st._ser(df2))["Close"].iloc[0] == pytest.approx(
        1.1234568, abs=1e-6
    )
    assert st._de(st._ser(df2))["Close"].iloc[0] != 1.123456789012


def test_de_without_date_column_uses_first(st):
    import io

    import pyarrow as pa
    import pyarrow.feather as feather

    t = pa.Table.from_pandas(pd.DataFrame({"when": [3, 1, 2], "Close": [30.0, 10.0, 20.0]}))
    buf = io.BytesIO()
    feather.write_feather(t, buf, compression="zstd")
    out = st._de(buf.getvalue())
    assert list(out.index) == [1, 2, 3] and list(out["Close"]) == [10.0, 20.0, 30.0]


def test_readers_when_no_store(st):
    assert st.get("A") is None
    assert st.get_many(["A"]) == {}
    assert st.load_all() == {}
    assert st.symbols() == []
    assert st.info() == {}


def test_build_from_dict_and_reads(st, capsys):
    a, b = _ohlcv(3), _ohlcv(4, "2026-01-07")
    a.index.name = b.index.name = "Date"
    n = st.build({"A": a, "B": b, "E": pd.DataFrame(), "N": None})
    assert n == 2
    assert "LMDB store built: 2 symbols" in capsys.readouterr().out
    assert sorted(st.symbols()) == ["A", "B"]
    assert len(st.get("A")) == 3 and st.get("ZZ") is None
    gm = st.get_many(["A", "B", "ZZ"])
    assert set(gm) == {"A", "B"}
    assert set(st.load_all()) == {"A", "B"}  # meta key excluded
    i = st.info()
    assert i["symbols"] == 2 and i["latest_bar"].startswith("2026-01-10")
    # rebuild replaces rather than appends
    st.build({"C": a}, verbose=False)
    assert st.symbols() == ["C"]


def test_build_latest_bar_is_max_not_last(st, capsys):
    early, late = _ohlcv(2, "2026-01-01"), _ohlcv(2, "2026-03-01")
    early.index.name = late.index.name = "Date"
    st.build({"L": late, "E": early}, verbose=True)
    assert "latest bar 2026-03-02" in capsys.readouterr().out
    assert st.info()["latest_bar"].startswith("2026-03-02")


def test_build_empty_hist_reports_unknown_bar(st, capsys):
    assert st.build({}, verbose=True) == 0
    assert "latest bar ?" in capsys.readouterr().out
    assert st.info()["symbols"] == 0 and st.info()["latest_bar"] == "None"


def test_build_from_seeds(st, capsys):
    st.CACHE.mkdir(parents=True, exist_ok=True)
    from frames import to_long

    to_long({"IN1": _ohlcv(3)}).to_parquet(st.CLEANED)
    to_long({"US1": _ohlcv(3)}).to_parquet(st.CACHE / "cleaned_long_US.parquet")
    n = st.build()
    assert n == 2 and sorted(st.symbols()) == ["IN1", "US1"]
    out = capsys.readouterr().out
    assert "ingesting 2 market seed(s): cleaned_long.parquet, cleaned_long_US.parquet" in out


def test_build_no_seed_raises(st):
    st.CACHE.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="no cleaned_long"):
        st.build()


def test_vacuum(st, capsys):
    (st.CACHE / "nse").mkdir(parents=True)
    (st.CACHE / "nse" / "a.csv").write_text("x" * 10)
    (st.CACHE / "nse" / "keep.txt").write_text("y")
    (st.CACHE / "bse").mkdir()
    (st.CACHE / "bse" / "b.csv").write_text("z" * 5)
    assert st.vacuum() == 15
    assert not (st.CACHE / "nse" / "a.csv").exists() and (st.CACHE / "nse" / "keep.txt").exists()
    assert "reclaimed 0.0 MB" in capsys.readouterr().out
    assert st.vacuum(verbose=False) == 0
    assert capsys.readouterr().out == ""


def test_vacuum_no_dirs(st):
    assert st.vacuum(verbose=False) == 0


def test_store_main_guard(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "bc2"))
    monkeypatch.setattr(sys, "argv", ["bhavcopy_store.py", "--vacuum"])
    runpy.run_path(
        str(__import__("pathlib").Path(__file__).resolve().parent.parent / "bhavcopy_store.py"),
        run_name="__main__",
    )
    assert "store info: {}" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["bhavcopy_store.py", "--build"])
    with pytest.raises(FileNotFoundError):
        runpy.run_path(
            str(__import__("pathlib").Path(__file__).resolve().parent.parent / "bhavcopy_store.py"),
            run_name="__main__",
        )


# ---------------- bhavcopy_history ----------------
@pytest.fixture
def bh(monkeypatch, tmp_path):
    return fresh(monkeypatch, tmp_path, "bhavcopy_history", {"BHAV_CACHE": tmp_path / "bc"})


def test_import_without_stock_utils(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "stock_utils", None)
    m = fresh(monkeypatch, tmp_path, "bhavcopy_history", {"BHAV_CACHE": tmp_path / "bc"})
    assert m.clean_ohlcv is None


NSE_CSV = "TckrSymb, SctySrs ,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\nAAA,EQ,1,2,0.5,1.5,100\nBBB,BE,1,2,0.5,1.5,100\n"


class _Nse:
    def __init__(self, path=None, exc=None):
        self.path, self.exc, self.dates = path, exc, []

    def equityBhavcopy(self, date):
        self.dates.append(date)
        if self.exc:
            raise self.exc
        return self.path


def test_nse_day_downloads_filters_caches_and_unlinks(bh, tmp_path):
    p = tmp_path / "dl.csv"
    p.write_text(NSE_CSV)
    cli = _Nse(p)
    df = bh._nse_day(dt.date(2026, 1, 5), cli)
    assert list(df["TckrSymb"]) == ["AAA"]
    assert cli.dates == [dt.datetime(2026, 1, 5)]
    assert not p.exists()
    cached = bh.NSE_DIR / "20260105.csv"
    assert cached.exists()
    # second call served from cache: client must not be hit
    assert list(bh._nse_day(dt.date(2026, 1, 5), _Nse(exc=RuntimeError()))["TckrSymb"]) == ["AAA"]


def test_nse_day_unlink_failure_ignored_and_errors(bh, tmp_path, monkeypatch):
    p = tmp_path / "dl.csv"
    p.write_text(NSE_CSV)

    def bad_unlink(self, *a, **k):
        raise OSError("nope")

    monkeypatch.setattr(type(p), "unlink", bad_unlink)
    assert len(bh._nse_day(dt.date(2026, 1, 6), _Nse(p))) == 1
    assert bh._nse_day(dt.date(2026, 1, 7), _Nse(exc=RuntimeError("x"))) is None
    assert not (bh.NSE_DIR / "20260107.csv").exists()


class _Resp:
    def __init__(self, status=200, content=b""):
        self.status_code, self.content = status, content


class _Sess:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.urls = resp, exc, []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        if self.exc:
            raise self.exc
        return self.resp


BSE_CSV = (
    "FinInstrmTp, TckrSymb,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol\n"
    + "STK,500325,1,2,0.5,1.5,100\n" * 1
    + "OTH,x,1,2,0.5,1.5,1\n"
    + "#" * 1000
).encode()


def test_bse_day(bh):
    sess = _Sess(_Resp(200, BSE_CSV))
    df = bh._bse_day(dt.date(2026, 1, 5), sess)
    assert sess.urls == [
        (
            "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_20260105_F_0000.CSV",
            25,
        )
    ]
    assert list(df["TckrSymb"]) == ["500325"]
    assert (bh.BSE_DIR / "20260105.csv").exists()
    # cached → no session use
    assert len(bh._bse_day(dt.date(2026, 1, 5), _Sess(exc=RuntimeError()))) == 1


def test_bse_day_failures(bh):
    assert bh._bse_day(dt.date(2026, 1, 6), _Sess(_Resp(404, BSE_CSV))) is None
    assert bh._bse_day(dt.date(2026, 1, 6), _Sess(_Resp(200, b"x" * 999))) is None
    assert bh._bse_day(dt.date(2026, 1, 6), _Sess(exc=RuntimeError("net"))) is None
    # parseable body lacking FinInstrmTp: the filter raises inside the try -> swallowed, not cached
    assert bh._bse_day(dt.date(2026, 1, 6), _Sess(_Resp(200, b"a,b\n" + b"1,2\n" * 250))) is None
    assert not (bh.BSE_DIR / "20260106.csv").exists()


def test_bse_day_size_gate_boundary(bh):
    head = b"FinInstrmTp,TckrSymb\nSTK,A\n"
    ok = head + b"#" * (1000 - len(head))
    assert len(ok) == 1000 and len(bh._bse_day(dt.date(2026, 1, 8), _Sess(_Resp(200, ok)))) >= 1
    short = head + b"#" * (999 - len(head))
    assert bh._bse_day(dt.date(2026, 1, 9), _Sess(_Resp(200, short))) is None


def test_trading_dates(bh, monkeypatch):
    today = dt.date(2026, 1, 12)  # Monday
    fake = types.ModuleType("market_calendar")
    fake.trading_days = lambda cal: [d for d in cal if d.day % 2 == 0]
    monkeypatch.setitem(sys.modules, "market_calendar", fake)
    got = bh._trading_dates(4, today)
    assert got == {pd.Timestamp(2026, 1, 12), pd.Timestamp(2026, 1, 10)}
    # calendar failure -> weekday fallback (Mon 12, Sun 11, Sat 10, Fri 9)
    fake.trading_days = lambda cal: 1 / 0
    assert bh._trading_dates(4, today) == {pd.Timestamp(2026, 1, 12), pd.Timestamp(2026, 1, 9)}
    assert bh._trading_dates(0, today) == set()


def test_load_no_data(bh):
    assert bh._load_no_data() == set()
    bh.NODATA_FILE.write_text(json.dumps(["2026-01-01", "2026-01-02"]))
    assert bh._load_no_data() == {pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")}
    bh.NODATA_FILE.write_text("garbage")
    assert bh._load_no_data() == set()


def _long(dates, syms=("A",), exch="NSE", base=10.0):
    rows = []
    for s in syms:
        for i, d in enumerate(dates):
            c = base + i
            rows.append(
                {
                    "Date": pd.Timestamp(d),
                    "Symbol": s,
                    "Open": c,
                    "High": c + 1,
                    "Low": c - 1,
                    "Close": c,
                    "Volume": 100,
                    "_exch": exch,
                }
            )
    return pd.DataFrame(rows)


def test_from_cleaned_filters_dates_and_min_bars(bh):
    cl = _long(["2026-01-05", "2026-01-06", "2026-01-07"]).drop(columns="_exch")
    cl.to_parquet(bh.CLEANED)
    out = bh._from_cleaned({pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")}, 2)
    assert list(out) == ["A"] and len(out["A"]) == 2
    assert bh._from_cleaned({pd.Timestamp("2026-01-05")}, 2) == {}


def test_load_assembled(bh, capsys):
    assert bh._load_assembled(True).empty
    cl = _long(["2026-01-05"]).drop(columns="_exch")
    cl.to_parquet(bh.CLEANED)
    got = bh._load_assembled(True)
    assert (got["_exch"] == "NSE").all() and bh.ASSEMBLED.exists()
    assert "seeded assembled cache" in capsys.readouterr().out
    # existing assembled wins; corrupt assembled -> empty
    bh.ASSEMBLED.write_text("junk")
    assert bh._load_assembled(True).empty
    # corrupt cleaned -> empty
    bh.ASSEMBLED.unlink()
    bh.CLEANED.write_text("junk")
    assert bh._load_assembled(False).empty


def test_load_assembled_existing(bh):
    _long(["2026-01-05"]).to_parquet(bh.ASSEMBLED)
    got = bh._load_assembled(False)
    assert len(got) == 1 and str(got["Date"].dtype).startswith("datetime64")


def _fake_nse_pkg(monkeypatch, client):
    m = types.ModuleType("nse")
    m.NSE = lambda download_folder: client
    monkeypatch.setitem(sys.modules, "nse", m)


def test_fetch_nse_days(bh, monkeypatch, capsys):
    calls = []

    class C:
        def exit(self):
            calls.append("exit")
            raise RuntimeError("ignored")

    _fake_nse_pkg(monkeypatch, C())
    days = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-07")]
    monkeypatch.setattr(
        bh,
        "_nse_day",
        lambda d, c: {
            dt.date(2026, 1, 5): pd.DataFrame({"x": [1]}),
            dt.date(2026, 1, 6): pd.DataFrame(),
        }.get(d),
    )
    fr = bh._fetch_nse_days(days, True)
    assert [(e, d) for e, d, _ in fr] == [("NSE", dt.date(2026, 1, 5))]
    assert calls == ["exit"] and "NSE: +1 new trading days" in capsys.readouterr().out
    assert bh._fetch_nse_days([], False) == []


def test_fetch_bse_days(bh, monkeypatch, capsys):
    monkeypatch.setattr(
        bh,
        "_bse_day",
        lambda d, s: (
            pd.DataFrame({"x": [1]}) if d.day == 5 else (pd.DataFrame() if d.day == 6 else None)
        ),
    )
    days = [pd.Timestamp("2026-01-0%d" % i) for i in (5, 6, 7)]
    fr = bh._fetch_bse_days(days, 2, True)
    assert [(e, d) for e, d, _ in fr] == [("BSE", dt.date(2026, 1, 5))]
    assert "BSE: +1 new trading days" in capsys.readouterr().out
    assert bh._fetch_bse_days(days, 2, False) and capsys.readouterr().out == ""


def test_frames_to_long(bh):
    df = pd.DataFrame(
        {
            "TckrSymb": [1, 2],
            "OpnPric": [1, 2],
            "HghPric": [2, 3],
            "LwPric": [0, 1],
            "ClsPric": [1.5, 2.5],
            "TtlTradgVol": [10, 20],
            "Z": 0,
        }
    )
    (out,) = bh._frames_to_long([("BSE", dt.date(2026, 1, 5), df)])
    assert list(out.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Symbol",
        "Date",
        "_exch",
    ]
    assert list(out["Symbol"]) == ["1", "2"] and (out["_exch"] == "BSE").all()
    assert (out["Date"] == pd.Timestamp("2026-01-05")).all()
    assert bh._frames_to_long([]) == []


def test_record_dead_dates(bh):
    today = dt.date(2026, 1, 12)
    missing = [pd.Timestamp(2026, 1, d) for d in (8, 9, 12)]
    frames = [("NSE", dt.date(2026, 1, 8), None)]
    nd = {pd.Timestamp("2025-12-25")}
    bh._record_dead_dates(missing, frames, nd, today)
    assert nd == {
        pd.Timestamp("2025-12-25"),
        pd.Timestamp("2026-01-09"),
    }  # today excluded, got-date excluded
    assert json.loads(bh.NODATA_FILE.read_text()) == ["2025-12-25", "2026-01-09"]
    # nothing dead → nothing written
    bh.NODATA_FILE.unlink()
    bh._record_dead_dates([pd.Timestamp(2026, 1, 12)], [], set(), today)
    assert not bh.NODATA_FILE.exists()


def test_record_dead_dates_write_failure(bh, monkeypatch):
    monkeypatch.setattr(bh, "NODATA_FILE", bh.CACHE / "no" / "dir" / "f.json")
    nd = set()
    bh._record_dead_dates([pd.Timestamp(2026, 1, 8)], [], nd, dt.date(2026, 1, 12))
    assert nd == {pd.Timestamp(2026, 1, 8)}


def test_merge_and_persist(bh, capsys):
    assert bh._merge_and_persist(pd.DataFrame(), [], True).empty
    assert bh._merge_and_persist(None, [pd.DataFrame()], True).empty
    old = _long(["2026-01-05"], base=10)
    new = _long(["2026-01-05", "2026-01-06"], base=50)
    out = bh._merge_and_persist(old, [new], True)
    assert len(out) == 2
    assert out[out["Date"] == pd.Timestamp("2026-01-05")]["Close"].iloc[0] == 50  # last wins
    assert bh.ASSEMBLED.exists() and "assembled cache updated" in capsys.readouterr().out
    # cache only: no persist
    bh.ASSEMBLED.unlink()
    assert len(bh._merge_and_persist(old, [], True)) == 1 and not bh.ASSEMBLED.exists()


def test_merge_persist_failure(bh, monkeypatch, capsys):
    monkeypatch.setattr(bh, "ASSEMBLED", bh.CACHE / "no" / "dir" / "a.parquet")
    out = bh._merge_and_persist(pd.DataFrame(), [_long(["2026-01-05"])], True)
    assert len(out) == 1 and "could not persist assembled cache" in capsys.readouterr().out
    bh._merge_and_persist(pd.DataFrame(), [_long(["2026-01-05"])], False)
    assert capsys.readouterr().out == ""


def test_pivot_and_clean_nse_precedence(bh, monkeypatch, capsys):
    monkeypatch.setattr(bh, "clean_ohlcv", None)
    rows = pd.concat(
        [
            _long(["2026-01-05", "2026-01-06"], ("A", "B"), "BSE", base=1),
            _long(["2026-01-05", "2026-01-06"], ("A",), "NSE", base=100),
        ]
    )
    out = bh._pivot_and_clean(rows, True)
    assert set(out) == {"A", "B"}
    assert out["A"]["Close"].iloc[0] == 100  # NSE processed last overwrites
    assert out["B"]["Close"].iloc[0] == 1
    assert "Pivoting 2 dates" in capsys.readouterr().out
    bh._pivot_and_clean(rows, False)
    assert capsys.readouterr().out == ""


def test_pivot_uses_cleaner_and_drops_empty(bh, monkeypatch):
    seen = []

    def clean(g, ticker, min_bars):
        seen.append((ticker, min_bars))
        return None if ticker == "B" else g

    monkeypatch.setattr(bh, "clean_ohlcv", clean)
    rows = _long(["2026-01-05"], ("A", "B"))
    out = bh._pivot_and_clean(rows, False)
    assert set(out) == {"A"} and sorted(seen) == [("A", 1), ("B", 1)]


def test_persist_cleaned(bh, capsys):
    bh._persist_cleaned({}, True)
    assert not bh.CLEANED.exists()
    g = _ohlcv(2)
    bh._persist_cleaned({"A": g}, True)
    got = pd.read_parquet(bh.CLEANED)
    assert list(got.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol"]
    assert "cleaned cache written" in capsys.readouterr().out


def test_persist_cleaned_failure(bh, monkeypatch, capsys):
    monkeypatch.setattr(bh, "CLEANED", bh.CACHE / "no" / "dir" / "c.parquet")
    bh._persist_cleaned({"A": _ohlcv(2)}, True)
    assert "could not persist cleaned cache" in capsys.readouterr().out
    bh._persist_cleaned({"A": _ohlcv(2)}, False)
    assert capsys.readouterr().out == ""


def test_sync_lmdb(bh, monkeypatch, capsys):
    import bhavcopy_store

    seen = {}
    monkeypatch.setattr(bhavcopy_store, "build", lambda h, verbose: seen.update(h=h, v=verbose))
    bh._sync_lmdb({"A": 1}, False)
    assert seen == {"h": {"A": 1}, "v": False}
    monkeypatch.setattr(bhavcopy_store, "build", lambda h, verbose: 1 / 0)
    bh._sync_lmdb({}, True)
    assert "LMDB store not synced" in capsys.readouterr().out
    bh._sync_lmdb({}, False)
    assert capsys.readouterr().out == ""


# --- fetch_history orchestration
TODAY = dt.date.today()


def _days(n):
    return [pd.Timestamp(TODAY - dt.timedelta(days=i)) for i in range(n)]


def test_fetch_history_fast_path(bh, monkeypatch, capsys):
    ds = _days(3)
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(ds))
    cl = _long(ds, ("A",)).drop(columns="_exch")
    cl.to_parquet(bh.CLEANED)
    _long(ds).to_parquet(bh.ASSEMBLED)
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: pytest.fail("no fetch"))
    out = bh.fetch_history(n_days=3, min_bars=3)
    assert list(out) == ["A"]
    assert "fast path (cleaned cache): 1 symbols" in capsys.readouterr().out
    assert bh.fetch_history(n_days=3, min_bars=4, verbose=False) == {}


def test_fetch_history_fast_path_unusable_cleaned(bh, monkeypatch, capsys):
    ds = _days(2)
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(ds))
    bh.CLEANED.write_text("junk")
    _long(ds).to_parquet(bh.ASSEMBLED)
    monkeypatch.setattr(bh, "clean_ohlcv", None)
    monkeypatch.setattr(bh, "_sync_lmdb", lambda *a: None)
    out = bh.fetch_history(n_days=2, min_bars=2)
    assert "cleaned cache unusable, rebuilding" in capsys.readouterr().out
    assert list(out) == ["A"]


def test_fetch_history_full_pipeline(bh, monkeypatch, capsys):
    ds = _days(3)
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(ds))
    got = {}
    nse_frames = [
        (
            "NSE",
            d.date(),
            pd.DataFrame(
                {
                    "TckrSymb": ["A"],
                    "OpnPric": [1],
                    "HghPric": [2],
                    "LwPric": [0],
                    "ClsPric": [1.5],
                    "TtlTradgVol": [9],
                }
            ),
        )
        for d in ds[:2]
    ]
    monkeypatch.setattr(
        bh, "_fetch_nse_days", lambda m, v: got.setdefault("nse", list(m)) and nse_frames
    )
    monkeypatch.setattr(
        bh, "_fetch_bse_days", lambda m, w, v: got.setdefault("bse", (list(m), w)) and []
    )
    synced = {}
    monkeypatch.setattr(bh, "_sync_lmdb", lambda o, v: synced.update(o))
    monkeypatch.setattr(bh, "clean_ohlcv", None)
    out = bh.fetch_history(n_days=3, workers=3, min_bars=2)
    assert list(out) == ["A"] and len(out["A"]) == 2
    assert sorted(got["nse"]) == sorted(ds) and got["bse"][1] == 3
    assert "A" in synced and bh.CLEANED.exists() and bh.ASSEMBLED.exists()
    # the third date produced no data: it was today (published later) -> may be dead only if not today
    nd = json.loads(bh.NODATA_FILE.read_text()) if bh.NODATA_FILE.exists() else []
    assert all(d != str(TODAY) for d in nd)
    assert "fetch_history complete: 1 symbols with ≥2 bars" in capsys.readouterr().out


def test_fetch_history_exchange_filter(bh, monkeypatch):
    ds = _days(2)
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(ds))
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: pytest.fail("nse"))
    called = []
    monkeypatch.setattr(bh, "_fetch_bse_days", lambda *a: called.append(1) or [])
    assert bh.fetch_history(n_days=2, exchanges=("BSE",), verbose=False) == {}
    assert called == [1]
    called.clear()
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: called.append("n") or [])
    monkeypatch.setattr(bh, "_fetch_bse_days", lambda *a: pytest.fail("bse"))
    bh.fetch_history(n_days=2, exchanges=("NSE",), verbose=False)
    assert called == ["n"]


def test_fetch_history_post_fetch_fast_path(bh, monkeypatch, capsys):
    ds = _days(3)
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(ds))
    _long(ds[1:]).to_parquet(bh.ASSEMBLED)  # missing: ds[0] == today only
    _long(ds).drop(columns="_exch").to_parquet(bh.CLEANED)
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: [])
    monkeypatch.setattr(bh, "_fetch_bse_days", lambda *a: [])
    out = bh.fetch_history(n_days=3, min_bars=3)
    assert len(out["A"]) == 3
    assert "fast path (no new EOD yet)" in capsys.readouterr().out
    assert not bh.NODATA_FILE.exists()  # today is never recorded as dead


def test_fetch_history_post_fetch_fast_path_unusable(bh, monkeypatch):
    ds = _days(2)
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(ds))
    bh.CLEANED.write_text("junk")
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: [])
    monkeypatch.setattr(bh, "_fetch_bse_days", lambda *a: [])
    # cleaned unusable, nothing cached, nothing fetched -> empty
    assert bh.fetch_history(n_days=2, verbose=False) == {}


def test_fetch_history_empty_everything(bh, monkeypatch):
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: set(_days(2)))
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: [])
    monkeypatch.setattr(bh, "_fetch_bse_days", lambda *a: [])
    assert bh.fetch_history(n_days=2) == {}


def test_fetch_history_dead_date_skipped_next_time(bh, monkeypatch):
    old = dt.date.today() - dt.timedelta(days=10)
    ds = {pd.Timestamp(old)}
    monkeypatch.setattr(bh, "_trading_dates", lambda n, t: ds)
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda m, v: [])
    monkeypatch.setattr(bh, "_fetch_bse_days", lambda m, w, v: [])
    bh.fetch_history(n_days=11, verbose=False)
    assert json.loads(bh.NODATA_FILE.read_text()) == [str(old)]
    monkeypatch.setattr(bh, "_fetch_nse_days", lambda *a: pytest.fail("dead date refetched"))
    bh.fetch_history(n_days=11, verbose=False)


# --- get_symbol
def test_get_symbol(bh, monkeypatch):
    assert bh.get_symbol("A") is None
    today = pd.Timestamp(dt.date.today())
    rows = pd.concat(
        [
            _long([today - pd.Timedelta(days=1), today], ("A", "B"), "BSE", base=1),
            _long([today - pd.Timedelta(days=1)], ("A",), "NSE", base=100),
            _long([today - pd.Timedelta(days=500)], ("A",), "NSE", base=9),
        ]
    )
    rows.to_parquet(bh.ASSEMBLED)
    monkeypatch.setattr(bh, "clean_ohlcv", None)
    g = bh.get_symbol("A", n_days=400)
    assert len(g) == 2  # old row cut off; duplicate date deduped
    assert g["Close"].iloc[0] == 100  # NSE wins the collision
    assert bh.get_symbol("ZZ") is None
    assert len(bh.get_symbol("A", n_days=600)) == 3
    monkeypatch.setattr(bh, "clean_ohlcv", lambda g, ticker, min_bars: g.assign(T=ticker))
    assert (bh.get_symbol("B")["T"] == "B").all()


def test_history_main_guard(monkeypatch, tmp_path, capsys):
    import pathlib

    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "bc"))
    monkeypatch.setattr(sys, "argv", ["bhavcopy_history.py", "1"])
    # network is absent: stub the day-fetchers' deps so run is a no-op
    fake = types.ModuleType("nse")

    class C:
        def __init__(self, download_folder):
            pass

        def equityBhavcopy(self, date):
            raise RuntimeError

        def exit(self):
            pass

    fake.NSE = C
    monkeypatch.setitem(sys.modules, "nse", fake)
    import requests

    def boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(requests.Session, "get", boom)
    runpy.run_path(
        str(pathlib.Path(__file__).resolve().parent.parent / "bhavcopy_history.py"),
        run_name="__main__",
    )
    assert "Symbols: 0" in capsys.readouterr().out
