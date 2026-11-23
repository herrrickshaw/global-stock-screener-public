"""Offline tests for symbol_master (HOME redirected; fake nse/bse/requests)."""

import importlib
import os
import sys
import time
import types

import pandas as pd
import pytest


@pytest.fixture
def sm(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    saved = sys.modules.pop("symbol_master", None)
    mod = importlib.import_module("symbol_master")
    yield mod
    sys.modules.pop("symbol_master", None)
    if saved is not None:
        sys.modules["symbol_master"] = saved


def test_master_path_under_home(sm, tmp_path):
    assert sm.MASTER == tmp_path / "home" / "Downloads" / "market_cache" / "symbol_master.parquet"
    assert sm.MASTER.parent.is_dir()


@pytest.mark.parametrize(
    "name,exp",
    [
        ("ADANI ENTERPRISES LIMITED", "ADANI ENTERPRISES"),
        ("MARUTI SUZUKI INDIA LTD.", "MARUTI SUZUKI INDIA"),
        ("Amara Raja Energy & Mobility L", "AMARA RAJA ENERGY"),
        ("The Tata Power Co Ltd", "TATA POWER"),
        ("A B Corp", ""),  # tokens shorter than 3 chars dropped
        ("ABC Inc", "ABC"),
        ("Foo123 Bar", "FOO BAR"),
        ("LIMITED", ""),
        ("", ""),
        (None, ""),
        (123, ""),
        ("one two three four five", "ONE TWO THREE"),
    ],
)
def test_clean_name(sm, name, exp):
    assert sm.clean_name(name) == exp


class FakePath:
    def __init__(self, exists=True, csv=None):
        self._e, self.csv = exists, csv

    def exists(self):
        return self._e


def install_nse(monkeypatch, behaviour, seen=None):
    class NSE:
        def __init__(self, download_folder, server):
            if seen is not None:
                seen.append((download_folder, server))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def equityBhavcopy(self, d):  # noqa: N802
            return behaviour(d)

    mod = types.ModuleType("nse")
    mod.NSE = NSE
    monkeypatch.setitem(sys.modules, "nse", mod)


def test_nse_names(sm, monkeypatch, tmp_path):
    csvp = tmp_path / "bh.csv"
    pd.DataFrame(
        {
            "SctySrs": ["EQ", "BE", "EQ"],
            "TckrSymb": [" TCS ", "XYZ", "INFY"],
            "FinInstrmNm": ["TATA CONSULTANCY", "x", None],
        }
    ).to_csv(csvp, index=False)
    calls = []
    seen = []

    class RealPath:
        def exists(self):
            return True

        def __fspath__(self):
            return str(csvp)

    def beh2(d):
        calls.append(d)
        n = len(calls)
        if n == 1:
            raise RuntimeError("holiday")
        if n == 2:
            return object()
        if n == 3:
            return FakePath(False)
        return RealPath()

    install_nse(monkeypatch, beh2, seen)
    rows = sm._nse_names()
    assert len(calls) == 4  # stops at first success
    assert seen == [("/tmp", False)]
    assert rows == [
        {"symbol": "TCS", "name": "TATA CONSULTANCY", "exchange": "NSE", "suffix": ".NS"},
        {"symbol": "INFY", "name": "nan", "exchange": "NSE", "suffix": ".NS"},
    ]
    # first attempt is today, then going back a day each time
    assert calls[0].date() > calls[1].date() > calls[2].date()


def test_nse_names_all_fail_and_importerror(sm, monkeypatch):
    calls = []
    install_nse(monkeypatch, lambda d: calls.append(d) or FakePath(False))
    assert sm._nse_names() == []
    assert len(calls) == 7
    monkeypatch.setitem(sys.modules, "nse", None)
    assert sm._nse_names() == []


def install_bse(monkeypatch, behaviour):
    class BSE:
        def __init__(self, download_folder):
            self.folder = download_folder

        def bhavcopyReport(self, d):  # noqa: N802
            return behaviour(d)

    mod = types.ModuleType("bse")
    mod.BSE = BSE
    monkeypatch.setitem(sys.modules, "bse", mod)


class Existing:
    def __init__(self, p):
        self.p = p

    def exists(self):
        return True

    def __fspath__(self):
        return str(self.p)


def test_bse_names(sm, monkeypatch, tmp_path):
    csvp = tmp_path / "b.csv"
    pd.DataFrame(
        {
            "SctySrs": ["A", "Z", "B", "XT"],
            "TckrSymb": ["500001", "500002", "TCS", "500003"],
            "FinInstrmNm": ["Foo Ltd", "Bar", "Dual", "Baz Ltd"],
        }
    ).to_csv(csvp, index=False)
    calls = []

    def beh(d):
        calls.append(d)
        if len(calls) == 1:
            raise RuntimeError("x")
        return Existing(csvp)

    install_bse(monkeypatch, beh)
    rows = sm._bse_names({"TCS"})
    assert [r["symbol"] for r in rows] == ["500001", "500003"]  # Z filtered, dual-listed skipped
    assert all(r["exchange"] == "BSE" and r["suffix"] == ".BO" for r in rows)
    assert len(calls) == 2


def test_bse_names_no_series_column_and_failures(sm, monkeypatch, tmp_path):
    csvp = tmp_path / "b.csv"
    pd.DataFrame({"TckrSymb": ["1", "2"], "FinInstrmNm": ["a", "b"]}).to_csv(csvp, index=False)
    install_bse(monkeypatch, lambda d: Existing(csvp))
    assert len(sm._bse_names(set())) == 2
    calls = []
    install_bse(monkeypatch, lambda d: calls.append(d) or object())
    assert sm._bse_names(set()) == [] and len(calls) == 7
    monkeypatch.setitem(sys.modules, "bse", None)
    assert sm._bse_names(set()) == []


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_us_names(sm, monkeypatch):
    data = {
        "data": [
            [1, "Apple Inc", "aapl", "Nasdaq"],
            [2, "Ford", "F", "NYSE"],
            [3, "Amex Co", "AMX", "NYSE American"],
            [4, "Too Long", "ABCDEF", "Nasdaq"],
            [5, "Dotted", "BRK.B", "NYSE"],
            [6, "Spaced", "A B", "NYSE"],
            [7, "Comma", "A,B", "NYSE"],
            [8, "OTC", "OTCX", "OTC"],
            [9, "Empty", "", "Nasdaq"],
            [10, "Short"],
            [11, "Five", "ABCDE", "NASDAQ"],
        ]
    }
    seen = {}

    def get(url, timeout, headers):
        seen.update(url=url, timeout=timeout, headers=headers)
        return FakeResp(data)

    req = types.ModuleType("requests")
    req.get = get
    monkeypatch.setitem(sys.modules, "requests", req)
    rows = sm._us_names()
    assert [(r["symbol"], r["exchange"]) for r in rows] == [
        ("AAPL", "NASDAQ"),
        ("F", "NYSE"),
        ("AMX", "NYSE"),
        ("ABCDE", "NASDAQ"),
    ]
    assert all(r["suffix"] == "" for r in rows)
    assert seen["url"].endswith("company_tickers_exchange.json") and seen["timeout"] == 30
    # network failure -> empty
    req.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net"))
    assert sm._us_names() == []
    req.get = lambda *a, **k: FakeResp({})
    assert sm._us_names() == []


def patch_builders(sm, monkeypatch):
    monkeypatch.setattr(
        sm,
        "_nse_names",
        lambda: [
            {
                "symbol": "TCS",
                "name": "TATA CONSULTANCY SERVICES LIMITED",
                "exchange": "NSE",
                "suffix": ".NS",
            }
        ],
    )
    seen = {}
    monkeypatch.setattr(
        sm,
        "_bse_names",
        lambda s: seen.setdefault("nse", s)
        and [{"symbol": "500", "name": "FOO LTD", "exchange": "BSE", "suffix": ".BO"}],
    )
    monkeypatch.setattr(
        sm,
        "_us_names",
        lambda: [
            {"symbol": "TCS", "name": "Tcs Inc", "exchange": "NASDAQ", "suffix": ""},
            {"symbol": "TCS", "name": "dup", "exchange": "NASDAQ", "suffix": ""},
        ],
    )
    return seen


def test_build_master(sm, monkeypatch, capsys):
    seen = patch_builders(sm, monkeypatch)
    df = sm.build_master()
    assert seen["nse"] == {"TCS"}
    assert list(df.columns) == ["symbol", "name", "name_clean", "exchange", "suffix", "yf_symbol"]
    assert len(df) == 3  # (TCS,NASDAQ) duplicate dropped; (TCS,NSE) kept
    assert set(df["yf_symbol"]) == {"TCS.NS", "500.BO", "TCS"}
    assert df.set_index("yf_symbol").loc["TCS.NS", "name_clean"] == "TATA CONSULTANCY SERVICES"
    assert sm.MASTER.exists() and len(pd.read_parquet(sm.MASTER)) == 3
    out = capsys.readouterr().out
    assert "NSE: 1 | BSE-only: 1 | US: 2" in out
    assert "Symbol master saved: 3 stocks" in out
    assert "(3 have a usable company-name root)" in out


def test_build_master_no_us_and_empty(sm, monkeypatch, capsys):
    patch_builders(sm, monkeypatch)
    monkeypatch.setattr(sm, "_us_names", lambda: pytest.fail("US should be skipped"))
    df = sm.build_master(include_us=False)
    assert len(df) == 2
    for n in ("_nse_names", "_us_names"):
        monkeypatch.setattr(sm, n, lambda: [])
    monkeypatch.setattr(sm, "_bse_names", lambda s: [])
    capsys.readouterr()
    assert sm.build_master().empty
    assert "No symbols fetched" in capsys.readouterr().out


def test_load_master_paths(sm, monkeypatch):
    built = []
    monkeypatch.setattr(
        sm, "build_master", lambda: built.append(1) or pd.DataFrame({"symbol": ["Z"]})
    )
    assert list(sm.load_master()["symbol"]) == ["Z"] and built == [1]  # missing -> build
    pd.DataFrame({"symbol": ["OK"]}).to_parquet(sm.MASTER)
    assert list(sm.load_master()["symbol"]) == ["OK"] and built == [1]  # fresh
    old = time.time() - (sm.STALE_HOURS * 3600 + 120)
    os.utime(sm.MASTER, (old, old))
    assert list(sm.load_master()["symbol"]) == ["Z"] and built == [1, 1]  # stale -> rebuild
    os.utime(sm.MASTER, (old, old))
    assert list(sm.load_master(auto_refresh=False)["symbol"]) == ["OK"]  # stale but allowed
    just = time.time() - (sm.STALE_HOURS * 3600 - 120)
    os.utime(sm.MASTER, (just, just))
    assert list(sm.load_master()["symbol"]) == ["OK"]  # just under 24h
    sm.MASTER.write_text("not parquet")
    assert list(sm.load_master()["symbol"]) == ["Z"] and built == [1, 1, 1]  # unreadable -> rebuild


def test_lookups(sm, monkeypatch):
    m = pd.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "name": ["First A", "Second A", "Bee"],
            "name_clean": ["FIRST", "SECOND", "BEE"],
        }
    )
    loads = []
    monkeypatch.setattr(sm, "load_master", lambda: loads.append(1) or m)
    monkeypatch.setattr(sm, "_LOOKUP", {})
    assert sm.name_for("A") == "First A"
    assert sm.clean_name_for("A") == "FIRST"
    assert sm.name_for("B") == "Bee" and sm.clean_name_for("B") == "BEE"
    assert sm.name_for("ZZ") == "" and sm.clean_name_for("ZZ") == ""
    assert loads == [1]  # built once per process


def test_lookup_empty_master_retries(sm, monkeypatch):
    loads = []
    monkeypatch.setattr(sm, "load_master", lambda: loads.append(1) or pd.DataFrame())
    monkeypatch.setattr(sm, "_LOOKUP", {})
    assert sm.name_for("A") == ""
    assert sm.name_for("A") == ""
    assert loads == [1, 1]


def test_main_block(sm, monkeypatch, capsys, tmp_path):
    import runpy

    # run the module as __main__ with builders stubbed via fake nse/bse/requests modules
    monkeypatch.setitem(sys.modules, "nse", None)
    monkeypatch.setitem(sys.modules, "bse", None)
    monkeypatch.setattr(sys, "argv", ["symbol_master.py", "--no-us"])
    runpy.run_path(sm.__file__, run_name="__main__")
    assert "No symbols fetched" in capsys.readouterr().out
