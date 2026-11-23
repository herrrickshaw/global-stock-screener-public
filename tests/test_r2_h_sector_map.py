import sys
import types

import numpy as np
import pandas as pd
import pytest

import sector_map as sm


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "CACHE", tmp_path)
    monkeypatch.setattr(sm.time, "sleep", lambda t: None)
    return tmp_path


def _yf(monkeypatch, infos, seen=None):
    def Ticker(sym):
        if seen is not None:
            seen.append(sym)
        v = infos.get(sym)
        if isinstance(v, Exception):
            raise v
        return types.SimpleNamespace(info=v if v is not None else {})

    m = types.ModuleType("yfinance")
    m.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", m)


def test_cache_path(cache):
    assert sm._cache_path("IN") == cache / "IN.parquet"


def test_fetch_explicit_symbols(cache, monkeypatch, capsys):
    seen = []
    _yf(
        monkeypatch,
        {
            "A.NS": {"sector": "Tech"},
            "B.NS": {"industryGroup": "Banks"},
            "C.NS": {},
            "D.NS": RuntimeError("x"),
            "E.NS": {"sector": ""},
        },
        seen,
    )
    got = sm.fetch("IN", limit=10, symbols=list("ABCDE"))
    assert got == 2
    assert seen == [f"{c}.NS" for c in "ABCDE"]
    d = pd.read_parquet(cache / "IN.parquet")
    assert dict(zip(d["Symbol"], d["Sector"])) == {"A": "Tech", "B": "Banks"}
    assert "+2 tagged, 2 total cached" in capsys.readouterr().out
    # known symbols are skipped; limit applies after filtering
    seen.clear()
    _yf(monkeypatch, {"X.NS": {"sector": "S"}, "Y.NS": {"sector": "S"}}, seen)
    got = sm.fetch("IN", limit=1, symbols=["A", "B", "X", "Y"], verbose=False)
    assert seen == ["X.NS"] and got == 1
    assert capsys.readouterr().out == ""
    d = pd.read_parquet(cache / "IN.parquet")
    assert sorted(d["Symbol"]) == ["A", "B", "X"]


def test_fetch_duplicate_keeps_last(cache, monkeypatch):
    pd.DataFrame({"Symbol": ["A"], "Sector": ["Old"]}).to_parquet(cache / "US.parquet", index=False)
    _yf(monkeypatch, {"A": {"sector": "New"}})
    # A is known -> skipped, so nothing changes
    assert sm.fetch("US", symbols=["A"], verbose=False) == 0
    assert pd.read_parquet(cache / "US.parquet")["Sector"].tolist() == ["Old"]


def test_fetch_unknown_market_no_suffix(cache, monkeypatch):
    seen = []
    _yf(monkeypatch, {"Q": {"sector": "S"}}, seen)
    sm.fetch("ZZ", symbols=["Q"], verbose=False)
    assert seen == ["Q"]


def test_fetch_from_serving(cache, monkeypatch, capsys):
    df = pd.DataFrame({"Symbol": ["A", "B", "C"], "Turnover_USD": [1, 3, 2]})
    monkeypatch.setattr(sm.sl, "serving", lambda m: df)
    seen = []
    _yf(monkeypatch, {"B.NS": {"sector": "S"}, "C.NS": {"sector": "S"}}, seen)
    assert sm.fetch("IN", limit=1, verbose=False) == 1
    assert seen == ["B.NS"]  # highest turnover first
    # without Turnover_USD keep order
    monkeypatch.setattr(sm.sl, "serving", lambda m: df[["Symbol"]])
    seen.clear()
    sm.fetch("JP", limit=1, verbose=False)
    assert seen == ["A.T"]


def test_fetch_empty_serving(cache, monkeypatch, capsys):
    monkeypatch.setattr(sm.sl, "serving", lambda m: pd.DataFrame())
    _yf(monkeypatch, {})
    assert sm.fetch("IN") == 0
    assert "no serving view for IN" in capsys.readouterr().out
    assert sm.fetch("IN", verbose=False) == 0
    assert capsys.readouterr().out == ""


def test_fetch_progress_and_no_rows(cache, monkeypatch, capsys):
    syms = [f"S{i}" for i in range(50)]
    _yf(monkeypatch, {})
    assert sm.fetch("US", limit=100, symbols=syms) == 0
    out = capsys.readouterr().out
    assert "tagged 50/50" in out
    assert not (cache / "US.parquet").exists()


def test_sector_map(cache):
    assert sm.sector_map("IN") == {}
    pd.DataFrame({"Symbol": ["A"], "Sector": ["S"]}).to_parquet(cache / "IN.parquet", index=False)
    assert sm.sector_map("IN") == {"A": "S"}


def _serving(rows):
    return pd.DataFrame(rows, columns=["Symbol", "Ret252", "Ret63", "Above200DMA"])


def test_rotation_empty():
    out = sm.rotation(pd.DataFrame(), {"A": "S"})
    assert list(out.columns) == ["Sector", "Names", "MedRet252", "MedRet63", "Breadth", "Phase"]
    assert out.empty
    assert sm.rotation(_serving([("A", 1, 1, 1)]), {}).empty


def test_rotation_phases_and_sort():
    rows, smap = [], {}

    def add(sec, n, r252, r63, above):
        for i in range(n):
            s = f"{sec}{i}"
            rows.append((s, r252, r63, above))
            smap[s] = sec

    add("peak", 3, 26.0, 1.0, True)  # breadth 1, r252>25
    add("early", 3, 25.0, 1.0, True)  # r252 == 25 -> Early
    add("cool", 4, 5.0, 0.0, True)  # breadth 1 but ret63 == 0 -> not boom; not laggard
    add("lag1", 3, 0.0, -5.5, True)  # ret63 < -5 -> Laggard
    add("lag2", 3, 0.0, 1.0, False)  # breadth 0 -> Laggard
    out = sm.rotation(_serving(rows), smap)
    ph = dict(zip(out["Sector"], out["Phase"]))
    assert ph == {
        "peak": "Peak Boom",
        "early": "Early Boom",
        "cool": "Cooling",
        "lag1": "Laggard",
        "lag2": "Laggard",
    }
    # sorted by Breadth desc then MedRet63 desc
    assert sorted(out["Sector"].tolist()[:2]) == ["early", "peak"]
    assert out["Sector"].tolist()[2:] == ["cool", "lag1", "lag2"]
    assert out.iloc[-1]["Breadth"] == 0.0
    assert out.loc[out.Sector == "peak", "Names"].iloc[0] == 3


def test_rotation_breadth_boundaries():
    def mk(frac_above, n=20):
        k = round(frac_above * n)
        rows = [(f"s{i}", 10.0, 1.0, i < k) for i in range(n)]
        return sm.rotation(_serving(rows), {f"s{i}": "X" for i in range(n)}).iloc[0]

    assert mk(0.6)["Phase"] == "Early Boom"
    assert mk(0.55)["Phase"] == "Cooling"
    assert mk(0.35)["Phase"] == "Laggard"
    assert mk(0.4)["Phase"] == "Cooling"


def test_rotation_min_names_and_unmapped():
    rows = [("a", 1.0, 1.0, True), ("b", 1.0, 1.0, True), ("c", 1.0, 1.0, True)]
    assert sm.rotation(_serving(rows), {"a": "S", "b": "S"}).empty  # only 2 mapped
    out = sm.rotation(_serving(rows), {"a": "S", "b": "S"}, min_names=2)
    assert out["Names"].tolist() == [2]
    assert sm.rotation(_serving(rows), {"a": "S", "b": "S", "c": "S"}, min_names=4).empty


def test_rotation_missing_columns_and_rounding():
    df = pd.DataFrame({"Symbol": ["a", "b", "c"]})
    out = sm.rotation(df, {"a": "S", "b": "S", "c": "S"})
    assert np.isnan(out.iloc[0]["MedRet252"]) and np.isnan(out.iloc[0]["Breadth"])
    assert out.iloc[0]["Phase"] == "Cooling"
    df = pd.DataFrame(
        {
            "Symbol": ["a", "b", "c"],
            "Ret252": [1.04, 1.06, 1.1],
            "Ret63": [0.0, 0.0, 0.0],
            "Above200DMA": [True, True, False],
        }
    )
    out = sm.rotation(df, {"a": "S", "b": "S", "c": "S"}).iloc[0]
    assert out["MedRet252"] == 1.1 and out["Breadth"] == 0.67 and out["MedRet63"] == 0.0


def test_report(cache, monkeypatch, capsys):
    assert sm.report("IN").empty
    assert "no sector cache" in capsys.readouterr().out
    assert sm.report("IN", verbose=False).empty
    assert capsys.readouterr().out == ""
    smap = {f"s{i}": "Hot" for i in range(3)}
    pd.DataFrame({"Symbol": list(smap), "Sector": list(smap.values())}).to_parquet(
        cache / "IN.parquet", index=False
    )
    monkeypatch.setattr(sm.sl, "serving", lambda m: _serving([(s, 30.0, 2.0, True) for s in smap]))
    out = sm.report("IN")
    txt = capsys.readouterr().out
    assert "Sector rotation — IN (3 tagged)" in txt
    assert "booming sectors (hunt here): ['Hot']" in txt
    assert out["Phase"].tolist() == ["Peak Boom"]
    sm.report("IN", verbose=False)
    assert capsys.readouterr().out == ""
    # no boom -> no booming line
    monkeypatch.setattr(sm.sl, "serving", lambda m: _serving([(s, 1.0, -9.0, False) for s in smap]))
    sm.report("IN")
    assert "booming" not in capsys.readouterr().out
    # nothing rotates (empty out) -> no print
    monkeypatch.setattr(sm.sl, "serving", lambda m: pd.DataFrame())
    assert sm.report("IN").empty
    assert capsys.readouterr().out == ""


def test_main(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, "fetch", lambda m, l: calls.append(("f", m, l)))
    monkeypatch.setattr(sm, "report", lambda m: calls.append(("r", m)))
    monkeypatch.setattr(sys, "argv", ["sm"])
    assert sm.main() == 0
    assert calls == [("r", "IN")]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["sm", "--fetch", "--market", "US", "--limit", "7"])
    sm.main()
    assert calls == [("f", "US", 7), ("r", "US")]
