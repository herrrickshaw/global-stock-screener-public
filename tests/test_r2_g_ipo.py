import datetime
import json
import sys

import pandas as pd
import pytest

from tests.r2g_support import fake_module, loader  # noqa: F401


@pytest.fixture
def mon(loader, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "cache"))
    us = fake_module("universe_sources", PROVIDERS={"US": 1, "CN": 2}, get_universe=None)
    m = loader("ipo_monitor", {"universe_sources": us})
    m._us = us
    return m


def test_snapshot_roundtrip_and_corrupt(mon):
    assert mon._load_snapshot("US") == set()
    mon._save_snapshot("US", {"B", "A"})
    d = json.loads((mon.SNAP_DIR / "US.json").read_text())
    assert d["tickers"] == ["A", "B"]
    assert d["updated"] == datetime.date.today().isoformat()
    assert mon._load_snapshot("US") == {"A", "B"}
    (mon.SNAP_DIR / "US.json").write_text("{bad")
    assert mon._load_snapshot("US") == set()


def test_append_log(mon):
    mon._append_log([{"a": 1}])
    mon._append_log([{"a": 2}])
    assert [e["a"] for e in json.loads(mon.LOG.read_text())] == [1, 2]
    mon.LOG.write_text("{bad")
    mon._append_log([{"a": 3}])
    assert json.loads(mon.LOG.read_text()) == [{"a": 3}]


def test_check_seed_diff_and_failures(mon, capsys):
    uni = {"US": ["A", "B"], "CN": [], "XX": None}

    def gu(m):
        if m == "BAD":
            raise RuntimeError("boom" * 30)
        return uni[m]

    mon.get_universe = gu
    out = mon.check(["US", "CN", "BAD"])
    txt = capsys.readouterr().out
    assert out == {}
    assert "US: snapshot seeded (2 tickers)" in txt
    assert "[reseeded]" not in txt
    assert "BAD: universe fetch failed" in txt
    assert "CN" not in txt
    # diff
    uni["US"] = ["B", "C"]
    out = mon.check(["US"])
    txt = capsys.readouterr().out
    assert out == {"US": {"new": ["C"], "delisted": ["A"]}}
    assert "1 new listing(s), 1 delisted" in txt
    assert "+ C" in txt and "logged 1 new listings" in txt
    log = json.loads(mon.LOG.read_text())
    assert log[0]["market"] == "US" and log[0]["new_listings"] == ["C"]
    assert mon._load_snapshot("US") == {"B", "C"}
    # reseed
    mon.check(["US"], seed=True)
    assert "[reseeded]" in capsys.readouterr().out
    # no new, quiet
    out = mon.check(["US"], verbose=False)
    assert out == {"US": {"new": [], "delisted": []}}
    assert capsys.readouterr().out == ""
    assert len(json.loads(mon.LOG.read_text())) == 1


def test_check_caps_listing_lines_at_15(mon, capsys):
    mon.get_universe = lambda m: [f"T{i:02d}" for i in range(20)] if m == "NEW" else []
    mon._save_snapshot("NEW", {"OLD"})
    mon.check(["NEW"])
    assert capsys.readouterr().out.count("      + ") == 15


def test_main_block(loader, tmp_path, monkeypatch, capsys):  # noqa: F811
    import runpy

    monkeypatch.setenv("BHAV_CACHE", str(tmp_path / "c"))
    us = fake_module("universe_sources", PROVIDERS={"US": 1}, get_universe=lambda m: ["Z"])
    monkeypatch.setitem(sys.modules, "universe_sources", us)
    monkeypatch.setattr(sys, "argv", ["ipo_monitor.py", "us", "--seed"])
    runpy.run_path("ipo_monitor.py", run_name="__main__")
    out = capsys.readouterr().out
    assert "SEED — ['US']" in out and "snapshot seeded" in out
    monkeypatch.setattr(sys, "argv", ["ipo_monitor.py"])
    runpy.run_path("ipo_monitor.py", run_name="__main__")
    assert "DIFF — ['US']" in capsys.readouterr().out


@pytest.fixture
def trk(loader, tmp_path, monkeypatch):  # noqa: F811
    t = loader("ipo_tracker")
    monkeypatch.setattr(t, "BASE", str(tmp_path))
    monkeypatch.setattr(t, "UNI_DIR", str(tmp_path / "uni"))
    monkeypatch.setattr(t, "CURATED", str(tmp_path / "cur"))
    monkeypatch.setattr(t, "LEDGER", str(tmp_path / "cur" / "led.parquet"))
    (tmp_path / "uni").mkdir()
    return t


def test_norm_and_exch(trk, monkeypatch):
    assert trk._norm(" RELIANCE ", "IN") == "RELIANCE.NS"
    assert trk._norm("ABC.BO", "IN") == "ABC.BO"
    assert trk._norm("AAPL", "US") == "AAPL"
    # build_universe missing/broken -> default label
    monkeypatch.setitem(sys.modules, "build_universe", None)
    assert trk._exch("X", "IN") == "NSE"
    assert trk._exch("X", "ZZ") == "ZZ"
    monkeypatch.setitem(
        sys.modules,
        "build_universe",
        fake_module("build_universe", _exch_from=lambda t, d: d + "!"),
    )
    assert trk._exch("X", "US") == "US!"


def test_append_universe(trk, tmp_path):
    assert trk._append_universe("US", [("A", "US"), ("B", "US")]) == 2
    assert trk._append_universe("US", [("A", "X"), ("C", "US")]) == 3
    df = pd.read_csv(tmp_path / "uni" / "US.csv", keep_default_na=False)
    assert list(df["yf_ticker"]) == ["A", "B", "C"]
    assert list(df["exchange"]) == ["US", "US", "US"]  # first kept on dup


def test_bse_new(trk, tmp_path, monkeypatch):
    bu = fake_module(
        "build_universe", build_bse=lambda: pd.DataFrame({"yf_ticker": ["1.BO", "2.BO", "3.BO"]})
    )
    monkeypatch.setitem(sys.modules, "build_universe", bu)
    assert trk._bse_new() == ["1.BO", "2.BO", "3.BO"]
    pd.DataFrame({"yf_ticker": ["1.BO", "A.NS"], "exchange": ["BSE", "NSE"]}).to_csv(
        tmp_path / "uni" / "IN.csv", index=False
    )
    assert trk._bse_new() == ["2.BO", "3.BO"]


def test_track_seed_returns_early(trk, monkeypatch):
    calls = []
    mon = fake_module(
        "ipo_monitor", check=lambda m, seed, verbose: calls.append((m, seed)) or {"US": 1}
    )
    monkeypatch.setitem(sys.modules, "ipo_monitor", mon)
    assert trk.track(["US"], seed=True) == {"US": 1}
    assert calls == [(["US"], True)]


def test_track_applies_and_ledger(trk, tmp_path, monkeypatch, capsys):
    found = {"IN": {"new": ["NEWCO"]}, "US": {"new": []}, "EU": {"new": ["X.DE"]}}
    mon = fake_module("ipo_monitor", check=lambda m, seed, verbose: found)
    bu = fake_module(
        "build_universe",
        _exch_from=lambda t, d: d,
        build_bse=lambda: pd.DataFrame({"yf_ticker": ["9.BO"]}),
    )
    monkeypatch.setitem(sys.modules, "ipo_monitor", mon)
    monkeypatch.setitem(sys.modules, "build_universe", bu)
    out = trk.track(["IN", "US", "EU"])
    assert out is found
    txt = capsys.readouterr().out
    assert "IN/BSE: 1 new scrip(s)" in txt and "IN: +2 new" in txt and "ledger" in txt
    inn = pd.read_csv(tmp_path / "uni" / "IN.csv")
    assert list(inn["yf_ticker"]) == ["NEWCO.NS", "9.BO"]
    assert list(inn["exchange"]) == ["NSE", "BSE"]
    assert not (tmp_path / "uni" / "US.csv").exists()
    led = pd.read_parquet(trk.LEDGER)
    assert len(led) == 3 and set(led["market"]) == {"IN", "EU"}
    # re-run: ledger deduped on (ticker, first_seen)
    trk.track(["IN", "US", "EU"], verbose=False)
    assert len(pd.read_parquet(trk.LEDGER)) == 3


def test_track_no_apply(trk, tmp_path, monkeypatch):
    mon = fake_module("ipo_monitor", check=lambda m, seed, verbose: {"US": {"new": ["Q"]}})
    monkeypatch.setitem(sys.modules, "ipo_monitor", mon)
    monkeypatch.setitem(sys.modules, "build_universe", None)
    trk.track(["US"], apply=False, verbose=False)
    assert not (tmp_path / "uni" / "US.csv").exists()
    assert list(pd.read_parquet(trk.LEDGER)["yf_ticker"]) == ["Q"]


def test_track_nothing_found(trk, monkeypatch):
    mon = fake_module("ipo_monitor", check=lambda m, seed, verbose: {})
    monkeypatch.setitem(sys.modules, "ipo_monitor", mon)
    trk.track(["US"])
    import os

    assert not os.path.exists(trk.LEDGER)


def test_main(trk, monkeypatch, capsys):
    got = {}
    monkeypatch.setattr(
        trk, "track", lambda m, seed, apply: got.update(m=m, seed=seed, apply=apply)
    )
    monkeypatch.setattr(sys, "argv", ["x"])
    trk.main()
    assert got == {"m": trk.DEFAULT_MARKETS, "seed": False, "apply": True}
    assert "DETECT" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["x", "us", "in", "--seed", "--no-apply"])
    trk.main()
    assert got == {"m": ["US", "IN"], "seed": True, "apply": False}
    assert "SEED" in capsys.readouterr().out
