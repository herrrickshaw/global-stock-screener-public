import datetime as dt
import json
import sys
import types

import pandas as pd
import pytest

from tests._r2i import fresh


# ---------------- build_universe ----------------
@pytest.fixture
def bu(monkeypatch, tmp_path):
    return fresh(monkeypatch, tmp_path, "build_universe")


def test_exch_from_suffixes(bu):
    assert bu._exch_from("VOLV-B.ST", "x") == "Nasdaq Stockholm"
    assert bu._exch_from("abc.l", "x") == "London Stock Exchange"
    assert bu._exch_from("NOVN.SW", "x") == "SIX Swiss"
    assert bu._exch_from("PLAIN", "dflt") == "dflt"
    assert bu._exch_from(None, "dflt") == "dflt"


class _Resp:
    def __init__(self, data):
        self._d = data

    def json(self):
        return self._d


DATA = [
    {"SCRIP_CD": " 500325 ", "Scrip_Name": " RELIANCE "},
    {"SCRIP_CD": "500325", "Scrip_Name": "dup"},
    {"SCRIP_CD": "ABC", "Scrip_Name": "non numeric"},
    {"SCRIP_CD": "532540"},
    {},
]


def test_build_bse_filters_and_dedups(bu, monkeypatch):
    seen = {}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return _Resp(DATA)

    monkeypatch.setattr(bu.requests, "get", get)
    df = bu.build_bse()
    assert list(df["yf_ticker"]) == ["500325.BO", "532540.BO"]
    assert df.iloc[0]["name"] == "RELIANCE"
    assert df.iloc[1]["name"] == ""
    assert set(df["exchange"]) == {"BSE"} and set(df["market"]) == {"IN"}
    assert seen["timeout"] == 30 and "bseindia" in seen["url"]


def test_merge_bse_new_file(bu, monkeypatch):
    monkeypatch.setattr(bu.requests, "get", lambda *a, **k: _Resp(DATA))
    assert bu.merge_bse_into_in() == (2, 2)
    out = pd.read_csv(bu.UNI_DIR + "/IN.csv")
    assert len(out) == 2


def test_merge_bse_replaces_old_bo_keeps_nse(bu, monkeypatch):
    cur = pd.DataFrame(
        {
            "yf_ticker": ["TCS.NS", "111111.bo", "500325.BO"],
            "name": ["a", "b", "c"],
            "exchange": ["NSE", "BSE", "BSE"],
            "market": ["IN"] * 3,
        }
    )
    cur.to_csv(bu.UNI_DIR + "/IN.csv", index=False)
    monkeypatch.setattr(bu.requests, "get", lambda *a, **k: _Resp(DATA))
    nb, nt = bu.merge_bse_into_in()
    out = pd.read_csv(bu.UNI_DIR + "/IN.csv")
    assert (nb, nt) == (2, 3)
    assert "111111.bo" not in set(out["yf_ticker"])
    assert "TCS.NS" in set(out["yf_ticker"])


def test_build_market(bu, monkeypatch):
    fake = types.ModuleType("universe_sources")
    fake.get_universe = lambda m: ["A.ST", "B.L", "A.ST", "C"]
    monkeypatch.setitem(sys.modules, "universe_sources", fake)
    assert bu.build_market("SE") == 3
    df = pd.read_csv(bu.UNI_DIR + "/SE.csv")
    assert list(df["exchange"]) == ["Nasdaq Stockholm", "London Stock Exchange", "Nasdaq Stockholm"]
    assert bu.build_market("ZZ") == 3
    assert set(pd.read_csv(bu.UNI_DIR + "/ZZ.csv")["market"]) == {"ZZ"}
    assert pd.read_csv(bu.UNI_DIR + "/ZZ.csv")["exchange"].iloc[2] == "ZZ"
    assert bu.build_market("TH") == 3
    assert pd.read_csv(bu.UNI_DIR + "/TH.csv")["exchange"].iloc[2] == "SET (Thailand)"


def _run_main(bu, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["build_universe.py", *argv])
    bu.main()


def test_main_bse_ok(bu, monkeypatch, capsys):
    monkeypatch.setattr(bu, "merge_bse_into_in", lambda: (5, 9))
    _run_main(bu, monkeypatch, ["--bse"])
    assert "BSE: +5 scrips → IN.csv now 9 tickers" in capsys.readouterr().out


def test_main_bse_fail(bu, monkeypatch, capsys):
    def boom():
        raise RuntimeError("x" * 300)

    monkeypatch.setattr(bu, "merge_bse_into_in", boom)
    _run_main(bu, monkeypatch, ["--bse"])
    out = capsys.readouterr().out
    assert out.startswith("BSE build failed: " + "x" * 120)
    assert "x" * 121 not in out


def test_main_markets_and_failure(bu, monkeypatch, capsys):
    def bm(m):
        if m == "BAD":
            raise ValueError("nope")
        return 7

    monkeypatch.setattr(bu, "build_market", bm)
    _run_main(bu, monkeypatch, ["--markets", " se, bad ,,"])
    out = capsys.readouterr().out.splitlines()
    assert out == ["SE: 7 tickers → SE.csv", "BAD: failed (nope)"]


def test_main_all_remaining(bu, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bu, "merge_bse_into_in", lambda: (1, 2))
    monkeypatch.setattr(bu, "build_market", lambda m: calls.append(m) or 1)
    _run_main(bu, monkeypatch, ["--all-remaining", "--markets", "xx"])
    assert calls == ["TH", "ID", "MY", "MX", "IL"]


def test_main_nothing(bu, monkeypatch, capsys):
    _run_main(bu, monkeypatch, [])
    assert capsys.readouterr().out == ""


# ---------------- market_calendar ----------------
@pytest.fixture
def mc(monkeypatch, tmp_path):
    return fresh(monkeypatch, tmp_path, "market_calendar", {"BHAV_CACHE": tmp_path / "bc"})


def test_parse(mc):
    assert mc._parse("15-Jan-2026") == "2026-01-15"
    assert mc._parse(" 2026-01-15 ") == "2026-01-15"
    assert mc._parse("15-January-2026") == "2026-01-15"
    assert mc._parse(" junk ") == "junk"


def test_load_cache_states(mc):
    assert mc._load_cache() == set()
    today = dt.date.today()
    mc.HOLIDAY_CACHE.write_text(json.dumps({"fetched": today.isoformat(), "dates": ["2030-01-01"]}))
    assert mc._load_cache() == {"2030-01-01"}
    edge = (today - dt.timedelta(days=mc.REFRESH_AFTER_DAYS)).isoformat()
    mc.HOLIDAY_CACHE.write_text(json.dumps({"fetched": edge, "dates": ["2030-01-02"]}))
    assert mc._load_cache() == {"2030-01-02"}  # exactly 14 days is not stale
    old = (today - dt.timedelta(days=mc.REFRESH_AFTER_DAYS + 1)).isoformat()
    mc.HOLIDAY_CACHE.write_text(json.dumps({"fetched": old, "dates": ["2030-01-02"]}))
    assert mc._load_cache() == set()
    mc.HOLIDAY_CACHE.write_text("not json")
    assert mc._load_cache() == set()
    mc.HOLIDAY_CACHE.write_text(json.dumps({"dates": ["x"]}))
    assert mc._load_cache() == set()


def _fake_nse(monkeypatch, payload=None, exc=None):
    m = types.ModuleType("nsepython")

    def nse_holidays():
        if exc:
            raise exc
        return payload

    m.nse_holidays = nse_holidays
    monkeypatch.setitem(sys.modules, "nsepython", m)


def test_fetch_live(mc, monkeypatch):
    _fake_nse(monkeypatch, {"CM": [{"tradingDate": "15-Jan-2026"}, {"x": 1}, {"tradingDate": ""}]})
    assert mc._fetch_live() == {"2026-01-15"}
    _fake_nse(monkeypatch, {"CBM": [{"tradingDate": "2026-02-02"}]})
    assert mc._fetch_live() == {"2026-02-02"}
    _fake_nse(monkeypatch, {})
    assert mc._fetch_live() == set()
    _fake_nse(monkeypatch, exc=RuntimeError("net"))
    assert mc._fetch_live() == set()


def test_get_holidays_live_persists(mc, monkeypatch):
    _fake_nse(monkeypatch, {"CM": [{"tradingDate": "01-Jan-2031"}]})
    h = mc.get_holidays()
    assert "2031-01-01" in h and mc._FALLBACK <= h
    saved = json.loads(mc.HOLIDAY_CACHE.read_text())
    assert saved["dates"] == ["2031-01-01"]
    assert saved["fetched"] == dt.date.today().isoformat()
    # second call uses the cache even if live now fails
    _fake_nse(monkeypatch, exc=RuntimeError("x"))
    assert "2031-01-01" in mc.get_holidays()
    # refresh=True bypasses cache
    assert "2031-01-01" not in mc.get_holidays(refresh=True)


def test_get_holidays_no_live_no_persist(mc, monkeypatch):
    _fake_nse(monkeypatch, exc=RuntimeError("x"))
    assert mc.get_holidays() == mc._FALLBACK
    assert not mc.HOLIDAY_CACHE.exists()


def test_get_holidays_write_failure_ignored(mc, monkeypatch):
    _fake_nse(monkeypatch, {"CM": [{"tradingDate": "01-Jan-2031"}]})
    monkeypatch.setattr(mc, "HOLIDAY_CACHE", mc.CACHE / "nodir" / "h.json")
    assert "2031-01-01" in mc.get_holidays()


def test_is_trading_day(mc):
    assert mc.is_trading_day(dt.date(2026, 1, 15), set()) is True  # Thursday
    assert mc.is_trading_day(dt.date(2026, 1, 15), {"2026-01-15"}) is False
    assert mc.is_trading_day(dt.date(2026, 1, 17), set()) is False  # Saturday
    assert mc.is_trading_day(dt.date(2026, 1, 18), set()) is False  # Sunday
    assert mc.is_trading_day(dt.date(2026, 1, 16), set()) is True  # Friday


def test_is_trading_day_default_holidays(mc, monkeypatch):
    monkeypatch.setattr(mc, "get_holidays", lambda: {"2026-01-15"})
    assert mc.is_trading_day(dt.date(2026, 1, 15)) is False
    assert mc.is_trading_day(dt.date(2026, 1, 14)) is True
    # an explicit empty set must not fall back to get_holidays
    assert mc.is_trading_day(dt.date(2026, 1, 15), set()) is True


def test_trading_days(mc, monkeypatch):
    monkeypatch.setattr(mc, "get_holidays", lambda: {"2026-01-14"})
    ds = [dt.date(2026, 1, d) for d in (13, 14, 15, 16, 17, 18)]
    assert mc.trading_days(ds) == [dt.date(2026, 1, 13), dt.date(2026, 1, 15), dt.date(2026, 1, 16)]
    assert mc.trading_days([]) == []
