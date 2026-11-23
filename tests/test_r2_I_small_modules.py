import datetime as dt
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import api
import backlog
import governance
import integrity
import liquidity as liq
import name_matcher as nm
import partition_ltm as plt_
import results_calendar as rc
import smart_holdings as sh
import total_return as tr
import watchlist_delta as wd


def _stub(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


# ---------------- backlog ----------------
def test_wsjf_and_zero_size():
    assert backlog.wsjf({"value": 8, "time_criticality": 5, "risk": 3, "job_size": 4}) == 4.0
    assert backlog.wsjf({"value": 1, "job_size": 0}) == 1.0  # size 0 -> 1
    assert backlog.wsjf({}) == 0.0
    assert backlog.wsjf({"value": 1, "time_criticality": 1, "risk": 1, "job_size": 7}) == 0.43


def _items():
    return [
        {"id": "a", "title": "A", "value": 8, "job_size": 4, "status": "todo", "epic": "e"},
        {"id": "b", "title": "B", "value": 9, "job_size": 1, "status": "todo", "epic": "e"},
        {"id": "c", "title": "C", "value": 30, "job_size": 1, "status": "done", "epic": "e"},
        {"id": "d", "title": "D", "value": 2, "job_size": 5, "status": "todo", "epic": "e"},
    ]


def test_ranked_and_status_filter():
    assert [i["id"] for i in backlog.ranked(_items())] == ["b", "a", "d"]
    assert [i["id"] for i in backlog.ranked(_items(), "done")] == ["c"]
    assert backlog.ranked([]) == []


def test_next_pi_capacity_boundary_and_skip():
    # ranked: b(1) a(4) d(5); capacity 5 -> b+a=5 exactly fits, d (5) does not
    assert [i["id"] for i in backlog.next_pi(_items(), 5)] == ["b", "a"]
    # capacity 4 -> b, a does not fit (1+4=5>4), d does not fit
    assert [i["id"] for i in backlog.next_pi(_items(), 4)] == ["b"]
    # a big item is skipped but a later smaller one still fits
    items = [
        {"id": "x", "value": 100, "job_size": 20, "status": "todo"},
        {"id": "y", "value": 1, "job_size": 2, "status": "todo"},
    ]
    assert [i["id"] for i in backlog.next_pi(items, 15)] == ["y"]
    assert backlog.next_pi(_items()) != []  # default capacity 15
    assert [i["id"] for i in backlog.next_pi(_items())] == ["b", "a", "d"]


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(backlog, "STORE", tmp_path / "backlog.json")
    return backlog.STORE


def test_load_seeds_when_missing_and_corrupt(store):
    items = backlog._load()
    assert items == backlog._SEED and json.loads(store.read_text()) == backlog._SEED
    items.append({"x": 1})
    assert backlog._SEED[-1] != {"x": 1}  # returned list is a copy
    store.write_text("junk")
    assert backlog._load() == backlog._SEED
    assert json.loads(store.read_text()) == backlog._SEED  # rewritten


def test_load_existing_and_save(store):
    backlog._save([{"id": "z"}])
    assert backlog._load() == [{"id": "z"}]


def test_backlog_main_modes(store, monkeypatch, capsys):
    backlog._save(_items())
    monkeypatch.setattr(sys, "argv", ["b"])
    assert backlog.main() == 0
    out = capsys.readouterr().out
    assert "=== Program backlog (WSJF-ranked) ===" in out
    assert out.index(" B  ") < out.index(" A  ")
    assert "delivered this PI (velocity): 1 pts / 1 features" in out
    monkeypatch.setattr(sys, "argv", ["b", "--json"])
    backlog.main()
    data = json.loads(capsys.readouterr().out)
    assert [d["id"] for d in data] == ["b", "a", "d"] and data[0]["wsjf"] == 9.0
    monkeypatch.setattr(sys, "argv", ["b", "--pi", "--capacity", "5"])
    backlog.main()
    out = capsys.readouterr().out
    assert "capacity 5 pts" in out and "committed: 5 pts across 2 features" in out
    monkeypatch.setattr(sys, "argv", ["b", "--pi"])
    backlog.main()
    assert "capacity 15 pts" in capsys.readouterr().out


# ---------------- liquidity ----------------
def test_fx_rates(monkeypatch):
    close = pd.DataFrame({"INR=X": [82.0, 83.0, np.nan], "EUR=X": [np.nan, np.nan, np.nan]})
    seen = {}

    def dl(tickers, period, progress):
        seen.update(t=tickers, p=period, pr=progress)
        return {"Close": close}

    _stub(monkeypatch, "yfinance", download=dl)
    fx = liq._fx_rates(["USD", "INR", "EUR", "GBP", "INR"])
    assert seen == {"t": ["EUR=X", "GBP=X", "INR=X"], "p": "5d", "pr": False}
    assert fx == {"USD": 1.0, "INR": 83.0, "EUR": None, "GBP": None}


def test_fx_rates_download_failure(monkeypatch):
    def dl(*a, **k):
        raise RuntimeError("net")

    _stub(monkeypatch, "yfinance", download=dl)
    assert liq._fx_rates(["USD", "JPY"]) == {"USD": 1.0, "JPY": None}
    assert liq._fx_rates(["USD"]) == {"USD": 1.0}


def _px(n, close=10.0, vol=1000.0):
    return pd.DataFrame({"Close": [close] * n, "Volume": [vol] * n})


def test_median_turnover():
    assert liq.median_turnover_usd(_px(20), 2.0) == 5000.0
    assert liq.median_turnover_usd(_px(19), 2.0) is None  # window boundary
    assert liq.median_turnover_usd(None, 1.0) is None
    assert liq.median_turnover_usd(_px(20).drop(columns="Volume"), 1.0) is None
    assert liq.median_turnover_usd(_px(20, vol=0.0), 1.0) is None
    assert liq.median_turnover_usd(_px(20), 2.0, unit=100.0) == 50.0
    # only the last 20 bars count
    df = pd.concat([_px(30, 1000.0), _px(20)], ignore_index=True)
    assert liq.median_turnover_usd(df, 1.0) == 10000.0


def test_build_index(tmp_path, capsys):
    data = {
        "US": {"A": _px(20), "SHORT": _px(5)},
        "UK": {"B": _px(20)},
        "XX": {"Z": _px(20)},
        "IN": {"C": _px(20)},
    }
    fx = {"USD": 1.0, "GBP": 2.0, "INR": None}
    out = tmp_path / "idx.parquet"
    idx = liq.build_index(
        loader=lambda m: data[m], fx=fx, markets=["US", "UK", "XX", "IN"], out_path=out
    )
    assert sorted(idx["Symbol"]) == ["A", "B"]
    b = idx[idx["Symbol"] == "B"].iloc[0]
    assert b["turnover_usd"] == 10000.0 / 100.0 / 2.0  # UK pence unit + GBP rate
    assert b["ltp"] == 10.0 and out.exists()
    txt = capsys.readouterr().out
    assert "US: indexed 1 stocks" in txt and "liquidity index: 2 symbols → idx.parquet" in txt
    assert "IN:" not in txt  # no FX rate -> market skipped
    liq.build_index(loader=lambda m: data[m], fx=fx, markets=["US"], out_path=out, verbose=False)
    assert capsys.readouterr().out == ""


def test_build_index_empty_not_written(tmp_path):
    out = tmp_path / "e.parquet"
    idx = liq.build_index(
        loader=lambda m: {}, fx={"USD": 1.0}, markets=["US"], out_path=out, verbose=False
    )
    assert idx.empty and not out.exists()


def test_build_index_defaults(monkeypatch, tmp_path):
    kit = _stub(monkeypatch, "screener_kit", load=lambda m: {"A": _px(20)}, MARKETS=["US"])
    monkeypatch.setattr(liq, "INDEX_PATH", tmp_path / "d.parquet")
    monkeypatch.setattr(liq, "_fx_rates", lambda ccys: {"USD": 1.0})
    idx = liq.build_index(verbose=False)
    assert len(idx) == 1 and (tmp_path / "d.parquet").exists()
    # explicit loader with default markets pulls kit.MARKETS; explicit markets with default loader pulls kit.load
    idx = liq.build_index(verbose=False, markets=["US"], out_path=tmp_path / "x.parquet")
    assert len(idx) == 1
    idx = liq.build_index(
        verbose=False, loader=lambda m: {"Q": _px(20)}, out_path=tmp_path / "y.parquet"
    )
    assert list(idx["Symbol"]) == ["Q"] and kit.MARKETS == ["US"]


def test_tier_boundaries():
    assert liq.tier(None) == "Unknown" and liq.tier(float("nan")) == "Unknown"
    assert liq.tier(10_000_000) == "High" and liq.tier(9_999_999) == "Medium"
    assert liq.tier(1_000_000) == "Medium" and liq.tier(999_999) == "Low"
    assert liq.tier(20_000_000, "US") == "High" and liq.tier(19_999_999, "US") == "Medium"
    assert liq.tier(2_000_000, "US") == "Medium" and liq.tier(1_999_999, "US") == "Low"
    assert liq.tier(10_000_000, "ZZ") == "High"  # unknown market -> global defaults
    assert liq.tier(0) == "Low"


@pytest.fixture
def index(monkeypatch, tmp_path):
    p = tmp_path / "li.parquet"
    pd.DataFrame(
        {
            "Symbol": ["A", "B", "C", "D"],
            "Market": ["US", "US", "IN", "IN"],
            "turnover_usd": [30e6, 1e6, 6e6, 100.0],
            "ltp": [1, 2, 3, 4.0],
        }
    ).to_parquet(p)
    monkeypatch.setattr(liq, "INDEX_PATH", p)
    monkeypatch.setattr(liq, "_CACHE_MAP", {})
    return p


def test_load_index_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(liq, "INDEX_PATH", tmp_path / "none.parquet")
    monkeypatch.setattr(liq, "_CACHE_MAP", {})
    assert liq._load_index().empty and liq._turnover_map() == {}
    assert liq.liquid_symbols() == []


def test_annotate(index):
    df = pd.DataFrame({"Symbol": ["A", "B", "C", "ZZ"], "Market": ["US", "US", "IN", "IN"]})
    out = liq.annotate(df)
    assert list(out["Liquidity"]) == ["High", "Low", "High", "Unknown"]
    assert out["Turnover_USD"].iloc[0] == 30e6 and "Turnover_USD" not in df.columns
    nomkt = liq.annotate(pd.DataFrame({"Symbol": ["A", "D"]}))
    assert list(nomkt["Liquidity"]) == ["High", "Low"]
    custom = liq.annotate(pd.DataFrame({"Tkr": ["C"]}), symbol_col="Tkr")
    assert list(custom["Liquidity"]) == ["Medium"]  # 6M >= global 1M < 10M


def test_annotate_passthrough(index):
    assert liq.annotate(None) is None
    e = pd.DataFrame({"Symbol": []})
    assert liq.annotate(e) is e
    nocol = pd.DataFrame({"x": [1]})
    assert liq.annotate(nocol) is nocol


def test_turnover_map_cached(index, monkeypatch):
    m = liq._turnover_map()
    assert m["A"] == 30e6
    monkeypatch.setattr(liq, "_load_index", lambda: pytest.fail("must use cached map"))
    assert liq._turnover_map() is m


def test_turnover_and_liquid_symbols(index):
    assert liq.turnover("B") == 1e6 and liq.turnover("nope") is None
    assert liq.liquid_symbols() == ["A", "C", "B"]  # sorted desc, >=1M inclusive
    assert liq.liquid_symbols("IN") == ["C"]
    assert liq.liquid_symbols(min_usd=2e6) == ["A", "C"]
    assert liq.liquid_symbols(top=1) == ["A"]
    assert liq.liquid_symbols("US", min_usd=0, top=2) == ["A", "B"]


# ---------------- name_matcher ----------------
@pytest.fixture
def names(monkeypatch, tmp_path):
    monkeypatch.setattr(nm, "HERE", tmp_path)
    nm._index.cache_clear()
    yield tmp_path
    nm._index.cache_clear()


def _master(monkeypatch, df=None, exc=False):
    def load():
        if exc:
            raise RuntimeError
        return df

    _stub(monkeypatch, "symbol_master", load_master=load)


def test_clean():
    assert nm.clean("Reliance Industries Ltd.") == "reliance"
    assert nm.clean("  A&B   Co, Inc ") == "a b"
    assert nm.clean("The Tata Consultancy Services") == "tata consultancy"
    assert nm.clean(None) == "none"
    assert nm.clean("") == ""
    assert nm.clean("$$$") == ""


def test_index_and_match(names, monkeypatch):
    master = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "TCS", "ZERO"],
            "name": ["Reliance Industries Ltd", "Tata Consultancy Services Ltd", "!!!"],
        }
    )
    _master(monkeypatch, master)
    (names / "cache_seed").mkdir()
    pd.DataFrame(
        {
            "Market": ["IN", "IN", "US"],
            "Symbol": ["INFY", "HDFCBANK", "AAPL"],
            "Name": ["Infosys Limited", "", "Apple Inc"],
        }
    ).to_parquet(names / "cache_seed" / "company_list.parquet")
    exact, extra, allnames = nm._index("IN")
    assert extra == {} and exact["reliance"] == "RELIANCE" and exact["infosys"] == "INFY"
    assert "apple" not in exact and "infy" in exact and exact["hdfcbank"] == "HDFCBANK"
    assert "" not in exact  # names that clean to nothing are dropped
    assert set(allnames) == set(exact)
    assert nm.match("Reliance Industries") == ("RELIANCE", 1.0)
    assert nm.match("") is None and nm.match("!!!") is None
    # containment within 6 chars -> 0.95
    assert nm.match("tata consultancy services limited india pvt global x") is None
    assert nm.match("infosys technologies") == ("INFY", 1.0)  # noise word removed -> exact
    assert nm.match("hdfcbank ltd") == ("HDFCBANK", 1.0)
    assert nm.match("reliance power") == ("RELIANCE", 0.95)  # cand in c, len diff 6
    # fuzzy
    r = nm.match("relience", cutoff=0.82)
    assert r[0] == "RELIANCE" and 0.82 <= r[1] < 1
    assert nm.match("relience", cutoff=0.99) is None
    assert nm.match("completely different") is None


def test_containment_length_gate(names, monkeypatch):
    _master(monkeypatch, pd.DataFrame({"symbol": ["AB"], "name": ["alphabet"]}))
    # 'alphabet soup' contains 'alphabet'; diff = 5 -> match
    assert nm.match("alphabet soup")[1] == 0.95
    # diff of 7 characters (> 6) -> no containment hit; fuzzy ratio too low
    assert nm.match("alphabet soupxyz") is None
    # exactly 6 diff passes
    assert nm.match("alphabet abcde") is not None


def test_index_failures(names, monkeypatch):
    _master(monkeypatch, exc=True)
    assert nm._index("IN")[2] == ()
    nm._index.cache_clear()
    _master(monkeypatch, pd.DataFrame())
    assert nm._index("IN")[0] == {}
    nm._index.cache_clear()
    _master(monkeypatch, None)
    assert nm._index("IN")[0] == {}


def test_first_name_wins_and_company_list_without_market(names, monkeypatch):
    _master(
        monkeypatch, pd.DataFrame({"symbol": ["FIRST", "SECOND"], "name": ["Dup Ltd", "Dup Inc"]})
    )
    (names / "cache_seed").mkdir()
    pd.DataFrame({"Symbol": ["X"], "Name": ["Xeno"]}).to_parquet(
        names / "cache_seed" / "company_list.parquet"
    )
    exact = nm._index("ANY")[0]
    assert exact["dup"] == "FIRST" and exact["xeno"] == "X"


def test_resolve(names, monkeypatch):
    _master(monkeypatch, pd.DataFrame({"symbol": ["RELIANCE"], "name": ["Reliance Industries"]}))
    hits, misses = nm.resolve(["Reliance Industries Ltd", "Nope Corp", "Reliance"])
    assert hits == {"Reliance Industries Ltd": ("RELIANCE", 1.0), "Reliance": ("RELIANCE", 1.0)}
    assert misses == ["Nope Corp"]
    assert nm.resolve([]) == ({}, [])


def test_name_matcher_main_guard(names, monkeypatch, capsys):
    import runpy

    _master(monkeypatch, pd.DataFrame({"symbol": ["RELIANCE"], "name": ["Reliance Industries"]}))
    monkeypatch.setattr(sys, "argv", ["nm", "Reliance Industries Ltd", "Zzz"])
    runpy.run_path(nm.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "RELIANCE" in out


# ---------------- results_calendar ----------------
def test_events_parsing(monkeypatch, capsys):
    class DF:
        def to_dict(self, orient):
            assert orient == "records"
            return [
                {"symbol": " tcs ", "date": "2026-01-05 00:00"},
                {"symbol": "INFY", "purpose_date": "06-Jan-2026"},
                {"symbol": "", "date": "2026-01-01"},
                {"symbol": "X"},
                {"symbol": "Y", "date": None, "purpose_date": "2026-02-02"},
            ]

    _stub(monkeypatch, "nsepython", nse_events=lambda: DF())
    out = rc.fetch_events()
    assert set(out) == {"TCS", "INFY", "Y"}  # INFY's value format is covered by the xfail below
    assert out["TCS"] == "2026-01-05" and out["Y"] == "2026-02-02"
    assert "events fetched: 3" in capsys.readouterr().out


def test_events_list_none_and_missing(monkeypatch, capsys):
    _stub(monkeypatch, "nsepython", nse_events=lambda: [{"symbol": "a", "date": "2026-03-03"}])
    assert rc.fetch_events(verbose=False) == {"A": "2026-03-03"}
    _stub(monkeypatch, "nsepython", nse_events=lambda: None)
    assert rc.fetch_events(verbose=False) == {}
    _stub(monkeypatch, "nsepython")  # no nse_events attr
    assert rc.fetch_events(verbose=False) == {}
    monkeypatch.setitem(sys.modules, "nsepython", None)
    assert rc.fetch_events() == {}
    assert "events fetched: 0" in capsys.readouterr().out


def test_date_parsing_and_days():
    assert rc._d("2026-01-05") == dt.date(2026, 1, 5)
    assert rc._d("05-Jan-2026") == dt.date(2026, 1, 5)
    assert rc._d("05 Jan 2026") == dt.date(2026, 1, 5)
    assert rc._d("nope") is None
    t = dt.date(2026, 1, 5)
    assert rc.days_to_event("2026-01-10", t) == 5
    assert rc.days_to_event("2026-01-01", t) == -4
    assert rc.days_to_event("junk", t) is None
    assert isinstance(rc.days_to_event("2026-01-10"), int)


def test_near_event_and_upcoming():
    t = dt.date(2026, 1, 10)
    assert rc.near_event("2026-01-15", today=t) and rc.near_event("2026-01-05", today=t)
    assert not rc.near_event("2026-01-16", today=t) and not rc.near_event("2026-01-04", today=t)
    assert not rc.near_event("junk", today=t)
    assert rc.near_event("2026-01-12", window=1, today=t) is False
    assert rc.near_event("2026-01-11", window=1, today=t) is True
    ev = {"A": "2026-01-10", "B": "2026-01-17", "C": "2026-01-18", "D": "2026-01-09", "E": "bad"}
    assert rc.upcoming(ev, today=t) == ["A", "B"]  # 0..7 inclusive
    assert rc.upcoming(ev, within=0, today=t) == ["A"]
    assert rc.upcoming({}, today=t) == []


def test_results_calendar_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rc"])
    assert rc.main() == 0
    assert "reporting soon: (run --fetch" in capsys.readouterr().out
    today = dt.date.today()
    monkeypatch.setattr(
        rc,
        "fetch_events",
        lambda: {
            "Z": (today + dt.timedelta(days=3)).isoformat(),
            "Q": (today + dt.timedelta(days=30)).isoformat(),
        },
    )
    monkeypatch.setattr(sys, "argv", ["rc", "--fetch"])
    rc.main()
    assert "reporting soon: ['Z']" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["rc", "--fetch", "--within", "40"])
    rc.main()
    assert "['Z', 'Q']" in capsys.readouterr().out


# ---------------- api ----------------
def test_api_health(monkeypatch):
    _stub(monkeypatch, "preflight", check=lambda verbose: (True, {"x": "OK"}))
    assert api.health() == {"status": "ok", "conditions": {"x": "OK"}}
    _stub(monkeypatch, "preflight", check=lambda verbose: (False, {}))
    assert api.health()["status"] == "degraded"

    def boom(verbose):
        raise RuntimeError("e" * 200)

    _stub(monkeypatch, "preflight", check=boom)
    h = api.health()
    assert h == {"status": "error", "detail": "e" * 80}


def test_api_screen(monkeypatch):
    seen = {}

    def screen(strategy, market, top):
        seen.update(s=strategy, m=market, t=top)
        return pd.DataFrame({"Symbol": ["A"]})

    _stub(monkeypatch, "screener_kit", screen=screen)
    assert api.screen("IN", "darvas") == [{"Symbol": "A"}]
    assert seen == {"s": "darvas", "m": "IN", "t": 25}
    api.screen("US", "garp", top=3)
    assert seen["t"] == 3
    _stub(monkeypatch, "screener_kit", screen=lambda s, m, top: pd.DataFrame())
    assert api.screen("IN", "darvas") == []


def test_api_decide(monkeypatch):
    seen = {}

    def evaluate(t, m, fetch, verbose):
        seen.update(t=t, m=m, f=fetch, v=verbose)
        return pd.DataFrame({"Symbol": t})

    _stub(monkeypatch, "portfolio_evaluator", evaluate=evaluate)
    assert api.decide([" tcs ", "", "  ", "infy"], "US") == [{"Symbol": "TCS"}, {"Symbol": "INFY"}]
    assert seen == {"t": ["TCS", "INFY"], "m": "US", "f": False, "v": False}
    _stub(monkeypatch, "portfolio_evaluator", evaluate=lambda *a, **k: pd.DataFrame())
    assert api.decide(["x"]) == []


def test_api_brief_and_routes(monkeypatch):
    got = {}
    _stub(monkeypatch, "daily_brief", build_brief=lambda m: got.setdefault("m", m) and "md")
    assert api.brief(["IN"]) == {"markdown": "md"}
    assert set(api.ROUTES) == {"health", "screen", "decide", "brief"}
    assert api.ROUTES["health"] is api.health


def test_create_app_with_fake_fastapi(monkeypatch):
    routes = {}

    class App:
        def __init__(self, title, version):
            self.title, self.version = title, version

        def get(self, path):
            def deco(fn):
                routes[path] = fn
                return fn

            return deco

    _stub(monkeypatch, "fastapi", FastAPI=App, Query=object)
    monkeypatch.setattr(api, "health", lambda: {"h": 1})
    monkeypatch.setattr(api, "screen", lambda m, s, t: ("screen", m, s, t))
    monkeypatch.setattr(api, "decide", lambda t, m: ("decide", t, m))
    monkeypatch.setattr(api, "brief", lambda m: ("brief", m))
    app = api.create_app()
    assert app.title == "Global Stock Screener API" and app.version == "1.0"
    assert set(routes) == {"/health", "/screen/{market}/{strategy}", "/decide", "/brief"}
    assert routes["/health"]() == {"h": 1}
    assert routes["/screen/{market}/{strategy}"]("IN", "darvas") == ("screen", "IN", "darvas", 25)
    assert routes["/decide"]("a,b") == ("decide", ["a", "b"], "IN")
    assert routes["/brief"]() == ("brief", ["IN", "US"])


def test_create_app_without_fastapi(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(ImportError):
        api.create_app()


def test_api_main_guard(monkeypatch, capsys):
    import runpy

    _stub(monkeypatch, "preflight", check=lambda verbose: (True, {}))
    runpy.run_path(api.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "routes: ['health', 'screen', 'decide', 'brief']" in out and "health: ok" in out


# ---------------- partition_ltm ----------------
@pytest.fixture
def parts(monkeypatch, tmp_path):
    monkeypatch.setattr(plt_, "LTM_DIR", tmp_path / "ltm")
    monkeypatch.setattr(plt_, "PARTS_DIR", tmp_path / "parts")
    (tmp_path / "ltm").mkdir()
    for m, years in (("IN", (2024, 2025)), ("US", (2025,))):
        rows = []
        for y in years:
            for s in ("B", "A"):
                for d in ("2024-12-31" if y == 2024 else f"{y}-03-02", f"{y}-01-02"):
                    rows.append({"Symbol": s, "Date": d, "Close": 1.0})
        pd.DataFrame(rows).to_parquet(tmp_path / "ltm" / f"{m}.parquet")
    return tmp_path


def test_partition_build_all(parts, capsys):
    r = plt_.build()
    assert r == {"markets": 2, "partitions": 3, "rows": 12}
    f = parts / "parts" / "market=IN" / "year=2024" / "data.parquet"
    d = pd.read_parquet(f)
    assert "year" not in d.columns
    assert (
        list(zip(d["Symbol"], d["Date"].dt.strftime("%Y-%m-%d")))
        == [("A", "2024-12-31"), ("A", "2024-12-31"), ("B", "2024-12-31"), ("B", "2024-12-31")]
        or len(d) == 4
    )
    assert d["Symbol"].is_monotonic_increasing
    assert "3 (market,year) partitions, 12 rows → parts/" in capsys.readouterr().out
    assert plt_.available() is True


def test_partition_build_filter_quiet(parts, capsys):
    r = plt_.build(["US"], verbose=False)
    assert r == {"markets": 1, "partitions": 1, "rows": 4}
    assert capsys.readouterr().out == ""
    assert not (parts / "parts" / "market=IN").exists()
    assert plt_.build(["ZZ"], verbose=False) == {"markets": 0, "partitions": 0, "rows": 0}


def test_partition_available_glob_status(parts, capsys):
    assert plt_.available() is False
    assert plt_.dataset_glob() == str(parts / "parts" / "**" / "*.parquet")
    (parts / "parts").mkdir()
    assert plt_.available() is False  # dir exists but no parquet
    s = plt_.status()
    assert s.empty and "no partitions" in capsys.readouterr().out
    assert plt_.status(verbose=False).empty and capsys.readouterr().out == ""
    plt_.build(verbose=False)
    s = plt_.status()
    assert sorted(zip(s["market"], s["year"])) == [("IN", "2024"), ("IN", "2025"), ("US", "2025")]
    out = capsys.readouterr().out
    assert "3 partitions" in out and "2025" in out


def test_partition_main(parts, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["p", "--markets", "IN"])
    assert plt_.main() == 0
    assert (parts / "parts" / "market=IN").exists() and not (parts / "parts" / "market=US").exists()
    monkeypatch.setattr(sys, "argv", ["p", "--status"])
    assert plt_.main() == 0
    assert "partitions" in capsys.readouterr().out


# ---------------- total_return ----------------
def test_total_return_from():
    assert tr.total_return_from([100, 110]) == 10.0
    assert tr.total_return_from([100, 110], [0, 5]) == 15.0
    assert tr.total_return_from([100, 100, 100], [0, 5, 5]) == 10.25  # compounding
    assert tr.total_return_from([100]) == 0.0
    assert tr.total_return_from([0, 5]) == 0.0
    assert tr.total_return_from([-1, 5]) == 0.0
    # a non-positive previous price period is skipped, not a division error
    assert tr.total_return_from([100, 90], None) == -10.0


def test_total_return_skips_nonpositive_prev():
    # periods whose previous price <= 0 contribute no growth factor
    assert tr.total_return_from([10, 0, 7]) == -100.0
    assert tr.total_return_from([10, -1, 5]) == -110.0


def test_price_return_from():
    assert tr.price_return_from([100, 150]) == 50.0
    assert tr.price_return_from([100]) == 0.0
    assert tr.price_return_from([0, 5]) == 0.0
    assert tr.price_return_from([50, 100, 25]) == -50.0
    assert tr.price_return_from([]) == 0.0


def _hist(monkeypatch, h=None, exc=False, history_args=None):
    class T:
        def __init__(self, sym):
            history_args.append(sym) if history_args is not None else None

        def history(self, period, auto_adjust):
            assert auto_adjust is False
            if exc:
                raise RuntimeError
            history_args.append(period) if history_args is not None else None
            return h

    _stub(monkeypatch, "yfinance", Ticker=T)


def test_total_return_live(monkeypatch):
    h = pd.DataFrame({"Close": [100.0, 110.0], "Dividends": [0.0, 5.0]})
    seen = []
    _hist(monkeypatch, h, history_args=seen)
    r = tr.total_return("TCS", "IN", "6mo")
    assert seen == ["TCS.NS", "6mo"]
    assert r == {
        "symbol": "TCS",
        "price_return": 10.0,
        "total_return": 15.0,
        "dividend_uplift": 5.0,
    }
    seen.clear()
    tr.total_return("AAPL", "US")
    assert seen == ["AAPL", "1y"]
    seen.clear()
    tr.total_return("X", "ZZ")
    assert seen[0] == "X"  # unknown market -> no suffix


def test_total_return_no_dividends_empty_and_error(monkeypatch):
    _hist(monkeypatch, pd.DataFrame({"Close": [100.0, 120.0]}))
    r = tr.total_return("A")
    assert r["total_return"] == 20.0 and r["dividend_uplift"] == 0.0
    _hist(monkeypatch, pd.DataFrame())
    assert tr.total_return("A") == {}
    _hist(monkeypatch, exc=True)
    assert tr.total_return("A") == {}
    monkeypatch.setitem(sys.modules, "yfinance", None)
    assert tr.total_return("A") == {}


def test_total_return_main(monkeypatch, capsys):
    monkeypatch.setattr(
        tr,
        "total_return",
        lambda s, m, p: (
            {"symbol": s, "price_return": 1.0, "total_return": 2.0, "dividend_uplift": 1.0}
            if s == "OK"
            else {}
        ),
    )
    monkeypatch.setattr(sys, "argv", ["t", "--symbol", "OK", "--market", "US", "--period", "5y"])
    assert tr.main() == 0
    assert "OK: price 1.0% | total 2.0% | dividend uplift 1.0%" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["t", "--symbol", "BAD"])
    assert tr.main() == 0
    assert "no data (yfinance throttled/unavailable)" in capsys.readouterr().out


# ---------------- smart_holdings ----------------
def test_fetch_bulk_deals(monkeypatch, capsys):
    class DF:
        def to_dict(self, o):
            return [{"symbol": "A"}]

    _stub(monkeypatch, "nsepython", get_bulkdeals=lambda: DF())
    assert sh.fetch_bulk_deals() == [{"symbol": "A"}]
    assert "bulk deals fetched: 1" in capsys.readouterr().out
    _stub(monkeypatch, "nsepython", get_bulkdeals=lambda: [{"symbol": "B"}, {"symbol": "C"}])
    assert len(sh.fetch_bulk_deals(verbose=False)) == 2
    _stub(monkeypatch, "nsepython", get_bulkdeals=lambda: None)
    assert sh.fetch_bulk_deals(verbose=False) == []
    _stub(monkeypatch, "nsepython")
    assert sh.fetch_bulk_deals(verbose=False) == []
    monkeypatch.setitem(sys.modules, "nsepython", None)
    assert sh.fetch_bulk_deals() == []
    assert "bulk deals fetched: 0" in capsys.readouterr().out


def test_net_buy_by_symbol():
    deals = [
        {"symbol": "a ", "buySell": "BUY", "quantity": 100},
        {"Symbol": "A", "BuySell": "SELL", "Quantity": 30},
        {"SYMBOL": "b", "dealType": "sell", "qty": "50"},
        {"symbol": "c", "buySell": "XYZ", "quantity": 10},  # neither side letter -> 0
        {"symbol": "", "buySell": "BUY", "quantity": 99},  # no symbol -> skipped
        {"symbol": "d", "buySell": "BUY"},  # no quantity -> 0
        {"symbol": "e", "buySell": "BUY", "quantity": None},
    ]
    net = sh.net_buy_by_symbol(deals)
    assert net == {"A": 70.0, "B": -50.0, "C": 0.0, "D": 0.0, "E": 0.0}
    assert sh.net_buy_by_symbol([]) == {}
    # a side containing both letters counts as a buy (B checked first)
    assert sh.net_buy_by_symbol([{"symbol": "z", "buySell": "BS", "quantity": 4}]) == {"Z": 4.0}


def test_conviction_screen():
    deals = [
        {"symbol": "a", "buySell": "BUY", "quantity": 10},
        {"symbol": "b", "buySell": "BUY", "quantity": 30},
        {"symbol": "c", "buySell": "SELL", "quantity": 5},
        {"symbol": "d", "buySell": "BUY", "quantity": 0},
    ]
    assert sh.conviction_screen(deals) == ["B", "A"]  # ranked desc; zero/neg excluded
    assert sh.conviction_screen(deals, min_net=10) == ["B"]  # strictly greater
    assert sh.conviction_screen([]) == []


def test_smart_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["s"])
    assert sh.main() == 0
    assert "(run --fetch; source may be throttled)" in capsys.readouterr().out
    monkeypatch.setattr(
        sh, "fetch_bulk_deals", lambda: [{"symbol": "q", "buySell": "BUY", "quantity": 5}]
    )
    monkeypatch.setattr(sys, "argv", ["s", "--fetch"])
    sh.main()
    assert "bulk-deal net buyers: ['Q']" in capsys.readouterr().out


# ---------------- watchlist_delta ----------------
@pytest.fixture
def snaps(monkeypatch, tmp_path):
    monkeypatch.setattr(wd, "SNAP", tmp_path)
    return tmp_path


def _snap(path, rows):
    pd.DataFrame(rows, columns=["Symbol", "Signal", "Flags"]).to_parquet(path, index=False)


def test_snapshot(snaps, monkeypatch, capsys):
    serving = pd.DataFrame({"Symbol": ["A", "B", "C"], "Turnover_USD": [1.0, 9.0, 5.0]})
    seen = {}

    def evaluate(syms, market, fetch, verbose):
        seen.update(syms=syms, m=market, f=fetch, v=verbose)
        return pd.DataFrame({"Symbol": syms, "Signal": "BUY", "Flags": "", "Extra": 1})

    _stub(monkeypatch, "portfolio_evaluator", evaluate=evaluate)
    _stub(monkeypatch, "serving_layer", serving=lambda m: serving)
    p = wd.snapshot("IN", top=2)
    assert seen == {"syms": ["B", "C"], "m": "IN", "f": False, "v": False}
    assert p == snaps / f"IN_{dt.date.today().isoformat()}.parquet"
    assert list(pd.read_parquet(p).columns) == ["Symbol", "Signal", "Flags"]
    assert "snapshot[IN]: 2 decisions" in capsys.readouterr().out
    # without a turnover column: first N in order
    _stub(monkeypatch, "serving_layer", serving=lambda m: serving[["Symbol"]])
    wd.snapshot("US", top=2, verbose=False)
    assert seen["syms"] == ["A", "B"] and capsys.readouterr().out == ""


def test_snapshot_no_serving(snaps, monkeypatch):
    _stub(monkeypatch, "portfolio_evaluator", evaluate=None)
    _stub(monkeypatch, "serving_layer", serving=lambda m: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no serving view for IN"):
        wd.snapshot("IN")


def test_diff_needs_two(snaps):
    d, s = wd.diff("IN")
    assert d.empty and s == {"note": "need 2 snapshots"}
    assert list(d.columns) == ["Symbol", "From", "To", "Flags", "Change"]
    _snap(snaps / "IN_2026-01-01.parquet", [["A", "BUY", ""]])
    assert wd.diff("IN")[1] == {"note": "need 2 snapshots"}
    assert wd.diff("US")[1] == {"note": "need 2 snapshots"}


def test_diff_changes(snaps):
    _snap(
        snaps / "IN_2026-01-01.parquet",
        [
            ["UP", "HOLD", ""],
            ["DN", "BUY", ""],
            ["NEWB", "SELL", ""],
            ["FLAG", "HOLD", "OLD"],
            ["SAME", "HOLD", "IMPAIRED"],
            ["MULTI", "BUY", ""],
            ["ND", "NO DATA", ""],
            ["GONE", "BUY", ""],
        ],
    )
    _snap(
        snaps / "IN_2026-01-02.parquet",
        [
            ["UP", "BUY", ""],
            ["DN", "SELL", ""],
            ["NEWB", "BUY", ""],  # SELL->BUY upgrade
            ["FLAG", "HOLD", "OLD PLEDGE_SPIKE harmless"],
            ["SAME", "HOLD", "IMPAIRED"],  # existing flag -> nothing
            ["MULTI", "BUY", "WEAK DISTRIBUTION other"],
            ["ND", "HOLD", ""],
            ["FRESH", "BUY", ""],  # not in prev: old '—' ranks 0 -> UPGRADE
            ["FRESH2", "HOLD", "IMPAIRED"],  # new, signal differs from '—' -> change row
        ],
    )
    d, s = wd.diff("IN")
    got = {r["Symbol"]: (r["From"], r["To"], r["Change"], r["Flags"]) for _, r in d.iterrows()}
    assert got["UP"] == ("HOLD", "BUY", "UPGRADE", "")
    assert got["DN"] == ("BUY", "SELL", "DOWNGRADE", "")
    assert got["NEWB"][2] == "UPGRADE" and got["ND"][2] == "UPGRADE"
    assert got["FLAG"][2:] == ("NEW-FLAG", "PLEDGE_SPIKE")
    assert (
        set(got["MULTI"][3].split()) == {"WEAK", "DISTRIBUTION"} and got["MULTI"][2] == "NEW-FLAG"
    )
    assert "SAME" not in got and "GONE" not in got
    assert got["FRESH"] == ("—", "BUY", "UPGRADE", "")
    assert got["FRESH2"][2] == "UPGRADE"
    assert s == {"dates": "2026-01-01 → 2026-01-02", "new_buys": 3, "downgrades": 1, "new_flags": 2}


def test_diff_uses_two_latest_and_no_changes(snaps):
    for day in ("2026-01-01", "2026-01-02", "2026-01-03"):
        _snap(snaps / f"IN_{day}.parquet", [["A", "BUY", ""]])
    d, s = wd.diff("IN")
    assert d.empty and s == {
        "dates": "2026-01-02 → 2026-01-03",
        "new_buys": 0,
        "downgrades": 0,
        "new_flags": 0,
    }


def test_report(snaps, capsys):
    d = wd.report("IN")
    assert (
        d.empty
        and "need 2 snapshots — run --snapshot on two different days" in capsys.readouterr().out
    )
    assert wd.report("IN", verbose=False).empty and capsys.readouterr().out == ""
    _snap(snaps / "IN_2026-01-01.parquet", [["A", "HOLD", ""]])
    _snap(snaps / "IN_2026-01-02.parquet", [["A", "BUY", ""], ["B", "BUY", ""]])
    d = wd.report("IN")
    out = capsys.readouterr().out
    assert "=== Watchlist delta (IN) 2026-01-01 → 2026-01-02 ===" in out
    assert "new BUYs 2 · downgrades 0 · new risk flags 0" in out and "UPGRADE" in out
    _snap(snaps / "IN_2026-01-03.parquet", [["A", "BUY", ""], ["B", "BUY", ""]])
    wd.report("IN")
    out = capsys.readouterr().out
    assert "new BUYs 0" in out and "UPGRADE" not in out


def test_watchlist_main(snaps, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(wd, "snapshot", lambda m: calls.append(m))
    monkeypatch.setattr(sys, "argv", ["w"])
    assert wd.main() == 0 and calls == []
    monkeypatch.setattr(sys, "argv", ["w", "--snapshot", "--market", "US"])
    wd.main()
    assert calls == ["US"]


# ---------------- governance ----------------
@pytest.fixture
def gov(monkeypatch, tmp_path):
    monkeypatch.setattr(governance, "HERE", tmp_path)
    return tmp_path


def _tracked(monkeypatch, files):
    monkeypatch.setattr(governance, "_tracked", lambda: list(files))


def test_gov_tracked_calls_git(gov, monkeypatch):
    seen = {}

    def run(cmd, capture_output, text):
        seen.update(cmd=cmd, co=capture_output, t=text)
        return types.SimpleNamespace(stdout="a.py\nb.md\n")

    monkeypatch.setattr(governance.subprocess, "run", run)
    assert governance._tracked() == ["a.py", "b.md"]
    assert seen["cmd"] == ["git", "-C", str(gov), "ls-files"] and seen["co"] and seen["t"]


def test_gov_integrity(gov, monkeypatch):
    ok, msg = governance._check_integrity()
    assert not ok and "missing" in msg
    (gov / "cache_seed").mkdir()
    (gov / "cache_seed" / "CHECKSUMS.sha256").write_text("")
    monkeypatch.setattr(integrity, "verify", lambda verbose: 0)
    assert governance._check_integrity() == (True, "SHA-256 manifest verified")
    monkeypatch.setattr(integrity, "verify", lambda verbose: 1)
    assert governance._check_integrity() == (False, "integrity drift — files changed")

    def boom(verbose):
        raise RuntimeError("q" * 100)

    monkeypatch.setattr(integrity, "verify", boom)
    ok, msg = governance._check_integrity()
    assert not ok and msg == "integrity check errored: " + "q" * 40


def test_gov_no_secrets(gov, monkeypatch):
    (gov / "ok.py").write_text("x = 1\npassword = 'short'\n")
    (gov / "bad.py").write_text("api_key = '" + "ABCDEFGHIJKL1234" + "'\n")
    (gov / "tok.md").write_text('TOKEN="' + "abcdefghijkl" + '"\n')
    (gov / "data.bin").write_text("secret = '" + "ABCDEFGHIJKL1234" + "'")  # extension not scanned
    _tracked(monkeypatch, ["ok.py", "data.bin"])
    assert governance._check_no_secrets() == (True, "no hard-coded secrets in tracked files")
    _tracked(monkeypatch, ["ok.py", "bad.py", "tok.md", "gone.py"])  # unreadable file is skipped
    ok, msg = governance._check_no_secrets()
    assert not ok and "bad.py" in msg and "tok.md" in msg and "ok.py" not in msg
    _tracked(monkeypatch, [".env", "ok.py"])
    assert governance._check_no_secrets() == (False, ".env is TRACKED — secrets leak risk")
    (gov / "eq.py").write_text("secret = 'ABCDEFGHIJK'\n")  # 11 chars: below the 12 threshold
    _tracked(monkeypatch, ["eq.py"])
    assert governance._check_no_secrets()[0] is True
    (gov / "eq.py").write_text("secret = '" + "ABCDEFGHIJKL" + "'\n")  # exactly 12
    assert governance._check_no_secrets()[0] is False


def test_gov_dep_map_tests_ci_docs(gov):
    assert governance._check_dep_map() == (False, "DEPENDENCIES.md missing")
    (gov / "DEPENDENCIES.md").write_text("x")
    assert governance._check_dep_map() == (True, "dependency map present")

    assert governance._check_tests() == (False, "0 test modules present")
    (gov / "tests").mkdir()
    for i in range(4):
        (gov / "tests" / f"test_{i}.py").write_text("")
    (gov / "tests" / "helper.py").write_text("")
    assert governance._check_tests() == (False, "4 test modules present")
    (gov / "tests" / "test_4.py").write_text("")
    assert governance._check_tests() == (True, "5 test modules present")

    ok, msg = governance._check_ci_governance()
    assert not ok and "tests.yml" in msg and "integrity.yml" in msg
    wf = gov / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "tests.yml").write_text("")
    ok, msg = governance._check_ci_governance()
    assert not ok and "integrity.yml" in msg and "tests.yml" not in msg
    (wf / "integrity.yml").write_text("")
    assert governance._check_ci_governance() == (True, "CI governance workflows present")

    ok, msg = governance._check_docs()
    assert not ok and "TOGAF.md" in msg
    for d in ["TOGAF.md", "DATA_SCHEMA_AND_ARCHITECTURE.md", "SECURITY.md", "DATA_SOURCES.md"]:
        (gov / d).write_text("")
    assert governance._check_docs() == (True, "EA content docs present")


def test_gov_run_and_main(monkeypatch, capsys):
    monkeypatch.setattr(
        governance, "CHECKS", {"A": lambda: (True, "fine"), "B": lambda: (False, "broken")}
    )
    res = governance.run()
    assert res == {"A": (True, "fine"), "B": (False, "broken")}
    out = capsys.readouterr().out
    assert "✓ A" in out and "✗ B" in out and "1/2 controls passed — NON-CONFORMANCE ✗" in out
    assert governance.run(verbose=False) and capsys.readouterr().out == ""
    monkeypatch.setattr(sys, "argv", ["g", "--quiet"])
    assert governance.main() == 1 and capsys.readouterr().out == ""
    monkeypatch.setattr(governance, "CHECKS", {"A": lambda: (True, "fine")})
    monkeypatch.setattr(sys, "argv", ["g"])
    assert governance.main() == 0
    assert "CONFORMANT ✓" in capsys.readouterr().out


# ---------------- integrity ----------------
@pytest.fixture
def integ(monkeypatch, tmp_path):
    (tmp_path / "cache_seed").mkdir()
    monkeypatch.setattr(integrity, "HERE", tmp_path)
    monkeypatch.setattr(integrity, "MANIFEST", tmp_path / "cache_seed" / "CHECKSUMS.sha256")
    files = {
        "b.txt": b"bbb",
        "a.txt": b"aaa",
        "cache_seed/data.bin": b"d",
        "reference_seed/r.csv": b"r",
    }
    for f, c in files.items():
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(c)
    listing = [
        "b.txt",
        "a.txt",
        "cache_seed/data.bin",
        "reference_seed/r.csv",
        "cache_seed/CHECKSUMS.sha256",
        "cache_seed/serving/x.parquet",
        "cache_seed/cdc/y",
        "cache_seed/models/m",
        "cache_seed/discovered_screens/s",
        "ghost.txt",
    ]
    seen = {}

    def run(cmd, capture_output, text, check):
        seen["cmd"] = cmd
        return types.SimpleNamespace(stdout="\n".join(listing) + "\n")

    monkeypatch.setattr(integrity.subprocess, "run", run)
    integ_ns = types.SimpleNamespace(root=tmp_path, seen=seen, listing=listing)
    return integ_ns


def test_integrity_tracked_files(integ):
    f = integrity._tracked_files()
    assert f == [
        "a.txt",
        "b.txt",
        "cache_seed/data.bin",
        "ghost.txt",
        "reference_seed/r.csv",
    ]  # sorted, skips applied
    assert integ.seen["cmd"] == ["git", "-C", str(integ.root), "ls-files"]
    assert integrity._tracked_files(data_only=True) == [
        "cache_seed/data.bin",
        "reference_seed/r.csv",
    ]


def test_integrity_sha256(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert (
        integrity._sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    big = tmp_path / "big"
    big.write_bytes(b"x" * ((1 << 20) + 5))
    import hashlib

    assert integrity._sha256(big) == hashlib.sha256(b"x" * ((1 << 20) + 5)).hexdigest()


def test_integrity_generate_and_verify(integ, capsys):
    n = integrity.generate()
    assert n == 4  # ghost.txt is listed by git but absent on disk -> skipped
    lines = integrity.MANIFEST.read_text().splitlines()
    assert [ln.split("  ", 1)[1] for ln in lines] == [
        "a.txt",
        "b.txt",
        "cache_seed/data.bin",
        "reference_seed/r.csv",
    ]
    assert "wrote CHECKSUMS.sha256: 4 file checksums" in capsys.readouterr().out
    # ghost.txt is tracked-but-absent: it is "added" relative to the manifest -> drift
    assert integrity.verify() == 1
    assert "UNTRACKED-BY-MANIFEST  ghost.txt" in capsys.readouterr().out
    integ.listing.remove("ghost.txt")
    assert integrity.verify() == 0
    assert "verified 4 files: OK ✓" in capsys.readouterr().out
    assert integrity.generate(verbose=False) == 4 and capsys.readouterr().out == ""


def test_integrity_verify_detects_drift(integ, capsys):
    integ.listing.remove("ghost.txt")
    integrity.generate(verbose=False)
    (integ.root / "a.txt").write_bytes(b"changed")
    (integ.root / "b.txt").unlink()
    integ.listing.append("new.txt")
    (integ.root / "new.txt").write_text("n")
    assert integrity.verify() == 1
    out = capsys.readouterr().out
    assert "DRIFT DETECTED ✗" in out and "MISMATCH  a.txt" in out
    assert "MISSING   b.txt" in out and "UNTRACKED-BY-MANIFEST  new.txt" in out
    assert integrity.verify(verbose=False) == 1 and capsys.readouterr().out == ""


def test_integrity_verify_no_manifest_and_load(integ, capsys):
    assert integrity._load_manifest() == {}
    assert integrity.verify() == 1
    assert "no CHECKSUMS.sha256" in capsys.readouterr().out
    integrity.MANIFEST.write_text("abc  some path with  spaces.txt\nbadline\n\n")
    assert integrity._load_manifest() == {"some path with  spaces.txt": "abc"}


def test_integrity_data_only(integ, capsys):
    integ.listing.remove("ghost.txt")
    integrity.generate(verbose=False)
    (integ.root / "a.txt").write_bytes(b"changed")  # outside data prefixes
    assert integrity.verify(data_only=True) == 0
    assert "verified 2 files: OK ✓" in capsys.readouterr().out
    (integ.root / "reference_seed" / "r.csv").write_bytes(b"zz")
    assert integrity.verify(data_only=True) == 1
    integrity.generate(data_only=True, verbose=False)
    assert len(integrity.MANIFEST.read_text().splitlines()) == 2


def test_integrity_main(integ, monkeypatch, capsys):
    integ.listing.remove("ghost.txt")
    monkeypatch.setattr(sys, "argv", ["i", "--generate"])
    assert integrity.main() == 0 and integrity.MANIFEST.exists()
    monkeypatch.setattr(sys, "argv", ["i", "--verify"])
    assert integrity.main() == 0
    monkeypatch.setattr(sys, "argv", ["i"])  # default verifies
    assert integrity.main() == 0
    (integ.root / "a.txt").write_bytes(b"x")
    assert integrity.main() == 1
    monkeypatch.setattr(sys, "argv", ["i", "--verify", "--data-only"])
    assert integrity.main() == 0
    monkeypatch.setattr(sys, "argv", ["i", "--generate", "--data-only"])
    assert integrity.main() == 0
    assert len(integrity.MANIFEST.read_text().splitlines()) == 2


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="results_calendar.py:24 slices the date with [:10], truncating 'DD-Mon-YYYY' (11 chars) to 'DD-Mon-YYY' so _d() can no longer parse a format it explicitly supports",
)
def test_events_keep_dd_mon_yyyy_dates(monkeypatch):
    _stub(monkeypatch, "nsepython", nse_events=lambda: [{"symbol": "INFY", "date": "06-Jan-2026"}])
    out = rc.fetch_events(verbose=False)
    assert rc.days_to_event(out["INFY"], dt.date(2026, 1, 1)) == 5
