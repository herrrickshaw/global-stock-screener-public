"""Offline tests for screener_kit.py: seed bootstrap, cached loading, single-symbol lookup, the
incremental update and the built-in / custom screens.

screener_kit creates its cache directory at import time, so the fixture points ``BHAV_CACHE`` at
tmp_path, imports the module fresh and redirects ``CACHE`` / ``SEED_DIR`` into tmp_path. Every lazily
imported collaborator (datalink, market_memory, bhavcopy_store, liquidity, screener_in,
bhavcopy_history, data_sources) is blocked by default (import raises) and replaced per test with a
fake, so nothing reads the repo's seeds or touches the network.
"""

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

LAZY = [
    "datalink",
    "market_memory",
    "bhavcopy_store",
    "liquidity",
    "screener_in",
    "bhavcopy_history",
    "data_sources",
]


@pytest.fixture
def sk(monkeypatch, tmp_path):
    cache, seeds = tmp_path / "cache", tmp_path / "seeds"
    cache.mkdir()
    seeds.mkdir()
    monkeypatch.setenv("BHAV_CACHE", str(cache))
    previous = sys.modules.pop("screener_kit", None)
    for name in LAZY:
        monkeypatch.setitem(sys.modules, name, None)  # None in sys.modules => ImportError
    mod = importlib.import_module("screener_kit")
    monkeypatch.setattr(mod, "CACHE", cache)
    monkeypatch.setattr(mod, "SEED_DIR", seeds)
    mod.seeds, mod.cache = seeds, cache
    yield mod
    sys.modules.pop("screener_kit", None)
    if previous is not None:
        sys.modules["screener_kit"] = previous


def fake(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def _ohlcv(end, n=3, base=10.0):
    idx = pd.date_range(end=end, periods=n, freq="D")
    c = np.linspace(base, base + n - 1, n)
    return pd.DataFrame(
        {"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": [1000] * n}, index=idx
    )


def _write_seed(path, data):
    rows = []
    for sym, df in data.items():
        d = df.copy()
        d.index.name = "Date"
        d = d.reset_index()
        d.insert(0, "Symbol", sym)
        rows.append(d)
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


# ---- paths ------------------------------------------------------------------------------------


def test_seed_names_special_case_india(sk):
    assert sk._seed_name("IN") == "cleaned_long.parquet"
    assert sk._seed_name("US") == "cleaned_long_US.parquet"


def test_live_path_prefers_the_working_copy_over_the_seed(sk):
    assert sk._live_path("US") == sk.seeds / "cleaned_long_US.parquet"  # nothing exists: the seed
    (sk.cache / "cleaned_long_US.parquet").write_bytes(b"x")
    assert sk._live_path("US") == sk.cache / "cleaned_long_US.parquet"


def test_markets_lists_only_markets_with_data_in_table_order(sk):
    (sk.seeds / "cleaned_long_US.parquet").write_bytes(b"x")
    (sk.cache / "cleaned_long.parquet").write_bytes(b"x")
    (sk.cache / "cleaned_long_ZZ.parquet").write_bytes(b"x")  # not a known market
    assert sk.markets() == ["IN", "US"]


def test_market_table_has_no_duplicates(sk):
    assert len(set(sk.MARKETS)) == len(sk.MARKETS) and {"IN", "US", "EU"} <= set(sk.MARKETS)


# ---- bootstrap --------------------------------------------------------------------------------


def test_bootstrap_copies_missing_seeds_only(sk, monkeypatch):
    (sk.seeds / "cleaned_long.parquet").write_bytes(b"seed-in")
    (sk.seeds / "cleaned_long_US.parquet").write_bytes(b"seed-us")
    (sk.cache / "cleaned_long_US.parquet").write_bytes(b"live-us")  # already there: kept
    out = sk.bootstrap(rebuild_store=False, verbose=False)
    assert out == {"seeded": ["IN"], "cache": str(sk.cache)}
    assert (sk.cache / "cleaned_long.parquet").read_bytes() == b"seed-in"
    assert (sk.cache / "cleaned_long_US.parquet").read_bytes() == b"live-us"


def test_bootstrap_reports_every_copied_market(sk):
    for name in ("cleaned_long.parquet", "cleaned_long_US.parquet", "cleaned_long_JP.parquet"):
        (sk.seeds / name).write_bytes(b"s")
    assert sk.bootstrap(rebuild_store=False, verbose=False)["seeded"] == ["IN", "US", "JP"]


def test_bootstrap_derives_short_term_seeds_first(sk, monkeypatch):
    calls = []
    fake(monkeypatch, "market_memory", ensure_stm_seeds=lambda verbose: calls.append(verbose))
    sk.bootstrap(rebuild_store=False, verbose=False)
    sk.bootstrap(rebuild_store=False, verbose=True)
    assert calls == [False, True]


def test_bootstrap_survives_a_failing_stm_step(sk, monkeypatch):
    def boom(verbose):
        raise RuntimeError("no LTM")

    fake(monkeypatch, "market_memory", ensure_stm_seeds=boom)
    (sk.seeds / "cleaned_long.parquet").write_bytes(b"s")
    assert sk.bootstrap(rebuild_store=False, verbose=False)["seeded"] == ["IN"]


def test_bootstrap_rebuilds_the_store_by_default(sk, monkeypatch, capsys):
    built = []
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: built.append(verbose) or 42)
    sk.bootstrap(verbose=True)
    out = capsys.readouterr().out
    assert built == [True] and "store ready: 42 symbols" in out
    assert "bootstrap: seeded 0 markets" in out


def test_bootstrap_without_a_store_rebuild_never_imports_it(sk):
    # bhavcopy_store is blocked in the fixture: importing it would raise
    assert sk.bootstrap(rebuild_store=False, verbose=False)["seeded"] == []


def test_bootstrap_quiet_mode_prints_nothing(sk, monkeypatch, capsys):
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: 1)
    sk.bootstrap(verbose=False)
    assert capsys.readouterr().out == ""


# ---- load -------------------------------------------------------------------------------------


def test_load_prefers_the_memoised_datalink_loader(sk, monkeypatch):
    seen = []
    fake(
        monkeypatch,
        "datalink",
        load_market=lambda m, t: seen.append((m, t)) or {"X": _ohlcv("2024-01-10")},
    )
    assert list(sk.load("US", 5_000.0)) == ["X"] and seen == [("US", 5_000.0)]


def test_load_reads_the_live_parquet_into_symbol_frames(sk):
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"B": _ohlcv("2024-01-10"), "A": _ohlcv("2024-01-10", n=5)},
    )
    got = sk.load("US")
    assert sorted(got) == ["A", "B"] and len(got["A"]) == 5
    assert list(got["A"].columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert got["A"].index.is_monotonic_increasing


def test_load_falls_back_to_the_committed_seed(sk):
    _write_seed(sk.seeds / "cleaned_long_US.parquet", {"S": _ohlcv("2024-01-10")})
    assert list(sk.load("US")) == ["S"]


def test_load_returns_empty_for_a_market_with_no_data(sk):
    assert sk.load("US") == {}


def test_load_applies_the_liquidity_prefilter(sk, monkeypatch):
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"A": _ohlcv("2024-01-10"), "B": _ohlcv("2024-01-10")},
    )
    args = []
    fake(
        monkeypatch,
        "liquidity",
        liquid_symbols=lambda m, min_usd: args.append((m, min_usd)) or ["B"],
    )
    assert list(sk.load("US", min_turnover_usd=1e6)) == ["B"] and args == [("US", 1e6)]


def test_load_skips_the_prefilter_when_the_floor_is_zero_or_negative(sk, monkeypatch):
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv("2024-01-10")})
    fake(monkeypatch, "liquidity", liquid_symbols=lambda m, min_usd: pytest.fail("not called"))
    assert list(sk.load("US", 0.0)) == ["A"] and list(sk.load("US", -5.0)) == ["A"]


def test_load_keeps_everything_when_the_liquidity_filter_fails(sk, monkeypatch):
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv("2024-01-10")})

    def boom(m, min_usd):
        raise RuntimeError("no index")

    fake(monkeypatch, "liquidity", liquid_symbols=boom)
    assert list(sk.load("US", 1e6)) == ["A"]


def test_load_can_filter_down_to_nothing(sk, monkeypatch):
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv("2024-01-10")})
    fake(monkeypatch, "liquidity", liquid_symbols=lambda m, min_usd: [])
    assert sk.load("US", 1e6) == {}


def test_load_survives_a_failing_datalink(sk, monkeypatch):
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv("2024-01-10")})

    def boom(m, t):
        raise RuntimeError("memo broke")

    fake(monkeypatch, "datalink", load_market=boom)
    assert list(sk.load("US")) == ["A"]


# ---- get --------------------------------------------------------------------------------------


def test_get_returns_the_store_hit_without_scanning_seeds(sk, monkeypatch):
    frame = _ohlcv("2024-01-10")
    fake(monkeypatch, "bhavcopy_store", get=lambda s: frame)
    monkeypatch.setattr(sk, "load", lambda m: pytest.fail("seeds must not be scanned"))
    assert sk.get("AAA") is frame


def test_get_falls_back_to_scanning_markets_in_table_order(sk, monkeypatch):
    fake(monkeypatch, "bhavcopy_store", get=lambda s: None)
    asked = []
    hit = _ohlcv("2024-01-10")

    def load(m):
        asked.append(m)
        return {"AAA": hit} if m == "JP" else {}

    monkeypatch.setattr(sk, "load", load)
    assert sk.get("AAA") is hit and asked == ["IN", "US", "JP"]  # stops at the first hit


def test_get_falls_back_when_the_store_is_unavailable(sk, monkeypatch):
    hit = _ohlcv("2024-01-10")
    monkeypatch.setattr(sk, "load", lambda m: {"AAA": hit} if m == "US" else {})
    assert sk.get("AAA") is hit  # bhavcopy_store is blocked: ImportError is swallowed


def test_get_unknown_symbol_returns_none_after_every_market(sk, monkeypatch):
    asked = []
    monkeypatch.setattr(sk, "load", lambda m: asked.append(m) or {})
    assert sk.get("NOPE") is None and asked == sk.MARKETS


# ---- update -----------------------------------------------------------------------------------


def test_update_india_uses_the_official_bhavcopy_history(sk, monkeypatch):
    calls = []
    fake(monkeypatch, "bhavcopy_history", fetch_history=lambda verbose: calls.append(verbose))
    assert sk.update("IN", verbose=False) == {"market": "IN", "source": "bhavcopy"}
    assert calls == [False]


def test_update_without_a_seed_reports_an_error(sk, monkeypatch):
    fake(monkeypatch, "data_sources", fetch=lambda *a, **k: pytest.fail("no fetch"))
    out = sk.update("US", verbose=False)
    assert out == {"market": "US", "error": "no seed; run a full fetch first"}


def _recording_fetch(monkeypatch, fresh=None):
    calls = []

    def multi_fetch(tickers, order, period, min_bars, verbose):
        calls.append(
            {
                "tickers": tickers,
                "order": order,
                "period": period,
                "min_bars": min_bars,
                "verbose": verbose,
            }
        )
        return fresh or {}

    fake(monkeypatch, "data_sources", fetch=multi_fetch)
    return calls


@pytest.mark.parametrize(
    "gap_days,period",
    [(0, "5d"), (7, "5d"), (8, "1mo"), (35, "1mo"), (36, "3mo"), (400, "3mo")],
)
def test_update_picks_the_fetch_window_from_the_staleness_gap(sk, monkeypatch, gap_days, period):
    last = pd.Timestamp.today().normalize() - pd.Timedelta(days=gap_days)
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv(last)})
    calls = _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: 1)
    sk.update("US", verbose=False)
    assert calls[0]["period"] == period


def test_update_counts_a_single_bar_symbol_when_finding_the_newest_bar(sk, monkeypatch):
    today = pd.Timestamp.today().normalize()
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"OLD": _ohlcv(today - pd.Timedelta(days=200), n=3), "ONE": _ohlcv(today, n=1)},
    )
    calls = _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: 1)
    sk.update("US", verbose=False)
    assert calls[0]["period"] == "5d"


def test_update_with_only_empty_frames_defaults_to_a_month(sk, monkeypatch, capsys):
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: {"A": _ohlcv("2024-01-10", n=0)})
    calls = _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: 1)
    sk.update("US", verbose=True)
    assert calls[0]["period"] == "1mo" and "last bar None" in capsys.readouterr().out


def test_update_hands_the_cleaner_a_sorted_deduplicated_frame(sk, monkeypatch):
    today = pd.Timestamp.today().normalize()
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv(today, n=3, base=10.0)})
    # fresh bars are OLDER than the baseline, so the plain concat is out of order
    older = _ohlcv(today - pd.Timedelta(days=10), n=3, base=50.0)
    _recording_fetch(monkeypatch, fresh={"A": older})
    seen = []
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: None)
    fake(
        monkeypatch,
        "stock_utils",
        clean_ohlcv=lambda df, ticker, min_bars: seen.append((df, ticker, min_bars)),
    )
    assert sk.update("US", verbose=False)["updated"] == 1
    df, ticker, min_bars = seen[0]
    assert df.index.is_monotonic_increasing and df.index.is_unique and len(df) == 6
    assert ticker == "A" and min_bars == 1


def test_update_fetch_parameters(sk, monkeypatch):
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"A": _ohlcv(pd.Timestamp.today().normalize()), "B": _ohlcv("2024-01-10")},
    )
    calls = _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: 1)
    sk.update("US", verbose=True)
    c = calls[0]
    assert c["tickers"] == ["A", "B"] and c["order"] == ("yahoo", "stooq")
    assert c["min_bars"] == 1 and c["verbose"] is True


def test_update_uses_the_newest_bar_across_all_symbols_for_the_gap(sk, monkeypatch):
    today = pd.Timestamp.today().normalize()
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"OLD": _ohlcv(today - pd.Timedelta(days=200)), "NEW": _ohlcv(today)},
    )
    calls = _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: 1)
    sk.update("US", verbose=False)
    assert calls[0]["period"] == "5d"


def test_update_merges_fresh_bars_and_rebuilds_the_store(sk, monkeypatch, capsys):
    today = pd.Timestamp.today().normalize()
    old = _ohlcv(today - pd.Timedelta(days=2), n=3, base=10.0)
    fresh_a = _ohlcv(today, n=3, base=99.0)  # overlaps 1 old day, adds 2 new
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": old, "B": _ohlcv(today, n=3)})
    _recording_fetch(monkeypatch, fresh={"A": fresh_a})
    built = []
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: built.append(verbose))
    fake(monkeypatch, "stock_utils", clean_ohlcv=lambda df, ticker, min_bars: None)
    out = sk.update("US", verbose=True)
    assert out == {"market": "US", "updated": 1}  # only A grew
    merged = sk.load("US")
    assert len(merged["A"]) == 5 and len(merged["B"]) == 3
    overlap = today - pd.Timedelta(days=2)
    assert merged["A"].loc[overlap, "Close"] == 99.0  # fresh bar beats the old one (12.0)
    assert merged["A"].loc[today, "Close"] == 101.0
    assert merged["A"].index.is_monotonic_increasing
    assert built == [False]
    assert "1 symbols got new bars; store rebuilt" in capsys.readouterr().out


def test_update_without_new_data_counts_nothing_updated(sk, monkeypatch):
    today = pd.Timestamp.today().normalize()
    seed = _ohlcv(today)
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": seed})
    _recording_fetch(monkeypatch, fresh={"A": seed})  # nothing newer than what we hold
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: None)
    fake(monkeypatch, "stock_utils", clean_ohlcv=lambda df, ticker, min_bars: None)
    assert sk.update("US", verbose=False)["updated"] == 0


def test_update_prints_the_plan_when_verbose(sk, monkeypatch, capsys):
    today = pd.Timestamp.today().normalize()
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv(today)})
    _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: None)
    sk.update("US", verbose=True)
    out = capsys.readouterr().out
    assert "update US: 1 symbols" in out and f"last bar {today.date()}" in out
    assert "fetching 5d" in out


def test_update_quiet_mode_prints_nothing(sk, monkeypatch, capsys):
    _write_seed(sk.cache / "cleaned_long_US.parquet", {"A": _ohlcv("2024-01-10")})
    _recording_fetch(monkeypatch)
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: None)
    sk.update("US", verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="`clean_ohlcv(...) or df` evaluates a DataFrame's truth value when cleaning succeeds",
)
def test_update_merges_with_the_real_cleaner(sk, monkeypatch):
    today = pd.Timestamp.today().normalize()
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"A": _ohlcv(today - pd.Timedelta(days=2), n=3)},
    )
    _recording_fetch(monkeypatch, fresh={"A": _ohlcv(today, n=3, base=50.0)})
    fake(monkeypatch, "bhavcopy_store", build=lambda verbose: None)
    assert sk.update("US", verbose=False)["updated"] == 1


# ---- India CCC and stock assembly ------------------------------------------------------------


def test_india_ccc_prefers_the_cached_datalink_map(sk, monkeypatch):
    fake(monkeypatch, "datalink", ccc_map_cached=lambda: {"TCS": 12.0})
    fake(monkeypatch, "screener_in", ccc_map=lambda: pytest.fail("network scrape"))
    assert sk._india_ccc() == {"TCS": 12.0}


def test_india_ccc_falls_back_to_screener_in_and_drops_nan(sk, monkeypatch):
    fake(monkeypatch, "screener_in", ccc_map=lambda: {"A": 5.0, "B": float("nan"), "C": -3.0})
    assert sk._india_ccc() == {"A": 5.0, "C": -3.0}


def test_india_ccc_falls_back_when_the_cached_map_raises(sk, monkeypatch):
    def boom():
        raise RuntimeError("stale")

    fake(monkeypatch, "datalink", ccc_map_cached=boom)
    fake(monkeypatch, "screener_in", ccc_map=lambda: {"A": 1.0})
    assert sk._india_ccc() == {"A": 1.0}


def test_india_ccc_is_empty_when_every_source_fails(sk):
    assert sk._india_ccc() == {}  # datalink and screener_in are both blocked


def test_stocks_attach_ccc_for_india_only(sk, monkeypatch):
    data = {"AAA": _ohlcv("2024-01-10"), "BBB": _ohlcv("2024-01-10")}
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: data)
    monkeypatch.setattr(sk, "_india_ccc", lambda: {"AAA": 30.0})
    india = {s.symbol: s for s in sk._stocks("IN")}
    assert india["AAA"].fundamentals == {"ccc": 30.0} and india["BBB"].fundamentals == {}
    assert india["AAA"].market == "IN" and india["AAA"].ohlcv is data["AAA"]
    monkeypatch.setattr(sk, "_india_ccc", lambda: pytest.fail("India-only"))
    assert all(s.fundamentals == {} for s in sk._stocks("US"))


def test_stocks_pass_the_turnover_floor_to_load(sk, monkeypatch):
    seen = []
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: seen.append((m, t)) or {})
    sk._stocks("US", 2_500.0)
    assert seen == [("US", 2_500.0)]


# ---- screen -----------------------------------------------------------------------------------


class _Strat:
    def __init__(self, results):
        self.results = results

    def screen(self, sd):
        r = self.results[sd.symbol]
        if isinstance(r, Exception):
            raise r
        return r


def _res(sym, passed, score):
    from strategies.base import Result

    return Result(sym, "fake", passed, score=score, metrics={"M": 1}, note="n")


@pytest.fixture
def fake_strategy(sk, monkeypatch):
    monkeypatch.setattr(
        sk, "load", lambda m, t=0.0: {s: _ohlcv("2024-01-10") for s in ("A", "B", "C", "D", "E")}
    )
    strat = _Strat(
        {
            "A": _res("A", True, 1.0),
            "B": _res("B", True, 9.0),
            "C": _res("C", False, 99.0),
            "D": RuntimeError("bad data"),
            "E": None,
        }
    )
    monkeypatch.setitem(sk.st.STRATEGIES, "fake", strat)
    return strat


def test_screen_rejects_an_unknown_strategy(sk):
    with pytest.raises(ValueError, match="unknown strategy"):
        sk.screen("not-a-strategy", "US")


def test_screen_keeps_passing_names_sorted_by_score_and_isolates_failures(sk, fake_strategy):
    out = sk.screen("fake", "US")
    assert list(out["Symbol"]) == ["B", "A"]  # failed, raising and None results dropped
    assert list(out["Score"]) == [9.0, 1.0] and list(out.index) == [0, 1]


def test_screen_top_limits_and_resets_the_index(sk, fake_strategy):
    out = sk.screen("fake", "US", top=1)
    assert list(out["Symbol"]) == ["B"] and list(out.index) == [0]


def test_screen_top_zero_means_no_limit(sk, fake_strategy):
    assert len(sk.screen("fake", "US", top=0)) == 2


def test_screen_with_no_matches_is_an_empty_frame(sk, monkeypatch):
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: {"A": _ohlcv("2024-01-10")})
    monkeypatch.setitem(sk.st.STRATEGIES, "fake", _Strat({"A": _res("A", False, 1.0)}))
    out = sk.screen("fake", "US", top=5)
    assert out.empty


def test_screen_annotates_with_liquidity_when_available(sk, fake_strategy, monkeypatch):
    fake(
        monkeypatch,
        "liquidity",
        annotate=lambda df: df.assign(Liquidity="High"),
    )
    assert set(sk.screen("fake", "US")["Liquidity"]) == {"High"}


def test_screen_passes_the_market_and_floor_through(sk, fake_strategy, monkeypatch):
    seen = []
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: seen.append((m, t)) or {})
    sk.screen("fake", "JP", min_turnover_usd=123.0)
    assert seen == [("JP", 123.0)]


def test_every_builtin_strategy_slug_is_accepted(sk, monkeypatch):
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: {})
    for slug in sk.st.STRATEGIES:
        assert sk.screen(slug, "US").empty


# ---- custom_screen / _add_liquidity -----------------------------------------------------------


def test_custom_screen_forwards_every_option_and_annotates(sk, monkeypatch):
    seen = {}

    def cs_screen(stocks, criteria, rank_by, top, ascending, show):
        seen.update(
            n=len(list(stocks)),
            criteria=criteria,
            rank_by=rank_by,
            top=top,
            ascending=ascending,
            show=show,
        )
        return pd.DataFrame({"Symbol": ["A"]})

    monkeypatch.setattr(sk.cs, "screen", cs_screen)
    monkeypatch.setattr(sk, "load", lambda m, t=0.0: {"A": _ohlcv("2024-01-10")})
    fake(monkeypatch, "liquidity", annotate=lambda df: df.assign(Liquidity="Low"))
    out = sk.custom_screen(
        {"ltp": (">", 1)}, "US", rank_by="ltp", top=5, ascending=True, show=["ltp"]
    )
    assert seen == {
        "n": 1,
        "criteria": {"ltp": (">", 1)},
        "rank_by": "ltp",
        "top": 5,
        "ascending": True,
        "show": ["ltp"],
    }
    assert list(out["Liquidity"]) == ["Low"]


def test_custom_screen_defaults_to_india_top_50_descending(sk, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        sk.cs,
        "screen",
        lambda stocks, criteria, **kw: seen.update(kw) or pd.DataFrame(),
    )
    monkeypatch.setattr(sk, "_stocks", lambda m, t=0.0: seen.update(market=m, floor=t) or [])
    sk.custom_screen({})
    assert seen["market"] == "IN" and seen["top"] == 50 and seen["ascending"] is False
    assert seen["rank_by"] is None and seen["show"] is None and seen["floor"] == 0.0


def test_custom_screen_passes_the_turnover_floor_to_stock_loading(sk, monkeypatch):
    floors = []
    monkeypatch.setattr(sk.cs, "screen", lambda stocks, criteria, **kw: pd.DataFrame())
    monkeypatch.setattr(sk, "_stocks", lambda m, t=0.0: floors.append(t) or [])
    sk.custom_screen({}, "US", min_turnover_usd=7.0)
    assert floors == [7.0]


def test_custom_screen_runs_the_real_screener_on_cached_data(sk):
    _write_seed(
        sk.cache / "cleaned_long_US.parquet",
        {"LOW": _ohlcv("2024-01-10", base=10.0), "HIGH": _ohlcv("2024-01-10", base=500.0)},
    )
    out = sk.custom_screen({"ltp": (">", 100)}, "US")
    assert list(out["Symbol"]) == ["HIGH"]


def test_add_liquidity_returns_the_frame_unchanged_without_the_module(sk):
    df = pd.DataFrame({"Symbol": ["A"]})
    assert sk._add_liquidity(df) is df


def test_add_liquidity_survives_a_failing_annotator(sk, monkeypatch):
    def boom(df):
        raise RuntimeError("no index")

    fake(monkeypatch, "liquidity", annotate=boom)
    df = pd.DataFrame({"Symbol": ["A"]})
    assert sk._add_liquidity(df) is df
