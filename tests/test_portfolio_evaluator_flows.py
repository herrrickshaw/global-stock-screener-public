"""Offline tests for portfolio_evaluator.py beyond the pure helpers: file parsing, P/E derivation,
the screen/fundamentals gathering, ``evaluate``, name resolution, manual input and the CLI.

Everything the module imports lazily (decision_engine, ticker_aliases, screen_metrics, validation,
fundamental_metrics, delivery_signal, name_matcher) is replaced by a fake in ``sys.modules`` and the
serving view / fundamentals cache / results directory are redirected into ``tmp_path``, so nothing
touches the network or the repo's cache_seed.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

import portfolio_evaluator as pev


def _mod(monkeypatch, name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def _serving(*rows):
    cols = ["Symbol", "Close", "Ret252", "Liquidity"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Redirect results/serving/fundamentals and fake every lazily imported collaborator."""
    monkeypatch.setattr(pev, "RESULTS", tmp_path)
    state = types.SimpleNamespace(
        serving=_serving(("AAA", 100.0, 12.345, "high"), ("BBB", 50.0, -3.0, "low")),
        funds=pd.DataFrame(),
        decide_calls=[],
        signals={},
        fetched=[],
    )
    monkeypatch.setattr(pev.sl, "serving", lambda market: state.serving)
    monkeypatch.setattr(pev, "_fundamentals", lambda market: state.funds)

    def decide(row, pe, med_pe, fundamentals=None):
        state.decide_calls.append((dict(row), pe, med_pe, fundamentals))
        sym = row.get("Close")
        sig = state.signals.get(sym, "HOLD")
        return {
            "signal": sig,
            "flags": ["F1", "F2"],
            "composite": 55.5,
            "recovery": 10,
            "rationale": f"r-{sig}",
        }

    _mod(monkeypatch, "decision_engine", decide=decide)
    _mod(monkeypatch, "ticker_aliases", apply_aliases=lambda ts: list(ts))
    _mod(monkeypatch, "screen_metrics", run_screen=lambda name, market: pd.DataFrame())
    _mod(monkeypatch, "validation", known_universe=lambda market: set())
    _mod(
        monkeypatch,
        "fundamental_metrics",
        _enrich=lambda f: f,
        load_fundamentals=lambda market, symbols, limit, verbose: state.fetched.append(
            list(symbols)
        ),
    )
    _mod(
        monkeypatch,
        "delivery_signal",
        CACHE=tmp_path / "deliv",
        accumulation_scores=lambda df: pd.DataFrame({"Symbol": ["AAA"], "AccumScore": [1.3]}),
    )
    return state


# ---- parse_file -------------------------------------------------------------------------------


def test_parse_file_reads_a_csv_ticker_column(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("Qty,Symbol\n1,tcs.ns\n2,INFY\n3,TCS\n")
    assert pev.parse_file(str(p)) == ["TCS", "INFY"]  # suffix stripped, de-duplicated, order kept


def test_parse_file_returns_nothing_when_no_column_matches(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("a,b\n1,2\n")
    assert pev.parse_file(str(p)) == []


def test_parse_file_skips_a_banner_row_in_excel(tmp_path, monkeypatch):
    frames = {
        0: pd.DataFrame({"Broker statement": ["x"]}),
        1: pd.DataFrame({"Scrip": ["reliance.bo", "TCS"]}),
    }
    monkeypatch.setattr(pd, "read_excel", lambda p, header=0: frames[header])
    assert pev.parse_file("holdings.xlsx") == ["RELIANCE", "TCS"]


def test_parse_file_excel_gives_up_after_four_header_rows(monkeypatch):
    seen = []

    def read(p, header=0):
        seen.append(header)
        raise ValueError("bad")

    monkeypatch.setattr(pd, "read_excel", read)
    assert pev.parse_file("x.XLS") == [] and seen == [0, 1, 2, 3]  # case-insensitive suffix


def test_clean_syms_drops_numbers_blanks_and_oversize_tokens():
    s = pd.Series(["500325", "M&M", "bajaj-auto", None, "THIS_HAS_UNDERSCORES", "A" * 21, "ok.nse"])
    assert pev._clean_syms(s) == ["M&M", "BAJAJ-AUTO", "OK"]


def test_ticker_col_prefers_the_earlier_keyword():
    df = pd.DataFrame({"Company Name": [1], "Symbol": [2]})
    assert pev._ticker_col(df) == "Symbol"  # 'symbol' outranks 'name' in TICKER_COLS


# ---- P/E derivation ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pe,want",
    [
        (None, None),
        (0.0, 0.0),
        (25.5, 25.5),
        (1000.0, 1000.0),
        (1000.1, None),  # near-zero-EPS artefact
        (-5.0, -5.0),  # loss-making is kept for the engine to flag
        (-5000.0, -5000.0),
    ],
)
def test_sane_pe_boundaries(pe, want):
    assert pev._sane_pe(pe) == want


def test_pe_of_prefers_the_stored_ratio_in_key_order():
    assert pev._pe_of({"pe": 12.34, "P/E": 99}, 100) == 12.3  # rounded to 1dp, first key wins
    assert pev._pe_of({"P/E": 8.0}, None) == 8.0
    assert pev._pe_of({"pe_ttm": 15.0}, None) == 15.0


def test_pe_of_skips_empty_stored_values_then_uses_eps():
    rec = {"pe": 0, "P/E": "", "pe_ttm": float("nan"), "eps_ttm": 5.0}
    assert pev._pe_of(rec, 100.0) == 20.0


def test_pe_of_eps_path_rounds_to_one_decimal():
    assert pev._pe_of({"eps_ttm": 3.0}, 100.0) == 33.3


def test_pe_of_eps_path_keeps_negative_eps_as_loss_making():
    assert pev._pe_of({"eps_ttm": -4.0}, 100.0) == -25.0


def test_pe_of_eps_without_a_price_does_not_fall_through_to_net_income():
    assert pev._pe_of({"eps_ttm": 5.0, "net_income": 10, "shares": 10}, None) is None


def test_pe_of_falls_back_to_market_cap_over_net_income():
    assert pev._pe_of({"net_income": 50.0, "shares": 10.0}, 100.0) == 20.0


def test_pe_of_artefact_and_empty_records_give_none():
    assert pev._pe_of({"pe": 5000}, 100) is None
    assert pev._pe_of({"eps_ttm": 0.0001}, 100.0) is None  # 1,000,000x
    assert pev._pe_of({}, 100.0) is None


# ---- _fundamentals / _fund_map ----------------------------------------------------------------


def test_fundamentals_is_empty_when_the_cache_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.undo()  # use the real function, not the fixture's stub
    monkeypatch.setattr(pev, "__file__", str(tmp_path / "portfolio_evaluator.py"))
    assert pev._fundamentals("ZZ").empty


def test_fundamentals_reads_the_market_parquet(monkeypatch, tmp_path):
    d = tmp_path / "cache_seed" / "fundamentals"
    d.mkdir(parents=True)
    pd.DataFrame({"Symbol": ["A"], "pe": [10.0]}).to_parquet(d / "IN.parquet")
    monkeypatch.setattr(pev, "__file__", str(tmp_path / "portfolio_evaluator.py"))
    assert list(pev._fundamentals("IN")["Symbol"]) == ["A"]


def test_fund_map_is_empty_without_fundamentals(env):
    assert pev._fund_map("IN", env.serving) == {}


def test_fund_map_derives_pe_from_serving_price_and_skips_symbol_less_rows(env):
    env.funds = pd.DataFrame(
        {
            "Symbol": ["AAA", "", "BBB"],
            "eps_ttm": [5.0, 1.0, np.nan],
            "pe": [np.nan, np.nan, 40.0],
        }
    )
    m = pev._fund_map("IN", env.serving)
    assert set(m) == {"AAA", "BBB"}
    assert m["AAA"]["pe"] == 20.0 and m["BBB"]["pe"] == 40.0


@pytest.mark.xfail(strict=True, reason="a NaN Symbol is truthy, so it is keyed as nan, not skipped")
def test_fund_map_skips_rows_with_a_missing_symbol(env):
    env.funds = pd.DataFrame(
        {"Symbol": pd.Series(["AAA", np.nan], dtype=object), "pe": [10.0, 20.0]}
    )
    assert set(pev._fund_map("IN", env.serving)) == {"AAA"}


def test_fund_map_survives_a_failing_enrichment(env, monkeypatch):
    def boom(f):
        raise RuntimeError("enrich failed")

    sys.modules["fundamental_metrics"]._enrich = boom
    env.funds = pd.DataFrame({"Symbol": ["AAA"], "pe": [11.0]})
    assert pev._fund_map("IN", env.serving)["AAA"]["pe"] == 11.0  # unenriched frame still used


def test_fund_map_with_an_empty_serving_view_has_no_prices(env):
    env.funds = pd.DataFrame({"Symbol": ["AAA"], "eps_ttm": [5.0]})
    assert pev._fund_map("IN", pd.DataFrame())["AAA"]["pe"] is None


# ---- enrich_missing ---------------------------------------------------------------------------


def test_enrich_missing_fetches_only_uncached_tickers(env, capsys):
    pev.enrich_missing("IN", ["AAA", "BBB", "CCC"], have={"AAA"})
    assert env.fetched == [["BBB", "CCC"]]
    assert "fetching live fundamentals for 2 uncached ticker(s)" in capsys.readouterr().out


def test_enrich_missing_limit_equals_the_number_of_missing_tickers(env):
    limits = []
    sys.modules["fundamental_metrics"].load_fundamentals = (
        lambda market, symbols, limit, verbose: limits.append(limit)
    )
    pev.enrich_missing("IN", ["A", "B", "C"], have=set())
    assert limits == [3]


def test_enrich_missing_does_nothing_when_everything_is_cached(env, capsys):
    pev.enrich_missing("IN", ["AAA"], have={"AAA"})
    assert env.fetched == [] and capsys.readouterr().out == ""


def test_enrich_missing_reports_but_survives_a_failed_fetch(env, capsys):
    def boom(market, symbols, limit, verbose):
        raise RuntimeError("x" * 100)

    sys.modules["fundamental_metrics"].load_fundamentals = boom
    pev.enrich_missing("IN", ["ZZZ"], have=set())
    out = capsys.readouterr().out
    assert "live fetch skipped (" + "x" * 40 + ")" in out  # message truncated to 40 chars


def test_enrich_missing_is_quiet_when_not_verbose(env, capsys):
    sys.modules["fundamental_metrics"].load_fundamentals = lambda *a, **k: 1 / 0
    pev.enrich_missing("IN", ["ZZZ"], have=set(), verbose=False)
    assert capsys.readouterr().out == ""


# ---- _screens_passed --------------------------------------------------------------------------


def test_screens_passed_collects_each_screen_and_the_popular_universe(env):
    asked = []

    def run_screen(name, market):
        asked.append((name, market))
        return pd.DataFrame({"Symbol": ["AAA"]}) if name == "golden_crossover" else pd.DataFrame()

    sys.modules["screen_metrics"].run_screen = run_screen
    sys.modules["validation"].known_universe = lambda market: {"AAA", "BBB"}
    got = pev._screens_passed("IN", env.serving)
    assert got == {"AAA": ["golden_crossover", "popular"], "BBB": ["popular"]}
    assert [a[0] for a in asked] == [
        "golden_crossover",
        "companies_creating_new_high",
        "breakout_stocks",
        "multibagger_momentum",
    ] and all(m == "IN" for _, m in asked)


def test_screens_passed_survives_failing_sources(env):
    def boom(*a, **k):
        raise RuntimeError("down")

    sys.modules["screen_metrics"].run_screen = boom
    sys.modules["validation"].known_universe = boom
    assert pev._screens_passed("IN", env.serving) == {}


def test_screens_passed_keeps_earlier_screens_when_a_later_one_fails(env):
    def run_screen(name, market):
        if name == "breakout_stocks":
            raise RuntimeError("boom")
        return pd.DataFrame({"Symbol": ["AAA"]}) if name == "golden_crossover" else pd.DataFrame()

    sys.modules["screen_metrics"].run_screen = run_screen
    assert pev._screens_passed("IN", env.serving) == {"AAA": ["golden_crossover"]}


# ---- _decide boundaries -----------------------------------------------------------------------


def test_decide_cheap_needs_a_trend_or_a_screen():
    assert pev._decide(15.9, 20, 0, True)[0] == "BUY"
    assert pev._decide(15.9, 20, 1, False)[0] == "BUY"
    assert pev._decide(15.9, 20, 0, False)[0] == "HOLD"  # cheap but no trend/screen: not a BUY
    assert pev._decide(25.0, 20, 0, False)[0] == "SELL"
    assert pev._decide(16.0, 20, 3, True)[0] == "HOLD"  # exactly 0.8x median is not "cheap"


def test_decide_expensive_boundaries():
    assert pev._decide(30.0, 20, 0, True)[0] == "HOLD"  # exactly 1.5x median is not "rich"
    assert pev._decide(30.1, 20, 0, True)[0] == "SELL"
    assert pev._decide(20.5, 20, 0, False)[0] == "SELL"  # above median in a downtrend
    assert pev._decide(20.0, 20, 0, False)[0] == "HOLD"  # at the median in a downtrend


def test_decide_rationales_describe_the_case():
    assert "no P/E" in pev._decide(None, 20, 0, True)[1]
    assert "downtrend" in pev._decide(None, 20, 0, False)[1]
    assert "hold if turning up" in pev._decide(-1, 20, 0, True)[1]
    assert "loss-making" in pev._decide(0, 20, 0, False)[1]  # P/E of exactly 0 is a loss
    assert (
        "undervalued" in pev._decide(10, 20, 2, True)[1]
        and "uptrend" in pev._decide(10, 20, 2, True)[1]
    )
    assert "flat" in pev._decide(10, 20, 2, False)[1]
    assert "+ downtrend" in pev._decide(50, 20, 0, False)[1]
    assert "+ downtrend" not in pev._decide(50, 20, 0, True)[1]


# ---- evaluate ---------------------------------------------------------------------------------


def test_evaluate_refuses_an_empty_serving_view(env):
    env.serving = pd.DataFrame()
    with pytest.raises(RuntimeError, match="no serving view for IN"):
        pev.evaluate(["AAA"], "IN", verbose=False)


def test_evaluate_marks_unknown_tickers_as_no_data_and_sorts_by_signal(env, tmp_path):
    env.signals = {100.0: "SELL", 50.0: "BUY"}
    out = pev.evaluate(["ZZZ", "AAA", "BBB"], "IN", fetch=False, verbose=False)
    assert list(out["Symbol"]) == ["BBB", "AAA", "ZZZ"]  # BUY, SELL, NO DATA
    assert list(out["Signal"]) == ["BUY", "SELL", "NO DATA"]
    zzz = out.iloc[2]
    assert zzz["Screens"] == 0 and zzz["Rationale"] == "not in our universe/cache"
    assert (tmp_path / "portfolio_review.csv").exists()


def test_evaluate_row_fields_and_rounding(env):
    out = pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    r = out.iloc[0]
    assert r["Close"] == 100.0 and r["Ret252"] == 12.3  # 12.345 -> 1dp
    assert r["Flags"] == "F1 F2" and r["Score"] == 55.5 and r["Recovery"] == 10
    assert r["Liquidity"] == "high" and r["Rationale"] == "r-HOLD"


def test_evaluate_median_pe_uses_only_plausible_positive_pes(env):
    env.funds = pd.DataFrame(
        {"Symbol": ["AAA", "BBB", "CCC", "DDD"], "pe": [10.0, 30.0, -5.0, 500.0]}
    )
    pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    row, pe, med, fund = env.decide_calls[0]
    assert med == 20.0  # median of 10 and 30; negative and >=200 excluded
    assert pe == 10.0 and fund["pe"] == 10.0


def test_evaluate_median_is_a_median_not_a_mean(env):
    env.funds = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "pe": [10.0, 10.0, 40.0]})
    pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    assert env.decide_calls[0][2] == 10.0


def test_evaluate_rounds_close_to_two_decimals(env):
    env.serving = _serving(("AAA", 100.456, 1.0, ""))
    out = pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    assert out.loc[0, "Close"] == 100.46


def test_evaluate_orders_hold_before_sell(env):
    env.signals = {100.0: "SELL", 50.0: "HOLD"}
    out = pev.evaluate(["AAA", "BBB"], "IN", fetch=False, verbose=False)
    assert list(out["Signal"]) == ["HOLD", "SELL"]


def test_evaluate_median_defaults_to_20_without_any_pe(env):
    pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    assert env.decide_calls[0][2] == 20.0 and env.decide_calls[0][1] is None


def test_evaluate_applies_ticker_aliases(env):
    sys.modules["ticker_aliases"].apply_aliases = lambda ts: [
        "AAA" if t == "OLD" else t for t in ts
    ]
    out = pev.evaluate(["OLD"], "IN", fetch=False, verbose=False)
    assert list(out["Symbol"]) == ["AAA"]


def test_evaluate_survives_a_broken_alias_module(env):
    def boom(ts):
        raise RuntimeError("no alias file")

    sys.modules["ticker_aliases"].apply_aliases = boom
    assert list(pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)["Symbol"]) == ["AAA"]


def test_evaluate_fetches_only_known_uncached_tickers(env):
    env.funds = pd.DataFrame({"Symbol": ["AAA"], "pe": [10.0]})
    pev.evaluate(["AAA", "BBB", "ZZZ"], "IN", fetch=True, verbose=False)
    assert env.fetched == [["BBB"]]  # AAA cached, ZZZ not in the serving view


def test_evaluate_fetch_with_no_fundamentals_cache_requests_everything_known(env):
    pev.evaluate(["AAA", "BBB"], "IN", fetch=True, verbose=False)
    assert env.fetched == [["AAA", "BBB"]]


def test_evaluate_fetch_false_never_fetches(env):
    pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    assert env.fetched == []


def test_evaluate_attaches_delivery_accumulation_for_india_only(env, tmp_path):
    deliv = tmp_path / "deliv"
    deliv.mkdir()
    pd.DataFrame({"Symbol": ["AAA"]}).to_parquet(deliv / "IN.parquet")
    pev.evaluate(["AAA", "BBB"], "IN", fetch=False, verbose=False)
    by_close = {c[0]["Close"]: c[0] for c in env.decide_calls}
    assert by_close[100.0]["accum_score"] == 1.3 and "accum_score" not in by_close[50.0]


def test_evaluate_skips_delivery_signal_for_other_markets(env, tmp_path):
    deliv = tmp_path / "deliv"
    deliv.mkdir()
    pd.DataFrame({"Symbol": ["AAA"]}).to_parquet(deliv / "IN.parquet")
    pev.evaluate(["AAA"], "US", fetch=False, verbose=False)
    assert "accum_score" not in env.decide_calls[0][0]


def test_evaluate_without_a_delivery_cache_adds_no_accumulation(env):
    pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    assert "accum_score" not in env.decide_calls[0][0]


def test_evaluate_survives_a_broken_delivery_module(env, tmp_path):
    deliv = tmp_path / "deliv"
    deliv.mkdir()
    pd.DataFrame({"Symbol": ["AAA"]}).to_parquet(deliv / "IN.parquet")

    def boom(df):
        raise RuntimeError("bad cache")

    sys.modules["delivery_signal"].accumulation_scores = boom
    assert len(pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)) == 1


def test_evaluate_counts_screens_per_symbol(env):
    sys.modules["validation"].known_universe = lambda market: {"AAA"}
    sys.modules["screen_metrics"].run_screen = lambda n, m: pd.DataFrame({"Symbol": ["AAA"]})
    out = pev.evaluate(["AAA", "BBB"], "IN", fetch=False, verbose=False)
    assert dict(zip(out["Symbol"], out["Screens"])) == {"AAA": 5, "BBB": 0}


def test_evaluate_prints_a_table_and_signal_counts_when_verbose(env, capsys):
    env.signals = {100.0: "BUY"}
    pev.evaluate(["AAA", "BBB", "ZZZ"], "IN", fetch=False, verbose=True)
    out = capsys.readouterr().out
    assert "Portfolio review (IN)" in out and "market median P/E ≈ 20" in out
    assert "BUY:1  HOLD:1  SELL:0  NO DATA:1" in out


def test_evaluate_is_silent_when_not_verbose(env, capsys):
    pev.evaluate(["AAA"], "IN", fetch=False, verbose=False)
    assert capsys.readouterr().out == ""


def test_evaluate_table_omits_columns_that_are_absent(env, capsys):
    pev.evaluate(["ZZZ"], "IN", fetch=False, verbose=True)  # only NO DATA rows
    out = capsys.readouterr().out
    assert "Symbol" in out and "Recovery" not in out


def test_evaluate_saves_the_csv_with_every_row(env, tmp_path):
    pev.evaluate(["AAA", "ZZZ"], "IN", fetch=False, verbose=False)
    saved = pd.read_csv(tmp_path / "portfolio_review.csv")
    assert list(saved["Symbol"]) == ["AAA", "ZZZ"]


# ---- resolve_names / evaluate_from_file -------------------------------------------------------


def test_resolve_names_dedups_and_reports_misses(monkeypatch, capsys):
    _mod(
        monkeypatch,
        "name_matcher",
        resolve=lambda names, market: (
            {"Reliance Ltd": ("RELIANCE", 98), "Reliance Industries": ("RELIANCE", 95)},
            ["Mystery Co"],
        ),
    )
    assert pev.resolve_names(["a", "b", "c"], "IN") == ["RELIANCE"]
    out = capsys.readouterr().out
    assert "RELIANCE  (98)" in out and "unmatched (skipped): ['Mystery Co']" in out


def test_resolve_names_quiet_mode_and_no_misses(monkeypatch, capsys):
    _mod(monkeypatch, "name_matcher", resolve=lambda names, market: ({"A": ("AAA", 90)}, []))
    assert pev.resolve_names(["A"], verbose=False) == ["AAA"]
    assert capsys.readouterr().out == ""


def test_resolve_names_does_not_print_unmatched_when_there_are_none(monkeypatch, capsys):
    _mod(monkeypatch, "name_matcher", resolve=lambda names, market: ({"A": ("AAA", 90)}, []))
    pev.resolve_names(["A"])
    assert "unmatched" not in capsys.readouterr().out


def test_evaluate_from_file_uses_tickers_when_present(env, tmp_path, capsys):
    p = tmp_path / "h.csv"
    p.write_text("Symbol\nAAA\nBBB\n")
    out = pev.evaluate_from_file(str(p), "IN", verbose=True)
    assert sorted(out["Symbol"]) == ["AAA", "BBB"]
    assert f"resolved 2 tickers from {p}" in capsys.readouterr().out


def test_evaluate_from_file_falls_back_to_resolving_names(env, tmp_path, monkeypatch):
    p = tmp_path / "h.csv"
    p.write_text("Name\n12345\n99999\n")  # digits only: no usable ticker tokens
    seen = {}

    def resolve(names, market):
        seen["names"] = names
        return {n: ("AAA", 90) for n in names[:1]}, []

    _mod(monkeypatch, "name_matcher", resolve=resolve)
    out = pev.evaluate_from_file(str(p), "IN", verbose=False)
    assert seen["names"] == ["12345", "99999"] and list(out["Symbol"]) == ["AAA"]


def test_evaluate_from_file_with_no_usable_column_passes_no_tickers(env, tmp_path, monkeypatch):
    p = tmp_path / "h.csv"
    p.write_text("foo,bar\n1,2\n")
    seen = []
    monkeypatch.setattr(pev, "evaluate", lambda t, m, verbose: seen.append(t))
    pev.evaluate_from_file(str(p), "IN", verbose=False)
    assert seen == [[]]


@pytest.mark.xfail(strict=True, raises=KeyError, reason="an empty portfolio has no Signal column")
def test_evaluate_with_no_tickers_returns_an_empty_frame(env):
    assert pev.evaluate([], "IN", fetch=False, verbose=False).empty


def test_evaluate_from_file_reads_excel_by_name_after_header_rows(env, tmp_path, monkeypatch):
    seen = []

    def read_excel(path, header=0):
        seen.append(header)
        if header < 3:
            raise ValueError("banner")
        return pd.DataFrame({"Instrument": ["1234"]})

    monkeypatch.setattr(pd, "read_excel", read_excel)
    _mod(monkeypatch, "name_matcher", resolve=lambda n, m: ({n[0]: ("BBB", 80)}, []))
    out = pev.evaluate_from_file("h.XLSX", "IN", verbose=False)
    assert list(out["Symbol"]) == ["BBB"] and 3 in seen


# ---- manual input -----------------------------------------------------------------------------


def test_row_from_manual_derives_distance_and_trend_fields():
    rec = {"price": 110.0, "pe": 14.0, "high_52w": 120.0, "low_52w": 80.0, "sma200": 100.0}
    row, pe, fund = pev._row_from_manual({**rec, "ret_1y": 7.0, "liquidity": "ok", "roe": 18.0})
    assert row["Close"] == 110.0 and row["SMA200"] == 100.0 and row["Ret252"] == 7.0
    assert row["Liquidity"] == "ok"
    assert row["PctFromHigh"] == pytest.approx((110 / 120 - 1) * 100)
    assert row["PctFromLow"] == pytest.approx((110 / 80 - 1) * 100)
    assert row["Above200DMA"] is True and row["GoldenCross"] is True
    assert pe == 14.0 and fund == {"roe": 18.0}


def test_row_from_manual_at_exactly_the_200dma_is_not_above():
    row, _, _ = pev._row_from_manual({"price": 100.0, "sma200": 100.0})
    assert row["Above200DMA"] is False


def test_row_from_manual_below_the_200dma():
    row, _, _ = pev._row_from_manual({"price": 90.0, "sma200": 100.0})
    assert row["Above200DMA"] is False and row["GoldenCross"] is False


def test_row_from_manual_with_sparse_input_has_no_derived_fields():
    row, pe, fund = pev._row_from_manual({"ticker": "X"})
    assert set(row) == {"Close", "SMA200", "Ret252", "Liquidity"}
    assert row["Liquidity"] == "" and pe is None and fund is None


def test_row_from_manual_zero_roe_is_still_provided():
    assert pev._row_from_manual({"roe": 0})[2] == {"roe": 0}  # None means missing; 0 does not


def test_row_from_manual_needs_both_price_and_reference_for_a_distance():
    assert "PctFromHigh" not in pev._row_from_manual({"high_52w": 100})[0]
    assert "PctFromLow" not in pev._row_from_manual({"low_52w": 100})[0]
    assert "Above200DMA" not in pev._row_from_manual({"sma200": 100})[0]


def test_evaluate_manual_builds_a_decision_row_per_record(env, capsys):
    out = pev.evaluate_manual(
        [{"ticker": "X", "price": 100.0, "pe": 12.0}, {"price": 5.0}], med_pe=18.0
    )
    assert list(out["Ticker"]) == ["X", "?"]  # missing ticker shown as '?'
    assert list(out["PE"].iloc[[0]]) == [12.0]
    assert out.loc[0, "Flags"] == "F1 F2" and out.loc[0, "Score"] == 55.5
    assert [c[2] for c in env.decide_calls] == [18.0, 18.0]  # caller's median is used
    assert "Manual-input decisions" in capsys.readouterr().out


def test_evaluate_manual_default_median_and_quiet_mode(env, capsys):
    pev.evaluate_manual([{"ticker": "X", "price": 1.0}], verbose=False)
    assert env.decide_calls[0][2] == 20.0 and capsys.readouterr().out == ""


def test_evaluate_manual_with_no_records_prints_nothing(env, capsys):
    out = pev.evaluate_manual([])
    assert out.empty and capsys.readouterr().out == ""


def test_evaluate_manual_passes_roe_as_fundamentals(env):
    pev.evaluate_manual([{"ticker": "X", "price": 1.0, "roe": 22.0}])
    assert env.decide_calls[0][3] == {"roe": 22.0}


# ---- CLI --------------------------------------------------------------------------------------


def _argv(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["portfolio_evaluator.py", *args])


def test_main_requires_some_input(monkeypatch, capsys):
    _argv(monkeypatch)
    assert pev.main() == 1
    assert "provide --tickers, --names or --file" in capsys.readouterr().out


def test_main_tickers_runs_evaluate_with_the_market(env, monkeypatch):
    calls = []
    monkeypatch.setattr(pev, "evaluate", lambda t, m: calls.append((t, m)))
    _argv(monkeypatch, "--tickers", "aaa, bbb", "--market", "US")
    assert pev.main() == 0 and calls == [(["AAA", "BBB"], "US")]


def test_main_names_are_resolved_first(monkeypatch):
    calls = []
    monkeypatch.setattr(pev, "resolve_names", lambda names, m: ["TCS"])
    monkeypatch.setattr(pev, "evaluate", lambda t, m: calls.append((t, m)))
    _argv(monkeypatch, "--names", "Tata Consultancy , Infosys")
    assert pev.main() == 0 and calls == [(["TCS"], "IN")]


def test_main_names_are_split_on_commas_and_stripped(monkeypatch):
    seen = []
    monkeypatch.setattr(pev, "resolve_names", lambda names, m: seen.append(names) or ["X"])
    monkeypatch.setattr(pev, "evaluate", lambda t, m: None)
    _argv(monkeypatch, "--names", " Tata Consultancy , Infosys")
    pev.main()
    assert seen == [["Tata Consultancy", "Infosys"]]


def test_main_fails_when_no_names_resolve(monkeypatch, capsys):
    monkeypatch.setattr(pev, "resolve_names", lambda names, m: [])
    _argv(monkeypatch, "--names", "Nothing Matches")
    assert pev.main() == 1 and "no tickers resolved" in capsys.readouterr().out


def test_main_file_takes_precedence_over_other_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(pev, "evaluate_from_file", lambda f, m: calls.append((f, m)))
    monkeypatch.setattr(pev, "evaluate", lambda *a: calls.append("evaluate"))
    _argv(monkeypatch, "--file", "h.csv", "--tickers", "AAA", "--market", "IN")
    assert pev.main() == 0 and calls == [("h.csv", "IN")]
