import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import global_20country_universe_screener as g

SRC = Path(g.__file__)


@pytest.fixture
def sc():
    return g.Global20CountryScreener()


def _co(**k):
    base = {
        "ticker": "T", "name": "N", "country": "A", "region": "R", "sector": "S",
        "market_cap_b": 1.0, "revenue_cagr": 0.0, "capex_cagr": 0.0, "avg_fcf": 0.0,
        "fcf_margin": 1.0, "roic_change": 0.0, "debt_to_equity": 1.0,
        "interest_coverage": 3.0, "debt_service_coverage": 1.2, "asset_turnover": 1.0,
        "wc_ratio": 0.3, "capex_to_revenue": 0.05, "payout_ratio": 0.1, "price_5yr_cagr": 5.0,
        "ni_margin_change": 0.0,
    }  # fmt: skip
    base.update(k)
    return base


def test_config_and_define_countries(sc):
    # NB: docs say "20 countries, 50,000+ companies"; the table defines 19 / 35,200
    assert len(g.COUNTRIES_CONFIG) == 19
    assert sum(c["companies"] for c in g.COUNTRIES_CONFIG.values()) == 35200
    assert {c["region"] for c in g.COUNTRIES_CONFIG.values()} == {
        "North America", "Europe", "Asia-Pacific", "Emerging Markets",
    }  # fmt: skip
    d = sc._define_countries()
    assert d == g.COUNTRIES_CONFIG and d is not g.COUNTRIES_CONFIG
    d["USA"]["sectors"].append("X")
    assert "X" not in g.COUNTRIES_CONFIG["USA"]["sectors"]  # deep copy
    assert sc.total_companies == 0 and sc.total_candidates == 0


def _small(sc):
    sc.countries_config = {
        "China": {"region": "Asia-Pacific", "companies": 3, "sectors": ["a", "b"],
                  "major_exchanges": ["X1"]},
        "Brazil": {"region": "Emerging Markets", "companies": 2, "sectors": ["c"],
                   "major_exchanges": ["B3"]},
    }  # fmt: skip


def test_generate_universe_structure(sc, capsys):
    _small(sc)
    cos = sc.generate_global_universe()
    assert len(cos) == 5 and sc.total_companies == 5
    assert [c["ticker"] for c in cos] == [
        "CHINA00000", "CHINA00001", "CHINA00002", "BRAZIL00000", "BRAZIL00001",
    ]  # fmt: skip
    assert cos[0]["name"] == "China Company 0" and cos[3]["region"] == "Emerging Markets"
    assert {c["sector"] for c in cos[:3]} <= {"a", "b"} and cos[3]["exchange"] == "B3"
    assert all(0.05 <= c["wc_ratio"] <= 0.3 and 0.01 <= c["capex_to_revenue"] <= 0.18 for c in cos)
    out = capsys.readouterr().out
    assert "Total Companies: 5" in out and "Countries: 2" in out
    assert "Brazil               Emerging Markets     2" in out
    sc2 = g.Global20CountryScreener()
    _small(sc2)
    assert sc2.generate_global_universe()[1] == cos[1]  # seeded


def test_generate_universe_parameter_shifts(sc, monkeypatch):
    _small(sc)
    monkeypatch.setattr(np.random, "normal", lambda mean, sd: mean)
    monkeypatch.setattr(np.random, "random", lambda: 0.5)
    monkeypatch.setattr(np.random, "lognormal", lambda m, s: 7.0)
    cos = sc.generate_global_universe()
    china, brazil = cos[0], cos[3]
    assert china["revenue_cagr"] == 5 and brazil["revenue_cagr"] == 4
    assert china["capex_cagr"] == 2 and brazil["capex_cagr"] == 2.5
    assert china["price_5yr_cagr"] == 8 and brazil["price_5yr_cagr"] == 10
    assert china["avg_fcf"] == 7.0
    monkeypatch.setattr(np.random, "random", lambda: 0.2)  # not > 0.2 -> negative FCF flag
    assert sc.generate_global_universe()[0]["avg_fcf"] == -50


def test_stage1_boundaries(sc):
    good = _co()
    edge = [
        _co(debt_to_equity=2.0), _co(capex_to_revenue=0.005), _co(avg_fcf=0),
        _co(ni_margin_change=-5), _co(avg_fcf=None), _co(ni_margin_change=None),
    ]  # fmt: skip
    bad = [
        _co(debt_to_equity=2.01), _co(capex_to_revenue=0.0049), _co(avg_fcf=-1),
        _co(ni_margin_change=-5.01),
    ]  # fmt: skip
    assert sc._stage1_filter([good, *bad, *edge]) == [good, *edge]
    assert sc._stage1_filter([{"x": 1}]) == []  # default capex_to_revenue 0 fails


def test_stage2_boundaries(sc):
    good = _co()
    edge = [_co(debt_service_coverage=1.0), _co(interest_coverage=2.0), _co(payout_ratio=0.8)]
    bad = [_co(debt_service_coverage=0.99), _co(interest_coverage=1.99), _co(payout_ratio=0.81)]
    assert sc._stage2_filter([good, *bad, *edge]) == [good, *edge]
    assert sc._stage2_filter([{"x": 1}]) == [{"x": 1}]
    z = _co(debt_service_coverage=0, interest_coverage=0, payout_ratio=None)
    assert sc._stage2_filter([z]) == [z]


@pytest.mark.parametrize(
    "score,tier",
    [(75, "Tier 1"), (74.9, "Tier 2"), (50, "Tier 2"), (49.9, "Tier 3"), (25, "Tier 3"),
     (24.9, "Tier 4")],
)  # fmt: skip
def test_tiers(sc, score, tier):
    assert sc._classify_tier(score) == tier


def _score(sc, **k):
    c = {"revenue_cagr": 0, "capex_cagr": 0, "avg_fcf": 0, "roic_change": 0,
         "debt_to_equity": 99, "debt_service_coverage": 0.5, "asset_turnover": 0.5,
         "wc_ratio": 0.5, "interest_coverage": 0}  # fmt: skip
    c.update(k)
    return sc._calculate_11d_score(c)


def test_score_components(sc):
    assert _score(sc) == 0
    assert _score(sc, capex_cagr=6) == pytest.approx(24 * 0.6)
    assert _score(sc, capex_cagr=5) == 0
    assert _score(sc, capex_cagr=11) == pytest.approx(24)
    assert _score(sc, avg_fcf=5e8) == pytest.approx(11)
    assert _score(sc, avg_fcf=3e9) == pytest.approx(22)
    assert _score(sc, revenue_cagr=7) == pytest.approx(13.3)
    assert _score(sc, revenue_cagr=11) == pytest.approx(19)
    assert _score(sc, roic_change=1) == pytest.approx(5)
    assert _score(sc, roic_change=3) == pytest.approx(10)
    assert _score(sc, debt_service_coverage=1.2) == pytest.approx(8)
    assert _score(sc, debt_service_coverage=1.0) == 0
    assert _score(sc, debt_service_coverage=2) == pytest.approx(18)  # 10 + sustainability 8
    assert _score(sc, debt_to_equity=0.75) == pytest.approx(5)
    assert _score(sc, debt_to_equity=1.5) == 0
    assert _score(sc, asset_turnover=1.5) == pytest.approx(5.25)
    assert _score(sc, asset_turnover=9) == pytest.approx(7)
    assert _score(sc, wc_ratio=0.19) == pytest.approx(4)
    assert _score(sc, wc_ratio=0.2) == 0
    assert _score(sc, capex_cagr=1, revenue_cagr=1) == pytest.approx(4)
    assert _score(sc, capex_cagr=1.5, revenue_cagr=1) == 0
    assert _score(sc, wc_ratio=0, debt_service_coverage=0) == pytest.approx(
        4 + 10
    )  # falsy defaults
    capped = _score(sc, capex_cagr=50, revenue_cagr=50, avg_fcf=9e9, roic_change=9,
                    debt_service_coverage=3, debt_to_equity=0.1, asset_turnover=5,
                    wc_ratio=0.1)  # fmt: skip
    assert capped == 100


def test_interest_coverage_threshold(sc):
    assert _score(sc, interest_coverage=9) == 2  # clearly above 5 earns the +2
    assert _score(sc, interest_coverage=0) == 0  # falsy -> default 5 -> not > 5


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug global_20country_universe_screener.py:314: "
    "`get('interest_coverage', 5) or 5 > 5` parses as `x or (5 > 5)`, so any non-zero "
    "interest coverage (even 1.0) earns the +2 leverage-health points",
)
def test_interest_coverage_threshold_low_value_bug(sc):
    assert _score(sc, interest_coverage=1.0) == 0


def test_run_screening(sc, capsys):
    cos = [
        _co(ticker="HI", capex_cagr=20, revenue_cagr=20, avg_fcf=5e9, roic_change=5,
            debt_service_coverage=3, debt_to_equity=0.1, interest_coverage=9),
        _co(ticker="LO", avg_fcf=1),
        _co(ticker="S1", debt_to_equity=5),
        _co(ticker="S2", payout_ratio=0.99),
    ]  # fmt: skip
    sc.total_companies = 4
    df = sc.run_global_screening(cos)
    assert df["ticker"].tolist() == ["HI", "LO"] and sc.total_candidates == 2
    assert df.iloc[0]["tier"] == "Tier 1"
    out = capsys.readouterr().out
    assert "Input: 4 | Output: 3" in out and "Input: 3 | Output: 2" in out
    assert "Input: 2 | Output: 2" in out and "(4 companies)" in out
    assert "region" in df.columns and "exchange" not in df.columns


def _results():
    rows = [
        ("A1", "A", "R1", "Tier 1", 90.0, 5.0, 2.0, 11.0),
        ("A2", "A", "R1", "Tier 2", 60.0, 0, 0, 9.0),
        ("B1", "B", "R2", "Tier 3", 30.0, 3.0, 1.0, 5.0),
        ("B2", "B", "R2", "Tier 4", 10.0, 1.0, 1.0, 7.0),
        ("B3", "B", "R2", "Tier 4", 0.0, 1.0, 1.0, 7.0),
    ]
    return pd.DataFrame(
        [
            {"ticker": t, "country": c, "region": r, "tier": ti, "expansion_score": s,
             "revenue_cagr": rv, "capex_cagr": cx, "price_5yr_cagr": p}
            for t, c, r, ti, s, rv, cx, p in rows
        ]
    )  # fmt: skip


def test_regional_analysis(sc, capsys):
    stats = sc.generate_regional_analysis(_results())
    assert stats == {
        "R1": {"total": 2, "tier1": 1, "tier2": 1, "avg_score": 75.0},
        "R2": {"total": 3, "tier1": 0, "tier2": 0, "avg_score": pytest.approx(40 / 3)},
    }
    out = capsys.readouterr().out
    assert "BY COUNTRY (Top 10)" in out
    assert "B               3" in out and "A               2               1          75.0" in out


def test_regional_top10_cap(sc, capsys):
    rows = [
        {"ticker": f"T{i}", "country": f"C{i}", "region": "R", "tier": "Tier 1",
         "expansion_score": 1.0, "price_5yr_cagr": 1.0} for i in range(12)
    ]  # fmt: skip
    sc.generate_regional_analysis(pd.DataFrame(rows))
    out = capsys.readouterr().out.split("BY COUNTRY")[1]
    assert sum(1 for ln in out.splitlines() if ln.startswith("C") and ln[1].isdigit()) == 10


def test_final_report(sc, capsys):
    sc.total_companies, sc.total_candidates = 20, 5
    r = sc.generate_final_report(_results())
    assert len(r) == 5
    out = capsys.readouterr().out
    assert "Pass rate: 25.0%" in out
    assert "Tier 1 (Aggressive): 1" in out and "Tier 4 (Passive): 2" in out
    assert "Tier 2 (Strong): 1" in out and "Tier 3 (Moderate): 1" in out
    row_a2 = [ln for ln in out.splitlines() if ln.startswith("2    ")][0]
    assert row_a2.split().count("N/A") == 2  # zero revenue/capex shown as N/A
    row_a1 = [ln for ln in out.splitlines() if ln.startswith("1    ")][0]
    assert "5.0%" in row_a1 and "2.0%" in row_a1 and "A1" in row_a1


def test_final_report_top15_cap(sc, capsys):
    sc.total_companies, sc.total_candidates = 100, 20
    rows = [
        {"ticker": f"T{i:02d}", "country": "C", "region": "R", "tier": "Tier 1",
         "expansion_score": 100 - i, "revenue_cagr": 1.0, "capex_cagr": 1.0} for i in range(20)
    ]  # fmt: skip
    sc.generate_final_report(pd.DataFrame(rows))
    out = capsys.readouterr().out
    assert "T14" in out and "T15" not in out


@pytest.fixture
def csvs(monkeypatch, tmp_path):
    written = []
    real = pd.DataFrame.to_csv

    def to_csv(self, path, *a, **k):
        p = tmp_path / Path(str(path)).name
        written.append(p)
        return real(self, p, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    return written


def test_deploy_global(sc, csvs, capsys):
    _small(sc)
    res, stats = sc.deploy_global()
    assert sc.total_companies == 5 and set(stats) <= {"Asia-Pacific", "Emerging Markets"}
    names = [p.name for p in csvs]
    assert names[0].startswith("global_screening_20countries_")
    per_country = {n.split("_")[1] for n in names[1:]}
    assert per_country == set(res["country"].unique())
    allrows = pd.read_csv(csvs[0])
    assert len(allrows) == len(res) and not allrows.empty or res.empty
    for p in csvs[1:]:
        d = pd.read_csv(p)
        assert d["expansion_score"].is_monotonic_decreasing
    assert "GLOBAL DEPLOYMENT COMPLETE" in capsys.readouterr().out


def test_main_block(csvs, monkeypatch, capsys):
    small = {
        "Brazil": {"region": "Emerging Markets", "companies": 300, "sectors": ["c"],
                   "major_exchanges": ["B3"]},
    }  # fmt: skip
    import copy

    monkeypatch.setattr(copy, "deepcopy", lambda x: small)
    runpy.run_path(str(SRC), run_name="__main__")
    out = capsys.readouterr().out
    assert "GLOBAL SCREENING LIVE" in out and "Coverage: 20 countries, 300 companies" in out
