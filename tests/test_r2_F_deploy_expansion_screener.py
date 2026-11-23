import builtins
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

import deploy_expansion_screener as de

SRC = Path(de.__file__)


def _co(**k):
    base = {
        "ticker": "T", "name": "N", "country": "C", "sector": "S", "market_cap_b": 1.0,
        "revenue_cagr": 0.0, "capex_cagr": 0.0, "debt_cagr": 0.0, "avg_fcf": 0.0,
        "fcf_margin": 0.0, "oi_margin_change": 0.0, "ni_margin_change": 0.0,
        "roic_change": 0.0, "debt_to_equity": 1.0, "interest_coverage": 3.0,
        "debt_service_coverage": 1.2, "asset_turnover": 1.0, "wc_ratio": 0.3,
        "capex_to_revenue": 0.05, "payout_ratio": 0.1, "price_5yr_cagr": 5.0,
    }  # fmt: skip
    base.update(k)
    return base


@pytest.fixture
def sd():
    return de.ScreenerDeployment()


@pytest.fixture
def redirect(monkeypatch, tmp_path):
    """Send every absolute-path write (csv + summary json) into tmp_path."""
    written = {}
    real_open = builtins.open
    real_to_csv = pd.DataFrame.to_csv

    def to_csv(self, path, *a, **k):
        p = tmp_path / Path(str(path)).name
        written["csv"] = p
        return real_to_csv(self, p, *a, **k)

    def fake_open(path, mode="r", *a, **k):
        if "w" in mode and str(path).startswith("/Users/"):
            p = tmp_path / Path(str(path)).name
            written["json"] = p
            return real_open(p, mode, *a, **k)
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    monkeypatch.setattr(de, "open", fake_open, raising=False)
    return written


def test_load_sources(sd, monkeypatch, capsys):
    monkeypatch.setattr(sd, "_generate_synthetic_universe", lambda: ["x"])
    assert sd.load_company_data() == ["x"]
    assert "LOADING COMPANY DATA (synthetic)" in capsys.readouterr().out
    assert sd.load_company_data("csv") is None  # unimplemented stub
    with pytest.raises(ValueError, match="Unknown data source: db"):
        sd.load_company_data("db")


def test_synthetic_universe_shape_and_determinism(sd):
    a = sd._generate_synthetic_universe()
    assert len(a) == 25000
    assert a[0]["ticker"] == "SYM00000" and a[24999]["ticker"] == "SYM24999"
    assert {c["country"] for c in a} <= {
        "USA",
        "China",
        "Japan",
        "Germany",
        "UK",
        "India",
        "Singapore",
    }
    neg = sum(1 for c in a if c["avg_fcf"] == -100)
    assert 0.15 < neg / len(a) < 0.25
    assert all(0.05 <= c["wc_ratio"] <= 0.25 and 0 <= c["payout_ratio"] <= 0.8 for c in a)
    b = sd._generate_synthetic_universe()
    assert a[100] == b[100]


def test_stage1_each_rule_and_boundary(sd):
    good = _co()
    cases = {
        "de>2": _co(debt_to_equity=2.01),
        "capex_int": _co(capex_to_revenue=0.0049),
        "capex_decl": _co(capex_cagr=-15.01),
        "fcf": _co(avg_fcf=-1),
        "margin": _co(ni_margin_change=-5.01),
    }
    edge = [
        _co(debt_to_equity=2.0), _co(capex_to_revenue=0.005), _co(capex_cagr=-15),
        _co(avg_fcf=0), _co(ni_margin_change=-5),
    ]  # fmt: skip
    out = sd._stage1_prefilter([good, *cases.values(), *edge])
    assert out == [good, *edge]
    # None-valued fields are tolerated for the `or 0` paths
    n = _co(avg_fcf=None, ni_margin_change=None, capex_cagr=None)
    assert sd._stage1_prefilter([n]) == [n]
    # missing keys use defaults: capex_to_revenue default 0 < 0.005 -> rejected
    assert sd._stage1_prefilter([{"ticker": "x"}]) == []


def test_stage1_rule_order_counts(sd, capsys):
    both = _co(debt_to_equity=3.0, avg_fcf=-1)  # counted once, in "de"
    sd._stage1_prefilter([both, _co()])
    out = capsys.readouterr().out
    assert "Input: 2 companies" in out and "Output: 1 companies (50.0%)" in out
    assert "Rejected: 1 (50.0%)" in out


def test_stage2(sd):
    good = _co()
    bad = [
        _co(debt_service_coverage=0.99),
        _co(interest_coverage=1.99),
        _co(payout_ratio=0.81),
    ]
    edge = [_co(debt_service_coverage=1.0), _co(interest_coverage=2.0), _co(payout_ratio=0.8)]
    assert sd._stage2_midfilter([good, *bad, *edge]) == [good, *edge]
    # falsy values fall back to the permissive defaults (1.5 / 5 / 0)
    f = _co(debt_service_coverage=0, interest_coverage=None, payout_ratio=None)
    assert sd._stage2_midfilter([f]) == [f]
    assert sd._stage2_midfilter([{"x": 1}]) == [{"x": 1}]


@pytest.mark.parametrize(
    "score,tier",
    [
        (100, "Tier 1"), (75, "Tier 1"), (74.99, "Tier 2"), (50, "Tier 2"),
        (49.99, "Tier 3"), (25, "Tier 3"), (24.99, "Tier 4"), (0, "Tier 4"),
    ],
)  # fmt: skip
def test_classify_tier(sd, score, tier):
    assert sd._classify_tier(score).startswith(tier)


def _score(sd, **k):
    c = {
        "revenue_cagr": 0, "capex_cagr": 0, "avg_fcf": 0, "roic_change": 0,
        "debt_to_equity": 99, "interest_coverage": 1, "debt_service_coverage": 0.5,
        "asset_turnover": 0.5, "wc_ratio": 0.5,
    }  # fmt: skip
    c.update(k)
    return sd._calculate_11d_score(c)


def test_score_components(sd):
    assert _score(sd) == 0
    # capex acceleration
    assert _score(sd, capex_cagr=10) == pytest.approx(24 * 1.0 * 0 + 24 * 1.0)  # >5, = 10/10*24
    assert _score(sd, capex_cagr=10.01) == pytest.approx(
        24
    )  # >10 flat; timing needs capex < rev*1.5
    assert _score(sd, capex_cagr=5) == 0
    assert _score(sd, capex_cagr=6) == pytest.approx(24 * 0.6)
    # fcf
    assert _score(sd, avg_fcf=5e8) == pytest.approx(11)
    assert _score(sd, avg_fcf=2e9) == pytest.approx(22)
    assert _score(sd, avg_fcf=-5) == 0
    # revenue
    assert _score(sd, revenue_cagr=10) == pytest.approx(19 * 1.0)
    assert _score(sd, revenue_cagr=10.5) == pytest.approx(19)
    assert _score(sd, revenue_cagr=5) == 0
    assert _score(sd, revenue_cagr=7) == pytest.approx(19 * 0.7)
    # roic
    assert _score(sd, roic_change=2) == pytest.approx(10)
    assert _score(sd, roic_change=2.5) == pytest.approx(10)
    assert _score(sd, roic_change=1) == pytest.approx(5)
    assert _score(sd, roic_change=-1) == 0
    # dsc (and sustainability)
    assert _score(sd, debt_service_coverage=1.5) == pytest.approx(10 * 1.0)
    assert _score(sd, debt_service_coverage=1.6) == pytest.approx(10 + 8)
    assert _score(sd, debt_service_coverage=1.0) == 0
    # debt expansion
    assert _score(sd, debt_to_equity=0) == 0 + 0  # `or` path irrelevant: 0 not in (0,1.5)
    assert _score(sd, debt_to_equity=0.75) == pytest.approx(5)
    assert _score(sd, debt_to_equity=1.5) == 0
    # asset turnover
    assert _score(sd, asset_turnover=1) == 0
    assert _score(sd, asset_turnover=1.5) == pytest.approx(7 * 0.75)
    assert _score(sd, asset_turnover=4) == pytest.approx(7)
    # leverage health
    assert _score(sd, interest_coverage=5) == 0
    assert _score(sd, interest_coverage=5.1) == pytest.approx(2)
    # wc
    assert _score(sd, wc_ratio=0.2) == 0
    assert _score(sd, wc_ratio=0.19) == pytest.approx(4)
    # timing
    assert _score(sd, capex_cagr=1, revenue_cagr=1) == pytest.approx(4)  # 1 < 1.5
    assert _score(sd, capex_cagr=1.5, revenue_cagr=1) == 0  # 1.5 !< 1.5


def test_score_falsy_defaults_and_cap(sd):
    # None / 0 for icr, dsc, asset_turnover, wc fall back to defaults
    s = sd._calculate_11d_score({"interest_coverage": 0, "debt_service_coverage": None,
                                 "asset_turnover": 0, "wc_ratio": None, "debt_to_equity": 5})  # fmt: skip
    # dsc->1.5 (not >1.5): 10*(1.5/1.5)=10 ; icr->5 no; at->1 no; wc->0.1: +4
    assert s == pytest.approx(14)
    big = _co(capex_cagr=50, revenue_cagr=50, avg_fcf=5e9, roic_change=9, debt_service_coverage=3,
              debt_to_equity=0.1, asset_turnover=5, interest_coverage=9, wc_ratio=0.1)  # fmt: skip
    assert sd._calculate_11d_score(big) == 100


def test_stage3_and_run_screening(sd, capsys):
    cos = [
        _co(ticker="LO"),
        _co(ticker="HI", capex_cagr=20, revenue_cagr=20, avg_fcf=5e9, roic_change=5,
            debt_service_coverage=3, debt_to_equity=0.1, interest_coverage=9),
        _co(ticker="DE", debt_to_equity=3),  # s1 reject
        _co(ticker="PAY", payout_ratio=0.9),  # s2 reject
        _co(ticker="FCF", avg_fcf=-1),  # s1 reject
    ]  # fmt: skip
    cos[0]["avg_fcf"] = 1
    df = sd.run_screening(cos)
    assert df["ticker"].tolist() == ["HI", "LO"]
    assert sd.results_summary == {
        "total_input": 5, "stage1_output": 3, "stage2_output": 2, "final_output": 2,
    }  # fmt: skip
    assert df.iloc[0]["tier"].startswith("Tier 1") and df.iloc[0]["de_ratio"] == 0.1
    assert set(df.columns) >= {"expansion_score", "price_5yr_cagr", "dsc_ratio", "fcf_margin"}
    out = capsys.readouterr().out
    assert "STAGE 1" in out and "STAGE 3" in out and "2 companies scored" in out


def test_stage3_missing_fields(sd):
    df = sd._stage3_fullscore([{}])
    r = df.iloc[0]
    assert r["ticker"] == "N/A" and r["country"] == "N/A" and r["market_cap_b"] == 0
    assert pd.isna(r["revenue_cagr"]) or r["revenue_cagr"] is None


def _results():
    rows = []
    for i, (t, tier) in enumerate(
        [("A", "Tier 1 (Aggressive Expander)"), ("B", "Tier 4 (Passive/Mature)"),
         ("C", "Tier 4 (Passive/Mature)")]  # fmt: skip
    ):
        rows.append(
            {"ticker": t, "name": t, "country": "X", "sector": "Tech" if i < 2 else "Energy",
             "market_cap_b": 1, "expansion_score": 90 - 40 * i, "tier": tier,
             "revenue_cagr": 5.0 if i else 0, "capex_cagr": 2.0, "debt_cagr": 1.0,
             "fcf_margin": 1.0, "roic_change": 1.0, "de_ratio": 1.0, "dsc_ratio": 1.0,
             "price_5yr_cagr": 12.0}  # fmt: skip
        )
    return pd.DataFrame(rows)


def test_generate_reports(sd, redirect, capsys):
    out = sd.generate_reports(_results())
    assert out is not None and len(out) == 3
    o = capsys.readouterr().out
    assert "Tier 1 (Aggressive Expander)" in o and "1 ( 33.3%)" in o
    assert "Tier 4 (Passive/Mature)" in o and "2 ( 66.7%)" in o
    assert f"   {'Tier 2 (Strong Expander)':30s}: {0:6,} ({0.0:5.1f}%)" in o
    assert "Tech" in o and "Energy" in o
    # falsy 0 revenue -> N/A, nonzero -> formatted
    lines = [ln for ln in o.splitlines() if ln.startswith("1  ")]
    assert "N/A" in lines[0] and "12.0%" in lines[0]
    assert "5.0%" in [ln for ln in o.splitlines() if ln.startswith("2  ")][0]
    saved = pd.read_csv(redirect["csv"])
    assert saved["ticker"].tolist() == ["A", "B", "C"]
    assert redirect["csv"].name.startswith("expansion_screening_results_")


def test_summary(sd):
    sd.results_summary = {"total_input": 8, "final_output": 3}
    s = sd.create_deployment_summary(_results())
    assert s["pass_rate"] == "37.5%" and s["total_candidates"] == 3
    assert s["total_companies_screened"] == 8
    assert s["tier_distribution"] == {
        "Tier 4 (Passive/Mature)": 2,
        "Tier 1 (Aggressive Expander)": 1,
    }
    assert s["average_score"] == pytest.approx((90 + 50 + 10) / 3)
    assert [r["ticker"] for r in s["top_10_candidates"]] == ["A", "B", "C"]
    assert s["deployment_time"] == sd.deployment_time.isoformat()
    sd.results_summary = {}
    assert sd.create_deployment_summary(_results())["pass_rate"] == "0.0%"
    json.dumps(s, default=str)


def test_deploy_and_main(sd, redirect, monkeypatch, capsys):
    cos = [_co(ticker="HI", capex_cagr=20, revenue_cagr=20, avg_fcf=5e9, roic_change=5,
               debt_service_coverage=3, debt_to_equity=0.1, interest_coverage=9),
           _co(ticker="LO", avg_fcf=1)]  # fmt: skip
    monkeypatch.setattr(sd, "_generate_synthetic_universe", lambda: cos)
    res = sd.deploy()
    assert res["ticker"].tolist() == ["HI", "LO"]
    summ = json.loads(redirect["json"].read_text())
    assert summ["total_companies_screened"] == 2 and summ["pass_rate"] == "100.0%"
    o = capsys.readouterr().out
    assert "DEPLOYMENT SUCCESSFUL" in o and "Screened: 2 companies" in o
    assert "Pass rate: 100.0%" in o and "Qualified: 2 candidates" in o


def test_main_block(redirect, monkeypatch, capsys):
    import numpy as np

    # shrink the 25k synthetic universe so the script-run stays fast
    monkeypatch.setattr(np.random, "seed", lambda s: None)
    real_choice = np.random.choice
    monkeypatch.setattr(np.random, "choice", lambda a, *k, **kw: a[0])
    runpy.run_path(str(SRC), run_name="__main__")
    out = capsys.readouterr().out
    assert "DEPLOYMENT COMPLETE" in out and "PRODUCTION READY" in out
    assert real_choice is not None
