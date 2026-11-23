import runpy
from pathlib import Path

import pandas as pd
import pytest

import phased_expansion_screener_11d as ph

SRC = Path(ph.__file__)


@pytest.fixture
def ps():
    return ph.PhasedExpansionScreener()


def _co(**k):
    base = {
        "ticker": "T", "name": "N", "country": "C", "sector": "S", "market_cap_b": 2.0,
        "revenue_cagr": 0.0, "capex_cagr": 0.0, "debt_cagr": 0.0, "avg_fcf": 0.0,
        "fcf_margin": 1.0, "oi_margin_change": 0.0, "ni_margin_change": 0.0,
        "roic_change": 0.0, "debt_to_equity": 1.0, "interest_coverage": 3.0,
        "debt_service_coverage": 1.2, "asset_turnover": 1.0, "wc_ratio": 0.3,
        "capex_to_revenue": 0.05, "payout_ratio": 0.1, "price_5yr_cagr": 5.0,
    }  # fmt: skip
    base.update(k)
    return base


def test_weights(ps):
    w = ps.define_criteria_weights()
    assert len(w) == 11 and sum(c["weight"] for c in w.values()) == 120
    by_stage = {s: sum(c["weight"] for c in w.values() if c["stage"] == s) for s in (1, 2, 3)}
    assert by_stage == {1: 66, 2: 29, 3: 25}
    assert w["capex_acceleration"]["weight"] == 24
    assert ps.stage1_filter == {} and ps.stage2_filter == {} and ps.stage3_results == {}


def test_stage1_each_rule(ps, capsys):
    good = _co()
    cos = [
        good,
        _co(debt_to_equity=2.01),
        _co(capex_to_revenue=0.0049),
        _co(capex_cagr=-15.01),
        _co(avg_fcf=-1),
        _co(avg_fcf=None),
        _co(net_margin=-5.01),
        _co(debt_to_equity=3, avg_fcf=-1),  # counted once, under debt
    ]  # fmt: skip
    out = ps.stage1_prefilter(cos)
    assert out == [good]
    assert ps.stage1_filter == {
        "failed_debt": 2, "failed_capex": 2, "failed_fcf": 2, "failed_profitability": 1,
    }  # fmt: skip
    o = capsys.readouterr().out
    assert "Stage 1 Weight: 66/100" in o
    assert "Passed: 1 / 8" in o and "Failed: 7 (87.5%)" in o


def test_stage1_boundaries(ps):
    edge = [
        _co(debt_to_equity=2.0), _co(capex_to_revenue=0.005), _co(capex_cagr=-15),
        _co(avg_fcf=0), _co(net_margin=-5), _co(net_margin=None), _co(capex_cagr=None),
    ]  # fmt: skip
    assert ps.stage1_prefilter(edge) == edge
    # missing keys: capex_to_revenue default 0 -> rejected; None fcf rejected
    assert ps.stage1_prefilter([{"ticker": "x"}]) == []


def test_stage2(ps, capsys):
    good = _co()
    cos = [
        good,
        _co(debt_service_coverage=0.99),
        _co(interest_coverage=1.99),
        _co(payout_ratio=0.81),
        _co(debt_service_coverage=0.5, interest_coverage=1, payout_ratio=0.9),  # one count only
    ]  # fmt: skip
    assert ps.stage2_midfilter(cos) == [good]
    assert ps.stage2_filter == {
        "failed_sustainability": 2, "failed_leverage": 1, "failed_reinvestment": 1,
    }  # fmt: skip
    assert "Failed: 4 (80.0%)" in capsys.readouterr().out
    edge = [
        _co(debt_service_coverage=1.0), _co(interest_coverage=2.0), _co(payout_ratio=0.8),
        _co(debt_service_coverage=0), _co(interest_coverage=None), _co(payout_ratio=0),
    ]  # fmt: skip
    assert ps.stage2_midfilter(edge) == edge
    assert ps.stage2_midfilter([{"x": 1}]) == [{"x": 1}]


def _s(ps, **k):
    c = {
        "revenue_cagr": 0, "capex_cagr": 0, "avg_fcf": 0, "oi_margin_change": 0,
        "ni_margin_change": 0, "roic_change": 0, "debt_to_equity": 99,
        "interest_coverage": 1, "debt_service_coverage": 0.5, "asset_turnover": 0.5,
        "wc_ratio": 0.5,
    }  # fmt: skip
    c.update(k)
    return ps.calculate_11d_score(c)


def test_score_debt_expansion(ps):
    assert _s(ps, debt_to_equity=0.75) == pytest.approx(5)
    assert _s(ps, debt_to_equity=0.0001) == pytest.approx(10 * (1.5 - 0.0001) / 1.5)
    assert _s(ps, debt_to_equity=1.5) == pytest.approx(5)  # second branch, 1.5 <2
    assert _s(ps, debt_to_equity=1.99) == pytest.approx(5)
    assert _s(ps, debt_to_equity=2.0) == 0
    assert _s(ps, debt_to_equity=0) == 0


def test_score_capex_and_revenue(ps):
    assert _s(ps, capex_cagr=11) == pytest.approx(24)
    assert _s(ps, capex_cagr=10) == pytest.approx(24)  # >5 branch: 24*1.0
    assert _s(ps, capex_cagr=6) == pytest.approx(14.4)
    assert _s(ps, capex_cagr=5) == pytest.approx(24 * 1.0 * 0.5)  # last branch
    assert _s(ps, capex_cagr=2.5) == pytest.approx(6)
    assert _s(ps, capex_cagr=-1) == 0
    assert _s(ps, revenue_cagr=11) == pytest.approx(19)
    assert _s(ps, revenue_cagr=6) == pytest.approx(11.4)
    assert _s(ps, revenue_cagr=5) == pytest.approx(9.5)
    assert _s(ps, revenue_cagr=1) == pytest.approx(1.9)
    assert _s(ps, revenue_cagr=-1) == 0


def test_score_profitability(ps):
    assert _s(ps, roic_change=2.01) == 10
    assert _s(ps, roic_change=2) == 0
    assert _s(ps, ni_margin_change=0.1) == pytest.approx(7)
    assert _s(ps, oi_margin_change=0.1) == pytest.approx(5)
    assert _s(ps, ni_margin_change=0.1, oi_margin_change=1) == pytest.approx(7)  # elif chain
    assert _s(ps, roic_change=3, ni_margin_change=1) == 10


def test_score_dsc_icr_fcf(ps):
    assert _s(ps, debt_service_coverage=2) == pytest.approx(8 + 10)
    assert _s(ps, debt_service_coverage=1.5) == pytest.approx(8 * 1.0 + 10 * 1.0)
    assert _s(ps, debt_service_coverage=1.2) == pytest.approx(8 * 0.8 + 10 * 0.8)
    assert _s(ps, debt_service_coverage=1.0) == 0
    assert _s(ps, debt_service_coverage=0) == pytest.approx(18)  # falsy -> default 1.5
    assert _s(ps, interest_coverage=5.01) == pytest.approx(2)
    assert _s(ps, interest_coverage=5) == pytest.approx(2 * 5 / 5)
    assert _s(ps, interest_coverage=4) == pytest.approx(1.6)
    assert _s(ps, interest_coverage=3) == 0
    assert _s(ps, interest_coverage=0) == pytest.approx(2)  # falsy -> default 5 -> 2*(5/5)
    assert _s(ps, avg_fcf=5e8) == pytest.approx(11)
    assert _s(ps, avg_fcf=2e9) == pytest.approx(22)
    assert _s(ps, avg_fcf=-5) == 0


def test_score_timing_asset_wc(ps):
    assert _s(ps, capex_cagr=2, revenue_cagr=2) == pytest.approx(
        24 * 0.4 * 0.5 + 19 * 0.4 * 0.5 + 4
    )
    assert _s(ps, capex_cagr=3, revenue_cagr=2) == pytest.approx(
        24 * 0.6 * 0.5 + 19 * 0.4 * 0.5
    )  # 3 < 2*1.5 is False -> no timing bonus
    assert _s(ps, asset_turnover=1) == 0
    assert _s(ps, asset_turnover=1.5) == pytest.approx(5.25)
    assert _s(ps, asset_turnover=10) == pytest.approx(7)
    assert _s(ps, asset_turnover=0) == 0  # falsy -> 1 -> not > 1
    assert _s(ps, wc_ratio=0.19) == pytest.approx(4)
    assert _s(ps, wc_ratio=0.2) == 0
    assert _s(ps, wc_ratio=0) == pytest.approx(4)  # falsy -> 0.1


def test_score_cap_and_defaults(ps):
    big = {"capex_cagr": 50, "revenue_cagr": 50, "avg_fcf": 9e9, "roic_change": 9,
           "debt_to_equity": 0.1, "interest_coverage": 9, "debt_service_coverage": 3,
           "asset_turnover": 5, "wc_ratio": 0.1}  # fmt: skip
    assert ps.calculate_11d_score(big) == 100
    # empty company: dsc->1.5 (8 + 10), icr->5 (2*1), wc->0.1 (4)
    assert ps.calculate_11d_score({}) == pytest.approx(8 + 10 + 2 + 4)
    assert ps.calculate_11d_score({"debt_cagr": None}) == pytest.approx(24)


@pytest.mark.parametrize(
    "score,tier",
    [(75, "Tier 1"), (74.99, "Tier 2"), (50, "Tier 2"), (49.99, "Tier 3"), (25, "Tier 3"),
     (24.99, "Tier 4")],
)  # fmt: skip
def test_tier(ps, score, tier):
    assert ps.classify_tier(score).startswith(tier)


def _hi(**k):
    return _co(capex_cagr=20, revenue_cagr=20, avg_fcf=5e9, roic_change=5,
               debt_service_coverage=3, debt_to_equity=0.1, interest_coverage=9,
               asset_turnover=3, wc_ratio=0.1, **k)  # fmt: skip


def test_screen_companies_end_to_end(ps, capsys):
    cos = [
        _hi(ticker="HI", price_5yr_cagr=30.0),
        _co(ticker="LO", avg_fcf=1.0, price_5yr_cagr=0.0, capex_cagr=0.0, revenue_cagr=0.0),
        _co(ticker="MID", avg_fcf=1e9, capex_cagr=6, revenue_cagr=6, price_5yr_cagr=10.0),
        _co(ticker="DEBT", debt_to_equity=5),
        _co(ticker="PAY", payout_ratio=0.95),
    ]  # fmt: skip
    df = ps.screen_companies(cos)
    assert df["ticker"].tolist() == ["HI", "MID", "LO"]
    assert df["expansion_score"].is_monotonic_decreasing
    assert df.iloc[0]["tier"].startswith("Tier 1") and df.iloc[2]["tier"].startswith("Tier")
    assert df.iloc[0]["company_name"] == "N" and df.iloc[0]["market_cap_usd_b"] == 2.0
    assert ps.stage3_results is df
    assert ps.stage1_filter["failed_debt"] == 1 and ps.stage2_filter["failed_reinvestment"] == 1
    o = capsys.readouterr().out
    assert "Total companies to screen: 5" in o and "on 3 candidates" in o
    assert "Stage 1 rejections: 1" in o and "Stage 2 rejections: 1" in o
    assert "Final candidates: 3" in o and "Overall pass rate: 0.01%" in o
    assert "Correlation (expansion score vs price CAGR)" in o
    row_lo = [ln for ln in o.splitlines() if ln.startswith("3    ")][0]
    assert "N/A" in row_lo  # 0.0 growth rates shown as N/A


def _summary_frame(price, score=(90.0, 60.0, 30.0)):
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C"], "expansion_score": list(score),
            "tier": ["Tier 1 (Aggressive Expander)", "Tier 2 (Strong Expander)",
                     "Tier 3 (Moderate Expander)"],
            "revenue_cagr": [1.0, 2.0, 3.0], "capex_cagr": [1.0, 2.0, 3.0],
            "price_5yr_cagr": price, "sector": ["X", "X", "Y"],
        }
    )  # fmt: skip


def test_summary_interpretation_and_tiers(ps, capsys):
    ps.print_screening_summary(_summary_frame([1.0, 2.0, 3.0]))
    o = capsys.readouterr().out
    assert "Correlation (expansion score vs price CAGR): -1.000" in o or "1.000" in o
    assert "Interpretation: Strong signal" in o
    assert f"   {'Tier 4 (Passive/Mature)':30s}: {0:6,} (  0.0%)" in o
    assert f"   {'Tier 1 (Aggressive Expander)':30s}: {1:6,} ( 33.3%)" in o
    assert "Avg expansion score: 60.0/100" in o and "Avg price 5-year CAGR: 2.0%" in o
    assert "X                             :     2" in o


@pytest.mark.parametrize(
    "price,label",
    [([3.0, 1.0, 2.0], "Weak"), ([1.0, 3.0, 2.0], "Weak")],
)
def test_summary_weak_labels(ps, capsys, price, label):
    f = _summary_frame(price, score=(1.0, 2.0, 3.0))
    r = f["expansion_score"].corr(f["price_5yr_cagr"])
    ps.print_screening_summary(f)
    expect = "Strong" if abs(r) > 0.3 else "Moderate" if abs(r) > 0.15 else "Weak"
    assert f"Interpretation: {expect} signal" in capsys.readouterr().out


def test_summary_moderate_and_boundaries(ps, capsys):
    n = 200
    import numpy as np

    x = np.arange(n, dtype=float)
    xs = (x - x.mean()) / x.std()
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, n)
    z = z - np.polyval(np.polyfit(xs, z, 1), xs)
    zs = (z - z.mean()) / z.std()
    for target, label in [(0.2, "Moderate"), (0.1, "Weak"), (0.5, "Strong")]:
        y = target * xs + np.sqrt(1 - target**2) * zs
        f = pd.DataFrame(
            {"ticker": "A", "expansion_score": x, "tier": "Tier 1 (Aggressive Expander)",
             "revenue_cagr": 1.0, "capex_cagr": 1.0, "price_5yr_cagr": y, "sector": "X"}
        )  # fmt: skip
        ps.print_screening_summary(f)
        assert f"Interpretation: {label} signal" in capsys.readouterr().out


def test_summary_no_valid_price(ps, capsys):
    f = _summary_frame([np_nan := float("nan")] * 3)
    ps.print_screening_summary(f)
    o = capsys.readouterr().out
    assert "Correlation (expansion" not in o and "PRICE CORRELATION" in o
    assert np_nan != np_nan


def test_summary_empty_results(ps, capsys):
    empty = _summary_frame([1.0, 2.0, 3.0]).iloc[0:0]
    ps.print_screening_summary(empty)  # pct guarded against /0
    assert "(  0.0%)" in capsys.readouterr().out


def test_summary_top15(ps, capsys):
    rows = [
        {"ticker": f"T{i:02d}", "expansion_score": float(i), "tier": "Tier 4 (Passive/Mature)",
         "revenue_cagr": 1.0, "capex_cagr": 1.0, "price_5yr_cagr": 1.0, "sector": "X"}
        for i in range(20)
    ]  # fmt: skip
    ps.print_screening_summary(pd.DataFrame(rows))
    o = capsys.readouterr().out
    assert "T19" in o and "T05" in o and "T04" not in o


def test_save_results(ps, monkeypatch, tmp_path, capsys):
    seen = {}
    real = pd.DataFrame.to_csv

    def to_csv(self, path, *a, **k):
        seen["path"] = str(path)
        return real(self, tmp_path / "o.csv", *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    df = pd.DataFrame({"a": [1]})
    p = ps.save_results(df)
    assert p.endswith("/expansion_screening_results_11d.csv") and seen["path"] == p
    p2 = ps.save_results(df, "x.csv")
    assert p2.endswith("/x.csv")
    assert "Results saved to: x.csv" in capsys.readouterr().out
    assert pd.read_csv(tmp_path / "o.csv")["a"].tolist() == [1]


def test_main_block(monkeypatch, tmp_path, capsys):
    real = pd.DataFrame.to_csv
    seen = {}

    def to_csv(self, path, *a, **k):
        seen["n"] = len(self)
        return real(self, tmp_path / "o.csv", *a, **k)

    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    runpy.run_path(str(SRC), run_name="__main__")
    o = capsys.readouterr().out
    assert "SCREENING COMPLETE" in o and f"Total candidates: {seen['n']:,}" in o
    assert "Total companies to screen: 25,000" in o
