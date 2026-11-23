import importlib
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "price_criterion_correlation_tracker.py"


def _rank_corr(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    rho = float(a.rank().corr(b.rank()))
    return rho, (0.001 if abs(rho) > 0.3 else 0.5)


def _pearson(a, b):
    r = float(pd.Series(list(a)).corr(pd.Series(list(b))))
    return r, 0.01


@pytest.fixture
def mod(monkeypatch):
    stats = types.ModuleType("scipy.stats")
    stats.spearmanr = _rank_corr
    stats.pearsonr = _pearson
    monkeypatch.setitem(sys.modules, "scipy", types.ModuleType("scipy"))
    monkeypatch.setitem(sys.modules, "scipy.stats", stats)
    monkeypatch.delitem(sys.modules, "price_criterion_correlation_tracker", raising=False)
    m = importlib.import_module("price_criterion_correlation_tracker")
    yield m
    sys.modules.pop("price_criterion_correlation_tracker", None)


def _frame(n=40):
    rng = np.random.default_rng(1)
    price = rng.normal(10, 5, n)
    return pd.DataFrame(
        {
            "expansion_score": price * 2 + rng.normal(0, 1, n),
            "tier": ["Tier 1 (Aggressive Expander)"] * 10
            + ["Tier 2 (Strong Expander)"] * 10
            + ["Tier 3 (Moderate Expander)"] * 10
            + ["Tier 4 (Passive/Mature)"] * (n - 30),
            "revenue_cagr": price + rng.normal(0, 0.1, n),  # strong +
            "capex_cagr": -price + rng.normal(0, 0.1, n),  # strong -
            "debt_cagr": rng.normal(0, 1, n),  # noise
            "fcf_margin": price,
            "price_5yr_cagr": price,
        }
    )


def test_calculate_correlations(mod, capsys):
    t = mod.PriceCriterionCorrelationTracker(_frame())
    c = t.calculate_correlations()
    assert set(c["criterion"]) == {
        "Revenue Growth CAGR",
        "Capex Acceleration CAGR",
        "Debt Growth CAGR",
        "FCF Margin",
    }
    assert abs(c["spearman_r"].iloc[0]) >= abs(c["spearman_r"].iloc[-1])
    row = c.set_index("criterion")
    assert row.loc["FCF Margin", "spearman_r"] == pytest.approx(1.0)
    assert bool(row.loc["FCF Margin", "is_significant"]) is True
    assert row.loc["FCF Margin", "n_samples"] == 40
    assert bool(row.loc["Debt Growth CAGR", "is_significant"]) is False
    out = capsys.readouterr().out
    assert "Analyzing 40 companies" in out
    assert "✓ SIGNIFICANT" in out and "✗ Not significant" in out
    assert "Negative" in out and "Positive" in out


def test_calculate_skips_short_and_missing(mod):
    df = _frame(40)
    df.loc[:30, "fcf_margin"] = np.nan  # 9 valid -> skipped (<10)
    df.loc[:29, "debt_cagr"] = np.nan  # exactly 10 valid -> kept
    c = mod.PriceCriterionCorrelationTracker(df).calculate_correlations()
    assert "FCF Margin" not in set(c["criterion"])
    assert "Debt Growth CAGR" in set(c["criterion"])
    no_target = df.drop(columns="price_5yr_cagr")
    with pytest.raises(KeyError):  # empty frame has no spearman_r column to sort on
        mod.PriceCriterionCorrelationTracker(no_target).calculate_correlations()


def test_validate_weights(mod, capsys):
    t = mod.PriceCriterionCorrelationTracker(_frame())
    t.correlations = pd.DataFrame(
        {
            "criterion": [
                "Capex Acceleration CAGR",  # w24 corr .05 -> Good
                "FCF Margin",  # w22 corr .01 -> Question
                "Revenue Growth CAGR",  # w19 corr .03 -> Good (>10, >.02)
                "Debt Growth CAGR",  # w10 corr .5 -> Question (not >10)
                "Debt-to-Equity Ratio",  # w2 corr 0 -> Mismatch
                "Unknown",
            ],
            "spearman_r": [-0.05, 0.01, 0.03, 0.5, 0.0, 0.9],
        }
    )
    v = t.validate_model_weights({}).set_index("criterion")
    assert v["match"].to_dict() == {
        "Capex Acceleration CAGR": "✓ Good",
        "FCF Margin": "⚠ Question",
        "Revenue Growth CAGR": "✓ Good",
        "Debt Growth CAGR": "⚠ Question",
        "Debt-to-Equity Ratio": "✗ Mismatch",
    }
    assert v.loc["Capex Acceleration CAGR", "effectiveness"] == 0.05
    assert v.loc["FCF Margin", "weight"] == 22
    v2 = t.validate_model_weights({"capex_acceleration": 20, "leverage_health": 5})
    assert v2.set_index("criterion").loc["Capex Acceleration CAGR", "weight"] == 20
    assert v2.set_index("criterion").loc["Debt-to-Equity Ratio", "match"] == "✗ Mismatch"
    capsys.readouterr()


def _corr_table():
    return pd.DataFrame(
        {
            "criterion": ["S", "Sx", "W", "W2", "N", "N2"],
            "spearman_r": [0.5, 0.05, 0.03, 0.04, 0.02, 0.0],
            "spearman_p": [0.001, 0.5, 0.01, 0.01, 0.01, 0.9],
            "is_significant": [True, False, True, True, True, False],
        }
    )


def test_signal_vs_noise(mod, capsys):
    t = mod.PriceCriterionCorrelationTracker(_frame())
    t.correlations = _corr_table()
    r = t.identify_signal_vs_noise()
    assert r["strong_signal"]["criterion"].tolist() == ["S"]  # Sx not significant
    assert r["weak_signal"]["criterion"].tolist() == ["W", "W2"]  # 0.04 incl, 0.02 excl
    assert r["noise"]["criterion"].tolist() == ["N", "N2"]
    out = capsys.readouterr().out
    assert "Count: 1" in out and "Count: 2" in out


def test_recommendations(mod, capsys):
    t = mod.PriceCriterionCorrelationTracker(_frame())
    tab = _corr_table()
    tab.loc[0, "criterion"] = "FCF Margin"
    tab.loc[1, ["spearman_r", "is_significant"]] = [0.6, True]
    tab.loc[1, "criterion"] = "Other"
    t.correlations = tab
    t.generate_recommendations()
    out = capsys.readouterr().out
    assert "FCF Margin" in out and "Already well-weighted" in out
    assert "Other" in out and "Consider +5pp" in out
    assert "Consider removing" in out
    # no weak / no noise / no strong
    t.correlations = tab.iloc[:1]
    t.generate_recommendations()
    out = capsys.readouterr().out
    assert "All criteria are either strong or noise" in out
    assert "No clear noise criteria identified" in out
    t.correlations = tab.iloc[0:0]
    t.generate_recommendations()


def test_tier_performance(mod, capsys):
    df = _frame()
    df["price_5yr_cagr"] = [30.0] * 10 + [20.0] * 10 + [10.0] * 10 + [1.0] * 10
    t = mod.PriceCriterionCorrelationTracker(df)
    p = t.tier_price_performance()
    assert p["count"].tolist() == [10, 10, 10, 10]
    assert p["avg_price_cagr"].tolist() == [30, 20, 10, 1]
    out = capsys.readouterr().out
    assert "+29.0%" in out and "VALIDATES" in out
    df["price_5yr_cagr"] = [3.0] * 10 + [2.0] * 10 + [2.0] * 10 + [1.0] * 10
    mod.PriceCriterionCorrelationTracker(df).tier_price_performance()
    assert "Weak validation" in capsys.readouterr().out
    df["price_5yr_cagr"] = [1.0] * 10 + [2.0] * 10 + [2.0] * 10 + [3.0] * 10
    mod.PriceCriterionCorrelationTracker(df).tier_price_performance()
    assert "Model FAILS" in capsys.readouterr().out
    df["price_5yr_cagr"] = [3.0] * 10 + [2.0] * 10 + [2.0] * 10 + [3.0] * 10
    mod.PriceCriterionCorrelationTracker(df).tier_price_performance()
    assert "Model FAILS" in capsys.readouterr().out  # tie is a failure (> 0 required)
    # single tier -> no comparison
    one = df[df["tier"] == "Tier 1 (Aggressive Expander)"]
    r = mod.PriceCriterionCorrelationTracker(one).tier_price_performance()
    assert len(r) == 1 and "outperformance" not in capsys.readouterr().out
    # 5.0 exactly is "weak"
    df["price_5yr_cagr"] = [6.0] * 10 + [2.0] * 10 + [2.0] * 10 + [1.0] * 10
    mod.PriceCriterionCorrelationTracker(df).tier_price_performance()
    assert "Weak validation" in capsys.readouterr().out


def _with_r2(r2, n=200):
    rng = np.random.default_rng(3)
    x = np.arange(n, dtype=float)
    xs = (x - x.mean()) / x.std()
    z = rng.normal(0, 1, n)
    z = z - np.polyval(np.polyfit(xs, z, 1), xs)
    zs = (z - z.mean()) / z.std()
    y = np.sqrt(r2) * xs + np.sqrt(1 - r2) * zs
    return pd.DataFrame({"expansion_score": x, "price_5yr_cagr": y, "tier": "t"})


@pytest.mark.parametrize(
    "r2,line",
    [
        (0.5, "✓ STRONG: Model explains >50% of price variance"),
        (0.2001, "✓ STRONG"),
        (0.1999, "✓ MODERATE: Model explains 20.0% of price variance"),
        (0.1001, "✓ MODERATE"),
        (0.0999, "⚠ WEAK: Model explains only 10.0% of price variance"),
        (0.0501, "⚠ WEAK"),
        (0.0499, "✗ VERY WEAK: Model explains <5.0% of price variance"),
    ],
)
def test_r_squared_labels(mod, capsys, r2, line):
    res = mod.PriceCriterionCorrelationTracker(_with_r2(r2)).coefficient_of_determination()
    assert res["r_squared"] == pytest.approx(r2, abs=1e-6)
    assert res["correlation"] ** 2 == pytest.approx(res["r_squared"])
    assert line in capsys.readouterr().out


def test_r_squared_too_few(mod):
    df = pd.DataFrame({"expansion_score": [1.0, 2.0], "price_5yr_cagr": [1.0, 2.0]})
    assert mod.PriceCriterionCorrelationTracker(df).coefficient_of_determination() is None
    df3 = pd.DataFrame({"expansion_score": [1.0, 2.0, 3.0], "price_5yr_cagr": [1.0, 2.0, 4.0]})
    assert mod.PriceCriterionCorrelationTracker(df3).coefficient_of_determination() is not None


def test_copy_of_input(mod):
    df = _frame()
    t = mod.PriceCriterionCorrelationTracker(df)
    t.results.loc[0, "revenue_cagr"] = 999
    assert df.loc[0, "revenue_cagr"] != 999


def test_main_demo_data(mod, monkeypatch, capsys):
    real = pd.read_csv

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(pd, "read_csv", boom)
    runpy.run_path(str(SRC), run_name="__main__")
    monkeypatch.setattr(pd, "read_csv", real)
    out = capsys.readouterr().out
    assert "Results file not found. Using demo data" in out
    assert "CORRELATION ANALYSIS COMPLETE" in out
    assert "Analyzing 1,000 companies" in out


def test_main_reads_file(mod, monkeypatch, capsys):
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: _frame())
    runpy.run_path(str(SRC), run_name="__main__")
    out = capsys.readouterr().out
    assert "Using demo data" not in out and "Analyzing 40 companies" in out
