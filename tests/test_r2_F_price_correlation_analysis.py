import importlib
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "price_correlation_analysis.py"


def _rank_corr(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    rho = float(a.rank().corr(b.rank()))
    return rho, (0.001 if abs(rho) > 0.3 else 0.5)


def _pearson(a, b):
    r = float(pd.Series(list(a)).corr(pd.Series(list(b))))
    return r, (0.04 if abs(r) > 0.3 else 0.05)  # 0.05 exactly -> not significant


@pytest.fixture
def mod(monkeypatch):
    stats = types.ModuleType("scipy.stats")
    stats.spearmanr = _rank_corr
    stats.pearsonr = _pearson
    monkeypatch.setitem(sys.modules, "scipy", types.ModuleType("scipy"))
    monkeypatch.setitem(sys.modules, "scipy.stats", stats)
    monkeypatch.delitem(sys.modules, "price_correlation_analysis", raising=False)
    m = importlib.import_module("price_correlation_analysis")
    yield m
    sys.modules.pop("price_correlation_analysis", None)


def _tier2(n=50):
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(n)], "score": np.arange(n)})


def test_init_synthetic_cap(mod):
    a = mod.PriceCorrelationAnalyzer(_tier2(20))
    assert a.scorecard is None and len(a.tier2) == 20
    assert (a.tier2["market_cap_b"] > 0).all()
    assert a.correlation_results == {}
    # seeded -> deterministic
    b = mod.PriceCorrelationAnalyzer(_tier2(20))
    assert a.tier2["market_cap_b"].tolist() == b.tier2["market_cap_b"].tolist()


def test_init_with_scorecard_left_merge(mod):
    sc = pd.DataFrame({"ticker": ["T0", "T1"], "market_cap_b": [5.0, 7.0], "extra": [1, 2]})
    a = mod.PriceCorrelationAnalyzer(_tier2(3), sc)
    assert a.tier2["market_cap_b"].tolist()[:2] == [5.0, 7.0]
    assert np.isnan(a.tier2["market_cap_b"].iloc[2])
    assert "extra" not in a.tier2.columns
    assert len(a.tier2) == 3
    sc.loc[0, "market_cap_b"] = 99
    assert a.scorecard.loc[0, "market_cap_b"] == 5.0  # copied


def test_synthetic_price(mod):
    a = mod.PriceCorrelationAnalyzer(_tier2(200))
    df = a.generate_synthetic_stock_price()
    assert len(df) == 200 and "market_cap_b" in df
    assert (df["stock_price"] == 100 + df["stock_return_12m"]).all()
    assert "stock_return_12m" not in a.tier2.columns
    r = df["stock_return_12m"]
    bins = df["price_movement"].astype(str)
    assert (bins[r <= -5] == "Down >5%").all()
    assert (bins[(r > -5) & (r <= 0)] == "Down <5%").all()
    assert (bins[(r > 0) & (r <= 5)] == "Flat").all()
    assert (bins[(r > 5) & (r <= 10)] == "Up <10%").all()
    assert (bins[r > 10] == "Up >10%").all()
    # high market cap branch: capex effect in [-10, -2] vs [-5,5]
    hi = mod.PriceCorrelationAnalyzer(_tier2(5))
    hi.tier2["market_cap_b"] = 5000.0
    lo = mod.PriceCorrelationAnalyzer(_tier2(5))
    lo.tier2["market_cap_b"] = 1000.0  # not > 1000
    ah, al = hi.generate_synthetic_stock_price(), lo.generate_synthetic_stock_price()
    assert ah["stock_return_12m"].tolist() != al["stock_return_12m"].tolist()
    # same seed => hi branch uniform(-10,-2) vs lo uniform(-5,5): same u, different scale
    np.random.seed(42)
    u = np.random.uniform(0, 1)
    assert (ah["stock_return_12m"] - al["stock_return_12m"]).iloc[0] == pytest.approx(
        ((-10 + 8 * u) - (-5 + 10 * u)) / 4
    )


def test_extract_dimensions(mod):
    a = mod.PriceCorrelationAnalyzer(_tier2(30))
    d = a.extract_8_dimensions(_tier2(30))
    assert len(d) == 8 and all(len(v) == 30 for v in d.values())
    assert all(20 <= x <= 80 for v in d.values() for x in v)
    assert a.extract_8_dimensions(_tier2(0)) == {k: [] for k in d}


def test_calculate_correlations(mod):
    a = mod.PriceCorrelationAnalyzer(_tier2(20))
    ret = np.arange(20, dtype=float)
    df = pd.DataFrame({"stock_return_12m": ret})
    c = a.calculate_correlations(df, {"up": list(ret), "dn": list(-ret)})
    assert c["up"]["pearson_r"] == pytest.approx(1) and c["dn"]["spearman_r"] == pytest.approx(-1)
    assert c["up"]["significant"] is True
    rng = np.random.default_rng(0)
    c2 = a.calculate_correlations(df, {"n": list(rng.normal(size=20))})
    assert c2["n"]["significant"] is False  # pval 0.05 is not < 0.05
    assert set(c["up"]) == {
        "pearson_r", "pearson_pval", "spearman_r", "spearman_pval", "significant"
    }  # fmt: skip


def _corrs(vals, pv=0.01):
    names = [
        "debt_expansion", "capex_acceleration", "profit_reinvestment", "profitability_quality",
        "sustainability", "timing_alignment", "leverage_health", "fcf_generation",
    ]  # fmt: skip
    return {n: {"spearman_r": v, "spearman_pval": pv} for n, v in zip(names, vals)}


def test_recommend_weights(mod):
    a = mod.PriceCorrelationAnalyzer(_tier2(5))
    c = _corrs([0.5, -0.5, 0.25, 0.1, 0.0, 0.5, 0.25, 0.5])
    r = a.recommend_weight_adjustments(c)
    assert r["debt_expansion"]["adjustment_factor"] == 1.5
    assert r["capex_acceleration"]["adjustment_factor"] == 1.5  # abs
    assert r["capex_acceleration"]["correlation"] == -0.5
    assert r["sustainability"]["adjustment_factor"] == 0.5  # zero corr -> floor 0.5x
    assert r["profit_reinvestment"]["adjustment_factor"] == 1.0
    assert r["fcf_generation"]["current"] == 0 and r["fcf_generation"]["recommended"] == 0
    assert r["debt_expansion"]["p_value"] == 0.01
    raw = {
        "debt_expansion": 20 * 1.5, "capex_acceleration": 20 * 1.5, "profit_reinvestment": 15.0,
        "profitability_quality": 15 * 0.7, "sustainability": 7.5, "timing_alignment": 15.0,
        "leverage_health": 5 * 1.0,
    }  # fmt: skip
    tot = sum(round(v, 1) for v in raw.values())
    for k, v in raw.items():
        assert r[k]["recommended"] == pytest.approx(round(round(v, 1) * 100 / tot, 1), abs=0.051)
    assert sum(x["recommended"] for x in r.values()) == pytest.approx(100, abs=0.5)


def test_run_analysis(mod, capsys):
    a = mod.PriceCorrelationAnalyzer(_tier2(60))
    recs, df = a.run_analysis()
    assert set(recs) == set(_corrs([0] * 8)) and len(df) == 60
    out = capsys.readouterr().out
    assert "PRICE CORRELATION ANALYSIS - WEIGHT OPTIMIZATION" in out
    assert out.count("📊") == 1 and "Calculating correlations" in out


def _recs(p_sig=0.01, p_ns=0.5):
    names = list(_corrs([0] * 8))
    cors = [0.6, -0.5, 0.2, 0.1, 0.05, 0.3, 0.4, 0.0]
    cur = [20, 20, 15, 15, 15, 10, 5, 0]
    out = {}
    for n, c, w in zip(names, cors, cur):
        out[n] = {
            "current": w, "adjustment_factor": 1.25, "recommended": w + 1.0, "correlation": c,
            "p_value": p_sig if abs(c) >= 0.3 else p_ns,
        }  # fmt: skip
    return out


def test_report(mod, capsys):
    a = mod.PriceCorrelationAnalyzer(_tier2(5))
    df = pd.DataFrame(
        {
            "stock_return_12m": [-6.0, 1.0, 2.0, 12.0],
            "price_movement": pd.Categorical(
                ["Down >5%", "Flat", "Flat", "Up >10%"],
                categories=["Down >5%", "Down <5%", "Flat", "Up <10%", "Up >10%"],
            ),
        }
    )
    a.generate_report(_recs(), df)
    out = capsys.readouterr().out
    assert "Companies analyzed: 4" in out
    assert "-6.0% to 12.0%" in out and "Average 12M return: 2.2%" in out
    assert "Strongest price predictor: debt_expansion" in out
    assert "Weakest price predictor: fcf_generation" in out
    assert "significant predictors (p<0.05): 4" in out
    assert out.count("   ✅ ") == 4
    lines = [ln for ln in out.splitlines() if ln.startswith("   Flat")]
    assert lines and "2 ( 50.0%)" in lines[0] and "█" * 25 in lines[0]
    assert "TOTAL" in out and "100.0" in out and "108.0" in out
    assert "1.08x multiplier" in out
    # p exactly 0.05 is not significant
    a.generate_report(_recs(p_sig=0.05), df)
    assert "significant predictors (p<0.05): 0" in capsys.readouterr().out


def test_export(mod, tmp_path, capsys):
    a = mod.PriceCorrelationAnalyzer(_tier2(5))
    f = tmp_path / "r.csv"
    a.export_recommendations(_recs(), str(f))
    d = pd.read_csv(f, index_col=0)
    assert list(d.index)[0] == "debt_expansion" and d.loc["debt_expansion", "current"] == 20
    assert f"exported to {f}" in capsys.readouterr().out


def test_main_block(mod, capsys):
    runpy.run_path(str(SRC), run_name="__main__")
    assert "PRICE CORRELATION ANALYSIS - WEIGHT OPTIMIZATION FRAMEWORK" in capsys.readouterr().out
