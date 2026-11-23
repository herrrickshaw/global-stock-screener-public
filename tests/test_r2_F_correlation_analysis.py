import sys

import numpy as np
import pandas as pd
import pytest

import correlation_analysis as ca


def _df(closes):
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float)})


def test_perf_features_filters_and_values(monkeypatch):
    up = np.linspace(100, 200, 300)
    data = {
        "SHORT": _df(np.ones(119)),
        "NONE": None,
        "NEG": _df(np.r_[0.0, np.ones(200)]),
        "FLAT": _df(np.ones(120) * 10),
        "UP": _df(up),
        "EXACT": _df(np.linspace(100, 110, 120)),
    }
    monkeypatch.setattr(ca.mm, "read", lambda m, tier: data)
    out = ca._perf_features("IN").set_index("Symbol")
    assert set(out.index) == {"FLAT", "UP", "EXACT"}
    assert np.isnan(out.loc["FLAT", "sharpe"])
    assert out.loc["FLAT", "vol"] == 0
    assert out.loc["UP", "mom_12m"] == pytest.approx(up[-1] / up[-252] - 1)
    assert np.isnan(out.loc["EXACT", "mom_12m"])  # len 120 <= 252
    assert out.loc["UP", "max_drawdown"] == 0
    assert out.loc["UP", "cagr"] == pytest.approx((2.0) ** (252 / 300) - 1)
    # exactly 252 long -> still NaN momentum (strict >)
    monkeypatch.setattr(ca.mm, "read", lambda m, tier: {"A": _df(np.linspace(1, 2, 252))})
    assert np.isnan(ca._perf_features("IN").iloc[0]["mom_12m"])
    monkeypatch.setattr(ca.mm, "read", lambda m, tier: {"A": _df(np.linspace(1, 2, 253))})
    assert not np.isnan(ca._perf_features("IN").iloc[0]["mom_12m"])


def test_spearman_small_and_fallback(monkeypatch):
    x = pd.Series([1, 2, 3, 4, np.nan])
    assert ca._spearman(x, x)[2] == 4 and np.isnan(ca._spearman(x, x)[0])
    x = pd.Series([1.0, 2, 3, 4, 5, 6])
    y = pd.Series([6.0, 5, 4, 3, 2, 1])
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    rho, p, n = ca._spearman(x, y)
    assert rho == pytest.approx(-1.0) and np.isnan(p) and n == 6


def test_spearman_scipy(monkeypatch):
    import types

    fake = types.ModuleType("scipy.stats")
    fake.spearmanr = lambda a, b: (0.5, 0.01)
    monkeypatch.setitem(sys.modules, "scipy", types.ModuleType("scipy"))
    monkeypatch.setitem(sys.modules, "scipy.stats", fake)
    x = pd.Series(range(5), dtype=float)
    assert ca._spearman(x, x) == (0.5, 0.01, 5)


def _setup(monkeypatch, tmp_path, fund):
    monkeypatch.setattr(ca, "FUND_DIR", tmp_path)
    if fund is not None:
        fund.to_parquet(tmp_path / "IN.parquet")
    rng = np.random.default_rng(0)
    data = {f"S{i}": _df(100 * np.cumprod(1 + rng.normal(0.001 * i, 0.01, 300))) for i in range(8)}
    monkeypatch.setattr(ca.mm, "read", lambda m, tier: data)


def test_analyse_missing_and_empty(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, None)
    assert ca.analyse("IN").empty
    pd.DataFrame({"Symbol": []}).to_parquet(tmp_path / "IN.parquet")
    assert ca.analyse("IN").empty
    pd.DataFrame({"Symbol": ["ZZ"], "roe": [1.0]}).to_parquet(tmp_path / "IN.parquet")
    assert ca.analyse("IN").empty  # no overlap
    monkeypatch.setattr(ca.mm, "read", lambda m, tier: {})
    pd.DataFrame({"Symbol": ["S1"], "roe": [1.0]}).to_parquet(tmp_path / "IN.parquet")
    assert ca.analyse("IN").empty  # perf empty


def test_analyse_ranks_and_main(monkeypatch, tmp_path, capsys):
    fund = pd.DataFrame(
        {
            "Symbol": [f"S{i}" for i in range(8)],
            "roe": np.arange(8, dtype=float),
            "noise": [3.0, 1, 4, 1, 5, 9, 2, 6],
            "txt": list("abcdefgh"),
            "const": [np.nan] * 8,
        }
    )
    _setup(monkeypatch, tmp_path, fund)
    res = ca.analyse("IN", "cagr")
    assert list(res.columns) == ["metric", "spearman_rho", "p_value", "n", "significant"]
    assert set(res["metric"]) == {"roe", "noise"}
    assert res.iloc[0]["metric"] == "roe"
    assert res["n"].tolist() == [8, 8]
    monkeypatch.setattr(sys, "argv", ["x", "--target", "cagr"])
    assert ca.main() == 0
    out = capsys.readouterr().out
    assert "IN — fundamentals vs cagr" in out
    assert "significant at p<0.05 (n up to 8)" in out


def test_analyse_all_nan_results_empty(monkeypatch, tmp_path, capsys):
    fund = pd.DataFrame({"Symbol": ["S0", "S1"], "roe": [1.0, 2.0]})
    _setup(monkeypatch, tmp_path, fund)
    assert ca.analyse("IN").empty
    monkeypatch.setattr(sys, "argv", ["x"])
    assert ca.main() == 0
    assert "IN: no overlap" in capsys.readouterr().out
