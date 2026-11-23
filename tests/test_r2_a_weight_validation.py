"""Group A: backtest_weight_validation.py."""

import numpy as np
import pandas as pd
import pytest

import backtest_weight_validation as m

N = 40


def _corr(rows):
    return pd.DataFrame(
        [{"metric": a, "spearman_rho": b, "p_value": c, "n": N} for a, b, c in rows]
    )


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    syms = [f"S{i:02d}" for i in range(N)]
    roe = np.arange(N, dtype=float)
    lev = rng.permutation(N).astype(float)
    fund = pd.DataFrame({"Symbol": syms, "roe": roe, "lev": lev, "junk": rng.random(N)})
    perf = pd.DataFrame(
        {"Symbol": syms, "cagr": roe / 100.0 - lev / 1000.0, "mom_12m": rng.random(N)}
    )
    (tmp_path / "IN.parquet").parent.mkdir(exist_ok=True)
    fund.to_parquet(tmp_path / "IN.parquet")
    monkeypatch.setattr(m, "FUND_DIR", tmp_path)
    state = {
        "corr": _corr([("roe", 0.9, 0.001), ("lev", -0.3, 0.04), ("junk", 0.05, 0.6)]),
        "perf": perf,
    }
    monkeypatch.setattr(m.ca, "analyse", lambda market, target="cagr": state["corr"])
    monkeypatch.setattr(m.ca, "_perf_features", lambda market: state["perf"].copy())
    return state


def test_zscore():
    z = m._zscore(pd.Series([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0.0)
    assert z.iloc[2] == pytest.approx(1.0)  # (3-2)/std(ddof=1)=1
    assert list(m._zscore(pd.Series([5.0, 5.0, 5.0]))) == [0.0, 0.0, 0.0]
    assert list(m._zscore(pd.Series([5.0]))) == [0.0]


def test_build_composite_empty_corr(world):
    world["corr"] = pd.DataFrame()
    merged, used, picks = m.build_composite("IN")
    assert merged is None and used is None and picks.empty


def test_build_composite_significant_only(world):
    merged, used, picks = m.build_composite("IN", p_max=0.10)
    assert used == ["roe", "lev"]  # junk (p=0.6) excluded
    assert list(picks["metric"]) == ["roe", "lev"]
    exp = m._zscore(merged["roe"]) - m._zscore(merged["lev"])  # sign-aligned
    assert merged["composite"].tolist() == pytest.approx(exp.tolist())


def test_build_composite_p_boundary(world):
    world["corr"] = _corr([("roe", 0.9, 0.10), ("lev", -0.3, 0.1001)])
    _, used, _ = m.build_composite("IN", p_max=0.10)
    assert used == ["roe"]  # p == p_max counts


def test_build_composite_top_k(world):
    _, used, picks = m.build_composite("IN", top_k=1)
    assert used == ["roe"] and len(picks) == 1


def test_build_composite_falls_back_when_nothing_significant(world):
    world["corr"] = _corr([("roe", 0.9, 0.5), ("lev", -0.3, 0.6), ("junk", 0.1, 0.9)])
    _, used, picks = m.build_composite("IN", top_k=2)
    assert used == ["roe", "lev"] and len(picks) == 2


def test_build_composite_skips_missing_column(world):
    world["corr"] = _corr([("ghost", 0.9, 0.001), ("roe", 0.5, 0.001)])
    merged, used, _ = m.build_composite("IN")
    assert used == ["ghost"][:0] + ["roe"]
    assert "ghost" not in merged


def test_build_composite_inner_join_on_symbol(world):
    world["perf"] = world["perf"].iloc[:10]
    merged, _, _ = m.build_composite("IN")
    assert len(merged) == 10


def test_validate_no_overlap_error(world):
    world["corr"] = pd.DataFrame()
    assert m.validate("IN") == {"market": "IN", "error": "no fundamentals/perf overlap"}


def test_validate_too_few(world):
    world["perf"] = world["perf"].iloc[:9]
    r = m.validate("IN", q=5)
    assert r["error"] == "too few names (9) for 5-tiles"
    r = m.validate("IN", q=4)  # 9 >= 8
    assert "error" not in r


def test_validate_quintile_table(world, capsys):
    r = m.validate("IN", "cagr", q=5)
    t = r["table"]
    assert list(t.index) == [1, 2, 3, 4, 5]
    assert t["count"].tolist() == [8] * 5
    assert r["n"] == N and r["target"] == "cagr"
    assert r["metrics_used"] == ["roe", "lev"]
    assert r["spread_top_minus_bottom"] == pytest.approx(t.loc[5, "mean"] - t.loc[1, "mean"])
    assert r["spread_top_minus_bottom"] > 0
    assert t["mean"].is_monotonic_increasing
    assert r["used_test_split"] is False
    # the data-quality hook warns about missing OHLC columns
    out = capsys.readouterr().out
    assert "Data validation warnings:" in out
    assert out.count("•") == 3  # capped at 3 messages


def test_validate_drops_nan_target(world):
    p = world["perf"].copy()
    p.loc[:4, "cagr"] = np.nan
    world["perf"] = p
    assert m.validate("IN")["n"] == N - 5


def test_validate_test_split_with_date_column(world, capsys):
    p = world["perf"].copy()
    p["date"] = pd.date_range("2023-01-01", periods=N, freq="D")
    world["perf"] = p
    r = m.validate("IN", use_test_split=True)
    out = capsys.readouterr().out
    assert "Applied TEST split" in out
    assert r["used_test_split"] is True


def test_validate_test_split_without_date_column(world, capsys):
    r = m.validate("IN", use_test_split=True)
    assert "No date column in performance data" in capsys.readouterr().out
    assert r["n"] == N


def test_validate_test_split_old_dates_leave_too_few(world):
    p = world["perf"].copy()
    p["date"] = pd.date_range("2016-01-01", periods=N, freq="D")  # all in TRAIN
    world["perf"] = p
    assert m.validate("IN", use_test_split=True)["error"].startswith("too few names (0)")


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="validate() accepts a capitalised 'Date' column but then calls "
    "filter_data_by_split(perf, 'date'), which raises KeyError",
)
def test_validate_test_split_capital_date_column(world):
    p = world["perf"].copy()
    p["Date"] = pd.date_range("2023-01-01", periods=N, freq="D")
    world["perf"] = p
    m.validate("IN", use_test_split=True)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="when no date column exists the split is NOT applied, yet validate() still "
    "reports used_test_split=True (main then prints 'Evaluated on TEST split')",
)
def test_validate_reports_split_not_applied_without_date(world):
    assert m.validate("IN", use_test_split=True)["used_test_split"] is False


def test_proper_split_banner(world, capsys):
    r = m.validate_with_proper_split("IN", "cagr", 5, use_test_split=True)
    out = capsys.readouterr().out
    assert "FUNDAMENTALS COMPOSITE VALIDATION" in out
    assert "Using TEST split: 2023-01-01 to 2024-06-30 (unseen data)" in out
    assert r["n"] == N
    m.validate_with_proper_split("IN", use_test_split=False)
    assert "Using all available data (not recommended for evaluation)" in capsys.readouterr().out


def _run_main(monkeypatch, *argv):
    if "--market" not in argv:
        argv = ("--market", "IN", *argv)
    monkeypatch.setattr("sys.argv", ["x", *argv])
    return m.main()


def test_main_report(world, monkeypatch, capsys):
    assert _run_main(monkeypatch, "--market", "IN", "--target", "cagr") == 0
    out = capsys.readouterr().out
    assert f"IN — fundamentals composite vs cagr (n={N})" in out
    assert "metrics: roe, lev" in out
    assert "Q5−Q1 spread (cagr):" in out and "signal separates winners" in out
    assert "✅ Validation complete" in out
    assert "No date column" in out  # test split requested by default


def test_main_no_split_flag(world, monkeypatch, capsys):
    _run_main(monkeypatch, "--no-split", "--quantiles", "4")
    out = capsys.readouterr().out
    assert "Using all available data" in out
    assert "Q4−Q1 spread" in out


def test_main_test_split_flag_reported(world, monkeypatch, capsys):
    p = world["perf"].copy()
    p["date"] = pd.date_range("2023-01-01", periods=N, freq="D")
    world["perf"] = p
    _run_main(monkeypatch)
    assert "Evaluated on TEST split" in capsys.readouterr().out


def test_main_negative_spread_message(world, monkeypatch, capsys):
    world["perf"] = world["perf"].assign(cagr=lambda d: -d["cagr"])
    _run_main(monkeypatch, "--no-split")
    assert "no/negative separation" in capsys.readouterr().out


def test_main_error_path(world, monkeypatch, capsys):
    world["corr"] = pd.DataFrame()
    assert _run_main(monkeypatch, "--market", "ZZ") == 0
    assert "ZZ: no fundamentals/perf overlap" in capsys.readouterr().out


def test_main_rejects_bad_target(world, monkeypatch):
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--target", "bogus")


@pytest.fixture
def no_data_config(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "data_config", None)  # import raises ImportError
    importlib.reload(m)
    yield m
    monkeypatch.undo()
    importlib.reload(m)


def test_without_data_config_module(no_data_config, world, monkeypatch, capsys):
    mod = no_data_config
    assert mod.HAS_DATA_CONFIG is False
    # reload re-bound ca/FUND_DIR; re-install the fakes on the reloaded module
    monkeypatch.setattr(mod, "FUND_DIR", m.FUND_DIR)
    monkeypatch.setattr(mod.ca, "analyse", lambda market, target="cagr": world["corr"])
    monkeypatch.setattr(mod.ca, "_perf_features", lambda market: world["perf"].copy())
    r = mod.validate_with_proper_split("IN", use_test_split=True)
    out = capsys.readouterr().out
    assert "Using all available data" in out  # cannot honour the split
    assert r["used_test_split"] is False
    assert "Data validation warnings" not in out
