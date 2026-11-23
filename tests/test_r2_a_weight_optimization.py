"""Group A: backtest_weight_optimization.py."""

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest_weight_optimization as m

ROOT = Path(m.__file__).resolve().parent
DIMS = list(m.WeightBacktester.BASELINE_WEIGHTS)


def _frame(n=40, seed=0, dates=False, signal=True):
    rng = np.random.default_rng(seed)
    d = {f"{k}_score": rng.random(n) * 100 for k in DIMS}
    ret = rng.normal(0, 10, n)
    if signal:  # make capex_acceleration informative
        d["capex_acceleration_score"] = ret * 3 + 50
    d["stock_return_12m"] = ret
    df = pd.DataFrame(d)
    if dates:
        df["date"] = pd.date_range("2023-01-02", periods=n, freq="D")
    return df


def test_weights_sum_to_hundred():
    assert sum(m.WeightBacktester.BASELINE_WEIGHTS.values()) == 100
    assert sum(m.WeightBacktester.OPTIMIZED_WEIGHTS.values()) == 100


def test_init_banner(capsys):
    m.WeightBacktester(_frame(), lookback_years=2, use_test_split=False)
    out = capsys.readouterr().out
    assert "Companies: 40" in out and "Lookback period: 2 years" in out
    assert "Using proper train/test split: False" in out
    assert "Date range:" in out
    m.WeightBacktester(_frame(), use_test_split=True)
    out = capsys.readouterr().out
    assert "TEST [2023-01-01 to 2024-06-30] (unseen data)" in out


def test_init_does_not_mutate_input():
    df = _frame()
    before = df.copy()
    b = m.WeightBacktester(df)
    b.data["x"] = 1
    pd.testing.assert_frame_equal(df, before)


def test_composite_score_weights():
    b = m.WeightBacktester(_frame())
    df = pd.DataFrame(
        {
            "debt_expansion_score": [100.0, 0.0],
            "capex_acceleration": [50.0, 10.0],  # raw column fallback
            "unrelated": [9.0, 9.0],
        }
    )
    s = b.calculate_composite_score(df, {"debt_expansion": 20, "capex_acceleration": 40, "x": 5})
    assert s.tolist() == pytest.approx([20.0 + 20.0, 0.0 + 4.0])


def test_composite_prefers_score_column_over_raw():
    b = m.WeightBacktester(_frame())
    df = pd.DataFrame({"a_score": [10.0], "a": [999.0]})
    assert b.calculate_composite_score(df, {"a": 50}).tolist() == [5.0]


def test_signals_strictly_above_percentile():
    b = m.WeightBacktester(_frame())
    sig = b.generate_signals(np.arange(10, dtype=float), 50.0)
    assert sig.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # median 4.5
    sig = b.generate_signals(np.array([1.0, 1.0, 1.0, 1.0]), 50.0)
    assert sig.sum() == 0  # ties are not > threshold
    assert b.generate_signals(np.arange(10.0), 90.0).sum() == 1
    assert b.generate_signals(np.arange(10.0)).sum() == 5  # default 50th pct


def test_evaluate_signals_confusion_core():
    b = m.WeightBacktester(_frame())
    ret = pd.Series([1.0, 2, 3, 4, 5, 6])
    sig = np.array([0, 0, 0, 1, 1, 1])  # perfect
    r = b._evaluate_signals(sig, ret, "perfect")
    assert (r["tp"], r["fp"], r["fn"]) == (3, 0, 0)
    assert (r["precision"], r["recall"], r["f1"]) == (1.0, 1.0, 1.0)
    assert r["n_signals"] == 3 and r["n_outperform"] == 3
    assert r["avg_buy_return"] == 5.0 and r["avg_sell_return"] == 2.0
    assert r["return_spread"] == 3.0 and r["label"] == "perfect"
    worst = b._evaluate_signals(np.array([1, 1, 1, 0, 0, 0]), ret, "inverse")
    assert (worst["tp"], worst["fp"], worst["fn"]) == (0, 3, 3)
    assert worst["precision"] == 0 and worst["recall"] == 0 and worst["f1"] == 0
    assert worst["return_spread"] == -3.0


def test_evaluate_signals_partial():
    b = m.WeightBacktester(_frame())
    ret = pd.Series([1.0, 2, 3, 4, 5, 6])
    sig = np.array([0, 1, 0, 1, 1, 0])
    r = b._evaluate_signals(sig, ret, "p")
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_evaluate_signals_no_buys():
    b = m.WeightBacktester(_frame())
    r = b._evaluate_signals(np.zeros(6, dtype=int), pd.Series([1.0, 2, 3, 4, 5, 6]), "none")
    assert r["precision"] == 0 and r["recall"] == 0 and r["f1"] == 0
    assert r["avg_buy_return"] == 0  # no buy rows -> 0, not NaN
    assert r["avg_sell_return"] == 3.5


def test_evaluate_signals_all_buys():
    b = m.WeightBacktester(_frame())
    r = b._evaluate_signals(np.ones(6, dtype=int), pd.Series([1.0, 2, 3, 4, 5, 6]), "all")
    assert r["avg_sell_return"] == 0
    assert r["recall"] == 1.0 and r["precision"] == 0.5


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="true-negatives use bitwise `~` on int arrays (~0 == -1), so `tn` is negative "
    "garbage and `accuracy` is wrong; should be 3 and 1.0 for a perfect signal",
)
def test_evaluate_signals_true_negatives_and_accuracy():
    b = m.WeightBacktester(_frame())
    r = b._evaluate_signals(np.array([0, 0, 0, 1, 1, 1]), pd.Series([1.0, 2, 3, 4, 5, 6]), "x")
    assert r["tn"] == 3
    assert r["accuracy"] == 1.0


def test_run_backtest_selects_sets(capsys):
    b = m.WeightBacktester(_frame())
    both = b.run_backtest("both")
    assert set(both) == {"baseline", "optimized"}
    assert set(b.run_backtest("baseline")) == {"baseline"}
    assert set(b.run_backtest("optimized")) == {"optimized"}
    assert set(b.run_backtest("nonsense")) == set()
    assert both["baseline"]["label"] == "Baseline"
    assert both["optimized"]["label"] == "Optimized"
    # the informative dimension carries 24% vs 20% weight, so the optimised
    # model buys the same top half here
    assert both["optimized"]["n_signals"] > 0
    out = capsys.readouterr().out
    assert "RUNNING BACKTEST - BOTH" in out


def test_format_change_pct_pp_raw():
    f = m.WeightBacktester._format_change
    imp, a, b, c = f("pct", 0.5, 0.6)
    assert imp == pytest.approx(10.0) and (a, b) == ("50.0%", "60.0%") and c == "+10.0pp"
    imp, a, b, c = f("pp", 1.0, 3.5)
    assert imp == 2.5 and (a, b, c) == ("+1.00%", "+3.50%", "+2.50pp")
    imp, a, b, c = f("raw", 0.25, 0.2)
    assert imp == pytest.approx(-0.05) and (a, b, c) == ("0.2500", "0.2000", "-0.0500")


def _metrics(**over):
    base = dict(
        label="x",
        n_signals=5,
        n_outperform=5,
        tp=3,
        fp=2,
        fn=2,
        tn=3,
        precision=0.6,
        recall=0.6,
        f1=0.6,
        accuracy=0.6,
        avg_buy_return=2.0,
        avg_sell_return=-1.0,
        return_spread=3.0,
    )
    base.update(over)
    return base


def test_print_comparison_thresholds(capsys):
    b = m.WeightBacktester(_frame())
    b._print_comparison(_metrics(f1=0.5), _metrics(f1=0.5601))
    out = capsys.readouterr().out
    assert "SIGNIFICANT IMPROVEMENT" in out and "DEPLOY OPTIMIZED WEIGHTS" in out
    b._print_comparison(_metrics(f1=0.5), _metrics(f1=0.54))  # 0 < gain <= 0.05 -> modest
    out = capsys.readouterr().out
    assert "MODEST IMPROVEMENT" in out and "DEPLOY with caution" in out
    b._print_comparison(_metrics(f1=0.5), _metrics(f1=0.5))  # zero -> no improvement
    out = capsys.readouterr().out
    assert "NO IMPROVEMENT" in out and "KEEP BASELINE WEIGHTS" in out
    assert "Improved metrics: 0/6" in out
    b._print_comparison(_metrics(f1=0.5), _metrics(f1=0.3, precision=0.9))
    out = capsys.readouterr().out
    assert "F1 score decreased by 0.2000" in out
    assert "Improved metrics: 1/6" in out
    assert "✅" in out and "❌" in out and " = " in out


def test_print_metrics_block(capsys):
    m.WeightBacktester._print_metrics_block("TITLE", _metrics(), 10)
    out = capsys.readouterr().out
    assert "TITLE" in out and "Buy signals: 5 out of 10 companies" in out
    assert "Precision:  60.0%" in out and "F1 Score:   0.6000" in out
    assert "Avg buy return:  +2.00%" in out and "Return spread:   +3.00pp" in out


def test_generate_report_variants(capsys):
    b = m.WeightBacktester(_frame())
    res = b.run_backtest("both")
    capsys.readouterr()
    b.generate_report(res)
    out = capsys.readouterr().out
    assert "BASELINE WEIGHTS (Original Model)" in out
    assert "OPTIMIZED WEIGHTS (F1-Tuned Model)" in out
    assert "IMPROVEMENTS (OPTIMIZED vs BASELINE)" in out
    b.generate_report({"baseline": res["baseline"]})
    out = capsys.readouterr().out
    assert "IMPROVEMENTS" not in out and "OPTIMIZED WEIGHTS" not in out
    b.generate_report({"optimized": res["optimized"]})
    out = capsys.readouterr().out
    assert "OPTIMIZED WEIGHTS" in out and "BASELINE WEIGHTS" not in out


def test_export_results(tmp_path, capsys):
    b = m.WeightBacktester(_frame())
    res = b.run_backtest("both")
    p = tmp_path / "out.csv"
    b.export_results(res, str(p))
    df = pd.read_csv(p)
    assert list(df["weight_set"]) == ["baseline", "optimized"]
    assert list(df.columns) == [
        "weight_set",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "avg_buy_return",
        "avg_sell_return",
        "return_spread",
    ]
    assert df.loc[0, "true_positives"] == res["baseline"]["tp"]
    assert f"exported to {p}" in capsys.readouterr().out


# ---------------------------------------------------------- backtest_from_data_source
def test_from_source_missing_file(tmp_path, capsys):
    assert m.backtest_from_data_source(str(tmp_path / "no.parquet")) == {}
    assert "Data file not found" in capsys.readouterr().out


def test_from_source_without_data_config(monkeypatch, capsys):
    monkeypatch.setattr(m, "HAS_DATA_CONFIG", False)
    assert m.backtest_from_data_source("whatever") == {}
    assert "data_config.py required" in capsys.readouterr().out


def test_from_source_test_split_filters_dates(tmp_path, capsys):
    df = _frame(n=60, dates=True)
    # first 10 rows before the TEST window, last rows after it
    df["date"] = list(pd.date_range("2022-06-01", periods=10)) + list(
        pd.date_range("2023-01-02", periods=50)
    )
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    res = m.backtest_from_data_source(str(p), use_test_split=True)
    out = capsys.readouterr().out
    assert "Loaded 60 records from d.parquet" in out
    assert "Using TEST split: 2023-01-01 to 2024-06-30 (50 records)" in out
    assert "Companies: 50" in out
    assert set(res) == {"baseline", "optimized"}


def test_from_source_weight_set_and_no_split(tmp_path, capsys):
    p = tmp_path / "d.parquet"
    _frame(n=30, dates=True).to_parquet(p)
    res = m.backtest_from_data_source(str(p), use_test_split=False, weight_set="baseline")
    out = capsys.readouterr().out
    assert set(res) == {"baseline"}
    assert "Using all data (no split; not recommended for evaluation)" in out
    assert "Companies: 30" in out


def test_from_source_no_date_column(tmp_path, capsys):
    p = tmp_path / "d.parquet"
    _frame(n=30).to_parquet(p)
    res = m.backtest_from_data_source(str(p), use_test_split=True)
    assert "No date column found; using all data" in capsys.readouterr().out
    assert set(res) == {"baseline", "optimized"}


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="a capitalised 'Date' column passes the `'date' in cols or 'Date' in cols` guard, "
    "then filter_data_by_split(full_data, 'date') raises KeyError",
)
def test_from_source_capital_date_column(tmp_path):
    p = tmp_path / "d.parquet"
    df = _frame(n=30, dates=True).rename(columns={"date": "Date"})
    df.to_parquet(p)
    m.backtest_from_data_source(str(p), use_test_split=True)


# ------------------------------------------------------------------ __main__ script
def _run_script(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["backtest_weight_optimization.py", *argv])
    runpy.run_path(str(ROOT / "backtest_weight_optimization.py"), run_name="__main__")


def test_script_main_with_export(tmp_path, monkeypatch, capsys):
    data = tmp_path / "d.parquet"
    _frame(n=50, dates=True).to_parquet(data)
    out_csv = tmp_path / "res.csv"
    _run_script(monkeypatch, "--data", str(data), "--export", str(out_csv))
    out = capsys.readouterr().out
    assert "BACKTEST FRAMEWORK - KARZ vs MAIN WEIGHT COMPARISON" in out
    assert "✅ Backtest complete" in out
    assert out_csv.exists()
    assert len(pd.read_csv(out_csv)) == 2


def test_script_main_no_split_single_set_no_export(tmp_path, monkeypatch, capsys):
    data = tmp_path / "d.parquet"
    _frame(n=50).to_parquet(data)
    _run_script(monkeypatch, "--data", str(data), "--no-split", "--weight-set", "optimized")
    out = capsys.readouterr().out
    assert "Using all data (no split" in out
    assert "OPTIMIZED WEIGHTS" in out and "IMPROVEMENTS" not in out


def test_script_main_missing_data_skips_export(tmp_path, monkeypatch, capsys):
    out_csv = tmp_path / "res.csv"
    _run_script(monkeypatch, "--data", str(tmp_path / "none.parquet"), "--export", str(out_csv))
    assert not out_csv.exists()
    assert "✅ Backtest complete" in capsys.readouterr().out
