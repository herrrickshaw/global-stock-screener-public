import importlib
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "f1_hyperparameter_tuning.py"


def _counts(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return (
        int(((y == 1) & (p == 1)).sum()),
        int(((y == 0) & (p == 1)).sum()),
        int(((y == 1) & (p == 0)).sum()),
        int(((y == 0) & (p == 0)).sum()),
    )


def _prec(y, p, zero_division=0):
    tp, fp, _, _ = _counts(y, p)
    return tp / (tp + fp) if tp + fp else zero_division


def _rec(y, p, zero_division=0):
    tp, _, fn, _ = _counts(y, p)
    return tp / (tp + fn) if tp + fn else zero_division


def _f1(y, p, zero_division=0):
    a, b = _prec(y, p), _rec(y, p)
    return 2 * a * b / (a + b) if a + b else zero_division


def _cm(y, p):
    tp, fp, fn, tn = _counts(y, p)
    return np.array([[tn, fp], [fn, tp]])


@pytest.fixture
def mod(monkeypatch):
    sk = types.ModuleType("sklearn")
    met = types.ModuleType("sklearn.metrics")
    met.f1_score, met.precision_score, met.recall_score = _f1, _prec, _rec
    met.confusion_matrix = _cm
    monkeypatch.setitem(sys.modules, "sklearn", sk)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", met)
    monkeypatch.delitem(sys.modules, "f1_hyperparameter_tuning", raising=False)
    m = importlib.import_module("f1_hyperparameter_tuning")
    yield m
    sys.modules.pop("f1_hyperparameter_tuning", None)


def _data(n=100, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(5, 10, n)
    d = {"stock_return_12m": ret}
    for i, dim in enumerate(
        [
            "debt_expansion", "capex_acceleration", "profit_reinvestment",
            "profitability_quality", "sustainability", "timing_alignment",
            "leverage_health", "fcf_generation",
        ]
    ):  # fmt: skip
        d[dim] = ret * (1 if i % 2 == 0 else 0.2) + rng.normal(0, 3, n)
    return pd.DataFrame(d)


def test_init_split(mod, capsys):
    np.random.seed(1)
    t = mod.F1HyperparameterTuner(_data(100), train_size=0.7)
    assert len(t.train) == 70 and len(t.test) == 30
    assert t.train["outperform"].isin([0, 1]).all()
    # strictly greater than median
    med = t.train["stock_return_12m"].median()
    assert (t.train["outperform"] == (t.train["stock_return_12m"] > med)).all()
    out = capsys.readouterr().out
    assert "Train: 70 samples (70.0%)" in out and "Test:  30 samples (30.0%)" in out
    assert t.results == []
    t2 = mod.F1HyperparameterTuner(_data(10), train_size=0.5)
    assert len(t2.train) == 5 and len(t2.test) == 5


def test_generate_scores(mod):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(20))
    df = pd.DataFrame(
        {"debt_expansion": [1.0, 2.0], "capex_acceleration": [10.0, 20.0], "other": [9, 9]}
    )
    s = t.generate_8d_scores(df, {"debt_expansion": 2, "capex_acceleration": 0.5, "other": 100})
    assert s.tolist() == [7.0, 14.0]
    # missing weight -> 0; no matching columns -> zeros
    assert t.generate_8d_scores(df, {}).tolist() == [0.0, 0.0]
    assert t.generate_8d_scores(pd.DataFrame({"x": [1, 2]}), {"debt_expansion": 1}).tolist() == [
        0,
        0,
    ]
    # *_score column is preferred when the bare column is also present
    both = pd.DataFrame({"debt_expansion": [100.0], "debt_expansion_score": [1.0]})
    assert t.generate_8d_scores(both, {"debt_expansion": 3}).tolist() == [3.0]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="source bug f1_hyperparameter_tuning.py:75: `if dim in df.columns` is checked "
    "before the *_score fallback, so frames with only <dim>_score columns score as all zeros",
)
def test_generate_scores_score_suffix_only(mod):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(20))
    df = pd.DataFrame({"debt_expansion_score": [1.0, 2.0]})
    assert t.generate_8d_scores(df, {"debt_expansion": 2}).tolist() == [2.0, 4.0]


def test_random_search(mod, capsys):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(200, 4))
    capsys.readouterr()
    r = t.random_search_weights(n_iterations=30, top_k=5)
    assert len(r) == 5
    f = [x["test_f1"] for x in r]
    assert f == sorted(f, reverse=True)
    for x in r:
        assert sum(x["weights"].values()) == pytest.approx(100)
        assert set(x["weights"]) == set(t.DIMENSIONS)
        assert x["overfitting_gap"] == pytest.approx(abs(x["train_f1"] - x["test_f1"]))
        assert 0 <= x["test_precision"] <= 1
    out = capsys.readouterr().out
    assert "Iterations: 30" in out and "Tested 30 random combinations" in out
    # deterministic: seeded inside
    r2 = t.random_search_weights(n_iterations=30, top_k=5)
    assert [x["weights"] for x in r2] == [x["weights"] for x in r]
    assert len(t.random_search_weights(n_iterations=3, top_k=100)) == 3
    assert len(t.random_search_weights(n_iterations=1, top_k=100)) == 1


def test_random_search_metric_failure_zeroed(mod, monkeypatch):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(50))

    def boom(*a, **k):
        raise ValueError("x")

    monkeypatch.setattr(mod, "f1_score", boom)
    r = t.random_search_weights(n_iterations=2, top_k=2)
    assert all(x["train_f1"] == 0 and x["test_f1"] == 0 and x["overfitting_gap"] == 0 for x in r)
    assert all(x["train_precision"] == 0 for x in r)


def test_zero_total_weight_skipped(mod, monkeypatch):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(40))
    monkeypatch.setattr(mod.np.random, "uniform", lambda a, b: 0.0)
    assert t.random_search_weights(n_iterations=3, top_k=5) == []


def test_run_tuning_and_recommend(mod):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(100))
    w, top = t.run_tuning(n_iterations=10)
    assert w == top[0]["weights"] == t.recommend_weights(top)
    assert len(top) == 10


def _res(gap, **k):
    w = {d: 12.5 for d in mod_dims()}
    r = {
        "weights": w, "test_f1": 0.6, "test_precision": 0.7, "test_recall": 0.5,
        "train_f1": 0.6 + gap, "overfitting_gap": gap,
    }  # fmt: skip
    r.update(k)
    return r


def mod_dims():
    return [
        "debt_expansion", "capex_acceleration", "profit_reinvestment", "profitability_quality",
        "sustainability", "timing_alignment", "leverage_health", "fcf_generation",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "gap,msg",
    [(0.0499, "LOW OVERFITTING"), (0.05, "MODERATE OVERFITTING"), (0.1499, "MODERATE"),
     (0.15, "HIGH OVERFITTING")],
)  # fmt: skip
def test_report_overfit_labels(mod, capsys, gap, msg):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(40))
    capsys.readouterr()
    top = [_res(gap)] * 12
    t.generate_report(top[0]["weights"], top)
    out = capsys.readouterr().out
    assert msg in out
    assert "TOTAL: 100.0" in out
    assert "Test F1 Score:    0.6000" in out and "Precision:  70.0%" in out
    assert out.count("   10  ") == 1 and "   11  " not in out  # top 10 only
    assert "28 held-out" in out or "12 held-out" in out


def test_report_weight_change_formatting(mod, capsys):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(40))
    capsys.readouterr()
    w = {d: 0.0 for d in mod_dims()}
    w["debt_expansion"] = 20.05  # |change| 0.05 -> arrow
    w["capex_acceleration"] = 25.0  # +5.0
    w["fcf_generation"] = 0.0  # change 0 -> arrow
    w["leverage_health"] = 0.0  # -5.0
    top = [_res(0.0, weights=w)]
    t.generate_report(w, top)
    out = capsys.readouterr().out
    line = {
        ln.split()[0]: ln
        for ln in out.splitlines()
        if ln.strip().split()[:1] and ln.strip().split()[0] in w
    }
    assert line["debt_expansion"].rstrip().endswith("→")
    assert line["capex_acceleration"].rstrip().endswith("+5.0")
    assert line["leverage_health"].rstrip().endswith("-5.0")
    assert line["fcf_generation"].rstrip().endswith("→")


def test_compare_with_baseline(mod, capsys):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(200, 9))
    capsys.readouterr()
    good = {"debt_expansion": 1.0, "profit_reinvestment": 1.0}
    bad = {"debt_expansion": -1.0, "profit_reinvestment": -1.0}
    t.compare_with_baseline(good, bad)
    out = capsys.readouterr().out
    assert "OPTIMIZED WEIGHTS" in out and "BASELINE WEIGHTS" in out
    assert "Use optimized weights" in out and "Improvement: +" in out
    t.compare_with_baseline(bad, good)
    out = capsys.readouterr().out
    assert "Stick with baseline" in out
    # identical -> improvement 0 -> not > 0
    t.compare_with_baseline(good, good)
    out = capsys.readouterr().out
    assert "Improvement: +0.0%" in out and "Stick with baseline" in out


def test_compare_baseline_f1_zero(mod, capsys, monkeypatch):
    np.random.seed(0)
    t = mod.F1HyperparameterTuner(_data(60, 2))
    t.test["outperform"] = 0  # no positives: f1 = 0 for any weights
    capsys.readouterr()
    monkeypatch.setattr(mod, "confusion_matrix", lambda y, p: np.array([[1, 2], [3, 4]]))
    t.compare_with_baseline({"debt_expansion": 1.0}, {"debt_expansion": 1.0})
    out = capsys.readouterr().out
    assert "Improvement: +0.0%" in out
    assert "True Positives:  4" in out and "False Negatives: 3" in out
    assert "Accuracy:      0.5000" in out


def test_main_block(mod, capsys):
    runpy.run_path(str(SRC), run_name="__main__")
    out = capsys.readouterr().out
    assert "F1-BASED HYPERPARAMETER TUNING FOR WEIGHT OPTIMIZATION" in out
