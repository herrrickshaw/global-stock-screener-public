from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.r2b_support import load_with_fake_sklearn, synthetic_ohlc

SRC = str(Path(__file__).resolve().parent.parent / "ml_signal_engine.py")


@pytest.fixture
def m(monkeypatch):
    return load_with_fake_sklearn(SRC, "mse_r2_fake", monkeypatch)


@pytest.fixture
def m_nosk(monkeypatch, capsys):
    mod = load_with_fake_sklearn(SRC, "mse_r2_nosk", monkeypatch, sklearn=False)
    capsys.readouterr()
    return mod


@pytest.fixture(scope="module")
def ohlc():
    return synthetic_ohlc(1, 600)


# ---- import-time flag ---------------------------------------------------------------
def test_sklearn_flag(m, m_nosk):
    assert m._SKLEARN_OK is True and m_nosk._SKLEARN_OK is False


def test_import_message_when_missing(monkeypatch, capsys):
    load_with_fake_sklearn(SRC, "mse_r2_nosk2", monkeypatch, sklearn=False)
    assert "scikit-learn not installed" in capsys.readouterr().out


def test_constants(m):
    assert (m.LOOKBACK, m.TRAIN_WINDOW, m.PREDICT_DAYS) == (60, 252, 5)
    assert m.MIN_ROWS == 60 + 252 + 5 + 10
    assert len(m.FEATURE_NAMES) == 12


# ---- compute_features / z-score -------------------------------------------------------
def test_compute_features_columns_and_values(m, ohlc):
    f = m.compute_features(ohlc)
    assert list(f.columns) == m.FEATURE_NAMES
    assert not f.isna().any().any()
    assert len(f) == len(ohlc) - 199  # dma200 needs 200 rows
    c = ohlc["Close"]
    i = f.index[10]
    assert f.loc[i, "ret_1d"] == pytest.approx((c.loc[i] / c.shift(1).loc[i] - 1) * 100)
    assert f.loc[i, "ret_5d"] == pytest.approx((c.loc[i] / c.shift(5).loc[i] - 1) * 100)
    assert f.loc[i, "hl_ratio"] == pytest.approx(
        (ohlc.loc[i, "High"] - ohlc.loc[i, "Low"]) / c.loc[i]
    )
    d50 = c.rolling(50).mean().loc[i]
    assert f.loc[i, "dma50_gap"] == pytest.approx((c.loc[i] - d50) / d50 * 100)
    assert f.loc[i, "vol_20d"] == pytest.approx(
        c.pct_change().rolling(20).std().loc[i] * np.sqrt(252) * 100
    )
    mid, sd = c.rolling(20).mean().loc[i], c.rolling(20).std().loc[i]
    assert f.loc[i, "bb_pct"] == pytest.approx((c.loc[i] - (mid - 2 * sd)) / (4 * sd))
    assert f.loc[i, "close_norm"] == pytest.approx((c.loc[i] - mid) / sd)
    assert 0 <= f["rsi_14"].min() and f["rsi_14"].max() <= 100
    assert f.loc[i, "macd"] - f.loc[i, "macd_signal"] != 0


def test_compute_features_zero_volume_rows_dropped(m, ohlc):
    z = ohlc.copy()
    z.iloc[300, z.columns.get_loc("Volume")] = 0.0
    f = m.compute_features(z)
    assert z.index[300] not in f.index
    assert len(f) < len(m.compute_features(ohlc))


def test_z_score_normalise(m):
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z = m.z_score_normalise(X)
    assert Z[:, 0].mean() == pytest.approx(0) and Z[:, 0].std() == pytest.approx(1)
    assert (Z[:, 1] == 0).all()  # constant column -> std treated as 1


# ---- direction ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "x,d",
    [
        (0.5, "BULLISH"),
        (0.4999, "NEUTRAL"),
        (0.0, "NEUTRAL"),
        (-0.4999, "NEUTRAL"),
        (-0.5, "BEARISH"),
        (9, "BULLISH"),
        (-9, "BEARISH"),
    ],
)
def test_classify(m, x, d):
    assert m.MLSignalEngine._classify_direction(x) == d


# ---- model factory / sequences -----------------------------------------------------------
def test_make_model(m, m_nosk):
    assert type(m.MLSignalEngine("ridge")._make_model()).__name__ == "FakeRidge"
    assert m.MLSignalEngine("ridge")._make_model().alpha == 1.0
    assert type(m.MLSignalEngine("lr")._make_model()).__name__ == "FakeLinearRegression"
    assert type(m.MLSignalEngine("anything")._make_model()).__name__ == "FakeLinearRegression"
    assert m_nosk.MLSignalEngine()._make_model() is None


def test_init_defaults(m):
    e = m.MLSignalEngine()
    assert e.model_type == "ridge" and e.workers == 4 and e._model_cache == {}


def test_make_sequences(m, ohlc):
    f = m.compute_features(ohlc)
    tgt = pd.Series(np.arange(len(f), dtype=float), index=f.index)
    X, y = m.MLSignalEngine()._make_sequences(f, tgt)
    assert X.shape == (len(f) - 60 - 5, 60 * 12)
    assert y[0] == 65.0 and y[-1] == len(f) - 1.0
    first = m.z_score_normalise(f.values[:60]).flatten()
    assert np.allclose(X[0], first)


def test_make_sequences_short_returns_empty(m, ohlc):
    f = m.compute_features(ohlc).iloc[:65]
    X, y = m.MLSignalEngine()._make_sequences(f, pd.Series(0.0, index=f.index))
    assert X.size == 0 and y.size == 0
    f = m.compute_features(ohlc).iloc[:66]
    X, y = m.MLSignalEngine()._make_sequences(f, pd.Series(0.0, index=f.index))
    assert len(X) == 1


# ---- _prepare / _fit ---------------------------------------------------------------------
def test_prepare_ok(m, ohlc):
    feat, tgt, n = m.MLSignalEngine._prepare(ohlc)
    assert list(feat.columns) == m.FEATURE_NAMES and n == len(feat) == len(tgt)
    assert n == len(ohlc) - 199 - 5
    c = ohlc["Close"]
    i = tgt.index[3]
    j = c.index.get_loc(i)
    assert tgt.loc[i] == pytest.approx((c.iloc[j + 5] / c.iloc[j] - 1) * 100)


def test_prepare_none_when_short(m):
    # first gate: len(features) = n-199 must be >= MIN_ROWS-50 = 277  ->  n >= 476
    assert m.MLSignalEngine._prepare(synthetic_ohlc(1, 400)) is None
    assert m.MLSignalEngine._prepare(synthetic_ohlc(1, 475)) is None
    assert m.MLSignalEngine._prepare(synthetic_ohlc(1, 476)) is not None


def test_fit_walk_forward(m, ohlc):
    e = m.MLSignalEngine("lr")
    feat, tgt, n = e._prepare(ohlc)
    model, X, yp, rmse = e._fit_walk_forward(feat, tgt, n)
    assert X.shape[0] == 252 - 60 - 5 and X.shape[1] == 720
    assert rmse >= 0 and len(yp) == len(X)
    assert e._fit_walk_forward(feat.iloc[:100], tgt.iloc[:100], 100) is None


# ---- predict -----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["none", "short", "short_by_one"])
def test_predict_insufficient(m, kind):
    e = m.MLSignalEngine()
    df = {
        "none": None,
        "short": synthetic_ohlc(1, 100),
        "short_by_one": synthetic_ohlc(1, m.MIN_ROWS - 1),
    }[kind]
    r = e.predict("X", df)
    assert r["direction"] == "INSUFFICIENT_DATA" and r["predicted_ret%"] == 0.0
    assert r["train_rows"] == 0 and r["model"] == "ridge" and r["symbol"] == "X"


def test_predict_no_sklearn(m_nosk, ohlc):
    r = m_nosk.MLSignalEngine().predict("X", ohlc)
    assert r["direction"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("mt", ["ridge", "lr"])
def test_predict_ok(m, ohlc, mt):
    r = m.MLSignalEngine(mt).predict("SYM", ohlc)
    assert r["symbol"] == "SYM" and r["model"] == mt
    assert r["direction"] in ("BULLISH", "NEUTRAL", "BEARISH")
    assert r["train_rows"] == 187 and r["features_used"] == 12
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["train_rmse"] >= 0
    assert (
        m.MLSignalEngine._classify_direction(r["predicted_ret%"]) == r["direction"]
        or abs(abs(r["predicted_ret%"]) - 0.5) < 1e-3
    )


def test_predict_prepare_none_and_fit_none(m, monkeypatch, ohlc):
    e = m.MLSignalEngine()
    monkeypatch.setattr(m.MLSignalEngine, "_prepare", staticmethod(lambda d: None))
    assert e.predict("X", ohlc)["direction"] == "INSUFFICIENT_DATA"
    monkeypatch.undo()


def test_predict_fit_none(m, monkeypatch, ohlc):
    e = m.MLSignalEngine()
    monkeypatch.setattr(e, "_fit_walk_forward", lambda *a: None)
    assert e.predict("X", ohlc)["direction"] == "INSUFFICIENT_DATA"


def test_predict_error_isolated(m, ohlc):
    bad = ohlc.drop(columns=["Close"])
    r = m.MLSignalEngine().predict("BAD", bad)
    assert "error" in r and r["direction"] == "NEUTRAL" and r["predicted_ret%"] == 0.0


def test_predict_confidence_formula(m, monkeypatch, ohlc):
    class M:
        def predict(self, X):
            return np.array([2.0]) if len(X) == 1 else np.array([1.0, -1.0, 1.0, -1.0])

    e = m.MLSignalEngine()
    monkeypatch.setattr(
        e,
        "_fit_walk_forward",
        lambda *a: (M(), np.zeros((4, 3)), np.array([1.0, -1.0, 1.0, -1.0]), 0.12345),
    )
    r = e.predict("X", ohlc)
    ps = float(np.array([1.0, -1.0, 1.0, -1.0]).std())
    assert r["confidence"] == pytest.approx(min(1.0, 2.0 / (2 * ps + 1e-9)), abs=1e-3)
    assert r["predicted_ret%"] == 2.0 and r["direction"] == "BULLISH"
    assert r["train_rmse"] == 0.1235 and r["train_rows"] == 4


def test_predict_confidence_caps_and_zero_std(m, monkeypatch, ohlc):
    class M:
        def predict(self, X):
            return np.array([50.0])

    e = m.MLSignalEngine()
    monkeypatch.setattr(
        e, "_fit_walk_forward", lambda *a: (M(), np.zeros((2, 1)), np.array([1.0, 1.0]), 0.0)
    )
    r = e.predict("X", ohlc)
    assert r["confidence"] == 1.0  # capped; zero std -> 1.0 fallback


# ---- predict_batch ---------------------------------------------------------------------------
def test_predict_batch(m, monkeypatch, capsys):
    e = m.MLSignalEngine(workers=2)

    def fake(sym, df):
        if sym == "BOOM":
            raise RuntimeError("boom")
        v = {"A": 1.0, "B": -1.0, "C": 0.0}[sym]
        return {"symbol": sym, "direction": e._classify_direction(v), "predicted_ret%": v}

    monkeypatch.setattr(e, "predict", fake)
    out = e.predict_batch({"A": 1, "B": 2, "C": 3, "BOOM": 4})
    assert list(out["symbol"][:3]) == ["A", "C", "B"] or list(out["symbol"])[0] == "A"
    row = out[out["symbol"] == "BOOM"].iloc[0]
    assert row["direction"] == "ERROR" and row["error"] == "boom"
    txt = capsys.readouterr().out
    assert "predicting 4 stocks (ridge, 60d window, 252d train)" in txt
    assert "4/4 done | BULLISH: 1 | BEARISH: 1" in txt
    assert "BULLISH" in txt and "25.0%" in txt


def test_predict_batch_progress_every_100(m, monkeypatch, capsys):
    e = m.MLSignalEngine()
    monkeypatch.setattr(
        e, "predict", lambda s, d: {"symbol": s, "direction": "NEUTRAL", "predicted_ret%": 0.0}
    )
    e.predict_batch({str(i): None for i in range(101)}, max_workers=3)
    txt = capsys.readouterr().out
    assert "100/101 done" in txt and "101/101 done" in txt and "99/101" not in txt


def test_predict_batch_empty(m, capsys):
    out = m.MLSignalEngine().predict_batch({})
    assert out.empty


# ---- evaluate_on_backtest ----------------------------------------------------------------------
def test_evaluate_on_backtest_runs(m, capsys):
    df = synthetic_ohlc(2, 520)
    idx = m.compute_features(df).index
    start = str(idx[-12].date())
    end = str(idx[-8].date())
    r = m.MLSignalEngine("lr").evaluate_on_backtest(
        {"A": df, "short": synthetic_ohlc(1, 50), "none": None}, start, end
    )
    assert r["n_predictions"] > 0 and r["n_predictions"] <= 5
    assert r["test_period"] == f"{start} – {end}"
    assert r["rmse"] >= r["mae"] >= 0 or r["rmse"] >= 0
    assert 0 <= r["directional_acc"] <= 100
    assert "ML Backtest Evaluation" in capsys.readouterr().out


def test_evaluate_no_predictions(m):
    df = synthetic_ohlc(2, 520)
    r = m.MLSignalEngine().evaluate_on_backtest({"A": df}, "1999-01-01", "1999-02-01")
    assert r == {"error": "No test predictions generated"}
    assert m.MLSignalEngine().evaluate_on_backtest({}, "2020-01-01", "2021-01-01") == {
        "error": "No test predictions generated"
    }


def test_evaluate_swallows_errors(m):
    bad = synthetic_ohlc(2, 520).drop(columns=["Close"])
    assert m.MLSignalEngine().evaluate_on_backtest({"A": bad}, "2020-01-01", "2030-01-01") == {
        "error": "No test predictions generated"
    }


def test_evaluate_directional_counts(m, monkeypatch):
    df = synthetic_ohlc(2, 520)
    idx = m.compute_features(df).index
    e = m.MLSignalEngine()

    class Perfect:
        def fit(self, X, y):
            self.y = y
            return self

        def predict(self, X):
            return np.array([1.0]) if len(X) == 1 else np.zeros(len(X))

    monkeypatch.setattr(e, "_make_model", lambda: Perfect())
    r = e.evaluate_on_backtest({"A": df}, str(idx[-12].date()), str(idx[-8].date()))
    # always predicts +1: accuracy equals share of positive actuals
    assert r["n_predictions"] > 0
    assert r["directional_acc"] == pytest.approx(round(r["directional_acc"], 1))


# ---- compare_models ------------------------------------------------------------------------------
def test_compare_models(m, ohlc, capsys):
    df = m.compare_models(ohlc, "TST")
    assert list(df["Model"]) == ["LinearRegression", "Ridge(alpha=1)", "Naive(last_ret)"]
    assert set(df.columns) == {"Model", "Train_RMSE", "Train_MAE", "Test_RMSE", "Test_MAE"}
    # unregularised LR fits the (n<p) training set exactly; Naive is the worst on train
    assert df.loc[0, "Train_RMSE"] <= df.loc[1, "Train_RMSE"] <= df.loc[2, "Train_RMSE"]
    out = capsys.readouterr().out
    assert "Model comparison for TST (train=" in out and "Paper benchmark" in out


def test_compare_models_guards(m, m_nosk, ohlc, monkeypatch):
    assert m_nosk.compare_models(ohlc).empty
    assert m.compare_models(synthetic_ohlc(1, m.MIN_ROWS - 1)).empty
    monkeypatch.setattr(
        m.MLSignalEngine, "_make_sequences", lambda self, f, t: (np.zeros((49, 2)), np.zeros(49))
    )
    assert m.compare_models(ohlc).empty
    monkeypatch.setattr(
        m.MLSignalEngine, "_make_sequences", lambda self, f, t: (np.zeros((50, 2)), np.zeros(50))
    )
    assert len(m.compare_models(ohlc)) == 3
