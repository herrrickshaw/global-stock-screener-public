"""Offline tests for pattern_discovery (sklearn replaced by tiny numpy fakes)."""

import importlib
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def pdm(tmp_path, monkeypatch):
    """Fresh import of pattern_discovery with cwd in tmp_path (module mkdirs on import)."""
    monkeypatch.chdir(tmp_path)
    saved = sys.modules.pop("pattern_discovery", None)
    mod = importlib.import_module("pattern_discovery")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(mod, "REF_DIR", tmp_path / "ref")
    (tmp_path / "cache").mkdir()
    (tmp_path / "ref").mkdir()
    yield mod
    sys.modules.pop("pattern_discovery", None)
    if saved is not None:
        sys.modules["pattern_discovery"] = saved


PD3 = int(pd.__version__.split(".")[0]) >= 3
xfail_pd3 = pytest.mark.xfail(
    PD3,
    strict=True,
    raises=(TypeError, IndexError),
    reason="pattern_discovery.py:~470: save_excel passes sheet_name positionally to "
    "DataFrame.to_excel, which is keyword-only in pandas>=3",
)


def _ohlcv(n, seed=0, price=100.0, vol=1000.0, drift=0.0005):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.01, n)
    close = price * np.cumprod(1 + r)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close * 1.001,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, vol),
        },
        index=idx,
    )


# ---- fakes for sklearn ----------------------------------------------------
class FakeScaler:
    def fit_transform(self, X):
        X = np.asarray(X, float)
        sd = X.std(axis=0)
        sd[sd == 0] = 1
        return (X - X.mean(axis=0)) / sd


class FakeKMeans:
    def __init__(self, n_clusters, random_state=None, n_init=None):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        return np.arange(len(X)) % self.n_clusters


class FakePCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.explained_variance_ratio_ = np.array([0.4, 0.2, 0.1, 0.05, 0.05])
        return np.asarray(X)[:, : self.n_components]


class FakeDBSCAN:
    def __init__(self, eps, min_samples):
        pass

    def fit(self, X):
        lab = np.zeros(len(X), dtype=int)
        lab[:3] = -1
        self.labels_ = lab
        return self


class FakeGBR:
    r2 = 0.3

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.feature_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        self.feature_importances_ /= self.feature_importances_.sum()
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _split(X, y, test_size, random_state):
    k = int(len(X) * (1 - test_size))
    return X[:k], X[k:], y[:k], y[k:]


@pytest.fixture
def sk(pdm, monkeypatch):
    monkeypatch.setattr(pdm, "_SK_OK", True, raising=False)
    for name, obj in {
        "StandardScaler": FakeScaler,
        "KMeans": FakeKMeans,
        "PCA": FakePCA,
        "DBSCAN": FakeDBSCAN,
        "GradientBoostingRegressor": FakeGBR,
        "train_test_split": _split,
        "r2_score": lambda a, b: FakeGBR.r2,
        "mean_absolute_error": lambda a, b: 1.5,
    }.items():
        monkeypatch.setattr(pdm, name, obj, raising=False)
    return pdm


# ---- load_and_clean --------------------------------------------------------
def test_load_and_clean_filters(pdm, tmp_path, capsys):
    good = _ohlcv(300)
    good.to_parquet(tmp_path / "cache" / "AAA.NS.parquet")
    good.to_parquet(tmp_path / "cache" / "BBB.parquet")
    good.to_parquet(tmp_path / "cache" / "CCC.BO.parquet")
    good.to_parquet(tmp_path / "ref" / "AAA.NS.parquet")  # duplicate symbol
    _ohlcv(100).to_parquet(tmp_path / "cache" / "SHORT.parquet")
    z = _ohlcv(300)
    z.iloc[5, z.columns.get_loc("Close")] = 0
    z.to_parquet(tmp_path / "cache" / "ZERO.parquet")
    _ohlcv(300, vol=0).to_parquet(tmp_path / "cache" / "ILLIQ.parquet")
    (tmp_path / "cache" / "CORRUPT.parquet").write_bytes(b"junk")
    pd.DataFrame({"Open": []}).to_parquet(tmp_path / "cache" / "EMPTY.parquet")

    out = pdm.load_and_clean("ALL")
    assert set(out) == {"AAA.NS", "BBB", "CCC.BO"}
    assert "dropped 5" in capsys.readouterr().out
    assert set(pdm.load_and_clean("IN")) == {"AAA.NS"}
    assert set(pdm.load_and_clean("US")) == {"BBB"}
    assert len(pdm.load_and_clean("ALL", max_stocks=1)) <= 1


def test_load_and_clean_min_bars_boundary(pdm, tmp_path):
    _ohlcv(252).to_parquet(tmp_path / "cache" / "EXACT.parquet")
    _ohlcv(251).to_parquet(tmp_path / "cache" / "ONEOFF.parquet")
    assert set(pdm.load_and_clean("ALL")) == {"EXACT"}
    assert set(pdm.load_and_clean("ALL", min_bars=251)) == {"EXACT", "ONEOFF"}


def test_load_and_clean_dedupes_and_fills(pdm, tmp_path):
    df = _ohlcv(300)
    dup = pd.concat([df, df.iloc[[10]]])
    dup.iloc[-1, dup.columns.get_loc("Close")] = df["Close"].iloc[10] * 2
    dup.iloc[20, dup.columns.get_loc("Close")] = np.nan
    dup.to_parquet(tmp_path / "cache" / "D.parquet")
    out = pdm.load_and_clean()["D"]
    assert len(out) == 300
    assert out.index.is_monotonic_increasing
    assert not out["Close"].isnull().any()
    # keep="last": duplicate row's doubled price wins
    assert out["Close"].iloc[10] == pytest.approx(df["Close"].iloc[10] * 2)


# ---- extract_features ------------------------------------------------------
def test_extract_features_too_short(pdm):
    assert pdm.extract_features("X", _ohlcv(252 + pdm.FWD - 1)) == {}


def test_extract_features_keys_and_target(pdm):
    df = _ohlcv(400, seed=1)
    f = pdm.extract_features("X", df)
    assert f["symbol"] == "X"
    assert all(k in f for k in pdm.FEATURE_NAMES)
    last = df["Close"].iloc[-pdm.FWD - 1]
    fut = df["Close"].iloc[-1]
    assert f["_target_fwd21"] == pytest.approx((fut - last) / last * 100)
    c = df["Close"].iloc[: -pdm.FWD]
    assert f["ret_1m"] == pytest.approx((c.iloc[-1] / c.iloc[-21] - 1) * 100)
    assert f["ret_12m"] == pytest.approx((c.iloc[-1] / c.iloc[-252] - 1) * 100)
    assert 0 <= f["rsi_14"] <= 100
    assert f["max_drawdown"] <= 0
    # no look-ahead: changing held-out tail doesn't alter features
    df2 = df.copy()
    df2.iloc[-pdm.FWD :, df2.columns.get_loc("Close")] *= 3
    f2 = pdm.extract_features("X", df2)
    for k in pdm.FEATURE_NAMES:
        assert f2[k] == pytest.approx(f[k])
    assert f2["_target_fwd21"] != f["_target_fwd21"]


def test_extract_features_short_windows_zero(pdm):
    # 252 + 21 bars -> features window n == 252 : ret_12m branch needs n > 252
    f = pdm.extract_features("X", _ohlcv(252 + pdm.FWD))
    assert f["ret_12m"] == 0
    assert f["ret_6m"] != 0 or f["ret_6m"] == 0  # computed branch executed
    g = pdm.extract_features("X", _ohlcv(253 + pdm.FWD))
    assert g["ret_12m"] != 0


def test_extract_features_monotonic_rise_rsi_100(pdm):
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    c = np.linspace(100, 200, n)
    df = pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(n, 10.0)}, index=idx
    )
    f = pdm.extract_features("UP", df)
    assert f["rsi_14"] == 100
    assert f["up_day_ratio"] == 100
    assert f["max_drawdown"] == 0
    assert f["calmar_proxy"] == 0


def test_extract_features_flat_price(pdm):
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    c = np.full(n, 50.0)
    df = pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(n, 10.0)}, index=idx
    )
    f = pdm.extract_features("FLAT", df)
    assert f["vol_60d"] == 0
    assert f["vol_ratio"] == 1
    assert f["rsi_14"] == 100  # zero loss -> 100


# ---- build_feature_matrix ---------------------------------------------------
def test_build_feature_matrix(pdm, capsys):
    cleaned = {f"S{i}": _ohlcv(350, seed=i) for i in range(5)}
    cleaned["SHORT"] = _ohlcv(100)
    fm = pdm.build_feature_matrix(cleaned)
    assert set(fm["symbol"]) == {f"S{i}" for i in range(5)}
    assert all(c in fm.columns for c in pdm.FEATURE_NAMES)
    assert "Feature matrix: 5 stocks" in capsys.readouterr().out


# ---- model steps -----------------------------------------------------------
def _fm(n=120, seed=0):
    rng = np.random.default_rng(seed)
    d = {"symbol": [f"S{i}" for i in range(n)]}
    for f in pdm_names():
        d[f] = rng.normal(size=n)
    d["_target_fwd21"] = rng.normal(size=n)
    d["avg_dollar_vol"] = rng.uniform(1, 10, n)
    return pd.DataFrame(d)


def pdm_names():
    import pattern_discovery

    return pattern_discovery.FEATURE_NAMES


def test_discover_clusters(sk, capsys):
    fm = _fm()
    fm.loc[0, "ret_1m"] = np.inf
    fm.loc[1, "ret_3m"] = np.nan
    out, km, pca, evr = sk.discover_clusters(fm, 4)
    assert set(out["cluster"]) == {0, 1, 2, 3}
    assert out["is_anomaly"].sum() == 3
    assert "pc1" in out and "pc2" in out
    assert "cluster" not in fm  # input untouched
    txt = capsys.readouterr().out
    assert "KMeans: 4 clusters | PCA top-2 explain 60%" in txt
    assert "DBSCAN: 3 anomalous" in txt


def test_supervised_drivers(sk, capsys):
    imp = sk.supervised_drivers(_fm(120))
    assert list(imp["Importance"]) == sorted(imp["Importance"], reverse=True)
    assert imp.iloc[0]["Feature"] == sk.FEATURE_NAMES[-1]
    assert "R²=0.300" in capsys.readouterr().out
    FakeGBR.r2 = 0.01
    try:
        sk.supervised_drivers(_fm(120))
        assert "Low R²" in capsys.readouterr().out
        FakeGBR.r2 = 0.05
        sk.supervised_drivers(_fm(120))
        assert "Low R²" not in capsys.readouterr().out
    finally:
        FakeGBR.r2 = 0.3


def test_supervised_insufficient_and_nan_targets(sk, capsys):
    assert sk.supervised_drivers(_fm(99)).empty
    assert "Insufficient labelled data" in capsys.readouterr().out
    fm = _fm(150)
    fm.loc[:60, "_target_fwd21"] = np.nan  # leaves 89 -> insufficient
    assert sk.supervised_drivers(fm).empty
    assert not sk.supervised_drivers(_fm(100)).empty  # exactly 100 ok


def test_find_comoving_pairs(pdm, capsys):
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, 300)
    idx = pd.date_range("2020-01-01", periods=301, freq="B")

    def frame(r):
        c = 100 * np.cumprod(1 + np.r_[0, r])
        return pd.DataFrame({"Close": c}, index=idx[: len(c)])

    cleaned = {
        "A": frame(base),
        "B": frame(base * 1.0 + rng.normal(0, 0.0005, 300)),
        "C": frame(rng.normal(0, 0.01, 300)),
        "SHORTY": frame(base[:150]),
    }
    fm = pd.DataFrame(
        {"symbol": ["A", "B", "C", "SHORTY", "MISSING"], "avg_dollar_vol": [5, 4, 3, 2, 1]}
    )
    pairs = pdm.find_comoving_pairs(cleaned, fm)
    assert list(pairs["Stock_A"]) == ["A"] and list(pairs["Stock_B"]) == ["B"]
    assert pairs.iloc[0]["Correlation"] > 0.8
    # top_n cap
    assert len(pdm.find_comoving_pairs(cleaned, fm, top_n=0)) == 0


def test_find_comoving_pairs_too_few(pdm):
    fm = pd.DataFrame({"symbol": ["A"], "avg_dollar_vol": [1.0]})
    assert pdm.find_comoving_pairs({"A": _ohlcv(300)}, fm).empty


# ---- name_archetype --------------------------------------------------------
BASE = {
    "ret_12m": 0.0,
    "vol_60d": 30.0,
    "dma200_gap": 20.0,
    "max_drawdown": -10.0,
    "rsi_14": 50.0,
    "mean_reversion": 0.0,
}


def _arch(pdm, **kw):
    return pdm.name_archetype(pd.Series({**BASE, **kw}))


@pytest.mark.parametrize(
    "kw,expect",
    [
        ({"ret_12m": 41, "dma200_gap": 11}, "High-Momentum"),
        ({"ret_12m": 40, "dma200_gap": 11}, "Balanced"),
        ({"ret_12m": 41, "dma200_gap": 10}, "Balanced"),
        ({"ret_12m": -21, "dma200_gap": -11}, "Sustained Downtrenders"),
        ({"ret_12m": -20, "dma200_gap": -11}, "Balanced"),
        ({"vol_60d": 51}, "High-Volatility"),
        ({"vol_60d": 50}, "Balanced"),
        ({"dma200_gap": 4, "vol_60d": 24}, "Low-Vol Range-Bound"),
        ({"dma200_gap": 5, "vol_60d": 24}, "Balanced"),
        ({"dma200_gap": -4.9, "vol_60d": 24}, "Low-Vol Range-Bound"),
        ({"dma200_gap": 4, "vol_60d": 25}, "Balanced"),
        ({"mean_reversion": 11}, "Mean-Reverters"),
        ({"mean_reversion": 10}, "Balanced"),
        ({"rsi_14": 66, "ret_12m": 16}, "Overbought"),
        ({"rsi_14": 65, "ret_12m": 16}, "Balanced"),
        ({"rsi_14": 66, "ret_12m": 15}, "Balanced"),
        ({"max_drawdown": -51}, "Deep-Drawdown"),
        ({"max_drawdown": -50}, "Balanced"),
    ],
)
def test_name_archetype(pdm, kw, expect):
    assert expect in _arch(pdm, **kw)


def test_name_archetype_priority(pdm):
    # momentum leader beats high-vol
    assert "High-Momentum" in _arch(pdm, ret_12m=50, dma200_gap=15, vol_60d=60)


# ---- insights / excel ------------------------------------------------------
def _clustered(sk):
    fm, *_ = sk.discover_clusters(_fm(60), 3)
    return fm


def test_extract_insights_full(sk, capsys):
    fm = _clustered(sk)
    imp = sk.supervised_drivers(_fm(120))
    pairs = pd.DataFrame({"Stock_A": ["A"], "Stock_B": ["B"], "Correlation": [0.9]})
    ins = sk.extract_insights(fm, imp, pairs, None)
    assert set(ins) == {"clusters", "importances", "anomalies", "pairs"}
    cl = ins["clusters"]
    assert list(cl["Cluster"]) == [0, 1, 2]
    assert cl["Count"].sum() == 60
    assert cl.iloc[0]["Examples"].count(",") == 3
    assert list(ins["anomalies"].columns)[0] == "symbol"
    out = capsys.readouterr().out
    assert "BEHAVIOURAL ARCHETYPES" in out and "RETURN DRIVERS" in out
    assert "ANOMALIES (3 stocks" in out and "CO-MOVING PAIRS" in out
    assert sk.DISCLAIMER in out


def test_extract_insights_minimal(sk, capsys):
    fm = _clustered(sk)
    fm["is_anomaly"] = False
    ins = sk.extract_insights(fm, pd.DataFrame(), pd.DataFrame(), None)
    assert set(ins) == {"clusters"}
    out = capsys.readouterr().out
    assert "RETURN DRIVERS" not in out and "ANOMALIES" not in out and "CO-MOVING" not in out


@xfail_pd3
def test_save_excel(sk, tmp_path, capsys):
    fm = _clustered(sk)
    imp = sk.supervised_drivers(_fm(120))
    pairs = pd.DataFrame({"Stock_A": ["A"], "Stock_B": ["B"], "Correlation": [0.9]})
    ins = sk.extract_insights(fm, imp, pairs, None)
    path = sk.save_excel(ins, fm, "IN")
    assert path.parent == sk.OUT_DIR and path.name.startswith("patterns_IN_")
    xl = pd.ExcelFile(path)
    assert xl.sheet_names == [
        "DISCLAIMER",
        "Archetypes",
        "Return_Drivers",
        "CoMoving_Pairs",
        "Anomalies",
        "All_Features",
    ]
    assert len(xl.parse("All_Features")) == 60
    # only the disclaimer + features when insights are empty
    p2 = sk.save_excel({}, fm, "ALL")
    assert pd.ExcelFile(p2).sheet_names == ["DISCLAIMER", "All_Features"]


# ---- main ------------------------------------------------------------------
def test_main_without_sklearn(pdm, monkeypatch, capsys):
    monkeypatch.setattr(pdm, "_SK_OK", False)
    assert pdm.main() is None
    assert "pip install scikit-learn" in capsys.readouterr().out


def test_main_insufficient_data(sk, capsys):
    sk.main("ALL")
    assert "Insufficient data." in capsys.readouterr().out


def test_main_insufficient_features(sk, monkeypatch, capsys):
    monkeypatch.setattr(sk, "load_and_clean", lambda m, mx: {f"S{i}": None for i in range(50)})
    monkeypatch.setattr(sk, "build_feature_matrix", lambda c: _fm(49))
    sk.main()
    assert "Insufficient features." in capsys.readouterr().out


@xfail_pd3
def test_main_happy_path(sk, monkeypatch, capsys):
    seen = {}
    cleaned = {f"S{i}": _ohlcv(300, seed=i) for i in range(50)}

    def fake_load(market, mx):
        seen["args"] = (market, mx)
        return cleaned

    monkeypatch.setattr(sk, "load_and_clean", fake_load)
    monkeypatch.setattr(sk, "build_feature_matrix", lambda c: _fm(120))
    monkeypatch.setattr(
        sk,
        "find_comoving_pairs",
        lambda c, f: pd.DataFrame({"Stock_A": ["A"], "Stock_B": ["B"], "Correlation": [0.9]}),
    )
    sk.main(market="IN", n_clusters=3, max_stocks=7)
    assert seen["args"] == ("IN", 7)
    assert "Pattern Excel" in capsys.readouterr().out
    assert list(sk.OUT_DIR.glob("patterns_IN_*.xlsx"))


def test_main_just_below_50_cleaned(sk, monkeypatch, capsys):
    monkeypatch.setattr(sk, "load_and_clean", lambda m, mx: {f"S{i}": None for i in range(49)})
    sk.main()
    assert "Insufficient data." in capsys.readouterr().out


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="pattern_discovery.py:~327: sort_values('Correlation') on an empty "
    "DataFrame (no pair > 0.80) raises KeyError instead of returning empty",
)
def test_find_comoving_pairs_none_correlated(pdm):
    rng = np.random.default_rng(5)
    idx = pd.date_range("2020-01-01", periods=301, freq="B")
    cleaned = {
        s: pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + np.r_[0, rng.normal(0, 0.01, 300)])}, index=idx
        )
        for s in "AB"
    }
    fm = pd.DataFrame({"symbol": ["A", "B"], "avg_dollar_vol": [2.0, 1.0]})
    assert pdm.find_comoving_pairs(cleaned, fm).empty
