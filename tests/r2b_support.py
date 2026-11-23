"""Synthetic, deterministic price / fundamentals panels for the group-B (reward_*) tests."""

import datetime
import sys
import types

import numpy as np
import pandas as pd


def make_prices(n_sym=120, start="2015-01-01", end="2025-12-31", seed=7, short=2, prefix="S"):
    """Long-format price frame (Date, Close, High, Volume, Symbol).

    Drift falls with volatility (so low-vol and momentum screens pay), volume level varies
    by symbol, the first `short` symbols have <300 rows so they are skipped.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    sig = np.linspace(0.006, 0.03, n_sym)
    rng.shuffle(sig)
    mu = 0.0012 - 0.045 * sig
    ret = mu[None, :] + sig[None, :] * rng.standard_normal((n, n_sym))
    close = 20.0 * np.exp(np.cumsum(ret, axis=0))
    vol = 1e5 * (1 + np.arange(n_sym)[None, :] / 10.0) * (1 + 0.3 * rng.random((n, n_sym)))
    frames = []
    for k in range(n_sym):
        sl = slice(None) if k >= short else slice(0, 100)
        d = dates[sl]
        frames.append(
            pd.DataFrame(
                {
                    "Date": d,
                    "Close": close[sl, k],
                    "High": close[sl, k] * 1.01,
                    "Volume": vol[sl, k],
                    "Symbol": f"{prefix}{k:03d}",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_fundamentals(prices, seed=3):
    """Annual fundamentals for each long-history symbol: fy_end Dec 31, filed = +90d."""
    rng = np.random.default_rng(seed)
    syms = prices.groupby("Symbol").size()
    syms = syms[syms > 300].index
    rows = []
    for k, s in enumerate(syms):
        base = rng.uniform(0.2, 1.0)
        for fy in range(2014, 2025):
            fy_end = pd.Timestamp(fy, 12, 31)
            ta = 1000.0 * (1 + 0.05 * (fy - 2014)) * (1 + 0.2 * rng.random())
            rev = ta * rng.uniform(0.5, 1.5)
            ni = ta * (0.02 + 0.1 * (1 - k / len(syms)) * base + 0.01 * rng.standard_normal())
            rows.append(
                {
                    "ticker": s,
                    "fy_end": fy_end,
                    "filed": (datetime.date(fy, 12, 31) + datetime.timedelta(days=90)).isoformat(),
                    "net_income": ni,
                    "total_assets": ta,
                    "revenue": rev,
                    "equity": ta * rng.uniform(0.2, 0.6),
                    "current_assets": ta * 0.3,
                    "current_liabilities": ta * rng.uniform(0.1, 0.3),
                    "long_term_debt": ta * rng.uniform(0.05, 0.3),
                    "gross_profit": rev * rng.uniform(0.2, 0.5),
                    "ebit": ni * 1.3,
                    "cfo": ni * rng.uniform(0.8, 1.6),
                    "shares": rng.uniform(5.0, 50.0),
                }
            )
    return pd.DataFrame(rows)


def fake_warehouse(prices=None, funds=None, calls=None):
    """A stand-in for the `warehouse` module (duckdb-backed in production)."""
    mod = types.ModuleType("warehouse")

    def _px(market):
        if calls is not None:
            calls.append(("prices", market))
        if prices is None or market not in prices:
            raise SystemExit(f"no warehouse prices for {market}")
        return prices[market].copy()

    def _fu(market):
        if calls is not None:
            calls.append(("fundamentals", market))
        if funds is None or market not in funds:
            return pd.DataFrame()
        return funds[market].copy()

    mod.prices = _px
    mod.fundamentals = _fu
    return mod


def install_warehouse(monkeypatch, **kw):
    mod = fake_warehouse(**kw)
    monkeypatch.setitem(sys.modules, "warehouse", mod)
    return mod


# ---- minimal numpy stand-in for scikit-learn (CI does not install it) --------------------
class _FakeLinear:
    def __init__(self, alpha=0.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        xm = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        ym = y.mean() if self.fit_intercept else 0.0
        Xc, yc = X - xm, y - ym
        if self.alpha > 0:
            n, p = Xc.shape
            if n >= p:  # primal: (Xc'Xc + a I) w = Xc'yc
                self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(p), Xc.T @ yc)
            else:  # dual: w = Xc' (Xc Xc' + a I)^-1 yc
                self.coef_ = Xc.T @ np.linalg.solve(Xc @ Xc.T + self.alpha * np.eye(n), yc)
        else:
            self.coef_ = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        self.intercept_ = ym - xm @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class FakeRidge(_FakeLinear):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(alpha, fit_intercept)


class FakeLinearRegression(_FakeLinear):
    def __init__(self, fit_intercept=True):
        super().__init__(0.0, fit_intercept)


class FakeStandardScaler:
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        s = X.std(axis=0)
        s[s == 0] = 1.0
        self.scale_ = s
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_


def fake_sklearn_modules():
    def mse(a, b):
        return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

    def mae(a, b):
        return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    root = types.ModuleType("sklearn")
    lin = types.ModuleType("sklearn.linear_model")
    lin.Ridge, lin.LinearRegression = FakeRidge, FakeLinearRegression
    met = types.ModuleType("sklearn.metrics")
    met.mean_squared_error, met.mean_absolute_error = mse, mae
    pre = types.ModuleType("sklearn.preprocessing")
    pre.StandardScaler = FakeStandardScaler
    return {
        "sklearn": root,
        "sklearn.linear_model": lin,
        "sklearn.metrics": met,
        "sklearn.preprocessing": pre,
    }


def load_with_fake_sklearn(path, alias, monkeypatch, sklearn=True):
    """Execute `path` as module `alias` with fake (or absent) scikit-learn; no leakage."""
    import importlib.util

    if sklearn:
        for k, v in fake_sklearn_modules().items():
            monkeypatch.setitem(sys.modules, k, v)
    else:
        for k in ("sklearn", "sklearn.linear_model", "sklearn.metrics", "sklearn.preprocessing"):
            monkeypatch.setitem(sys.modules, k, None)  # import -> ImportError
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_ohlc(seed=1, n=600, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Low": close * (1 - rng.uniform(0, 0.01, n)),
            "Close": close,
            "Volume": rng.uniform(1e5, 1e6, n),
        },
        index=pd.bdate_range(start, periods=n),
    )
