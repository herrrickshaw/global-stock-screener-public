import sys

import numpy as np
import pandas as pd
import pytest

import reward_universe_check as U
from tests.r2b_support import install_warehouse, make_prices


@pytest.fixture(scope="module")
def px():
    return make_prices(n_sym=260)


@pytest.fixture(scope="module")
def panel(px):
    orig = U._prices
    U._prices = lambda m: px[["Date", "Close", "Volume", "Symbol"]]
    try:
        return U._panel("XX")
    finally:
        U._prices = orig


def test_prices_us_reads_parquet(tmp_path, monkeypatch):
    d = tmp_path / "cache_seed" / "ltm"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "Date": ["2020-01-02", "2020-01-03"],
            "Close": [1.0, 2.0],
            "Volume": [10, 20],
            "Symbol": ["A", "A"],
            "Extra": [0, 0],
        }
    ).to_parquet(d / "US.parquet")
    monkeypatch.setattr(U, "BASE", str(tmp_path))
    out = U._prices("US")
    assert list(out.columns) == ["Date", "Close", "Volume", "Symbol"]
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])


def test_prices_other_uses_warehouse(monkeypatch, px):
    calls = []
    install_warehouse(monkeypatch, prices={"JP": px}, calls=calls)
    out = U._prices("JP")
    assert list(out.columns) == ["Date", "Close", "Volume", "Symbol"]
    assert calls == [("prices", "JP")]
    assert "warehouse" in sys.modules


def test_panel_columns_and_years(panel):
    assert list(panel.columns) == (
        ["symbol", "year"]
        + list(U.FACTORS)
        + ["fwd", "dvol", "close", "fwd_w", "excess", "excess_w"]
    )
    assert sorted(panel["year"].unique()) == list(range(2017, 2025))
    # short symbols (<300 rows) are skipped entirely
    assert not panel["symbol"].isin(["S000", "S001"]).any()
    # excess is neutral within each cohort year
    assert np.allclose(panel.groupby("year")["excess"].median(), 0.0)
    lo, hi = panel["fwd"].quantile([0.01, 0.99])
    assert panel["fwd_w"].between(lo, hi).all()


def test_panel_factor_formulas_single_symbol(monkeypatch):
    dates = pd.bdate_range("2015-01-01", "2025-12-31")
    n = len(dates)
    close = 10.0 * 1.0005 ** np.arange(n)
    vol = np.full(n, 100.0)
    frame = pd.DataFrame({"Date": dates, "Close": close, "Volume": vol, "Symbol": "ONE"})
    monkeypatch.setattr(U, "_prices", lambda m: frame)
    df = U._panel("X")
    r = df[df["year"] == 2017].iloc[0]
    i = int(np.searchsorted(dates.to_numpy(), np.datetime64("2017-06-30")))
    assert r["mom_12_1"] == pytest.approx(close[i - 21] / close[i - 273] - 1)
    assert r["above_200dma"] == pytest.approx(close[i] / close[i - 200 : i].mean() - 1)
    assert r["reversal_1m"] == pytest.approx(close[i] / close[i - 21] - 1)
    assert r["dist_52w_high"] == pytest.approx(0.0)  # monotone rising -> at the high
    assert r["fwd"] == pytest.approx(close[i + 252] / close[i] - 1)
    assert r["dvol"] == pytest.approx(float(np.median(close[i - 60 : i] * 100.0)))
    assert r["close"] == pytest.approx(close[i])
    assert r["vol_60"] == pytest.approx(0.0, abs=1e-9)


def test_panel_drops_bad_closes_and_nan_volume(monkeypatch):
    dates = pd.bdate_range("2015-01-01", "2025-12-31")
    n = len(dates)
    close = 10.0 * 1.0005 ** np.arange(n)
    close[5] = np.nan
    close[6] = -1.0
    close[7] = 0.0
    vol = np.full(n, np.nan)
    frame = pd.DataFrame({"Date": dates, "Close": close, "Volume": vol, "Symbol": "ONE"})
    monkeypatch.setattr(U, "_prices", lambda m: frame)
    df = U._panel("X")
    assert len(df) == 8
    assert (df["dvol"] == 0.0).all()  # NaN volume -> 0


def test_reward_threshold_is_60():
    sub = pd.DataFrame({"excess": [0.1] * 59, "excess_w": [0.1] * 59})
    assert U._reward(sub) is None
    sub = pd.DataFrame({"excess": [0.1] * 59 + [-0.1], "excess_w": [0.2] * 60})
    r = U._reward(sub)
    assert r["n"] == 60 and r["med"] == pytest.approx(0.1)
    assert r["mean"] == pytest.approx(0.2)
    assert r["win"] == pytest.approx(59 / 60)


def test_reward_win_is_strict():
    sub = pd.DataFrame({"excess": [0.0] * 60, "excess_w": [0.0] * 60})
    assert U._reward(sub)["win"] == 0.0


def test_best_factor_finds_low_vol(panel):
    b = U._best_factor(panel)
    assert b["factor"] == "vol_60" and b["dir"] == -1
    assert b["worse"] > 0 and b["test"]["med"] > 0
    assert b["keep"] in U.KEEP


def test_best_factor_needs_two_train_years_and_test(panel):
    assert U._best_factor(panel[panel["year"] <= 2017]) is None  # no test years
    only_train_year = panel[panel["year"].isin([2017, 2022])]
    assert U._best_factor(only_train_year) is None  # single train year
    assert U._best_factor(panel, split=2030) is None  # test empty


def test_best_factor_small_sample_none(panel):
    small = panel[panel["symbol"].isin(sorted(panel["symbol"].unique())[:15])]
    assert U._best_factor(small) is None  # <200 factor obs


@pytest.fixture
def run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(U, "BASE", str(tmp_path))

    def go(panels, argv=()):
        def fake(m):
            p = panels[m]
            if isinstance(p, BaseException):
                raise p
            return p

        monkeypatch.setattr(U, "_panel", fake)
        monkeypatch.setattr("sys.argv", ["x", *argv])
        U.main()
        return capsys.readouterr().out

    return go


def test_main_robust_whole(run, tmp_path, panel):
    out = run({"US": panel}, ["--markets", "us"])
    assert "none (entire universe)" in out
    assert "OOS-robust at full universe: 1/1 — US" in out
    assert "vol_60↓" in out and "✅" in out
    csv = pd.read_csv(tmp_path / "reports" / "reward_universe_whole.csv")
    assert csv.loc[0, "market"] == "US" and bool(csv.loc[0, "robust"])
    assert csv.loc[0, "obs"] == len(panel)
    assert "written → reports/reward_universe_whole.csv" in out


def test_main_liq_filter_and_edge_cases(run, tmp_path, panel):
    out = run(
        {"US": panel, "JP": SystemExit("x"), "HK": panel.iloc[0:0], "CN": panel.iloc[:30]},
        ["--markets", "US, JP,HK,CN", "--liq", "0.5"],
    )
    assert "drop least-liquid 50%" in out
    assert " JP   (no data)" in out
    assert out.count("— none robust —") == 2  # HK empty, CN tiny
    csv = pd.read_csv(tmp_path / "reports" / "reward_universe_liq50.csv")
    assert list(csv["market"]) == ["US", "HK", "CN"]
    liq_obs = csv.loc[0, "obs"]
    assert liq_obs == (panel["dvol"] >= panel["dvol"].quantile(0.5)).sum()
    assert csv.loc[1, "names"] == 0 and csv.loc[1, "obs"] == 0


def test_main_non_robust_marker(run, monkeypatch, panel):
    fake_best = {
        "factor": "mom_12_1",
        "dir": 1,
        "keep": 0.3,
        "worse": 0.01,
        "train": {"med": 0.02, "n": 5},
        "test": {"med": 0.005, "win": 0.9, "n": 10},
    }
    monkeypatch.setattr(U, "_best_factor", lambda df: fake_best)
    out = run({"US": panel}, ["--markets", "US"])
    assert "mom_12_1↑30%" in out and "OOS-robust at full universe: 0/1 — none" in out
    # med == 0.005 is NOT robust (strict >)
    assert "✅" not in out.split("OOS-robust")[0].split("best price factor")[1]
    fake_best["test"] = {"med": 0.0051, "win": 0.5, "n": 10}
    out = run({"US": panel}, ["--markets", "US"])
    assert "1/1 — US" in out
    fake_best["test"] = {"med": 0.0051, "win": 0.4999, "n": 10}
    out = run({"US": panel}, ["--markets", "US"])
    assert "0/1" in out
