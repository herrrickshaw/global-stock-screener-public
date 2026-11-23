import numpy as np
import pandas as pd
import pytest

import reward_liquidity_check as L
import reward_universe_check as U
from tests.r2b_support import make_prices


@pytest.fixture(scope="module")
def px():
    return make_prices(n_sym=300)


@pytest.fixture(scope="module")
def panel(px):
    orig = U._prices
    U._prices = lambda m: px[["Date", "Close", "Volume", "Symbol"]]
    try:
        return L._panel("XX")
    finally:
        U._prices = orig


def test_panel_shape(panel):
    assert list(panel.columns) == ["symbol", "year"] + L.FACTORS + [
        "fwd",
        "dvol",
        "fwd_w",
        "excess",
    ]
    assert sorted(panel["year"].unique()) == list(range(2017, 2025))
    assert not panel["symbol"].isin(["S000", "S001"]).any()
    assert np.allclose(panel.groupby("year")["excess"].median(), 0.0)
    assert not np.isinf(panel[L.FACTORS].to_numpy(dtype=float)).any()


def test_panel_formulas_single_symbol(monkeypatch):
    dates = pd.bdate_range("2015-01-01", "2025-12-31")
    n = len(dates)
    close = 10.0 * 1.0003 ** np.arange(n)
    v = 1000.0 * (1.001 ** np.arange(n))  # rising volume
    frame = pd.DataFrame({"Date": dates, "Close": close, "Volume": v, "Symbol": "ONE"})
    monkeypatch.setattr(U, "_prices", lambda m: frame)
    df = L._panel("X")
    r = df[df["year"] == 2018].iloc[0]
    i = int(np.searchsorted(dates.to_numpy(), np.datetime64("2018-06-30")))
    dvol = close * v
    assert r["liq_3_12"] == pytest.approx(
        dvol[i - 63 : i].mean() / dvol[i - 252 : i - 63].mean() - 1
    )
    assert r["liq_6_6"] == pytest.approx(
        dvol[i - 126 : i].mean() / dvol[i - 252 : i - 126].mean() - 1
    )
    assert r["vol_surge"] == pytest.approx(v[i - 21 : i].mean() / v[i - 252 : i].mean() - 1)
    assert r["fwd"] == pytest.approx(close[i + 252] / close[i] - 1)
    assert r["dvol"] == pytest.approx(np.median(dvol[i - 60 : i]))
    illiq = np.abs(np.diff(close)) / close[:-1] / dvol[1:]
    rec, pri = illiq[i - 63 : i - 1].mean(), illiq[i - 252 : i - 63].mean()
    assert r["amihud_trend"] == pytest.approx((pri - rec) / pri)
    assert r["liq_3_12"] > 0 and r["liq_6_6"] > 0


def test_panel_zero_volume_gives_nan_factors(monkeypatch):
    dates = pd.bdate_range("2015-01-01", "2025-12-31")
    n = len(dates)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Close": 10.0 * 1.0003 ** np.arange(n),
            "Volume": np.zeros(n),
            "Symbol": "Z",
        }
    )
    monkeypatch.setattr(U, "_prices", lambda m: frame)
    df = L._panel("X")
    assert len(df) == 8
    assert df[["liq_3_12", "liq_6_6", "vol_surge"]].isna().all().all()
    assert df["amihud_trend"].isna().all()


def test_panel_empty_when_no_history(monkeypatch):
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2020-01-01", periods=50),
            "Close": 1.0,
            "Volume": 1.0,
            "Symbol": "A",
        }
    )
    monkeypatch.setattr(U, "_prices", lambda m: frame)
    assert L._panel("X").empty


def test_reward_threshold():
    mk = lambda n, v=0.1: pd.DataFrame({"excess": [v] * n})  # noqa: E731
    assert L._reward(mk(59)) is None
    r = L._reward(mk(60))
    assert r == {"n": 60, "med": pytest.approx(0.1), "win": 1.0}
    assert L._reward(mk(60, 0.0))["win"] == 0.0


def test_best_selects_in_sample_regardless_of_sign(panel):
    g = panel[panel["dvol"] >= panel["dvol"].quantile(L.LIQ_GATE)]
    b = L._best(g)
    assert b is not None
    assert b["factor"] in L.FACTORS and b["dir"] in (1, -1) and b["keep"] in L.KEEP
    assert b["test"]["n"] >= 60 and b["train"]["n"] >= 2 * L.N_FOLD


def test_best_guards(panel):
    assert L._best(panel[panel["year"] <= 2017]) is None
    assert L._best(panel[panel["year"].isin([2017, 2022])]) is None
    assert L._best(panel, split=2040) is None
    assert L._best(panel.iloc[:100]) is None


@pytest.fixture
def run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(L, "BASE", str(tmp_path))

    def go(panels, best=None):
        def fake(m):
            p = panels[m]
            if isinstance(p, BaseException):
                raise p
            return p

        monkeypatch.setattr(L, "_panel", fake)
        if best is not None:
            monkeypatch.setattr(L, "_best", lambda df: best)
        return go

    return go


def _best(med=0.02, win=0.6, worse=0.01, d=1):
    return {
        "factor": "liq_3_12",
        "dir": d,
        "keep": 0.3,
        "worse": worse,
        "train": {"n": 100, "med": 0.03, "win": 0.6},
        "test": {"n": 70, "med": med, "win": win},
    }


def test_sweep_branches(run, tmp_path, panel, capsys):
    run({"A": panel, "B": SystemExit("x"), "C": panel.iloc[0:0], "D": panel.iloc[:50]})
    res = L.sweep(["A", "B", "C", "D"])
    out = capsys.readouterr().out
    assert "B    (no data)" in out and "C    (empty)" in out
    assert res["D"] == {"market": "D", "factor": None, "robust": False}
    assert "— none robust —" in out
    assert set(res) == {"A", "D"}
    csv = pd.read_csv(tmp_path / "reports" / "reward_liquidity.csv")
    assert "D" in set(csv["market"]) and "B" not in set(csv["market"])


@pytest.mark.parametrize(
    "med,win,worse,ok",
    [
        (0.0051, 0.50, 0.01, True),
        (0.005, 0.60, 0.01, False),
        (0.0051, 0.4999, 0.01, False),
        (0.02, 0.6, 0.0, False),
        (0.02, 0.6, -0.01, False),
    ],
)
def test_sweep_robust_boundaries(run, panel, capsys, med, win, worse, ok):
    run({"A": panel}, best=_best(med, win, worse))
    res = L.sweep(["A"])
    out = capsys.readouterr().out
    assert res["A"]["robust"] is ok
    assert ("1/1: A" in out) is ok
    assert ("✅" in out) is ok
    assert "liq_3_12 ↑ rising 30%" in out
    assert res["A"]["direction"] == "rising" and res["A"]["oos_n"] == 70


def test_sweep_falling_label(run, panel, capsys):
    run({"A": panel}, best=_best(d=-1))
    res = L.sweep(["A"])
    assert "↓ falling" in capsys.readouterr().out
    assert res["A"]["direction"] == "falling"


def test_sweep_applies_liquidity_gate(run, monkeypatch, panel):
    seen = []
    run({"A": panel})
    monkeypatch.setattr(L, "_best", lambda df: seen.append(len(df)) or None)
    L.sweep(["A"])
    assert seen == [(panel["dvol"] >= panel["dvol"].quantile(0.5)).sum()]


def test_picks_rising_and_falling(run, panel, capsys):
    run({"A": panel, "B": panel})
    res = {
        "A": {"robust": True, "factor": "liq_3_12", "direction": "rising", "keep": 0.3},
        "B": {"robust": True, "factor": "liq_3_12", "direction": "falling", "keep": 0.3},
        "C": {"robust": False},
    }
    L.picks(["A", "B", "C", "D"], res, 5)
    out = capsys.readouterr().out
    assert "C: no robust liquidity-trend edge" in out
    assert "D: no robust liquidity-trend edge" in out
    latest = panel[panel["year"] == panel["year"].max()]
    latest = latest[latest["dvol"] >= latest["dvol"].quantile(0.5)]
    hi = latest[latest["liq_3_12"] >= latest["liq_3_12"].quantile(0.7)].sort_values(
        "dvol", ascending=False
    )
    lo = latest[latest["liq_3_12"] <= latest["liq_3_12"].quantile(0.3)].sort_values(
        "dvol", ascending=False
    )
    a, b = out.split("B — liq_3_12 falling")
    assert "A — liq_3_12 rising (top 30%, liquidity-gated)" in a
    for s in hi["symbol"].head(5):
        assert s in a
    assert hi["symbol"].iloc[5] not in a
    for s in lo["symbol"].head(5):
        assert s in b


def test_main_sweep_only(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(L, "sweep", lambda m: calls.append(("sweep", m)) or {"A": 1})
    monkeypatch.setattr(L, "picks", lambda *a: calls.append(("picks", a)))
    monkeypatch.setattr("sys.argv", ["x", "--markets", "a, b"])
    L.main()
    assert calls == [("sweep", ["A", "B"])]


def test_main_with_picks(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "sweep", lambda m: {"r": 1})
    monkeypatch.setattr(L, "picks", lambda *a: calls.append(a))
    monkeypatch.setattr("sys.argv", ["x", "--markets", "US", "--picks", "us,in", "--top", "3"])
    L.main()
    assert calls == [(["US", "IN"], {"r": 1}, 3)]


def test_main_defaults(monkeypatch):
    seen = []
    monkeypatch.setattr(L, "sweep", lambda m: seen.append(m) or {})
    monkeypatch.setattr("sys.argv", ["x"])
    L.main()
    assert seen == [U.ALL]
