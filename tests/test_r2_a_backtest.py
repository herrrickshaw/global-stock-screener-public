"""Group A: backtest.py (walk-forward price screens)."""

import numpy as np
import pytest

import backtest as bt


def _close(n=400, kind="up"):
    x = np.arange(n, dtype=float)
    if kind == "up":
        return 100 + x
    if kind == "down":
        return 500 - x * 0.5
    return np.full(n, 100.0)


def test_pit_requires_200_bars():
    assert bt._pit(_close(), 199) is None
    assert bt._pit(_close(), 200) is not None


def test_pit_nonpositive_last_price():
    c = _close()
    c[250] = 0.0
    assert bt._pit(c, 250) is None
    c[250] = -1.0
    assert bt._pit(c, 250) is None


def test_pit_uptrend_features():
    c = _close(400, "up")
    f = bt._pit(c, 300)
    assert f["rsi"] == 100.0  # no down moves -> dn == 0
    assert f["sma50"] > f["sma200"]
    assert f["sma50"] == pytest.approx(c[251:301].mean())
    assert f["sma200"] == pytest.approx(c[101:301].mean())
    assert f["sma50_prev"] == pytest.approx(c[-60 + 301 : 301 - 10].mean())
    assert f["pct_from_high"] == 0.0
    assert f["ret126"] == pytest.approx((c[300] / c[300 - 125] - 1) * 100)


def test_pit_downtrend_rsi_zero_and_pct_from_high():
    c = _close(400, "down")
    f = bt._pit(c, 300)
    assert f["rsi"] == pytest.approx(0.0)
    assert f["pct_from_high"] < 0
    hi = c[:301][-252:].max()
    assert f["pct_from_high"] == pytest.approx((c[300] / hi - 1) * 100)


def test_pit_flat_series_rsi_defaults_to_100():
    f = bt._pit(_close(400, "flat"), 300)
    assert f["rsi"] == 100.0
    assert f["ret126"] == 0.0


def test_pit_ret126_needs_more_than_126_bars_zero_base():
    c = _close(400, "up")
    c[300 - 125] = 0.0
    assert bt._pit(c, 300)["ret126"] == 0.0


def test_screens_boundaries():
    base = {
        "sma50": 10.0,
        "sma200": 9.0,
        "sma50_prev": 8.0,
        "sma200_prev": 9.0,
        "rsi": 30.0,
        "pct_from_high": -3.0,
        "ret126": 25.0,
    }
    S = bt.SCREENS
    assert S["golden_state"](base)
    assert not S["golden_state"]({**base, "sma50": 9.0})
    assert S["golden_crossover"](base)
    assert S["golden_crossover"]({**base, "sma50_prev": 9.0})  # <= boundary
    assert not S["golden_crossover"]({**base, "sma50_prev": 9.01})
    assert not S["golden_crossover"]({**base, "sma50": 8.0})
    assert S["near_high"](base)  # >= -3
    assert not S["near_high"]({**base, "pct_from_high": -3.01})
    assert not S["rsi_oversold"](base)  # < 30 strict
    assert S["rsi_oversold"]({**base, "rsi": 29.99})
    assert not S["momentum"](base)  # > 25 strict
    assert S["momentum"]({**base, "ret126": 25.01})
    assert not S["momentum"]({**base, "ret126": 40, "sma50": 9.0})


def test_edge_metrics_empty():
    assert bt.edge_metrics([], [1.0]) == {
        "trades": 0,
        "avg_ret": 0.0,
        "mkt_ret": 0.0,
        "edge": 0.0,
        "win_rate": 0.0,
    }


def test_edge_metrics_values():
    m = bt.edge_metrics([10.0, 2.0, 5.0], [4.0, 4.0, 5.0])
    assert m["trades"] == 3
    assert m["avg_ret"] == pytest.approx(5.67, abs=0.005)
    assert m["mkt_ret"] == pytest.approx(4.33, abs=0.005)
    assert m["edge"] == pytest.approx(1.33, abs=0.005)
    assert m["win_rate"] == pytest.approx(0.333)  # ties (5 vs 5) are not wins


def test_edge_metrics_nan_ignored():
    m = bt.edge_metrics([1.0, np.nan], [0.0, 0.0])
    assert m["avg_ret"] == 1.0
    assert m["trades"] == 2


class _Df:
    """Minimal frame with len() and ['Close'].to_numpy()."""

    def __init__(self, close):
        self._c = np.asarray(close, float)

    def __len__(self):
        return len(self._c)

    def __getitem__(self, k):
        import pandas as pd

        return pd.Series(self._c)


@pytest.fixture
def fake_market(monkeypatch):
    calls = []

    def load(market, tier=None):
        calls.append((market, tier))
        if tier == "ltm":
            return {}  # falls through to the un-tiered loader
        return {
            "A": _Df(_close(300, "up")),
            "B": _Df(_close(300, "down")),
            "SHORT": _Df(_close(50, "up")),
            "NONE": None,
        }

    monkeypatch.setattr(bt.datalink, "load_market", load)
    return calls


def test_backtest_screen_unknown():
    with pytest.raises(ValueError, match="unknown screen"):
        bt.backtest_screen("IN", "nope")


def test_backtest_screen_metrics(fake_market):
    m = bt.backtest_screen("IN", "golden_state", horizon=21, step=21)
    assert fake_market == [("IN", "ltm"), ("IN", None)]
    assert (m["market"], m["screen"], m["horizon"]) == ("IN", "golden_state", 21)
    # A (up) always golden; B (down) never -> picks are A's windows only
    assert m["trades"] > 0
    assert m["avg_ret"] > m["mkt_ret"]
    assert m["edge"] == pytest.approx(m["avg_ret"] - m["mkt_ret"], abs=0.011)
    # windows t in range(200, 300-21-1, 21): 200,221,242,263 -> 4 per eligible symbol
    assert m["trades"] == 4
    assert m["win_rate"] == 1.0


def test_backtest_screen_max_symbols(fake_market):
    m = bt.backtest_screen("IN", "golden_state", max_symbols=1)  # only A
    assert m["trades"] == 4
    assert m["avg_ret"] == pytest.approx(m["mkt_ret"])  # market == A only
    m2 = bt.backtest_screen("IN", "golden_state", max_symbols=2)
    assert m2["mkt_ret"] < m["mkt_ret"]  # adding B (down) lowers the market mean


def test_backtest_screen_no_data(monkeypatch):
    monkeypatch.setattr(bt.datalink, "load_market", lambda m, tier=None: {"X": _Df(_close(50))})
    m = bt.backtest_screen("IN", "near_high")
    assert m["trades"] == 0 and m["mkt_ret"] == 0.0 and m["edge"] == 0.0


def test_backtest_screen_zero_close_skipped(monkeypatch):
    c = _close(300, "up")
    c[200] = 0.0  # window t=200 has fwd=nan and must be dropped
    monkeypatch.setattr(bt.datalink, "load_market", lambda m, tier=None: {"X": _Df(c)})
    m = bt.backtest_screen("IN", "golden_state")
    assert m["trades"] == 3


def test_run_all_sorted_and_verbose(fake_market, capsys):
    df = bt.run_all("IN", 21)
    assert list(df["edge"]) == sorted(df["edge"], reverse=True)
    assert set(df["screen"]) == set(bt.SCREENS)
    out = capsys.readouterr().out
    assert "Walk-forward backtest — IN (21d hold)" in out
    assert "edge = screen avg forward return" in out


def test_run_all_quiet(fake_market, capsys):
    bt.run_all("IN", 21, verbose=False)
    assert capsys.readouterr().out == ""


def test_main_all_and_single(fake_market, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["backtest.py", "--market", "US"])
    assert bt.main() == 0
    assert "Walk-forward backtest — US" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["backtest.py", "--screen", "golden_state", "--horizon", "21"])
    assert bt.main() == 0
    out = capsys.readouterr().out
    assert out.startswith("golden_state on IN: 4 trades")
    assert "win 100%" in out


def test_main_all_flag(fake_market, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["backtest.py", "--all", "--screen", "near_high"])
    assert bt.main() == 0
    assert "Walk-forward backtest" in capsys.readouterr().out
