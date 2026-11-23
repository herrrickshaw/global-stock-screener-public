"""Group A: backtest_price_sweep.py."""

import numpy as np
import pandas as pd
import pytest

import backtest
import backtest_price_sweep as m


def _df(close):
    return pd.DataFrame({"Close": np.asarray(close, float)})


UP = 100 + np.arange(600, dtype=float)
DOWN = 1000 - np.arange(600, dtype=float) * 0.9


def _windows(n, h=21, step=21):
    return len(range(200, n - h - 1, step))


@pytest.fixture
def loader(monkeypatch):
    state = {"data": {}, "calls": []}

    def load(market, tier=None):
        state["calls"].append((market, tier))
        return state["data"] if tier is None or state["ltm_ok"] else {}

    state["ltm_ok"] = True
    monkeypatch.setattr(m.datalink, "load_market", load)
    return state


def test_all_markets_lists_parquets(tmp_path, monkeypatch):
    d = tmp_path / "cache_seed" / "ltm"
    d.mkdir(parents=True)
    for n in ["US.parquet", "IN.parquet", "notes.txt", "EU.parquet"]:
        (d / n).write_text("x")
    monkeypatch.setattr(m, "BASE", str(tmp_path))
    assert m._all_markets() == ["EU", "IN", "US"]


def test_sweep_market_all_up(loader):
    loader["data"] = {k: _df(UP * (1 + i)) for i, k in enumerate("ABCD")}
    res = m.sweep_market("XX", 21)
    n = 4 * _windows(600)
    assert res["_baseline"]["n"] == n
    for s in ("golden_state", "near_high"):
        assert res[s]["trades"] == n
        assert res[s]["edge"] == 0.0 and res[s]["edge_med"] == 0.0
        assert res[s]["win"] == 1.0
    assert res["rsi_oversold"] == {"edge": 0.0, "edge_med": 0.0, "win": 0.0, "trades": 0}
    assert res["golden_crossover"]["trades"] == 0
    assert res["_baseline"]["median"] > 0
    # tier ltm tried first
    assert loader["calls"] == [("XX", "ltm")]


def test_sweep_market_falls_back_to_default_tier(loader):
    loader["data"] = {k: _df(UP) for k in "ABCD"}
    loader["ltm_ok"] = False
    assert m.sweep_market("XX", 21) is not None
    assert loader["calls"] == [("XX", "ltm"), ("XX", None)]


def test_sweep_market_edge_positive_for_trend_screen(loader):
    loader["data"] = {"U1": _df(UP), "U2": _df(UP), "D1": _df(DOWN), "D2": _df(DOWN)}
    res = m.sweep_market("XX", 21)
    gs = res["golden_state"]
    assert gs["trades"] == 2 * _windows(600)
    assert gs["edge"] > 0 and gs["edge_med"] > 0
    assert gs["win"] == 1.0
    assert res["_baseline"]["n"] == 4 * _windows(600)
    # edge is rounded to 2 dp
    assert gs["edge"] == round(gs["edge"], 2)


def test_sweep_market_insufficient(loader):
    assert m.sweep_market("XX", 21) is None  # empty data
    loader["data"] = {"A": _df(UP[:230])}  # too short for 200+h+1? 230>=222 but <50 windows
    assert m.sweep_market("XX", 21) is None
    loader["data"] = {"S": _df(UP[:100]), "N": None}
    assert m.sweep_market("XX", 21) is None


def test_sweep_min_length_boundary(loader):
    # len == 200 + h + 1 is accepted by the filter but yields zero windows
    loader["data"] = {"A": _df(UP[:222])}
    assert m.sweep_market("XX", 21) is None


def test_sweep_skips_zero_close_and_nonfinite(loader):
    c = UP.copy()
    c[221] = 0.0  # window t=221 skipped
    c[305] = np.nan  # window t=284 has a non-finite forward return
    loader["data"] = {"A": _df(c), "B": _df(UP), "C": _df(UP), "D": _df(UP)}
    res = m.sweep_market("XX", 21)
    # dropped windows: t=221 (zero close), t=284 (fwd uses the NaN at 305),
    # t=305 (the NaN close itself)
    assert res["_baseline"]["n"] == 4 * _windows(600) - 3


def test_sweep_negative_close_has_no_features(loader):
    c = UP.copy()
    c[242] = -5.0  # finite fwd but _pit returns None -> window counted, never a pick
    loader["data"] = {"A": _df(c), "B": _df(UP), "C": _df(UP), "D": _df(UP)}
    res = m.sweep_market("XX", 21)
    assert res["_baseline"]["n"] == 4 * _windows(600)
    assert res["near_high"]["trades"] == 4 * _windows(600) - 1


def test_sweep_predicate_error_is_isolated(loader, monkeypatch):
    monkeypatch.setitem(backtest.SCREENS, "boom", lambda f: 1 / 0)
    loader["data"] = {k: _df(UP) for k in "ABCD"}
    res = m.sweep_market("XX", 21)
    assert res["boom"]["trades"] == 0
    assert res["near_high"]["trades"] == 4 * _windows(600)


def test_sweep_winsorizes_baseline_mean(loader):
    spike = UP.copy()
    spike[300:] *= 50  # absurd one-off jump
    loader["data"] = {"A": _df(spike), "B": _df(UP), "C": _df(UP), "D": _df(UP)}
    res = m.sweep_market("XX", 21)
    b = res["_baseline"]
    assert b["median"] < 10  # median ignores the spike
    assert b["mean"] < 500  # clipped at the 99th pct, not raw (~4900%)
    # raw mean would exceed the winsorized one
    raw = []
    for s in (spike, UP, UP, UP):
        for t in range(200, 600 - 21 - 1, 21):
            raw.append((s[t + 21] / s[t] - 1) * 100)
    assert np.mean(raw) > b["mean"]


def test_main_requires_datalink(monkeypatch):
    monkeypatch.setattr(m, "datalink", None)
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit, match="datalink module unavailable"):
        m.main()


def _canned(edge_med_by_screen):
    out = {"_baseline": {"mean": 1.0, "median": 0.5, "n": 1234}}
    for s in backtest.SCREENS:
        e = edge_med_by_screen.get(s, 0.0)
        out[s] = {"edge": e / 2, "edge_med": e, "win": 0.5, "trades": 7}
    return out


@pytest.fixture
def main_env(tmp_path, monkeypatch):
    d = tmp_path / "cache_seed" / "ltm"
    d.mkdir(parents=True)
    for n in ["US", "IN", "EU", "UK", "KR"]:
        (d / f"{n}.parquet").write_text("x")
    monkeypatch.setattr(m, "BASE", str(tmp_path))
    monkeypatch.setattr(m, "OUT", str(tmp_path / "results" / "price_sweep.csv"))
    seen = []

    def fake(mkt, h):
        seen.append((mkt, h))
        if mkt == "UK":
            return None
        return _canned({"near_high": 2.5 if mkt == "EU" else -1.0})

    monkeypatch.setattr(m, "sweep_market", fake)
    return seen, tmp_path


def test_main_default_skips_done_markets(main_env, monkeypatch, capsys):
    seen, tmp = main_env
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert seen == [("EU", 21), ("UK", 21)]  # US/IN/KR are in DONE
    out = capsys.readouterr().out
    assert "Robust price-screen sweep — 21d hold — 2 markets" in out
    assert "UK   (insufficient data)" in out
    assert "EU   best(median): near_high" in out
    assert "+2.50pp" in out and "n=1,234" in out
    assert "=== MEDIAN edge (pp, outlier-proof) — market × screen ===" in out
    assert "=== winsorized-MEAN edge (pp) — market × screen ===" in out
    assert "near_high" in out and "positive in 1/1" in out
    df = pd.read_csv(tmp / "results" / "price_sweep.csv")
    assert len(df) == len(backtest.SCREENS)
    assert set(df["market"]) == {"EU"}
    assert "full results → results" in out


def test_main_all_flag(main_env, monkeypatch, capsys):
    seen, _ = main_env
    monkeypatch.setattr("sys.argv", ["x", "--all", "--horizon", "10"])
    m.main()
    assert [s[0] for s in seen] == ["EU", "IN", "KR", "UK", "US"]
    assert {s[1] for s in seen} == {10}
    assert "5 markets" in capsys.readouterr().out


def test_main_markets_option_overrides_all(main_env, monkeypatch, capsys):
    seen, _ = main_env
    monkeypatch.setattr("sys.argv", ["x", "--markets", "eu, us", "--all"])
    m.main()
    assert [s[0] for s in seen] == ["EU", "US"]


def test_main_no_results(main_env, monkeypatch, capsys):
    monkeypatch.setattr(m, "sweep_market", lambda *a: None)
    monkeypatch.setattr("sys.argv", ["x", "--markets", "UK"])
    m.main()
    out = capsys.readouterr().out
    assert "no results" in out
    assert not (main_env[1] / "results").exists()


def test_main_screen_ordering_in_matrix(main_env, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--markets", "EU"])
    m.main()
    out = capsys.readouterr().out
    hdr = next(ln for ln in out.splitlines() if "rsi_oversold" in ln and "near_high" in ln)
    cols = hdr.split()
    assert cols[1:] == m.SCREEN_ORDER
