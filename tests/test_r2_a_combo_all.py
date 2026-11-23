"""Group A: backtest_combo_all.py."""

import types

import numpy as np
import pandas as pd
import pytest

import backtest_combo_all as m
from tests.r2a_synth import fund_hist, lin_prices

TICKERS = [f"T{i}" for i in range(10)]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    fd = tmp_path / "fund"
    lt = tmp_path / "ltm"
    fd.mkdir()
    lt.mkdir()
    monkeypatch.setattr(m, "FUND_DIR", str(fd))
    monkeypatch.setattr(m, "LTM", str(lt))
    monkeypatch.setattr(m, "BASE", str(tmp_path))
    monkeypatch.setattr(m, "_USE_WH", False)
    return fd, lt


def _write(dirs, mkt="XX", tickers=TICKERS, n=1000, hist=None, lower=False):
    fd, lt = dirs
    (hist if hist is not None else fund_hist(tickers, years=7, start_year=2018)).to_parquet(
        fd / f"{mkt}.parquet"
    )
    px = lin_prices(tickers, start="2019-01-01", n=n)
    if lower:
        px["Symbol"] = px["Symbol"].str.lower()
    px.to_parquet(lt / f"{mkt}.parquet")


def test_fscore_missing_market(dirs):
    assert m._fscore_by_ticker("NOPE") == {}


def test_fscore_values_and_keys(dirs):
    h = fund_hist(["abc"], years=4, start_year=2018)
    _write(dirs, tickers=["abc"], hist=h)
    out = m._fscore_by_ticker("XX")
    assert list(out) == ["ABC"]  # upper-cased
    filed, F = out["ABC"]
    assert len(filed) == 3 and len(F) == 3  # first fiscal year has no prior -> dropped
    assert filed.dtype.kind == "M"
    assert np.all(np.diff(filed.astype("int64")) > 0)
    assert F.min() >= 0 and F.max() <= 9


def test_fscore_nine_point_and_worst(dirs):
    rows = []
    cols = [
        "ticker",
        "fy_end",
        "filed",
        "net_income",
        "revenue",
        "gross_profit",
        "cfo",
        "total_assets",
        "current_assets",
        "current_liabilities",
        "long_term_debt",
        "shares",
    ]
    y1 = [10, 100, 30, 8, 100, 40, 20, 30, 100]
    best = [20, 120, 60, 30, 100, 60, 20, 20, 102]  # shares +2% still counts
    worst = [-5, 90, 10, -1, 100, 20, 20, 40, 103]  # +3% does not
    for t, y2 in (("A", best), ("B", worst)):
        rows.append(dict(zip(cols, [t, "2018-12-31", "2019-02-01", *y1], strict=True)))
        rows.append(dict(zip(cols, [t, "2019-12-31", "2020-02-01", *y2], strict=True)))
    _write(dirs, tickers=["A", "B"], hist=pd.DataFrame(rows))
    out = m._fscore_by_ticker("XX")
    assert out["A"][1][0] == 9.0
    assert out["B"][1][0] == 1.0


def test_fscore_zero_prior_assets_nan_dropped(dirs):
    h = fund_hist(["A"], years=3)
    h.loc[0, "total_assets"] = 0
    _write(dirs, tickers=["A"], hist=h)
    filed, F = m._fscore_by_ticker("XX")["A"]
    assert len(F) == 1  # only the row whose prior assets are non-zero


def test_fscore_warehouse_path(monkeypatch):
    h = fund_hist(["A", "B"], years=3)
    fake = types.SimpleNamespace(markets=lambda: ["XX"], fundamentals=lambda mk: h.copy(), DB="x")
    monkeypatch.setattr(m, "_USE_WH", True)
    monkeypatch.setattr(m, "_wh", fake, raising=False)
    assert m._fscore_by_ticker("ZZ") == {}
    assert set(m._fscore_by_ticker("XX")) == {"A", "B"}


def test_f_asof_boundaries():
    filed = np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[ns]")
    fd = (filed, np.array([5.0, 8.0]))
    assert np.isnan(m._f_asof(fd, np.datetime64("2019-12-31")))
    assert m._f_asof(fd, np.datetime64("2020-01-01")) == 5.0  # same day counts
    assert m._f_asof(fd, np.datetime64("2020-12-31")) == 5.0
    assert m._f_asof(fd, np.datetime64("2021-01-01")) == 8.0


def test_darvas_logic():
    n = 100
    high = np.full(n, 90.0)
    high[10] = 100.0
    close = np.full(n, 80.0)
    vol = np.full(n, 100.0)
    t = 70
    close[t], vol[t] = 101.0, 121.0
    assert m._darvas(high, vol, close, t) is True
    vol[t] = 120.0
    assert m._darvas(high, vol, close, t) is False
    vol[t] = 121.0
    close[t - 1] = 101.0
    assert m._darvas(high, vol, close, t) is False
    close[t - 1] = 80.0
    assert m._darvas(high, vol, close, 60) is False


def test_backtest_market_no_data(dirs):
    assert m.backtest_market("NOPE", 20) is None
    fd, lt = dirs
    fund_hist(["A"], years=3).to_parquet(fd / "XX.parquet")
    assert m.backtest_market("XX", 20) is None  # fundamentals but no prices


def test_backtest_market_too_few_windows(dirs):
    _write(dirs, tickers=TICKERS[:3], n=400)
    assert m.backtest_market("XX", 20) is None


def test_backtest_market_result(dirs):
    _write(dirs, lower=True)
    r = m.backtest_market("XX", 20)
    w = len(TICKERS) * len(range(200, 1000 - 20 - 1, 21))
    assert r["market"] == "XX" and r["windows"] == w
    assert r["base"] > 0  # uptrending panel
    # monotonic uptrend: near_high always fires, so its edge is exactly 0
    assert r["near_high_n"] == w
    assert r["near_high"] == pytest.approx(0.0)
    assert r["rsi_oversold_n"] == 0 and r["rsi_oversold"] is None
    assert r["piotroski7_n"] > 0 and r["piotroski7"] is not None
    # near_high is always on, so the combo is exactly the overlay population
    assert r["near_high×F7_n"] == r["piotroski7_n"]
    assert r["darvas_n"] == 0 and r["darvas"] is None
    assert r["momentum_n"] > 0
    assert r["momentum×F7_n"] <= min(r["momentum_n"], r["piotroski7_n"])
    for lab in ["darvas", "near_high", "rsi_oversold", "momentum"]:
        assert f"{lab}×F7" in r


def test_backtest_market_no_fundamentals_for_symbol(dirs):
    # prices for 10 names but history for only 2 -> overlay population shrinks
    h = fund_hist(TICKERS[:2], years=7)
    _write(dirs, hist=h)
    full = m.backtest_market("XX", 20)
    _write(dirs)
    both = m.backtest_market("XX", 20)
    assert full["piotroski7_n"] < both["piotroski7_n"]


def test_backtest_market_warehouse_prices(dirs, monkeypatch):
    px = lin_prices(TICKERS, start="2019-01-01", n=1000)
    fake = types.SimpleNamespace(
        markets=lambda: ["XX"],
        fundamentals=lambda mk: fund_hist(TICKERS, years=7),
        prices=lambda mk: px.copy(),
        DB="x",
    )
    monkeypatch.setattr(m, "_USE_WH", True)
    monkeypatch.setattr(m, "_wh", fake, raising=False)
    r = m.backtest_market("XX", 20)
    assert r["windows"] == len(TICKERS) * len(range(200, 979, 21))


def _canned(**over):
    r = {"market": "XX", "windows": 300, "base": 1.0}
    for lab in ["darvas", "near_high", "rsi_oversold", "momentum"]:
        r[lab] = None
        r[f"{lab}×F7"] = None
    r["piotroski7"] = 0.0
    r.update(over)
    return r


def test_main_output_and_parquet(dirs, monkeypatch, capsys, tmp_path):
    results = {
        "AA": _canned(
            market="AA", near_high=2.0, **{"near_high×F7": 3.5, "darvas": 1.0, "darvas×F7": 4.0}
        ),
        "BB": _canned(
            market="BB", momentum=5.0, **{"momentum×F7": 4.0, "darvas": 2.0, "darvas×F7": 6.0}
        ),
        "CC": _canned(market="CC", near_high=1.0),  # combo n<20
        "DD": None,
        "EE": _canned(market="EE"),  # no price edge at all -> silently skipped
    }
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: results[mk])
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "63", "--markets", "aa,bb,cc,dd,ee"])
    m.main()
    out = capsys.readouterr().out
    assert "Cross-market price×fundamental combo — 63d forward" in out
    assert "  AA       near_high   +2.0  →       +3.5   +1.5" in out
    assert "  BB        momentum   +5.0  →       +4.0   -1.0" in out
    assert "  CC       near_high   +1.0  →       n<20" in out
    assert "  DD  (insufficient data)" in out
    assert "EE" not in out.split("overlay improved")[0].split("DD")[1]
    assert "F≥7 overlay improved the best price screen in 1/2 markets." in out
    assert "darvas×F7 avg edge: +5.0pp (vs darvas +1.5)" in out
    pq = tmp_path / "data" / "curated" / "combo_backtest_all.parquet"
    df = pd.read_parquet(pq)
    assert list(df["market"]) == ["AA", "BB", "CC", "EE"]
    assert "→ data/curated/combo_backtest_all.parquet" in out


def test_main_no_rows_no_output_file(dirs, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: None)
    monkeypatch.setattr("sys.argv", ["x", "--markets", "AA"])
    m.main()
    assert "overlay improved" not in capsys.readouterr().out
    assert not (tmp_path / "data").exists()


def test_main_default_markets_from_fund_dir(dirs, monkeypatch, capsys):
    fd, _ = dirs
    for n in ["US", "IN"]:
        (fd / f"{n}.parquet").write_text("x")
    (fd / "README.md").write_text("x")
    seen = []
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: seen.append((mk, h)))
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert seen == [("IN", 126), ("US", 126)]


def test_main_default_markets_from_warehouse(dirs, monkeypatch):
    monkeypatch.setattr(m, "_USE_WH", True)
    monkeypatch.setattr(
        m, "_wh", types.SimpleNamespace(markets=lambda: ["JP", "KR"]), raising=False
    )
    seen = []
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: seen.append(mk))
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert seen == ["JP", "KR"]


def test_backtest_market_skips_bad_bars(dirs):
    _write(dirs)
    base = m.backtest_market("XX", 20)
    fd, lt = dirs
    px = pd.read_parquet(lt / "XX.parquet")
    sel = px["Symbol"] == "T0"
    close = px.loc[sel, "Close"].to_numpy().copy()
    close[200] = 0.0  # window t=200: zero close
    close[242 + 20] = np.nan  # window t=242: non-finite forward return
    close[305] = np.nan  # window t=305: non-finite close
    close[326] = -3.0  # window t=326: finite fwd but no price features
    px.loc[sel, "Close"] = close
    px.to_parquet(lt / "XX.parquet")
    r = m.backtest_market("XX", 20)
    # t=200,242,305 dropped (and 284 loses its fwd=close[304]? no: 284+20=304 ok)
    dropped = base["windows"] - r["windows"]
    assert dropped == 3
    assert r["near_high_n"] <= r["windows"] - 1  # the negative-close window is never a pick


def test_backtest_market_skips_short_symbols(dirs):
    _write(dirs)
    base = m.backtest_market("XX", 20)
    fd, lt = dirs
    px = pd.read_parquet(lt / "XX.parquet")
    extra = lin_prices(["T0X"], start="2019-01-01", n=220)  # == 200 + 20 : too short
    pd.concat([px, extra]).to_parquet(lt / "XX.parquet")
    assert m.backtest_market("XX", 20)["windows"] == base["windows"]
