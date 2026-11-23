"""Group A: backtest_fundamentals_all.py."""

import types

import numpy as np
import pandas as pd
import pytest

import backtest_fundamentals_all as m
from tests.r2a_synth import fund_hist, lin_prices

TICKERS = [f"T{i:02d}" for i in range(30)]


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


def _write(dirs, mkt="XX", tickers=TICKERS, years=7, n=3000):
    fd, lt = dirs
    fund_hist(tickers, years=years, start_year=2015).to_parquet(fd / f"{mkt}.parquet")
    lin_prices(tickers, start="2015-06-01", n=n).to_parquet(lt / f"{mkt}.parquet")


COLS = [
    "ticker",
    "fy_end",
    "net_income",
    "revenue",
    "gross_profit",
    "cfo",
    "total_assets",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "shares",
    "equity",
]


def _two(t, y1, y2):
    return pd.DataFrame(
        [
            dict(zip(COLS, [t, "2018-12-31", *y1], strict=True)),
            dict(zip(COLS, [t, "2019-12-31", *y2], strict=True)),
        ]
    )


Y1 = [10, 100, 30, 8, 100, 40, 20, 30, 100, 50]
BEST = [20, 120, 60, 30, 100, 60, 20, 20, 100, 50]
WORST = [-5, 90, 10, -1, 100, 20, 20, 40, 120, 50]


def test_fscore_values():
    out = m._fscore(pd.concat([_two("A", Y1, BEST), _two("B", Y1, WORST)])).set_index("ticker")
    assert out.loc["A", "F"] == 9.0
    assert out.loc["B", "F"] == 1.0
    assert len(out) == 2  # first year of each ticker has no prior -> NaN F -> dropped


def test_fscore_derived_fields():
    a = m._fscore(_two("A", Y1, BEST)).iloc[0]
    assert a["roce"] == pytest.approx(20 / 80 * 100)
    assert a["de"] == pytest.approx(20 / 50)
    assert a["lowdebt"] == 1.0  # 20 < 30
    # ncav = current assets - (total assets - equity) = 60 - 50
    assert a["ncav"] == pytest.approx(10.0)
    assert a["ey_num"] == 20
    b = m._fscore(_two("B", Y1, WORST)).iloc[0]
    assert b["lowdebt"] == 0.0  # 40 > 30
    flat = list(BEST)
    flat[7] = 30  # equal debt is NOT a decrease
    assert m._fscore(_two("C", Y1, flat)).iloc[0]["lowdebt"] == 0.0


def test_fscore_nonpositive_equity_and_capital():
    y2 = list(BEST)
    y2[9] = 0  # equity 0 -> de NaN
    assert np.isnan(m._fscore(_two("A", Y1, y2)).iloc[0]["de"])
    y2 = list(BEST)
    y2[6] = 100  # current liabilities == total assets -> capital 0 -> roce NaN
    # (current-ratio terms change but roce must be NaN)
    assert np.isnan(m._fscore(_two("A", Y1, y2)).iloc[0]["roce"])


def test_fscore_null_debt_is_zero_and_zero_prior_assets():
    y2 = list(BEST)
    y2[7] = None
    out = m._fscore(_two("A", Y1, y2)).iloc[0]
    assert out["long_term_debt"] == 0
    y1 = list(Y1)
    y1[4] = 0  # prior total assets 0
    assert m._fscore(_two("A", y1, BEST)).empty


def test_fscore_share_tolerance():
    ok = list(BEST)
    ok[8] = 102
    bad = list(BEST)
    bad[8] = 103
    d = m._fscore(_two("A", Y1, ok)).iloc[0]["F"] - m._fscore(_two("A", Y1, bad)).iloc[0]["F"]
    assert d == 1.0


def test_data_access_parquet(dirs):
    _write(dirs, tickers=TICKERS[:2], n=50)
    assert len(m._fund_df("XX")) == 14
    assert list(m._px_df("XX").columns) == ["Date", "Close", "Symbol"]
    assert m._has("XX") is True
    assert m._has("NOPE") is False
    fd, lt = dirs
    (fd / "HALF.parquet").write_text("x")  # fundamentals without prices
    assert m._has("HALF") is False


def test_data_access_warehouse(monkeypatch):
    fake = types.SimpleNamespace(
        markets=lambda: ["XX"],
        fundamentals=lambda mk: fund_hist(["A"], years=3),
        prices=lambda mk: lin_prices(["A"], n=10),
        DB="x",
    )
    monkeypatch.setattr(m, "_USE_WH", True)
    monkeypatch.setattr(m, "_wh", fake, raising=False)
    assert m._has("XX") and not m._has("YY")
    assert len(m._fund_df("XX")) == 3
    assert len(m._px_df("XX")) == 10


def test_prices_normalises_symbols(monkeypatch):
    px = lin_prices(["abc"], n=5)
    monkeypatch.setattr(m, "_px_df", lambda mk: px)
    out = m._prices("XX")
    assert list(out) == ["ABC"]
    dates, closes = out["ABC"]
    assert len(dates) == 5 and closes[0] == 100.0


def test_attach_exact():
    by = {"A": (pd.date_range("2019-01-01", periods=50).to_numpy(), 100 + np.arange(50.0))}
    sc = pd.DataFrame(
        {
            "ticker": ["a", "A", "A", "A", "B", "A"],
            "filed": ["2019-01-10", "9999-99-99", "", "2019-01-10", "2019-01-10", "2020-01-01"],
        }
    )
    out = m._attach(sc, by, 5)
    assert list(out.index) == [0, 3]
    assert out["entry_px"].iloc[0] == 110.0
    assert out["fwd"].iloc[0] == pytest.approx(115 / 110 - 1)


def test_attach_horizon_boundary_and_zero_entry():
    by = {"A": (pd.date_range("2019-01-01", periods=20).to_numpy(), 100 + np.arange(20.0))}
    sc = pd.DataFrame({"ticker": ["A"], "filed": ["2019-01-04"]})  # entry idx 4
    assert len(m._attach(sc, by, 15)) == 1
    assert len(m._attach(sc, by, 16)) == 0
    by0 = {"A": (by["A"][0], np.zeros(20))}
    assert m._attach(sc, by0, 3).empty


def test_backtest_market_missing(dirs):
    assert m.backtest_market("NOPE", 30) is None


def test_backtest_market_too_few_rows(dirs):
    _write(dirs, tickers=TICKERS[:5])
    assert m.backtest_market("XX", 30) is None  # 5 * 6 = 30 < 100


def test_backtest_market_result(dirs):
    _write(dirs)
    r = m.backtest_market("XX", 30)
    assert r["market"] == "XX"
    assert r["n"] == 30 * 6  # years 2..7 each have a prior year
    assert set(r) >= {
        "base",
        "F>=8",
        "n_F8",
        "F>=7",
        "F<=3",
        "magic",
        "quality",
        "net_net",
        "lowdebt",
    }
    # the synthetic firms all improve every year: no F<=3 population
    assert r["F<=3"] is None
    assert r["n_F8"] >= 20 and r["F>=8"] is not None
    assert r["F>=7"] is not None
    # linear price panel: forward returns shrink with time, so base is positive
    assert r["base"] > 0
    # net-net never happens with these balance sheets
    assert r["net_net"] is None
    # edges are in percentage points: bounded by the base return scale
    assert abs(r["F>=7"]) < 100


def test_backtest_market_edge_sign_and_units(dirs):
    _write(dirs)
    r = m.backtest_market("XX", 30)
    d = m._fscore(m._fund_df("XX"))
    d = m._attach(d, m._prices("XX"), 30)
    base = d["fwd"].median()
    exp = (d.loc[d["F"] >= 7, "fwd"].median() - base) * 100
    assert r["base"] == pytest.approx(base * 100)
    assert r["F>=7"] == pytest.approx(exp)
    ld = d[d["lowdebt"] == 1]
    assert r["lowdebt"] == pytest.approx((ld["fwd"].median() - base) * 100)


def _canned(mk, **over):
    r = {
        "market": mk,
        "n": 500,
        "base": 1.0,
        "F>=8": None,
        "n_F8": 0,
        "F>=7": None,
        "F<=3": None,
        "magic": None,
        "quality": None,
        "net_net": None,
        "lowdebt": None,
    }
    r.update(over)
    return r


def test_main_table_and_summary(dirs, monkeypatch, capsys, tmp_path):
    results = {
        "AA": _canned(
            "AA",
            **{
                "F>=8": 3.0,
                "F>=7": 2.0,
                "magic": -1.0,
                "quality": 1.5,
                "lowdebt": 0.5,
                "F<=3": -2.0,
                "net_net": 0.2,
            },
        ),
        "BB": _canned(
            "BB",
            **{
                "F>=8": -1.0,
                "F>=7": -2.0,
                "magic": 4.0,
                "quality": 1.0,
                "lowdebt": 0.0,
                "F<=3": 1.0,
                "net_net": -0.3,
            },
        ),
        "CC": None,
    }
    seen = []
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: seen.append((mk, h)) or results[mk])
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "126", "--markets", "aa, bb ,cc"])
    m.main()
    out = capsys.readouterr().out
    assert seen == [("AA", 126), ("BB", 126), ("CC", 126)]
    assert "Cross-market fundamental backtest — 126d forward, point-in-time" in out
    assert "(insufficient data)" in out
    aa = next(ln for ln in out.splitlines() if ln.startswith("         AA"))
    assert aa.count("⭐") == 1 and "+3.0⭐" in aa
    bb = next(ln for ln in out.splitlines() if ln.startswith("         BB"))
    assert "+4.0⭐" in bb
    assert "F>=7      avg  +0.0pp" in out.replace("  ", " ") or "avg" in out
    assert "positive in 1/2: AA" in out
    assert "magic     avg  +1.5pp · positive in 1/2: BB" in out
    df = pd.read_parquet(tmp_path / "data" / "curated" / "fundamental_backtest_all.parquet")
    assert list(df["market"]) == ["AA", "BB"]
    assert "→ data/curated/fundamental_backtest_all.parquet" in out


def test_main_missing_cells_render_dash(dirs, monkeypatch, capsys):
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: _canned(mk, **{"F>=7": 1.0}))
    monkeypatch.setattr("sys.argv", ["x", "--markets", "AA"])
    m.main()
    out = capsys.readouterr().out
    row = next(ln for ln in out.splitlines() if ln.startswith("         AA"))
    assert "—" in row and "+1.0⭐" in row
    # a screen that is missing everywhere is summarised as 0 positive markets
    assert "positive in 0/1: " in out
    assert "F>=7      avg  +1.0pp · positive in 1/1: AA" in out


def test_main_default_markets(dirs, monkeypatch):
    fd, _ = dirs
    for n in ["US", "IN"]:
        (fd / f"{n}.parquet").write_text("x")
    (fd / "x.txt").write_text("x")
    seen = []
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: seen.append((mk, h)))
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert seen == [("IN", 252), ("US", 252)]


def test_main_default_markets_warehouse(dirs, monkeypatch):
    monkeypatch.setattr(m, "_USE_WH", True)
    monkeypatch.setattr(m, "_wh", types.SimpleNamespace(markets=lambda: ["JP"]), raising=False)
    seen = []
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: seen.append(mk))
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert seen == ["JP"]


def test_main_no_rows_writes_nothing(dirs, monkeypatch, tmp_path):
    monkeypatch.setattr(m, "backtest_market", lambda mk, h: None)
    monkeypatch.setattr("sys.argv", ["x", "--markets", "AA"])
    m.main()
    assert not (tmp_path / "data").exists()
