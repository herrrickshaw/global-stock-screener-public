"""Group A: backtest_piotroski_us.py."""

import numpy as np
import pandas as pd
import pytest

import backtest_piotroski_us as m
from tests.r2a_synth import fund_hist, lin_prices

COLS = [
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


def _row(t, fy, filed, ni, rev, gp, cfo, ta, ca, cl, ltd, sh):
    return dict(zip(COLS, [t, fy, filed, ni, rev, gp, cfo, ta, ca, cl, ltd, sh], strict=True))


Y1 = (10, 100, 30, 8, 100, 40, 20, 30, 100)


def _two_years(t, y2):
    return [
        _row(t, "2018-12-31", "2019-02-01", *Y1),
        _row(t, "2019-12-31", "2020-02-01", *y2),
    ]


def test_piotroski_all_nine():
    # ni up, cfo > ni, cr up, lev down, shares flat (<=), gm up, asset turnover up
    df = pd.DataFrame(_two_years("A", (20, 120, 60, 30, 100, 60, 20, 20, 100)))
    out = m._piotroski(df)
    assert list(out["ticker"]) == ["A"]  # first year dropped (no prior)
    assert out["F"].iloc[0] == 9.0


def test_piotroski_worst_case_scores_one():
    df = pd.DataFrame(_two_years("B", (-5, 90, 10, -1, 100, 20, 20, 40, 120)))
    out = m._piotroski(df)
    # only (cfo/ta) > roa holds: -0.01 > -0.05
    assert out["F"].iloc[0] == 1.0


def test_piotroski_share_equality_counts():
    base = (20, 120, 60, 30, 100, 60, 20, 20, 100)
    hi = (20, 120, 60, 30, 100, 60, 20, 20, 101)
    f_eq = m._piotroski(pd.DataFrame(_two_years("A", base)))["F"].iloc[0]
    f_hi = m._piotroski(pd.DataFrame(_two_years("A", hi)))["F"].iloc[0]
    assert f_eq - f_hi == 1.0


def test_piotroski_sorts_and_groups_by_ticker():
    rows = _two_years("B", (-5, 90, 10, -1, 100, 20, 20, 40, 120))
    rows += _two_years("A", (20, 120, 60, 30, 100, 60, 20, 20, 100))
    rows.reverse()  # unsorted input
    out = m._piotroski(pd.DataFrame(rows)).set_index("ticker")["F"]
    assert out["A"] == 9.0 and out["B"] == 1.0


def test_forward_returns_exact(tmp_path, monkeypatch):
    px = lin_prices(["AAA"], start="2019-01-01", n=100)
    p = tmp_path / "px.parquet"
    px.to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    sc = pd.DataFrame(
        {
            "ticker": ["aaa", "AAA", "AAA", "AAA", "ZZZ", "AAA"],
            "filed": [
                "2019-01-10",  # entry 2019-01-11 -> idx 10
                "2019-01-10",
                "9999-99-99",
                "",
                "2019-01-10",
                "2019-01-10",
            ],
        }
    )
    out = m._forward_returns(sc, 5)
    # rows 0,1,5 valid; sentinel / empty / unknown ticker dropped
    assert list(out.index) == [0, 1, 5]
    assert out["fwd"].iloc[0] == pytest.approx(115 / 110 - 1)


def test_forward_returns_horizon_boundary(tmp_path, monkeypatch):
    px = lin_prices(["AAA"], start="2019-01-01", n=20)
    p = tmp_path / "px.parquet"
    px.to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    sc = pd.DataFrame({"ticker": ["AAA"], "filed": ["2019-01-04"]})  # entry idx 4
    assert len(m._forward_returns(sc, 15)) == 1  # j = 19 = last bar
    assert len(m._forward_returns(sc, 16)) == 0  # j = 20 out of range
    late = pd.DataFrame({"ticker": ["AAA"], "filed": ["2020-01-01"]})  # entry beyond data
    assert len(m._forward_returns(late, 1)) == 0


def test_forward_returns_nonpositive_entry(tmp_path, monkeypatch):
    px = lin_prices(["AAA"], start="2019-01-01", n=20, base=0.0, slope=1.0)
    p = tmp_path / "px.parquet"
    px.to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    sc = pd.DataFrame({"ticker": ["AAA", "AAA"], "filed": ["2018-12-31", "2019-01-04"]})
    # first entry idx 0 has close 0 -> dropped; second idx 4 valid
    out = m._forward_returns(sc, 3)
    assert list(out.index) == [1]


@pytest.fixture
def world(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(8)]
    hist = fund_hist(tickers, years=5, start_year=2018, seed=3)
    # make a few tickers deteriorate so several F values appear
    bad = hist["ticker"].isin(["T0", "T1", "T2"]) & (hist["fy_end"] > "2019-12-31")
    hist.loc[bad, "net_income"] = -5.0
    hist.loc[bad, "cfo"] = -1.0
    hp = tmp_path / "hist.parquet"
    hist.to_parquet(hp)
    px = lin_prices(tickers, start="2018-06-01", n=2000)
    pp = tmp_path / "px.parquet"
    px.to_parquet(pp)
    monkeypatch.setattr(m, "HIST", str(hp))
    monkeypatch.setattr(m, "PRICES", str(pp))
    return tickers


def test_main_report(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "30"])
    m.main()
    out = capsys.readouterr().out
    assert "Piotroski F-Score backtest — US — 30-day forward, point-in-time" in out
    # 8 tickers x 4 scored years
    assert "scored rows: 32 | tickers: 8 | years: 2019–2022" in out
    assert "baseline (winsorized mean" in out
    assert "by exact F (winsorized mean · median):" in out
    assert "  F=" in out or "    F=" in out


def test_main_default_horizon(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert "126-day forward" in capsys.readouterr().out


def test_main_missing_hist(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "HIST", str(tmp_path / "nope.parquet"))
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert "missing" in str(e.value) and "sec_history_collector.py" in str(e.value)


def test_main_band_rows(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "30"])
    m.main()
    out = capsys.readouterr().out
    n_by_f = 0
    for line in out.splitlines():
        if line.strip().startswith("F=") and "n=" in line:
            n_by_f += int(line.split("n=")[1].split()[0].replace(",", ""))
    assert n_by_f == 32
    assert np.isfinite(n_by_f)
