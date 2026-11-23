"""Group A: backtest_coffee_can_us.py."""

import pandas as pd
import pytest

import backtest_coffee_can_us as m
from tests.r2a_synth import lin_prices


def _tk(t, roe=20.0, de=0.5, growth=1.15, years=6, equity=100.0, roes=None):
    rows = []
    for y in range(years):
        r = roes[y] if roes is not None else roe
        rows.append(
            {
                "ticker": t,
                "fy_end": f"{2015 + y}-12-31",
                "filed": f"{2016 + y}-02-20",
                "net_income": equity * r / 100,
                "equity": equity,
                "revenue": 100.0 * growth**y,
                "long_term_debt": equity * de,
            }
        )
    return rows


def _score(*groups):
    rows = [r for g in groups for r in g]
    return m._coffee_can(pd.DataFrame(rows)).set_index("ticker")


def test_pass_all_four():
    s = _score(_tk("A"))
    assert len(s) == 1
    assert bool(s.loc["A", "PASS"]) is True
    assert s.loc["A", "n_criteria"] == 4.0
    assert s.loc["A", "rev_cagr5"] == pytest.approx(15.0)


def test_only_evaluable_after_five_years():
    s = m._coffee_can(pd.DataFrame(_tk("A", years=8)))
    assert len(s) == 3  # rows idx 5,6,7 have a 5y-back revenue
    assert m._coffee_can(pd.DataFrame(_tk("A", years=5))).empty


def test_roe_boundary_is_inclusive():
    assert bool(_score(_tk("A", roe=15.0)).loc["A", "PASS"]) is True
    s = _score(_tk("B", roe=14.99))
    assert bool(s.loc["B", "PASS"]) is False
    assert s.loc["B", "n_criteria"] == 2.0  # only revenue + leverage


def test_de_boundary_is_inclusive():
    assert bool(_score(_tk("A", de=1.0)).loc["A", "PASS"]) is True
    s = _score(_tk("B", de=1.01))
    assert bool(s.loc["B", "PASS"]) is False
    assert s.loc["B", "n_criteria"] == 3.0


def test_revenue_cagr_threshold():
    assert bool(_score(_tk("A", growth=1.11)).loc["A", "PASS"]) is True
    s = _score(_tk("B", growth=1.09))
    assert bool(s.loc["B", "PASS"]) is False
    assert s.loc["B", "rev_cagr5"] < 10


def test_consistency_requires_all_recent_years():
    # latest ROE fine but one of the trailing 5 years is below 15
    s = _score(_tk("A", roes=[20, 20, 10, 20, 20, 20]))
    assert bool(s.loc["A", "roe_consistent"]) is False
    assert s.loc["A", "n_criteria"] == 3.0  # roe, rev, de
    # the dip is outside the trailing-5 window (rows 1..5)
    s2 = _score(_tk("B", roes=[5, 20, 20, 20, 20, 20]))
    assert bool(s2.loc["B", "roe_consistent"]) is True


def test_nonpositive_equity_not_evaluable():
    assert _score(_tk("A", equity=-10.0)).empty


def test_numeric_coercion():
    rows = _tk("A")
    for r in rows:
        r["net_income"] = str(r["net_income"])
    assert bool(_score(rows).loc["A", "PASS"]) is True


@pytest.fixture
def prices(tmp_path, monkeypatch):
    p = tmp_path / "px.parquet"
    lin_prices(["AAA", "bbb"], start="2016-01-01", n=3000).to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))


def test_forward_returns(prices):
    sc = pd.DataFrame(
        {
            "ticker": ["aaa", "AAA", "AAA", "AAA", "ZZZ"],
            "filed": ["2016-01-10", "9999-99-99", "", "2016-01-10", "2016-01-10"],
        }
    )
    out = m._forward_returns(sc, 10)
    assert list(out.index) == [0, 3]
    assert out["fwd"].iloc[0] == pytest.approx(120 / 110 - 1)


def test_forward_returns_end_boundary(prices):
    sc = pd.DataFrame({"ticker": ["AAA"], "filed": ["2016-01-01"]})  # entry idx 1
    assert len(m._forward_returns(sc, 2998)) == 1  # j = 2999
    assert len(m._forward_returns(sc, 2999)) == 0  # j = 3000


@pytest.fixture
def world(tmp_path, monkeypatch):
    rows = []
    rows += _tk("AAA")  # passes
    rows += _tk("BBB", roe=10.0)  # fails ROE
    rows += _tk("CCC", de=3.0)
    rows += _tk("DDD", growth=1.02)
    for r in rows:
        r["filed"] = r["filed"]
    hp = tmp_path / "h.parquet"
    pd.DataFrame(rows).to_parquet(hp)
    pp = tmp_path / "p.parquet"
    lin_prices(["AAA", "BBB", "CCC", "DDD"], start="2016-01-01", n=3000).to_parquet(pp)
    monkeypatch.setattr(m, "HIST", str(hp))
    monkeypatch.setattr(m, "PRICES", str(pp))


def test_main(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "50"])
    m.main()
    out = capsys.readouterr().out
    assert "Coffee Can backtest — US — 50-day forward" in out
    assert "scored rows: 4 | tickers: 4 | years: 2020–2020" in out
    lines = [ln for ln in out.splitlines()]
    pass_line = next(ln for ln in lines if ln.strip().startswith("PASS (all 4)"))
    fail_line = next(ln for ln in lines if ln.strip().startswith("FAIL"))
    assert pass_line.split()[3] == "1"  # n column
    assert fail_line.split()[1] == "3"
    assert "by # criteria met:" in out
    assert any(ln.strip().startswith("4/4") for ln in lines)
    assert any(ln.strip().startswith("3/4") for ln in lines)
    assert not any(ln.strip().startswith("0/4") for ln in lines)  # empty groups skipped


def test_main_default_horizon(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert "252-day forward" in capsys.readouterr().out


def test_main_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "HIST", str(tmp_path / "x.parquet"))
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit):
        m.main()
