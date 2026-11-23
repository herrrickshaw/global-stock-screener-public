"""Group A: backtest_piotroski_in.py (direct + screener.in schemas)."""

import pandas as pd
import pytest

import backtest_piotroski_in as m
from tests.r2a_synth import fund_hist, lin_prices

DCOLS = [
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
]


def _d(t, fy, *v):
    return dict(zip(DCOLS, [t, fy, *v], strict=True))


D1 = (10, 100, 30, 8, 100, 40, 20, 30, 100)
D_BEST = (20, 120, 60, 30, 100, 60, 20, 20, 100)


def test_direct_schema_best_and_worst():
    df = pd.DataFrame(
        [
            _d("A", "2018-03-31", *D1),
            _d("A", "2019-03-31", *D_BEST),
            _d("B", "2018-03-31", *D1),
            _d("B", "2019-03-31", -5, 90, 10, -1, 100, 20, 20, 40, 120),
        ]
    )
    out = m._piotroski(df).set_index("ticker")["F"]
    assert out["A"] == 9.0
    assert out["B"] == 1.0


def test_direct_share_tolerance_two_percent():
    ok = list(D_BEST)
    ok[-1] = 102  # == 1.02 * 100 -> counted (<=)
    bad = list(D_BEST)
    bad[-1] = 103  # beyond tolerance
    f = []
    for v in (ok, bad):
        df = pd.DataFrame([_d("A", "2018-03-31", *D1), _d("A", "2019-03-31", *v)])
        f.append(m._piotroski(df)["F"].iloc[0])
    assert f[0] - f[1] == 1.0


def test_direct_debt_free_nan_ltd_treated_as_zero():
    y2 = list(D_BEST)
    y2[7] = None  # long_term_debt NaN -> 0 -> lev 0 < 0.3
    y1 = list(D1)
    df = pd.DataFrame([_d("A", "2018-03-31", *y1), _d("A", "2019-03-31", *y2)])
    assert m._piotroski(df)["F"].iloc[0] == 9.0


def test_direct_zero_prior_assets_gives_nan_and_drops():
    y1 = list(D1)
    y1[4] = 0  # total_assets prior == 0
    df = pd.DataFrame([_d("A", "2018-03-31", *y1), _d("A", "2019-03-31", *D_BEST)])
    assert m._piotroski(df).empty


SCOLS = [
    "ticker",
    "fy_end",
    "net_income",
    "revenue",
    "cfo",
    "borrowings",
    "reserves",
    "equity_capital",
    "other_liab",
    "inventory",
    "receivables",
    "cash",
    "shares",
]


def _s(t, fy, *v):
    return dict(zip(SCOLS, [t, fy, *v], strict=True))


def test_screener_schema_derives_totals():
    # year1: eq=100 (10+90) ta = 100+50+50 = 200 ; cur_assets 40 / other_liab 50
    y1 = (10, 100, 8, 50, 90, 10, 50, 10, 20, 10, 100)
    # year2: ni up, cfo > ni, borrowings down, cur ratio up, net margin up, turnover up
    y2 = (30, 150, 40, 20, 100, 10, 50, 30, 20, 20, 100)
    df = pd.DataFrame([_s("A", "2018-03-31", *y1), _s("A", "2019-03-31", *y2)])
    out = m._piotroski(df)
    assert "total_assets" in out.columns
    assert out["total_assets"].iloc[0] == pytest.approx(180.0)  # 110 + 20 + 50
    assert out["F"].iloc[0] == 9.0


def test_screener_schema_all_total_assets_nan_uses_screener_branch():
    # a total_assets column that is all-NaN must NOT select the direct branch
    y1 = (10, 100, 8, 50, 90, 10, 50, 10, 20, 10, 100)
    df = pd.DataFrame([_s("A", "2018-03-31", *y1), _s("A", "2019-03-31", *y1)])
    df["total_assets"] = float("nan")
    out = m._piotroski(df)
    assert out["total_assets"].iloc[0] == pytest.approx(200.0)


def test_forward_returns_lag_and_horizon(tmp_path, monkeypatch):
    px = lin_prices(["aaa"], start="2019-01-01", n=200)
    p = tmp_path / "px.parquet"
    px.to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    sc = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "NOPE"],
            "fy_end": ["2019-01-01", "2019-03-01", "2019-01-01"],
        }
    )
    out = m._forward_returns(sc, 10, 30)
    # row0 entry 2019-01-31 -> idx 30 -> 140/130 ... close = 100 + idx
    assert out["fwd"].iloc[0] == pytest.approx(140 / 130 - 1)
    # lag 0 -> idx 0
    out0 = m._forward_returns(sc.iloc[:1], 10, 0)
    assert out0["fwd"].iloc[0] == pytest.approx(110 / 100 - 1)
    # out of range
    assert m._forward_returns(sc.iloc[:1], 10, 195).empty
    assert len(m._forward_returns(sc.iloc[:1], 169, 30)) == 1  # j = 199
    assert len(m._forward_returns(sc.iloc[:1], 170, 30)) == 0  # j = 200


@pytest.fixture
def world(tmp_path, monkeypatch):
    tickers = [f"T{i}" for i in range(6)]
    hist = fund_hist(tickers, years=5, start_year=2018, seed=5)
    hist.loc[hist["ticker"].isin(["T0", "T1"]) & (hist["fy_end"] > "2019-12-31"), "cfo"] = -2.0
    hp = tmp_path / "hist.parquet"
    hist.to_parquet(hp)
    px = lin_prices(tickers, start="2018-06-01", n=2000)
    pp = tmp_path / "px.parquet"
    px.to_parquet(pp)
    monkeypatch.setattr(m, "HIST", str(hp))
    monkeypatch.setattr(m, "PRICES", str(pp))


def test_main(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "20", "--lag", "45"])
    m.main()
    out = capsys.readouterr().out
    assert "INDIA — 20-day forward, entry=FY-end+45d" in out
    assert "scored rows: 24 | tickers: 6 | years: 2019–2022" in out
    assert "F band" in out
    assert "by exact F" in out


def test_main_defaults(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert "126-day forward, entry=FY-end+90d" in capsys.readouterr().out


def test_main_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "HIST", str(tmp_path / "no.parquet"))
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert "screener_history_collector.py" in str(e.value)
