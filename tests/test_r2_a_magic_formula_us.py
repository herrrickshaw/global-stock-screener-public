"""Group A: backtest_magic_formula_us.py."""

import numpy as np
import pandas as pd
import pytest

import backtest_magic_formula_us as m
from tests.r2a_synth import lin_prices


def _hist(n=20, filed_year=2017, ltd=0.0):
    rows = []
    for k in range(1, n + 1):
        rows.append(
            {
                "ticker": f"T{k:02d}",
                "fy_end": f"{filed_year - 1}-12-31",
                "filed": f"{filed_year}-03-01",
                "ebit": 10.0 * k,
                "long_term_debt": ltd,
                "shares": 10.0,
                "total_assets": 600.0,
                "current_liabilities": 100.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def prices(tmp_path, monkeypatch):
    syms = [f"T{k:02d}" for k in range(1, 21)] + ["ZZ"]
    p = tmp_path / "px.parquet"
    lin_prices(syms, start="2017-01-01", n=1200).to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    return p


def test_attach_entry_and_fwd(prices):
    df = pd.DataFrame(
        {
            "ticker": ["t01", "T01", "T01", "T01", "NOPE", "T01"],
            "filed": [
                "2017-03-01",
                "9999-99-99",
                "",
                "2017-03-01",
                "2017-03-01",
                "2030-01-01",  # entry beyond the data
            ],
        }
    )
    out = m._attach_entry_and_fwd(df, 10)
    assert list(out.index) == [0, 3]
    # entry = 2017-03-02 -> idx 60 -> close 160 ; exit idx 70
    assert out["entry_px"].iloc[0] == 160.0
    assert out["fwd"].iloc[0] == pytest.approx(170 / 160 - 1)


def test_attach_horizon_boundary(prices):
    df = pd.DataFrame({"ticker": ["T01"], "filed": ["2017-03-01"]})
    assert len(m._attach_entry_and_fwd(df, 1139)) == 1  # j = 1199
    assert len(m._attach_entry_and_fwd(df, 1140)) == 0


def test_attach_nonpositive_entry(tmp_path, monkeypatch):
    p = tmp_path / "px.parquet"
    lin_prices(["T01"], start="2017-01-01", n=100, base=0.0).to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    df = pd.DataFrame({"ticker": ["T01"], "filed": ["2016-12-31"]})  # entry idx 0, close 0
    assert m._attach_entry_and_fwd(df, 5).empty


def test_magic_ranking_deciles(prices):
    out = m._magic(_hist(), 10)
    assert len(out) == 20
    out = out.set_index("ticker")
    # ev = shares * entry(160) + 0 ; cap_emp = 500
    assert out.loc["T05", "mcap"] == pytest.approx(1600.0)
    assert out.loc["T05", "ey"] == pytest.approx(50 / 1600)
    assert out.loc["T05", "roc"] == pytest.approx(50 / 500)
    assert out.loc["T20", "mf_rank"] == 2.0  # best on both
    assert out.loc["T01", "mf_rank"] == 40.0
    assert out.loc["T20", "decile"] == 1 and out.loc["T19", "decile"] == 1
    assert out.loc["T01", "decile"] == 10 and out.loc["T02", "decile"] == 10
    assert set(out["decile"]) == set(range(1, 11))
    assert (out["cohort"] == 2017).all()


def test_magic_long_term_debt_added_to_ev(prices):
    out = m._magic(_hist(ltd=400.0), 10).set_index("ticker")
    assert out.loc["T05", "ev"] == pytest.approx(2000.0)
    # NaN debt treated as zero
    h = _hist()
    h["long_term_debt"] = np.nan
    out2 = m._magic(h, 10).set_index("ticker")
    assert out2.loc["T05", "ev"] == pytest.approx(1600.0)


def test_magic_filters(prices):
    h = _hist()
    h.loc[0, "total_assets"] = 100.0  # cap_emp = 0 -> excluded
    h.loc[1, "current_liabilities"] = 700.0  # negative -> excluded
    h.loc[2, "ebit"] = np.nan  # dropna
    h.loc[3, "shares"] = np.nan
    out = m._magic(h, 10)
    assert sorted(out["ticker"]) == [f"T{k:02d}" for k in range(5, 21)]


def test_magic_cohorts_ranked_separately(tmp_path, monkeypatch):
    p = tmp_path / "px.parquet"
    lin_prices([f"T{k:02d}" for k in range(1, 21)], start="2017-01-01", n=1500).to_parquet(p)
    monkeypatch.setattr(m, "PRICES", str(p))
    h = pd.concat([_hist(20, 2017), _hist(20, 2018)], ignore_index=True)
    # same tickers appear in both cohorts; each cohort has its own 1..N ranking
    out = m._magic(h, 10)
    assert set(out["cohort"]) == {2017, 2018}
    for c in (2017, 2018):
        sub = out[out["cohort"] == c]
        assert sub["r_ey"].min() == 1.0 and sub["r_ey"].max() == 20.0
        assert set(sub["decile"]) == set(range(1, 11))


@pytest.fixture
def world(tmp_path, monkeypatch, prices):
    hp = tmp_path / "h.parquet"
    _hist().to_parquet(hp)
    monkeypatch.setattr(m, "HIST", str(hp))


def test_main(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "50"])
    m.main()
    out = capsys.readouterr().out
    assert "Magic Formula backtest — US — 50-day forward, point-in-time" in out
    assert "scored rows: 20 | tickers: 20 | cohorts: 2017–2017" in out
    assert "D1 = MF port" in out and "D10 = worst" in out
    drows = [ln for ln in out.splitlines() if ln.strip().startswith("D") and "%" in ln]
    assert len(drows) == 10
    assert "D1−D10 winsorized-mean spread:" in out


def test_main_default_horizon(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    assert "252-day forward" in capsys.readouterr().out


def test_main_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "HIST", str(tmp_path / "x.parquet"))
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert "sec_history_collector.py" in str(e.value)
