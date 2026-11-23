"""Group A: reward_backtest.py."""

import math

import numpy as np
import pandas as pd
import pytest

import reward_backtest as m


def test_parse_labels():
    assert m._parse("vol_60↓10%") == ("vol_60", -1, 0.10)
    assert m._parse("mom_12_1↑30%") == ("mom_12_1", +1, 0.30)
    assert m._parse("above_200dma↑50%") == ("above_200dma", 1, 0.5)
    f, d, k = m._parse("x↓5%")
    assert k == pytest.approx(0.05)


def test_compound_and_cagr():
    assert m._compound([0.1, 0.1]) == pytest.approx(0.21)
    assert m._compound([0.5, -0.5]) == pytest.approx(-0.25)
    assert math.isnan(m._compound([]))
    assert m._cagr([0.1, 0.1]) == pytest.approx(0.1)
    assert m._cagr([0.44, 0.0]) == pytest.approx(1.44**0.5 - 1)
    assert math.isnan(m._cagr([]))
    assert m._compound(np.array([0.2])) == pytest.approx(0.2)


def _panel(years=(2020, 2022, 2023), n=20, flip=(), extra_illiquid=True, short=()):
    """Per year: n liquid rows with factor f=0..n-1 ; top half earns 0.2, bottom 0."""
    parts = []
    for y in years:
        k = n - 1 if y in short else n
        f = np.arange(k, dtype=float)
        fwd = np.where(f >= k / 2, 0.2, 0.0)
        if y in flip:
            fwd = 0.2 - fwd
        df = pd.DataFrame({"year": y, "mom_12_1": f, "fwd": fwd, "dvol": 1.0})
        parts.append(df)
        if extra_illiquid:
            parts.append(pd.DataFrame({"year": y, "mom_12_1": f, "fwd": 9.0, "dvol": 0.1}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def panel(monkeypatch):
    state = {"df": _panel()}
    monkeypatch.setattr(m.U, "_panel", lambda market: state["df"].copy())
    return state


def test_backtest_market_empty(panel):
    panel["df"] = pd.DataFrame(columns=["year", "mom_12_1", "fwd", "dvol"])
    assert m.backtest_market("XX", "mom_12_1↑50%") is None


def test_backtest_market_values(panel):
    r = m.backtest_market("XX", "mom_12_1↑50%")
    assert (r["market"], r["factor"]) == ("XX", "mom_12_1↑50%")
    # 2021 absent; liquidity gate removes the dvol=0.1 rows (fwd=9.0)
    assert r["years"] == 3
    d = r["detail"]
    assert d["year"].tolist() == [2020, 2022, 2023]
    assert d["n"].tolist() == [10, 10, 10]
    assert d["port"].tolist() == pytest.approx([0.2] * 3)
    assert d["bench"].tolist() == pytest.approx([0.1] * 3)
    assert r["port_cagr"] == pytest.approx(0.2)
    assert r["bench_cagr"] == pytest.approx(0.1)
    assert r["port_cum"] == pytest.approx(1.2**3 - 1)
    assert r["bench_cum"] == pytest.approx(1.1**3 - 1)
    assert r["alpha_yr"] == pytest.approx(0.1)
    assert r["win_yrs"] == 1.0
    assert r["oos_port_cagr"] == pytest.approx(0.2)
    assert r["oos_bench_cagr"] == pytest.approx(0.1)
    assert r["oos_alpha_yr"] == pytest.approx(0.1)
    assert r["oos_win"] == 1.0


def test_backtest_market_low_side_factor(panel):
    r = m.backtest_market("XX", "mom_12_1↓50%")
    # bottom half (f <= median) earns 0 -> negative alpha
    assert r["detail"]["port"].tolist() == pytest.approx([0.0] * 3)
    assert r["alpha_yr"] == pytest.approx(-0.1)
    assert r["win_yrs"] == 0.0
    assert r["oos_win"] == 0.0


def test_backtest_market_losing_year_and_oos_split(panel):
    panel["df"] = _panel(flip=(2023,))
    r = m.backtest_market("XX", "mom_12_1↑50%")
    assert r["win_yrs"] == pytest.approx(2 / 3)
    assert r["oos_win"] == pytest.approx(0.5)  # 2022 wins, 2023 loses
    assert r["oos_alpha_yr"] == pytest.approx(0.0)
    assert r["alpha_yr"] == pytest.approx(0.1 / 3)


def test_oos_boundary_year(panel):
    # OOS_FROM is 2022 inclusive: a 2021 row is full-sample only
    panel["df"] = _panel(years=(2021, 2022))
    r = m.backtest_market("XX", "mom_12_1↑50%")
    assert r["years"] == 2
    assert r["oos_alpha_yr"] == pytest.approx(0.1)
    assert len(r["detail"][r["detail"]["year"] >= m.OOS_FROM]) == 1


def test_backtest_market_no_oos_years(panel):
    panel["df"] = _panel(years=(2019, 2020))
    r = m.backtest_market("XX", "mom_12_1↑50%")
    assert math.isnan(r["oos_alpha_yr"]) and math.isnan(r["oos_win"])
    assert math.isnan(r["oos_port_cagr"])
    assert r["years"] == 2


def test_year_cross_section_minimum(panel):
    # exactly 20 liquid rows are kept, 19 are skipped
    panel["df"] = _panel(short=(2022,))
    r = m.backtest_market("XX", "mom_12_1↑50%")
    assert r["detail"]["year"].tolist() == [2020, 2023]


def test_minimum_selected_names(panel):
    # keep 20% of 20 rows -> thr at q0.8 -> 4 names (<5) -> year skipped
    assert m.backtest_market("XX", "mom_12_1↑20%") is None
    # keep 30% -> thr = q0.7 = 13.3 -> f>=13.3 -> 6 names
    r = m.backtest_market("XX", "mom_12_1↑30%")
    assert r["detail"]["n"].tolist() == [6, 6, 6]


def test_liquidity_gate_uses_median(panel):
    # without the illiquid half nothing is filtered
    panel["df"] = _panel(extra_illiquid=False)
    r = m.backtest_market("XX", "mom_12_1↑50%")
    assert r["detail"]["n"].tolist() == [10, 10, 10]
    # with it, the 0.1-dvol rows (fwd=9) would wreck returns if kept
    assert r["port_cagr"] == pytest.approx(0.2)


def test_winsorization_clips_outlier(panel):
    df = _panel(years=(2020,), extra_illiquid=False, n=100)
    df.loc[99, "fwd"] = 50.0  # one lottery name in the selected half
    panel["df"] = df
    r = m.backtest_market("XX", "mom_12_1↑50%")
    raw_mean = df.loc[df["mom_12_1"] >= 49.5, "fwd"].mean()
    assert r["detail"]["port"].iloc[0] < raw_mean
    assert r["detail"]["bench"].iloc[0] < df["fwd"].mean()


# ----------------------------------------------------------------------- main
@pytest.fixture
def main_env(tmp_path, monkeypatch, panel):
    csv = tmp_path / "reports" / "reward_universe_liq50.csv"
    csv.parent.mkdir()
    pd.DataFrame(
        {
            "market": ["AA", "BB", "CC", "DD"],
            "factor": ["mom_12_1↑50%", "mom_12_1↓50%", "mom_12_1↑50%", "mom_12_1↑50%"],
            "robust": [True, True, True, False],
        }
    ).to_csv(csv, index=False)
    monkeypatch.setattr(m, "LIQ_CSV", str(csv))
    monkeypatch.setattr(m, "BASE", str(tmp_path))

    def fake_panel(market):
        if market == "CC":
            return pd.DataFrame(columns=["year", "mom_12_1", "fwd", "dvol"])
        return panel["df"].copy()

    monkeypatch.setattr(m.U, "_panel", fake_panel)
    return tmp_path


def test_main_report_and_csv(main_env, capsys):
    m.main()
    out = capsys.readouterr().out
    assert "PORTFOLIO BACKTEST — reward-optimised screens" in out
    assert " CC   (no backtest)" in out
    assert " DD" not in out  # non-robust rows are never run
    aa = next(ln for ln in out.splitlines() if ln.startswith(" AA"))
    assert "CAGR   20.0%  10.0%    10.0%" in aa and "100%" in aa
    bb = next(ln for ln in out.splitlines() if ln.startswith(" BB"))
    assert "-10.0%" in bb
    assert "Portfolio beat benchmark OOS (positive alpha 2022-24): 1/2 — AA" in out
    df = pd.read_csv(main_env / "reports" / "reward_backtest.csv")
    assert list(df["market"]) == ["AA", "BB"]
    assert "detail" not in df.columns
    assert df.loc[0, "alpha_yr"] == pytest.approx(0.1)
    assert "written → reports/reward_backtest.csv" in out


def test_main_nothing_robust(main_env, monkeypatch, capsys):
    csv = main_env / "reports" / "reward_universe_liq50.csv"
    pd.DataFrame({"market": ["AA"], "factor": ["mom_12_1↑50%"], "robust": [False]}).to_csv(
        csv, index=False
    )
    m.main()
    out = capsys.readouterr().out
    assert "beat benchmark OOS (positive alpha 2022-24): 0/0" in out
