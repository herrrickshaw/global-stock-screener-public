"""Group A: backtest_combo_us.py."""

import numpy as np
import pandas as pd
import pytest

import backtest_combo_us as m
from tests.r2a_synth import fund_hist, price_panel


def _hist_two_years(t="A", best=True):
    y1 = dict(
        ticker=t,
        fy_end="2018-12-31",
        filed="2019-02-01",
        net_income=10,
        revenue=100,
        gross_profit=30,
        cfo=8,
        ebit=15,
        total_assets=100,
        current_assets=40,
        current_liabilities=20,
        long_term_debt=30,
        equity=50,
        shares=100,
    )
    if best:
        y2 = dict(
            y1,
            fy_end="2019-12-31",
            filed="2020-02-01",
            net_income=20,
            revenue=120,
            gross_profit=60,
            cfo=30,
            current_assets=60,
            long_term_debt=20,
        )
    else:
        y2 = dict(
            y1,
            fy_end="2019-12-31",
            filed="2020-02-01",
            net_income=-5,
            revenue=90,
            gross_profit=10,
            cfo=-1,
            current_assets=20,
            long_term_debt=40,
            shares=120,
        )
    return [y1, y2]


def test_fundamentals_by_ticker(tmp_path, monkeypatch):
    p = tmp_path / "h.parquet"
    pd.DataFrame(_hist_two_years("A") + _hist_two_years("B", best=False)).to_parquet(p)
    monkeypatch.setattr(m, "HIST", str(p))
    out = m._fundamentals_by_ticker()
    assert set(out) == {"A", "B"}
    a = out["A"]
    assert np.isnan(a["F"][0])  # no prior year
    assert a["F"][1] == 9.0
    assert out["B"]["F"][1] == 1.0
    assert a["filed"].dtype.kind == "M"
    # year-2 derived metrics: cap_emp = 100-20 = 80 ; roce = 15/80*100
    assert a["roce"][1] == pytest.approx(18.75)
    assert a["de"][1] == pytest.approx(20 / 50)
    assert a["lowdebt"][1] == 1.0 and out["B"]["lowdebt"][1] == 0.0
    assert a["total_liab"][1] == pytest.approx(50.0)


def test_fundamentals_drops_unparseable_filed_and_coerces(tmp_path, monkeypatch):
    rows = _hist_two_years("A")
    rows.append(dict(rows[1], ticker="C", filed="not-a-date", fy_end="2019-12-31"))
    df = pd.DataFrame(rows)
    df["net_income"] = df["net_income"].astype(str)
    p = tmp_path / "h.parquet"
    df.to_parquet(p)
    monkeypatch.setattr(m, "HIST", str(p))
    out = m._fundamentals_by_ticker()
    assert "C" not in out
    assert out["A"]["F"][1] == 9.0


def test_fundamentals_growth_cagr(tmp_path, monkeypatch):
    h = fund_hist(["A"], years=7, seed=2)
    p = tmp_path / "h.parquet"
    h.to_parquet(p)
    monkeypatch.setattr(m, "HIST", str(p))
    out = m._fundamentals_by_ticker()["A"]
    assert np.isnan(out["rev_cagr5"][4]) and not np.isnan(out["rev_cagr5"][5])
    exp = ((h["revenue"].iloc[5] / h["revenue"].iloc[0]) ** 0.2 - 1) * 100
    assert out["rev_cagr5"][5] == pytest.approx(exp)


# ---------------------------------------------------------------------- _darvas
def _arrays(n=100, top=100.0):
    high = np.full(n, 90.0)
    high[10] = top
    close = np.full(n, 80.0)
    vol = np.full(n, 100.0)
    return high, vol, close


def test_darvas_breakout_true():
    high, vol, close = _arrays()
    t = 70
    close[t] = 101.0
    vol[t] = 121.0  # > 1.2 * 100
    assert m._darvas(high, vol, close, t) is True


def test_darvas_boundaries():
    high, vol, close = _arrays()
    t = 70
    close[t], vol[t] = 101.0, 120.0  # volume exactly 1.2x -> not enough
    assert m._darvas(high, vol, close, t) is False
    vol[t] = 121.0
    close[t] = 100.0  # equal to box top -> not a breakout
    assert m._darvas(high, vol, close, t) is False
    close[t] = 101.0
    close[t - 1] = 100.01  # yesterday already above -> not fresh
    assert m._darvas(high, vol, close, t) is False
    close[t - 1] = 100.0  # <= box_top counts as inside
    assert m._darvas(high, vol, close, t) is True


def test_darvas_zero_avg_volume_and_early_t():
    high, vol, close = _arrays()
    t = 70
    close[t] = 101.0
    vol[:] = 0.0
    vol[t] = 500.0
    assert m._darvas(high, vol, close, t) is False
    high, vol, close = _arrays()
    close[60], vol[60] = 101.0, 500.0
    assert m._darvas(high, vol, close, 60) is False  # t < box + 1
    close[60] = 80.0
    close[61], vol[61] = 101.0, 500.0
    assert m._darvas(high, vol, close, 61) is True  # t == box + 1 allowed


def test_darvas_current_bar_excluded_from_box():
    high, vol, close = _arrays()
    t = 70
    high[t] = 500.0  # today's own high must not raise the box top
    close[t], vol[t] = 101.0, 200.0
    assert m._darvas(high, vol, close, t) is True


def test_darvas_box_window_edges():
    high, vol, close = _arrays(top=90.0)
    t = 70
    high[t - 60] = 120.0  # oldest bar still inside the 60-bar box
    close[t], vol[t] = 110.0, 200.0
    assert m._darvas(high, vol, close, t) is False
    high[t - 60] = 90.0
    high[t - 61] = 120.0  # one bar older than the box
    assert m._darvas(high, vol, close, t) is True


# ------------------------------------------------------------------ _fund_flags
def _fd(**over):
    base = dict(
        F=8.0,
        roce=20.0,
        de=0.5,
        lowdebt=1.0,
        rev_cagr5=10.0,
        ni_cagr5=20.0,
        ebit=100.0,
        long_term_debt=50.0,
        current_assets=2000.0,
        total_liab=100.0,
        net_income=50.0,
        shares=10.0,
    )
    base.update(over)
    d = {k: np.array([v]) for k, v in base.items()}
    d["filed"] = np.array(["2020-01-01"], dtype="datetime64[ns]")
    return d


D = np.datetime64("2020-06-01")


def test_flags_none_before_first_filing():
    assert not any(m._fund_flags(_fd(), np.datetime64("2019-12-31"), 100.0).values())
    # filed on the as-of date counts (side='right')
    assert m._fund_flags(_fd(), np.datetime64("2020-01-01"), 100.0)["piotroski7"]


def test_flags_uses_latest_filing_only():
    fd = _fd()
    fd["filed"] = np.array(["2019-01-01", "2020-01-01"], dtype="datetime64[ns]")
    for k in list(fd):
        if k != "filed":
            fd[k] = np.array([3.0, fd[k][0]])
    fd["F"] = np.array([3.0, 8.0])
    assert m._fund_flags(fd, np.datetime64("2019-06-01"), 100.0)["piotroski7"] is False
    assert m._fund_flags(fd, np.datetime64("2020-06-01"), 100.0)["piotroski7"] is True


def test_flag_piotroski_boundary():
    assert m._fund_flags(_fd(F=7.0), D, 100.0)["piotroski7"] is True
    assert m._fund_flags(_fd(F=6.99), D, 100.0)["piotroski7"] is False
    assert m._fund_flags(_fd(F=np.nan), D, 100.0)["piotroski7"] is False


def test_flag_quality_boundaries():
    assert m._fund_flags(_fd(roce=15.0, de=1.0), D, 100.0)["quality"] is True
    assert m._fund_flags(_fd(roce=14.99), D, 100.0)["quality"] is False
    assert m._fund_flags(_fd(de=1.01), D, 100.0)["quality"] is False
    assert m._fund_flags(_fd(de=np.nan), D, 100.0)["quality"] is False
    assert m._fund_flags(_fd(roce=np.nan), D, 100.0)["quality"] is False


def test_flag_lowdebt():
    assert m._fund_flags(_fd(lowdebt=1.0), D, 100.0)["lowdebt"] is True
    assert m._fund_flags(_fd(lowdebt=0.0), D, 100.0)["lowdebt"] is False


def test_flag_magic_boundaries():
    # mcap = 10*100 = 1000, ev = 1050
    assert m._fund_flags(_fd(ebit=84.0), D, 100.0)["magic"] is True  # ey == 0.08
    assert m._fund_flags(_fd(ebit=83.9), D, 100.0)["magic"] is False
    assert m._fund_flags(_fd(ebit=84.0, roce=14.9), D, 100.0)["magic"] is False
    assert m._fund_flags(_fd(ebit=np.nan), D, 100.0)["magic"] is False
    # NaN debt counts as zero: ev = 1000 -> ey 0.084 passes with ebit 84
    assert m._fund_flags(_fd(ebit=80.0, long_term_debt=np.nan), D, 100.0)["magic"] is True


def test_flag_garp_boundaries():
    # eps = 5 ; price 100 -> pe 20 ; needs pe <= ni_cagr5
    assert m._fund_flags(_fd(ni_cagr5=20.0), D, 100.0)["garp"] is True
    assert m._fund_flags(_fd(ni_cagr5=20.0), D, 100.1)["garp"] is False
    assert m._fund_flags(_fd(ni_cagr5=14.99), D, 1.0)["garp"] is False  # growth < 15
    assert m._fund_flags(_fd(ni_cagr5=15.0), D, 70.0)["garp"] is True  # growth == 15
    assert m._fund_flags(_fd(net_income=0.0), D, 1.0)["garp"] is False
    assert m._fund_flags(_fd(ni_cagr5=np.nan), D, 1.0)["garp"] is False


def test_flag_net_net_boundaries():
    # ncav = 2000 - 100 = 1900 ; mcap = 10 * price
    assert m._fund_flags(_fd(), D, 189.9)["net_net"] is True
    assert m._fund_flags(_fd(), D, 190.0)["net_net"] is False  # mcap == ncav
    assert m._fund_flags(_fd(current_assets=50.0), D, 0.1)["net_net"] is False  # ncav < 0
    assert m._fund_flags(_fd(total_liab=np.nan), D, 1.0)["net_net"] is False


def test_flags_bad_shares_short_circuit():
    for sh in (0.0, np.nan, -5.0):
        assert not any(m._fund_flags(_fd(shares=sh), D, 100.0).values())


# --------------------------------------------------------------------- main
@pytest.fixture
def world(tmp_path, monkeypatch):
    syms = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    px = price_panel(syms, n=500, seed=11, start="2018-01-01")
    px["Symbol"] = px["Symbol"].str.lower()  # must be upper-cased by main
    pp = tmp_path / "px.parquet"
    px.to_parquet(pp)
    hist = fund_hist(syms[:4], years=7, start_year=2013, seed=4)  # EEE/FFF: no history
    hp = tmp_path / "h.parquet"
    hist.to_parquet(hp)
    monkeypatch.setattr(m, "PRICES", str(pp))
    monkeypatch.setattr(m, "HIST", str(hp))


def test_main_report(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "20", "--step", "5"])
    m.main()
    out = capsys.readouterr().out
    assert "loading fundamentals…" in out
    assert "  4 tickers with SEC history" in out
    # 6 symbols * windows range(200, 500-20-1, 5)
    n = 6 * len(range(200, 479, 5))
    assert f"windows: {n:,}" in out
    assert "Combo backtest — US — 20-day forward, point-in-time" in out
    assert "── PRICE screens ──" in out
    assert "── FUNDAMENTAL screens (screener.in-style) ──" in out
    assert "── COMBINATIONS (price × fundamental) ──" in out
    for name in m.PRICE_SCREENS + m.FUND_SCREENS:
        assert any(ln.strip().startswith(name) for ln in out.splitlines())
    for a, b in m.COMBOS:
        assert f"{a}×{b}" in out
    # sparse combos are reported as skipped, populated ones print stats
    assert "(n<30 — skipped)" in out
    assert "%" in out


def test_main_defaults(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x"])
    m.main()
    out = capsys.readouterr().out
    assert "126-day forward" in out


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="no eligible windows -> np.nanpercentile of empty array returns a scalar and "
    "the `lo, hi = ...` unpack raises TypeError instead of reporting 'no data'",
)
def test_main_short_history_reports_no_data(world, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "400"])
    # n=500 < 200+400+1 -> every symbol skipped -> no windows
    m.main()


def test_main_skips_nan_zero_and_negative_bars(world, tmp_path, monkeypatch, capsys):
    px = pd.read_parquet(m.PRICES)
    sel = px["Symbol"] == "aaa"
    close = px.loc[sel, "Close"].to_numpy().copy()
    close[200] = np.nan  # window t=200: non-finite close
    close[270] = np.nan  # window t=250 (h=20): non-finite forward return
    close[300] = -5.0  # window t=300: finite fwd but no price features
    px.loc[sel, "Close"] = close
    q = tmp_path / "px2.parquet"
    px.to_parquet(q)
    monkeypatch.setattr(m, "PRICES", str(q))
    monkeypatch.setattr("sys.argv", ["x", "--horizon", "20", "--step", "5"])
    m.main()
    out = capsys.readouterr().out
    # t=200 (nan close), t=250 (nan fwd), t=270 (nan close) are dropped; t=300 kept
    full = 6 * len(range(200, 479, 5))
    assert f"windows: {full - 3:,}" in out


def test_main_missing_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "HIST", str(tmp_path / "a.parquet"))
    monkeypatch.setattr(m, "PRICES", str(tmp_path / "b.parquet"))
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit, match="missing"):
        m.main()
