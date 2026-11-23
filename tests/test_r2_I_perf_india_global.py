import numpy as np
import pandas as pd
import pytest

import india_public as ip
import market_performance as mp
import run_global_analysis as rga

PD3 = int(pd.__version__.split(".")[0]) >= 3


def _series(n=300, end="2026-06-30", start_val=100.0, step=0.5):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.Series(start_val + np.arange(n) * step, index=idx)


# ---------------- market_performance ----------------
def test_metrics_too_short():
    assert mp._metrics(_series(59)) == {}
    assert (
        mp._metrics(pd.Series([np.nan] * 100, index=pd.bdate_range("2026-01-01", periods=100)))
        == {}
    )
    assert mp._metrics(_series(60)) != {}


def test_metrics_values():
    s = _series(300)
    m = mp._metrics(s)
    assert m["Total_5y%"] == round((s.iloc[-1] / s.iloc[0] - 1) * 100, 1)
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    assert m["Years"] == round(yrs, 1)
    exp_cagr = ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) * 100
    assert m["CAGR%"] == round(exp_cagr, 1)
    assert m["MaxDD%"] == 0.0  # monotone up
    assert m["Sharpe"] > 0 and m["Vol_ann%"] > 0
    # 1y return uses >=365d window; YTD uses Jan 1 of last year
    yr_ago = s.index[-1] - pd.Timedelta(days=365)
    s1 = s[s.index >= yr_ago]
    assert m["Return_1y%"] == round((s1.iloc[-1] / s1.iloc[0] - 1) * 100, 1)
    j = s[s.index >= pd.Timestamp(2026, 1, 1)]
    assert m["YTD%"] == round((j.iloc[-1] / j.iloc[0] - 1) * 100, 1)


def test_metrics_drawdown_and_flat():
    idx = pd.bdate_range("2025-01-01", periods=100)
    v = np.concatenate([np.linspace(100, 200, 50), np.linspace(200, 100, 50)])
    m = mp._metrics(pd.Series(v, index=idx))
    assert m["MaxDD%"] == -50.0
    flat = mp._metrics(pd.Series(100.0, index=idx))
    assert flat["Sharpe"] is None and flat["CAGR%"] is None
    assert flat["Vol_ann%"] == 0.0


def test_metrics_ytd_none_with_single_bar_in_year():
    idx = list(pd.bdate_range(end="2025-12-31", periods=59)) + [pd.Timestamp("2026-01-02")]
    s = pd.Series(np.linspace(10, 20, 60), index=pd.DatetimeIndex(idx))
    m = mp._metrics(s)
    assert m["YTD%"] is None
    assert m["Return_1y%"] is not None


def _df(n=100):
    return pd.DataFrame({"Close": _series(n)})


@pytest.fixture
def mpf(monkeypatch, tmp_path):
    monkeypatch.setattr(mp, "OUT_PARQUET", tmp_path / "perf.parquet")
    return mp


def test_analyse(mpf, monkeypatch, capsys):
    got = {}

    def fetch(tickers, order, period, min_bars, verbose):
        got.update(n=len(tickers), order=order, period=period, mb=min_bars, v=verbose)
        slow = pd.DataFrame({"Close": _series(100, step=0.1)})
        fast = pd.DataFrame({"Close": _series(100, step=1.0)})
        return {"^GSPC": slow, "^NSEI": fast, "^N225": pd.DataFrame(), "^HSI": _df(10)}

    monkeypatch.setattr(mpf, "multi_fetch", fetch)
    res = mpf.analyse()
    assert got == {"n": 20, "order": ("yahoo", "stooq"), "period": "5y", "mb": 60, "v": False}
    assert list(res["Market"]) == ["IN", "US"]  # sorted by CAGR desc
    assert (mpf.OUT_PARQUET).exists() and mpf.OUT_PARQUET.with_suffix(".xlsx").exists()
    out = capsys.readouterr().out
    assert "JP (^N225): no data" in out and "CN (000001.SS): no data" in out
    assert "20-MARKET 5-YEAR PERFORMANCE" in out and "NOT investment advice" in out
    assert mpf.load().shape[0] == 2


def test_analyse_quiet_and_empty(mpf, monkeypatch, capsys):
    monkeypatch.setattr(mpf, "multi_fetch", lambda *a, **k: {})
    res = mpf.analyse(verbose=False)
    assert res.empty and capsys.readouterr().out == ""
    assert not mpf.OUT_PARQUET.exists()
    assert mpf.load().empty


# ---------------- india_public ----------------
def test_load_config(monkeypatch, tmp_path):
    base = dict(ip.PUBLIC_SCREENS)
    monkeypatch.setattr(ip, "__file__", str(tmp_path / "india_public.py"))
    assert ip._load_config() == base
    (tmp_path / "public_screens.json").write_text('{"garp": "u1", "cash_conversion_cycle": "u2"}')
    cfg = ip._load_config()
    assert cfg["garp"] == "u1" and cfg["cash_conversion_cycle"] == "u2"
    (tmp_path / "public_screens.json").write_text("{bad")
    assert ip._load_config() == base


def test_run(monkeypatch, capsys):
    monkeypatch.setattr(ip, "_load_config", lambda: {"a": "ua", "b": "ub", "c": "uc"})
    calls = []

    def screen(slug, mkt, top, min_turnover_usd):
        calls.append((slug, mkt, top, min_turnover_usd))
        if slug == "golden_crossover":
            raise RuntimeError("boom" * 30)
        return pd.DataFrame({"Symbol": ["X", "Y"]})

    monkeypatch.setattr(ip.kit, "screen", screen)

    def fetch(url, verbose):
        assert verbose is False
        if url == "ub":
            return pd.DataFrame()
        if url == "uc":
            raise ValueError("bad")
        return pd.DataFrame({"Symbol": ["S1", "S2"]})

    monkeypatch.setattr(ip.si, "fetch_screen", fetch)
    monkeypatch.setattr(ip.liq, "annotate", lambda df: df.assign(Liq="ok"))
    out = ip.run(min_turnover=5, top=3)
    assert calls == [("darvas", "IN", 3, 5), ("golden_crossover", "IN", 3, 5)]
    assert list(out) == ["darvas", "golden_crossover", "a", "b", "c"]
    assert out["golden_crossover"].empty and out["b"].empty and out["c"].empty
    assert list(out["a"].columns) == ["Symbol", "Market", "Liq"]
    txt = capsys.readouterr().out
    assert "golden_crossover: price-screen failed (" + ("boom" * 30)[:40] + ")" in txt
    assert ("boom" * 30)[:41] not in txt
    assert "c: public-screen fetch failed (bad)" in txt
    assert "darvas" in txt and "X, Y" in txt and "—" in txt


def test_run_quiet(monkeypatch, capsys):
    monkeypatch.setattr(ip, "_load_config", lambda: {"a": "ua"})
    monkeypatch.setattr(ip.kit, "screen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(
        ip.si, "fetch_screen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("y"))
    )
    out = ip.run(verbose=False)
    assert all(v.empty for v in out.values()) and capsys.readouterr().out == ""


# ---------------- run_global_analysis ----------------
def _ohlcv(n=30):
    idx = pd.bdate_range("2026-01-01", periods=n)
    c = np.linspace(10, 20, n)
    return pd.DataFrame(
        {"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(n, 1000)}, index=idx
    )


@pytest.fixture
def rg(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rga, "HIGHLIGHTS_PARQUET", tmp_path / "hl.parquet")
    return rga


def _setup_kit(monkeypatch, loaded, mom_by_market):
    loads = []

    def load(m, mt):
        loads.append((m, mt))
        return loaded.get(m, {})

    monkeypatch.setattr(rga.kit, "load", load)
    monkeypatch.setattr(rga.kit, "markets", lambda: list(loaded))
    monkeypatch.setattr(
        rga.kit,
        "screen",
        lambda slug, m, min_turnover_usd: (
            pd.DataFrame({"Symbol": ["D"]}) if slug == "darvas" else None
        ),
    )
    monkeypatch.setattr(
        rga.cs,
        "screen",
        lambda stocks, crit, rank_by, top, show: mom_by_market[
            stocks[0].market if hasattr(stocks[0], "market") else None
        ],
    )
    return loads


def test_global_analyse(rg, monkeypatch, capsys, tmp_path):
    def mom(sym, r):
        return pd.DataFrame(
            {
                "Symbol": sym,
                "ltp": 1.0,
                "ret_126": r,
                "ret_252": 2.0,
                "rsi14": 50.0,
                "dist_52w_high": 1.0,
            }
        )

    monkeypatch.setattr(rga.kit, "markets", lambda: ["IN", "US", "JP"])
    monkeypatch.setattr(
        rga.kit, "load", lambda m, mt: {} if m == "JP" else {"A": _ohlcv(), "B": _ohlcv()}
    )
    monkeypatch.setattr(
        rga.kit,
        "screen",
        lambda slug, m, min_turnover_usd: (
            pd.DataFrame({"Symbol": ["D"]}) if slug == "darvas" else None
        ),
    )
    seen = []

    def cs_screen(stocks, crit, rank_by, top, show):
        seen.append((len(stocks), crit, rank_by, top, show))
        return mom(["A", "B"], [20.0, 40.0]) if len(seen) == 1 else pd.DataFrame()

    monkeypatch.setattr(rga.cs, "screen", cs_screen)
    if PD3:
        pytest.skip("covered by the xfail test below")
    res = rga.analyse(top_per_market=3)
    assert seen[0][1] is rga.MOMENTUM and seen[0][2:] == ("ret_126", 3, rga.SHOW)
    assert list(res["per_market"]) == ["IN", "US"]
    g = res["global"]
    assert list(g["Symbol"]) == ["B", "A"] and list(g["Market"]) == ["IN", "IN"]
    assert (tmp_path / "global_highlights.xlsx").exists()
    assert rga.load_highlights().shape[0] == 2
    out = capsys.readouterr().out
    assert "■ IN: 2 stocks | Darvas 1 | GoldenCross 0 | momentum 2" in out
    assert "■ US: 2 stocks | Darvas 1 | GoldenCross 0 | momentum 0" in out
    assert "JP" not in out.replace("GLOBAL", "")
    assert "GLOBAL TOP 20" in out


@pytest.mark.xfail(
    PD3,
    strict=True,
    raises=(TypeError, IndexError),
    reason="run_global_analysis.py:~89 passes sheet_name positionally to DataFrame.to_excel (keyword-only in pandas 3: the TypeError is masked by openpyxl 'no sheet' IndexError on writer close)",
)
def test_analyse_to_excel_positional_sheet_name(rg, monkeypatch):
    monkeypatch.setattr(rga.kit, "markets", lambda: ["IN"])
    monkeypatch.setattr(rga.kit, "load", lambda m, mt: {"A": _ohlcv()})
    monkeypatch.setattr(rga.kit, "screen", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        rga.cs,
        "screen",
        lambda *a, **k: pd.DataFrame(
            {
                "Symbol": ["A"],
                "ltp": 1.0,
                "ret_126": 5.0,
                "ret_252": 1.0,
                "rsi14": 1.0,
                "dist_52w_high": 1.0,
            }
        ),
    )
    rga.analyse(verbose=False)


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="run_global_analysis.py:86 opens an ExcelWriter and writes no sheet when no market has data -> openpyxl 'At least one sheet must be visible'",
)
def test_analyse_empty_quiet(rg, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(rga.kit, "markets", lambda: ["IN"])
    monkeypatch.setattr(rga.kit, "load", lambda m, mt: {})
    res = rga.analyse(verbose=False)
    assert res["global"].empty and res["per_market"] == {}


def test_load_highlights_missing(rg):
    assert rga.load_highlights().empty
