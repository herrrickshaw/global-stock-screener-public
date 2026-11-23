import numpy as np
import pandas as pd
import pytest

import reward_screener_picks as P
from tests.r2b_support import install_warehouse


def _series(n=400, g=1.001, v=100.0, sym="AAA", start="2023-01-02"):
    dates = pd.bdate_range(start, periods=n)
    close = 10.0 * g ** np.arange(n)
    return pd.DataFrame(
        {"Date": dates, "Close": close, "High": close, "Volume": np.full(n, v), "Symbol": sym}
    )


# ---- _prices / _clean ------------------------------------------------------------
def test_prices_local(tmp_path, monkeypatch):
    d = tmp_path / "cache_seed" / "ltm"
    d.mkdir(parents=True)
    df = _series()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.assign(Extra=1).to_parquet(d / "US.parquet")
    monkeypatch.setattr(P, "BASE", str(tmp_path))
    out = P._prices("US", "local")
    assert list(out.columns) == ["Date", "High", "Close", "Volume", "Symbol"]
    assert pd.api.types.is_datetime64_any_dtype(out["Date"])


def test_prices_warehouse(monkeypatch):
    calls = []
    install_warehouse(monkeypatch, prices={"CA": _series()}, calls=calls)
    assert len(P._prices("CA", "warehouse")) == 400
    assert calls == [("prices", "CA")]


def test_clean_drops_bad_rows():
    g = pd.DataFrame(
        {
            "Close": [1.0, np.nan, 0.0, -2.0, 3.0, "x"],
            "Volume": [10.0, 5.0, 5.0, 5.0, np.nan, 5.0],
        }
    )
    c, v = P._clean(g)
    assert c.tolist() == [1.0, 3.0]
    assert v.tolist() == [10.0, 0.0]


# ---- _latest_factor ----------------------------------------------------------------
@pytest.fixture
def fake_px(monkeypatch):
    frames = {}

    def go(df):
        monkeypatch.setattr(P, "_prices", lambda m, s: df)

    return go


def test_latest_mom(fake_px):
    s = _series(n=400)
    fake_px(pd.concat([s, _series(n=272, sym="SHORT")]))
    df = P._latest_factor("CA", "warehouse", "mom_12_1")
    assert list(df["symbol"]) == ["AAA"]
    c = s["Close"].to_numpy()
    r = df.iloc[0]
    assert r["factor"] == pytest.approx(c[-21] / c[-273] - 1)
    assert r["ret_6m"] == pytest.approx(c[-1] / c[-126] - 1)
    assert r["last"] == pytest.approx(c[-1])
    assert r["dvol"] == pytest.approx(np.median(c[-60:] * 100.0))


def test_latest_min_history_273(fake_px):
    fake_px(_series(n=273))
    assert len(P._latest_factor("CA", "warehouse", "mom_12_1")) == 1
    fake_px(_series(n=272))
    assert P._latest_factor("CA", "warehouse", "mom_12_1").empty


def test_latest_above_200(fake_px):
    s = _series()
    fake_px(s)
    c = s["Close"].to_numpy()
    r = P._latest_factor("IN", "warehouse", "above_200dma").iloc[0]
    assert r["factor"] == pytest.approx(c[-1] / np.mean(c[-200:-1]) - 1)


def test_latest_vol(fake_px):
    rng = np.random.default_rng(2)
    n = 400
    close = 10 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    s = pd.DataFrame(
        {
            "Date": pd.bdate_range("2023-01-02", periods=n),
            "Close": close,
            "High": close,
            "Volume": 1.0,
            "Symbol": "V",
        }
    )
    fake_px(s)
    r = P._latest_factor("TW", "warehouse", "vol_60").iloc[0]
    exp = np.std(np.diff(close[-61:]) / close[-61:-1])
    assert r["factor"] == pytest.approx(exp)


def test_latest_unknown_factor_all_nan(fake_px):
    fake_px(_series())
    assert P._latest_factor("CA", "warehouse", "bogus").empty


# ---- _us_fund ---------------------------------------------------------------------
@pytest.fixture
def us_env(tmp_path, monkeypatch):
    d = tmp_path / "cache_seed" / "fundamentals_history"
    d.mkdir(parents=True)
    fund = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BIG", "TINY", "NOPX", "SHORTH"],
            "filed": [
                "2020-01-01",
                "2021-01-01",
                "2021-01-01",
                "2021-01-01",
                "2021-01-01",
                "2021-01-01",
            ],
            "equity": [10.0, 20.0, 30.0, 5.0, 5.0, 5.0],
            "shares": [1e6, 2e6, 3e6, 1e3, 1e6, 1e8],
            "cfo": [1.0, 4.0, 3.0, 1.0, 1.0, 1.0],
            "total_assets": [100.0, 200.0, 100.0, 10.0, 10.0, 10.0],
            "net_income": [2.0, 8.0, 6.0, 1.0, 1.0, 1.0],
        }
    )
    fund.to_parquet(d / "US.parquet")
    (tmp_path / "cache_seed" / "ltm").mkdir()
    px = pd.concat(
        [
            _series(sym="AAA", g=1.001),  # last ~ 14.9 ; mktcap 2e6*14.9 = 3e7 < 1e8!
            _series(sym="BIG", g=1.001).assign(Close=lambda d: d["Close"] * 20),
            _series(sym="TINY"),
            _series(sym="SHORTH", n=125),
        ]
    )
    monkeypatch.setattr(P, "BASE", str(tmp_path))
    px["Date"] = px["Date"].dt.strftime("%Y-%m-%d")
    px.to_parquet(tmp_path / "cache_seed" / "ltm" / "US.parquet")
    return px


def test_us_fund_cfo_ta_and_filters(us_env):
    out = P._us_fund("cfo_ta")
    # AAA: latest filing used (cfo 4 / TA 200), but mktcap = 2e6*~14.9 = 3e7 < 1e8 -> dropped
    # BIG: price ~ 298 * 3e6 shares = 8.9e8 -> kept; TINY mktcap tiny; NOPX has no prices;
    # SHORTH has <126 closes
    assert list(out["symbol"]) == ["BIG"]
    assert out.iloc[0]["factor"] == pytest.approx(3.0 / 100.0)
    assert list(out.columns) == ["symbol", "factor", "dvol", "last", "ret_6m"]


@pytest.mark.parametrize(
    "factor,expected",
    [
        ("cfo_ta", 0.03),
        ("roa", 0.06),
    ],
)
def test_us_fund_factors(us_env, factor, expected):
    assert P._us_fund(factor).iloc[0]["factor"] == pytest.approx(expected)


def test_us_fund_btm_ey_and_unknown(us_env):
    big = P._us_fund("book_to_market").iloc[0]
    cap = 3e6 * big["last"]
    assert big["factor"] == pytest.approx(30.0 / cap)
    ey = P._us_fund("earnings_yield").iloc[0]
    assert ey["factor"] == pytest.approx(6.0 / cap)
    assert P._us_fund("bogus").empty


def test_us_fund_market_cap_boundary(us_env, tmp_path):
    # AAA latest filing -> shares 2e6; raise to exactly pass the $100M floor
    f = pd.read_parquet(tmp_path / "cache_seed" / "fundamentals_history" / "US.parquet")
    f.loc[1, "shares"] = 1e9
    f.to_parquet(tmp_path / "cache_seed" / "fundamentals_history" / "US.parquet")
    assert set(P._us_fund("cfo_ta")["symbol"]) == {"AAA", "BIG"}


def test_us_fund_inf_factor_dropped(us_env, tmp_path):
    f = pd.read_parquet(tmp_path / "cache_seed" / "fundamentals_history" / "US.parquet")
    f.loc[2, "total_assets"] = 0.0
    f.to_parquet(tmp_path / "cache_seed" / "fundamentals_history" / "US.parquet")
    assert P._us_fund("cfo_ta").empty


# ---- screen -------------------------------------------------------------------------
def _cross(n=100, last=50.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "symbol": [f"T{i}" for i in range(n)],
            "factor": np.linspace(0.0, 1.0, n),
            "dvol": np.arange(1, n + 1, dtype=float),
            "last": np.full(n, last),
            "ret_6m": rng.random(n),
        }
    )


def test_screen_top_side(monkeypatch):
    df = _cross()
    monkeypatch.setattr(P, "_latest_factor", lambda m, s, f: df)
    label, picks = P.screen("CA", top=5)  # mom_12_1 cap [0,3], top 40%, liquid 40%
    assert label == "12-mo momentum"
    liquid = df[df["dvol"] >= df["dvol"].quantile(0.6)]
    thr = liquid["factor"].quantile(0.6)
    exp = liquid[liquid["factor"] >= thr].sort_values("dvol", ascending=False).head(5)
    assert picks["symbol"].tolist() == exp["symbol"].tolist()
    assert len(picks) == 5


def test_screen_bottom_side(monkeypatch):
    df = _cross().assign(factor=lambda d: 0.01 + d["factor"] * 0.1)
    monkeypatch.setattr(P, "_latest_factor", lambda m, s, f: df)
    label, picks = P.screen("TW", top=100)  # vol_60 bottom 40%, min price 10
    assert label == "low volatility"
    liquid = df[df["dvol"] >= df["dvol"].quantile(0.6)]
    thr = liquid["factor"].quantile(0.4)
    assert set(picks["symbol"]) == set(liquid[liquid["factor"] <= thr]["symbol"])
    assert picks["dvol"].is_monotonic_decreasing


def test_screen_caps_inclusive_and_min_price(monkeypatch):
    df = pd.DataFrame(
        {
            "symbol": list("abcdef"),
            "factor": [0.0, 3.0, 3.0001, -0.0001, 1.0, 1.0],
            "dvol": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "last": [5.0, 5.0, 5.0, 5.0, 1.99, 2.0],
            "ret_6m": 0.1,
        }
    )
    monkeypatch.setattr(P, "_latest_factor", lambda m, s, f: df)
    # keep=1.0-ish: patch ROBUST to keep everything
    monkeypatch.setitem(P.ROBUST, "CA", ("m", "mom_12_1", "top", 1.0, "warehouse"))
    _, picks = P.screen("CA", 10)
    assert sorted(picks["symbol"]) == ["a", "b", "f"]  # e excluded: price 1.99 < 2.0


def test_screen_empty_branches(monkeypatch):
    monkeypatch.setattr(P, "_latest_factor", lambda m, s, f: _cross().iloc[0:0])
    label, picks = P.screen("CA", 5)
    assert label == "12-mo momentum" and picks.empty
    out_of_cap = _cross().assign(factor=9.0)
    monkeypatch.setattr(P, "_latest_factor", lambda m, s, f: out_of_cap)
    assert P.screen("CA", 5)[1].empty


def test_screen_us_uses_fund(monkeypatch):
    seen = []
    monkeypatch.setattr(
        P, "_us_fund", lambda f: seen.append(f) or _cross(last=10.0).assign(factor=0.3)
    )
    monkeypatch.setattr(P, "_latest_factor", lambda *a: pytest.fail("not for US"))
    label, picks = P.screen("US", 3)
    assert seen == ["cfo_ta"] and label == "cash-flow quality"
    assert len(picks) == 3


def test_screen_uses_robust_source(monkeypatch):
    seen = []
    monkeypatch.setattr(
        P, "_latest_factor", lambda m, s, f: seen.append((m, s, f)) or _cross().iloc[0:0]
    )
    P.screen("IN", 5)
    assert seen == [("IN", "warehouse", "above_200dma")]


# ---- main -----------------------------------------------------------------------------
def test_main(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(P, "BASE", str(tmp_path))
    top = _cross(n=3, last=12.3456).assign(factor=[0.1234, 0.2, 0.3])
    calls = []

    def fake_screen(m, t):
        calls.append((m, t))
        return P.ROBUST[m][0], (top if m != "CN" else top.iloc[0:0])

    monkeypatch.setattr(P, "screen", fake_screen)
    monkeypatch.setattr("sys.argv", ["x", "--top", "7"])
    P.main()
    out = capsys.readouterr().out
    assert calls == [(m, 7) for m in P.ROBUST]
    assert "US — cash-flow quality  [cfo_ta]" in out and "CFO/TA" in out
    assert "TW — low volatility  [vol_60]" in out and "  vol" in out
    assert "IN — uptrend (>200-DMA)  [above_200dma]" in out
    assert "12.35" in out and "0.123" in out
    assert "12 names written → reports/reward_picks.csv" in out
    df = pd.read_csv(tmp_path / "reports" / "reward_picks.csv")
    assert len(df) == 12 and set(df["market"]) == {"US", "CA", "TW", "IN"}
    assert list(df.columns) == [
        "market",
        "screen",
        "factor_name",
        "ticker",
        "factor_value",
        "ret_6m",
        "last_price",
    ]
    r = df[df["market"] == "US"].iloc[0]
    assert r["factor_value"] == pytest.approx(0.1234) and r["last_price"] == pytest.approx(12.35)


def test_main_book_to_market_format(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(P, "BASE", str(tmp_path))
    monkeypatch.setitem(P.ROBUST, "US", ("v", "book_to_market", "top", 0.2, "local"))
    monkeypatch.setattr(P, "ROBUST", {"US": P.ROBUST["US"]})
    monkeypatch.setattr(P, "screen", lambda m, t: ("v", _cross(n=1).assign(factor=1.23456)))
    monkeypatch.setattr("sys.argv", ["x"])
    P.main()
    out = capsys.readouterr().out
    assert "b/m" in out and "1.23 " in out and "1.235" not in out
