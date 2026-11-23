import numpy as np
import pandas as pd
import pytest

import reward_liquidity_check as L
import reward_overlay as O
import reward_universe_check as U
from tests.r2b_support import make_prices


@pytest.fixture(scope="module")
def px():
    return make_prices(n_sym=300)


@pytest.fixture(scope="module")
def merged(px):
    orig = U._prices
    U._prices = lambda m: px[["Date", "Close", "Volume", "Symbol"]]
    try:
        return O._merged("XX")
    finally:
        U._prices = orig


def test_merged(merged):
    assert {"mom_12_1", "liq_6_6", "vol_surge", "fwd", "dvol"} <= set(merged.columns)
    assert not merged.duplicated(["symbol", "year"]).any()


def test_merged_empty_branches(monkeypatch):
    empty = pd.DataFrame()
    full = pd.DataFrame({"symbol": ["a"], "year": [1]})
    monkeypatch.setattr(U, "_panel", lambda m: empty)
    monkeypatch.setattr(L, "_panel", lambda m: full)
    assert O._merged("X") is empty
    monkeypatch.setattr(U, "_panel", lambda m: full)
    monkeypatch.setattr(L, "_panel", lambda m: empty)
    assert O._merged("X") is full


def test_price_mask_directions():
    sub = pd.DataFrame({"x": np.arange(10.0)})
    assert O._price_mask(sub, ("x", +1, 0.3)).tolist() == [False] * 7 + [True] * 3
    assert O._price_mask(sub, ("x", -1, 0.3)).tolist() == [True] * 3 + [False] * 7


def test_price_screen(merged):
    s = O._price_screen(merged)
    assert s is not None and s[0] in U.FACTORS and s[1] in (1, -1)
    assert O._price_screen(merged.iloc[:10]) is None


def test_alpha_rows_and_guards():
    rng = np.random.default_rng(1)
    rows = []
    for y in (2022, 2023):
        for k in range(40):
            rows.append({"year": y, "x": float(k), "liq_6_6": float(rng.random()), "fwd": 0.01 * k})
    df = pd.DataFrame(rows)
    a = O._alpha(df, ("x", +1, 0.5), [2022, 2023])
    assert len(a) == 2 and list(a.columns) == ["price", "rise", "fall", "n_p", "n_r", "n_f"]
    assert (a["price"] > 0).all() and (a["n_p"] == 20).all()
    assert (a["n_r"] + a["n_f"] == a["n_p"]).all()
    # year with <20 rows is skipped
    assert len(O._alpha(df.iloc[:59], ("x", +1, 0.5), [2022, 2023])) == 1
    # portfolio smaller than MIN_NAMES -> skipped (keep 0.1 of 40 = 4)
    assert O._alpha(df, ("x", +1, 0.1), [2022]).empty
    # keep 0.125 of 40 = 5 names: price ok, but rise/fall each < 5 -> NaN
    a = O._alpha(df, ("x", +1, 0.125), [2022])
    assert a.loc[0, "n_p"] == 5
    assert np.isnan(a.loc[0, "rise"]) and np.isnan(a.loc[0, "fall"])
    assert a.loc[0, "n_r"] + a.loc[0, "n_f"] == 5


def test_alpha_nan_factor_excluded():
    df = pd.DataFrame(
        {
            "year": 2022,
            "x": [np.nan] * 10 + list(np.arange(30.0)),
            "liq_6_6": np.linspace(0, 1, 40),
            "fwd": np.linspace(0, 0.4, 40),
        }
    )
    a = O._alpha(df, ("x", -1, 0.5), [2022])
    assert a.loc[0, "n_p"] == 15  # NaN names never pass


def test_run_ok(monkeypatch, merged):
    monkeypatch.setattr(O, "_merged", lambda m: merged)
    r = O.run("JP")
    assert r["market"] == "JP" and r["screen"][-1] == "%"
    assert {"price", "rise", "fall", "n_p", "n_r"} <= set(r)
    assert r["n_p"] > 0


def test_run_empty_and_no_screen(monkeypatch, merged):
    monkeypatch.setattr(O, "_merged", lambda m: pd.DataFrame())
    assert O.run("JP") is None
    monkeypatch.setattr(O, "_merged", lambda m: merged)
    monkeypatch.setattr(O, "_price_screen", lambda df: None)
    assert O.run("JP") == {"market": "JP", "screen": None}


def test_run_no_oos_years(monkeypatch, merged):
    early = merged[merged["year"] < 2022]
    monkeypatch.setattr(O, "_merged", lambda m: early)
    monkeypatch.setattr(O, "_price_screen", lambda df: ("vol_60", -1, 0.3))
    assert O.run("JP") == {"market": "JP", "screen": None}


def test_run_label_format(monkeypatch, merged):
    monkeypatch.setattr(O, "_merged", lambda m: merged)
    monkeypatch.setattr(O, "_price_screen", lambda df: ("vol_60", -1, 0.3))
    assert O.run("JP")["screen"] == "vol_60↓30%"
    monkeypatch.setattr(O, "_price_screen", lambda df: ("mom_12_1", 1, 0.2))
    assert O.run("JP")["screen"] == "mom_12_1↑20%"


# ---- _current / overlay_picks -----------------------------------------------------
def test_current_values_single(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=400)
    close = 10.0 * 1.001 ** np.arange(400)
    v = np.full(400, 100.0)
    v[-126:] = 200.0
    frame = pd.DataFrame({"Date": dates, "Close": close, "Volume": v, "Symbol": "AAA"})
    short = frame.iloc[:272].copy()
    short["Symbol"] = "SHORT"
    monkeypatch.setattr(U, "_prices", lambda m: pd.concat([frame, short]))
    df = O._current("X")
    assert list(df["ticker"]) == ["AAA"]
    r = df.iloc[0]
    assert r["liq_6_6"] == pytest.approx(
        (close[-126:] * 200).mean() / (close[-252:-126] * 100).mean() - 1
    )
    assert r["dist_52w_high"] == pytest.approx(0.0)
    assert r["last"] == pytest.approx(close[-1])
    assert r["ret_6m"] == pytest.approx(close[-1] / close[-126] - 1)
    assert r["dvol"] == pytest.approx(np.median(close[-60:] * 200))
    assert r["vol_60"] == pytest.approx(0.0, abs=1e-6)


def test_current_zero_prior_volume_nan(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=400)
    close = 10.0 * 1.001 ** np.arange(400)
    v = np.zeros(400)
    v[-126:] = 5.0
    frame = pd.DataFrame({"Date": dates, "Close": close, "Volume": v, "Symbol": "AAA"})
    monkeypatch.setattr(U, "_prices", lambda m: frame)
    assert np.isnan(O._current("X").loc[0, "liq_6_6"])


def _fake_current(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(n)],
            "vol_60": rng.random(n) * 0.05,
            "dist_52w_high": -rng.random(n),
            "liq_6_6": rng.standard_normal(n),
            "dvol": np.arange(1, n + 1) * 1000.0,
            "last": np.full(n, 500.0),
            "ret_6m": rng.standard_normal(n) * 0.1,
        }
    )


def test_overlay_picks_logic(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "BASE", str(tmp_path))
    frames = {m: _fake_current(seed=i) for i, m in enumerate(O.OVERLAY_PICKS)}
    monkeypatch.setattr(O, "_current", lambda m: frames[m])
    out, path = O.overlay_picks(top=3)
    assert path == str(tmp_path / "reports" / "reward_overlay_picks.csv")
    df = pd.DataFrame(out)
    assert set(df["market"]) <= set(O.OVERLAY_PICKS)
    assert df.groupby("market").size().max() <= 3
    # replicate HK (low-vol bottom 10%, rising)
    f = frames["HK"]
    f = f[f["last"] >= 1.0]
    f = f[f["dvol"] >= f["dvol"].quantile(0.5)]
    thr = f["vol_60"].quantile(0.10)
    exp = f[(f["vol_60"] <= thr) & (f["liq_6_6"] >= f["liq_6_6"].median())]
    exp = exp.sort_values("dvol", ascending=False).head(3)["ticker"].tolist()
    assert df[df["market"] == "HK"]["ticker"].tolist() == exp
    # CA is "falling": every pick has liq below the market median (strict <)
    f = frames["CA"]
    f = f[f["dvol"] >= f["dvol"].quantile(0.5)]
    ca = df[df["market"] == "CA"]
    med = f["liq_6_6"].median()
    for t in ca["ticker"]:
        assert f[f["ticker"] == t]["liq_6_6"].iloc[0] < med
    # SE is a "top" breakout screen
    se = df[df["market"] == "SE"]
    f = frames["SE"]
    f = f[f["last"] >= 10]
    f = f[f["dvol"] >= f["dvol"].quantile(0.5)]
    thr = f["dist_52w_high"].quantile(0.7)
    for t in se["ticker"]:
        assert f[f["ticker"] == t]["dist_52w_high"].iloc[0] >= thr
    assert pd.read_csv(path).shape[0] == len(out)
    assert set(df.columns) == {"market", "screen", "ticker", "liq_6_6", "ret_6m", "last_price"}


def test_overlay_picks_min_price_and_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "BASE", str(tmp_path))
    cheap = _fake_current()
    cheap["last"] = 0.5  # below every MIN_PRICE except none (SG=0.2 is below 0.5)

    def fake(m):
        if m == "HK":
            return _fake_current().iloc[0:0]
        if m == "UK":
            return cheap.assign(last=49.99)  # UK min 50
        if m == "JP":
            return cheap.assign(last=100.0, liq_6_6=1.0)  # exactly the minimum -> kept
        return _fake_current().assign(liq_6_6=np.nan)  # all dropped by dropna

    monkeypatch.setattr(O, "_current", fake)
    out, _ = O.overlay_picks()
    assert {r["market"] for r in out} == {"JP"}


def test_overlay_picks_no_rows_writes_empty_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(O, "BASE", str(tmp_path))
    monkeypatch.setattr(O, "_current", lambda m: _fake_current().iloc[0:0])
    out, path = O.overlay_picks()
    assert out == []
    assert (tmp_path / "reports" / "reward_overlay_picks.csv").exists()


# ---- main --------------------------------------------------------------------------
def _run_result(mkt, price, rise, fall):
    return {
        "market": mkt,
        "screen": "vol_60↓10%",
        "price": price,
        "rise": rise,
        "fall": fall,
        "n_p": 10,
        "n_r": 5,
    }


@pytest.fixture
def main_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(O, "BASE", str(tmp_path))

    def go(results, argv=()):
        monkeypatch.setattr(O, "run", lambda m: results.get(m))
        monkeypatch.setattr("sys.argv", ["x", *argv])
        O.main()
        return capsys.readouterr().out

    return go


def test_main_verdicts(main_run, tmp_path):
    res = {
        "US": _run_result("US", 0.01, 0.03, 0.0),  # +2.0pp via rising -> helps
        "JP": _run_result("JP", 0.01, 0.012, 0.0),  # +0.2pp neutral
        "HK": _run_result(
            "HK", 0.05, 0.0, 0.03
        ),  # fall -> best is fall 0.03? max(rise 0, fall .03)
        "TW": _run_result("TW", 0.05, -0.02, -0.03),  # -> hurts (best rising -7pp)
        "CA": {"market": "CA", "screen": None},
        "AU": None,
        "UK": _run_result("UK", 0.01, float("nan"), 0.02),
    }
    out = main_run(res)
    assert "(no robust price screen)" in out
    lines = {
        ln.split()[0]: ln
        for ln in out.splitlines()
        if ln.strip()[:2] in ("US", "JP", "HK", "TW", "UK")
    }
    assert "rising +2.0pp  ✅ helps" in lines["US"]
    assert "rising +0.2pp  – neutral" in lines["JP"]
    assert "falling -2.0pp  ✗ hurts" in lines["HK"]
    assert "rising -7.0pp  ✗ hurts" in lines["TW"]
    assert "falling +1.0pp  ✅ helps" in lines["UK"]
    assert "overlay IMPROVES the price screen OOS in 2/5: US, UK" in out
    csv = pd.read_csv(tmp_path / "reports" / "reward_overlay.csv")
    assert list(csv["market"]) == ["US", "JP", "HK", "TW", "UK"]
    assert csv.loc[0, "best_overlay"] == "rising" and csv.loc[0, "delta"] == pytest.approx(0.02)
    assert "written → reports/reward_overlay.csv" in out
    assert "liquidity-confirmed picks" not in out


@pytest.mark.parametrize(
    "rise,flag",
    [(0.0151, "✅ helps"), (0.015, "– neutral"), (0.0049 + 0.01 - 0.0098, "– neutral")],
)
def test_main_threshold_boundaries(main_run, rise, flag):
    out = main_run({"US": _run_result("US", 0.01, rise, -1.0)}, [])
    line = [ln for ln in out.splitlines() if ln.startswith(" US ")][0]
    assert flag in line


def test_main_hurts_boundary(main_run):
    # delta exactly -0.005 -> not "> -0.005" -> hurts
    out = main_run({"US": _run_result("US", 0.02, 0.015, -1.0)})
    assert "✗ hurts" in out
    out = main_run({"US": _run_result("US", 0.02, 0.0151, -1.0)})
    assert "– neutral" in out


def test_main_picks_flag(main_run, monkeypatch, tmp_path):
    monkeypatch.setattr(
        O, "overlay_picks", lambda top: ([{"a": 1}] * 2, str(tmp_path / "reports" / "x.csv"))
    )
    out = main_run({}, ["--picks", "--top", "4"])
    assert "2 liquidity-confirmed picks → reports/x.csv" in out
    assert "overlay IMPROVES the price screen OOS in 0/0: none" in out
