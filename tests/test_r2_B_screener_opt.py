import numpy as np
import pandas as pd
import pytest

import reward_screener_opt as R
from tests.r2b_support import install_warehouse, make_fundamentals, make_prices


@pytest.fixture(scope="module")
def px():
    return make_prices(n_sym=200)


@pytest.fixture(scope="module")
def funds(px):
    return make_fundamentals(px)


@pytest.fixture(scope="module")
def built(funds, px):
    df = R.build_factors(funds)
    return R.attach_forward_and_price(df, px, 60)


def _two_year():
    return pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "fy_end": pd.to_datetime(["2020-12-31", "2021-12-31"]),
            "total_assets": [100.0, 120.0],
            "net_income": [5.0, 10.0],
            "revenue": [50.0, 90.0],
            "equity": [40.0, 50.0],
            "current_assets": [30.0, 60.0],
            "current_liabilities": [20.0, 20.0],
            "long_term_debt": [30.0, 20.0],
            "shares": [10.0, 10.0],
            "gross_profit": [10.0, 30.0],
            "cfo": [6.0, 15.0],
        }
    )


# ---- _load ------------------------------------------------------------------
def test_load_local_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "BASE", str(tmp_path))
    with pytest.raises(SystemExit, match="missing fundamentals"):
        R._load("US", None)
    fd = tmp_path / "fh"
    fd.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(fd / "US.parquet")
    with pytest.raises(SystemExit, match="missing prices"):
        R._load("US", str(fd))


def test_load_local_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "BASE", str(tmp_path))
    d = tmp_path / "cache_seed" / "fundamentals_history"
    d.mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_parquet(d / "US.parquet")
    (tmp_path / "cache_seed" / "ltm").mkdir()
    (tmp_path / "cache_seed" / "ltm" / "US.parquet").write_bytes(b"x")
    hist, prices = R._load("US", None)
    assert list(hist["a"]) == [1]
    assert prices == str(tmp_path / "cache_seed" / "ltm" / "US.parquet")


def test_load_warehouse(monkeypatch, funds, px):
    calls = []
    install_warehouse(monkeypatch, prices={"JP": px}, funds={"JP": funds}, calls=calls)
    h, p = R._load("JP", None, "warehouse")
    assert len(h) == len(funds) and isinstance(p, pd.DataFrame)
    with pytest.raises(SystemExit, match="no warehouse fundamentals for KR"):
        R._load("KR", None, "warehouse")


# ---- factors ----------------------------------------------------------------
def test_prev_groups_by_ticker():
    df = pd.DataFrame({"ticker": ["A", "A", "B"], "x": [1.0, 2.0, 3.0]})
    assert R._prev(df, "x").tolist()[1] == 1.0
    assert np.isnan(R._prev(df, "x").iloc[2])


def test_fscore_all_nine_and_first_year_nan():
    f = R._fscore(_two_year())
    assert np.isnan(f[0])
    assert f[1] == 9.0


def test_fscore_partial():
    df = _two_year()
    df.loc[1, "net_income"] = -1.0  # roa<0, roa<roa1, cfo/ta(0.125) > roa(-0.0083) still
    df.loc[1, "long_term_debt"] = 40.0  # leverage up
    f = R._fscore(df)[1]
    # lost: roa>0, roa>roa1, lev<lev1 -> 6
    assert f == 6.0


def test_build_factors_full(funds):
    df = R.build_factors(funds)
    d = df.attrs["dirs"]
    assert d["accruals"] == -1 and d["leverage"] == -1 and d["roa"] == 1
    assert {"gross_margin", "cfo_ta", "current_ratio", "ebit_margin", "fscore"} <= set(d)
    assert df.attrs["has_ebit"] and df.attrs["has_ltd"]
    assert df["ticker"].is_monotonic_increasing
    r = df.iloc[3]
    assert r["roa"] == pytest.approx(r["net_income"] / r["total_assets"])
    assert r["accruals"] == pytest.approx((r["net_income"] - r["cfo"]) / r["total_assets"])
    assert r["roe"] == pytest.approx(r["net_income"] / r["equity"])


@pytest.mark.xfail(
    strict=True,
    raises=(TypeError, KeyError),
    reason="source bug: build_factors guards optional columns with `is not None` but _fscore "
    "(reward_screener_opt.py:79-103) dereferences them unconditionally, so a frame lacking "
    "any of these optional inputs crashes instead of yielding NaN factors",
)
@pytest.mark.parametrize("col", ["gross_profit", "cfo", "current_assets", "long_term_debt"])
def test_build_factors_tolerates_missing_optional_column(col):
    out = R.build_factors(_two_year().drop(columns=[col]))
    assert len(out) == 2


def test_build_factors_inf_and_nonpositive_denominators():
    df = _two_year()
    df.loc[0, "equity"] = -5.0
    df.loc[0, "revenue"] = 0.0
    out = R.build_factors(df)
    assert np.isnan(out.loc[0, "roe"]) and np.isnan(out.loc[0, "net_margin"])
    assert np.isinf(out[["roa", "asset_turn"]].to_numpy()).sum() == 0


# ---- attach_forward_and_price --------------------------------------------------
def test_attach_columns(built):
    for c in [
        "fwd",
        "fwd_w",
        "ey",
        "excess",
        "excess_w",
        "earnings_yield",
        "book_to_market",
        "mf_earn_yield",
        "mf_roc",
        "mom_12_1",
        "reversal_1m",
        "dist_52w_high",
        "above_200dma",
        "vol_60",
    ]:
        assert c in built.columns
    assert built["fwd"].notna().all()
    d = built.attrs["dirs"]
    assert d["vol_60"] == -1 and d["mom_12_1"] == 1 and d["reversal_1m"] == -1
    assert d["earnings_yield"] == 1 and d["mf_roc"] == 1
    assert np.allclose(built.groupby("ey")["excess"].median(), 0.0)
    lo, hi = built["fwd"].quantile([0.01, 0.99])
    assert built["fwd_w"].between(lo, hi).all()


def test_attach_values_single_ticker():
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    n = len(dates)
    close = 10.0 * 1.001 ** np.arange(n)
    px = pd.DataFrame({"Date": dates, "Close": close, "Symbol": "AAA"})
    fund = _two_year()
    fund["ticker"] = "aaa"  # lookups upper-case the ticker
    fund["filed"] = ["2018-03-31", "2019-03-31"]
    fund["ebit"] = [8.0, 12.0]
    df = R.attach_forward_and_price(R.build_factors(fund), px, 100)
    assert len(df) == 2
    r = df.iloc[0]
    i = int(np.searchsorted(dates.to_numpy(), np.datetime64("2018-04-01")))
    assert r["fwd"] == pytest.approx(close[i + 100] / close[i] - 1)
    assert r["_mktcap"] == pytest.approx(close[i] * 10.0)
    assert r["mom_12_1"] == pytest.approx(close[i - 21] / close[i - 273] - 1)
    assert r["reversal_1m"] == pytest.approx(close[i] / close[i - 21] - 1)
    assert r["dist_52w_high"] == pytest.approx(0.0)
    assert r["above_200dma"] == pytest.approx(close[i] / close[i - 200 : i].mean() - 1)
    assert r["earnings_yield"] == pytest.approx(5.0 / r["_mktcap"])
    assert r["book_to_market"] == pytest.approx(40.0 / r["_mktcap"])
    ev = r["_mktcap"] + 30.0
    assert r["mf_earn_yield"] == pytest.approx(8.0 / ev)
    assert r["mf_roc"] == pytest.approx(8.0 / (100.0 - 20.0))
    assert r["ey"] == 2018


def test_attach_skips_unusable_rows(funds, px):
    f = funds[funds["ticker"].isin(["S100", "S101"])].copy()
    f = f.reset_index(drop=True)
    f.loc[0, "filed"] = "9999-99-99"
    f.loc[1, "filed"] = "nan"
    f.loc[2, "filed"] = None
    f.loc[3, "filed"] = "2030-01-01"  # entry beyond price history
    f.loc[4, "ticker"] = "NOPE"
    out = R.attach_forward_and_price(R.build_factors(f), px, 60)
    assert len(out) == len(f) - 5
    assert "NOPE" not in set(out["ticker"])


def test_attach_parquet_path(tmp_path, funds, px):
    p = tmp_path / "px.parquet"
    px.to_parquet(p)
    f = funds[funds["ticker"] == "S050"]
    out = R.attach_forward_and_price(R.build_factors(f), str(p), 60)
    assert len(out) > 0


def test_attach_missing_shares_gives_nan_mktcap(funds, px):
    f = funds[funds["ticker"] == "S060"].copy()
    f["shares"] = np.nan
    out = R.attach_forward_and_price(R.build_factors(f), px, 60)
    assert out["_mktcap"].isna().all() and out["earnings_yield"].isna().all()


# ---- reward / candidates / mask ----------------------------------------------
def test_reward():
    assert R.reward(pd.DataFrame({"excess": [], "excess_w": []})) is None
    sub = pd.DataFrame({"excess": [0.1, -0.1, 0.0, 0.3], "excess_w": [1.0, 2.0, 3.0, 4.0]})
    r = R.reward(sub)
    assert r["n"] == 4 and r["med"] == pytest.approx(0.05)
    assert r["mean"] == pytest.approx(2.5) and r["win"] == 0.5


def test_candidates_orientation_and_min_sample():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"hi": rng.random(200), "lo": rng.random(200), "few": np.nan})
    train.loc[:148, "few"] = 1.0  # 149 non-null < N_MIN_TRAIN
    out = R.candidates(train, {"hi": +1, "lo": -1, "few": +1, "absent": +1})
    assert {c[0] for c in out} == {"hi", "lo"}
    hi = [c for c in out if c[0] == "hi"]
    assert [c[3] for c in hi] == R.KEEP_FRACS and all(c[1] == ">=" for c in hi)
    assert hi[0][2] == pytest.approx(train["hi"].quantile(0.5))
    assert hi[-1][2] == pytest.approx(train["hi"].quantile(0.9))
    lo = [c for c in out if c[0] == "lo"]
    assert all(c[1] == "<=" for c in lo)
    assert lo[-1][2] == pytest.approx(train["lo"].quantile(0.1))
    train2 = pd.DataFrame({"x": np.arange(150.0)})
    assert len(R.candidates(train2, {"x": 1})) == 5  # exactly N_MIN_TRAIN is allowed


def test_mask_of_and_evaluate():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, np.nan],
            "b": [5.0, 1.0, 1.0, 1.0],
            "excess": [0.1, 0.2, -0.1, 0.5],
            "excess_w": [0.1, 0.2, -0.1, 0.5],
        }
    )
    m = R.mask_of(df, [("a", ">=", 2.0, 0.5)])
    assert m.tolist() == [False, True, True, False]
    m = R.mask_of(df, [("a", "<=", 2.0, 0.5), ("b", "<=", 1.0, 0.5)])
    assert m.tolist() == [False, True, False, False]
    assert R.mask_of(df, []).all()
    assert R.evaluate(df, [("a", ">=", 2.0, 0.5)])["n"] == 2
    assert R.evaluate(df, [("a", ">=", 99.0, 0.5)]) is None


# ---- discover ---------------------------------------------------------------
@pytest.fixture(scope="module")
def result(funds, px):
    import sys

    mod = sys.modules.get("warehouse")
    import types

    w = types.ModuleType("warehouse")
    w.fundamentals = lambda m: funds.copy()
    w.prices = lambda m: px.copy()
    sys.modules["warehouse"] = w
    try:
        return R.discover("xx", "warehouse", 60, 2021, 3)
    finally:
        if mod is None:
            del sys.modules["warehouse"]
        else:
            sys.modules["warehouse"] = mod


def test_discover_structure(result):
    r = result
    assert r["market"] == "XX" and r["source"] == "warehouse"
    assert r["horizon"] == 60 and r["split"] == 2021
    assert r["n_train"] + r["n_test"] == r["n_panel"]
    assert r["tickers"] > 100
    assert 1 <= len(r["chosen"]) <= 3
    assert len({c[0] for c in r["chosen"]}) == len(r["chosen"])
    assert r["pool"] == sorted(r["pool"], key=lambda x: -x[2]["med"])
    assert all(p[2]["med"] > 0 for p in r["pool"])
    assert r["train"]["n"] >= 1 and r["per_yr"] > 0
    assert r["robust"] == bool(r["test"] and r["test"]["med"] > 0.005 and r["test"]["win"] >= 0.5)
    assert r["mid"] in range(2015, 2022)


def test_discover_max_filters_one(funds, px, monkeypatch):
    monkeypatch.setattr(R, "_load", lambda *a: (funds, px))
    r = R.discover("xx", "warehouse", 60, 2021, 1)
    assert len(r["chosen"]) == 1


def test_discover_no_pool(funds, px, monkeypatch):
    monkeypatch.setattr(R, "_load", lambda *a: (funds, px))
    monkeypatch.setattr(R, "N_MIN_TRAIN", 10**9)
    r = R.discover("xx", "warehouse", 60, 2021, 3)
    assert r["chosen"] == [] and r["pool"] == [] and r["per_yr"] == 0
    assert r["train"]["n"] == r["n_train"]  # empty filter list passes everyone
    assert r["robust"] in (True, False)


def test_discover_default_args(funds, px, monkeypatch):
    seen = {}

    def fake_load(m, fd, src):
        seen["a"] = (m, fd, src)
        return funds, px

    monkeypatch.setattr(R, "_load", fake_load)
    R.discover("us")
    assert seen["a"] == ("US", None, "local")


# ---- print_report -------------------------------------------------------------
def _res(**kw):
    base = {
        "market": "US",
        "horizon": 252,
        "split": 2021,
        "mid": 2018,
        "n_panel": 1000,
        "n_train": 500,
        "n_test": 200,
        "tickers": 90,
        "pool": [
            (
                ("roa", ">=", 0.05, 0.3),
                {"med": 0.04, "n": 5},
                {"med": 0.02, "n": 5},
                {"med": 0.01, "win": 0.55, "n": 40},
            ),
            (("vol_60", "<=", 0.02, 0.2), {"med": 0.03}, {"med": 0.01}, None),
        ],
        "chosen": [("roa", ">=", 0.05, 0.3), ("vol_60", "<=", 0.02, 0.2)],
        "train": {"med": 0.04, "mean": 0.05, "win": 0.6, "n": 100},
        "test": {"med": 0.02, "mean": 0.03, "win": 0.58, "n": 50},
        "full": {"med": 0.03, "mean": 0.04, "win": 0.59, "n": 150},
        "per_yr": 12.4,
        "robust": True,
    }
    base.update(kw)
    return base


def test_print_report_robust(capsys):
    R.print_report(_res())
    out = capsys.readouterr().out
    assert "REWARD-OPTIMISED SCREENER — US  (252d horizon)" in out
    assert "panel 1,000 obs | train(≤2021) 500 | test(>2021) 200 | 90 tickers" in out
    assert "thin panel" not in out
    assert "split@2018): 2" in out
    assert "★ DISCOVERED SCREENER (2 filters)" in out
    assert "(top 30%)" in out and "(bottom 20%)" in out
    assert "VERDICT: ✅ ROBUST" in out and "+2.0% median, 58% win" in out
    assert "avg names selected / year: 12" in out
    assert "nan%" in out  # missing OOS for vol_60 row


@pytest.mark.parametrize("ntr,nte,thin", [(399, 200, True), (400, 150, False), (400, 149, True)])
def test_print_report_thin_boundaries(capsys, ntr, nte, thin):
    R.print_report(_res(n_train=ntr, n_test=nte))
    assert ("thin panel" in capsys.readouterr().out) is thin


def test_print_report_not_robust_and_missing_test(capsys):
    R.print_report(_res(robust=False))
    out = capsys.readouterr().out
    assert "NOT robust out-of-sample (OOS median +2.0%)" in out
    R.print_report(_res(robust=False, test=None, train=None, full=None))
    out = capsys.readouterr().out
    assert "OOS median +nan%" in out or "OOS median nan%" in out
    assert out.count("(no names)") == 3


def test_print_report_no_chosen(capsys):
    R.print_report(_res(chosen=[]))
    out = capsys.readouterr().out
    assert "no factor pays in both train folds" in out
    assert "DISCOVERED" not in out


def test_main(monkeypatch, capsys, funds, px):
    seen = {}
    monkeypatch.setattr(R, "discover", lambda *a: seen.setdefault("a", a) and _res())
    monkeypatch.setattr(
        "sys.argv",
        [
            "x",
            "--market",
            "JP",
            "--source",
            "warehouse",
            "--horizon",
            "63",
            "--split",
            "2020",
            "--max-filters",
            "2",
            "--fund-dir",
            "/d",
        ],
    )
    R.main()
    assert seen["a"] == ("JP", "warehouse", 63, 2020, 2, "/d")
    assert "VERDICT" in capsys.readouterr().out
    seen.clear()
    monkeypatch.setattr("sys.argv", ["x"])
    R.main()
    assert seen["a"] == ("US", "local", 252, 2021, 3, None)
