import numpy as np
import pandas as pd
import pytest

import reward_alpha_opt as A
import reward_universe_check as U
from tests.r2b_support import make_fundamentals, make_prices


@pytest.fixture(scope="module")
def px():
    return make_prices(n_sym=300)


@pytest.fixture(scope="module")
def base_panel(px):
    orig = U._prices
    U._prices = lambda m: px[["Date", "Close", "Volume", "Symbol"]]
    try:
        return U._panel("XX")
    finally:
        U._prices = orig


@pytest.fixture
def fund_base(tmp_path, monkeypatch, px):
    d = tmp_path / "cache_seed" / "fundamentals_history"
    d.mkdir(parents=True)
    make_fundamentals(px).to_parquet(d / "US.parquet")
    monkeypatch.setattr(A, "BASE", str(tmp_path))
    monkeypatch.setattr(U, "BASE", str(tmp_path))
    return tmp_path


def test_dirs_and_mask():
    d = A._dirs(["mom_12_1", "leverage", "roa"])
    assert d == {"mom_12_1": 1, "leverage": -1, "roa": 1}
    sub = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.nan]})
    assert list(A._mask(sub, "x", +1, 0.5)) == [False, False, True, True, False]
    assert list(A._mask(sub, "x", -1, 0.5)) == [True, True, False, False, False]


def _df(years=range(2017, 2021), n=30):
    rows = []
    for y in years:
        for k in range(n):
            rows.append({"year": y, "x": float(k), "fwd": 0.01 * k})
    return pd.DataFrame(rows)


def test_alpha_over_values_and_guards():
    df = _df()
    r = A._alpha_over(df, [2017, 2018], [("x", +1, 0.5)])
    assert r["yrs"] == 2 and r["alpha"] > 0 and r["win"] == 1.0
    assert r["n"] == 15
    assert r["cagr"] == pytest.approx((1 + r["alpha"]) ** 1 - 1)
    # fewer than 2 usable years -> None
    assert A._alpha_over(df, [2017], [("x", +1, 0.5)]) is None
    # year with <20 rows skipped
    small = pd.concat([df, pd.DataFrame({"year": [2030] * 19, "x": 1.0, "fwd": 0.1})])
    assert A._alpha_over(small, [2017, 2030], [("x", +1, 0.5)]) is None
    twenty = pd.concat([df, pd.DataFrame({"year": [2030] * 20, "x": np.arange(20.0), "fwd": 0.1})])
    assert A._alpha_over(twenty, [2017, 2030], [("x", +1, 0.5)])["yrs"] == 2
    # portfolio with <5 names skipped: keep 0.1 of 30 -> 3
    assert A._alpha_over(df, [2017, 2018], [("x", +1, 0.1)]) is None
    # 5 names ok (keep 0.2 of 30 -> 6)
    assert A._alpha_over(df, [2017, 2018], [("x", +1, 0.2)])["n"] == 6


def test_alpha_over_negative_alpha_and_win():
    df = _df()
    df["fwd"] = -df["fwd"]
    r = A._alpha_over(df, [2017, 2018, 2019], [("x", +1, 0.5)])
    assert r["alpha"] < 0 and r["win"] == 0.0


def test_alpha_over_two_filters_intersect():
    df = _df()
    df["y"] = (df["x"] % 2).astype(float)
    one = A._alpha_over(df, [2017, 2018], [("x", +1, 0.5)])
    two = A._alpha_over(df, [2017, 2018], [("x", +1, 0.5), ("y", +1, 0.5)])
    assert two["n"] < one["n"]


def test_lbl():
    assert A._lbl([]) == "(none)"
    assert A._lbl([("a", 1, 0.3), ("b", -1, 0.25)]) == "a↑30% & b↓25%"


def test_with_fundamentals_non_us_passthrough(base_panel):
    d, f = A._with_fundamentals(base_panel, "JP")
    assert d is base_panel and f == list(U.FACTORS)


def test_with_fundamentals_us_missing_file(tmp_path, monkeypatch, base_panel):
    monkeypatch.setattr(A, "BASE", str(tmp_path))
    d, f = A._with_fundamentals(base_panel, "US")
    assert d is base_panel and f == list(U.FACTORS)


def test_with_fundamentals_us_joins_point_in_time(fund_base, base_panel, px):
    d, f = A._with_fundamentals(base_panel, "US")
    assert f == list(U.FACTORS) + list(A.FUND_DIRS)
    row = d[(d["symbol"] == "S150") & (d["year"] == 2020)].iloc[0]
    fund = make_fundamentals(px)
    fund = fund[fund["ticker"] == "S150"]
    # latest filing on/before 30-Jun-2020 is FY2019 (filed 2020-03-30); FY2020 is filed 2021
    fy = fund[fund["fy_end"] == pd.Timestamp(2019, 12, 31)].iloc[0]
    assert row["total_assets"] == pytest.approx(fy["total_assets"])
    assert row["roa"] == pytest.approx(fy["net_income"] / fy["total_assets"])
    cap = fy["shares"] * row["close"]
    assert row["book_to_market"] == pytest.approx(fy["equity"] / cap)
    assert row["earnings_yield"] == pytest.approx(fy["net_income"] / cap)
    assert row["cfo_ta"] == pytest.approx(fy["cfo"] / fy["total_assets"])
    assert row["leverage"] == pytest.approx(fy["long_term_debt"] / fy["total_assets"])
    assert row["gross_margin"] == pytest.approx(fy["gross_profit"] / fy["revenue"])
    # 2017 rebalance: filing filed 2017-03-31 for FY2016 is the latest
    r17 = d[(d["symbol"] == "S150") & (d["year"] == 2017)].iloc[0]
    fy16 = fund[fund["fy_end"] == pd.Timestamp(2016, 12, 31)].iloc[0]
    assert r17["total_assets"] == pytest.approx(fy16["total_assets"])


def test_with_fundamentals_drops_unparseable_filed(fund_base, base_panel, px):
    p = fund_base / "cache_seed" / "fundamentals_history" / "US.parquet"
    f = make_fundamentals(px)
    f.loc[f["fy_end"] == pd.Timestamp(2019, 12, 31), "filed"] = "garbage"
    f.to_parquet(p)
    d, _ = A._with_fundamentals(base_panel, "US")
    row = d[(d["symbol"] == "S150") & (d["year"] == 2020)].iloc[0]
    assert row["total_assets"] == pytest.approx(
        f[(f["ticker"] == "S150") & (f["fy_end"] == pd.Timestamp(2018, 12, 31))][
            "total_assets"
        ].iloc[0]
    )


def test_optimise_price_only(monkeypatch, base_panel):
    monkeypatch.setattr(U, "_panel", lambda m: base_panel)
    r = A.optimise("JP")
    assert r["market"] == "JP" and 1 <= len(r["chosen"]) <= 2
    names = [c[0] for c in r["chosen"]]
    assert len(set(names)) == len(names)
    assert all(n in U.FACTORS for n in names)
    assert r["pool"] == sorted(r["pool"], key=lambda x: -x[1])
    assert r["train"]["yrs"] <= 5 and r["test"]["yrs"] == 3
    assert r["full"]["yrs"] == 8
    assert all(c > 0 for _, c in r["pool"])
    # low volatility must be a pool member for this planted panel
    assert "vol_60" in [p[0][0] for p in r["pool"]]


def test_optimise_us_includes_fundamentals(fund_base, monkeypatch, base_panel):
    monkeypatch.setattr(U, "_panel", lambda m: base_panel)
    r = A.optimise("US")
    pool_names = {p[0][0] for p in r["pool"]}
    assert pool_names & set(A.FUND_DIRS)  # some fundamental factor pays in both folds


def test_optimise_empty_and_short(monkeypatch, base_panel):
    monkeypatch.setattr(U, "_panel", lambda m: base_panel.iloc[0:0])
    assert A.optimise("JP") is None
    short = base_panel[base_panel["year"].between(2017, 2020)]
    monkeypatch.setattr(U, "_panel", lambda m: short)
    assert A.optimise("JP") is None  # <2 test years
    few_train = base_panel[base_panel["year"].isin([2019, 2020, 2021, 2022, 2023, 2024])]
    monkeypatch.setattr(U, "_panel", lambda m: few_train)
    assert A.optimise("JP") is None  # 3 train years (<4)


def test_optimise_four_train_years_is_enough(monkeypatch, base_panel):
    ok = base_panel[base_panel["year"].isin([2018, 2019, 2020, 2021, 2022, 2023])]
    monkeypatch.setattr(U, "_panel", lambda m: ok)
    assert A.optimise("JP") is not None


def test_optimise_no_pool_gives_empty_chosen(monkeypatch, base_panel):
    bad = base_panel.copy()
    bad["fwd"] = np.random.default_rng(0).standard_normal(len(bad)) * 0
    monkeypatch.setattr(U, "_panel", lambda m: bad)
    # all-zero forward returns -> zero alpha everywhere -> nothing > 0
    r = A.optimise("JP")
    assert r["chosen"] == [] and r["pool"] == []
    assert r["train"]["alpha"] == pytest.approx(0.0, abs=1e-12)


def test_optimise_liquidity_gate(monkeypatch, base_panel):
    seen = []
    orig = A._with_fundamentals

    def spy(df, m):
        seen.append(len(df))
        return orig(df, m)

    monkeypatch.setattr(A, "_with_fundamentals", spy)
    monkeypatch.setattr(U, "_panel", lambda m: base_panel)
    A.optimise("JP")
    assert seen == [(base_panel["dvol"] >= base_panel["dvol"].quantile(0.5)).sum()]


def _res(chosen, tr, te, pool=None):
    return {
        "market": "XX",
        "chosen": chosen,
        "train": tr,
        "test": te,
        "full": None,
        "pool": pool or [],
    }


def _stat(alpha, win=0.6):
    return {"alpha": alpha, "yrs": 3, "win": win, "n": 20, "cagr": alpha}


@pytest.fixture
def mainrun(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(A, "BASE", str(tmp_path))

    def go(results, argv=()):
        monkeypatch.setattr(A, "optimise", lambda m: results[m])
        monkeypatch.setattr("sys.argv", ["x", *argv])
        A.main()
        return capsys.readouterr().out

    return go


def test_main_outputs_and_csv(mainrun, tmp_path):
    f = [("vol_60", -1, 0.4)]
    out = mainrun(
        {
            "US": _res(f, _stat(0.02), _stat(0.01), [(f[0], 0.03)]),
            "JP": None,
            "XX": _res(f, None, _stat(-0.01)),
            "YY": _res(f, _stat(0.02), _stat(0.01, win=0.49)),
            "ZZ": _res(f, _stat(0.02), None),
        },
        ["--markets", "us, jp,xx,yy,zz"],
    )
    assert "JP: no data" in out
    assert "US — alpha-optimal screen: vol_60↓40%" in out
    assert "vol_60↓40%=+3.0%" in out
    assert "TRAIN  alpha/yr  +2.0%  win  60%  (~20 names)" in out
    assert "OOS    alpha/yr  +1.0%" in out
    assert out.count("✅ beats benchmark OOS") == 1
    assert out.count("⚠ no OOS alpha") == 2  # XX (neg alpha) and YY (win<0.5)
    csv = pd.read_csv(tmp_path / "reports" / "reward_alpha_reopt.csv")
    assert list(csv["market"]) == ["US", "XX", "YY"]
    assert list(csv["robust"]) == [True, False, False]
    assert csv.loc[1, "train_alpha"] != csv.loc[1, "train_alpha"]  # NaN (no train)
    assert "written → reports/reward_alpha_reopt.csv" in out


def test_main_win_boundary_is_inclusive(mainrun):
    f = [("vol_60", -1, 0.4)]
    out = mainrun({"US": _res(f, _stat(0.02), _stat(0.001, win=0.5))}, ["--markets", "US"])
    assert "✅ beats" in out
    out = mainrun({"US": _res(f, _stat(0.02), _stat(0.0, win=0.9))}, ["--markets", "US"])
    assert "⚠ no OOS alpha" in out


def test_main_no_csv_when_nothing(mainrun, tmp_path):
    out = mainrun({"US": None, "JP": None})
    assert not (tmp_path / "reports").exists()
    assert "written" not in out and "Research only" in out
