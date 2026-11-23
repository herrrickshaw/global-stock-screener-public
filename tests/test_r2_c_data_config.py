"""Offline tests for data_config."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import data_config as dc


def test_date_split_ranges_and_no_overlap():
    assert dc.DateSplits.get_train_range() == ("2015-01-01", "2021-12-31")
    assert dc.DateSplits.get_validation_range() == ("2022-01-01", "2022-12-31")
    assert dc.DateSplits.get_test_range() == ("2023-01-01", "2024-06-30")
    assert dc.DateSplits.validate_no_overlap() is True


def test_validate_no_overlap_detects(monkeypatch):
    monkeypatch.setattr(dc.DateSplits, "TRAIN_END", "2022-06-01")
    with pytest.raises(AssertionError, match="Train-Val overlap"):
        dc.DateSplits.validate_no_overlap()
    monkeypatch.setattr(dc.DateSplits, "TRAIN_END", "2021-12-31")
    monkeypatch.setattr(dc.DateSplits, "VAL_END", "2023-01-01")  # equal to TEST_START
    with pytest.raises(AssertionError, match="Val-Test overlap"):
        dc.DateSplits.validate_no_overlap()
    monkeypatch.setattr(dc.DateSplits, "TRAIN_END", "2022-01-01")  # equal to VAL_START
    monkeypatch.setattr(dc.DateSplits, "VAL_END", "2022-12-31")
    with pytest.raises(AssertionError, match="Train-Val overlap"):
        dc.DateSplits.validate_no_overlap()


def test_registry():
    assert dc.DataSourceRegistry.get_source("IN")["symbols"] == 2681
    assert dc.DataSourceRegistry.get_source("US")["name"] == "cleaned_long_US.parquet"
    assert dc.DataSourceRegistry.get_source("ZZ") is None
    assert dc.DataSourceRegistry.get_all_markets() == ["IN", "US", "CN", "JP", "KR", "AU", "EU"]


def good_df(n=10):
    c = 100 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": c,
            "high": c + 1,
            "low": c - 1,
            "close": c,
            "volume": 1.0,
            "date": pd.date_range("2024-01-01", periods=n),
        }
    )


def test_validate_dataframe_clean():
    assert dc.DataValidationRules.validate_dataframe(good_df()) == []


def test_validate_dataframe_missing_columns():
    errs = dc.DataValidationRules.validate_dataframe(good_df().drop(columns=["volume", "date"]))
    assert errs == ["Missing critical column: volume", "Missing critical column: date"]
    with pytest.raises(ValueError, match="Missing critical column: open"):
        dc.DataValidationRules.validate_dataframe(
            good_df().drop(columns=["open"]), raise_on_error=True
        )


def test_validate_dataframe_nulls_boundary():
    df = good_df(20)
    df.loc[0, "volume"] = np.nan  # 5% exactly -> allowed
    assert dc.DataValidationRules.validate_dataframe(df) == []
    df.loc[1, "volume"] = np.nan  # 10%
    errs = dc.DataValidationRules.validate_dataframe(df)
    assert errs == ["Column volume has 10.0% nulls (max: 5.0%)"]


def test_validate_dataframe_high_below_low():
    df = good_df()
    df.loc[2, "high"] = df.loc[2, "low"] - 5
    df.loc[3, "high"] = df.loc[3, "low"] - 5
    assert "2 rows with high < low" in dc.DataValidationRules.validate_dataframe(df)
    df = good_df()
    df.loc[2, "high"] = df.loc[2, "low"]  # equal is fine
    assert dc.DataValidationRules.validate_dataframe(df) == []


def test_validate_dataframe_high_low_needs_all_four():
    df = good_df().drop(columns=["open"])
    df.loc[0, "high"] = -999
    errs = dc.DataValidationRules.validate_dataframe(df)
    assert not any("high < low" in e for e in errs)


def test_validate_dataframe_adds_returns_column():
    df = good_df()
    dc.DataValidationRules.validate_dataframe(df)
    assert "returns" in df.columns  # documented side effect


def test_validate_dataframe_no_close_skips_move_check():
    df = good_df().drop(columns=["close"])
    errs = dc.DataValidationRules.validate_dataframe(df)
    assert errs == ["Missing critical column: close"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="data_config.py:~173 compares fractional pct_change to MAX_DAILY_MOVE_PCT=50 "
    "(percent), so a +100% daily move is never flagged as extreme",
)
def test_validate_dataframe_flags_extreme_move():
    df = good_df(10)
    df["close"] = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200.0]
    errs = dc.DataValidationRules.validate_dataframe(df)
    assert any("extreme daily moves" in e for e in errs)


def test_validate_dataframe_extreme_moves_when_huge():
    # a >5000% move (fraction 60) trips the (mis-scaled) check
    df = good_df(10)
    df["close"] = [1.0, 100.0] + [100.0] * 8
    errs = dc.DataValidationRules.validate_dataframe(df)
    assert errs == ["1 extreme daily moves (>50%)"]


def test_universe_presets():
    u = dc.Universe
    assert u.get_universe() is u.INDIA_FOCUS
    assert u.get_universe("us_large_cap") is u.US_LARGE_CAP
    assert u.get_universe("global_all") is u.GLOBAL_ALL
    assert u.get_universe("unknown") is u.INDIA_FOCUS
    assert len(u.INDIA_FOCUS["symbols"]) == 10


@pytest.mark.parametrize(
    "age,req,fresh,msg",
    [
        (
            timedelta(minutes=15),
            dc.FreshnessRequirements.INTRADAY_SCREENER,
            True,
            "Age: 15 min (max: 15 min)",
        ),
        (
            timedelta(minutes=16),
            dc.FreshnessRequirements.INTRADAY_SCREENER,
            False,
            "Age: 16 min (max: 15 min)",
        ),
        (
            timedelta(hours=24),
            dc.FreshnessRequirements.DAILY_SCREENER,
            True,
            "Age: 24.0 hours (max: 24 hours)",
        ),
        (
            timedelta(hours=25),
            dc.FreshnessRequirements.DAILY_SCREENER,
            False,
            "Age: 25.0 hours (max: 24 hours)",
        ),
        (
            timedelta(days=7, hours=1),
            dc.FreshnessRequirements.WEEKLY_ANALYSIS,
            True,
            "Age: 7 days (max: 7 days)",
        ),
        (
            timedelta(days=8),
            dc.FreshnessRequirements.WEEKLY_ANALYSIS,
            False,
            "Age: 8 days (max: 7 days)",
        ),
        (
            timedelta(days=366),
            dc.FreshnessRequirements.BACKTEST_HISTORICAL,
            False,
            "Age: 366 days (max: 365 days)",
        ),
    ],
)
def test_check_freshness(age, req, fresh, msg):
    # small cushion so elapsed wall time between calls cannot cross a boundary
    ts = (
        datetime.now() - age + timedelta(seconds=1)
        if fresh
        else datetime.now() - age - timedelta(seconds=1)
    )
    got_fresh, got_msg = dc.FreshnessRequirements.check_freshness(ts, req)
    assert got_fresh is fresh
    assert got_msg == msg


def test_check_freshness_messages_exact():
    ts = datetime.now() - timedelta(minutes=30)
    f, m = dc.FreshnessRequirements.check_freshness(ts, {"max_age_minutes": 60})
    assert f is True and m.startswith("Age: 30 min") and m.endswith("(max: 60 min)")
    ts = datetime.now() - timedelta(days=3, hours=2)
    f, m = dc.FreshnessRequirements.check_freshness(ts, {"max_age_days": 3})
    assert f is True and m == "Age: 3 days (max: 3 days)"
    ts = datetime.now() - timedelta(days=4, minutes=1)
    assert dc.FreshnessRequirements.check_freshness(ts, {"max_age_days": 3})[0] is False
    f, m = dc.FreshnessRequirements.check_freshness(datetime.now(), {"purpose": "x"})
    assert f is True and m == "Unknown freshness requirement"


@pytest.mark.parametrize(
    "a,b,tol,ok",
    [
        (100, 102, 2.0, True),
        (100, 103, 2.0, False),
        (102, 100, 2.0, True),
        (0, 0.005, 5.0, True),
        (0, 0.01, 5.0, False),
        (0.009, 0, 5.0, True),
        (-100, -101, 2.0, True),
        (-100, 100, 2.0, False),
    ],
)
def test_acceptable_difference(a, b, tol, ok):
    assert dc.ConsistencyRules.acceptable_difference(a, b, tol) is ok


def test_data_config_splits():
    cfg = dc.DataConfig()
    assert cfg.get_split_for_purpose("backtest") == ("2023-01-01", "2024-06-30")
    assert cfg.get_split_for_purpose("validation") == ("2022-01-01", "2022-12-31")
    assert cfg.get_split_for_purpose("training") == ("2015-01-01", "2021-12-31")
    assert cfg.get_split_for_purpose("other") == ("2023-01-01", "2024-06-30")
    assert cfg.consistency.PRICE_DISCREPANCY_PCT == 2.0


def test_filter_data_by_split_inclusive():
    df = pd.DataFrame(
        {"d": ["2022-12-31", "2023-01-01", "2024-06-30", "2024-07-01", "2021-12-31", "2022-01-01"]}
    )
    assert list(dc.filter_data_by_split(df.copy(), "d")["d"].dt.strftime("%Y-%m-%d")) == [
        "2023-01-01",
        "2024-06-30",
    ]
    assert len(dc.filter_data_by_split(df.copy(), "d", "validation")) == 2
    assert len(dc.filter_data_by_split(df.copy(), "d", "train")) == 1
    with pytest.raises(ValueError, match="Unknown split: nope"):
        dc.filter_data_by_split(df.copy(), "d", "nope")


def test_validate_universe_consistency():
    syms = list("ABCDEFGHIJ")
    u = {"symbols": syms, "min_market_cap_usd": 1}
    df = pd.DataFrame({"symbol": list("ABCDEFGH")})  # 8/10 = 80% -> ok
    assert dc.validate_universe_consistency(u, df) == []
    df = pd.DataFrame({"symbol": list("ABCDEFG")})
    assert dc.validate_universe_consistency(u, df) == ["Only 7/10 defined symbols found in data"]
    assert dc.validate_universe_consistency({"symbols": None}, df) == []
    assert dc.validate_universe_consistency({}, df) == []


def test_main_block_runs(capsys):
    import runpy

    runpy.run_path(dc.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "Date splits are non-overlapping" in out
    assert "Configuration validated successfully" in out
    assert "India Primary: cleaned_long.parquet" in out
