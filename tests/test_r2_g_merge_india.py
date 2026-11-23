import sys

import numpy as np
import pandas as pd
import pytest

import merge_india_fundamentals as mi

T = pd.Timestamp


def _ns(df):
    """Pin datetime columns to one resolution (pandas>=3 infers s/us per literal)."""
    for c in ("fy_end", "filed"):
        if c in df.columns and df[c].dtype.kind != "O":
            df[c] = df[c].astype("datetime64[ns]")
    return df


def yf_df(rows):
    return _ns(pd.DataFrame(rows))


def nse_df(rows):
    base = {
        "revenue": 100.0,
        "net_income": 10.0,
        "pretax_income": 14.0,
        "interest_expense": 2.0,
        "depreciation": 3.0,
        "eps": 5.0,
    }
    return _ns(pd.DataFrame([{**base, **r} for r in rows]))


def scr_df(rows):
    base = {
        "equity_capital": 10.0,
        "reserves": 90.0,
        "cash": 5.0,
        "cfo": 6.0,
        "cogs": 40.0,
        "ebit": 20.0,
        "revenue": 100.0,
        "gross_profit": 60.0,
        "shares": 7.0,
        "net_income": 8.0,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_map_yfinance_fills_missing_cols_and_filed_source():
    df = yf_df(
        [
            {"ticker": "A", "fy_end": T("2024-03-31"), "filed": T("2024-05-01"), "revenue": 1.0},
            {"ticker": "A", "fy_end": T("2024-06-30"), "filed": pd.NaT, "revenue": 2.0},
        ]
    )
    out = mi._map_yfinance(df)
    assert list(out["filed_source"]) == ["yfinance", "missing"]
    assert set(out["source"]) == {"yfinance"}
    assert out["filed_nse"].isna().all()
    assert out["total_assets"].isna().all() and list(out["revenue"]) == [1.0, 2.0]
    assert out.index.equals(df.index)


def test_map_nse_derivations():
    df = nse_df(
        [
            {"ticker": "A", "fy_end": T("2024-03-31"), "filed": T("2024-04-20")},
            {
                "ticker": "B",
                "fy_end": T("2024-03-31"),
                "filed": T("2024-04-20"),
                "interest_expense": np.nan,
            },
            {"ticker": "C", "fy_end": T("2024-03-31"), "filed": T("2024-04-20"), "eps": 0.0},
            {"ticker": "D", "fy_end": T("2024-03-31"), "filed": T("2024-04-20"), "eps": np.nan},
            {
                "ticker": "E",
                "fy_end": T("2024-03-31"),
                "filed": T("2024-04-20"),
                "eps": -2.0,
                "net_income": 10.0,
            },
        ]
    )
    out = mi._map_nse(df)
    assert list(out["ebit"]) == [16.0, 14.0, 16.0, 16.0, 16.0]  # NaN interest treated as 0
    assert out["shares"].iloc[0] == 2.0  # 10/5
    assert out["shares"].iloc[1] == 2.0
    assert out["shares"].iloc[2:4].isna().all()  # eps 0 / NaN guarded -> NaN, not inf
    assert out["shares"].iloc[4] == -5.0
    assert out["d_and_a"].iloc[0] == 3.0
    assert set(out["source"]) == {"nse"} and set(out["filed_source"]) == {"nse"}
    assert out["total_assets"].isna().all() and out["cash"].isna().all()


def test_map_screener_proxy_filed_and_equity():
    df = scr_df(
        [
            {"ticker": "A", "fy_end": "2024-03-31"},
            {"ticker": "B", "fy_end": "2024-03-31", "equity_capital": np.nan, "reserves": 7.0},
            {"ticker": "C", "fy_end": "2024-03-31", "equity_capital": 3.0, "reserves": np.nan},
        ]
    )
    out = mi._map_screener(df)
    assert list(out["equity"]) == [100.0, 7.0, 3.0]
    assert (out["filed"] - out["fy_end"]).dt.days.eq(90).all()
    assert out["fy_end"].iloc[0] == T("2024-03-31")
    assert out["cost_of_revenue"].iloc[0] == 40.0 and out["cash"].iloc[0] == 5.0
    assert set(out["filed_source"]) == {"screener_in_proxy"} and set(out["source"]) == {
        "screener_in"
    }
    assert out["long_term_debt"].isna().all() and out["capex"].isna().all()


def test_cross_reference_early_returns():
    y = mi._map_yfinance(yf_df([{"ticker": "A", "fy_end": T("2024-03-31"), "filed": pd.NaT}]))
    assert mi._attach_nse_cross_reference(y, pd.DataFrame()) is y
    nse = nse_df([{"ticker": "A", "fy_end": T("2024-03-31"), "filed": T("2024-04-20")}])
    empty = y.iloc[0:0]
    assert mi._attach_nse_cross_reference(empty, nse) is empty


def test_cross_reference_tolerance_and_fallback():
    y = mi._map_yfinance(
        yf_df(
            [
                {"ticker": "A", "fy_end": T("2024-03-31"), "filed": T("2024-05-01")},  # own filed
                {"ticker": "A", "fy_end": T("2024-06-30"), "filed": pd.NaT},  # NSE fallback
                {"ticker": "A", "fy_end": T("2024-09-30"), "filed": pd.NaT},  # NSE 46d away
                {"ticker": "B", "fy_end": T("2024-06-30"), "filed": pd.NaT},  # other ticker
            ],
        )
    )
    nse = nse_df(
        [
            {"ticker": "A", "fy_end": T("2024-03-30"), "filed": T("2024-04-20")},  # 1d off
            {"ticker": "A", "fy_end": T("2024-08-15"), "filed": T("2024-09-01")},  # 46d from 09-30
            {"ticker": "A", "fy_end": T("2024-07-14"), "filed": T("2024-07-20")},  # 14d from 06-30
        ]
    )
    out = mi._attach_nse_cross_reference(y, nse)
    assert list(out.index) == list(y.index)
    assert out.loc[0, "filed_nse"] == T("2024-04-20") and out.loc[0, "filed"] == T("2024-05-01")
    assert out.loc[0, "filed_source"] == "yfinance"  # own date not overwritten
    assert out.loc[1, "filed_nse"] == T("2024-07-20")
    assert out.loc[1, "filed"] == T("2024-07-20") and out.loc[1, "filed_source"] == "nse_fallback"
    assert pd.isna(out.loc[2, "filed_nse"]) and pd.isna(out.loc[2, "filed"])
    assert out.loc[2, "filed_source"] == "missing"
    assert pd.isna(out.loc[3, "filed_nse"])  # never crosses tickers
    # input frame not mutated
    assert y["filed_nse"].isna().all()


def test_cross_reference_tolerance_boundary_45_days():
    y = mi._map_yfinance(yf_df([{"ticker": "A", "fy_end": T("2024-06-30"), "filed": pd.NaT}]))
    for off, hit in [(45, True), (46, False)]:
        nse = nse_df(
            [
                {
                    "ticker": "A",
                    "fy_end": T("2024-06-30") + pd.Timedelta(days=off),
                    "filed": T("2024-12-01"),
                }
            ]
        )
        out = mi._attach_nse_cross_reference(y, nse)
        assert out["filed_nse"].notna().iloc[0] is np.bool_(hit)


def test_cross_reference_ignores_null_fy_end():
    y = mi._map_yfinance(yf_df([{"ticker": "A", "fy_end": T("2024-06-30"), "filed": pd.NaT}]))
    nse = nse_df([{"ticker": "A", "fy_end": pd.NaT, "filed": T("2024-07-01")}])
    assert mi._attach_nse_cross_reference(y, nse)["filed_nse"].isna().all()


def full_inputs():
    yf = yf_df(
        [
            {
                "ticker": "YF",
                "fy_end": T("2024-06-30"),
                "filed": T("2024-07-30"),
                "total_assets": 5.0,
            },
            {
                "ticker": "YF",
                "fy_end": T("2024-03-31"),
                "filed": T("2024-04-30"),
                "total_assets": 4.0,
            },
        ]
    )
    nse = nse_df(
        [
            {
                "ticker": "YF",
                "fy_end": T("2024-03-31"),
                "filed": T("2024-04-18"),
            },  # dropped (yf wins)
            {"ticker": "NSE", "fy_end": T("2024-03-31"), "filed": T("2024-04-20")},
        ]
    )
    scr = scr_df(
        [
            {"ticker": "YF", "fy_end": "2024-03-31"},  # dropped
            {"ticker": "nse", "fy_end": "2024-03-31"},  # dropped (nse covers, case-insens.)
            {"ticker": "SCR", "fy_end": "2024-03-31"},
        ]
    )
    return yf, nse, scr


def test_build_merged_priority_and_order():
    m = mi.build_merged(*full_inputs())
    assert list(m.columns) == mi.TARGET_COLS
    assert list(m["ticker"]) == ["NSE", "SCR", "YF", "YF"]  # sorted by ticker, fy_end
    assert list(m["source"]) == ["nse", "screener_in", "yfinance", "yfinance"]
    assert list(m[m.ticker == "YF"]["fy_end"]) == [T("2024-03-31"), T("2024-06-30")]
    yfrows = m[m.ticker == "YF"]
    # cross-validated filed date from NSE is recorded for the matching period only
    assert yfrows.iloc[0]["filed_nse"] == T("2024-04-18")
    assert pd.isna(yfrows.iloc[1]["filed_nse"])
    assert list(m.index) == [0, 1, 2, 3]


def test_build_merged_empty_sources():
    yf, nse, scr = full_inputs()
    only_scr = mi.build_merged(pd.DataFrame(), pd.DataFrame(), scr)
    assert set(only_scr["source"]) == {"screener_in"} and len(only_scr) == 3
    only_nse = mi.build_merged(pd.DataFrame(), nse, pd.DataFrame())
    assert set(only_nse["ticker"]) == {"YF", "NSE"} and set(only_nse["source"]) == {"nse"}
    only_yf = mi.build_merged(yf, pd.DataFrame(), pd.DataFrame())
    assert len(only_yf) == 2 and set(only_yf["source"]) == {"yfinance"}
    nothing = mi.build_merged(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert nothing.empty and list(nothing.columns) == mi.TARGET_COLS


# ----------------------------------------------------------------- main
@pytest.fixture
def env(tmp_path, monkeypatch):
    fh = tmp_path / "fh"
    fh.mkdir()
    for n, v in {
        "BASE": str(tmp_path),
        "YF_PATH": str(fh / "IN_yfinance.parquet"),
        "NSE_PATH": str(fh / "IN_nse_results.parquet"),
        "SCREENER_PATH": str(fh / "IN.parquet"),
        "BACKUP_PATH": str(fh / "IN_screener_only_backup.parquet"),
        "DRY_RUN_PATH": str(fh / "IN_merged_dryrun.parquet"),
    }.items():
        monkeypatch.setattr(mi, n, v)
    yf, nse, scr = full_inputs()
    yf.to_parquet(mi.YF_PATH)
    nse.to_parquet(mi.NSE_PATH)
    scr.to_parquet(mi.SCREENER_PATH)
    return fh


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    mi.main()


def test_main_dry_run_does_not_touch_real_file(env, monkeypatch, capsys):
    before = pd.read_parquet(mi.SCREENER_PATH)
    run(monkeypatch)
    out = capsys.readouterr().out
    assert (
        "Loaded: yfinance 2 rows / 1 tickers, nse 2 rows / 2 tickers, screener.in 3 rows / 3 tickers"
        in out
    )
    assert "Merged: 4 rows / 3 tickers" in out
    assert "Tickers with at least one full-balance-sheet row:" in out
    assert "Dry run only -> fh/IN_merged_dryrun.parquet" in out
    assert len(pd.read_parquet(mi.DRY_RUN_PATH)) == 4
    pd.testing.assert_frame_equal(pd.read_parquet(mi.SCREENER_PATH), before)
    assert not (env / "IN_screener_only_backup.parquet").exists()


def test_main_commit_backs_up_once(env, monkeypatch, capsys):
    orig = pd.read_parquet(mi.SCREENER_PATH)
    run(monkeypatch, "--commit")
    out = capsys.readouterr().out
    assert "Backed up original screener.in-only file" in out and "Committed merged result" in out
    assert len(pd.read_parquet(mi.SCREENER_PATH)) == 4
    pd.testing.assert_frame_equal(pd.read_parquet(mi.BACKUP_PATH), orig)
    assert not (env / "IN_merged_dryrun.parquet").exists()


def test_main_commit_never_overwrites_existing_backup(env, monkeypatch, capsys):
    pd.DataFrame({"marker": [1]}).to_parquet(mi.BACKUP_PATH)
    run(monkeypatch, "--commit")
    assert "Backed up" not in capsys.readouterr().out
    assert list(pd.read_parquet(mi.BACKUP_PATH).columns) == ["marker"]


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="merge_india_fundamentals.py:~294: after one --commit, IN.parquet holds the MERGED "
    "schema, but a second run still feeds it to _map_screener (expects equity_capital/reserves) "
    "instead of the IN_screener_only_backup, so re-running --commit raises KeyError",
)
def test_main_commit_is_rerunnable(env, monkeypatch):
    run(monkeypatch, "--commit")
    run(monkeypatch, "--commit")


def test_main_commit_without_screener_file_and_missing_sources(env, monkeypatch, capsys):
    import os

    os.remove(mi.SCREENER_PATH)
    os.remove(mi.NSE_PATH)
    run(monkeypatch, "--commit")
    out = capsys.readouterr().out
    assert "nse 0 rows / 0 tickers, screener.in 0 rows / 0 tickers" in out
    assert "Backed up" not in out and os.path.exists(mi.SCREENER_PATH)
    assert not os.path.exists(mi.BACKUP_PATH)
