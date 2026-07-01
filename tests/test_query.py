# Tests for the DuckDB query layer. Skips cleanly if duckdb isn't installed.
import pytest

duckdb = pytest.importorskip("duckdb")
import query  # noqa: E402


def test_ltm_view_and_market_col():
    df = query.sql("SELECT DISTINCT Market FROM ltm ORDER BY 1 LIMIT 25")
    mkts = set(df["Market"])
    assert {"IN", "US"} <= mkts  # markets derived from filename
    assert all(len(m) <= 3 for m in mkts)  # clean market codes, not paths


def test_bars_range_query():
    b = query.bars("RELIANCE", "IN", start="2026-01-01")
    assert not b.empty
    assert list(b.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert b["Date"].is_monotonic_increasing


def test_latest_one_row_per_symbol():
    df = query.latest("IN")
    assert df["Symbol"].is_unique  # newest bar per symbol
    assert len(df) > 1000


def test_sql_aggregate():
    df = query.sql("SELECT Market, count(*) AS n FROM ltm GROUP BY 1")
    assert df["n"].sum() > 1_000_000  # ~9M rows across markets
