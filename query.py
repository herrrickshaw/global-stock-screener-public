#!/usr/bin/env python3
# query.py
# ========
# A DuckDB SQL layer over the committed parquet seeds — vectorised, out-of-core,
# with row-group pruning — so ad-hoc historical queries ("all bars for TCS in 2025",
# "top turnover by market") run in one indexed scan instead of loading whole frames.
#
# Why DuckDB (vs the LMDB B+tree + pandas path): LMDB is O(log n) for *one symbol*;
# for cross-symbol/range analytics over millions of EOD rows DuckDB's columnar,
# vectorised engine reads parquet directly, prunes by row-group min/max, and streams
# larger-than-RAM results — the right tool for range/aggregate queries.
#
#   from query import sql, bars, latest, con
#   sql("SELECT Market, count(DISTINCT Symbol) FROM ltm GROUP BY 1")
#   bars("TCS", "IN", start="2025-01-01")            # range query, pruned
#   latest("IN", where="Close > 1000")               # newest bar per symbol
#
#   python3 query.py "SELECT Market, count(*) FROM ltm GROUP BY 1 ORDER BY 2 DESC"
#
# Optional dependency: needs `duckdb`. If absent, functions raise a clear message.
# ⚠️ Research/education only.

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
LTM_GLOB = str(HERE / "cache_seed" / "ltm" / "*.parquet")
SERVING_GLOB = str(HERE / "cache_seed" / "serving" / "*.parquet")
_CON = None


def _require():
    try:
        import duckdb  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "query.py needs DuckDB — `pip install duckdb`. "
            "Falls outside the zero-dep core; install it to use the SQL layer."
        ) from e


def con():
    """Return a cached DuckDB connection with `ltm` and `serving` views registered.

    `ltm` derives Market from the filename (…/ltm/<MKT>.parquet) so a single view
    spans all 20 markets with predicate pushdown."""
    global _CON
    if _CON is not None:
        return _CON
    _require()
    import duckdb

    c = duckdb.connect(":memory:")
    # LTM view: prefer the year-partitioned dataset (Hive market=/year= → prunes whole
    # years); else the single-file-per-market parquet. Both expose a `Market` column.
    parts = HERE / "cache_seed" / "ltm_parts"
    if parts.exists() and any(parts.rglob("*.parquet")):
        # hive gives lowercase `market`/`year`; alias `market`→`Market` for parity
        # with the single-file view so callers use one column name either way.
        c.execute(
            f"""CREATE VIEW ltm AS SELECT * RENAME (market AS Market) FROM
                read_parquet('{parts}/**/*.parquet', hive_partitioning=true)"""
        )
    elif list(Path(LTM_GLOB).parent.glob("*.parquet")):
        c.execute(
            f"""CREATE VIEW ltm AS
                SELECT *, regexp_extract(filename, 'ltm/([A-Z]+)\\.parquet', 1) AS Market
                FROM read_parquet('{LTM_GLOB}', filename=true)"""
        )
    if list((HERE / "cache_seed" / "serving").glob("*.parquet")):
        c.execute(f"CREATE VIEW serving AS SELECT * FROM read_parquet('{SERVING_GLOB}')")
    _CON = c
    return c


def sql(query: str) -> pd.DataFrame:
    """Run arbitrary SQL against the `ltm` / `serving` views → DataFrame."""
    return con().execute(query).fetchdf()


def bars(
    symbol: str,
    market: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """OHLCV bars for a symbol over an optional date range (range query, pruned)."""
    where = [f"Symbol = '{symbol}'"]
    if market:
        where.append(f"Market = '{market}'")
    if start:
        where.append(f"Date >= DATE '{start}'")
    if end:
        where.append(f"Date <= DATE '{end}'")
    return sql(
        f"SELECT Date, Open, High, Low, Close, Volume FROM ltm "
        f"WHERE {' AND '.join(where)} ORDER BY Date"
    )


def latest(market: str, where: str = "TRUE") -> pd.DataFrame:
    """Newest bar per symbol for a market (QUALIFY window) with an optional filter."""
    return sql(
        f"""SELECT * FROM (
              SELECT *, row_number() OVER (PARTITION BY Symbol ORDER BY Date DESC) rn
              FROM ltm WHERE Market = '{market}'
            ) WHERE rn = 1 AND ({where}) ORDER BY Symbol"""
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="DuckDB SQL over the parquet seeds")
    ap.add_argument("query", nargs="?", help="SQL against views `ltm` / `serving`")
    args = ap.parse_args()
    q = args.query or (
        "SELECT Market, count(*) AS n_rows, count(DISTINCT Symbol) AS n_syms "
        "FROM ltm GROUP BY 1 ORDER BY 2 DESC"
    )
    with pd.option_context("display.max_rows", 40, "display.width", 120):
        print(sql(q).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
