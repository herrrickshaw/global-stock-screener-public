#!/usr/bin/env python3
# partition_ltm.py
# ================
# Year-partition the 5y LTM into a Hive-partitioned parquet dataset so queries scale
# as the LTM fills toward 5 years: DuckDB/pandas can then prune whole years and read
# only the (market, year) partitions a query touches — instead of scanning the full
# per-market file.
#
#   Layout:  cache_seed/ltm_parts/market=<MKT>/year=<YYYY>/data.parquet
#   Each partition is sorted by (Symbol, Date) so row-group min/max stats prune well.
#
# The single-file LTM (cache_seed/ltm/<MKT>.parquet) stays the source of truth for
# the rest of the system; this is a derived, query-optimised mirror (gitignored).
#
#   python3 partition_ltm.py            # (re)build the partitioned dataset from LTM
#   python3 partition_ltm.py --status   # show partition sizes
#
# ⚠️ Research/education only.

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
LTM_DIR = HERE / "cache_seed" / "ltm"
PARTS_DIR = HERE / "cache_seed" / "ltm_parts"


def build(markets: Optional[List[str]] = None, verbose: bool = True) -> dict:
    """Rewrite each LTM market file as market=<MKT>/year=<YYYY>/data.parquet."""
    src = sorted(LTM_DIR.glob("*.parquet"))
    if markets:
        src = [p for p in src if p.stem in markets]
    total_parts, total_rows = 0, 0
    for p in src:
        mkt = p.stem
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        df["year"] = df["Date"].dt.year
        for yr, g in df.groupby("year"):
            out = PARTS_DIR / f"market={mkt}" / f"year={yr}"
            out.mkdir(parents=True, exist_ok=True)
            (
                g.drop(columns=["year"])
                .sort_values(["Symbol", "Date"])
                .to_parquet(out / "data.parquet", compression="zstd", index=False)
            )
            total_parts += 1
            total_rows += len(g)
    if verbose:
        print(
            f"  partitioned {len(src)} market(s) → {total_parts} (market,year) partitions, "
            f"{total_rows:,} rows → {PARTS_DIR.name}/"
        )
    return {"markets": len(src), "partitions": total_parts, "rows": total_rows}


def dataset_glob() -> str:
    """Glob for the partitioned dataset (for DuckDB read_parquet hive_partitioning)."""
    return str(PARTS_DIR / "**" / "*.parquet")


def available() -> bool:
    return PARTS_DIR.exists() and any(PARTS_DIR.rglob("*.parquet"))


def status(verbose: bool = True) -> pd.DataFrame:
    rows = []
    for p in sorted(PARTS_DIR.rglob("*.parquet")):
        mkt = p.parent.parent.name.split("=")[-1]
        yr = p.parent.name.split("=")[-1]
        rows.append({"market": mkt, "year": yr, "MB": round(p.stat().st_size / 1e6, 2)})
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        piv = df.pivot_table(
            index="market", columns="year", values="MB", aggfunc="sum", fill_value=0
        )
        print(piv.to_string())
        print(f"\n  {len(df)} partitions, {df['MB'].sum():.1f} MB total")
    elif verbose:
        print("no partitions — run: python3 partition_ltm.py")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Year-partition the LTM for scalable pruned queries")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--markets", nargs="*")
    args = ap.parse_args()
    if args.status:
        status()
    else:
        build(args.markets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
