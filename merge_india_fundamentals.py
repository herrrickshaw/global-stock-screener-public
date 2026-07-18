#!/usr/bin/env python3
"""
merge_india_fundamentals.py -- combine the three India PIT fundamentals
sources into a single consolidated fundamentals_history/IN.parquet:

  1. IN_yfinance.parquet   -- yfinance quarterly statements + earnings_dates.
                               Most complete schema (full balance sheet) AND
                               a genuine PIT filed date. PRIMARY source.
  2. IN_nse_results.parquet -- NSE results-comparison API. Genuine PIT filed
                               date, but P&L-only (no balance sheet).
  3. IN.parquet (original)  -- screener.in. Weakest PIT (fy_end+90d proxy,
                               not a real filing date), P&L-heavy with a
                               thin balance sheet (no total_assets/current_
                               assets/current_liabilities/long_term_debt/
                               retained_earnings/total_liabilities at all).

MERGE RULE (ticker-level source priority, not period-level mixing): if a
ticker has ANY yfinance coverage, ALL of that ticker's rows come from
yfinance (it has the fullest schema) -- rows from the other two sources for
that same ticker are dropped, not blended row-by-row, so a screener never
sees a period that mixes fields from two different data providers. Tickers
with no yfinance coverage fall back to NSE (all their rows); tickers in
neither fall back to screener.in (all their rows). This mirrors the task's
own framing ("for tickers covered by yfinance, use the yfinance row as the
base... tickers with NO yfinance coverage... include them too").

CROSS-VALIDATION: for yfinance-based tickers, NSE's `filed` date is looked
up for the nearest matching fiscal period (+/-45 days) and stored in
`filed_nse` for comparison -- NOT silently preferred over yfinance's own
`filed`, since yfinance's balance-sheet numbers are already tied to its own
`filed` date. `filed_source` records which source's date ended up in the
`filed` column for every row (not just the cross-validated ones), so no
row's PIT provenance is ambiguous.

HONESTY, NOT COMPLETENESS: fields a source structurally cannot supply
(e.g. NSE has no balance sheet at all; screener.in's `borrowings` mixes
short+long-term debt so is deliberately NOT split into long_term_debt/
short_term_debt) are left as NaN. Nothing here is derived by guessing or
back-filling across sources.
"""
from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

BASE = "/Users/umashankar/repos/global-stock-screener"
FH_DIR = os.path.join(BASE, "cache_seed", "fundamentals_history")

YF_PATH = os.path.join(FH_DIR, "IN_yfinance.parquet")
NSE_PATH = os.path.join(FH_DIR, "IN_nse_results.parquet")
SCREENER_PATH = os.path.join(FH_DIR, "IN.parquet")
BACKUP_PATH = os.path.join(FH_DIR, "IN_screener_only_backup.parquet")
DRY_RUN_PATH = os.path.join(FH_DIR, "IN_merged_dryrun.parquet")

TARGET_COLS = [
    "ticker",
    "fy_end",
    "filed",
    "filed_source",
    "filed_nse",
    "source",
    "equity",
    "cash",
    "cfo",
    "cost_of_revenue",
    "ebit",
    "capex",
    "d_and_a",
    "total_assets",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "retained_earnings",
    "total_liabilities",
    "revenue",
    "short_term_debt",
    "gross_profit",
    "shares",
    "net_income",
]

_NSE_TOLERANCE = pd.Timedelta(days=45)


def _map_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in [
        "ticker",
        "fy_end",
        "filed",
        "equity",
        "cash",
        "cfo",
        "cost_of_revenue",
        "ebit",
        "capex",
        "d_and_a",
        "total_assets",
        "current_assets",
        "current_liabilities",
        "long_term_debt",
        "retained_earnings",
        "total_liabilities",
        "revenue",
        "short_term_debt",
        "gross_profit",
        "shares",
        "net_income",
    ]:
        out[c] = df[c] if c in df.columns else pd.NA
    out["source"] = "yfinance"
    out["filed_source"] = out["filed"].notna().map({True: "yfinance", False: "missing"})
    out["filed_nse"] = pd.NaT
    return out


def _map_nse(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ticker"] = df["ticker"]
    out["fy_end"] = df["fy_end"]
    out["filed"] = df["filed"]
    out["revenue"] = df["revenue"]
    out["net_income"] = df["net_income"]
    # EBIT proxy = PBT + interest expense (adds back interest since PBT is
    # already net of it) -- same derivation convention already established
    # for screener.in raw data in this account's earlier collection work.
    out["ebit"] = df["pretax_income"] + df["interest_expense"].fillna(0)
    out["d_and_a"] = df["depreciation"]
    # shares derived algebraically from reported net_income/EPS -- not a
    # fabrication, just arithmetic on two already-reported figures. Guarded
    # against eps==0/NaN to avoid inf/garbage.
    eps = df["eps"]
    valid_eps = eps.notna() & (eps != 0)
    out["shares"] = (df["net_income"] / eps).where(valid_eps)
    for c in [
        "equity",
        "cash",
        "cfo",
        "cost_of_revenue",
        "capex",
        "total_assets",
        "current_assets",
        "current_liabilities",
        "long_term_debt",
        "retained_earnings",
        "total_liabilities",
        "short_term_debt",
        "gross_profit",
    ]:
        out[c] = pd.NA
    out["source"] = "nse"
    out["filed_source"] = "nse"
    out["filed_nse"] = pd.NaT
    return out


def _map_screener(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ticker"] = df["ticker"]
    out["fy_end"] = pd.to_datetime(df["fy_end"])
    # No real filing date in this source -- established fy_end+90d PROXY,
    # explicitly NOT a genuine PIT date (see SCREENER_RESEARCH_DATA_SOURCES.md
    # Sec 2). filed_source flags this so downstream code can distinguish/
    # exclude proxy dates from genuine ones if PIT discipline matters.
    out["filed"] = out["fy_end"] + pd.Timedelta(days=90)
    out["equity"] = df["equity_capital"].fillna(0) + df["reserves"].fillna(0)
    out["cash"] = df["cash"]
    out["cfo"] = df["cfo"]
    out["cost_of_revenue"] = df["cogs"]
    out["ebit"] = df["ebit"]
    out["revenue"] = df["revenue"]
    out["gross_profit"] = df["gross_profit"]
    out["shares"] = df["shares"]
    out["net_income"] = df["net_income"]
    # Deliberately NOT populated: capex, d_and_a, total_assets, current_assets,
    # current_liabilities, retained_earnings, total_liabilities, short_term_debt.
    # `borrowings` exists in the source but mixes short+long-term debt with no
    # split available -- mapping it to long_term_debt alone would be a
    # fabricated split, so long_term_debt stays NaN too.
    for c in [
        "capex",
        "d_and_a",
        "total_assets",
        "current_assets",
        "current_liabilities",
        "long_term_debt",
        "retained_earnings",
        "total_liabilities",
        "short_term_debt",
    ]:
        out[c] = pd.NA
    out["source"] = "screener_in"
    out["filed_source"] = "screener_in_proxy"
    out["filed_nse"] = pd.NaT
    return out


def _attach_nse_cross_reference(yf_rows: pd.DataFrame, nse_raw: pd.DataFrame) -> pd.DataFrame:
    """For yfinance-based rows, look up NSE's own `filed` date for the
    nearest matching ticker+fiscal-period (+/-45 days) and store it in
    filed_nse. Does NOT overwrite `filed`/`filed_source` when yfinance's own
    filed date is present -- only used as the fallback when yfinance itself
    has no earnings_dates match (rare small/mid-cap gap, e.g. 20MICRONS in
    the smoke test)."""
    if nse_raw.empty or yf_rows.empty:
        return yf_rows

    # merge_asof's `on` key must be sorted GLOBALLY across the whole frame,
    # not just within each `by` group -- sorting by ["ticker","fy_end"]
    # looks right but leaves fy_end non-monotonic overall (it resets every
    # time ticker changes), which is what actually tripped "left keys must
    # be sorted" here. `by="ticker"` already handles the per-ticker
    # grouping/matching; the frame only needs to be sorted by fy_end alone.
    nse_small = nse_raw[["ticker", "fy_end", "filed"]].rename(columns={"filed": "_nse_filed"})
    nse_small = nse_small.dropna(subset=["fy_end"]).sort_values("fy_end")

    out = yf_rows.sort_values("fy_end").copy()
    out["_orig_idx"] = out.index
    merged = pd.merge_asof(
        out,
        nse_small,
        by="ticker",
        on="fy_end",
        direction="nearest",
        tolerance=_NSE_TOLERANCE,
    )
    merged = merged.set_index("_orig_idx")
    yf_rows = yf_rows.copy()
    yf_rows.loc[merged.index, "filed_nse"] = merged["_nse_filed"]

    missing_own_filed = yf_rows["filed"].isna() & yf_rows["filed_nse"].notna()
    yf_rows.loc[missing_own_filed, "filed"] = yf_rows.loc[missing_own_filed, "filed_nse"]
    yf_rows.loc[missing_own_filed, "filed_source"] = "nse_fallback"
    return yf_rows


def build_merged(
    yf_df: pd.DataFrame, nse_df: pd.DataFrame, screener_df: pd.DataFrame
) -> pd.DataFrame:
    yf_tickers = set(yf_df["ticker"].astype(str).str.upper()) if not yf_df.empty else set()
    nse_tickers = set(nse_df["ticker"].astype(str).str.upper()) if not nse_df.empty else set()

    yf_mapped = _map_yfinance(yf_df) if not yf_df.empty else pd.DataFrame(columns=TARGET_COLS)
    yf_mapped = _attach_nse_cross_reference(
        yf_mapped, nse_df if not nse_df.empty else pd.DataFrame()
    )

    nse_only_raw = (
        nse_df[~nse_df["ticker"].astype(str).str.upper().isin(yf_tickers)]
        if not nse_df.empty
        else nse_df
    )
    nse_mapped = (
        _map_nse(nse_only_raw) if not nse_only_raw.empty else pd.DataFrame(columns=TARGET_COLS)
    )

    screener_fallback_raw = (
        screener_df[~screener_df["ticker"].astype(str).str.upper().isin(yf_tickers | nse_tickers)]
        if not screener_df.empty
        else screener_df
    )
    screener_mapped = (
        _map_screener(screener_fallback_raw)
        if not screener_fallback_raw.empty
        else pd.DataFrame(columns=TARGET_COLS)
    )

    merged = pd.concat([yf_mapped, nse_mapped, screener_mapped], ignore_index=True)
    merged = merged[TARGET_COLS]
    merged = merged.sort_values(["ticker", "fy_end"]).reset_index(drop=True)
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge India PIT fundamentals sources")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write to IN_merged_dryrun.parquet instead of overwriting IN.parquet",
    )
    ap.add_argument(
        "--commit",
        action="store_true",
        help="back up current IN.parquet and overwrite it with the merged result",
    )
    args = ap.parse_args()

    yf_df = pd.read_parquet(YF_PATH) if os.path.exists(YF_PATH) else pd.DataFrame()
    nse_df = pd.read_parquet(NSE_PATH) if os.path.exists(NSE_PATH) else pd.DataFrame()
    screener_df = (
        pd.read_parquet(SCREENER_PATH) if os.path.exists(SCREENER_PATH) else pd.DataFrame()
    )

    print(
        f"Loaded: yfinance {len(yf_df)} rows / {yf_df['ticker'].nunique() if not yf_df.empty else 0} tickers, "
        f"nse {len(nse_df)} rows / {nse_df['ticker'].nunique() if not nse_df.empty else 0} tickers, "
        f"screener.in {len(screener_df)} rows / {screener_df['ticker'].nunique() if not screener_df.empty else 0} tickers"
    )

    merged = build_merged(yf_df, nse_df, screener_df)

    print(f"\nMerged: {len(merged)} rows / {merged['ticker'].nunique()} tickers")
    print(merged["source"].value_counts())
    print("\nBy source, ticker counts:")
    print(merged.groupby("source")["ticker"].nunique())
    full_bs_cols = ["total_assets", "current_assets", "current_liabilities", "long_term_debt"]
    has_full_bs = merged.groupby("ticker")[full_bs_cols].apply(lambda g: g.notna().any().all())
    print(
        f"\nTickers with at least one full-balance-sheet row: {has_full_bs.sum()} / {merged['ticker'].nunique()}"
    )

    if args.commit:
        if os.path.exists(SCREENER_PATH) and not os.path.exists(BACKUP_PATH):
            shutil.copy2(SCREENER_PATH, BACKUP_PATH)
            print(
                f"\nBacked up original screener.in-only file -> {os.path.relpath(BACKUP_PATH, BASE)}"
            )
        merged.to_parquet(SCREENER_PATH, index=False)
        print(f"Committed merged result -> {os.path.relpath(SCREENER_PATH, BASE)}")
    else:
        out_path = DRY_RUN_PATH
        merged.to_parquet(out_path, index=False)
        print(
            f"\nDry run only -> {os.path.relpath(out_path, BASE)} (pass --commit to overwrite the real IN.parquet)"
        )


if __name__ == "__main__":
    main()
