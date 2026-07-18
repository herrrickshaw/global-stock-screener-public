#!/usr/bin/env python3
"""
collect_yfinance_pit_fundamentals.py -- point-in-time India fundamentals
sourced from yfinance's quarterly financial statements + earnings_dates,
built to plug the balance-sheet gap left by the other two India sources:

  - IN.parquet (screener.in): P&L-heavy, thin balance sheet (no
    total_assets/current_assets/current_liabilities/long_term_debt/
    retained_earnings/total_liabilities at all), and its `filed` date is a
    fy_end+90d PROXY, not a real disclosure date.
  - IN_nse_results.parquet (NSE results-comparison API): genuine variable
    `filed` date (re_create_dt), but also no proper balance sheet, and the
    source is currently blocked (NSE WAF 403 domain-wide as of 2026-07-17).

yfinance is the only one of the three that has BOTH a full quarterly
balance sheet AND is not currently blocked. `Ticker.quarterly_balance_
sheet` / `.quarterly_income_stmt` / `.quarterly_cashflow` supply the
figures; `Ticker.earnings_dates` supplies a REAL historical announcement
timestamp (verified this session: 9-53 day variable lag vs fiscal period
end on ABBOTINDIA/7203.T/005930.KS -- not a fake constant offset). Each
statement period's `filed` date here is the nearest earnings_dates entry
on/after that period's end -- this is the entire point of using yfinance
over the screener.in fy_end+90d proxy.

FIELD MAPPING: yfinance's line-item labels are not perfectly standardized
across companies (confirmed by inspecting RELIANCE/TCS/ABBOTINDIA's actual
index labels before writing this) -- e.g. long_term_debt appears as
"Long Term Debt" on some tickers and only "Long Term Debt And Capital
Lease Obligation" on others. Every target field below is resolved via an
ordered candidate list (first match wins), not a single hardcoded label.

CAPEX SIGN: yfinance's "Capital Expenditure" is a cash-flow-statement
outflow, typically <= 0. This repo's existing convention (verified against
US.parquet, whose `fcf = cfo - capex` implies capex is stored POSITIVE)
takes abs() here to match.

CASHFLOW GAPS: some tickers (RELIANCE included, verified live) return a
genuinely empty quarterly_cashflow frame from Yahoo -- not a bug, Yahoo
just doesn't have it for that name. cfo/capex/d_and_a stay NaN in that
case rather than crashing the ticker.

CHECKPOINTING: same resumable convention as collect_nse_results.py -- save
every 100 tickers, skip tickers already present in the output on restart.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import yfinance as yf

BASE = "/Users/umashankar/repos/global-stock-screener"
OUT_DIR = os.path.join(BASE, "cache_seed", "fundamentals_history")
OUT = os.path.join(OUT_DIR, "IN_yfinance.parquet")

sys.path.insert(0, BASE)

_DELAY = 0.4
_CHECKPOINT_EVERY = 100
_EARNINGS_TOLERANCE_DAYS = 120  # widest plausible reporting lag (NSE genuine lag was 9-53d; leave headroom)

# Ordered candidate labels -- first one present in the statement wins.
# Built from actually inspecting RELIANCE.NS / TCS.NS / ABBOTINDIA.NS
# index labels live (2026-07-17), not guessed blind.
BS_CANDIDATES = {
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash Financial"],
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "retained_earnings": ["Retained Earnings"],
    "total_liabilities": ["Total Liabilities Net Minority Interest"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
}

INC_CANDIDATES = {
    "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "ebit": ["EBIT"],
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
}

CF_CANDIDATES = {
    "cfo": ["Operating Cash Flow"],
    "capex": ["Capital Expenditure"],
    "d_and_a": ["Depreciation And Amortization", "Reconciled Depreciation"],
}


def _pick(stmt: pd.DataFrame, candidates: list[str], col) -> float | None:
    for label in candidates:
        if label in stmt.index:
            val = stmt.loc[label, col]
            if pd.notna(val):
                return float(val)
    return None


def fetch_ticker(nse_symbol: str) -> tuple[list[dict], str | None]:
    """Returns (rows, suffix_used). Tries .NS first, falls back to .BO.
    suffix_used is None if neither yielded data (log-and-skip case)."""
    for suffix in (".NS", ".BO"):
        yf_ticker = f"{nse_symbol}{suffix}"
        try:
            t = yf.Ticker(yf_ticker)
            bs = t.quarterly_balance_sheet
            inc = t.quarterly_income_stmt
            if bs is None or bs.empty or inc is None or inc.empty:
                continue
            try:
                cf = t.quarterly_cashflow
            except Exception:
                cf = pd.DataFrame()
            try:
                ed = t.earnings_dates
            except Exception:
                ed = None

            # Columns are period-end Timestamps, newest first typically.
            periods = [c for c in bs.columns if c in inc.columns]
            if not periods:
                continue

            ed_dates = None
            if ed is not None and not ed.empty:
                idx = ed.index
                # yfinance earnings_dates index is tz-aware; normalize to naive date.
                ed_dates = pd.Series(pd.to_datetime(idx).tz_localize(None)).sort_values()

            rows = []
            for period in periods:
                fy_end = pd.Timestamp(period)
                rec = {"ticker": nse_symbol, "yf_ticker": yf_ticker, "fy_end": fy_end}

                for dst, cands in BS_CANDIDATES.items():
                    rec[dst] = _pick(bs, cands, period)
                for dst, cands in INC_CANDIDATES.items():
                    rec[dst] = _pick(inc, cands, period)
                if not cf.empty and period in cf.columns:
                    for dst, cands in CF_CANDIDATES.items():
                        v = _pick(cf, cands, period)
                        rec[dst] = abs(v) if (dst == "capex" and v is not None) else v
                else:
                    for dst in CF_CANDIDATES:
                        rec[dst] = None

                # Match to the nearest genuine earnings-announcement date
                # ON OR AFTER this period's end -- this is the entire point
                # of using yfinance over the fy_end+90d proxy.
                filed = None
                if ed_dates is not None:
                    candidates_after = ed_dates[ed_dates >= fy_end]
                    if not candidates_after.empty:
                        cand = candidates_after.iloc[0]
                        if (cand - fy_end).days <= _EARNINGS_TOLERANCE_DAYS:
                            filed = cand
                rec["filed"] = filed
                rows.append(rec)

            if rows:
                return rows, suffix
        except Exception:
            continue
    return [], None


def _universe(limit: int) -> list[str]:
    from collect_nse_results import _universe as _nse_universe
    return _nse_universe(limit)


def main():
    ap = argparse.ArgumentParser(description="Collect India PIT fundamentals from yfinance quarterly statements")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--smoke-test", action="store_true", help="use a fixed 20-ticker smoke sample")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke_test:
        universe = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",  # large-cap
            "ABBOTINDIA", "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA",
            "TATASTEEL", "COALINDIA", "NTPC", "ITC", "HINDUNILVR",  # more large-cap
            "PAGEIND", "MASTEK", "VBL", "SCHNEIDER", "20MICRONS",  # mid/small-cap mix
        ]
    elif args.tickers:
        universe = [t.upper() for t in args.tickers]
    else:
        universe = _universe(args.limit)

    done, existing = set(), []
    if os.path.exists(OUT) and not args.smoke_test:
        prev = pd.read_parquet(OUT)
        done = set(prev["ticker"].astype(str).str.upper())
        existing = [prev]
    todo = [t for t in universe if t not in done]
    print(f"yfinance PIT collector: {len(universe)} tickers, {len(done)} already done, "
          f"{len(todo)} to fetch (~{len(todo) * _DELAY / 60:.1f} min minimum at {_DELAY}s/req)")

    def _checkpoint(rows):
        df = pd.concat(existing + [pd.DataFrame(rows)], ignore_index=True) if (existing or rows) else pd.DataFrame()
        if not df.empty:
            df.to_parquet(OUT, index=False)
        return df

    rows, n_ok, n_fail, t0 = [], 0, 0, time.time()
    for i, tk in enumerate(todo, 1):
        time.sleep(_DELAY)
        r, suffix = fetch_ticker(tk)
        if not r:
            # retry once
            time.sleep(_DELAY)
            r, suffix = fetch_ticker(tk)
        if r:
            rows.extend(r)
            n_ok += 1
        else:
            n_fail += 1
            print(f"  [{i}/{len(todo)}] {tk}: no data from yfinance (.NS or .BO), skipping")
        if i % _CHECKPOINT_EVERY == 0 or i == len(todo):
            df = _checkpoint(rows)
            print(f"  {i}/{len(todo)} | {n_ok} ok | {n_fail} failed | {len(df)} rows | {time.time() - t0:.0f}s")

    df = _checkpoint(rows)
    print(f"done -> {os.path.relpath(OUT, BASE)} ({len(df)} rows, {df['ticker'].nunique() if not df.empty else 0} tickers)")


if __name__ == "__main__":
    main()
