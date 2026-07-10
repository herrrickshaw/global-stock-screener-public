#!/usr/bin/env python3
"""
build_universe.py — materialize / refresh ticker universes into
data/global_universe/<MKT>.csv (columns: yf_ticker, name, exchange, market).

Adds the markets the global_universe set was missing:
  • BSE  — India's second exchange (~5k scrips), via the BSE ListofScripData API
           → <SCRIP_CD>.BO, written into IN.csv alongside NSE (market=IN, exchange=BSE).
  • SE/DK/FI/UK/DE/CH — the remaining universe_sources PROVIDERS (Damodaran feeds).

Usage:
  python3 build_universe.py --bse                 # (re)build BSE into IN.csv
  python3 build_universe.py --markets SE,DK,UK     # remaining markets → their CSVs
  python3 build_universe.py --all-remaining        # BSE + all not-yet-materialized markets
"""
import argparse
import os

import pandas as pd
import requests

BASE = os.path.expanduser("~/Downloads/code/python_files")
UNI_DIR = os.path.join(BASE, "data", "global_universe")
os.makedirs(UNI_DIR, exist_ok=True)

_UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json",
}

# yfinance suffix → exchange label, for the Damodaran-feed markets we tag on.
_SUFFIX_EXCH = {
    ".ST": "Nasdaq Stockholm",
    ".CO": "Nasdaq Copenhagen",
    ".HE": "Nasdaq Helsinki",
    ".L": "London Stock Exchange",
    ".DE": "XETRA",
    ".F": "Deutsche Boerse Frankfurt",
    ".SW": "SIX Swiss",
    ".VX": "SIX Swiss",
}
# Non-European markets to materialize (local exchange only).
REMAINING = ["TH", "ID", "MY", "MX", "IL"]


def _exch_from(ticker, default):
    t = str(ticker).upper()
    for suf, ex in _SUFFIX_EXCH.items():
        if t.endswith(suf):
            return ex
    return default


def build_bse():
    """Fetch the full active BSE equity scrip list → rows for IN.csv (exchange=BSE)."""
    url = (
        "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
        "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
    )
    data = requests.get(url, headers=_UA, timeout=30).json()
    rows = [
        {
            "yf_ticker": f"{str(r['SCRIP_CD']).strip()}.BO",
            "name": str(r.get("Scrip_Name", "")).strip(),
            "exchange": "BSE",
            "market": "IN",
        }
        for r in data
        if str(r.get("SCRIP_CD", "")).strip().isdigit()
    ]
    return pd.DataFrame(rows).drop_duplicates("yf_ticker")


def merge_bse_into_in():
    """Append BSE rows to IN.csv without disturbing the existing NSE rows."""
    bse = build_bse()
    in_csv = os.path.join(UNI_DIR, "IN.csv")
    if os.path.exists(in_csv):
        cur = pd.read_csv(in_csv)
        keep = cur[~cur["yf_ticker"].astype(str).str.upper().str.endswith(".BO")]  # drop old BSE
        out = pd.concat([keep, bse], ignore_index=True).drop_duplicates("yf_ticker")
    else:
        out = bse
    out.to_csv(in_csv, index=False)
    return len(bse), len(out)


def build_market(market):
    """Materialize a universe_sources PROVIDERS market → its own CSV."""
    import universe_sources as us

    tickers = us.get_universe(market)
    default_exch = {
        "TH": "SET (Thailand)",
        "ID": "IDX (Indonesia)",
        "MY": "Bursa Malaysia",
        "MX": "BMV (Mexico)",
        "IL": "TASE (Israel)",
        "UK": "London Stock Exchange",
        "DE": "XETRA",
        "CH": "SIX Swiss",
        "SE": "Nasdaq Stockholm",
        "DK": "Nasdaq Copenhagen",
        "FI": "Nasdaq Helsinki",
    }.get(market, market)
    df = pd.DataFrame(
        {
            "yf_ticker": tickers,
            "name": "",
            "exchange": [_exch_from(t, default_exch) for t in tickers],
            "market": market,
        }
    ).drop_duplicates("yf_ticker")
    df.to_csv(os.path.join(UNI_DIR, f"{market}.csv"), index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser(description="Build/refresh ticker universes")
    ap.add_argument("--bse", action="store_true", help="(re)build BSE into IN.csv")
    ap.add_argument("--markets", default="", help="comma list of PROVIDERS markets to materialize")
    ap.add_argument("--all-remaining", action="store_true", help="BSE + all remaining markets")
    args = ap.parse_args()

    if args.bse or args.all_remaining:
        try:
            n_bse, n_in = merge_bse_into_in()
            print(f"BSE: +{n_bse} scrips → IN.csv now {n_in} tickers (NSE+BSE)")
        except Exception as e:
            print(f"BSE build failed: {str(e)[:120]}")

    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    if args.all_remaining:
        markets = REMAINING
    for m in markets:
        try:
            n = build_market(m)
            print(f"{m}: {n} tickers → {m}.csv")
        except Exception as e:
            print(f"{m}: failed ({str(e)[:100]})")


if __name__ == "__main__":
    main()
