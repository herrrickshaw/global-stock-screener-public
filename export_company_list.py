#!/usr/bin/env python3
# export_company_list.py
# ======================
# Build ONE downloadable list of every company analysed across all 20 markets, from
# the serving views (the deduped, liquidity-annotated universe the screeners run on).
#
# Output (both formats):
#   cache_seed/company_list.csv        ← open in Excel / share
#   cache_seed/company_list.parquet    ← compact, for code
#
# Columns: Market, Symbol, Name (where known), Exchange, Close, Currency,
#          Turnover_USD, Liquidity, LastDate, Bars, Above200DMA, Ret252
#
#   python3 export_company_list.py            # write both files + summary
#   python3 export_company_list.py --market IN
#
# Names come from symbol_master (NSE/BSE/US) where available; other markets list the
# ticker (yfinance suffix identifies the exchange).

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

import datalink
import serving_layer as sl

OUT_DIR = Path(__file__).parent / "cache_seed"

# market → (exchange label, currency, yfinance suffix)
MARKET_META = {
    "IN": ("NSE/BSE", "INR", ".NS/.BO"),
    "US": ("NYSE/NASDAQ", "USD", ""),
    "JP": ("TSE", "JPY", ".T"),
    "KR": ("KRX", "KRW", ".KS/.KQ"),
    "CN": ("SSE/SZSE", "CNY", ".SS/.SZ"),
    "SG": ("SGX", "SGD", ".SI"),
    "EU": ("Euronext", "EUR", ".PA/.AS"),
    "HK": ("HKEX", "HKD", ".HK"),
    "TW": ("TWSE", "TWD", ".TW"),
    "CA": ("TSX", "CAD", ".TO"),
    "AU": ("ASX", "AUD", ".AX"),
    "UK": ("LSE", "GBP", ".L"),
    "DE": ("Xetra", "EUR", ".DE"),
    "SA": ("Tadawul", "SAR", ".SR"),
    "BR": ("B3", "BRL", ".SA"),
    "CH": ("SIX", "CHF", ".SW"),
    "ZA": ("JSE", "ZAR", ".JO"),
    "SE": ("Nasdaq Stockholm", "SEK", ".ST"),
    "FI": ("Nasdaq Helsinki", "EUR", ".HE"),
    "DK": ("Nasdaq Copenhagen", "DKK", ".CO"),
}


def _names() -> dict:
    """{symbol: company name} from the symbol master (NSE/BSE/US), best-effort."""
    try:
        from symbol_master import load_master

        m = load_master()
        if m is not None and not m.empty and "name" in m.columns:
            return dict(zip(m["symbol"].astype(str), m["name"].astype(str)))
    except Exception:
        pass
    return {}


def build(markets=None) -> pd.DataFrame:
    names = _names()
    parts = []
    for mkt in markets or datalink.MARKETS:
        df = sl.serving(mkt)
        if df.empty:
            continue
        exch, ccy, suffix = MARKET_META.get(mkt, (mkt, "", ""))
        keep = df[
            [
                "Symbol",
                "Close",
                "Turnover_USD",
                "Liquidity",
                "LastDate",
                "Bars",
                "Above200DMA",
                "Ret252",
            ]
        ].copy()
        keep.insert(0, "Market", mkt)
        keep.insert(2, "Name", keep["Symbol"].map(names).fillna(""))
        keep.insert(3, "Exchange", exch)
        keep.insert(4, "Currency", ccy)
        parts.append(keep)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["Market", "Turnover_USD"], ascending=[True, False]).reset_index(
        drop=True
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the master company list (all markets)")
    ap.add_argument("--market", nargs="*", help="limit to these markets")
    args = ap.parse_args()

    df = build(args.market)
    if df.empty:
        print("no serving views — run serving_layer.py --refresh first")
        return 1
    csv, pq = OUT_DIR / "company_list.csv", OUT_DIR / "company_list.parquet"
    df.to_csv(csv, index=False)
    df.to_parquet(pq, compression="zstd", index=False)
    named = int((df["Name"] != "").sum())
    print(
        f"  {len(df):,} companies across {df['Market'].nunique()} markets "
        f"({named:,} with names)"
    )
    print(f"  → {csv.relative_to(OUT_DIR.parent)}")
    print(f"  → {pq.relative_to(OUT_DIR.parent)}")
    print("\n  per market:")
    for m, n in df["Market"].value_counts().sort_index().items():
        print(f"    {m:3} {n:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
