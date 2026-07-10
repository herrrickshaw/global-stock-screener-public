#!/usr/bin/env python3
"""
ipo_tracker.py — detect new listings (IPOs) and ADD them to the universe.
========================================================================
Wraps ipo_monitor's snapshot-diff detection (reliable and format-consistent),
then for each market:
  • normalizes the new tickers to yfinance form and APPENDS them to
    data/global_universe/<MKT>.csv, so scans + the freshness reference
    automatically pick up post-IPO stocks; and
  • records every addition to data/curated/ipo_additions.parquet — a rolling
    IPO ledger (yf_ticker, exchange, market, first_seen).

BSE has no official-feed snapshot in universe_sources, so it is tracked directly
by diffing the live BSE scrip list against the BSE rows already in IN.csv.

First run per market SEEDS the baseline (no additions); later runs detect deltas.

  python3 ipo_tracker.py                 # default: US, IN(+BSE), EU, TH, ID, MY, MX, IL
  python3 ipo_tracker.py US IN --seed    # seed baselines only, no update
  python3 ipo_tracker.py --no-apply      # detect + record only; don't edit CSVs
"""
import argparse
import datetime
import os

import pandas as pd

BASE = os.path.expanduser("~/Downloads/code/python_files")
UNI_DIR = os.path.join(BASE, "data", "global_universe")
CURATED = os.path.join(BASE, "data", "curated")
LEDGER = os.path.join(CURATED, "ipo_additions.parquet")
DEFAULT_MARKETS = ["US", "IN", "EU", "TH", "ID", "MY", "MX", "IL"]

# market → suffix to normalize a bare official symbol into yfinance form.
_SUFFIX = {"IN": ".NS"}
# market → default exchange label (matches build_universe / the reference).
_EXCH_LABEL = {
    "US": "US",
    "IN": "NSE",
    "EU": "Europe",
    "TH": "SET (Thailand)",
    "ID": "IDX (Indonesia)",
    "MY": "Bursa Malaysia",
    "MX": "BMV (Mexico)",
    "IL": "TASE (Israel)",
}


def _norm(ticker, market):
    t = str(ticker).strip()
    suf = _SUFFIX.get(market)
    return t + suf if (suf and "." not in t) else t


def _exch(ticker, market):
    default = _EXCH_LABEL.get(market, market)
    try:
        import build_universe as bu

        return bu._exch_from(ticker, default)
    except Exception:
        return default


def _append_universe(market, pairs):
    """pairs = [(yf_ticker, exchange), …]; append (deduped) to <MKT>.csv."""
    csv = os.path.join(UNI_DIR, f"{market}.csv")
    cols = ["yf_ticker", "name", "exchange", "market"]
    cur = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame(columns=cols)
    rows = [{"yf_ticker": t, "name": "", "exchange": e, "market": market} for t, e in pairs]
    out = pd.concat([cur, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("yf_ticker")
    out.to_csv(csv, index=False)
    return len(out)


def _bse_new():
    """New BSE scrips = live scrip list minus the BSE rows already in IN.csv."""
    import build_universe as bu

    live = bu.build_bse()
    in_csv = os.path.join(UNI_DIR, "IN.csv")
    existing = set()
    if os.path.exists(in_csv):
        d = pd.read_csv(in_csv)
        existing = set(d[d["exchange"] == "BSE"]["yf_ticker"].astype(str))
    return sorted(set(live["yf_ticker"]) - existing)


def track(markets, seed=False, apply=True, verbose=True):
    import ipo_monitor

    found = ipo_monitor.check(markets, seed=seed, verbose=verbose)
    if seed:
        return found

    today = datetime.date.today().isoformat()
    ledger_rows = []
    for m in markets:
        new = list(found.get(m, {}).get("new", []))
        pairs = [(_norm(t, m), _exch(t, m)) for t in new]
        # India: also catch BSE IPOs from the live scrip list
        if m == "IN":
            bse_new = _bse_new()
            pairs += [(t, "BSE") for t in bse_new]
            if verbose and bse_new:
                print(f"  IN/BSE: {len(bse_new)} new scrip(s)")
        if not pairs:
            continue
        if apply:
            n = _append_universe(m, pairs)
            if verbose:
                print(f"  {m}: +{len(pairs)} new → {m}.csv ({n} tickers)")
        ledger_rows += [
            {"yf_ticker": t, "exchange": e, "market": m, "first_seen": today} for t, e in pairs
        ]

    if ledger_rows:
        led = pd.DataFrame(ledger_rows)
        if os.path.exists(LEDGER):
            led = pd.concat([pd.read_parquet(LEDGER), led], ignore_index=True).drop_duplicates(
                ["yf_ticker", "first_seen"]
            )
        os.makedirs(CURATED, exist_ok=True)
        led.to_parquet(LEDGER, index=False)
        if verbose:
            print(f"  ledger → {os.path.relpath(LEDGER, BASE)} ({len(led)} rows total)")
    return found


def main():
    ap = argparse.ArgumentParser(description="Detect IPOs and add new stocks to the universe")
    ap.add_argument("markets", nargs="*", help="markets (default: US IN EU TH ID MY MX IL)")
    ap.add_argument("--seed", action="store_true", help="seed baselines only, no update")
    ap.add_argument("--no-apply", action="store_true", help="detect + record only; don't edit CSVs")
    args = ap.parse_args()
    mkts = [m.upper() for m in args.markets] or DEFAULT_MARKETS
    print(f"IPO tracker — {'SEED' if args.seed else 'DETECT'} — {mkts}")
    track(mkts, seed=args.seed, apply=not args.no_apply)


if __name__ == "__main__":
    main()
