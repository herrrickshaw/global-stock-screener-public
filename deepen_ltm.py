#!/usr/bin/env python3
# deepen_ltm.py
# =============
# F-05 — backfill deeper history into the LTM so momentum (Ret252) and the backtest
# have a full multi-year window. Pulls a longer period via data_sources and upserts
# into the market's LTM (dedupe on Symbol+Date). Best-effort / network-bound.
#   python3 deepen_ltm.py --market US --years 5 --limit 200
# Research/education only.
from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")


def deepen(market: str, years: int = 5, limit: int = 0, verbose: bool = True) -> dict:
    import datalink
    import market_memory as mm
    from data_sources import fetch as multi_fetch
    from frames import to_long

    syms = list((datalink.load_market(market, tier="ltm") or datalink.load_market(market)).keys())
    if limit:
        syms = syms[:limit]
    if not syms:
        return {"market": market, "error": "no LTM symbols"}
    if verbose:
        print(f"  deepen {market}: fetching {years}y for {len(syms)} symbols…")
    hist = multi_fetch(syms, period=f"{years}y", min_bars=1, verbose=False)
    res = mm.commit(market, to_long(hist), verbose=verbose)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Deepen LTM history")
    ap.add_argument("--market", default="US")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    print(deepen(args.market, args.years, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
