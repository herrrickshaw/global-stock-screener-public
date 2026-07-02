#!/usr/bin/env python3
# smart_holdings.py
# =================
# F-18 — "follow smart money": surface stocks with recent bulk/block deals or high
# promoter/institutional conviction. Best-effort source (nsepython bulk deals); the
# aggregation is pure + tested. Research/education only. Not advice.
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CACHE = HERE / "cache_seed" / "smart"
CACHE.mkdir(parents=True, exist_ok=True)


def fetch_bulk_deals(verbose: bool = True) -> list:
    """Recent NSE bulk deals (best-effort via nsepython)."""
    try:
        import nsepython as nse

        d = nse.get_bulkdeals() if hasattr(nse, "get_bulkdeals") else []
        rows = d.to_dict("records") if hasattr(d, "to_dict") else (d or [])
    except Exception:
        rows = []
    if verbose:
        print(f"  bulk deals fetched: {len(rows)}")
    return rows


def net_buy_by_symbol(deals: List[dict]) -> Dict[str, float]:
    """Aggregate deal quantities into net (buy − sell) per symbol.
    Expects rows with symbol, buySell (BUY/SELL), quantity."""
    net: Dict[str, float] = defaultdict(float)
    for r in deals:
        sym = str(r.get("symbol") or r.get("Symbol") or r.get("SYMBOL", "")).upper().strip()
        side = str(r.get("buySell") or r.get("BuySell") or r.get("dealType", "")).upper()
        qty = float(r.get("quantity") or r.get("Quantity") or r.get("qty", 0) or 0)
        if not sym:
            continue
        net[sym] += qty if "B" in side else (-qty if "S" in side else 0)
    return dict(net)


def conviction_screen(deals: List[dict], min_net: float = 0) -> List[str]:
    """Symbols with net accumulation via bulk deals, ranked by net quantity."""
    net = net_buy_by_symbol(deals)
    return [s for s, q in sorted(net.items(), key=lambda kv: -kv[1]) if q > min_net]


def main() -> int:
    ap = argparse.ArgumentParser(description="Smart-money (bulk-deal) conviction screen")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    deals = fetch_bulk_deals() if args.fetch else []
    picks = conviction_screen(deals)
    print(
        "bulk-deal net buyers:", picks[:20] if picks else "(run --fetch; source may be throttled)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
