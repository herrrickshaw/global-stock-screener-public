#!/usr/bin/env python3
# ticker_aliases.py
# =================
# Map old / delisted / renamed NSE tickers to their current symbol, so holdings and
# tax statements that still use the old code resolve instead of showing NO DATA.
#
# Only high-confidence corporate-action renames/mergers are included. Extend via
# ticker_aliases.json (same {OLD: NEW} shape) next to this file.
#
#   from ticker_aliases import canonical, apply_aliases
#   canonical("ZOMATO")   # -> "ETERNAL"
#
# ⚠️ Verify against the exchange before relying on any mapping for real decisions.

from __future__ import annotations

import json
from pathlib import Path
from typing import List

HERE = Path(__file__).parent

# OLD ticker -> current NSE symbol (confirmed corporate actions)
ALIASES = {
    "ZOMATO": "ETERNAL",  # Zomato Ltd renamed Eternal Ltd
    "PVR": "PVRINOX",  # PVR merged with INOX Leisure -> PVR INOX
    "AMARAJABAT": "ARE&M",  # Amara Raja -> Amara Raja Energy & Mobility
    "GMRINFRA": "GMRAIRPORT",  # GMR Infrastructure -> GMR Airports
    "GET&D": "GEVERNOVA",  # GE T&D India -> GE Vernova T&D India
    "HDFC": "HDFCBANK",  # HDFC Ltd merged into HDFC Bank
    "MINDTREE": "LTIM",  # Mindtree merged -> LTIMindtree
    "L&TFH": "LTF",  # L&T Finance Holdings -> L&T Finance
    "MOTHERSUMI": "MOTHERSON",  # Motherson Sumi -> Samvardhana Motherson (MOTHERSON)
    "SHRIRAMCIT": "SHRIRAMFIN",  # Shriram City Union merged -> Shriram Finance
    "SRTRANSFIN": "SHRIRAMFIN",  # Shriram Transport -> Shriram Finance
    "CADILAHC": "ZYDUSLIFE",  # Cadila Healthcare -> Zydus Lifesciences
    "IBULHSGFIN": "SAMMAANCAP",  # Indiabulls Housing -> Sammaan Capital
    "RELINFRA": "RELINFRA",  # (kept as-is; still listed)
}


def _load() -> dict:
    m = dict(ALIASES)
    f = HERE / "ticker_aliases.json"
    if f.exists():
        try:
            m.update(json.loads(f.read_text()))
        except Exception:
            pass
    return m


def canonical(symbol: str) -> str:
    """Return the current symbol for a possibly-old ticker (unchanged if no alias)."""
    return _load().get(str(symbol).strip().upper(), str(symbol).strip().upper())


def apply_aliases(symbols: List[str]) -> List[str]:
    """Remap a list of tickers to current symbols, de-duplicated, order-preserving."""
    m = _load()
    out = [m.get(s.strip().upper(), s.strip().upper()) for s in symbols]
    return list(dict.fromkeys(out))


if __name__ == "__main__":
    import sys

    for s in sys.argv[1:] or ["ZOMATO", "PVR", "HDFC", "RELIANCE"]:
        print(f"  {s} -> {canonical(s)}")
