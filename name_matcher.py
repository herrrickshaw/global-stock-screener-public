#!/usr/bin/env python3
# name_matcher.py
# ===============
# Resolve COMPANY NAMES → tickers. Many broker capital-gains / tax P&L statements
# list only the company name ("Reliance Industries Ltd"), not the ticker (RELIANCE).
# This maps names to tickers using the symbol master (NSE/BSE/US name↔ticker) plus
# the company list, with normalisation + fuzzy matching.
#
#   from name_matcher import resolve
#   resolve(["Reliance Industries", "Tata Consultancy", "Infosys Ltd"], market="IN")
#   → {"Reliance Industries": ("RELIANCE", 1.0), ...}, unmatched=[...]
#
# ⚠️ Research/education only.

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).parent

# corporate suffixes / noise stripped before matching
_NOISE = re.compile(
    r"\b(ltd|limited|inc|incorporated|corp|corporation|co|company|plc|"
    r"industries|enterprises|holdings|group|the|and|&|pvt|private|"
    r"technologies|tech|services|international|intl)\b",
    re.I,
)


def clean(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    s = _NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(maxsize=8)
def _index(market: str) -> Tuple[dict, dict, tuple]:
    """(exact_clean→ticker, first_token→[tickers], all_clean_names) for a market."""
    pairs: List[Tuple[str, str]] = []  # (name, ticker)
    # 1) symbol master (has name / name_clean / symbol; NSE/BSE/US)
    try:
        from symbol_master import load_master

        m = load_master()
        if m is not None and not m.empty:
            for _, r in m.iterrows():
                pairs.append((str(r.get("name", "")), str(r["symbol"])))
    except Exception:
        pass
    # 2) company list (Market-scoped Name → Symbol)
    p = HERE / "cache_seed" / "company_list.parquet"
    if p.exists():
        cl = pd.read_parquet(p)
        cl = cl[cl["Market"] == market] if "Market" in cl else cl
        for _, r in cl.iterrows():
            if str(r.get("Name", "")):
                pairs.append((str(r["Name"]), str(r["Symbol"])))
        # also allow ticker-as-name (so a ticker typed in the name field still resolves)
        for s in cl["Symbol"]:
            pairs.append((str(s), str(s)))

    exact: Dict[str, str] = {}
    for name, tk in pairs:
        c = clean(name)
        if c and c not in exact:
            exact[c] = tk
    return exact, {}, tuple(exact.keys())


def match(name: str, market: str = "IN", cutoff: float = 0.82) -> Optional[Tuple[str, float]]:
    """Best (ticker, score) for a company name, or None."""
    exact, _, all_names = _index(market)
    c = clean(name)
    if not c:
        return None
    if c in exact:
        return exact[c], 1.0
    # substring containment (either direction) — strong signal
    for cand in all_names:
        if cand and (c in cand or cand in c) and abs(len(cand) - len(c)) <= 6:
            return exact[cand], 0.95
    # fuzzy
    near = difflib.get_close_matches(c, all_names, n=1, cutoff=cutoff)
    if near:
        return exact[near[0]], round(difflib.SequenceMatcher(None, c, near[0]).ratio(), 2)
    return None


def resolve(names: List[str], market: str = "IN") -> Tuple[Dict[str, Tuple[str, float]], List[str]]:
    """names → ({name: (ticker, score)}, unmatched_names)."""
    hits, misses = {}, []
    for n in names:
        m = match(n, market)
        (hits.__setitem__(n, m) if m else misses.append(n))
    return hits, misses


if __name__ == "__main__":
    import sys

    market = "IN"
    names = sys.argv[1:] or ["Reliance Industries Ltd", "Tata Consultancy Services", "Infosys"]
    hits, misses = resolve(names, market)
    for n, (tk, sc) in hits.items():
        print(f"  {n:35} → {tk:12} ({sc})")
    if misses:
        print("  unmatched:", misses)
