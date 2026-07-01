#!/usr/bin/env python3
# sector_map.py
# =============
# DATA_SOURCES.md opportunity #3 — attach SECTOR / industry classification to the
# universe and rank sectors by momentum & breadth (sector rotation), the input the
# doc's "sector boom phase detector" needs but which the price-only path lacks.
#
# Sector source: yfinance `Ticker.info['sector']` (best-effort, cached). For IN it
# also accepts the symbol-master exchange grouping as a coarse fallback. Cached to
# cache_seed/sectors/<MKT>.parquet (Symbol, Sector); the ROTATION analytics run
# offline on whatever is cached joined to the serving views.
#
#   python3 sector_map.py --fetch --market IN --limit 200   # tag sectors → cache
#   python3 sector_map.py --market IN                       # sector-rotation leaderboard
#
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import serving_layer as sl

HERE = Path(__file__).parent
CACHE = HERE / "cache_seed" / "sectors"
CACHE.mkdir(parents=True, exist_ok=True)

# yfinance suffix per market (to query the right symbol)
SUFFIX = {
    "IN": ".NS",
    "US": "",
    "JP": ".T",
    "KR": ".KS",
    "HK": ".HK",
    "TW": ".TW",
    "AU": ".AX",
    "UK": ".L",
    "DE": ".DE",
    "CA": ".TO",
    "SG": ".SI",
    "BR": ".SA",
    "CH": ".SW",
    "ZA": ".JO",
    "SE": ".ST",
    "FI": ".HE",
    "DK": ".CO",
    "EU": ".PA",
    "CN": ".SS",
    "SA": ".SR",
}


def _cache_path(market: str) -> Path:
    return CACHE / f"{market}.parquet"


# ── fetch (best-effort) ──────────────────────────────────────────────────────────
def fetch(
    market: str, limit: int = 200, symbols: Optional[List[str]] = None, verbose: bool = True
) -> int:
    """Tag sectors via yfinance for the top-`limit` liquid names not yet cached."""
    import yfinance as yf

    path = _cache_path(market)
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["Symbol", "Sector"])
    known = set(have["Symbol"]) if not have.empty else set()
    if symbols is None:
        df = sl.serving(market)
        if df.empty:
            if verbose:
                print(f"  no serving view for {market}")
            return 0
        if "Turnover_USD" in df:
            df = df.sort_values("Turnover_USD", ascending=False)
        symbols = df["Symbol"].head(limit * 2).tolist()
    want = [s for s in symbols if s not in known][:limit]
    suf = SUFFIX.get(market, "")
    rows, got = [], 0
    for i, sym in enumerate(want):
        try:
            info = yf.Ticker(f"{sym}{suf}").info
            sec = info.get("sector") or info.get("industryGroup")
            if sec:
                rows.append({"Symbol": sym, "Sector": sec})
                got += 1
        except Exception:
            pass
        time.sleep(0.05)
        if verbose and (i + 1) % 50 == 0:
            print(f"  tagged {i+1}/{len(want)} …")
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
            "Symbol", keep="last"
        )
        have.to_parquet(path, index=False)
    if verbose:
        print(f"  sectors[{market}]: +{got} tagged, {len(have)} total cached")
    return got


def sector_map(market: str) -> Dict[str, str]:
    p = _cache_path(market)
    if not p.exists():
        return {}
    d = pd.read_parquet(p)
    return dict(zip(d["Symbol"], d["Sector"]))


# ── rotation analytics (offline, testable) ───────────────────────────────────────
def rotation(serving: pd.DataFrame, smap: Dict[str, str], min_names: int = 3) -> pd.DataFrame:
    """Rank sectors by momentum + breadth. serving = serving-view frame; smap =
    {symbol: sector}. Returns per-sector medians + a rotation phase."""
    if serving.empty or not smap:
        return pd.DataFrame(
            columns=["Sector", "Names", "MedRet252", "MedRet63", "Breadth", "Phase"]
        )
    df = serving.copy()
    df["Sector"] = df["Symbol"].map(smap)
    df = df.dropna(subset=["Sector"])
    rows = []
    for sec, g in df.groupby("Sector"):
        if len(g) < min_names:
            continue
        r252 = float(np.nanmedian(g["Ret252"])) if "Ret252" in g else np.nan
        r63 = float(np.nanmedian(g["Ret63"])) if "Ret63" in g else np.nan
        breadth = float(g["Above200DMA"].mean()) if "Above200DMA" in g else np.nan
        rows.append(
            {
                "Sector": sec,
                "Names": len(g),
                "MedRet252": round(r252, 1),
                "MedRet63": round(r63, 1),
                "Breadth": round(breadth, 2),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # phase: strong momentum + broad participation = Early/Peak boom; weak = Laggard
    def phase(r):
        if r["Breadth"] >= 0.6 and r["MedRet63"] > 0:
            return "Peak Boom" if r["MedRet252"] > 25 else "Early Boom"
        if r["Breadth"] <= 0.35 or r["MedRet63"] < -5:
            return "Laggard"
        return "Cooling"

    out["Phase"] = out.apply(phase, axis=1)
    return out.sort_values(["Breadth", "MedRet63"], ascending=False).reset_index(drop=True)


def report(market: str = "IN", verbose: bool = True) -> pd.DataFrame:
    smap = sector_map(market)
    if not smap:
        if verbose:
            print(f"no sector cache — run: python3 sector_map.py --fetch --market {market}")
        return pd.DataFrame()
    out = rotation(sl.serving(market), smap)
    if verbose and not out.empty:
        print(f"\n=== Sector rotation — {market} ({sum(v for v in [len(smap)])} tagged) ===")
        print(out.to_string(index=False))
        booms = out[out["Phase"].str.contains("Boom")]["Sector"].tolist()
        if booms:
            print(f"\n  booming sectors (hunt here): {booms}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Sector classification + rotation leaderboard")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--market", default="IN")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    if args.fetch:
        fetch(args.market, args.limit)
    report(args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
