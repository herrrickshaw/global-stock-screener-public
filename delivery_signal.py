#!/usr/bin/env python3
# delivery_signal.py
# ==================
# DATA_SOURCES.md opportunity #1 — mine NSE **delivery %** (deliverable quantity as a
# fraction of traded volume) for an accumulation/distribution signal the plain OHLCV
# path ignores.
#
# High and *rising* delivery % means buyers are taking delivery (holding), not day-
# trading — a classic accumulation tell; falling delivery % on a rally warns of weak,
# speculative hands.
#
# Source: NSE full security-wise bhavcopy
#   nsearchives.nseindia.com/products/content/sec_bhavdata_full_<DDMMYYYY>.csv
#   columns include DELIV_QTY, DELIV_PER, TTL_TRD_QNTY.
# Fetched best-effort + cached to cache_seed/delivery/IN.parquet (Date,Symbol,DelivPct,Volume);
# the SCORING works offline on whatever is cached.
#
#   python3 delivery_signal.py --fetch --days 30   # pull recent delivery data → cache
#   python3 delivery_signal.py                      # accumulation report from cache
#
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import datetime as _dt
import io
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CACHE = HERE / "cache_seed" / "delivery"
CACHE.mkdir(parents=True, exist_ok=True)
_UA = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}


# ── fetch (best-effort) ──────────────────────────────────────────────────────────
def _fetch_day(day: _dt.date) -> Optional[pd.DataFrame]:
    import requests

    url = "https://nsearchives.nseindia.com/products/content/" f"sec_bhavdata_full_{day:%d%m%Y}.csv"
    try:
        r = requests.get(url, headers=_UA, timeout=25)
        if r.status_code != 200 or not r.content:
            return None
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        df = df[df.get("SERIES", "EQ").astype(str).str.strip() == "EQ"]
        out = pd.DataFrame(
            {
                "Date": pd.Timestamp(day),
                "Symbol": df["SYMBOL"].astype(str).str.strip(),
                "DelivPct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
                "Volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
            }
        )
        return out.dropna(subset=["DelivPct"])
    except Exception:
        return None


def fetch(days: int = 30, verbose: bool = True) -> int:
    """Pull recent daily delivery data → cache_seed/delivery/IN.parquet (dedup by date)."""
    path = CACHE / "IN.parquet"
    have = (
        pd.read_parquet(path)
        if path.exists()
        else pd.DataFrame(columns=["Date", "Symbol", "DelivPct", "Volume"])
    )
    have["Date"] = pd.to_datetime(have.get("Date", pd.Series(dtype="datetime64[ns]")))
    known = set(have["Date"].dt.date) if not have.empty else set()
    got = 0
    for i in range(days):
        d = _dt.date.today() - _dt.timedelta(days=i)
        if d.weekday() >= 5 or d in known:
            continue
        day = _fetch_day(d)
        if day is not None and not day.empty:
            have = pd.concat([have, day], ignore_index=True)
            got += 1
    if got:
        have.drop_duplicates(["Date", "Symbol"], keep="last").to_parquet(path, index=False)
    if verbose:
        print(f"  delivery: +{got} new day(s), {path.name} now {len(have)} rows")
    return got


# ── scoring (offline, testable) ──────────────────────────────────────────────────
def accumulation_scores(df: pd.DataFrame, recent: int = 5, base: int = 20) -> pd.DataFrame:
    """Per-symbol accumulation score from a delivery-% history frame.

    score = mean(recent DelivPct) / mean(prior DelivPct) ; >1 = rising delivery
    (accumulation). Also flags absolute high delivery (recent mean > 60%)."""
    if df.empty:
        return pd.DataFrame(columns=["Symbol", "DelivRecent", "DelivBase", "AccumScore", "Signal"])
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    rows = []
    for sym, g in df.sort_values("Date").groupby("Symbol"):
        d = g["DelivPct"].to_numpy("float64")
        if len(d) < recent + 3:
            continue
        rec = np.nanmean(d[-recent:])
        bas = (
            np.nanmean(d[-(base + recent) : -recent])
            if len(d) > recent
            else np.nanmean(d[:-recent])
        )
        if not np.isfinite(bas) or bas <= 0:
            continue
        ratio = rec / bas
        if rec >= 60 and ratio >= 1.1:
            sig = "STRONG ACCUMULATION"
        elif ratio >= 1.15:
            sig = "ACCUMULATION"
        elif ratio <= 0.8:
            sig = "DISTRIBUTION"
        else:
            sig = "NEUTRAL"
        rows.append(
            {
                "Symbol": sym,
                "DelivRecent": round(rec, 1),
                "DelivBase": round(bas, 1),
                "AccumScore": round(ratio, 2),
                "Signal": sig,
            }
        )
    out = pd.DataFrame(rows)
    return (
        out.sort_values("AccumScore", ascending=False).reset_index(drop=True)
        if not out.empty
        else out
    )


def delivery_map(recent: int = 5, base: int = 20) -> Dict[str, float]:
    """{symbol: recent delivery %} — usable by liquidity/decision layers."""
    path = CACHE / "IN.parquet"
    if not path.exists():
        return {}
    sc = accumulation_scores(pd.read_parquet(path), recent, base)
    return dict(zip(sc["Symbol"], sc["DelivRecent"])) if not sc.empty else {}


def report(top: int = 20, verbose: bool = True) -> pd.DataFrame:
    path = CACHE / "IN.parquet"
    if not path.exists():
        if verbose:
            print("no delivery cache — run: python3 delivery_signal.py --fetch")
        return pd.DataFrame()
    sc = accumulation_scores(pd.read_parquet(path))
    if verbose and not sc.empty:
        print("\n=== NSE delivery-% accumulation (top) ===")
        print(sc.head(top).to_string(index=False))
        print(
            f"\n  {int((sc.Signal.str.contains('ACCUMULATION')).sum())} accumulating, "
            f"{int((sc.Signal=='DISTRIBUTION').sum())} distributing"
        )
    return sc


def main() -> int:
    ap = argparse.ArgumentParser(description="NSE delivery-% accumulation signal")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    if args.fetch:
        fetch(args.days)
    report(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
