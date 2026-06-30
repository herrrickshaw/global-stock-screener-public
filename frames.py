#!/usr/bin/env python3
# frames.py
# =========
# Shared OHLCV frame helpers (DRY) — the long↔per-symbol conversions and the
# compact-parquet write were duplicated across screener_kit, bhavcopy_store,
# bhavcopy_history, fetch_market_ohlc and build_market_seeds. One place now.

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def long_to_dict(long: pd.DataFrame, min_bars: int = 1) -> Dict[str, pd.DataFrame]:
    """Long frame (Symbol/Date/OHLCV rows) → {symbol: Date-indexed OHLCV}."""
    if long is None or long.empty:
        return {}
    long = long.copy()
    long["Date"] = pd.to_datetime(long["Date"])
    out: Dict[str, pd.DataFrame] = {}
    for sym, g in long.groupby("Symbol"):
        frame = g.set_index("Date").sort_index()[OHLCV]
        if len(frame) >= min_bars:
            out[str(sym)] = frame
    return out


def to_long(hist: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{symbol: OHLCV df} → tidy long frame (float32 prices, int64 volume)."""
    parts = []
    for sym, d in hist.items():
        if d is None or d.empty:
            continue
        g = d.reset_index()
        g.columns = ["Date", *OHLCV][: len(g.columns)]
        for c in ("Open", "High", "Low", "Close"):
            g[c] = g[c].astype("float32")
        g["Volume"] = pd.to_numeric(g["Volume"], errors="coerce").fillna(0).astype("int64")
        g["Symbol"] = str(sym)
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["Date", *OHLCV, "Symbol"])
    return pd.concat(parts, ignore_index=True).sort_values(["Symbol", "Date"])


def write_seed(hist: Dict[str, pd.DataFrame], path: Path, level: int = 9) -> int:
    """Write {symbol: df} to a compact zstd parquet seed. Returns #symbols."""
    long = to_long(hist)
    if long.empty:
        return 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(path, compression="zstd", compression_level=level, index=False)
    return long["Symbol"].nunique()


def read_seed(path: Path, min_bars: int = 1) -> Dict[str, pd.DataFrame]:
    """Read a seed parquet back into {symbol: OHLCV df}."""
    p = Path(path)
    return long_to_dict(pd.read_parquet(p), min_bars) if p.exists() else {}
