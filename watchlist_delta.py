#!/usr/bin/env python3
# watchlist_delta.py
# ==================
# F-13 — "what changed on the watchlist today": snapshot the Buy/Hold/Sell decisions
# for a market's universe and diff against the previous snapshot, surfacing signal
# flips (e.g. HOLD→SELL), new BUYs, and new risk flags (IMPAIRED / PLEDGE spikes).
# Pairs with the CDC price-delta log; output feeds the daily brief / mailer.
#
#   python3 watchlist_delta.py --snapshot --market IN   # save today's decisions
#   python3 watchlist_delta.py --market IN              # report change vs last snapshot
#
# Snapshots: cache_seed/decision_snapshots/<MKT>_<date>.parquet (Symbol, Signal, Flags).
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import datetime as _dt
import warnings
from pathlib import Path
from typing import List, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
SNAP = HERE / "cache_seed" / "decision_snapshots"
SNAP.mkdir(parents=True, exist_ok=True)


def snapshot(market: str = "IN", top: int = 400, verbose: bool = True) -> Path:
    """Compute Buy/Hold/Sell for the market's top-liquid names and save a snapshot."""
    import portfolio_evaluator as pe
    import serving_layer as sl

    df = sl.serving(market)
    if df.empty:
        raise RuntimeError(f"no serving view for {market}")
    syms = (
        df.sort_values("Turnover_USD", ascending=False)["Symbol"].head(top).tolist()
        if "Turnover_USD" in df
        else df["Symbol"].head(top).tolist()
    )
    out = pe.evaluate(syms, market, fetch=False, verbose=False)[["Symbol", "Signal", "Flags"]]
    path = SNAP / f"{market}_{_dt.date.today().isoformat()}.parquet"
    out.to_parquet(path, index=False)
    if verbose:
        print(f"  snapshot[{market}]: {len(out)} decisions → {path.name}")
    return path


def _last_two(market: str) -> List[Path]:
    return sorted(SNAP.glob(f"{market}_*.parquet"))[-2:]


def diff(market: str = "IN") -> Tuple[pd.DataFrame, dict]:
    """Diff the two latest snapshots → (changes DataFrame, summary)."""
    snaps = _last_two(market)
    if len(snaps) < 2:
        return pd.DataFrame(columns=["Symbol", "From", "To", "Flags", "Change"]), {
            "note": "need 2 snapshots"
        }
    prev = pd.read_parquet(snaps[0]).set_index("Symbol")
    cur = pd.read_parquet(snaps[1]).set_index("Symbol")
    rank = {"BUY": 3, "HOLD": 2, "SELL": 1, "NO DATA": 0}
    rows = []
    for sym in cur.index:
        c = cur.loc[sym]
        p = prev.loc[sym] if sym in prev.index else None
        old = p["Signal"] if p is not None else "—"
        if old != c["Signal"]:
            change = "UPGRADE" if rank.get(c["Signal"], 0) > rank.get(old, 0) else "DOWNGRADE"
            rows.append(
                {
                    "Symbol": sym,
                    "From": old,
                    "To": c["Signal"],
                    "Flags": c.get("Flags", ""),
                    "Change": change,
                }
            )
        else:  # same signal but a new risk flag appeared
            newf = (
                set(str(c.get("Flags", "")).split()) - set(str(p["Flags"]).split())
                if p is not None
                else set()
            )
            risk = {
                f
                for f in newf
                if any(k in f for k in ("IMPAIRED", "PLEDGE", "DISTRIBUTION", "WEAK"))
            }
            if risk:
                rows.append(
                    {
                        "Symbol": sym,
                        "From": old,
                        "To": c["Signal"],
                        "Flags": " ".join(risk),
                        "Change": "NEW-FLAG",
                    }
                )
    d = pd.DataFrame(rows)
    summ = {
        "dates": f"{snaps[0].stem.split('_')[-1]} → {snaps[1].stem.split('_')[-1]}",
        "new_buys": (
            int((d["Change"].eq("UPGRADE") & d["To"].eq("BUY")).sum()) if not d.empty else 0
        ),
        "downgrades": int(d["Change"].eq("DOWNGRADE").sum()) if not d.empty else 0,
        "new_flags": int(d["Change"].eq("NEW-FLAG").sum()) if not d.empty else 0,
    }
    return d, summ


def report(market: str = "IN", verbose: bool = True) -> pd.DataFrame:
    d, s = diff(market)
    if verbose:
        if "note" in s:
            print(f"  {s['note']} — run --snapshot on two different days")
        else:
            print(f"=== Watchlist delta ({market}) {s['dates']} ===")
            print(
                f"  new BUYs {s['new_buys']} · downgrades {s['downgrades']} · new risk flags {s['new_flags']}"
            )
            if not d.empty:
                print(d.head(30).to_string(index=False))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="CDC watchlist decision-delta")
    ap.add_argument("--market", default="IN")
    ap.add_argument("--snapshot", action="store_true", help="save today's decisions")
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.market)
    report(args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
