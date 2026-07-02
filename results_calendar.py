#!/usr/bin/env python3
# results_calendar.py
# ===================
# F-15 — event-aware screening: know which stocks report results (or go ex-corp-action)
# soon, so screens can avoid pre-results uncertainty or catch post-results movers.
# Best-effort source (nsepython events); the date logic is pure + tested.
from __future__ import annotations

import argparse
import datetime as _dt
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")


def fetch_events(verbose: bool = True) -> Dict[str, str]:
    """{symbol: 'YYYY-MM-DD'} upcoming results/corp-action dates (best-effort)."""
    out: Dict[str, str] = {}
    try:
        import nsepython as nse

        ev = nse.nse_events() if hasattr(nse, "nse_events") else []
        for r in (ev.to_dict("records") if hasattr(ev, "to_dict") else (ev or [])):
            sym = str(r.get("symbol", "")).upper().strip()
            dt = str(r.get("date") or r.get("purpose_date") or "")[:10]
            if sym and dt:
                out[sym] = dt
    except Exception:
        pass
    if verbose:
        print(f"  events fetched: {len(out)}")
    return out


def _d(s: str) -> Optional[_dt.date]:
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def days_to_event(event_date: str, today: Optional[_dt.date] = None) -> Optional[int]:
    d = _d(event_date)
    return (d - (today or _dt.date.today())).days if d else None


def near_event(event_date: str, window: int = 5, today: Optional[_dt.date] = None) -> bool:
    """True if the event is within ±window days (the caution/opportunity zone)."""
    n = days_to_event(event_date, today)
    return n is not None and abs(n) <= window


def upcoming(
    events: Dict[str, str], within: int = 7, today: Optional[_dt.date] = None
) -> List[str]:
    """Symbols reporting within `within` days (forward)."""
    out = []
    for sym, dt in events.items():
        n = days_to_event(dt, today)
        if n is not None and 0 <= n <= within:
            out.append(sym)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Results / corp-action calendar")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--within", type=int, default=7)
    args = ap.parse_args()
    ev = fetch_events() if args.fetch else {}
    print("reporting soon:", upcoming(ev, args.within) or "(run --fetch; source may be throttled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
