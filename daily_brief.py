#!/usr/bin/env python3
# daily_brief.py
# ==============
# F-16 — a concise autonomous daily brief from the serving views: per-market regime
# (breadth, median return, tape) + the day's top momentum names. Written to
# results/daily_brief.md and (if GMAIL_* creds are set) emailed via send_mailer.
#
#   python3 daily_brief.py                 # write results/daily_brief.md
#   python3 daily_brief.py --mail          # also email it (needs GMAIL_* in .env)
#
# Scheduled by daily_memory.sh (or standalone cron/launchd — see footer).
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import datetime as _dt
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datalink
import serving_layer as sl

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
LIQ_GOOD = {"High", "Medium"}


def _tape(breadth: float, medret: float) -> str:
    if breadth > 0.55 and medret > 0:
        return "Bull"
    if breadth < 0.35 or medret < -5:
        return "Bear"
    return "Neutral"


def build_brief(markets: Optional[List[str]] = None) -> str:
    markets = markets or datalink.MARKETS
    asof = ""
    rows = []
    for m in markets:
        d = sl.serving(m)
        if d.empty:
            continue
        asof = max(asof, str(d["LastDate"].max()))
        breadth = float(d["Above200DMA"].mean()) if "Above200DMA" in d else 0.0
        medret = float(np.nanmedian(d.get("Ret252", pd.Series([np.nan]))))
        liq = d[d["Liquidity"].isin(LIQ_GOOD)] if "Liquidity" in d else d
        top = ", ".join(liq.sort_values("Ret252", ascending=False)["Symbol"].head(3))
        rows.append((m, len(d), breadth, medret, _tape(breadth, medret), top))

    lines = [f"# Daily Market Brief — {asof or _dt.date.today().isoformat()}", ""]
    if not rows:
        return "\n".join(lines + ["_No serving views built — run serving_layer --refresh._"])
    rows.sort(key=lambda r: (-(r[2]), -(r[3] if r[3] == r[3] else -1e9)))
    bull = [r[0] for r in rows if r[4] == "Bull"]
    bear = [r[0] for r in rows if r[4] == "Bear"]
    lines += [
        f"**Tape:** {len(bull)} Bull · {len(rows)-len(bull)-len(bear)} Neutral · {len(bear)} Bear",
        f"Bull: {', '.join(bull) or '—'}  |  Bear: {', '.join(bear) or '—'}",
        "",
        "| Market | Stocks | Breadth | Med 1y | Tape | Top momentum |",
        "|--------|-------:|--------:|-------:|------|--------------|",
    ]
    for m, n, b, mr, t, top in rows:
        mrs = f"{mr:.0f}%" if mr == mr else "n/a"
        lines.append(f"| {m} | {n} | {b:.0%} | {mrs} | {t} | {top} |")
    lines += ["", "_Educational/research only. Not investment advice._"]
    return "\n".join(lines)


def write_brief(markets: Optional[List[str]] = None, verbose: bool = True) -> Path:
    text = build_brief(markets)
    out = RESULTS / "daily_brief.md"
    out.write_text(text)
    if verbose:
        print(text)
        print(f"\n  → {out}")
    return out


def mail_brief(markets: Optional[List[str]] = None) -> bool:
    """Email the brief — delegates to the full morning mailer (correct send signature)."""
    try:
        import morning_mailer

        return morning_mailer.send(markets, verbose=False)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Autonomous daily market brief")
    ap.add_argument("--market", nargs="*")
    ap.add_argument("--mail", action="store_true", help="also email (needs GMAIL_* creds)")
    args = ap.parse_args()
    write_brief(args.market)
    if args.mail:
        print("  mailed ✓" if mail_brief(args.market) else "  mail skipped (no mailer/creds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ── Scheduling (append to daily_memory.sh, or standalone) ───────────────────────
# cron (evening, weekdays):
#   30 19 * * 1-5 cd /path/to/repo && python3 daily_brief.py --mail >> ~/brief.log 2>&1
# macOS launchd: add a StartCalendarInterval job running `python3 daily_brief.py --mail`.
