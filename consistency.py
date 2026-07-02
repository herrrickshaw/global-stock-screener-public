#!/usr/bin/env python3
# consistency.py
# ==============
# F-14 — consistency metrics from a multi-year fundamentals series (the "consistent
# compounder" lens): fraction of years with growth, no drawdown years, trend slope.
# Source series come from Trendlyne/screener 10y sheets (best-effort); the SCORING
# is pure and tested.  Research/education only.
from __future__ import annotations

import argparse
import warnings
from typing import List

warnings.filterwarnings("ignore")


def growth_consistency(series: List[float]) -> float:
    """Share of year-over-year steps that grew (0..1)."""
    s = [x for x in series if x is not None]
    if len(s) < 2:
        return 0.0
    ups = sum(1 for a, b in zip(s, s[1:]) if b > a)
    return round(ups / (len(s) - 1), 3)


def positive_years(series: List[float]) -> float:
    """Share of years with a positive value (e.g. profit)."""
    s = [x for x in series if x is not None]
    return round(sum(1 for x in s if x > 0) / len(s), 3) if s else 0.0


def cagr(series: List[float]) -> float:
    s = [x for x in series if x is not None and x > 0]
    if len(s) < 2:
        return 0.0
    yrs = len(s) - 1
    return round(((s[-1] / s[0]) ** (1 / yrs) - 1) * 100, 2)


def compounder_score(profit: List[float], roe: List[float] | None = None) -> dict:
    """Composite: consistent + growing profit, optionally durable ROE."""
    gc, py, cg = growth_consistency(profit), positive_years(profit), cagr(profit)
    score = 40 * gc + 30 * py + min(30, max(0, cg))
    if roe:
        score = 0.8 * score + 0.2 * min(100, positive_years(roe) * 100)
    return {
        "growth_consistency": gc,
        "positive_years": py,
        "profit_cagr": cg,
        "score": round(min(100, score), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="10y fundamentals consistency")
    ap.add_argument("--profit", required=True, help="comma series, oldest→newest")
    args = ap.parse_args()
    ser = [float(x) for x in args.profit.split(",")]
    print(compounder_score(ser))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
