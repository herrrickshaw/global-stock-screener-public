#!/usr/bin/env python3
# decision_engine.py
# ==================
# A BUY / HOLD / SELL model built the way investors actually analyse a stock — and,
# critically, one that flags structurally IMPAIRED names that are effectively never
# going to recover (value traps / falling knives), instead of calling them a bargain
# just because they look "cheap".
#
# Four factors people typically weigh (each scored 0–100):
#   • VALUATION  — P/E vs the market median (cheap is good; losses score low)
#   • MOMENTUM   — 200-DMA position, golden/death cross, 3/6/12-mo returns, dist-to-high
#   • QUALITY    — ROE, ROCE, Piotroski, debt/equity, profit growth (when available)
#   • RECOVERY   — odds a beaten-down name comes back (LOW = likely permanently impaired)
#
# Impairment override (the "never recover" guard): if several structural breaks line
# up — deep multi-year drawdown, below 200-DMA, death cross, loss-making, collapsed
# liquidity — the stock is tagged SELL · IMPAIRED regardless of how cheap it screens.
#
#   from decision_engine import decide
#   decide(serving_row_dict, pe=12.3, med_pe=22, fundamentals={...})
#
# ⚠️ Research/education only. A mechanical, explainable model — NOT investment advice.

from __future__ import annotations

from typing import Optional

import numpy as np


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def _g(d: dict, k: str, default=None):
    v = d.get(k, default)
    if v is None:
        return default
    if isinstance(v, float) and np.isnan(v):
        return default
    return v


# ── individual factor scores (0–100, higher = more attractive) ───────────────────
def valuation_score(pe: Optional[float], med_pe: float) -> float:
    if pe is None:
        return 40.0  # unknown earnings → neutral-low
    if pe <= 0:
        return 10.0  # loss-making → poor value
    r = pe / med_pe if med_pe else 1.0
    if r < 0.6:
        return 92.0
    if r < 0.8:
        return 78.0
    if r < 1.1:
        return 58.0
    if r < 1.5:
        return 38.0
    if r < 2.5:
        return 20.0
    return 8.0


def momentum_score(row: dict) -> float:
    s = 50.0
    s += 15 if _g(row, "Above200DMA") else -15
    s += 8 if _g(row, "GoldenCross") else -8
    r252, r63 = _g(row, "Ret252", 0.0), _g(row, "Ret63", 0.0)
    s += _clamp(r252 / 4, -20, 20)  # 1y return
    s += _clamp(r63 / 3, -12, 12)  # 3mo return
    pfh = _g(row, "PctFromHigh", -50.0)  # closer to 52w high = stronger
    s += _clamp((pfh + 20) / 2, -15, 12)
    # blend Trendlyne DVM Momentum score (0–100) when supplied on the row
    dvm_m = _g(row, "dvm_momentum")
    if dvm_m is not None:
        s = 0.6 * s + 0.4 * _clamp(dvm_m)
    return _clamp(s)


def quality_score(fund: Optional[dict]) -> Optional[float]:
    if not fund:
        return None
    s, n = 0.0, 0
    roe = _g(fund, "roe")
    if roe is not None:
        s += _clamp(50 + roe, 0, 100)
        n += 1
    roce = _g(fund, "roce")
    if roce is not None:
        s += _clamp(50 + roce, 0, 100)
        n += 1
    pio = _g(fund, "piotroski")
    if pio is not None:
        s += pio / 9 * 100
        n += 1
    de = _g(fund, "debt_to_equity")
    if de is not None:
        s += _clamp(100 - de * 40)  # D/E 0→100, 2.5→0
        n += 1
    pg = _g(fund, "profit_growth")
    if pg is not None:
        s += _clamp(50 + pg, 0, 100)
        n += 1
    # Trendlyne DVM Durability score (0–100) is a ready-made quality gauge
    dur = _g(fund, "dvm_durability")
    if dur is not None:
        s += _clamp(dur)
        n += 1
    return _clamp(s / n) if n else None


def recovery_score(row: dict, pe: Optional[float], fund: Optional[dict]) -> float:
    """Odds a beaten-down name recovers. Low = likely permanently impaired."""
    s = 70.0
    close, sma200 = _g(row, "Close"), _g(row, "SMA200")
    if close and sma200:
        gap = close / sma200 - 1
        s += _clamp(gap * 120, -35, 15)  # far below 200-DMA → big penalty
    if not _g(row, "GoldenCross"):
        s -= 12  # death cross
    r252 = _g(row, "Ret252", 0.0)
    s += _clamp(r252 / 3, -30, 10)
    pfh = _g(row, "PctFromHigh", 0.0)
    if pfh < -60:
        s -= 25  # deep, prolonged drawdown
    elif pfh < -40:
        s -= 12
    pfl = _g(row, "PctFromLow", 100.0)
    if pfl < 8:
        s -= 12  # scraping 52-week lows, no bounce
    roe = _g(fund, "roe") if fund else None
    if (pe is not None and pe <= 0) or (roe is not None and roe < 0):
        s -= 18  # losing money
    if _g(row, "Liquidity") == "Low":
        s -= 10
    return _clamp(s)


# ── impairment ("never recover") detector ────────────────────────────────────────
def is_impaired(row: dict, pe: Optional[float], fund: Optional[dict], rec: float) -> bool:
    below200 = not _g(row, "Above200DMA", False)
    death = not _g(row, "GoldenCross", False)
    r252 = _g(row, "Ret252", 0.0)
    pfh = _g(row, "PctFromHigh", 0.0)
    roe = _g(fund, "roe") if fund else None
    loss = (pe is not None and pe <= 0) or (roe is not None and roe < 0)
    illiquid = _g(row, "Liquidity") == "Low"

    # structural break patterns that rarely recover
    broken_trend = below200 and death and r252 < -40 and pfh < -55
    bleeding = loss and below200 and r252 < -30
    collapsed_illiquid = illiquid and r252 < -50
    return bool(rec < 35 and (broken_trend or bleeding or collapsed_illiquid))


# ── the decision ─────────────────────────────────────────────────────────────────
WEIGHTS = {"valuation": 0.35, "momentum": 0.30, "quality": 0.20, "recovery": 0.15}


def decide(
    row: dict, pe: Optional[float], med_pe: float = 20.0, fundamentals: Optional[dict] = None
) -> dict:
    fund = fundamentals or {}
    # surface Trendlyne DVM Momentum onto the row so momentum_score can blend it
    if _g(fund, "dvm_momentum") is not None and _g(row, "dvm_momentum") is None:
        row = {**row, "dvm_momentum": fund["dvm_momentum"]}
    val = valuation_score(pe, med_pe)
    mom = momentum_score(row)
    qual = quality_score(fundamentals)
    rec = recovery_score(row, pe, fundamentals)

    # composite (renormalise weights if quality is unavailable)
    comps = {"valuation": val, "momentum": mom, "quality": qual, "recovery": rec}
    w = {k: v for k, v in WEIGHTS.items() if comps[k] is not None}
    tot = sum(w.values())
    composite = sum(comps[k] * (w[k] / tot) for k in w)

    flags, reasons = [], []
    if is_impaired(row, pe, fundamentals, rec):
        pfh = _g(row, "PctFromHigh", 0.0)
        flags.append("IMPAIRED")
        reasons.append(
            f"structurally broken — {pfh:.0f}% from high, below 200-DMA, "
            + ("loss-making, " if (pe is not None and pe <= 0) else "")
            + "unlikely to recover"
        )
        signal = "SELL"
    else:
        if composite >= 64:
            signal = "BUY"
        elif composite <= 42:
            signal = "SELL"
        else:
            signal = "HOLD"
        # explain the main drivers
        if pe is not None and pe > 0:
            rel = pe / med_pe if med_pe else 1
            reasons.append(
                f"P/E {pe:.0f} {'cheap' if rel < 0.8 else 'rich' if rel > 1.5 else 'fair'} "
                f"vs median {med_pe:.0f}"
            )
        elif pe is not None:
            reasons.append("loss-making (no P/E)")
        reasons.append("uptrend" if _g(row, "Above200DMA") else "below 200-DMA")
        if qual is not None:
            reasons.append(f"quality {qual:.0f}/100")
        if rec < 45:
            flags.append("WEAK-RECOVERY")

    # promoter-pledge risk control (applies to any non-impaired signal): pledged
    # promoter shares are a governance/liquidation risk — flag, and veto a BUY at
    # high pledge (a cheap stock with heavily-pledged promoters is a value trap risk).
    pledge = _g(fund, "promoter_pledge")
    if pledge is not None and pledge > 25:
        flags.append(f"PLEDGE-{int(pledge)}%")
        if signal == "BUY" and pledge > 50:
            signal = "HOLD"
            reasons.append(f"high promoter pledge {pledge:.0f}% — de-risked BUY→HOLD")

    return {
        "signal": signal,
        "composite": round(composite, 1),
        "valuation": round(val, 1),
        "momentum": round(mom, 1),
        "quality": None if qual is None else round(qual, 1),
        "recovery": round(rec, 1),
        "flags": flags,
        "rationale": "; ".join(reasons),
    }
