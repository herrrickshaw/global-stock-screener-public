#!/usr/bin/env python3
# portfolio_evaluator.py
# ======================
# Upload your holdings / broker tax-P&L statement (or just paste tickers) and get a
# BUY / HOLD / SELL call for each, validated against the system's screens and driven
# by P/E (valuation) relative to the market.
#
# Inputs (any of):
#   • a CSV/XLSX file — a broker capital-gains / holdings statement; the ticker
#     column is auto-detected (Symbol / Ticker / Scrip / Instrument / ISIN / Stock)
#   • a comma/space/newline list of tickers
#
# For each holding it pulls the central data (serving view + fundamentals cache),
# computes P/E, counts how many screens it clears, and decides:
#
#   BUY   — undervalued (P/E well below market median) AND in an uptrend / clears
#           quality screens
#   SELL  — overvalued (P/E well above market median) OR loss-making in a downtrend
#   HOLD  — everything in between
#
#   python3 portfolio_evaluator.py --tickers RELIANCE,TCS,INFY --market IN
#   python3 portfolio_evaluator.py --file ~/Downloads/holdings.xlsx --market IN
#   → prints a decision table + saves results/portfolio_review.csv
#
# ⚠️ Research/education only. A mechanical valuation screen, NOT investment advice.

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import serving_layer as sl

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

TICKER_COLS = ["symbol", "ticker", "scrip", "instrument", "stock", "security", "name", "isin"]


# ── input parsing ────────────────────────────────────────────────────────────────
def parse_tickers(text: str) -> List[str]:
    toks = re.split(r"[,\s;]+", text.strip().upper())
    return [t for t in dict.fromkeys(toks) if t]


def parse_file(path: str) -> List[str]:
    """Extract tickers from a holdings / tax-P&L CSV or XLSX (auto-detect the column)."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        # broker statements often have a title banner; try a few header rows
        for hdr in (0, 1, 2, 3):
            try:
                df = pd.read_excel(p, header=hdr)
                col = _ticker_col(df)
                if col:
                    return _clean_syms(df[col])
            except Exception:
                continue
        return []
    df = pd.read_csv(p, on_bad_lines="skip")
    col = _ticker_col(df)
    return _clean_syms(df[col]) if col else []


def _ticker_col(df: pd.DataFrame) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for k in TICKER_COLS:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None


def _clean_syms(series: pd.Series) -> List[str]:
    out = []
    for v in series.dropna().astype(str):
        s = v.strip().upper()
        # strip common exchange suffixes / ISIN → keep ticker-like tokens
        s = re.sub(r"\.(NS|BO|NSE|BSE)$", "", s)
        if re.fullmatch(r"[A-Z0-9&.\-]{1,20}", s) and not s.isdigit():
            out.append(s)
    return list(dict.fromkeys(out))


# ── valuation + screens ──────────────────────────────────────────────────────────
def _fundamentals(market: str) -> pd.DataFrame:
    p = Path(__file__).parent / "cache_seed" / "fundamentals" / f"{market}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _pe_map(market: str, serving: pd.DataFrame) -> Dict[str, float]:
    """P/E per symbol: prefer a stored ratio, else price/EPS, else market_cap/NI."""
    f = _fundamentals(market)
    price = dict(zip(serving["Symbol"], serving["Close"])) if not serving.empty else {}
    pe: Dict[str, float] = {}
    if not f.empty:
        for _, r in f.iterrows():
            sym = r["Symbol"]
            val = None
            for key in ("pe", "P/E", "pe_ttm"):
                if key in r and pd.notna(r.get(key)) and r.get(key) not in (0, ""):
                    val = float(r[key])
                    break
            if val is None and r.get("eps_ttm") and price.get(sym):
                val = price[sym] / r["eps_ttm"] if r["eps_ttm"] else None
            if val is None and r.get("net_income") and r.get("shares") and price.get(sym):
                mc = price[sym] * r["shares"]
                val = mc / r["net_income"] if r["net_income"] else None
            if val is not None and np.isfinite(val):
                pe[sym] = round(val, 1)
    return pe


def _screens_passed(market: str, serving: pd.DataFrame) -> Dict[str, list]:
    """{symbol: [screens it clears]} from the price/technical + popular screens."""
    hits: Dict[str, set] = {}

    def add(name, syms):
        for s in syms:
            hits.setdefault(s, set()).add(name)

    try:
        import screen_metrics as sm

        for name in (
            "golden_crossover",
            "companies_creating_new_high",
            "breakout_stocks",
            "multibagger_momentum",
        ):
            df = sm.run_screen(name, market)
            if not df.empty:
                add(name, df["Symbol"])
    except Exception:
        pass
    try:
        from validation import known_universe

        add("popular", known_universe(market))
    except Exception:
        pass
    return {s: sorted(v) for s, v in hits.items()}


def _decide(pe: Optional[float], med_pe: float, n_screens: int, uptrend: bool) -> tuple:
    """BUY/HOLD/SELL driven by P/E vs the market median, tempered by screens+trend."""
    if pe is None:
        base = "HOLD" if uptrend else "SELL"
        return base, "no P/E (no earnings data); " + ("uptrend" if uptrend else "downtrend")
    if pe <= 0:
        return (
            "SELL",
            f"loss-making (P/E {pe}); " + ("hold if turning up" if uptrend else "downtrend"),
        )
    cheap, rich = 0.8 * med_pe, 1.5 * med_pe
    if pe < cheap and (uptrend or n_screens >= 1):
        return (
            "BUY",
            f"undervalued (P/E {pe} < {cheap:.0f} = 0.8×median {med_pe:.0f}), {n_screens} screens, {'uptrend' if uptrend else 'flat'}",
        )
    if pe > rich or (not uptrend and pe > med_pe):
        return (
            "SELL",
            f"expensive (P/E {pe} vs median {med_pe:.0f}){' + downtrend' if not uptrend else ''}",
        )
    return "HOLD", f"fair value (P/E {pe} near median {med_pe:.0f}), {n_screens} screens"


def evaluate(tickers: List[str], market: str = "IN", verbose: bool = True) -> pd.DataFrame:
    serving = sl.serving(market)
    if serving.empty:
        raise RuntimeError(f"no serving view for {market} — run serving_layer.py --refresh")
    serving = serving.set_index("Symbol")
    pe_map = _pe_map(market, serving.reset_index())
    med_pe = float(np.median([v for v in pe_map.values() if 0 < v < 200])) if pe_map else 20.0
    screens = _screens_passed(market, serving.reset_index())

    rows = []
    for t in tickers:
        if t not in serving.index:
            rows.append(
                {
                    "Symbol": t,
                    "Signal": "NO DATA",
                    "PE": None,
                    "Screens": 0,
                    "Rationale": "not in our universe/cache",
                }
            )
            continue
        row = serving.loc[t]
        pe = pe_map.get(t)
        scr = screens.get(t, [])
        uptrend = bool(row.get("Above200DMA", False))
        sig, why = _decide(pe, med_pe, len(scr), uptrend)
        rows.append(
            {
                "Symbol": t,
                "Signal": sig,
                "PE": pe,
                "Close": round(float(row["Close"]), 2),
                "Ret252": round(float(row.get("Ret252", np.nan)), 1),
                "Liquidity": row.get("Liquidity", ""),
                "Screens": len(scr),
                "Passed": ",".join(scr),
                "Rationale": why,
            }
        )
    out = pd.DataFrame(rows)
    order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NO DATA": 3}
    out = out.sort_values("Signal", key=lambda s: s.map(order)).reset_index(drop=True)
    if verbose:
        print(f"\n=== Portfolio review ({market}) — market median P/E ≈ {med_pe:.0f} ===")
        cols = [
            c
            for c in ["Symbol", "Signal", "PE", "Close", "Ret252", "Screens", "Rationale"]
            if c in out.columns
        ]
        print(out[cols].to_string(index=False))
        print(
            "\n  "
            + "  ".join(
                f"{k}:{int((out['Signal']==k).sum())}" for k in ("BUY", "HOLD", "SELL", "NO DATA")
            )
        )
    out.to_csv(RESULTS / "portfolio_review.csv", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate holdings against screens → Buy/Hold/Sell by P/E"
    )
    ap.add_argument("--tickers", help="comma/space list of tickers")
    ap.add_argument("--file", help="holdings / tax-P&L CSV or XLSX")
    ap.add_argument("--market", default="IN")
    args = ap.parse_args()

    if args.file:
        tickers = parse_file(args.file)
        print(f"  parsed {len(tickers)} tickers from {args.file}")
    elif args.tickers:
        tickers = parse_tickers(args.tickers)
    else:
        print("provide --tickers or --file")
        return 1
    if not tickers:
        print("no tickers found")
        return 1
    evaluate(tickers, args.market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
