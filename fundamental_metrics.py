#!/usr/bin/env python3
# fundamental_metrics.py
# ======================
# Compute the FUNDAMENTAL Screener.in screen formulas on real financial data — the
# screens screen_metrics.py can't do from price alone (Piotroski, Magic Formula,
# ROCE, ROE, debt-free, FCF yield, dividend yield, growth, coffee-can …).
#
# Fundamentals feed (public/authenticated):
#   US → SEC EDGAR companyfacts (sec_fundamentals) — free, public, works offline-ish.
#   IN → screener.in authenticated export (screener_in_auth) when SCREENER_EMAIL /
#        SCREENER_PASSWORD are set; else falls back to the cached public screen
#        memberships (public_screens.py).
# Market cap / earnings & FCF yields use the latest price from the OHLCV DB × shares.
#
#   python3 fundamental_metrics.py --market US --limit 40      # fetch + run on 40 liquid US names
#   python3 fundamental_metrics.py --market US --screen piotroski_9 --limit 60
#
# SEC EDGAR asks for polite rate limits, so --limit caps how many symbols are fetched
# per run (default 40); results are cached to cache_seed/fundamentals/<MKT>.parquet
# and reused, so coverage grows across runs.
#
# ⚠️ Research/education only. Not advice.

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import datalink

CACHE = Path(__file__).parent / "cache_seed" / "fundamentals"
CACHE.mkdir(parents=True, exist_ok=True)


# ── fundamentals acquisition ─────────────────────────────────────────────────────
def _fetch_one(market: str, sym: str) -> dict:
    if market == "US":
        from sec_fundamentals import fundamentals

        return fundamentals(sym) or {}
    if market == "IN":
        try:
            from screener_in_auth import company_financials

            return company_financials(sym) or {}
        except Exception:
            return {}
    return {}


def _liquid_symbols(market: str, n: int) -> List[str]:
    import serving_layer as sl

    df = sl.serving(market)
    if df.empty:
        return []
    if "Turnover_USD" in df:
        df = df.sort_values("Turnover_USD", ascending=False)
    return df["Symbol"].head(n).tolist()


def _prices(market: str) -> Dict[str, float]:
    import serving_layer as sl

    df = sl.serving(market)
    return dict(zip(df["Symbol"], df["Close"])) if not df.empty else {}


def load_fundamentals(market: str, limit: int = 40, verbose: bool = True) -> pd.DataFrame:
    """Fetch fundamentals for the top-`limit` liquid names not yet cached, merge with
    the cache, and return a per-symbol fundamentals frame (+ price/market cap)."""
    path = CACHE / f"{market}.parquet"
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(cached["Symbol"]) if not cached.empty else set()
    want = [s for s in _liquid_symbols(market, limit * 3) if s not in have][:limit]

    rows = []
    for i, sym in enumerate(want):
        f = _fetch_one(market, sym)
        if f:
            f["Symbol"] = sym
            rows.append(f)
        if market == "US":
            time.sleep(0.12)  # SEC politeness (<10 req/s)
        if verbose and (i + 1) % 20 == 0:
            print(f"  fetched {i+1}/{len(want)} …")
    fresh = pd.DataFrame(rows)
    out = pd.concat([cached, fresh], ignore_index=True) if not fresh.empty else cached
    if not out.empty:
        out = out.drop_duplicates("Symbol", keep="last")
        out.to_parquet(path, index=False)
    if verbose:
        print(f"  fundamentals[{market}]: +{len(fresh)} fetched, {len(out)} total cached")
    # attach market cap from latest price × shares
    px = _prices(market)
    if not out.empty:
        out["price"] = out["Symbol"].map(px)
        if "shares" in out:
            out["market_cap"] = out["price"] * out["shares"]
    return out


# ── fundamental screen formulas (screener.in query semantics) ────────────────────
def _piotroski(r) -> int:
    """Piotroski F-score (0–9). A point is awarded only when the inputs it needs are
    present — missing data never earns a free point (avoids score inflation)."""
    def has(*ks):
        return all(r.get(k) is not None and not (isinstance(r.get(k), float) and np.isnan(r[k]))
                   for k in ks)

    s = 0
    s += has("net_income") and r["net_income"] > 0
    s += has("roa") and r["roa"] > 0
    s += has("cfo") and r["cfo"] > 0
    s += has("cfo", "net_income") and r["cfo"] > r["net_income"]                # accrual
    s += has("debt_to_assets", "debt_to_assets_prev") and r["debt_to_assets"] < r["debt_to_assets_prev"]
    s += has("current_ratio", "current_ratio_prev") and r["current_ratio"] > r["current_ratio_prev"]
    s += has("shares", "shares_prev") and r["shares"] <= r["shares_prev"]       # no dilution
    s += has("gross_margin", "gross_margin_prev") and r["gross_margin"] > r["gross_margin_prev"]
    s += has("asset_turnover", "asset_turnover_prev") and r["asset_turnover"] > r["asset_turnover_prev"]
    return int(s)


def _roce(r):
    # ROCE ≈ EBIT / total assets (capital employed proxy) × 100
    ebit, cap = r.get("ebit"), r.get("total_assets") or r.get("assets")
    if ebit and cap:
        return ebit / cap * 100
    # fallback: ROA is a lower bound proxy when assets absent
    return r.get("roa")


def _first_val(v):
    """First element of a history field, robust to list/np.ndarray/scalar/NaN."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0] if len(v) else None
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def _earnings_yield(r):
    ebit, mc = r.get("ebit"), r.get("market_cap")
    return ebit / mc * 100 if ebit and mc else None


def _fcf_yield(r):
    cfo = r.get("cfo")
    capex = _first_val(r.get("capex_history"))
    mc = r.get("market_cap")
    if cfo is None or mc in (None, 0) or (isinstance(mc, float) and np.isnan(mc)):
        return None
    fcf = cfo - (capex or 0)
    return fcf / mc * 100


def _dividend_yield(r):
    div = _first_val(r.get("dividend_history"))
    mc = r.get("market_cap")
    if not div or not mc or (isinstance(mc, float) and np.isnan(mc)):
        return None
    return abs(div) / mc * 100


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    recs = df.to_dict("records")
    df["piotroski"] = [_piotroski(r) for r in recs]
    df["roce"] = [_roce(r) for r in recs]
    df["earnings_yield"] = [_earnings_yield(r) for r in recs]
    df["fcf_yield"] = [_fcf_yield(r) for r in recs]
    df["div_yield"] = [_dividend_yield(r) for r in recs]
    return df


FUND_SCREENS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "piotroski_9":     lambda d: d["piotroski"] >= 8,
    "high_roce":       lambda d: d["roce"] > 20,
    "high_roe":        lambda d: d.get("roe", pd.Series(dtype=float)) > 20,
    "debt_free":       lambda d: d.get("debt_to_equity", pd.Series(dtype=float)) < 0.1,
    "low_debt_equity": lambda d: d.get("debt_to_equity", pd.Series(dtype=float)) < 0.3,
    "magic_formula":   lambda d: (d["earnings_yield"] > 8) & (d["roce"] > 15),
    "fcf_yield":       lambda d: d["fcf_yield"] > 5,
    "high_dividend":   lambda d: d["div_yield"] > 3,
    "eps_growth":      lambda d: d.get("eps_growth", pd.Series(dtype=float)) > 15,
    "coffee_can":      lambda d: (d.get("roe", pd.Series(dtype=float)) > 15)
                                 & (d.get("debt_to_equity", pd.Series(dtype=float)) < 0.5),
}


def run_all(market: str, limit: int = 40, verbose: bool = True) -> Dict[str, int]:
    df = load_fundamentals(market, limit=limit, verbose=verbose)
    if df.empty:
        if verbose:
            print(f"no fundamentals for {market} (US=SEC; IN needs SCREENER creds)")
        return {}
    df = _enrich(df)
    counts = {}
    if verbose:
        print(f"\n=== fundamental screens on {market} ({len(df)} names with financials) ===")
    for name, pred in FUND_SCREENS.items():
        try:
            m = pred(df).fillna(False)
        except Exception:
            m = pd.Series(False, index=df.index)
        counts[name] = int(m.sum())
        if verbose:
            ex = ", ".join(df.loc[m, "Symbol"].head(6))
            print(f"  {name:18} {counts[name]:>4}   {ex}")
    return counts


def run_screen(name: str, market: str, limit: int = 40, top: Optional[int] = 25) -> pd.DataFrame:
    df = _enrich(load_fundamentals(market, limit=limit))
    if df.empty or name not in FUND_SCREENS:
        return pd.DataFrame()
    out = df[FUND_SCREENS[name](df).fillna(False)]
    sort = "roce" if name in ("high_roce", "magic_formula") else "piotroski"
    if sort in out.columns:
        out = out.sort_values(sort, ascending=False)
    cols = [c for c in ["Symbol", "piotroski", "roce", "roe", "earnings_yield",
                        "fcf_yield", "div_yield", "debt_to_equity", "eps_growth"] if c in out.columns]
    out = out[cols]
    return out.head(top).reset_index(drop=True) if top else out.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run fundamental Screener.in formulas on real financials")
    ap.add_argument("--market", default="US")
    ap.add_argument("--screen", help="run one fundamental screen")
    ap.add_argument("--limit", type=int, default=40, help="max new symbols to fetch this run")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    if args.screen:
        df = run_screen(args.screen, args.market, limit=args.limit, top=args.top)
        print(f"\n{args.screen} on {args.market} — {len(df)} names:")
        print(df.round(2).to_string(index=False) if not df.empty else "  none / no fundamentals")
        return 0
    run_all(args.market, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
