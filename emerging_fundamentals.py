#!/usr/bin/env python3
"""
emerging_fundamentals.py
========================
Fundamentals source for markets yfinance does NOT cover with financial
statements — Borsa Istanbul (.IS), Prague (.PR), Budapest (.BD) — plus any
other ticker whose Yahoo statements come back empty.

Why this exists
---------------
`full_european_market_scan.py` computes Piotroski + Coffee Can from
yfinance income/balance/cash-flow statements. For Turkish/Czech/Hungarian
equities Yahoo returns `no_financial_statements`, so those names could only
ever appear as Darvas momentum signals. This module fills that gap by pulling
statements from a keyed provider and computing the SAME F-score / Coffee-Can
fields (identical keys), so the scan can merge the result transparently.

Providers (first one with a key + data wins)
--------------------------------------------
  1. Financial Modeling Prep   env FMP_API_KEY        (widest .IS/.PR/.BD coverage)
  2. Twelve Data               env TWELVEDATA_KEY      (fallback)

Both are free-tier friendly (FMP 250 req/day, Twelve Data 800/day). Get a key:
  FMP:         https://site.financialmodelingprep.com/developer/docs  (Free plan)
  Twelve Data: https://twelvedata.com/pricing                         (Basic/free)
  export FMP_API_KEY=xxxx   (or)   export TWELVEDATA_KEY=xxxx

With NO key set, get_fundamentals() returns None and the caller keeps the
momentum-only behaviour — nothing breaks, the section just stays empty.

Public API
----------
  get_fundamentals(ticker) -> dict | None
      Returns the scan-compatible dict (f_score, cc_qualifies, cc_score,
      pe_ratio, sector, cc_rev_cagr, cc_roce_avg, cc_debt_equity, source, error)
      or None if no provider can serve the ticker.
  score_from_statements(inc, bal, cf, profile) -> dict
      Pure scorer (provider-agnostic) — unit-testable without network.
"""
import os
import time

import requests

_TIMEOUT = 20
_UA = {"User-Agent": "Mozilla/5.0 (compatible; market-report/1.0)"}
_LAST_CALL = {"t": 0.0}
_MIN_INTERVAL = 0.20  # be polite to free tiers


def _throttle():
    dt = time.time() - _LAST_CALL["t"]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _LAST_CALL["t"] = time.time()


# ─────────────────────────────── the scorer ───────────────────────────────
# Works on NORMALISED statement rows: each is a list of yearly dicts, newest
# first, with these canonical keys (providers map their own names into these):
#   inc:  net_income, revenue, gross_profit, ebit, shares
#   bal:  total_assets, long_term_debt, current_assets, current_liabilities
#   cf:   operating_cash_flow, free_cash_flow, capex
#   profile: {pe, sector}
def _g(row, key):
    v = row.get(key) if row else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def score_from_statements(inc, bal, cf, profile):
    """Compute Piotroski F-score (0–9) + Coffee-Can (0–6) from normalised
    statements. Mirrors full_european_market_scan.fundamental_scan exactly so
    downstream scoring/PE-zone logic is identical across data sources."""
    res = {
        "f_score": None,
        "cc_qualifies": "FAIL",
        "cc_score": None,
        "cc_rev_cagr": None,
        "cc_roce_avg": None,
        "cc_debt_equity": None,
        "pe_ratio": None,
        "sector": None,
        "error": None,
    }
    profile = profile or {}
    res["sector"] = profile.get("sector")
    pe = profile.get("pe")
    res["pe_ratio"] = round(float(pe), 2) if (pe not in (None, 0) and float(pe) > 0) else None

    if not inc or not bal:
        res["error"] = "no_financial_statements"
        return res

    i0 = inc[0]
    i1 = inc[1] if len(inc) > 1 else {}
    b0 = bal[0]
    b1 = bal[1] if len(bal) > 1 else {}
    c0 = cf[0] if cf else {}

    # ── Piotroski F-score (9 points) ──
    try:
        ni0, a0 = _g(i0, "net_income"), _g(b0, "total_assets")
        ni1, a1 = _g(i1, "net_income"), _g(b1, "total_assets")
        roa0 = (ni0 / a0) if (ni0 is not None and a0) else None
        roa1 = (ni1 / a1) if (ni1 is not None and a1) else None
        ocf0 = _g(c0, "operating_cash_flow")

        f1 = 1 if (roa0 and roa0 > 0) else 0
        f2 = 1 if (ocf0 and ocf0 > 0) else 0
        f3 = 1 if (roa0 is not None and roa1 is not None and roa0 > roa1) else 0
        f4 = 1 if (ocf0 and a0 and roa0 is not None and (ocf0 / a0) > roa0) else 0

        ltd0 = _g(b0, "long_term_debt") or 0
        ltd1 = _g(b1, "long_term_debt") or 0
        f5 = 1 if (a0 and a1 and (ltd0 / a0) < (ltd1 / a1)) else 0

        ca0, cl0 = _g(b0, "current_assets"), _g(b0, "current_liabilities")
        ca1, cl1 = _g(b1, "current_assets"), _g(b1, "current_liabilities")
        f6 = 1 if (ca0 and cl0 and ca1 and cl1 and (ca0 / cl0) > (ca1 / cl1)) else 0

        sh0, sh1 = _g(i0, "shares"), _g(i1, "shares")
        f7 = (1 if sh0 <= sh1 else 0) if (sh0 and sh1) else 1

        rev0, gp0 = _g(i0, "revenue"), _g(i0, "gross_profit")
        rev1, gp1 = _g(i1, "revenue"), _g(i1, "gross_profit")
        f8 = 1 if (gp0 and rev0 and gp1 and rev1 and (gp0 / rev0) > (gp1 / rev1)) else 0
        f9 = 1 if (rev0 and a0 and rev1 and a1 and (rev0 / a0) > (rev1 / a1)) else 0

        res["f_score"] = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    except Exception:
        pass

    # ── Coffee Can (6 points) ──
    try:
        c = {}
        revs = [_g(x, "revenue") for x in inc if _g(x, "revenue")]
        cagr = (
            ((revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1) * 100
            if len(revs) >= 2 and revs[-1] > 0
            else None
        )
        c["C1"] = 1 if (cagr and cagr > 10) else 0

        roce_l = []
        for i, b in zip(inc, bal):
            ebit = _g(i, "ebit")
            ta = _g(b, "total_assets")
            cl = _g(b, "current_liabilities")
            if ebit is not None and ta and cl is not None and (ta - cl) > 0:
                roce_l.append(ebit / (ta - cl) * 100)
        avg_roce = sum(roce_l) / len(roce_l) if roce_l else None
        c["C2"] = 1 if (avg_roce and avg_roce > 15) else 0

        de = None
        td0 = _g(b0, "long_term_debt")
        eq0 = (_g(b0, "total_assets") or 0) - (_g(b0, "current_liabilities") or 0)  # rough proxy
        if td0 is not None and eq0:
            de = td0 / eq0
        c["C3"] = 1 if (de is not None and de < 1.0) else 0

        gps = [_g(x, "gross_profit") for x in inc]
        revs2 = [_g(x, "revenue") for x in inc]
        margins = [g / r for g, r in zip(gps, revs2) if g is not None and r]
        c["C4"] = 1 if (len(margins) >= 2 and margins[0] > margins[-1]) else 0

        nis = [_g(x, "net_income") for x in inc if _g(x, "net_income") is not None]
        c["C5"] = 1 if (nis and all(n > 0 for n in nis)) else 0

        fcf0 = _g(c0, "free_cash_flow")
        if fcf0 is None:
            ocf = _g(c0, "operating_cash_flow")
            cap = _g(c0, "capex")
            fcf0 = (ocf - abs(cap)) if (ocf is not None and cap is not None) else None
        c["C6"] = 1 if (fcf0 is not None and fcf0 > 0) else 0

        res.update(
            {
                "cc_qualifies": "PASS" if sum(c.values()) == len(c) else "FAIL",
                "cc_score": f"{sum(c.values())}/{len(c)}",
                "cc_rev_cagr": round(cagr, 2) if cagr else None,
                "cc_roce_avg": round(avg_roce, 2) if avg_roce else None,
                "cc_debt_equity": round(de, 2) if de else None,
            }
        )
    except Exception:
        pass

    return res


# ─────────────────────────────── providers ───────────────────────────────
class _FMP:
    name = "FMP"
    base = "https://financialmodelingprep.com/api/v3"

    def __init__(self):
        self.key = os.environ.get("FMP_API_KEY", "").strip()

    @property
    def available(self):
        return bool(self.key)

    def _get(self, path, **params):
        params["apikey"] = self.key
        _throttle()
        r = requests.get(f"{self.base}/{path}", params=params, headers=_UA, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        j = r.json()
        if isinstance(j, dict) and j.get("Error Message"):
            return None
        return j

    def fetch(self, ticker):
        inc_raw = self._get(f"income-statement/{ticker}", period="annual", limit=5)
        bal_raw = self._get(f"balance-sheet-statement/{ticker}", period="annual", limit=5)
        cf_raw = self._get(f"cash-flow-statement/{ticker}", period="annual", limit=5)
        if not inc_raw or not bal_raw:
            return None
        prof = self._get(f"profile/{ticker}") or []
        quote = self._get(f"quote/{ticker}") or []
        sector = prof[0].get("sector") if prof else None
        pe = quote[0].get("pe") if quote else None
        inc = [
            {
                "net_income": r.get("netIncome"),
                "revenue": r.get("revenue"),
                "gross_profit": r.get("grossProfit"),
                "ebit": r.get("operatingIncome") or r.get("ebitda"),
                "shares": r.get("weightedAverageShsOut"),
            }
            for r in inc_raw
        ]
        bal = [
            {
                "total_assets": r.get("totalAssets"),
                "long_term_debt": r.get("longTermDebt"),
                "current_assets": r.get("totalCurrentAssets"),
                "current_liabilities": r.get("totalCurrentLiabilities"),
            }
            for r in bal_raw
        ]
        cf = [
            {
                "operating_cash_flow": r.get("operatingCashFlow"),
                "free_cash_flow": r.get("freeCashFlow"),
                "capex": r.get("capitalExpenditure"),
            }
            for r in (cf_raw or [])
        ]
        return inc, bal, cf, {"pe": pe, "sector": sector}


class _TwelveData:
    name = "TwelveData"
    base = "https://api.twelvedata.com"

    def __init__(self):
        self.key = os.environ.get("TWELVEDATA_KEY", "").strip()

    @property
    def available(self):
        return bool(self.key)

    def _get(self, path, **params):
        params["apikey"] = self.key
        _throttle()
        r = requests.get(f"{self.base}/{path}", params=params, headers=_UA, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        j = r.json()
        if isinstance(j, dict) and j.get("code") and j.get("status") == "error":
            return None
        return j

    def fetch(self, ticker):
        inc_raw = self._get("income_statement", symbol=ticker, period="annual")
        bal_raw = self._get("balance_sheet", symbol=ticker, period="annual")
        cf_raw = self._get("cash_flow", symbol=ticker, period="annual")
        stmts = (inc_raw or {}).get("income_statement") if inc_raw else None
        bsts = (bal_raw or {}).get("balance_sheet") if bal_raw else None
        if not stmts or not bsts:
            return None
        quote = self._get("quote", symbol=ticker) or {}
        prof = self._get("profile", symbol=ticker) or {}
        inc = [
            {
                "net_income": r.get("net_income"),
                "revenue": r.get("sales") or r.get("revenue"),
                "gross_profit": r.get("gross_profit"),
                "ebit": r.get("operating_income"),
                "shares": r.get("basic_shares_outstanding"),
            }
            for r in stmts
        ]
        bal = [
            {
                "total_assets": (
                    (r.get("assets") or {}).get("total_assets")
                    if isinstance(r.get("assets"), dict)
                    else r.get("total_assets")
                ),
                "long_term_debt": r.get("long_term_debt"),
                "current_assets": r.get("total_current_assets"),
                "current_liabilities": r.get("total_current_liabilities"),
            }
            for r in bsts
        ]
        cfl = (cf_raw or {}).get("cash_flow") or []
        cf = [
            {
                "operating_cash_flow": r.get("operating_cash_flow"),
                "free_cash_flow": r.get("free_cash_flow"),
                "capex": r.get("capital_expenditures"),
            }
            for r in cfl
        ]
        return inc, bal, cf, {"pe": quote.get("pe") or prof.get("pe"), "sector": prof.get("sector")}


_PROVIDERS = None


def _providers():
    global _PROVIDERS
    if _PROVIDERS is None:
        _PROVIDERS = [p for p in (_FMP(), _TwelveData()) if p.available]
    return _PROVIDERS


def available_source():
    ps = _providers()
    return ps[0].name if ps else None


def get_fundamentals(ticker):
    """Fetch + score fundamentals for a ticker yfinance can't serve.
    Returns a scan-compatible dict, or None if no keyed provider has data."""
    for p in _providers():
        try:
            fetched = p.fetch(ticker)
        except Exception:
            fetched = None
        if not fetched:
            continue
        inc, bal, cf, profile = fetched
        scored = score_from_statements(inc, bal, cf, profile)
        if scored.get("f_score") is not None:
            scored["source"] = p.name
            return scored
    return None


if __name__ == "__main__":
    import json
    import sys

    print("available source:", available_source())
    for t in sys.argv[1:]:
        print(t, "->", json.dumps(get_fundamentals(t)))
