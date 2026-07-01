#!/usr/bin/env python3
# screener_in_auth.py
# ===================
# Authenticated screener.in access — logs in with YOUR credentials to unlock
# per-company financial exports (full P&L / balance sheet / cash flow / ratios),
# which fills the India fundamentals gap so the fundamental screeners (piotroski,
# magic_formula, garp, debt_reduction, dividend_yield, loss_to_profit, coffee_can)
# can run on real Indian data.
#
# ⚠️ SECURITY: credentials come ONLY from environment variables — never hard-code
# them, never paste them into chat, never commit them:
#     export SCREENER_EMAIL="you@example.com"
#     export SCREENER_PASSWORD="••••••••"
# Use a dedicated/low-value password; respect screener.in's Terms of Service and
# rate limits (this is for your own account's data, fetched politely).
#
#   from screener_in_auth import session, company_financials
#   s = session()                       # logged-in requests.Session (cached)
#   f = company_financials("RELIANCE")  # -> fundamentals dict for strategies

from __future__ import annotations

import io
import os
import re
import time
import warnings
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests

warnings.filterwarnings("ignore")
_UA = {"User-Agent": "Mozilla/5.0 (research)"}
BASE = "https://www.screener.in"


class AuthError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def session() -> requests.Session:
    """Return a logged-in session (cached). Reads SCREENER_EMAIL / SCREENER_PASSWORD."""
    email = os.environ.get("SCREENER_EMAIL")
    pw = os.environ.get("SCREENER_PASSWORD")
    if not (email and pw):
        raise AuthError("set SCREENER_EMAIL and SCREENER_PASSWORD env vars (never hard-code them)")
    s = requests.Session()
    s.headers.update(_UA)
    r = s.get(f"{BASE}/login/", timeout=25)
    token = re.search(r'name=["\']csrfmiddlewaretoken["\'] value=["\']([^"\']+)', r.text)
    if not token:
        raise AuthError("could not read CSRF token from login page")
    s.post(
        f"{BASE}/login/",
        data={
            "csrfmiddlewaretoken": token.group(1),
            "username": email,
            "password": pw,
            "next": "/",
        },
        headers={"Referer": f"{BASE}/login/"},
        timeout=25,
    )
    # verify: the dashboard/account page is only reachable when logged in
    who = s.get(f"{BASE}/", timeout=25)
    if "/logout/" not in who.text and "logout" not in who.text.lower():
        raise AuthError("login failed — check credentials")
    return s


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def company_export_bytes(symbol: str) -> Optional[bytes]:
    """Download the authenticated 'Export to Excel' workbook for a company."""
    s = session()
    page = s.get(f"{BASE}/company/{symbol}/", timeout=25)
    href = re.search(r'href=["\']([^"\']*/export/[^"\']*)["\']', page.text)
    if not href:
        return None
    url = href.group(1)
    if url.startswith("/"):
        url = BASE + url
    r = s.get(url, timeout=40)
    return r.content if r.status_code == 200 and len(r.content) > 1000 else None


def company_financials(symbol: str) -> dict:
    """Parse the screener.in export into a strategy-ready fundamentals dict.

    Returns {} if not logged in / no export. The export's 'Data Sheet' holds P&L,
    balance-sheet and cash-flow rows keyed by year — enough for the fundamental
    strategies (net income, ROE/ROCE, debt, current ratio, inventory/receivables/
    payables for CCC, dividends, EPS growth)."""
    raw = company_export_bytes(symbol)
    if not raw:
        return {}
    try:
        xl = pd.ExcelFile(io.BytesIO(raw))
        sheet = next((s for s in xl.sheet_names if "data" in s.lower()), xl.sheet_names[0])
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)
    except Exception:
        return {}

    def row(*labels):
        for lab in labels:
            m = df[
                df.apply(
                    lambda r: r.astype(str).str.strip().str.lower().eq(lab.lower()).any(), axis=1
                )
            ]
            if not m.empty:
                vals = [_num(v) for v in m.iloc[0].tolist()[1:] if _num(v) is not None]
                if vals:
                    return vals
        return []

    ni = row("Net profit", "Net Profit")
    rev = row("Sales", "Revenue", "Total Revenue")
    debt = row("Borrowings", "Total Debt")
    inv = row("Inventory", "Inventories")
    recv = row("Trade receivables", "Debtors", "Receivables")
    pay = row("Trade Payables", "Creditors", "Payables")
    div = row("Dividend Amount", "Dividend Paid")
    eps = row("EPS in Rs", "EPS")

    def g(lst, i=0):
        return lst[i] if len(lst) > i else None

    def pct(a, b):
        return (a - b) / abs(b) * 100 if (a is not None and b not in (None, 0)) else None

    out = {
        "source": "screener.in",
        "net_income": g(ni),
        "net_income_prev": g(ni, 1),
        "revenue": g(rev),
        "revenue_5y_ago": g(rev, -1) if rev else None,
        "debt_history": debt or None,
        "eps_ttm": g(eps),
        "eps_growth": pct(g(eps), g(eps, 1)),
        "inventory": g(inv),
        "receivables": g(recv),
        "payables": g(pay),
        "dividend_history": div or None,
    }
    return {k: v for k, v in out.items() if v is not None}


def fundamentals_batch(symbols, pause: float = 1.0, verbose: bool = True) -> dict:
    """Fetch fundamentals for many symbols (polite pause between requests)."""
    out = {}
    for i, sym in enumerate(symbols, 1):
        try:
            f = company_financials(sym)
            if f:
                out[sym] = f
        except AuthError:
            raise
        except Exception:
            pass
        if verbose and i % 25 == 0:
            print(f"  {i}/{len(symbols)} fetched ({len(out)} ok)")
        time.sleep(pause)
    return out


if __name__ == "__main__":
    import sys

    try:
        s = session()
        print("logged in ✓")
        for t in sys.argv[1:] or ["RELIANCE"]:
            print(t, company_financials(t))
    except AuthError as e:
        print("AUTH:", e)
