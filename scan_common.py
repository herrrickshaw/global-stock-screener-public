"""Helpers shared by the Europe/Japan/Korea market-scan scripts (pure or I/O at the edge)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

SheetSpec = tuple[list[dict], str, "str | None"]


def trend_signal(dist_ma, fallback: str) -> str:
    """Label the % distance from the 200-day MA (None/0 -> `fallback`; +-5% is the band)."""
    return (
        "Above 200MA (Uptrend)"
        if dist_ma and dist_ma > 5
        else (
            "Below 200MA (Downtrend)"
            if dist_ma and dist_ma < -5
            else "Near 200MA (Consolidation)" if dist_ma else fallback
        )
    )


def write_scan_workbook(
    path: Path,
    sheets: Sequence[SheetSpec],
    style_sheet: Callable | None = None,
) -> None:
    """Write (rows, sheet_name, sort_col) specs to an .xlsx; sort descending, style if given."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for rows, name, sort_col in sheets:
            if not rows:
                pd.DataFrame().to_excel(writer, sheet_name=name, index=False)
                continue
            df = pd.DataFrame(rows)
            if sort_col and sort_col in df.columns:
                df = df.sort_values(sort_col, ascending=False)
            df.to_excel(writer, sheet_name=name, index=False)
            if style_sheet is not None:
                style_sheet(writer.sheets[name])


# ── yfinance statement accessors (identical in all three scanners) ───────────
def first_df(ticker, *attrs):
    """First non-empty DataFrame attribute among `attrs` on a yfinance Ticker, else None."""
    for attr in attrs:
        df = getattr(ticker, attr, None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def row(df, *names, col: int = 0):
    """Value of the first matching statement row at column `col` (0 = latest year), or None."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            try:
                val = df.loc[name].iloc[col]
                return float(val) if pd.notna(val) else None
            except Exception:
                pass
    return None


def series(df, *names) -> list[float]:
    """All non-NaN values (latest first) of the first matching statement row, else []."""
    if df is None or df.empty:
        return []
    for name in names:
        if name in df.index:
            return [float(v) for v in df.loc[name].dropna() if pd.notna(v)]
    return []


def piotroski_f_score(inc, bal, cf) -> int:
    """Piotroski F-score (0-9) from yfinance income/balance/cash-flow frames.

    May raise on malformed statements; callers decide how to degrade.
    """
    _row = row
    ni0 = _row(inc, "Net Income", col=0)
    a0 = _row(bal, "Total Assets", col=0)
    ni1 = _row(inc, "Net Income", col=1)
    a1 = _row(bal, "Total Assets", col=1)
    roa0 = (ni0 / a0) if (ni0 and a0) else None
    roa1 = (ni1 / a1) if (ni1 and a1) else None
    ocf0 = _row(cf, "Operating Cash Flow", "Total Cash From Operating Activities", col=0)

    f1 = 1 if (roa0 and roa0 > 0) else 0
    f2 = 1 if (ocf0 and ocf0 > 0) else 0
    f3 = 1 if (roa0 and roa1 and roa0 > roa1) else 0
    f4 = 1 if (ocf0 and a0 and roa0 and (ocf0 / a0) > roa0) else 0

    ltd0 = _row(bal, "Long Term Debt", col=0) or 0
    ltd1 = _row(bal, "Long Term Debt", col=1) or 0
    f5 = 1 if (a0 and a1 and (ltd0 / a0) < (ltd1 / a1)) else 0

    ca0 = _row(bal, "Current Assets", "Total Current Assets", col=0)
    cl0 = _row(bal, "Current Liabilities", "Total Current Liabilities", col=0)
    ca1 = _row(bal, "Current Assets", "Total Current Assets", col=1)
    cl1 = _row(bal, "Current Liabilities", "Total Current Liabilities", col=1)
    f6 = 1 if (ca0 and cl0 and ca1 and cl1 and (ca0 / cl0) > (ca1 / cl1)) else 0

    sh0 = _row(bal, "Share Issued", col=0)
    sh1 = _row(bal, "Share Issued", col=1)
    f7 = (1 if sh0 <= sh1 else 0) if (sh0 and sh1) else 1

    rev0 = _row(inc, "Total Revenue", col=0)
    gp0 = _row(inc, "Gross Profit", col=0)
    rev1 = _row(inc, "Total Revenue", col=1)
    gp1 = _row(inc, "Gross Profit", col=1)
    f8 = 1 if (gp0 and rev0 and gp1 and rev1 and (gp0 / rev0) > (gp1 / rev1)) else 0
    f9 = 1 if (rev0 and a0 and rev1 and a1 and (rev0 / a0) > (rev1 / a1)) else 0

    return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9


def revenue_cagr_pct(revs: list[float]) -> float | None:
    """CAGR % between newest (revs[0]) and oldest (revs[-1]); None unless >=2 points, oldest>0."""
    if len(revs) >= 2 and revs[-1] > 0:
        return ((revs[0] / revs[-1]) ** (1 / (len(revs) - 1)) - 1) * 100
    return None


def average_roce_pct(ebit_s: list[float], ta_s: list[float], cl_s: list[float]) -> float | None:
    """Mean of EBIT / (total assets - current liabilities) in %, skipping non-positive capital."""
    roce_l = [
        ebit_s[i] / (ta_s[i] - cl_s[i]) * 100
        for i in range(min(len(ebit_s), len(ta_s), len(cl_s)))
        if (ta_s[i] - cl_s[i]) > 0
    ]
    return sum(roce_l) / len(roce_l) if roce_l else None


def free_cash_flow_positive(cf) -> int:
    """1 if latest FCF > 0 (or OCF - |capex| > 0 when FCF is absent), else 0."""
    fcf_s = series(cf, "Free Cash Flow")
    if fcf_s:
        return 1 if fcf_s[0] > 0 else 0
    ocf_s, capex_s = series(cf, "Operating Cash Flow"), series(cf, "Capital Expenditure")
    return 1 if (ocf_s and capex_s and (ocf_s[0] - abs(capex_s[0])) > 0) else 0


def coffee_can_core(inc, bal, cf, ticker, mcap_floor: float) -> tuple[dict, float]:
    """Japan/Korea-adapted Coffee Can (C1-C6); returns (result_updates, market_cap).

    C1 revenue CAGR >10%, C2 avg ROCE >15%, C3 D/E <1, C4 market cap >= `mcap_floor`,
    C5 net income positive in every reported year, C6 positive FCF. May raise.
    """
    c = {}
    cagr = revenue_cagr_pct(series(inc, "Total Revenue"))
    c["C1"] = 1 if (cagr is not None and cagr > 10) else 0

    ebit_s = series(inc, "EBIT", "Operating Income", "Ebit")
    ta_s = series(bal, "Total Assets")
    cl_s = series(bal, "Current Liabilities", "Total Current Liabilities")
    avg_roce = average_roce_pct(ebit_s, ta_s, cl_s)
    c["C2"] = 1 if (avg_roce and avg_roce > 15) else 0

    ltd_s = series(bal, "Long Term Debt")
    eq_s = series(
        bal,
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Total Equity Gross Minority Interest",
    )
    de = (ltd_s[0] / abs(eq_s[0])) if (ltd_s and eq_s and eq_s[0] != 0) else None
    c["C3"] = 1 if (de is not None and de < 1) else 0

    try:
        mcap = ticker.fast_info.market_cap or 0
    except Exception:
        mcap = 0
    c["C4"] = 1 if mcap >= mcap_floor else 0

    ni_s = series(inc, "Net Income")
    c["C5"] = 1 if (ni_s and all(n > 0 for n in ni_s)) else 0
    c["C6"] = free_cash_flow_positive(cf)

    qualifies = sum(c.values()) == len(c)
    return (
        {
            "cc_qualifies": "PASS" if qualifies else "FAIL",
            "cc_score": f"{sum(c.values())}/{len(c)}",
            "cc_rev_cagr": round(cagr, 2) if cagr else None,
            "cc_roce_avg": round(avg_roce, 2) if avg_roce else None,
            "cc_debt_equity": round(de, 2) if de is not None else None,
        },
        mcap,
    )
