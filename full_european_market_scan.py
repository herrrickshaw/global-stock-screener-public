# full_european_market_scan.py
# ===========================
# Optimized EURO STOXX 50 universe scanner (50 major Eurozone equity stocks).
# Features:
#   - Static Metadata: Instant Name & Sector mapping (zero API calls for index list).
#   - Restricts slow financials & info scans exclusively to breakout candidates.
#   - Auto-caches Ticker data to prevent duplicate HTTP calls.
#
# Pipeline:
#   Stage 1 & 2 — Bulk download 1-year OHLC for all 50 tickers in one batch.
#   Stage 3 — Compute Darvas Box & 200-day MA trend locally (instant).
#   Stage 4 — Run parallel fundamental scans (Piotroski + Coffee Can + PEGY) on breakouts only.
#   Stage 5 — Save a beautifully styled Excel workbook.
#
# Usage:
#   python full_european_market_scan.py
#   python full_european_market_scan.py --top 10      # Test run first 10
#   python full_european_market_scan.py --no-scans    # Skip fundamental scans
#
# Install:
#   pip install yfinance pandas openpyxl requests

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from scan_common import (
    average_roce_pct,
    first_df,
    free_cash_flow_positive,
    piotroski_f_score,
    revenue_cagr_pct,
    row,
    series,
    trend_signal,
    write_scan_workbook,
)

try:
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    OPENPYXL_OK = True
except ImportError:
    OPENPYXL_OK = False

# ── Configuration & Constants ────────────────────────────────────────────────
DOWNLOAD_DIR = Path("./european_scan")
DOWNLOAD_DIR.mkdir(exist_ok=True)

DARVAS_CONFIRM = 3
MAX_WORKERS = 10
PIOTROSKI_STRONG = 7

# Static Metadata for Euro Stoxx 50 components (removes 50 slow .info API calls)
EURO_STOXX_50_META = {
    "ADS.DE": ("Adidas", "Consumer Cyclical"),
    "ADYEN.AS": ("Adyen", "Financial Services"),
    "AD.AS": ("Ahold Delhaize", "Consumer Defensive"),
    "AI.PA": ("Air Liquide", "Basic Materials"),
    "AIR.PA": ("Airbus", "Industrials"),
    "ALV.DE": ("Allianz", "Financial Services"),
    "ABI.BR": ("Anheuser-Busch InBev", "Consumer Defensive"),
    "ARGX.BR": ("Argenx", "Healthcare"),
    "ASML.AS": ("ASML Holding", "Technology"),
    "CS.PA": ("Axa", "Financial Services"),
    "BAS.DE": ("BASF", "Basic Materials"),
    "BAYN.DE": ("Bayer", "Healthcare"),
    "BBVA.MC": ("BBVA", "Financial Services"),
    "SAN.MC": ("Banco Santander", "Financial Services"),
    "BMW.DE": ("BMW", "Consumer Cyclical"),
    "BNP.PA": ("BNP Paribas", "Financial Services"),
    "BN.PA": ("Danone", "Consumer Defensive"),
    "DBK.DE": ("Deutsche Bank", "Financial Services"),
    "DB1.DE": ("Deutsche Börse", "Financial Services"),
    "DHL.DE": ("Deutsche Post", "Industrials"),
    "DTE.DE": ("Deutsche Telekom", "Communication Services"),
    "ENEL.MI": ("Enel", "Utilities"),
    "ENI.MI": ("Eni", "Energy"),
    "EL.PA": ("EssilorLuxottica", "Healthcare"),
    "RACE.MI": ("Ferrari", "Consumer Cyclical"),
    "RMS.PA": ("Hermès", "Consumer Cyclical"),
    "IBE.MC": ("Iberdrola", "Utilities"),
    "ITX.MC": ("Inditex", "Consumer Cyclical"),
    "IFX.DE": ("Infineon Technologies", "Technology"),
    "INGA.AS": ("ING Group", "Financial Services"),
    "ISP.MI": ("Intesa Sanpaolo", "Financial Services"),
    "OR.PA": ("L'Oréal", "Consumer Defensive"),
    "MC.PA": ("LVMH", "Consumer Cyclical"),
    "MBG.DE": ("Mercedes-Benz Group", "Consumer Cyclical"),
    "MUV2.DE": ("Munich Re", "Financial Services"),
    "NDA-FI.HE": ("Nordea", "Financial Services"),
    "PRX.AS": ("Prosus", "Technology"),
    "RHM.DE": ("Rheinmetall", "Industrials"),
    "SAF.PA": ("Safran", "Industrials"),
    "SGO.PA": ("Saint-Gobain", "Industrials"),
    "SAN.PA": ("Sanofi", "Healthcare"),
    "SAP.DE": ("SAP", "Technology"),
    "SU.PA": ("Schneider Electric", "Industrials"),
    "SIE.DE": ("Siemens", "Industrials"),
    "ENR.DE": ("Siemens Energy", "Utilities"),
    "TTE.PA": ("TotalEnergies", "Energy"),
    "DG.PA": ("Vinci SA", "Industrials"),
    "UCG.MI": ("UniCredit", "Financial Services"),
    "VOW.DE": ("Volkswagen", "Consumer Cyclical"),
    "WKL.AS": ("Wolters Kluwer", "Professional Services"),
}


# ── Financials Extractors ─────────────────────────────────────────────────────
_first_df, _row, _series = first_df, row, series


# ── Technical Scans ───────────────────────────────────────────────────────────
def compute_darvas_box(df: pd.DataFrame, confirm: int = DARVAS_CONFIRM) -> dict:
    """Detect Darvas Box from historical Close/High/Low series."""
    if df is None or df.empty or len(df) < confirm + 5:
        return {"signal": "INSUFFICIENT_DATA", "box_top": None, "box_bottom": None}

    highs = pd.to_numeric(df["High"], errors="coerce").fillna(0).tolist()
    lows = pd.to_numeric(df["Low"], errors="coerce").fillna(0).tolist()
    closes = pd.to_numeric(df["Close"], errors="coerce").fillna(0).tolist()

    current = closes[-1]
    highs_h = highs[:-1]  # Exclude current bar to avoid lookahead contamination
    lows_h = lows[:-1]
    n = len(highs_h)

    # Step 1: Box Top
    box_top_idx = box_top = None
    for i in range(n - confirm - 1, -1, -1):
        c = highs_h[i]
        if c == 0:
            continue
        w = highs_h[i + 1 : i + 1 + confirm]
        if len(w) == confirm and all(h < c for h in w):
            box_top_idx, box_top = i, c
            break

    if box_top is None:
        return {"signal": "NO_BOX", "box_top": None, "box_bottom": None}

    # Step 2: Box Bottom (historical segment starting from the box top)
    seg = lows_h[box_top_idx:]
    box_bottom = None
    for i in range(len(seg) - confirm):
        c = seg[i]
        if c == 0:
            continue
        w = seg[i + 1 : i + 1 + confirm]
        if len(w) == confirm and all(l > c for l in w):
            box_bottom = c
            break
    if box_bottom is None:
        valid = [l for l in seg if l > 0]
        box_bottom = min(valid) if valid else None

    if box_bottom is None:
        return {"signal": "NO_BOX", "box_top": round(box_top, 2), "box_bottom": None}

    # Step 3: Classify Signal
    if current > box_top:
        signal = "BREAKOUT_BUY"
    elif current < box_bottom:
        signal = "BREAKDOWN_SELL"
    else:
        signal = "IN_BOX"

    rng = box_top - box_bottom
    upside_pct = (box_top - current) / current * 100 if current else 0
    pos_in_box = (current - box_bottom) / rng * 100 if rng else 0

    return {
        "signal": signal,
        "box_top": round(box_top, 2),
        "box_bottom": round(box_bottom, 2),
        "current_price": round(current, 2),
        "upside_pct": round(upside_pct, 2),
        "pos_in_box": round(pos_in_box, 1),
    }


# ── Fundamental Scan ──────────────────────────────────────────────────────────
_EMERG_FUND = {"mod": None, "checked": False}


def _emerging_fundamentals(symbol: str):
    """Pull statements from a keyed provider (FMP/Twelve Data) for tickers
    yfinance can't serve. Returns a res-compatible dict or None (no key/no data).
    Only attempted for exchanges Yahoo lacks statements for, to save quota."""
    if not str(symbol).upper().endswith((".IS", ".PR", ".BD")):
        return None
    if not _EMERG_FUND["checked"]:
        _EMERG_FUND["checked"] = True
        try:
            import emerging_fundamentals as ef

            _EMERG_FUND["mod"] = ef if ef.available_source() else None
        except Exception:
            _EMERG_FUND["mod"] = None
    ef = _EMERG_FUND["mod"]
    if ef is None:
        return None
    try:
        return ef.get_fundamentals(symbol)
    except Exception:
        return None


def _coffee_can(inc, bal, cf, info, ticker, state: dict) -> dict:
    """Coffee Can C1-C6 (market cap floor 1bn EUR); returns result updates. May raise.

    Records the market cap in `state["mcap"]` so the valuation step can reuse it (and,
    as before, is skipped if this step failed before reaching it).
    """
    c = {}
    cagr = revenue_cagr_pct(_series(inc, "Total Revenue"))
    c["C1"] = 1 if (cagr and cagr > 10) else 0

    avg_roce = average_roce_pct(
        _series(inc, "EBIT", "Operating Income"),
        _series(bal, "Total Assets"),
        _series(bal, "Current Liabilities", "Total Current Liabilities"),
    )
    c["C2"] = 1 if (avg_roce and avg_roce > 15) else 0

    de_raw = info.get("debtToEquity")
    de = de_raw / 100 if de_raw is not None and de_raw > 10 else de_raw
    if de is None:
        ltd_s = _series(bal, "Long Term Debt")
        eq_s = _series(bal, "Stockholders Equity", "Total Stockholder Equity")
        de = (ltd_s[0] / abs(eq_s[0])) if (ltd_s and eq_s and eq_s[0] != 0) else None
    c["C3"] = 1 if (de is not None and de < 1) else 0

    mcap = info.get("marketCap") or getattr(ticker.fast_info, "market_cap", 0)
    state["mcap"] = mcap
    c["C4"] = 1 if mcap >= 1e9 else 0  # >= 1 Billion EUR

    ni_s = _series(inc, "Net Income")
    c["C5"] = 1 if (ni_s and all(n > 0 for n in ni_s)) else 0
    c["C6"] = free_cash_flow_positive(cf)

    qualifies = sum(c.values()) == len(c)
    return {
        "cc_qualifies": "PASS" if qualifies else "FAIL",
        "cc_score": f"{sum(c.values())}/{len(c)}",
        "cc_rev_cagr": round(cagr, 2) if cagr else None,
        "cc_roce_avg": round(avg_roce, 2) if avg_roce else None,
        "cc_debt_equity": round(de, 2) if de else None,
    }


def _valuation(info: dict, mcap, res: dict) -> None:
    """Write valuation + PEGY fields into `res` from yfinance `info` and the market cap.

    `sector` is written first, so it survives a later failure (as before). May raise.
    """
    pe = (
        info.get("trailingPE")
        or info.get("forwardPE")
        or info.get("regularMarketPrice", 0) / (info.get("trailingEps") or 1)
    )
    peg = info.get("pegRatio")
    dy_raw = info.get("dividendYield")
    dy = (dy_raw if dy_raw >= 0.15 else dy_raw * 100) if dy_raw is not None else 0.0
    pegy = float(peg) + dy if peg is not None else None
    res["sector"] = info.get("sector")
    res.update(
        {
            "market_cap_b": round(mcap / 1e9, 2) if mcap else None,
            "pe_ratio": round(pe, 2) if pe and pe > 0 else None,
            "pb_ratio": round(info.get("priceToBook"), 2) if info.get("priceToBook") else None,
            "dividend_yield": round(dy, 2),
            "pegy_ratio": round(pegy, 2) if pegy else None,
        }
    )


def fundamental_scan(symbol: str) -> dict:
    """Fetch financials and execute F-Score, Coffee Can, and PEGY calculations in one Ticker call."""
    res = {"symbol": symbol, "f_score": None, "cc_qualifies": "FAIL", "error": None}
    try:
        ticker = yf.Ticker(symbol)
        inc = _first_df(ticker, "income_stmt", "financials")
        bal = _first_df(ticker, "balance_sheet")
        cf = _first_df(ticker, "cash_flow", "cashflow")

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        if inc is None or inc.empty:
            # yfinance has no statements for this market (Istanbul/Prague/Budapest
            # and other emerging tickers). Fall back to a keyed fundamentals
            # provider (FMP / Twelve Data) if one is configured; otherwise keep
            # the momentum-only behaviour.
            alt = _emerging_fundamentals(symbol)
            if alt is not None:
                res.update(alt)
                return res
            res["error"] = "no_financial_statements"
            return res
    except Exception as e:
        res["error"] = str(e)[:80]
        return res

    try:
        res["f_score"] = piotroski_f_score(inc, bal, cf)
    except Exception:
        pass

    state: dict = {}
    try:
        res.update(_coffee_can(inc, bal, cf, info, ticker, state))
    except Exception:
        pass

    try:
        _valuation(info, state["mcap"], res)
    except Exception:
        pass

    return res


# ── Sheet Styling ─────────────────────────────────────────────────────────────
def style_excel_sheet(ws):
    """Format and style an openpyxl sheet to premium standards."""
    if not OPENPYXL_OK:
        return

    fill_header = PatternFill(start_color="1A365D", end_color="1A365D", fill_type="solid")
    fill_zebra = PatternFill(start_color="F7FAFC", end_color="F7FAFC", fill_type="solid")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_body = Font(name="Calibri", size=11, bold=False, color="000000")

    border_thin = Border(
        left=Side(style="thin", color="E2E8F0"),
        right=Side(style="thin", color="E2E8F0"),
        top=Side(style="thin", color="E2E8F0"),
        bottom=Side(style="thin", color="E2E8F0"),
    )

    # Style Header
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border_thin

    # Style Data Rows
    for row_idx in range(2, ws.max_row + 1):
        is_zebra = row_idx % 2 == 1
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = font_body
            cell.border = border_thin
            if is_zebra:
                cell.fill = fill_zebra

            header = str(ws.cell(row=1, column=col_idx).value or "").upper()
            val = cell.value

            # Alignment
            if any(
                k in header for k in ["SYMBOL", "SIGNAL", "SCORE", "CLASS", "QUALIFIES", "EXCHANGE"]
            ):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

            # Formats
            if isinstance(val, (int, float)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
                if "%" in header or "YIELD" in header or "CAGR" in header or "ROCE" in header:
                    cell.number_format = '0.00"%"'
                elif (
                    "CAP" in header
                    or "LTP" in header
                    or "BOX" in header
                    or "PRICE" in header
                    or "200MA" in header
                ):
                    cell.number_format = "#,##0.00"
                elif (
                    "PEG" in header
                    or "PE_RATIO" in header
                    or "PB_RATIO" in header
                    or "DEBT_EQUITY" in header
                ):
                    cell.number_format = "0.00"

    ws.row_dimensions[1].height = 28
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 20

    # Auto Column Widths
    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    ws.views.sheetView[0].showGridLines = True


# ── Main Scanner ──────────────────────────────────────────────────────────────
def load_universe(path):
    """Load a broad European universe CSV (yf_ticker,name,index,exchange) into
    an ordered {ticker: (name, sector_or_exchange)} meta dict."""
    df = pd.read_csv(os.path.expanduser(path))
    tcol = "yf_ticker" if "yf_ticker" in df.columns else df.columns[0]
    ncol = "name" if "name" in df.columns else (df.columns[1] if len(df.columns) > 1 else tcol)
    xcol = "exchange" if "exchange" in df.columns else None
    meta = {}
    for _, r in df.iterrows():
        t = str(r[tcol]).strip()
        if not t or t.lower() == "nan":
            continue
        # Sector is unknown from index lists → seed with exchange; enriched from
        # yfinance .info during the fundamental stage where available.
        meta[t] = (str(r[ncol]), str(r[xcol]) if xcol else "—")
    return meta


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="European market scanner (Darvas + Piotroski + Coffee Can)"
    )
    parser.add_argument("--top", type=int, default=0, help="Scan only first N tickers")
    parser.add_argument(
        "--no-scans", action="store_true", default=False, help="Skip Stage 4 fundamental scans"
    )
    parser.add_argument(
        "--universe",
        default=None,
        help="CSV of a broad universe (yf_ticker,name,index,exchange). Default: EURO STOXX 50.",
    )
    parser.add_argument("--label", default=None, help="Scan label used in the output filename.")
    parser.add_argument(
        "--batch", type=int, default=120, help="OHLC download batch size (default 120)."
    )
    return parser.parse_args(argv)


def _print_banner(label: str, n_symbols: int) -> None:
    print(f"\n{'#'*60}")
    print(f"  EUROPEAN MARKET SCANNER — {label}  ({n_symbols} tickers)")
    print(f"  Started: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    try:
        import emerging_fundamentals as _ef

        _src = _ef.available_source()
        print(
            f"  Emerging-market fundamentals (.IS/.PR/.BD): "
            f"{_src + ' active' if _src else 'no provider key set (FMP_API_KEY / TWELVEDATA_KEY) — momentum-only'}"
        )
    except Exception:
        pass
    print(f"{'#'*60}")


def _download_chunk(chunk: list, batch_no: int):
    """Download one chunk of 1y OHLC; back off once on a rate-limit-shaped error."""
    raw = None
    for attempt in range(2):
        try:
            raw = yf.download(
                chunk,
                period="1y",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            break
        except Exception as e:
            msg = str(e)
            rate_limited = "Rate limited" in msg or "Too Many Requests" in msg
            if rate_limited and attempt == 0:
                print(f"    ⏳ batch {batch_no} rate-limited; backing off 20s and retrying …")
                time.sleep(20)
                continue
            print(f"    ⚠️  batch {batch_no} download failed ({msg[:60]}); skipping.")
    return raw


def _split_chunk(raw, chunk: list) -> dict:
    """Pick each ticker's non-empty OHLC frame out of a (possibly multi-index) download."""
    out = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for s in chunk:
        try:
            sub = (
                raw[s]
                if (multi and s in raw.columns.get_level_values(0))
                else (raw if (not multi and len(chunk) == 1) else None)
            )
            if sub is not None and not sub.dropna().empty:
                out[s] = sub.dropna()
        except Exception:
            continue
    return out


def _download_ohlc(symbols: list, batch: int) -> dict:
    """Stages 1 & 2: chunked yfinance download (chunked so ~1000-ticker universes don't trip
    Yahoo's burst rate limiter); pause briefly between chunks."""
    print(
        f"\nStage 1 & 2 — Downloading 1y history for {len(symbols)} stocks in batches of {batch} …"
    )
    ohlc_by_sym = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i : i + batch]
        raw = _download_chunk(chunk, i // batch + 1)
        if raw is None or raw.empty:
            time.sleep(2)
            continue
        ohlc_by_sym.update(_split_chunk(raw, chunk))
        time.sleep(2)
        print(
            f"    batch {i//batch+1}/{(len(symbols)-1)//batch+1} — {len(ohlc_by_sym)} tickers with data so far"
        )
    return ohlc_by_sym


def _technical_row(sym: str, df: pd.DataFrame, meta: dict) -> tuple[dict, dict]:
    """Darvas box + 200-day MA row for one symbol; returns (row, darvas_result)."""
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    ltp = round(float(closes.iloc[-1]), 2)
    prev = round(float(closes.iloc[-2]), 2) if len(closes) >= 2 else None
    chg_pct = round((ltp - prev) / prev * 100, 2) if prev else None

    ma_200 = round(closes.rolling(200).mean().iloc[-1], 2) if len(closes) >= 200 else None
    dist_ma = round(((ltp - ma_200) / ma_200 * 100), 2) if ma_200 else None

    darv = compute_darvas_box(df)
    name, sector = meta.get(sym, (sym, "—"))
    row = {
        "Symbol": sym,
        "Name": name,
        "Sector": sector,
        "LTP": ltp,
        "Change%": chg_pct,
        "200_Day_MA": ma_200,
        "Distance_to_200MA%": dist_ma,
        "Trend_Signal": trend_signal(dist_ma, "Insufficient History"),
        "Darvas_Signal": darv.get("signal"),
        "Box_Top": darv.get("box_top"),
        "Box_Bottom": darv.get("box_bottom"),
        "Upside_to_Top%": darv.get("upside_pct"),
        "Position_in_Box%": darv.get("pos_in_box"),
        # Valuation fallbacks
        "Market_Cap_B": None,
        "P_E_Ratio": None,
        "P_B_Ratio": None,
        "Dividend_Yield%": 0.0,
    }
    return row, darv


def _technical_stage(symbols: list, ohlc_by_sym: dict, meta: dict) -> tuple[dict, list]:
    """Stage 3: (all_stocks_map, breakout_symbols) from the local Darvas/MA maths."""
    all_stocks_map = {}
    breakout_symbols = []
    for sym in symbols:
        try:
            df = ohlc_by_sym.get(sym)
            if df is None or df.empty or len(df) < 20:
                continue
            row, darv = _technical_row(sym, df, meta)
            all_stocks_map[sym] = row
            if darv.get("signal") == "BREAKOUT_BUY":
                breakout_symbols.append(sym)
        except Exception as e:
            print(f"  ⚠️ Technical error on {sym}: {e}")
    return all_stocks_map, breakout_symbols


def _fundamental_row(sym: str, tech: dict, res: dict) -> dict:
    """Merge a technical row with a fundamental_scan result into one Fundamentals row."""
    return {
        "Symbol": sym,
        "Name": tech.get("Name"),
        "Sector": res.get("sector") or tech.get("Sector"),
        "LTP": tech.get("LTP"),
        "Change%": tech.get("Change%"),
        "Darvas_Signal": tech.get("Darvas_Signal"),
        "200_Day_MA": tech.get("200_Day_MA"),
        "Distance_to_200MA%": tech.get("Distance_to_200MA%"),
        "P_E_Ratio": res.get("pe_ratio"),
        "P_B_Ratio": res.get("pb_ratio"),
        "Dividend_Yield%": res.get("dividend_yield"),
        "PEGY_Ratio": res.get("pegy_ratio"),
        "Piotroski_Score": res.get("f_score"),
        "Piotroski_Class": res.get("f_interpretation"),
        "CoffeeCan_Score": res.get("cc_score"),
        "CoffeeCan_Class": res.get("cc_qualifies"),
        "CAGR_Revenue%": res.get("cc_rev_cagr"),
        "ROCE_Average%": res.get("cc_roce_avg"),
        "Debt_Equity": res.get("cc_debt_equity"),
        "Error": res.get("error"),
    }


def _fundamental_stage(breakout_symbols: list, all_stocks_map: dict) -> tuple[list, list]:
    """Stage 4: parallel fundamental scans on breakouts; also back-fills PE/PB/yield into
    `all_stocks_map`. Returns (fund_rows, triple_hits)."""
    fund_rows = []
    triple_hits = []
    print(
        f"\nStage 4 — Fetching financials & scans for {len(breakout_symbols)} breakouts ({MAX_WORKERS} workers) …"
    )
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fundamental_scan, sym): sym for sym in breakout_symbols}
        for future in as_completed(futures):
            sym = futures[future]
            done += 1
            try:
                res = future.result()
                tech = all_stocks_map.get(sym, {})
                fund_row = _fundamental_row(sym, tech, res)
                fund_rows.append(fund_row)

                # Update the master mapping so All_Stocks sheet contains PE/PB/Yield for breakouts
                tech.update(
                    {
                        "Market_Cap_B": res.get("market_cap_b"),
                        "P_E_Ratio": res.get("pe_ratio"),
                        "P_B_Ratio": res.get("pb_ratio"),
                        "Dividend_Yield%": res.get("dividend_yield"),
                    }
                )

                if (
                    res.get("f_score")
                    and res.get("f_score") >= PIOTROSKI_STRONG
                    and res.get("cc_qualifies") == "PASS"
                ):
                    triple_hits.append(fund_row.copy())

                if done % 5 == 0 or done == len(breakout_symbols):
                    print(
                        f"    {done}/{len(breakout_symbols)} processed (Triple Hits so far: {len(triple_hits)})"
                    )
            except Exception as e:
                print(f"    ❌ Fundamental error on {sym}: {e}")
    return fund_rows, triple_hits


def _save_workbook(label, all_stocks, darvas_signals, fund_rows, triple_hits) -> Path:
    """Stage 5: write the four-sheet Excel workbook to DOWNLOAD_DIR and return its path."""
    date_str = datetime.today().strftime("%Y%m%d_%H%M")
    fname = (
        "european_market_scan" if label in ("european", None) else f"european_market_scan_{label}"
    )
    excel_path = DOWNLOAD_DIR / f"{fname}_{date_str}.xlsx"
    write_scan_workbook(
        excel_path,
        [
            (all_stocks, "All_Stocks", "Change%"),
            (darvas_signals, "Darvas_Signals", "Upside_to_Top%"),
            (fund_rows, "Fundamentals", "Piotroski_Score"),
            (triple_hits, "Triple_Hits", "Piotroski_Score"),
        ],
        style_excel_sheet if OPENPYXL_OK else None,
    )
    return excel_path


def _print_summary(all_stocks, breakout_symbols, fund_rows, triple_hits, excel_path) -> None:
    print(f"\n{'='*60}")
    print(f"  SCAN COMPLETE — {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"  Total European Stocks Scanned: {len(all_stocks)}")
    print(f"  Darvas Breakouts Found:       {len(breakout_symbols)}")
    print(f"  Fundamentals Scanned:         {len(fund_rows)}")
    print(f"  ★ TRIPLE HITS FOUND:          {len(triple_hits)}")
    print(f"{'='*60}")

    if triple_hits:
        print("\n  Triple Hit Stocks:")
        for r in sorted(triple_hits, key=lambda x: x.get("Piotroski_Score") or 0, reverse=True):
            print(
                f"    • {r['Symbol']:<10} {r['Name']:<30} F-Score: {r['Piotroski_Score']}/9  CC: {r['CoffeeCan_Score']}  LTP: €{r['LTP']}"
            )
    else:
        print("\n  No Triple Hit stocks found today.")

    print(f"\n  📊 Excel report saved → {excel_path}\n")


def main():
    args = _parse_args()

    if args.universe:
        meta = load_universe(args.universe)
        label = args.label or "european_broad"
    else:
        meta = dict(EURO_STOXX_50_META)
        label = args.label or "european"
    symbols = list(meta.keys())
    if args.top:
        symbols = symbols[: args.top]

    _print_banner(label, len(symbols))

    ohlc_by_sym = _download_ohlc(symbols, args.batch)
    if not ohlc_by_sym:
        sys.exit("❌ No OHLC data downloaded for any ticker.")

    print("\nStage 3 — Computing Darvas Box & 200-day MA Trend locally …")
    all_stocks_map, breakout_symbols = _technical_stage(symbols, ohlc_by_sym, meta)
    print(
        f"  Processed {len(all_stocks_map)} stocks. Found {len(breakout_symbols)} Darvas breakouts."
    )

    fund_rows, triple_hits = [], []
    if not args.no_scans and breakout_symbols:
        fund_rows, triple_hits = _fundamental_stage(breakout_symbols, all_stocks_map)
    else:
        print("\nStage 4 — Skipped (No active breakouts or --no-scans set)")

    print("\nStage 5 — Saving Excel workbook …")
    all_stocks = list(all_stocks_map.values())
    darvas_signals = [
        s for s in all_stocks if s["Darvas_Signal"] in ("BREAKOUT_BUY", "BREAKDOWN_SELL")
    ]
    excel_path = _save_workbook(label, all_stocks, darvas_signals, fund_rows, triple_hits)
    _print_summary(all_stocks, breakout_symbols, fund_rows, triple_hits, excel_path)


if __name__ == "__main__":
    main()
